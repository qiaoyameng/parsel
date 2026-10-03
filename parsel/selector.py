"""XPath and JMESPath selectors based on the lxml and jmespath Python
packages."""

from __future__ import annotations

import codecs
import json
import re
import typing
import warnings
from functools import lru_cache
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypedDict,
    TypeVar,
)

import jmespath
from lxml import etree, html

from .csstranslator import GenericTranslator, HTMLTranslator
from .utils import extract_regex, flatten, iflatten, shorten

if TYPE_CHECKING:
    from collections.abc import Mapping
    from re import Pattern

    # typing.Self requires Python 3.11
    from typing_extensions import Self


_SelectorType = TypeVar("_SelectorType", bound="Selector")
_ParserType: TypeAlias = etree.XMLParser | etree.HTMLParser
# simplified _OutputMethodArg from types-lxml
_TostringMethodType = Literal[
    "html",
    "xml",
]


class CannotRemoveElementWithoutRoot(Exception):
    pass


class CannotRemoveElementWithoutParent(Exception):
    pass


class CannotDropElementWithoutParent(CannotRemoveElementWithoutParent):
    pass


class TreeRewriteError(Exception):
    """Base class for errors raised by :class:`TreeRewrite`."""


class RewriteConflictError(TreeRewriteError):
    """Raised when a single rewrite plan targets the same node more than once
    with different actions, or when a destructive action targets both an
    ancestor and a node it contains.

    The tree is never modified when this is raised.
    """

    def __init__(
        self,
        message: str,
        *,
        same_node: list[tuple["Selector", str, str]] | None = None,
        nested: list[tuple["Selector", str, "Selector", str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.same_node = same_node or []
        self.nested = nested or []


class InvalidFragmentError(TreeRewriteError):
    """Raised when a replacement fragment cannot be parsed for the document
    type of the node it is meant to replace."""


class SafeXMLParser(etree.XMLParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("resolve_entities", False)
        super().__init__(*args, **kwargs)


class CTGroupValue(TypedDict):
    _parser: type[etree.XMLParser | html.HTMLParser]
    _csstranslator: GenericTranslator | HTMLTranslator
    _tostring_method: _TostringMethodType


_ctgroup: dict[str, CTGroupValue] = {
    "html": {
        "_parser": html.HTMLParser,
        "_csstranslator": HTMLTranslator(),
        "_tostring_method": "html",
    },
    "xml": {
        "_parser": SafeXMLParser,
        "_csstranslator": GenericTranslator(),
        "_tostring_method": "xml",
    },
}


def _xml_or_html(type_: str | None) -> str:
    return "xml" if type_ == "xml" else "html"


_ENCODING_ERROR_TYPES = frozenset(
    {etree.ErrorTypes.ERR_INVALID_CHAR, etree.ErrorTypes.ERR_INVALID_ENCODING}
)


def _has_encoding_error(parser: _ParserType) -> bool:
    return any(error.type in _ENCODING_ERROR_TYPES for error in parser.error_log)


def _parse_html_fragment(
    fragment: str,
) -> tuple[str, list[etree._Element], str]:
    """Parse an HTML fragment into ``(leading_text, elements, trailing_text)``.

    HTML fragments are allowed to contain several root elements as well as
    text before the first element and after the last element.
    """
    try:
        parts = html.fragments_fromstring(fragment)
    except (etree.LxmlError, TypeError) as exc:
        raise InvalidFragmentError(f"Invalid HTML fragment: {fragment!r}") from exc
    if parts and isinstance(parts[0], str):
        leading = typing.cast("str", parts.pop(0))
    else:
        leading = ""
    trailing = ""
    if parts:
        trailing = parts[-1].tail or ""
        parts[-1].tail = None
    return leading, typing.cast("list[etree._Element]", parts), trailing


def _parse_xml_fragment(
    fragment: str,
) -> tuple[str, list[etree._Element], str]:
    """Parse an XML fragment into ``(leading_text, elements, trailing_text)``.

    XML fragments must contain exactly one root element, following XML
    document rules. Text-only fragments, multiple root elements and text
    outside the root element are rejected.
    """
    try:
        wrapper = etree.fromstring(
            f"<__parsel_fragment__>{fragment}</__parsel_fragment__>",
            parser=SafeXMLParser(),
        )
    except etree.LxmlError as exc:
        raise InvalidFragmentError(f"Invalid XML fragment: {fragment!r}") from exc
    children = list(wrapper)
    if len(children) != 1 or (wrapper.text or "").strip() or any(
        child.tail and child.tail.strip() for child in children
    ):
        raise InvalidFragmentError(
            f"XML fragments must contain exactly one root element: {fragment!r}"
        )
    return "", [children[0]], ""


def _parse_fragment(
    fragment: str, type_: str
) -> tuple[str, list[etree._Element], str]:
    if type_ == "xml":
        return _parse_xml_fragment(fragment)
    return _parse_html_fragment(fragment)


def _root_from_decoded_body(
    body: bytes,
    parser_cls: type[_ParserType],
    base_url: str | None,
    huge_tree: bool,
    encoding: str,
) -> etree._Element:
    """Create a root node from *body*, with undecodable bytes replaced."""
    return create_root_node(
        body.decode(encoding, errors="replace"),
        parser_cls,
        base_url=base_url,
        huge_tree=huge_tree,
    )


_XML_DECLARATION = re.compile(r"[\s\ufeff]*<\?xml\s")


def _detect_xml_or_html(text: str) -> str:
    return "xml" if _XML_DECLARATION.match(text) else "html"


def create_root_node(
    text: str,
    parser_cls: type[_ParserType],
    base_url: str | None = None,
    huge_tree: bool = True,
    body: bytes = b"",
    encoding: str = "utf-8",
) -> etree._Element:
    """Create root node for text using given parser class."""
    if text:
        body = text.strip().replace("\x00", "").encode(encoding) or b"<html/>"
    elif codecs.lookup(encoding).name != "utf-8":
        # lxml handles bytes that the declared encoding cannot decode by
        # truncating the document there, and characters that span several
        # bytes, as in UTF-16, are broken by the null and whitespace stripping
        # below. Only UTF-8 bytes are passed as is, since lxml does report
        # invalid UTF-8, and it covers most input.
        return _root_from_decoded_body(body, parser_cls, base_url, huge_tree, encoding)
    else:
        body = body.replace(b"\x00", b"").strip() or b"<html/>"

    parser = parser_cls(recover=True, encoding=encoding, huge_tree=huge_tree)
    root = None
    try:
        root = etree.fromstring(body, parser=parser, base_url=base_url)
    except etree.XMLSyntaxError:
        if text or not _has_encoding_error(parser):
            raise
    if not text and _has_encoding_error(parser):
        # Invalid bytes reach the tree as is, and reading them raises
        # UnicodeDecodeError from lxml.
        return _root_from_decoded_body(body, parser_cls, base_url, huge_tree, encoding)
    if not huge_tree:
        for error in parser.error_log:
            if "use XML_PARSE_HUGE option" in error.message:
                warnings.warn(
                    "Input data is too big. Set huge_tree=True for huge_tree support.",
                    stacklevel=2,
                )
    if root is None:
        root = etree.fromstring(b"<html/>", parser=parser, base_url=base_url)
    return root


class SelectorList(list[_SelectorType]):
    """
    The :class:`SelectorList` class is a subclass of the builtin ``list``
    class, which provides a few additional methods.
    """

    @typing.overload
    def __getitem__(self, pos: SupportsIndex) -> _SelectorType:
        pass

    @typing.overload
    def __getitem__(self, pos: slice) -> SelectorList[_SelectorType]:
        pass

    def __getitem__(
        self, pos: SupportsIndex | slice
    ) -> _SelectorType | SelectorList[_SelectorType]:
        o = super().__getitem__(pos)
        if isinstance(pos, slice):
            return self.__class__(typing.cast("SelectorList[_SelectorType]", o))
        return typing.cast("_SelectorType", o)

    def __getstate__(self) -> None:
        raise TypeError("can't pickle SelectorList objects")

    def jmespath(self, query: str, **kwargs: Any) -> SelectorList[_SelectorType]:
        """
        Call the ``.jmespath()`` method for each element in this list and return
        their results flattened as another :class:`SelectorList`.

        ``query`` is the same argument as the one in :meth:`Selector.jmespath`.

        Any additional named arguments are passed to the underlying
        ``jmespath.search`` call, e.g.::

            selector.jmespath('author.name', options=jmespath.Options(dict_cls=collections.OrderedDict))
        """
        return self.__class__(flatten([x.jmespath(query, **kwargs) for x in self]))

    def xpath(
        self,
        xpath: str,
        namespaces: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> SelectorList[_SelectorType]:
        """
        Call the ``.xpath()`` method for each element in this list and return
        their results flattened as another :class:`SelectorList`.

        ``xpath`` is the same argument as the one in :meth:`Selector.xpath`

        ``namespaces`` is an optional ``prefix: namespace-uri`` mapping (dict)
        for additional prefixes to those registered with ``register_namespace(prefix, uri)``.
        Contrary to ``register_namespace()``, these prefixes are not
        saved for future calls.

        Any additional named arguments can be used to pass values for XPath
        variables in the XPath expression, e.g.::

            selector.xpath('//a[href=$url]', url="http://www.example.com")
        """
        return self.__class__(
            flatten([x.xpath(xpath, namespaces=namespaces, **kwargs) for x in self])
        )

    def css(self, query: str) -> SelectorList[_SelectorType]:
        """
        Call the ``.css()`` method for each element in this list and return
        their results flattened as another :class:`SelectorList`.

        ``query`` is the same argument as the one in :meth:`Selector.css`
        """
        return self.__class__(flatten([x.css(query) for x in self]))

    def re(self, regex: str | Pattern[str], replace_entities: bool = True) -> list[str]:
        """
        Call the ``.re()`` method for each element in this list and return
        their results flattened, as a list of strings.

        By default, character entity references are replaced by their
        corresponding character (except for ``&amp;`` and ``&lt;``).
        Passing ``replace_entities`` as ``False`` switches off these
        replacements.
        """
        return flatten([x.re(regex, replace_entities=replace_entities) for x in self])

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: None = None,
        replace_entities: bool = True,
    ) -> str | None:
        pass

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: str,
        replace_entities: bool = True,
    ) -> str:
        pass

    def re_first(
        self,
        regex: str | Pattern[str],
        default: str | None = None,
        replace_entities: bool = True,
    ) -> str | None:
        """
        Call the ``.re()`` method for elements in this list and return the
        first matching result as a string. If the list is empty or the
        regex doesn't match anything, return the default value (``None`` if
        the argument is not provided).

        By default, character entity references are replaced by their
        corresponding character (except for ``&amp;`` and ``&lt;``).
        Passing ``replace_entities`` as ``False`` switches off these
        replacements.
        """
        for el in iflatten(
            x.re(regex, replace_entities=replace_entities) for x in self
        ):
            return typing.cast("str", el)
        return default

    def getall(self) -> list[str]:
        """
        Call the ``.get()`` method for each element in this list and return
        their results flattened, as a list of strings.
        """
        return [x.get() for x in self]

    extract = getall

    @typing.overload
    def get(self, default: None = None) -> str | None:
        pass

    @typing.overload
    def get(self, default: str) -> str:
        pass

    def get(self, default: str | None = None) -> Any:
        """
        Return the result of ``.get()`` for the first element in this list.
        If the list is empty, return the default value.
        """
        for x in self:
            return x.get()
        return default

    extract_first = get

    @property
    def attrib(self) -> Mapping[str, str]:
        """Return the attributes dictionary for the first element.
        If the list is empty, return an empty dict.
        """
        for x in self:
            return x.attrib
        return {}

    def drop(self) -> None:
        """
        Drop matched nodes from the parent for each element in this list.
        """
        for x in self:
            x.drop()

    def rewrite(self) -> TreeRewrite:
        """Return an empty :class:`TreeRewrite` plan.

        See the :class:`TreeRewrite` documentation for how to schedule
        multiple deletions, unwrappings and fragment replacements that are
        resolved against the original tree and applied in a single commit.
        """
        return TreeRewrite()


_ACTION_DROP = "drop"
_ACTION_UNWRAP = "unwrap"
_ACTION_REPLACE = "replace"
_DESTRUCTIVE_ACTIONS = frozenset({_ACTION_DROP, _ACTION_REPLACE})


def _rewrite_target_element(selector: Selector) -> etree._Element:
    """Return the element targeted by a rewrite action, rejecting selectors
    that point at text nodes, attributes or non markup documents."""
    if selector.type not in ("html", "xml"):
        raise CannotRemoveElementWithoutRoot(
            "Tree rewrites only work on Selector objects of the html or xml "
            "type, with an element as the selected node."
        )
    root = selector.root
    if not isinstance(root, etree._Element):
        raise CannotRemoveElementWithoutRoot(
            "The node you're trying to rewrite is not an element, "
            "are you trying to rewrite a pseudo-element? "
            "Try to use 'li' as a selector instead of 'li::text' or "
            "'//li' instead of '//li/text()', for example."
        )
    return root


def _drop_element(element: etree._Element) -> None:
    """Remove *element* while keeping its tail text next to the siblings."""
    parent = element.getparent()
    if parent is None:
        raise CannotDropElementWithoutParent(
            "The node you're trying to remove has no parent, "
            "are you trying to remove a root element?"
        )
    tail = element.tail
    if tail:
        previous = element.getprevious()
        if previous is not None:
            previous.tail = (previous.tail or "") + tail
        else:
            parent.text = (parent.text or "") + tail
    parent.remove(element)


def _unwrap_element(element: etree._Element) -> None:
    """Replace *element* with its children, keeping all surrounding text."""
    parent = element.getparent()
    if parent is None:
        raise CannotRemoveElementWithoutParent(
            "The node you're trying to unwrap has no parent, "
            "are you trying to unwrap a root element?"
        )
    index = parent.index(element)
    element_text = element.text
    tail = element.tail
    children = list(element)
    parent.remove(element)
    if not children:
        # Nothing of the element survives; its text and tail collapse the
        # same way they do when an element is dropped.
        trailing = (element_text or "") + (tail or "")
        if trailing:
            if index:
                previous = parent[index - 1]
                previous.tail = (previous.tail or "") + trailing
            else:
                parent.text = (parent.text or "") + trailing
        return
    for offset, child in enumerate(children):
        parent.insert(index + offset, child)
    first = children[0]
    first.text = (element_text or "") + (first.text or "")
    last = children[-1]
    last.tail = (last.tail or "") + (tail or "")


def _replace_element(
    element: etree._Element,
    leading: str,
    nodes: list[etree._Element],
    trailing: str,
) -> None:
    """Replace *element* with parsed fragment nodes and surrounding text."""
    parent = element.getparent()
    if parent is None:
        raise CannotRemoveElementWithoutParent(
            "The node you're trying to replace has no parent, "
            "are you trying to replace a root element?"
        )
    index = parent.index(element)
    tail = element.tail
    parent.remove(element)
    if leading:
        if index:
            previous = parent[index - 1]
            previous.tail = (previous.tail or "") + leading
        else:
            parent.text = (parent.text or "") + leading
    for offset, node in enumerate(nodes):
        parent.insert(index + offset, node)
    # The original tail follows the whole fragment; fragment trailing text
    # precedes it, so neither piece can overwrite the other.
    trailing = trailing + (tail or "")
    if trailing:
        if nodes:
            nodes[-1].tail = (nodes[-1].tail or "") + trailing
        elif index:
            previous = parent[index - 1]
            previous.tail = (previous.tail or "") + trailing
        else:
            parent.text = (parent.text or "") + trailing


class TreeRewrite:
    """A plan of tree modifications applied in a single commit.

    Create a plan through :meth:`Selector.rewrite` or
    :meth:`SelectorList.rewrite`, schedule actions with :meth:`drop`,
    :meth:`unwrap` and :meth:`replace`, then apply everything at once with
    :meth:`commit` or by using the plan as a context manager::

        with sel.css(".ad").rewrite() as plan:
            plan.drop(sel.css(".ad"))
            plan.unwrap(sel.css(".ad > span"))
            plan.replace(sel.css("span.note"), "<strong>note</strong>")

    All targets are resolved from the original tree before anything is
    modified. The same element selected several times is only acted upon once,
    while different actions on overlapping ancestor/descendant nodes are
    reported as a :class:`RewriteConflictError` before the tree is touched.
    Pass ``on_conflict="ancestor"`` to :meth:`commit` to explicitly keep the
    ancestor action and discard the conflicting descendant ones.

    HTML replacement fragments may contain several root elements and
    surrounding text; XML replacement fragments must contain exactly one root
    element and are validated with the XML parser.
    """

    def __init__(self) -> None:
        self._entries: dict[etree._Element, tuple[Selector, str, str | None]] = {}
        self._committed = False

    def _add(
        self,
        selectors: Selector | SelectorList[Selector],
        action: str,
        fragment: str | None = None,
    ) -> Self:
        items = (
            [selectors] if isinstance(selectors, Selector) else list(selectors)
        )
        for selector in items:
            element = _rewrite_target_element(selector)
            previous = self._entries.get(element)
            if previous is not None:
                _previous_selector, previous_action, previous_fragment = previous
                if previous_action == action and previous_fragment == fragment:
                    # The same node selected more than once is acted once.
                    continue
                raise RewriteConflictError(
                    "The same node is scheduled with different actions: "
                    f"{previous_action!r} and {action!r}."
                )
            self._entries[element] = (selector, action, fragment)
        return self

    def drop(self, selectors: Selector | SelectorList[Selector]) -> Self:
        """Schedule the selected elements for removal.

        The tail text of every removed element is kept next to the remaining
        siblings, in both HTML and XML documents.
        """
        return self._add(selectors, _ACTION_DROP)

    def unwrap(self, selectors: Selector | SelectorList[Selector]) -> Self:
        """Schedule the selected elements to be unwrapped in place.

        Each element is replaced by its own children. Its text becomes the
        leading text of the first child, and its tail becomes the tail of the
        last child, so no surrounding text is lost.
        """
        return self._add(selectors, _ACTION_UNWRAP)

    def replace(
        self, selectors: Selector | SelectorList[Selector], fragment: str
    ) -> Self:
        """Schedule the selected elements to be replaced by *fragment*.

        The fragment is parsed according to the type of each targeted
        document: HTML fragments allow multiple root elements and leading or
        trailing text, while XML fragments must contain exactly one root
        element. The tail of the replaced element is moved after the fragment.
        """
        if not isinstance(fragment, str):
            raise TypeError("fragment must be a string")
        return self._add(selectors, _ACTION_REPLACE, fragment)

    def commit(
        self,
        *,
        on_conflict: str = "error",
    ) -> None:
        """Validate and apply all scheduled actions in one pass.

        ``on_conflict`` is ``"error"`` by default, which raises
        :class:`RewriteConflictError` for overlapping destructive actions.
        Use ``"ancestor"`` to keep the action of the outermost element and
        discard the conflicting descendant actions.

        If the plan is invalid, no element is modified.
        """
        if on_conflict not in ("error", "ancestor"):
            raise ValueError(
                f"Invalid on_conflict value: {on_conflict!r} "
                "(expected 'error' or 'ancestor')."
            )
        if self._committed:
            return

        nested: list[tuple[Selector, str, Selector, str]] = []
        for element, (selector, action, _fragment) in self._entries.items():
            if action not in _DESTRUCTIVE_ACTIONS:
                continue
            ancestor = element.getparent()
            while ancestor is not None:
                ancestor_entry = self._entries.get(ancestor)
                if (
                    ancestor_entry is not None
                    and ancestor_entry[1] in _DESTRUCTIVE_ACTIONS
                ):
                    nested.append(
                        (
                            ancestor_entry[0],
                            ancestor_entry[1],
                            selector,
                            action,
                        )
                    )
                ancestor = ancestor.getparent()

        kept = dict(self._entries)
        if nested:
            if on_conflict == "error":
                details = "; ".join(
                    f"{ancestor_action}({ancestor.root.tag!r}) conflicts "
                    f"with {descendant_action}({descendant.root.tag!r})"
                    for (
                        ancestor,
                        ancestor_action,
                        descendant,
                        descendant_action,
                    ) in nested
                )
                raise RewriteConflictError(
                    "Conflicting actions on ancestor and descendant nodes: "
                    f"{details}. Pass on_conflict='ancestor' to keep the "
                    "ancestor actions.",
                    nested=nested,
                )
            for _ancestor, _a_action, descendant, _d_action in nested:
                kept.pop(descendant.root, None)

        # Parse every fragment and re-check the parents before touching a
        # tree, so a bad fragment leaves all documents unchanged.
        parsed: dict[etree._Element, tuple[str, list[etree._Element], str]] = {}
        for element, (selector, action, fragment) in kept.items():
            if element.getparent() is None:
                if action == _ACTION_DROP:
                    raise CannotDropElementWithoutParent(
                        "The node you're trying to remove has no parent, "
                        "are you trying to remove a root element?"
                    )
                raise CannotRemoveElementWithoutParent(
                    "The node you're trying to rewrite has no parent, "
                    "are you trying to rewrite a root element?"
                )
            if action == _ACTION_REPLACE:
                assert fragment is not None  # nosec
                parsed[element] = _parse_fragment(fragment, selector.type)

        # Apply deepest nodes first so that moving children never invalidates
        # the targets of remaining actions.
        for element, _entry in self._order_by_depth(kept):
            action = _entry[1]
            if action == _ACTION_DROP:
                _drop_element(element)
            elif action == _ACTION_UNWRAP:
                _unwrap_element(element)
            else:
                leading, nodes, trailing = parsed[element]
                _replace_element(element, leading, nodes, trailing)
        self._committed = True

    @staticmethod
    def _order_by_depth(
        entries: dict[etree._Element, tuple[Selector, str, str | None]]
    ) -> list[tuple[etree._Element, tuple[Selector, str, str | None]]]:
        def depth(element: etree._Element) -> int:
            count = 0
            parent = element.getparent()
            while parent is not None:
                count += 1
                parent = parent.getparent()
            return count

        return sorted(entries.items(), key=lambda item: depth(item[0]), reverse=True)

    def __enter__(self) -> TreeRewrite:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> None:
        if exc_type is None:
            self.commit()


_NOT_SET = object()


def _get_root_from_text(text: str, *, type_: str, **lxml_kwargs: Any) -> etree._Element:
    return create_root_node(text, _ctgroup[type_]["_parser"], **lxml_kwargs)


def _get_root_and_type_from_bytes(
    body: bytes,
    encoding: str,
    *,
    input_type: str | None,
    **lxml_kwargs: Any,
) -> tuple[Any, str]:
    if input_type == "text":
        return body.decode(encoding), input_type
    if input_type in ("json", None) and codecs.lookup(encoding).name == "utf-8":
        try:
            data = json.load(BytesIO(body))
        except ValueError:
            data = _NOT_SET
        if data is not _NOT_SET and (input_type == "json" or _is_json_document(data)):
            return data, "json"
    if input_type == "json":
        return None, "json"
    assert input_type in ("html", "xml", None)  # nosec
    type_ = input_type or _detect_xml_or_html(body[:256].decode(encoding, "ignore"))
    root = create_root_node(
        text="",
        body=body,
        encoding=encoding,
        parser_cls=_ctgroup[type_]["_parser"],
        **lxml_kwargs,
    )
    return root, type_


def _get_root_and_type_from_text(
    text: str, *, input_type: str | None, **lxml_kwargs: Any
) -> tuple[Any, str]:
    if input_type == "text":
        return text, input_type
    if input_type in ("json", None):
        try:
            data = json.loads(text)
        except ValueError:
            data = _NOT_SET
        if data is not _NOT_SET and (input_type == "json" or _is_json_document(data)):
            return data, "json"
    if input_type == "json":
        return None, "json"
    assert input_type in ("html", "xml", None)  # nosec
    type_ = input_type or _detect_xml_or_html(text)
    root = _get_root_from_text(text, type_=type_, **lxml_kwargs)
    return root, type_


# lxml registers the functions of these namespaces on every call of an
# evaluator that maps them, and with libxml2 2.13+ every registration after the
# first adds an entry to the evaluator's error log, which cannot be cleared.
_EXSLT_NAMESPACES = frozenset(
    f"http://exslt.org/{name}"
    for name in ("dates-and-times", "math", "sets", "strings")
)


@lru_cache(maxsize=2048)
def _compile_xpath(
    query: str, namespaces: tuple[tuple[str, str], ...], smart_strings: bool
) -> etree.XPath:
    return etree.XPath(query, namespaces=dict(namespaces), smart_strings=smart_strings)


def _get_xpath_evaluator(
    query: str, namespaces: dict[str, str], smart_strings: bool
) -> etree.XPath:
    namespaces = {
        prefix: uri
        for prefix, uri in namespaces.items()
        if uri not in _EXSLT_NAMESPACES
        or not isinstance(query, str)
        or f"{prefix}:" in query
    }
    if _EXSLT_NAMESPACES.intersection(namespaces.values()):
        return etree.XPath(query, namespaces=namespaces, smart_strings=smart_strings)
    return _compile_xpath(query, tuple(sorted(namespaces.items())), smart_strings)


def _get_root_type(root: Any, *, input_type: str | None) -> str:
    if isinstance(root, etree._Element):
        if input_type in {"json", "text"}:
            raise ValueError(
                f"Selector got an lxml.etree._Element object as root, "
                f"and {input_type!r} as type."
            )
        return _xml_or_html(input_type)
    return input_type or "json"


def _is_json_document(data: Any) -> bool:
    """Return whether *data* is a JSON object or array.

    Scalar JSON values are valid JSON documents as well, but they are also
    valid text and HTML, which is what they usually are when type detection
    gets to see them, so type detection ignores them.
    """
    return isinstance(data, (dict, list))


def _load_json_or_none(text: str | None) -> Any:
    try:
        return json.loads(text)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class Selector:
    """Wrapper for input data in HTML, JSON, or XML format, that allows
    selecting parts of it using selection expressions.

    You can write selection expressions in CSS or XPath for HTML and XML
    inputs, or in JMESPath for JSON inputs.

    ``text`` is an ``str`` object.

    ``body`` is a ``bytes`` or ``bytearray`` object. It can be used together with the
    ``encoding`` argument instead of the ``text`` argument.

    ``type`` defines the selector type. It can be ``"html"``, ``"json"``,
    ``"xml"`` or ``"text"``. If not specified, the input is handled as
    ``"json"`` if it is a JSON object or array, as ``"xml"`` if it starts with
    an XML declaration, and as ``"html"`` otherwise.

    ``base_url`` allows setting a URL for the document. This is needed when looking up external entities with relative paths.
    See the documentation for :func:`lxml.etree.fromstring` for more information.

    ``huge_tree`` controls the lxml/libxml2 feature that forbids parsing
    certain large documents to protect from possible memory exhaustion. The
    argument is ``True`` by default, which disables the protection to allow
    parsing such documents. Set it to ``False`` if you want to enable the
    protection. See `this lxml FAQ entry
    <https://lxml.de/FAQ.html#is-lxml-vulnerable-to-xml-bombs>`_ for more
    information.
    """

    __slots__ = [
        "__weakref__",
        "_expr",
        "_huge_tree",
        "_text",
        "body",
        "namespaces",
        "root",
        "type",
    ]

    _default_namespaces = {
        "re": "http://exslt.org/regular-expressions",
        # supported in libxslt:
        # set:difference
        # set:has-same-node
        # set:intersection
        # set:leading
        # set:trailing
        "set": "http://exslt.org/sets",
    }
    _lxml_smart_strings = False
    selectorlist_cls = SelectorList["Selector"]

    def __init__(
        self,
        text: str | None = None,
        type: str | None = None,  # noqa: A002
        body: bytes | bytearray = b"",
        encoding: str = "utf-8",
        namespaces: Mapping[str, str] | None = None,
        root: Any | None = _NOT_SET,
        base_url: str | None = None,
        _expr: str | None = None,
        huge_tree: bool = True,
    ) -> None:
        self.root: Any
        if type not in ("html", "json", "text", "xml", None):
            raise ValueError(f"Invalid type: {type}")

        if text is not None:
            if root is not _NOT_SET:
                warnings.warn(
                    "Selector got both text and root, root is being ignored.",
                    stacklevel=2,
                )
            if not isinstance(text, str):
                msg = f"text argument should be of type str, got {text.__class__}"
                raise TypeError(msg)

            root, type = _get_root_and_type_from_text(  # noqa: A001
                text,
                input_type=type,
                base_url=base_url,
                huge_tree=huge_tree,
            )
            self.root = root
            self.type = type
        elif body:
            if not isinstance(body, (bytes, bytearray)):
                msg = f"body argument should be of type bytes or bytearray, got {body.__class__}"
                raise TypeError(msg)
            root, type = _get_root_and_type_from_bytes(  # noqa: A001
                body=bytes(body),
                encoding=encoding,
                input_type=type,
                base_url=base_url,
                huge_tree=huge_tree,
            )
            self.root = root
            self.type = type
        elif root is _NOT_SET:
            raise ValueError("Selector needs text, body, or root arguments")
        else:
            self.root = root
            self.type = _get_root_type(root, input_type=type)

        self.namespaces = dict(self._default_namespaces)
        if namespaces is not None:
            self.namespaces.update(namespaces)

        self._expr = _expr
        self._huge_tree = huge_tree
        self._text = text

    def __getstate__(self) -> Any:
        raise TypeError("can't pickle Selector objects")

    def _get_root(
        self,
        text: str = "",
        base_url: str | None = None,
        huge_tree: bool = True,
        type_: str | None = None,
        body: bytes = b"",
        encoding: str = "utf-8",
    ) -> etree._Element:
        return create_root_node(
            text,
            body=body,
            encoding=encoding,
            parser_cls=_ctgroup[type_ or self.type]["_parser"],
            base_url=base_url,
            huge_tree=huge_tree,
        )

    def jmespath(
        self,
        query: str,
        **kwargs: Any,
    ) -> SelectorList[Self]:
        """
        Find objects matching the JMESPath ``query`` and return the result as a
        :class:`SelectorList` instance with all elements flattened. List
        elements implement :class:`Selector` interface too.

        ``query`` is a string containing the `JMESPath
        <https://jmespath.org/>`_ query to apply.

        Any additional named arguments are passed to the underlying
        ``jmespath.search`` call, e.g.::

            selector.jmespath('author.name', options=jmespath.Options(dict_cls=collections.OrderedDict))
        """
        if isinstance(self.root, str):
            data = _load_json_or_none(self.root)
        elif self.type == "json":
            data = self.root
        else:
            assert self.type in {"html", "xml"}  # nosec
            data = _load_json_or_none(self.root.text)

        result = jmespath.search(query, data, **kwargs)
        if result is None:
            result = []
        elif not isinstance(result, list):
            result = [result]

        def make_selector(x: Any) -> Selector:  # closure function
            if isinstance(x, str):
                return self.__class__(text=x, _expr=query, type="text")
            return self.__class__(root=x, _expr=query)

        result = [make_selector(x) for x in result]
        return typing.cast("SelectorList[Self]", self.selectorlist_cls(result))

    def xpath(
        self,
        query: str,
        namespaces: Mapping[str, str] | None = None,
        **kwargs: Any,
    ) -> SelectorList[Self]:
        """
        Find nodes matching the xpath ``query`` and return the result as a
        :class:`SelectorList` instance with all elements flattened. List
        elements implement :class:`Selector` interface too.

        ``query`` is a string containing the XPATH query to apply.

        ``namespaces`` is an optional ``prefix: namespace-uri`` mapping (dict)
        for additional prefixes to those registered with ``register_namespace(prefix, uri)``.
        Contrary to ``register_namespace()``, these prefixes are not
        saved for future calls.

        Any additional named arguments can be used to pass values for XPath
        variables in the XPath expression, e.g.::

            selector.xpath('//a[href=$url]', url="http://www.example.com")
        """
        if self.type not in ("html", "xml", "text"):
            raise ValueError(f"Cannot use xpath on a Selector of type {self.type!r}")
        if self.type in ("html", "xml"):
            root = self.root
            if not hasattr(root, "xpath"):
                if isinstance(root, str) and query.strip() == ".":
                    return typing.cast(
                        "SelectorList[Self]",
                        self.selectorlist_cls(
                            [
                                self.__class__(
                                    root=self.root,
                                    _expr=query,
                                    namespaces=self.namespaces,
                                    type=self.type,
                                )
                            ]
                        ),
                    )
                return typing.cast("SelectorList[Self]", self.selectorlist_cls([]))
        else:
            root = self._get_root(self._text or "", type_="html")

        nsp = dict(self.namespaces)
        if namespaces is not None:
            nsp.update(namespaces)
        try:
            xpathev = _get_xpath_evaluator(query, nsp, self._lxml_smart_strings)
            result = xpathev(root, **kwargs)
        except etree.XPathError as exc:
            raise ValueError(f"XPath error: {exc} in {query}")

        if not isinstance(result, list):
            result = [result]

        result = [
            self.__class__(
                root=x,
                _expr=query,
                namespaces=self.namespaces,
                type=_xml_or_html(self.type),
            )
            for x in result
        ]
        return typing.cast("SelectorList[Self]", self.selectorlist_cls(result))

    def css(self, query: str) -> SelectorList[Self]:
        """
        Apply the given CSS selector and return a :class:`SelectorList` instance.

        ``query`` is a string containing the CSS selector to apply.

        In the background, CSS queries are translated into XPath queries using
        `cssselect`_ library and run ``.xpath()`` method.

        .. _cssselect: https://pypi.python.org/pypi/cssselect/
        """
        if self.type not in ("html", "xml", "text"):
            raise ValueError(f"Cannot use css on a Selector of type {self.type!r}")
        return self.xpath(self._css2xpath(query))

    def _css2xpath(self, query: str) -> str:
        type_ = _xml_or_html(self.type)
        return _ctgroup[type_]["_csstranslator"].css_to_xpath(query)

    def re(self, regex: str | Pattern[str], replace_entities: bool = True) -> list[str]:
        """
        Apply the given regex and return a list of strings with the
        matches.

        ``regex`` can be either a compiled regular expression or a string which
        will be compiled to a regular expression using ``re.compile(regex)``.

        By default, character entity references are replaced by their
        corresponding character (except for ``&amp;`` and ``&lt;``).
        Passing ``replace_entities`` as ``False`` switches off these
        replacements.
        """
        data = self.get()
        return extract_regex(regex, data, replace_entities=replace_entities)

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: None = None,
        replace_entities: bool = True,
    ) -> str | None:
        pass

    @typing.overload
    def re_first(
        self,
        regex: str | Pattern[str],
        default: str,
        replace_entities: bool = True,
    ) -> str:
        pass

    def re_first(
        self,
        regex: str | Pattern[str],
        default: str | None = None,
        replace_entities: bool = True,
    ) -> str | None:
        """
        Apply the given regex and return the first string which matches. If
        there is no match, return the default value (``None`` if the argument
        is not provided).

        By default, character entity references are replaced by their
        corresponding character (except for ``&amp;`` and ``&lt;``).
        Passing ``replace_entities`` as ``False`` switches off these
        replacements.
        """
        return next(
            iflatten(self.re(regex, replace_entities=replace_entities)),
            default,
        )

    def get(self) -> Any:
        """
        Serialize and return the matched nodes.

        For HTML and XML, the result is always a string.
        """
        if self.type in ("text", "json"):
            return self.root
        try:
            return etree.tostring(
                self.root,
                method=_ctgroup[self.type]["_tostring_method"],
                encoding="unicode",
                with_tail=False,
            )
        except (AttributeError, TypeError):
            if self.root is True:
                return "1"
            if self.root is False:
                return "0"
            return str(self.root)

    extract = get

    def getall(self) -> list[str]:
        """
        Serialize and return the matched node in a 1-element list of strings.
        """
        return [self.get()]

    def register_namespace(self, prefix: str, uri: str) -> None:
        """
        Register the given namespace to be used in this :class:`Selector`.
        Without registering namespaces you can't select or extract data from
        non-standard namespaces. See :ref:`selector-examples-xml`.
        """
        self.namespaces[prefix] = uri

    def remove_namespaces(self) -> None:
        """
        Remove all namespaces, allowing to traverse the document using
        namespace-less xpaths. See :ref:`removing-namespaces`.
        For JSON selectors, this method does nothing.
        """
        if self.type == "json":
            return

        for el in self.root.iter("*"):
            if el.tag.startswith("{"):
                el.tag = el.tag.split("}", 1)[1]
            # loop on element attributes also
            for an in el.attrib:
                if an.startswith("{"):
                    el.attrib[an.split("}", 1)[1]] = el.attrib.pop(an)
        # remove namespace declarations
        etree.cleanup_namespaces(self.root)

    def drop(self) -> None:
        """
        Drop matched nodes from the parent element.
        """
        try:
            parent = self.root.getparent()
        except AttributeError:
            # 'str' object has no attribute 'getparent'
            raise CannotRemoveElementWithoutRoot(
                "The node you're trying to drop has no root, "
                "are you trying to drop a pseudo-element? "
                "Try to use 'li' as a selector instead of 'li::text' or "
                "'//li' instead of '//li/text()', for example."
            )

        no_parent_message = (
            "The node you're trying to remove has no parent, "
            "are you trying to remove a root element?"
        )
        try:
            if self.type == "xml":
                if parent is None:
                    raise CannotDropElementWithoutParent(no_parent_message)
                # Unlike HtmlElement.drop_tree(), _Element.remove() also
                # removes the tail text, so we need to preserve it manually
                if self.root.tail:
                    previous = self.root.getprevious()
                    if previous is not None:
                        previous.tail = (previous.tail or "") + self.root.tail
                    else:
                        parent.text = (parent.text or "") + self.root.tail
                parent.remove(self.root)
            else:
                typing.cast("html.HtmlElement", self.root).drop_tree()
        except (AttributeError, AssertionError):
            # 'NoneType' object has no attribute 'drop_tree'
            raise CannotDropElementWithoutParent(no_parent_message)

    def rewrite(self) -> TreeRewrite:
        """Return an empty :class:`TreeRewrite` plan.

        The plan is independent of this selector; pass selectors to its
        :meth:`TreeRewrite.drop`, :meth:`TreeRewrite.unwrap` and
        :meth:`TreeRewrite.replace` methods, then call
        :meth:`TreeRewrite.commit` to apply everything in one pass.
        """
        return TreeRewrite()

    @property
    def attrib(self) -> dict[str, str]:
        """
        Return the attributes dictionary for underlying element.
        For JSON selectors, return an empty dict.
        """
        if self.type == "json":
            return {}

        return dict(self.root.attrib)

    def __bool__(self) -> bool:
        """
        Return ``True`` if there is any real content selected or ``False``
        otherwise.  In other words, the boolean value of a :class:`Selector` is
        given by the contents it selects.
        """
        return bool(self.get())

    __nonzero__ = __bool__

    def __str__(self) -> str:
        return str(self.get())

    def __repr__(self) -> str:
        data = repr(shorten(str(self.get()), width=40))
        return f"<{type(self).__name__} query={self._expr!r} data={data}>"
