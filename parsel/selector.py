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
    """Base class for errors raised by :class:`TreeRewritePlan`."""


class TreeRewriteConflict(TreeRewriteError):
    """Raised when planned actions target a node and one of its ancestors.

    Descendant actions can be dropped in favor of the ancestor actions by
    passing ``keep_ancestors=True`` when applying the plan.
    """


class InvalidTreeRewriteError(TreeRewriteError):
    """Raised when a rewrite plan targets nodes that cannot be rewritten."""


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

    def rewrite(
        self,
        action: typing.Literal["drop", "unwrap", "replace"],
        fragment: str | None = None,
        *,
        keep_ancestors: bool = False,
    ) -> None:
        """
        Apply one tree-rewriting *action* to every element in this list in a
        single commit.

        All targets are determined from the original tree first, the whole
        plan is validated before anything is modified, and either every
        action is applied or the tree is left untouched.

        ``action`` is one of:

        * ``"drop"``: remove each node. Its ``tail`` text is kept as the tail
          of the preceding node or as the parent's leading text, so adjacent
          removals never swallow unrelated text.
        * ``"unwrap"``: replace each node with its children. The node's
          ``text`` becomes the tail of the preceding sibling (or the parent's
          leading text), and its ``tail`` becomes the tail of the last moved
          child.
        * ``"replace"``: replace each node with the nodes parsed from
          *fragment*. The fragment may contain several roots and leading text.
          It is parsed as HTML for :class:`Selector` objects of ``html`` type
          and as a strict XML fragment for those of ``xml`` type; the two
          parsing modes are never mixed. The target node's ``tail`` becomes
          the tail of the last inserted node.

        ``fragment`` is required (and only accepted) with ``action="replace"``.

        ``keep_ancestors`` resolves conflicts where both an ancestor and one
        of its descendants are targeted: when ``False`` (the default), such a
        plan raises :class:`TreeRewriteConflict` before any modification;
        when ``True``, the actions on descendants are discarded in favor of
        the actions on their ancestors.

        Repeated occurrences of the same node only execute once. Targets from
        different documents are supported. Selectors that still point to
        surviving nodes keep working after the rewrite.

        See :class:`TreeRewritePlan` for mixing different actions in a single
        commit.
        """
        if action not in ("drop", "unwrap", "replace"):
            raise ValueError(
                f"Invalid rewrite action: {action!r} "
                "(expected 'drop', 'unwrap', or 'replace')"
            )
        if fragment is not None and action != "replace":
            raise ValueError(
                "The fragment argument is only used with 'replace', "
                f"not {action!r}"
            )
        if fragment is None and action == "replace":
            raise ValueError("A fragment is required with the 'replace' action")
        plan = TreeRewritePlan()
        if action == "drop":
            plan.drop(self)
        elif action == "unwrap":
            plan.unwrap(self)
        else:
            plan.replace(self, fragment or "")
        plan.apply(keep_ancestors=keep_ancestors)


_REWRITE_DROP = "drop"
_REWRITE_UNWRAP = "unwrap"
_REWRITE_REPLACE = "replace"


class TreeRewritePlan:
    """
    A single-commit plan of tree rewrites for :class:`Selector` targets.

    Actions are added with :meth:`drop`, :meth:`unwrap` and :meth:`replace`
    and applied atomically with :meth:`apply`. Nothing is modified until
    :meth:`apply` is called: all targets are resolved against the original
    tree, duplicates are collapsed, conflicts are reported, and replacement
    fragments are parsed. If any of these steps fails, every targeted tree is
    left exactly as it was.

    The plan can combine different actions and targets from more than one
    document. Each target element is acted upon at most once.
    """

    def __init__(
        self,
        selectors: typing.Iterable[Selector] | Selector | None = None,
    ) -> None:
        self._entries: dict[etree._Element, dict[str, Any]] = {}
        self._applied = False
        if selectors is not None:
            self.drop(selectors)

    def _add(
        self,
        selectors: typing.Iterable[Selector] | Selector,
        action: str,
        fragment: str | None = None,
    ) -> "TreeRewritePlan":
        if self._applied:
            raise RuntimeError("This TreeRewritePlan has already been applied")
        if isinstance(selectors, Selector):
            targets: list[Selector] = [selectors]
        elif isinstance(selectors, SelectorList):
            targets = list(selectors)
        else:
            targets = list(selectors)
        for selector in targets:
            if not isinstance(selector, Selector):
                raise TypeError(
                    f"Tree rewrite targets must be Selector objects, got {selector!r}"
                )
            node = selector.root
            if not isinstance(node, etree._Element):
                raise InvalidTreeRewriteError(
                    "The node you're trying to rewrite is not an element, "
                    "are you trying to rewrite a pseudo-element? "
                    "Try to use 'li' as a selector instead of 'li::text' or "
                    "'//li' instead of '//li/text()', for example."
                )
            if selector.type not in ("html", "xml"):
                raise InvalidTreeRewriteError(
                    f"Tree rewriting is only supported for html and xml selectors, "
                    f"got type {selector.type!r}"
                )
            previous = self._entries.get(node)
            if previous is not None and previous["action"] != action:
                raise TreeRewriteConflict(
                    f"The <{node.tag}> element is scheduled for more than one "
                    f"action: {previous['action']!r} and {action!r}"
                )
            if previous is None:
                self._entries[node] = {
                    "action": action,
                    "fragment": fragment,
                    "node": node,
                    "type": selector.type,
                }
        return self

    def drop(
        self, selectors: typing.Iterable[Selector] | Selector
    ) -> "TreeRewritePlan":
        """Schedule each target element for removal, keeping its ``tail`` text."""
        return self._add(selectors, _REWRITE_DROP)

    def unwrap(
        self, selectors: typing.Iterable[Selector] | Selector
    ) -> "TreeRewritePlan":
        """Schedule each target element for replacement with its children."""
        return self._add(selectors, _REWRITE_UNWRAP)

    def replace(
        self, selectors: typing.Iterable[Selector] | Selector, fragment: str
    ) -> "TreeRewritePlan":
        """Schedule each target element for replacement with *fragment*.

        HTML selectors parse *fragment* as an HTML fragment (leading text and
        multiple roots are allowed); xml selectors parse it as a strict XML
        fragment.
        """
        if not isinstance(fragment, str):
            raise TypeError(f"fragment must be a str, got {type(fragment).__name__}")
        return self._add(selectors, _REWRITE_REPLACE, fragment=fragment)

    def apply(self, *, keep_ancestors: bool = False) -> None:
        """Validate the plan and apply all of its actions at once.

        ``keep_ancestors`` controls the handling of plans that target both an
        ancestor and one of its descendants: with ``False`` (the default) a
        :class:`TreeRewriteConflict` is raised and no tree is modified; with
        ``True`` the descendant actions are discarded and the ancestor actions
        are applied.
        """
        if self._applied:
            raise RuntimeError("This TreeRewritePlan has already been applied")
        entries = self._validate()
        entries = self._resolve_conflicts(entries, keep_ancestors=keep_ancestors)
        self._parse_fragments(entries)
        self._execute(entries)
        self._applied = True

    def _validate(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for entry in self._entries.values():
            node = entry["node"]
            parent = node.getparent()
            if parent is None:
                raise InvalidTreeRewriteError(
                    "The node you're trying to rewrite has no parent, "
                    "are you trying to rewrite a root element?"
                )
            entries.append(entry)
        return entries

    @staticmethod
    def _resolve_conflicts(
        entries: list[dict[str, Any]], *, keep_ancestors: bool
    ) -> list[dict[str, Any]]:
        nodes = {entry["node"] for entry in entries}
        surviving: list[dict[str, Any]] = []
        for entry in entries:
            node = entry["node"]
            ancestor = node.getparent()
            while ancestor is not None and ancestor not in nodes:
                ancestor = ancestor.getparent()
            if ancestor is not None:
                if not keep_ancestors:
                    raise TreeRewriteConflict(
                        f"Conflicting rewrite actions on <{ancestor.tag}> and its "
                        f"descendant <{node.tag}>; pass keep_ancestors=True to apply "
                        "the ancestor actions and discard the descendant actions"
                    )
                continue
            surviving.append(entry)
        return surviving

    @staticmethod
    def _parse_fragment(fragment: str, type_: str) -> tuple[str, list[etree._Element]]:
        """Parse *fragment* into its leading text and ordered root elements.

        Leading text (text before the first element) is returned separately;
        interspersed and trailing text is kept as the ``tail`` of the elements,
        following lxml's document model.
        """
        if type_ == "html":
            items: list[Any] = list(html.fragments_fromstring(fragment))
        else:
            parser = SafeXMLParser(recover=False)
            try:
                wrapper = etree.fromstring(
                    f"<fragment>{fragment}</fragment>", parser=parser
                )
            except etree.XMLSyntaxError as exc:
                raise InvalidTreeRewriteError(f"Invalid XML fragment: {exc}") from exc
            items = ([wrapper.text] if wrapper.text else []) + list(wrapper)
        leading = ""
        elements: list[etree._Element] = []
        for item in items:
            if isinstance(item, etree._Element):
                elements.append(item)
            elif elements:
                elements[-1].tail = (elements[-1].tail or "") + item
            else:
                leading += item
        return leading, elements

    def _parse_fragments(self, entries: list[dict[str, Any]]) -> None:
        for entry in entries:
            if entry["action"] == _REWRITE_REPLACE:
                entry["leading"], entry["elements"] = self._parse_fragment(
                    entry["fragment"], entry["type"]
                )

    @staticmethod
    def _execute(entries: list[dict[str, Any]]) -> None:
        groups: dict[etree._Element, list[dict[str, Any]]] = {}
        for entry in entries:
            groups.setdefault(entry["node"].getparent(), []).append(entry)
        for parent, group in groups.items():
            parent = typing.cast("etree._Element", parent)
            order = {child: index for index, child in enumerate(parent)}
            for entry in sorted(
                group, key=lambda item: order[item["node"]], reverse=True
            ):
                TreeRewritePlan._apply_entry(parent, entry)

    @staticmethod
    def _append(
        parent: etree._Element, anchor: etree._Element | None, text: str | None
    ) -> None:
        """Append *text* at the position right after *anchor*.

        With *anchor* of ``None`` the text extends the parent's leading text.
        """
        if not text:
            return
        if anchor is not None:
            anchor.tail = (anchor.tail or "") + text
        else:
            parent.text = (parent.text or "") + text

    @staticmethod
    def _apply_entry(parent: etree._Element, entry: dict[str, Any]) -> None:
        node = entry["node"]
        action = entry["action"]
        # Anchor before the target in the original document order. Entries of
        # the same parent are applied from right to left, so earlier siblings
        # are untouched at this point.
        anchor = node.getprevious()
        last: etree._Element | None
        if action == _REWRITE_UNWRAP:
            children = list(node)
            # node.text precedes the moved children in document order.
            TreeRewritePlan._append(parent, anchor, node.text)
            for child in children:
                node.addprevious(child)
            last = children[-1] if children else None
        elif action == _REWRITE_REPLACE:
            TreeRewritePlan._append(parent, anchor, entry["leading"])
            for new_node in entry["elements"]:
                node.addprevious(new_node)
            last = entry["elements"][-1] if entry["elements"] else None
        else:
            last = None
        # node.tail follows the surviving content (or the original anchor).
        TreeRewritePlan._append(parent, last if last is not None else anchor, node.tail)
        parent.remove(node)


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
