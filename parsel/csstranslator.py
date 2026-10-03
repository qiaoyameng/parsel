from __future__ import annotations

from functools import lru_cache
import re
from typing import TYPE_CHECKING, Any, Protocol, cast

from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import (
    CombinedSelector,
    Element,
    FunctionalPseudoElement,
    Function,
    Pseudo,
    PseudoElement,
    Selector as ParsedSelector,
    SelectorSyntaxError,
    TokenStream,
    tokenize,
)
from cssselect.xpath import ExpressionError, is_safe_name, parse_series
from cssselect.xpath import XPathExpr as OriginalXPathExpr

if TYPE_CHECKING:
    # typing.Self requires Python 3.11
    from typing_extensions import Self


_NTH_CHILD_NAMES = {"nth-child", "nth-last-child"}
_NTH_CHILD_OF_RE = re.compile(
    r"nth-(?:last-)?child\([^)]*\bof\b", re.IGNORECASE
)


def _contains_scope(tree: Any) -> bool:
    if isinstance(tree, Pseudo) and tree.ident == "scope":
        return True
    if isinstance(tree, CombinedSelector):
        return _contains_scope(tree.selector) or _contains_scope(tree.subselector)

    for child_name in ("selector", "subselector"):
        child = getattr(tree, child_name, None)
        if hasattr(child, "parsed_tree"):
            if _contains_scope(child.parsed_tree):
                return True
        elif child is not None and hasattr(child, "selector"):
            if _contains_scope(child):
                return True

    if hasattr(tree, "selector_list"):
        if any(_contains_scope(selector) for selector in tree.selector_list):
            return True
    return False


def _find_close(tokens: list[Any], open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(tokens)):
        token = tokens[index]
        if token == ("DELIM", "("):
            depth += 1
        elif token == ("DELIM", ")"):
            depth -= 1
            if depth == 0:
                return index
    raise SelectorSyntaxError("Expected ')'")


def _transform_nth_child_tokens(
    tokens: list[Any],
    css: str,
    nth_children: dict[str, tuple[bool, list[ParsedSelector]]],
) -> list[Any]:
    transformed: list[Any] = []
    index = 0

    while index < len(tokens):
        token = tokens[index]
        is_nth_child = (
            token.type == "IDENT"
            and token.value.lower() in _NTH_CHILD_NAMES
            and index >= 1
            and tokens[index - 1] == ("DELIM", ":")
            and (index < 2 or tokens[index - 2] != ("DELIM", ":"))
            and index + 1 < len(tokens)
            and tokens[index + 1] == ("DELIM", "(")
        )
        if not is_nth_child:
            transformed.append(token)
            index += 1
            continue

        open_index = index + 1
        close_index = _find_close(tokens, open_index)
        of_index = _find_of_index(tokens, open_index, close_index)
        if of_index is None:
            transformed.extend(tokens[index : close_index + 1])
            index = close_index + 1
            continue

        series_tokens = [
            token
            for token in tokens[open_index + 1 : of_index]
            if token.type != "S"
        ]
        if not series_tokens:
            raise SelectorSyntaxError("Expected an An+B argument")
        try:
            parse_series(series_tokens)
        except ValueError as ex:
            raise SelectorSyntaxError(f"Invalid series: {series_tokens!r}") from ex

        selector_start = of_index + 1
        while (
            selector_start < close_index
            and tokens[selector_start].type == "S"
        ):
            selector_start += 1
        if selector_start >= close_index:
            raise SelectorSyntaxError("Expected a selector after 'of'")

        selector_css = css[tokens[selector_start].pos : tokens[close_index].pos]
        selector_tokens = list(tokenize(selector_css))
        selector_tokens = _transform_nth_child_tokens(
            selector_tokens, selector_css, nth_children
        )
        selector_stream = TokenStream(selector_tokens, selector_css)
        from cssselect.parser import parse_selector_group

        selector_list = list(parse_selector_group(selector_stream))
        nth_id = f"nth-{len(nth_children)}"
        nth_children[nth_id] = (token.value == "nth-last-child", selector_list)

        transformed.append(type(token)("IDENT", f"parsel-{nth_id}", token.pos))
        transformed.append(tokens[open_index])
        transformed.extend(series_tokens)
        transformed.append(tokens[close_index])
        index = close_index + 1

    return transformed


def _find_of_index(
    tokens: list[Any], open_index: int, close_index: int
) -> int | None:
    depth = 0

    for index in range(open_index + 1, close_index):
        token = tokens[index]
        if token == ("DELIM", "(") or token == ("DELIM", "["):
            depth += 1
        elif token == ("DELIM", ")") or token == ("DELIM", "]"):
            depth -= 1
        elif (
            depth == 0
            and token.type == "IDENT"
            and token.value.lower() == "of"
        ):
            try:
                parse_series(tokens[open_index + 1 : index])
            except ValueError:
                continue
            return index

    return None


def _parse_nth_child_selectors(
    css: str,
) -> tuple[
    list[ParsedSelector], dict[str, tuple[bool, list[ParsedSelector]]]
]:
    original_tokens = list(tokenize(css))
    nth_children: dict[str, tuple[bool, list[ParsedSelector]]] = {}
    transformed_tokens = _transform_nth_child_tokens(
        original_tokens, css, nth_children
    )
    stream = TokenStream(transformed_tokens, css)
    from cssselect.parser import parse_selector_group

    selectors = list(parse_selector_group(stream))
    _attach_nth_child_selectors(selectors, nth_children)
    return selectors, nth_children


def _attach_nth_child_selectors(
    selectors: list[ParsedSelector],
    nth_children: dict[str, tuple[bool, list[ParsedSelector]]],
) -> None:
    for selector in selectors:
        _attach_to_tree(selector.parsed_tree, nth_children)


def _attach_to_tree(
    tree: Any,
    nth_children: dict[str, tuple[bool, list[ParsedSelector]]],
) -> None:
    if isinstance(tree, Function) and tree.name.startswith("parsel-nth-"):
        nth_id = tree.name.removeprefix("parsel-")
        last, selector_list = nth_children[nth_id]
        setattr(tree, "parsel_nth_child_last", last)
        setattr(tree, "parsel_nth_child_selectors", selector_list)

    if hasattr(tree, "selector"):
        _attach_to_tree(tree.selector, nth_children)
    if hasattr(tree, "subselector") and not hasattr(tree.subselector, "parsed_tree"):
        _attach_to_tree(tree.subselector, nth_children)
    if isinstance(tree, CombinedSelector):
        _attach_to_tree(tree.subselector, nth_children)
    relation_subselector = getattr(tree, "subselector", None)
    if hasattr(relation_subselector, "parsed_tree"):
        _attach_to_tree(relation_subselector.parsed_tree, nth_children)

    for child in getattr(tree, "selector_list", ()):
        _attach_to_tree(child, nth_children)


class XPathExpr(OriginalXPathExpr):
    textnode: bool = False
    attribute: str | None = None

    @classmethod
    def from_xpath(
        cls,
        xpath: OriginalXPathExpr,
        textnode: bool = False,
        attribute: str | None = None,
    ) -> Self:
        x = cls(path=xpath.path, element=xpath.element, condition=xpath.condition)
        x.textnode = textnode
        x.attribute = attribute
        return x

    def __str__(self) -> str:
        path = super().__str__()
        if self.textnode:
            if path == "*":
                path = "text()"
            elif path.endswith("::*/*"):
                path = path[:-3] + "text()"
            else:
                path += "/text()"

        if self.attribute is not None:
            if path.endswith("::*/*"):
                path = path[:-2]
            if is_safe_name(self.attribute):
                path += f"/@{self.attribute}"
            else:
                literal = OriginalGenericTranslator.xpath_literal(self.attribute)
                path += f"/attribute::*[name() = {literal}]"

        return path

    def join(
        self: Self,
        combiner: str,
        other: OriginalXPathExpr,
        *args: Any,
        **kwargs: Any,
    ) -> Self:
        if not isinstance(other, XPathExpr):
            raise ValueError(
                f"Expressions of type {__name__}.XPathExpr can only join expressions"
                f" of the same type (or its descendants), got {type(other)}"
            )
        super().join(combiner, other, *args, **kwargs)
        self.textnode = other.textnode
        self.attribute = other.attribute
        return self


# e.g. cssselect.GenericTranslator, cssselect.HTMLTranslator
class TranslatorProtocol(Protocol):
    def xpath_element(self, selector: Element) -> OriginalXPathExpr: ...

    def css_to_xpath(self, css: str, prefix: str = ...) -> str: ...


class TranslatorMixin:
    """This mixin adds support to CSS pseudo elements via dynamic dispatch.

    Currently supported pseudo-elements are ``::text`` and ``::attr(ATTR_NAME)``.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._build_cache()

    # The cache is per instance because translation depends on instance state,
    # and because a cache shared by all instances, keyed on the instance, keeps
    # every translator ever used alive.
    def _build_cache(self) -> None:
        self._cache = lru_cache(maxsize=256)(self._translate)

    def _translate(self, css: str, prefix: str) -> str:
        if not _NTH_CHILD_OF_RE.search(css):
            # https://github.com/python/mypy/issues/14757
            return super().css_to_xpath(css, prefix)  # type: ignore[misc,no-any-return]

        selectors, _nth_children = _parse_nth_child_selectors(css)
        return " | ".join(
            self.selector_to_xpath(selector, prefix, translate_pseudo_elements=True)
            for selector in selectors
        )

    def xpath_function(self, function: Function) -> XPathExpr:
        if not function.name.startswith("parsel-nth-"):
            return cast("XPathExpr", super().xpath_function(function))

        selectors = cast(
            "list[ParsedSelector]", getattr(function, "parsel_nth_child_selectors")
        )
        last = cast("bool", getattr(function, "parsel_nth_child_last"))
        xpath = self.xpath(function.selector)
        return self._xpath_nth_child_of(xpath, function.arguments, selectors, last)

    def _xpath_nth_child_of(
        self,
        xpath: OriginalXPathExpr,
        arguments: Any,
        selectors: list[ParsedSelector],
        last: bool,
    ) -> XPathExpr:
        try:
            a, b = parse_series(arguments)
        except ValueError as ex:
            raise ExpressionError(f"Invalid series: {arguments!r}") from ex

        conditions = [
            condition or "true()"
            for condition in (
                self._nth_child_selector_condition(selector) for selector in selectors
            )
        ]

        current_matches = conditions[0] if len(conditions) == 1 else " or ".join(
            f"({condition})" for condition in conditions
        )
        axis = "following-sibling" if last else "preceding-sibling"
        siblings_count = f"count({axis}::*[{current_matches}])"

        b_min_1 = b - 1
        series_conditions: list[str] = []
        if a == 0:
            series_conditions.append(f"{siblings_count} = {b_min_1}")
        elif a < 0:
            series_conditions.append(f"{siblings_count} <= {b_min_1}")

        if abs(a) > 1:
            offset = (-b_min_1) % abs(a)
            left = f"({siblings_count} +{offset})" if offset else siblings_count
            series_conditions.append(f"{left} mod {abs(a)} = 0")
        if a > 0 and b_min_1 > 0:
            series_conditions.insert(0, f"{siblings_count} >= {b_min_1}")

        predicates = [current_matches]
        if a < 0 and b_min_1 < 0:
            predicates = ["0"]
        elif not (a == 1 and b_min_1 <= 0):
            predicates.extend(series_conditions)

        return cast(
            "XPathExpr",
            xpath.add_condition(
                " and ".join(f"({predicate})" for predicate in predicates)
            ),
        )

    def _nth_child_selector_condition(
        self, selector: ParsedSelector
    ) -> str | None:
        if selector.pseudo_element is not None:
            raise ExpressionError(
                "Pseudo-elements are not supported in nth-child() selector lists"
            )
        return self._selector_matches_condition(selector.parsed_tree)

    def _selector_matches_condition(self, tree: Any) -> str | None:
        if isinstance(tree, CombinedSelector):
            return self._combined_selector_matches_condition(tree)

        if _contains_scope(tree):
            raise ExpressionError(
                ":scope is not supported in nth-child() selector lists"
            )
        return self._compound_matches_condition(tree)

    def _combined_selector_matches_condition(
        self, combined: CombinedSelector
    ) -> str | None:
        right = self._selector_matches_condition(combined.subselector)
        if right is None:
            right_condition = ""
        else:
            right_condition = f" and ({right})"

        left = combined.selector
        if isinstance(left, CombinedSelector):
            left_condition = self._combined_selector_matches_condition(left)
        elif _contains_scope(left):
            raise ExpressionError(
                ":scope is not supported in nth-child() selector lists"
            )
        else:
            left_condition = self._compound_matches_condition(left)

        if combined.combinator == " ":
            if left_condition is None:
                relation = "ancestor::*"
            else:
                relation = f"ancestor::*[({left_condition})]"
        elif combined.combinator == ">":
            if left_condition is None:
                relation = "parent::*[1]"
            else:
                relation = f"parent::*[1][({left_condition})]"
        elif combined.combinator == "+":
            condition = left_condition or "self::*"
            relation = f"preceding-sibling::*[1][({condition})]"
        else:
            condition = left_condition or "self::*"
            relation = f"preceding-sibling::*[({condition})]"

        return f"{relation}{right_condition}"

    def _compound_matches_condition(self, tree: Any) -> str | None:
        xpath = self.xpath(tree)
        xpath.add_name_test()
        return xpath.condition or None

    def css_to_xpath(self, css: str, prefix: str = "descendant-or-self::") -> str:
        return self._cache(css, prefix)

    # The cache holds a reference to a bound method, which cannot be pickled.
    def __getstate__(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if k != "_cache"}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._build_cache()

    def xpath_element(self: TranslatorProtocol, selector: Element) -> XPathExpr:
        # https://github.com/python/mypy/issues/14757
        xpath = super().xpath_element(selector)  # type: ignore[safe-super]
        return XPathExpr.from_xpath(xpath)

    def xpath_pseudo_element(
        self, xpath: OriginalXPathExpr, pseudo_element: PseudoElement
    ) -> OriginalXPathExpr:
        """
        Dispatch method that transforms XPath to support pseudo-element
        """
        if isinstance(pseudo_element, FunctionalPseudoElement):
            method_name = f"xpath_{pseudo_element.name.replace('-', '_')}_functional_pseudo_element"
            method = getattr(self, method_name, None)
            if not method:
                raise ExpressionError(
                    f"The functional pseudo-element ::{pseudo_element.name}() is unknown"
                )
            xpath = method(xpath, pseudo_element)
        else:
            method_name = (
                f"xpath_{pseudo_element.replace('-', '_')}_simple_pseudo_element"
            )
            method = getattr(self, method_name, None)
            if not method:
                raise ExpressionError(
                    f"The pseudo-element ::{pseudo_element} is unknown"
                )
            xpath = method(xpath)
        return xpath

    def xpath_attr_functional_pseudo_element(
        self, xpath: OriginalXPathExpr, function: FunctionalPseudoElement
    ) -> XPathExpr:
        """Support selecting attribute values using ::attr() pseudo-element"""
        if function.argument_types() not in (["STRING"], ["IDENT"]):
            raise ExpressionError(
                f"Expected a single string or ident for ::attr(), got {function.arguments!r}"
            )
        return XPathExpr.from_xpath(xpath, attribute=function.arguments[0].value)

    def xpath_text_simple_pseudo_element(self, xpath: OriginalXPathExpr) -> XPathExpr:
        """Support selecting text nodes using ::text pseudo-element"""
        return XPathExpr.from_xpath(xpath, textnode=True)


class GenericTranslator(TranslatorMixin, OriginalGenericTranslator):
    pass


class HTMLTranslator(TranslatorMixin, OriginalHTMLTranslator):
    pass


_translator = HTMLTranslator()


def css2xpath(query: str) -> str:
    """Return translated XPath version of a given CSS query"""
    return _translator.css_to_xpath(query)
