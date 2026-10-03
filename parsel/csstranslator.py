from __future__ import annotations

import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Protocol

from cssselect import GenericTranslator as OriginalGenericTranslator
from cssselect import HTMLTranslator as OriginalHTMLTranslator
from cssselect.parser import (
    CombinedSelector,
    EOFToken,
    Element,
    FunctionalPseudoElement,
    Pseudo,
    PseudoElement,
    Selector,
    Token,
    TokenStream,
    parse_selector_group,
    parse_series,
    tokenize,
)
from cssselect.xpath import ExpressionError, is_safe_name
from cssselect.xpath import XPathExpr as OriginalXPathExpr

if TYPE_CHECKING:
    # typing.Self requires Python 3.11
    from typing_extensions import Self


class _ScopeSentinel(Pseudo):
    """Replaces a ``:scope`` pseudo while compiling an in-predicate selector.

    It adds no condition on its own; the identity check against the scoping
    root is added by the caller.
    """


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

    # Pseudo-name used internally while translating
    # ":nth-child(An+B of selector-list)" and its nth-last-child variant.
    # cssselect 1.x does not parse the "of S" syntax, so it is parsed here
    # and encoded as this simple pseudo-class.
    _NTH_OF_PSEUDO_PREFIX = "parsel-nth-child-of-"

    # XPath 1.0 node-identity and subtree-containment tests against the
    # CSS scoping root, passed as the $parsel_scope XPath variable.
    _SCOPE_IDENTITY = "count(. | $parsel_scope) = count($parsel_scope)"

    def _translate(self, css: str, prefix: str) -> str:
        # Parsed ":nth-child(... of ...)" payloads, indexed by the number in
        # the generated pseudo-class name.
        self._nth_of_payloads: list[tuple[bool, int, int, list[Selector]]] = []
        try:
            tokens = self._transform_nth_child_of_tokens(list(tokenize(css)))
            stream = TokenStream(tokens, source=css)
            selectors = list(parse_selector_group(stream))
            return " | ".join(
                self.selector_to_xpath(
                    selector, prefix, translate_pseudo_elements=True
                )
                for selector in selectors
            )
        finally:
            self._nth_of_payloads = []

    def _transform_nth_child_of_tokens(self, tokens: list[Token]) -> list[Token]:
        """Rewrite ``:nth-child(An+B of S)`` tokens into simple pseudo-classes.

        The selector list ``S`` after ``of`` is parsed with the CSS selector
        grammar (never with string splitting), so commas, brackets, strings,
        nested functions and their parentheses are handled correctly.
        """
        result: list[Token] = []
        index = 0
        while index < len(tokens):
            colon = tokens[index]
            if (
                colon == ("DELIM", ":")
                and (index == 0 or tokens[index - 1] != ("DELIM", ":"))
                and index + 1 < len(tokens)
                and tokens[index + 1].type == "IDENT"
                and tokens[index + 1].value is not None
                and tokens[index + 1].value.lower()
                in (
                    "nth-child",
                    "nth-last-child",
                    "nth-of-type",
                    "nth-last-of-type",
                )
                and index + 2 < len(tokens)
                and tokens[index + 2] == ("DELIM", "(")
            ):
                name = tokens[index + 1].value.lower()
                body_start = index + 3
                depth = 1
                scan = body_start
                split_at: int | None = None
                bracket_depth = 0
                saw_whitespace = False
                last_series_token: Token | None = None
                while scan < len(tokens):
                    token = tokens[scan]
                    if token == ("DELIM", "("):
                        depth += 1
                    elif token == ("DELIM", ")"):
                        depth -= 1
                        if depth == 0:
                            break
                    elif token == ("DELIM", "[") and depth == 1:
                        bracket_depth += 1
                    elif token == ("DELIM", "]") and depth == 1:
                        bracket_depth -= 1
                    elif token.type == "S" and split_at is None:
                        saw_whitespace = True
                    elif (
                        depth == 1
                        and bracket_depth == 0
                        and split_at is None
                        and token.type == "IDENT"
                        and token.value is not None
                        and token.value.lower() == "of"
                    ):
                        # The An+B grammar requires whitespace before "of"
                        # unless the series ends in "n" (as in "2n of .a").
                        ends_in_n = (
                            last_series_token is not None
                            and last_series_token.type == "IDENT"
                            and last_series_token.value is not None
                            and last_series_token.value.lower().endswith("n")
                        )
                        if saw_whitespace or ends_in_n:
                            split_at = scan
                    if split_at is None and depth == 1 and token.type != "S":
                        last_series_token = token
                    scan += 1
                if split_at is None:
                    # No "of S" clause: keep the tokens untouched, cssselect
                    # handles plain :nth-child() / :nth-last-child().
                    end = scan + 1 if scan < len(tokens) else len(tokens)
                    result.extend(tokens[index:end])
                    index = end
                    continue

                if name in ("nth-of-type", "nth-last-of-type"):
                    raise ExpressionError(
                        f"The 'of S' syntax is not supported in :{name}(): "
                        "use :nth-child() or :nth-last-child()"
                    )

                # The An+B part ends at the first "of" keyword; whitespace
                # separates it from "of" (unless another token does).
                series_tokens = [
                    token
                    for token in tokens[body_start:split_at]
                    if token.type != "S"
                ]
                try:
                    a, b = parse_series(series_tokens)
                except ValueError as exc:
                    raise ExpressionError(
                        f"Invalid series in :{name}(): {exc}"
                    ) from exc

                selector_tokens = self._transform_nth_child_of_tokens(
                    tokens[split_at + 1 : scan]
                )
                if scan >= len(tokens):
                    raise ExpressionError(
                        f"Unterminated :{name}(): missing closing ')'"
                    )
                selector_tokens.append(EOFToken(tokens[scan].pos))
                selector_list = list(
                    parse_selector_group(TokenStream(selector_tokens))
                )
                if not selector_list:
                    raise ExpressionError(
                        f":{name}() requires a selector list after 'of'"
                    )
                for selector in selector_list:
                    if selector.pseudo_element is not None:
                        raise ExpressionError(
                            "Pseudo-elements are not supported in the selector "
                            f"list of :{name}(... of ...)"
                        )

                payload_index = len(self._nth_of_payloads)
                self._nth_of_payloads.append(
                    (name == "nth-last-child", a, b, selector_list)
                )
                result.append(tokens[index])
                result.append(
                    Token(
                        "IDENT",
                        f"{self._NTH_OF_PSEUDO_PREFIX}{payload_index}",
                        colon.pos,
                    )
                )
                index = scan + 1
                continue
            result.append(colon)
            index += 1
        return result

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

    def __getattr__(self, name: str) -> Any:
        # cssselect dispatches pseudo-class translation to methods named
        # xpath_<ident>_pseudo, but the generated pseudo-class names are
        # dynamic. Create the translation method on demand.
        prefix = "xpath_parsel_nth_child_of_"
        suffix = "_pseudo"
        if name.startswith(prefix) and name.endswith(suffix):
            middle = name[len(prefix) : -len(suffix)]
            if middle.isdigit():
                payload_index = int(middle)

                def translate(
                    xpath: OriginalXPathExpr,
                    payload_index: int = payload_index,
                ) -> OriginalXPathExpr:
                    payloads = getattr(self, "_nth_of_payloads", [])
                    if payload_index >= len(payloads):
                        raise ExpressionError(
                            f"The pseudo-class "
                            f":{self._NTH_OF_PSEUDO_PREFIX}{payload_index} "
                            "is unknown"
                        )
                    last, a, b, selector_list = payloads[payload_index]
                    return self._add_nth_child_of_condition(
                        xpath, last, a, b, selector_list
                    )

                return translate
        raise AttributeError(name)

    def _add_nth_child_of_condition(
        self,
        xpath: OriginalXPathExpr,
        last: bool,
        a: int,
        b: int,
        selector_list: list[Selector],
    ) -> OriginalXPathExpr:
        """Add the predicates for ``:nth-child(An+B of S)``.

        The candidate element must match one of the branches of ``S`` (its
        siblings are filtered first), and its 1-based position *within the
        filtered sibling subset* must satisfy ``An+B``. Repeated matches of
        the same element through different branches of ``S`` count once,
        because the branch conditions are combined with ``or`` inside a
        single node test rather than counted separately.
        """
        match_conditions = [
            self._match_condition(selector.parsed_tree)
            for selector in selector_list
        ]
        # An empty condition means the branch matches any element.
        match_conditions = [condition or "true()" for condition in match_conditions]
        if len(match_conditions) == 1:
            node_test = match_conditions[0]
        else:
            node_test = " or ".join(
                f"({condition})" for condition in match_conditions
            )

        conditions = [node_test]

        # Reproduce cssselect's series arithmetic, but count only the
        # preceding/following siblings that match the node test.
        axis = "following-sibling" if last else "preceding-sibling"
        siblings_count = f"count({axis}::*[{node_test}])"
        b_min_1 = b - 1
        if a == 1 and b_min_1 <= 0:
            series: list[str] = []
        elif a < 0 and b_min_1 < 0:
            return xpath.add_condition("0")
        elif a == 0:
            series = [f"{siblings_count} = {b_min_1}"]
        else:
            series = []
            if a > 0:
                if b_min_1 > 0:
                    series.append(f"{siblings_count} >= {b_min_1}")
            else:
                series.append(f"{siblings_count} <= {b_min_1}")
            if abs(a) != 1:
                left = siblings_count
                b_neg = (-b_min_1) % abs(a)
                if b_neg != 0:
                    left = f"({left} +{b_neg})"
                series.append(f"{left} mod {a} = 0")
        conditions.extend(series)
        predicate = " and ".join(f"({condition})" for condition in conditions)
        return xpath.add_condition(predicate)

    # Sentinel Pseudo node replacing :scope while compiling an in-predicate
    # selector to XPath: it adds no condition by itself, the identity check
    # against the scoping root is added around it.
    def xpath__scopesentinel(self, tree: _ScopeSentinel) -> XPathExpr:
        # The sentinel replaces :scope in an in-predicate selector. It adds
        # no condition itself: _compound_match_condition() adds the identity
        # check against the scoping root around the whole compound instead.
        return XPathExpr.from_xpath(self.xpath(tree.selector))

    def _without_scope(self, tree: Any) -> tuple[Any, bool]:
        """Return ``(tree, has_scope)`` with every :scope pseudo removed."""
        if isinstance(tree, Selector):
            # Relation.subselector wraps a full Selector.
            inner, found = self._without_scope(tree.parsed_tree)
            if found:
                tree = copy.copy(tree)
                tree.parsed_tree = inner
            return tree, found
        if isinstance(tree, Pseudo) and tree.ident == "scope":
            inner, _ = self._without_scope(tree.selector)
            return _ScopeSentinel(inner, "scope"), True
        found = False
        if hasattr(tree, "selector"):
            inner, inner_found = self._without_scope(tree.selector)
            found = inner_found
            if inner_found:
                tree = copy.copy(tree)
                tree.selector = inner
        if hasattr(tree, "subselector"):
            inner, inner_found = self._without_scope(tree.subselector)
            found = found or inner_found
            if inner_found:
                tree = copy.copy(tree)
                tree.subselector = inner
        return tree, found

    def _compound_match_condition(self, tree: Any) -> str:
        """XPath predicate for a compound (combinator-free) selector."""
        tree, has_scope = self._without_scope(tree)
        expression = self.xpath(tree)
        expression.add_name_test()
        condition = expression.condition
        if has_scope:
            condition = (
                f"({condition}) and {self._SCOPE_IDENTITY}"
                if condition
                else self._SCOPE_IDENTITY
            )
        return condition

    def _match_condition(self, tree: Any) -> str:
        """XPath predicate, evaluated on a candidate sibling element, that is
        true exactly when the candidate matches the parsed selector ``tree``.

        Ancestor/sibling structure imposed by combinators is expressed with
        reverse axes so the whole check is a predicate on the candidate.
        """
        if not isinstance(tree, CombinedSelector):
            return self._compound_match_condition(tree)

        left = self._match_condition(tree.selector) or "true()"
        right = self._match_condition(tree.subselector) or "true()"
        combinator = tree.combinator
        if combinator == " ":
            # Some ancestor matching the left branch.
            structure = f"ancestor::*[{left}]"
        elif combinator == ">":
            structure = f"parent::*[{left}]"
        elif combinator == "+":
            structure = f"preceding-sibling::*[position() = 1][{left}]"
        elif combinator == "~":
            structure = f"preceding-sibling::*[{left}]"
        else:
            raise ExpressionError(
                f"Unsupported combinator {combinator!r} in "
                ":nth-child(... of ...)"
            )
        condition = f"boolean({structure})"
        if right != "true()":
            condition = f"{condition} and ({right})"
        return condition

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
