"""Tests for the ``:nth-child(An+B of S)`` / ``:nth-last-child(...)`` syntax."""

from __future__ import annotations

import pickle

import pytest
from cssselect.parser import SelectorSyntaxError
from cssselect.xpath import ExpressionError

from parsel import Selector
from parsel.csstranslator import GenericTranslator, HTMLTranslator

# The .v subset has 3 elements at raw positions 1, 3, 5 which receive
# filtered positions 1, 2, 3. This distinguishes filter-first matching
# (even filtered position 2 -> raw item "3") from count-first matching
# (even raw positions 2, 4 -> the non-.v items "2", "4").
FILTERED_HTML = """
<ul id="u">
  <li class="v">1</li>
  <li>2</li>
  <li class="v">3</li>
  <li>4</li>
  <li class="v" data-priority="">5</li>
</ul>
"""


@pytest.mark.parametrize(
    ("css", "expected"),
    [
        ("li:nth-child(odd of .v)", ["1", "5"]),
        ("li:nth-child(2n+1 of .v)", ["1", "5"]),
        ("li:nth-child(even of .v)", ["3"]),
        ("li:nth-child(2n of .v)", ["3"]),
        ("li:nth-child(1 of .v)", ["1"]),
        ("li:nth-child(2 of .v)", ["3"]),
        ("li:nth-child(3 of .v)", ["5"]),
        ("li:nth-child(4 of .v)", []),
        ("li:nth-child(-n+2 of .v)", ["1", "3"]),
        ("li:nth-child(n+2 of .v)", ["3", "5"]),
        ("li:nth-child(0n+2 of .v)", ["3"]),
        ("li:nth-last-child(1 of .v)", ["5"]),
        ("li:nth-last-child(2 of .v)", ["3"]),
        ("li:nth-last-child(odd of .v)", ["1", "5"]),
        ("li:nth-last-child(even of .v)", ["3"]),
        (
            "li:nth-child(odd of .v, [data-priority])",
            ["1", "5"],
        ),
        (
            "li:nth-child(even of .v, [data-priority])",
            ["3"],
        ),
        ("li:nth-child(1 of [data-priority])", ["5"]),
        # The plain syntax keeps working unchanged.
        ("li:nth-child(odd)", ["1", "3", "5"]),
        ("li:nth-child(even)", ["2", "4"]),
        ("li:nth-last-child(1)", ["5"]),
    ],
)
def test_nth_child_of_filters_before_counting(
    css: str, expected: list[str]
) -> None:
    sel = Selector(text=FILTERED_HTML)
    assert sel.css(css + "::text").getall() == expected


# Two ul siblings: positions must be computed within each parent, not
# globally.
TWO_PARENTS_HTML = """
<ul><li class="v">A1</li><li>A2</li><li class="v">A3</li></ul>
<ul><li class="v">B1</li><li>B2</li><li class="v">B3</li></ul>
"""


def test_position_is_within_each_parent() -> None:
    sel = Selector(text=TWO_PARENTS_HTML)
    assert sel.css("li:nth-child(2 of .v)::text").getall() == ["A3", "B3"]
    assert sel.css("li:nth-last-child(2 of .v)::text").getall() == ["A1", "B1"]


# When several branches of the selector list match the same element it is
# counted once: the first element of the subset is still position 1 even if
# it matches every branch.
DEDUP_HTML = """
<ul>
  <li class="a b">X</li>
  <li class="a">Y</li>
  <li class="b">Z</li>
</ul>
"""


def test_selector_list_branches_deduplicate() -> None:
    sel = Selector(text=DEDUP_HTML)
    assert sel.css("li:nth-child(1 of .a, .b)::text").getall() == ["X"]
    assert sel.css("li:nth-child(2 of .a, .b)::text").getall() == ["Y"]
    assert sel.css("li:nth-child(3 of .a, .b)::text").getall() == ["Z"]
    # Element X matches both branches but counts once also for the subset
    # of .a only (X, Y): Y is the second .a element.
    assert sel.css("li:nth-child(2 of .a)::text").getall() == ["Y"]


# Commas and brackets inside the selector list must not split the argument.
def test_selectors_with_commas_and_brackets() -> None:
    sel = Selector(
        text="""
        <ul>
          <li data-v="a,b" class="v">P</li>
          <li data-v="other">Q</li>
          <li data-v="a,b" class="v">R</li>
        </ul>
        """
    )
    assert sel.css(
        "li:nth-child(2 of [data-v='a,b'].v)::text"
    ).getall() == ["R"]


def test_nested_nth_child_of() -> None:
    sel = Selector(
        text="""
        <ul>
          <li class="a"><b class="v">one</b></li>
          <li class="a"><b>two</b><b class="v">three</b></li>
        </ul>
        """
    )
    # Inside each li.a: first b.v is "one", then "three".
    assert sel.css("li:nth-child(2 of .a) b:nth-child(1 of .v)::text").getall() == [
        "three"
    ]


def test_nth_child_of_inside_not_and_is() -> None:
    sel = Selector(
        text="""
        <ul>
          <li class="a v">1</li>
          <li class="a">2</li>
          <li class="a v">3</li>
        </ul>
        """
    )
    assert sel.css("li:not(:nth-child(even of .v))::text").getall() == ["1", "2"]
    assert sel.css("li:is(:nth-child(2 of .v))::text").getall() == ["3"]


def test_selector_group_mixes_plain_and_of_forms() -> None:
    sel = Selector(text=FILTERED_HTML)
    assert sel.css(
        "li:nth-child(2)::text, li:nth-child(1 of .v)::text"
    ).getall() == ["1", "2"]
    assert sel.css(
        "li:nth-child(9)::text, li:nth-child(1 of .v)::text"
    ).getall() == ["1"]


def test_pseudo_elements_still_usable_on_the_result() -> None:
    sel = Selector(
        text="""
        <ul>
          <li class="v" data-id="a">one<span>tail</span></li>
          <li class="v" data-id="b">two</li>
        </ul>
        """
    )
    assert sel.css("li:nth-child(2 of .v)::attr(data-id)").getall() == ["b"]
    assert sel.css("li:nth-child(1 of .v) span::text").getall() == ["tail"]


def test_scope_restricts_the_selector_list() -> None:
    sel = Selector(
        text="""
        <div>
          <ul><li class="v">U1</li><li class="v">U2</li></ul>
        </div>
        <ul><li class="v">O1</li></ul>
        """
    )
    uls = sel.css("div ul")
    assert len(uls) == 1
    # :scope .v does not match li.v elements of the outer ul, which is a
    # sibling of the scope rather than its descendant.
    assert uls.css("li:nth-child(2 of :scope .v)::text").getall() == ["U2"]
    assert uls.css(":scope li:nth-child(2 of .v)::text").getall() == ["U2"]


def test_scope_child_combinator_in_selector_list() -> None:
    sel = Selector(
        text="""
        <ul>
          <li class="v">a</li>
          <li><span class="v">nested</span></li>
          <li class="v">b</li>
        </ul>
        """
    )
    ul = sel.css("ul")[0]
    # Only direct children of the scope are candidates; nested span.v is not.
    assert ul.css("li:nth-child(2 of :scope > .v)::text").getall() == ["b"]


COMBINATOR_HTML = """
<ul>
  <div></div>
  <li class="v">P1</li>
  <li class="v">P2</li>
  <div></div>
  <li class="v">Q1</li>
</ul>
"""


def test_combinators_in_selector_list() -> None:
    sel = Selector(text=COMBINATOR_HTML)
    # Elements immediately following a div: P1 and Q1, numbered within the
    # subset starting at 1 (P1 = 1, Q1 = 2).
    assert sel.css("li:nth-child(1 of div + .v)::text").getall() == ["P1"]
    assert sel.css("li:nth-child(2 of div + .v)::text").getall() == ["Q1"]
    # Subset {P1, P2, Q1} (preceded somewhere by a div), second is P2.
    assert sel.css("li:nth-child(2 of div ~ .v)::text").getall() == ["P2"]
    assert sel.css("li:nth-child(3 of div ~ .v)::text").getall() == ["Q1"]
    # Children of a ul (all li here) vs. the div>li variant excludes them.
    sel2 = Selector(
        text="<ul><li class='v'>solo</li></ul>"
    )
    assert sel2.css("li:nth-child(1 of ul > .v)::text").getall() == ["solo"]


def test_xml_namespace_in_selector_list() -> None:
    xml = """<root xmlns:ns="http://example.com/ns">
      <ns:item class="v">one</ns:item>
      <item>two</item>
      <ns:item class="v">three</ns:item>
    </root>"""
    sel = Selector(text=xml, type="xml")
    sel.register_namespace("n", "http://example.com/ns")
    assert sel.css("n|item:nth-child(2 of n|item.v)::text").getall() == ["three"]
    assert sel.css(
        "n|item:nth-last-child(1 of n|item.v)::text"
    ).getall() == ["three"]


@pytest.mark.parametrize("tr_cls", [HTMLTranslator, GenericTranslator])
def test_translation_cache_is_reusable(tr_cls: type) -> None:
    tr = tr_cls()
    first = tr.css_to_xpath("li:nth-child(2n+1 of .v, [data-x])")
    second = tr.css_to_xpath("li:nth-child(2n+1 of .v, [data-x])")
    assert first == second
    assert tr._cache.cache_info().hits == 1
    # Per-translation payload state must not leak between calls.
    assert tr._nth_of_payloads == []
    restored = pickle.loads(pickle.dumps(tr))  # noqa: S301
    assert restored.css_to_xpath("li:nth-child(2 of .v)") == tr.css_to_xpath(
        "li:nth-child(2 of .v)"
    )


@pytest.mark.parametrize("tr_cls", [HTMLTranslator, GenericTranslator])
@pytest.mark.parametrize(
    "css",
    [
        "li:nth-child(2 of ::text)",
        "li:nth-last-child(1 of .a::attr(href))",
        "li:nth-child(2 of)",
        "li:nth-child(of .a)",
        "li:nth-child(2 of :unknown-pseudo)",
        "li:nth-child(2 of .a,)",
        "li:nth-child(2 of",
        ":nth-of-type(2 of .a)",
        ":nth-last-of-type(2 of .a)",
    ],
)
def test_untranslatable_constructs_raise(tr_cls: type, css: str) -> None:
    with pytest.raises((ExpressionError, SelectorSyntaxError)):
        tr_cls().css_to_xpath(css)


def test_errors_do_not_look_like_empty_results() -> None:
    sel = Selector(text="<ul><li>x</li></ul>")
    with pytest.raises((ExpressionError, SelectorSyntaxError)):
        sel.css("li:nth-child(1 of ::text)")


def test_local_repeated_calls_performance() -> None:
    # Small-scale local smoke check: hundreds of repeated calls stay fast
    # thanks to the translation cache; no external benchmarking required.
    import time

    sel = Selector(text=FILTERED_HTML)
    query = "li:nth-child(2n+1 of .v, [data-priority])::text"
    sel.css(query).getall()  # populate the cache
    start = time.perf_counter()
    for _ in range(1000):
        assert sel.css(query).getall() == ["1", "5"]
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0
