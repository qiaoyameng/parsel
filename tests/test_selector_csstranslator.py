"""
Selector tests for cssselect backend
"""

from __future__ import annotations

import gc
import pickle
import weakref
from abc import ABC, abstractmethod
from typing import Any

import pytest
from cssselect.parser import SelectorSyntaxError
from cssselect.xpath import ExpressionError
from cssselect.xpath import XPathExpr as OriginalXPathExpr

from parsel import Selector, css2xpath
from parsel.csstranslator import (
    GenericTranslator,
    HTMLTranslator,
    TranslatorProtocol,
    XPathExpr,
)

HTMLBODY = """
<html>
<body>
<div>
 <a id="name-anchor" name="foo"></a>
 <a id="tag-anchor" rel="tag" href="http://localhost/foo">link</a>
 <a id="nofollow-anchor" rel="nofollow" href="https://example.org"> link</a>
 <p id="paragraph">
   lorem ipsum text
   <b id="p-b">hi</b> <em id="p-em">there</em>
   <b id="p-b2">guy</b>
   <input type="checkbox" id="checkbox-unchecked" />
   <input type="checkbox" id="checkbox-disabled" disabled="" />
   <input type="text" id="text-checked" checked="checked" />
   <input type="hidden" />
   <input type="hidden" disabled="disabled" />
   <input type="checkbox" id="checkbox-checked" checked="checked" />
   <input type="checkbox" id="checkbox-disabled-checked"
          disabled="disabled" checked="checked" />
   <fieldset id="fieldset" disabled="disabled">
     <input type="checkbox" id="checkbox-fieldset-disabled" />
     <input type="hidden" />
   </fieldset>
 </p>
 <map name="dummymap">
   <area shape="circle" coords="200,250,25" href="foo.html" id="area-href" />
   <area shape="default" id="area-nohref" />
 </map>
</div>
<div class="cool-footer" id="foobar-div" foobar="ab bc cde">
    <span id="foobar-span">foo ter</span>
</div>
</body></html>
"""


class TestTranslatorBase(ABC):
    @property
    @abstractmethod
    def tr_cls(self) -> type[TranslatorProtocol]:
        raise NotImplementedError

    def c2x(self, css: str) -> str:
        return self.tr_cls().css_to_xpath(css)

    @pytest.mark.parametrize(
        ("css", "xpath"),
        [
            ("::attr(name)", "descendant-or-self::*/@name"),
            ("a::attr(href)", "descendant-or-self::a/@href"),
            (
                "a ::attr(img)",
                "descendant-or-self::a/descendant-or-self::*/@img",
            ),
            ("a > ::attr(class)", "descendant-or-self::a/*/@class"),
            (
                r"a::attr(\:foo)",
                "descendant-or-self::a/attribute::*[name() = ':foo']",
            ),
            (
                r"a::attr(foo\:bar)",
                "descendant-or-self::a/attribute::*[name() = 'foo:bar']",
            ),
        ],
    )
    def test_attr_function(self, css: str, xpath: str) -> None:
        assert self.c2x(css) == xpath, css

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::attr(12)", ExpressionError),
            ("::attr(34test)", ExpressionError),
            ("::attr(@href)", SelectorSyntaxError),
        ],
    )
    def test_attr_function_exception(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "xpath"),
        [
            ("::text", "descendant-or-self::text()"),
            ("p::text", "descendant-or-self::p/text()"),
            ("p ::text", "descendant-or-self::p/descendant-or-self::text()"),
            ("#id::text", "descendant-or-self::*[@id = 'id']/text()"),
            ("p#id::text", "descendant-or-self::p[@id = 'id']/text()"),
            (
                "p#id ::text",
                "descendant-or-self::p[@id = 'id']/descendant-or-self::text()",
            ),
            ("p#id > ::text", "descendant-or-self::p[@id = 'id']/*/text()"),
            (
                "p#id ~ ::text",
                "descendant-or-self::p[@id = 'id']/following-sibling::*/text()",
            ),
            ("a[href]::text", "descendant-or-self::a[@href]/text()"),
            (
                "a[href] ::text",
                "descendant-or-self::a[@href]/descendant-or-self::text()",
            ),
            (
                "p::text, a::text",
                "descendant-or-self::p/text() | descendant-or-self::a/text()",
            ),
        ],
    )
    def test_text_pseudo_element(self, css: str, xpath: str) -> None:
        assert self.c2x(css) == xpath, css

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::attribute(12)", ExpressionError),
            ("::text()", ExpressionError),
            ("::attr(@href)", SelectorSyntaxError),
        ],
    )
    def test_pseudo_function_exception(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            ("::text-node", ExpressionError),
        ],
    )
    def test_unknown_pseudo_element(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)

    @pytest.mark.parametrize(
        ("css", "exc"),
        [
            (":text", ExpressionError),
            (":attribute(name)", ExpressionError),
        ],
    )
    def test_unknown_pseudo_class(self, css: str, exc: type[Exception]) -> None:
        with pytest.raises(exc):
            self.c2x(css)


class TestHTMLTranslator(TestTranslatorBase):
    tr_cls = HTMLTranslator


class TestGenericTranslator(TestTranslatorBase):
    tr_cls = GenericTranslator


@pytest.mark.parametrize("tr_cls", [GenericTranslator, HTMLTranslator])
class TestTranslatorCache:
    def test_cache(self, tr_cls: type[GenericTranslator | HTMLTranslator]) -> None:
        tr = tr_cls()
        assert tr.css_to_xpath("a::text") == tr.css_to_xpath("a::text")
        assert tr._cache.cache_info().hits == 1

    def test_garbage_collection(
        self, tr_cls: type[GenericTranslator | HTMLTranslator]
    ) -> None:
        tr = tr_cls()
        tr.css_to_xpath("a::text")
        ref = weakref.ref(tr)
        del tr
        gc.collect()
        assert ref() is None

    def test_pickle(self, tr_cls: type[GenericTranslator | HTMLTranslator]) -> None:
        tr = pickle.loads(pickle.dumps(tr_cls()))  # noqa: S301
        assert tr.css_to_xpath("a::text") == "descendant-or-self::a/text()"


def test_css2xpath() -> None:
    expected_xpath = (
        "descendant-or-self::*[@class and contains("
        "concat(' ', normalize-space(@class), ' '), ' some-class ')]"
    )
    assert css2xpath(".some-class") == expected_xpath


def test_xpathexpr_join_wrong_type() -> None:
    expr = XPathExpr(path="foo")
    with pytest.raises(ValueError, match="can only join expressions"):
        expr.join("/", OriginalXPathExpr(path="bar"))


class TestCSSSelector:
    sel = Selector(text=HTMLBODY)

    def x(self, *a: Any, **kw: Any) -> list[str]:
        return [v.strip() for v in self.sel.css(*a, **kw).extract() if v.strip()]

    def test_selector_simple(self) -> None:
        for x in self.sel.css("input"):
            assert isinstance(x, self.sel.__class__), x
        assert self.sel.css("input").extract() == [
            x.extract() for x in self.sel.css("input")
        ]

    def test_text_pseudo_element(self) -> None:
        assert self.x("#p-b2") == ['<b id="p-b2">guy</b>']
        assert self.x("#p-b2::text") == ["guy"]
        assert self.x("#p-b2 ::text") == ["guy"]
        assert self.x("#paragraph::text") == ["lorem ipsum text"]
        assert self.x("#paragraph ::text") == ["lorem ipsum text", "hi", "there", "guy"]
        assert self.x("p::text") == ["lorem ipsum text"]
        assert self.x("p ::text") == ["lorem ipsum text", "hi", "there", "guy"]

    def test_attribute_function(self) -> None:
        assert self.x("#p-b2::attr(id)") == ["p-b2"]
        assert self.x(".cool-footer::attr(class)") == ["cool-footer"]
        assert self.x(".cool-footer ::attr(id)") == ["foobar-div", "foobar-span"]
        assert self.x('map[name="dummymap"] ::attr(shape)') == ["circle", "default"]

    def test_nested_selector(self) -> None:
        assert self.sel.css("p").css("b::text").extract() == ["hi", "guy"]
        assert self.sel.css("div").css("area:last-child").extract() == [
            '<area shape="default" id="area-nohref">'
        ]

    def test_pseudoclass_has(self) -> None:
        assert self.x("p:has(b)::text") == ["lorem ipsum text"]


class TestCSSSelectorBytes(TestCSSSelector):
    sel = Selector(body=bytes(HTMLBODY, encoding="utf-8"))


NTH_CHILD_HTML = """
<ul>
  <li class="visible" id="one">1</li>
  <li id="hidden-one">h1</li>
  <li class="visible" id="two">2</li>
  <li class="visible extra" id="duplicate">d</li>
  <li data-priority id="priority">p</li>
</ul>
<section>
  <div><span class="target" id="deep">deep</span></div>
  <div class="group" id="group"></div>
  <p class="middle" id="middle"></p>
  <span class="target" id="adjacent">adjacent</span>
  <span class="target" id="following">following</span>
</section>
"""


@pytest.mark.parametrize("tr_cls", [GenericTranslator, HTMLTranslator])
class TestNthChildOf:
    def test_filter_before_counting(self, tr_cls: type[GenericTranslator]) -> None:
        sel = Selector(text=NTH_CHILD_HTML)
        query = "li:nth-child(odd of .visible, [data-priority])"
        assert sel.css(query).xpath("@id").getall() == [
            "one",
            "duplicate",
        ]

    def test_duplicate_selector_branch_counts_once(
        self, tr_cls: type[GenericTranslator]
    ) -> None:
        sel = Selector(text=NTH_CHILD_HTML)
        query = "li:nth-child(even of .visible, .visible.extra, [data-priority])"
        assert sel.css(query).xpath("@id").getall() == ["two", "priority"]

    @pytest.mark.parametrize(
        ("query", "expected"),
        [
            (
                "li:nth-last-child(odd of .visible, [data-priority])",
                ["two", "priority"],
            ),
            (
                "li:nth-child(even of .visible, [data-priority])",
                ["two", "priority"],
            ),
            ("li:nth-child(-n+2 of .visible, [data-priority])", ["one", "two"]),
        ],
    )
    def test_series(
        self,
        tr_cls: type[GenericTranslator],
        query: str,
        expected: list[str],
    ) -> None:
        sel = Selector(text=NTH_CHILD_HTML)
        assert sel.css(query).xpath("@id").getall() == expected

    def test_combined_selector(self, tr_cls: type[GenericTranslator]) -> None:
        sel = Selector(text=NTH_CHILD_HTML)
        queries = {
            "span:nth-child(1 of section div > .target, div .target)": ["deep"],
            "span:nth-child(1 of .group + .middle + .target)": ["adjacent"],
            "span:nth-child(2 of .group ~ .target)": ["following"],
        }
        for query, expected in queries.items():
            assert sel.css(query).xpath("@id").getall() == expected

    def test_pseudo_elements_and_cache(
        self, tr_cls: type[GenericTranslator]
    ) -> None:
        translator = tr_cls()
        query = "li:nth-child(1 of .visible)::text"
        assert translator.css_to_xpath(query) == translator.css_to_xpath(query)
        assert translator._cache.cache_info().hits == 1
        assert Selector(text=NTH_CHILD_HTML).css(query).getall() == ["1"]

        attr_query = "li:nth-child(1 of .visible)::attr(id)"
        assert Selector(text=NTH_CHILD_HTML).css(attr_query).getall() == ["one"]

    def test_namespace(self, tr_cls: type[GenericTranslator]) -> None:
        xml = """
        <root xmlns:n="http://example.com">
          <n:item class="visible" id="one"/>
          <n:item id="hidden"/>
          <n:item class="visible" id="two"/>
        </root>
        """
        sel = Selector(
            text=xml,
            type="xml",
            namespaces={"n": "http://example.com"},
        )
        query = "n|item:nth-child(even of n|item.visible)"
        assert sel.css(query).xpath("@id").getall() == ["two"]

    @pytest.mark.parametrize(
        "query",
        [
            "li:nth-child(1 of ::text)",
            "li:nth-child(1 of ::attr(id))",
            "li:nth-child(1 of :scope)",
            "li:nth-child(1 of > .visible)",
        ],
    )
    def test_unsupported_selector(
        self, tr_cls: type[GenericTranslator], query: str
    ) -> None:
        with pytest.raises((ExpressionError, SelectorSyntaxError)):
            tr_cls().css_to_xpath(query)

    def test_comma_in_pseudo_is_not_split(
        self, tr_cls: type[GenericTranslator]
    ) -> None:
        query = "li:nth-child(1 of .a[title='x,y'], .b)"
        xpath = tr_cls().css_to_xpath(query)
        assert "title = 'x,y'" in xpath
        assert " or " in xpath

    def test_type_and_class_named_of(
        self, tr_cls: type[GenericTranslator]
    ) -> None:
        sel = Selector(text='<ul><li class="of" id="of">of</li></ul>')
        query = "li:nth-child(1 of .of, :is(.of))"
        assert sel.css(query).xpath("@id").getall() == ["of"]
