# Basic usage of the Selector, strongly typed to test the typing of parsel's API.
from __future__ import annotations

import re

from parsel import NodeKind, Selector


def correct() -> None:
    selector = Selector(
        text="<html><body><ul><li>1</li><li>2</li><li>3</li></ul></body></html>"
    )

    li_values: list[str] = selector.css("li").getall()
    selector.re_first(re.compile(r"[32]"), "").strip()
    xpath_values: list[str] = selector.xpath(
        "//somens:a/text()", namespaces={"somens": "http://scrapy.org"}
    ).extract()

    class MySelector(Selector):
        def my_own_func(self) -> int:
            return 3

    my_selector = MySelector()
    res: int = my_selector.my_own_func()
    sub_res: int = my_selector.xpath("//somens:a/text()")[0].my_own_func()
    jsonpath_res: int = my_selector.jsonpath("$.a")[0].my_own_func()

    provenance_selector = Selector(
        text="<html><body><p>x</p></body></html>", provenance=True
    )
    provenance = provenance_selector.xpath("//p")[0].provenance
    provenance_kind: str = provenance.kind
    assert provenance_kind == NodeKind.ELEMENT
    located_value: object = provenance.locate().value
    assert located_value
    provenance_list = provenance_selector.xpath("//p").provenance
    located_list = provenance_selector.xpath("//p").locate()
    assert len(provenance_list) == len(located_list)


# Negative checks: all the code lines below have typing errors.
# the "# type: ignore" comment makes sure that mypy identifies them as errors.


def incorrect() -> None:
    selector = Selector(
        text="<html><body><ul><li>1</li><li>2</li><li>3</li></ul></body></html>"
    )

    # Wrong query type in css.
    selector.css(5).getall()  # type: ignore[arg-type]

    # Cannot assign a list of str to an int.
    li_values: int = selector.css("li").getall()  # type: ignore[assignment]

    # Cannot use a string to define namespaces in xpath.
    selector.xpath(
        "//somens:a/text()",
        namespaces='{"somens": "http://scrapy.org"}',  # type: ignore[arg-type]
    ).extract()

    # Typo in the extract method name.
    selector.css("li").extact()  # type: ignore[attr-defined]

    class MySelector(Selector):
        def my_own_func(self) -> int:
            return 3

    my_selector = MySelector()
    res: str = my_selector.my_own_func()  # type: ignore[assignment]
    sub_res: str = my_selector.xpath("//somens:a/text()")[0].my_own_func()  # type: ignore[assignment]
