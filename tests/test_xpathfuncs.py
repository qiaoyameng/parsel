from __future__ import annotations

import threading
from typing import Any

import pytest

from parsel import Selector
from parsel.xpathfuncs import XPathFunctionSession, set_xpathfunc


def test_has_class_simple() -> None:
    body = """
    <p class="foo bar-baz">First</p>
    <p class="foo">Second</p>
    <p class="bar">Third</p>
    <p>Fourth</p>
    """
    sel = Selector(text=body)
    assert [x.extract() for x in sel.xpath('//p[has-class("foo")]/text()')] == [
        "First",
        "Second",
    ]
    assert [x.extract() for x in sel.xpath('//p[has-class("bar")]/text()')] == ["Third"]
    assert [x.extract() for x in sel.xpath('//p[has-class("foo","bar")]/text()')] == []
    assert [
        x.extract() for x in sel.xpath('//p[has-class("foo","bar-baz")]/text()')
    ] == ["First"]


def test_has_class_error_no_args() -> None:
    body = """
    <p CLASS="foo">First</p>
    """
    sel = Selector(text=body)
    with pytest.raises(ValueError, match="has-class must have at least 1 argument"):
        sel.xpath("has-class()")


def test_has_class_error_invalid_arg_type() -> None:
    body = """
    <p CLASS="foo">First</p>
    """
    sel = Selector(text=body)
    with pytest.raises(ValueError, match="has-class arguments must be strings"):
        sel.xpath("has-class(.)")


def test_has_class_error_invalid_unicode() -> None:
    body = """
    <p CLASS="foo">First</p>
    """
    sel = Selector(text=body)
    with pytest.raises(ValueError, match="All strings must be XML compatible"):
        sel.xpath('has-class("héllö")'.encode())  # type: ignore[arg-type]


def test_has_class_unicode() -> None:
    body = """
    <p CLASS="fóó">First</p>
    """
    sel = Selector(text=body)
    assert [x.extract() for x in sel.xpath('//p[has-class("fóó")]/text()')] == ["First"]


def test_has_class_uppercase() -> None:
    body = """
    <p CLASS="foo">First</p>
    """
    sel = Selector(text=body)
    assert [x.extract() for x in sel.xpath('//p[has-class("foo")]/text()')] == ["First"]


def test_has_class_newline() -> None:
    body = """
    <p CLASS="foo
    bar">First</p>
    """
    sel = Selector(text=body)
    assert [x.extract() for x in sel.xpath('//p[has-class("foo")]/text()')] == ["First"]


def test_has_class_tab() -> None:
    body = """
    <p CLASS="foo\tbar">First</p>
    """
    sel = Selector(text=body)
    assert [x.extract() for x in sel.xpath('//p[has-class("foo")]/text()')] == ["First"]


def test_set_xpathfunc() -> None:
    def myfunc(ctx: Any) -> None:
        myfunc.call_count += 1  # type: ignore[attr-defined]

    myfunc.call_count = 0  # type: ignore[attr-defined]

    body = """
    <p CLASS="foo">First</p>
    """
    sel = Selector(text=body)
    with pytest.raises(ValueError, match=r"Unregistered function(: myfunc)? in myfunc"):
        sel.xpath("myfunc()")

    set_xpathfunc("myfunc", myfunc)
    sel.xpath("myfunc()")
    assert myfunc.call_count == 1  # type: ignore[attr-defined]

    set_xpathfunc("myfunc", None)
    with pytest.raises(ValueError, match=r"Unregistered function(: myfunc)? in myfunc"):
        sel.xpath("myfunc()")


def test_session_interleaved_same_name() -> None:
    calls: list[str] = []

    def func_a(ctx: Any) -> str:
        calls.append("a")
        return "A"

    def func_b(ctx: Any) -> str:
        calls.append("b")
        return "B"

    sel = Selector(text="<p>x</p>")
    session_a = XPathFunctionSession({"f": func_a})
    session_b = XPathFunctionSession({"f": func_b})

    with session_a:
        assert sel.xpath("f()").get() == "A"  # cold cache
        assert sel.xpath("f()").get() == "A"  # hot cache
        with session_b:
            assert sel.xpath("f()").get() == "B"  # cold cache
            assert sel.xpath("f()").get() == "B"  # hot cache
        assert sel.xpath("f()").get() == "A"  # outer session unaffected
    assert calls == ["a", "a", "b", "b", "a"]


def test_session_nested_query_from_function() -> None:
    calls: list[str] = []

    def inner_func(ctx: Any) -> str:
        calls.append("inner")
        return "inner-result"

    def outer_func(ctx: Any) -> str:
        calls.append("outer")
        with XPathFunctionSession({"g": inner_func}):
            return Selector(text="<p>y</p>").xpath("g()").get()

    sel = Selector(text="<p>x</p>")
    with XPathFunctionSession({"f": outer_func}):
        assert sel.xpath("f()").get() == "inner-result"
        # The inner session is gone; g is not visible here.
        with pytest.raises(ValueError, match="Unregistered function"):
            sel.xpath("g()")
    assert calls == ["outer", "inner"]


def test_session_threads() -> None:
    barrier = threading.Barrier(2)
    results: dict[str, str] = {}
    calls: list[str] = []
    sel = Selector(text="<p>x</p>")

    def make_func(tag: str) -> Any:
        def func(ctx: Any) -> str:
            calls.append(tag)
            barrier.wait()
            return tag

        return func

    def run(tag: str) -> None:
        with XPathFunctionSession({"f": make_func(tag)}):
            barrier.wait()
            results[tag] = sel.xpath("f()").get()

    threads = [
        threading.Thread(target=run, args=(tag,)) for tag in ("first", "second")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results == {"first": "first", "second": "second"}
    assert sorted(calls) == ["first", "second"]


def test_session_function_error() -> None:
    class FuncError(Exception):
        pass

    error = FuncError("boom")

    def bad(ctx: Any) -> None:
        raise error

    sel = Selector(text="<p>x</p>")
    with XPathFunctionSession({"bad": bad}), pytest.raises(FuncError) as excinfo:
        sel.xpath("bad()")
    # The original exception propagates, it is not reported as an XPath
    # syntax error.
    assert excinfo.value is error


def test_session_sub_selectors() -> None:
    sel = Selector(text="<div><p>a</p><p>b</p></div>")
    with XPathFunctionSession({"f": lambda ctx: "ok"}):
        subs = sel.xpath("//p")
        assert [sub.xpath("f()").get() for sub in subs] == ["ok", "ok"]
        assert sel.xpath("//p")[0].xpath("f()").get() == "ok"


def test_session_cleanup_on_exit() -> None:
    sel = Selector(text="<p>x</p>")
    with XPathFunctionSession({"f": lambda ctx: "session"}) as session:
        result = sel.xpath("//p")
        assert sel.xpath("f()").get() == "session"
    # Internal references are dropped.
    assert not session._functions
    assert not session._cache
    # Results produced during the session remain usable.
    assert result.get() == "<p>x</p>"
    # The function is no longer registered.
    with pytest.raises(ValueError, match="Unregistered function"):
        sel.xpath("f()")


def test_session_namespaced_function() -> None:
    calls: list[str] = []

    def func(ctx: Any) -> str:
        calls.append("ns")
        return "namespaced"

    sel = Selector(text="<p>x</p>")
    session = XPathFunctionSession()
    session.register("f", func, namespace="http://example.com/ns")
    with session:
        assert (
            sel.xpath("p:f()", namespaces={"p": "http://example.com/ns"}).get()
            == "namespaced"
        )
        # The same local name in no namespace is a different function.
        with pytest.raises(ValueError, match="Unregistered function"):
            sel.xpath("f()")
    assert calls == ["ns"]


def test_session_does_not_leak_into_global_cache() -> None:
    calls: list[str] = []

    def session_func(ctx: Any) -> str:
        calls.append("session")
        return "session"

    sel = Selector(text='<p class="foo">x</p>')
    # Warm the shared compiled-expression cache with the same query string.
    with pytest.raises(ValueError, match="Unregistered function"):
        sel.xpath("leak-check()")
    with XPathFunctionSession({"leak-check": session_func}):
        assert sel.xpath("leak-check()").get() == "session"  # cold session cache
        assert sel.xpath("leak-check()").get() == "session"  # hot session cache
        # Globally registered functions remain available inside a session.
        assert sel.xpath('//p[has-class("foo")]').get() == '<p class="foo">x</p>'
    # The session-bound compiled expression must not leak into the shared
    # cache: the function is unregistered again outside the session.
    with pytest.raises(ValueError, match="Unregistered function"):
        sel.xpath("leak-check()")
    assert calls == ["session", "session"]


def test_session_global_function_precedence() -> None:
    # lxml resolves functions registered in the global function namespace
    # (set_xpathfunc) before per-evaluator extension functions.
    sel = Selector(text="<p>x</p>")
    set_xpathfunc("shared-name", lambda ctx: "global")
    try:
        with XPathFunctionSession({"shared-name": lambda ctx: "session"}):
            assert sel.xpath("shared-name()").get() == "global"
    finally:
        set_xpathfunc("shared-name", None)
