from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Self

from lxml import etree
from w3lib.html import HTML5_WHITESPACE

from .selector import _compile_xpath, _xpath_function_session

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from contextvars import Token

regex = f"[{HTML5_WHITESPACE}]+"
replace_html5_whitespaces = re.compile(regex).sub


def set_xpathfunc(fname: str, func: Callable | None) -> None:  # type: ignore[type-arg]
    """Register a custom extension function to use in XPath expressions.

    The function ``func`` registered under ``fname`` identifier will be called
    for every matching node, being passed a ``context`` parameter as well as
    any parameters passed from the corresponding XPath expression.

    If ``func`` is ``None``, the extension function will be removed.

    See more `in lxml documentation`_.

    .. _`in lxml documentation`: https://lxml.de/extensions.html#xpath-extension-functions

    """
    ns_fns = etree.FunctionNamespace(None)
    if func is not None:
        ns_fns[fname] = func
    else:
        del ns_fns[fname]
        # Compiled expressions keep resolving functions that were registered
        # when they were first evaluated.
        _compile_xpath.cache_clear()


class XPathFunctionSession:
    """Query-session-scoped XPath extension function registry.

    Use it as a context manager to make a set of extension functions
    available to XPath queries without touching the process-wide function
    namespace used by :func:`set_xpathfunc`::

        with XPathFunctionSession({"my-func": my_func}):
            sel.xpath("my-func()")

    Functions are identified by a ``(namespace URI, local name)`` pair, so
    the mapping keys can also be ``(namespace, name)`` tuples, and
    :meth:`register` accepts a ``namespace`` argument.

    The session applies to every query made while it is active, including
    queries on :class:`~parsel.selector.Selector` objects produced by other
    queries. Nested sessions and concurrent threads each see their own
    functions; a session never sees the functions of another session, and
    the shared compiled-expression cache is never populated with
    session-bound expressions.

    Functions registered globally with :func:`set_xpathfunc` remain
    available inside a session, and take precedence over session functions
    with the same ``(namespace, name)`` identifier (an lxml limitation).

    Leaving the session drops its internal references (including its
    compiled-expression cache); results produced while it was active remain
    usable. A session is single-use: register the functions again or create
    a new session to re-enter.
    """

    def __init__(
        self,
        functions: Mapping[tuple[str | None, str] | str, Callable] | None = None,  # type: ignore[type-arg]
    ) -> None:
        self._functions: dict[tuple[str | None, str], Callable] = {}  # type: ignore[type-arg]
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...], bool], etree.XPath] = {}
        self._token: Token | None = None  # type: ignore[type-arg]
        if functions:
            for key, func in functions.items():
                if isinstance(key, tuple):
                    namespace, name = key
                else:
                    namespace, name = None, key
                self.register(name, func, namespace=namespace)

    def register(
        self, name: str, func: Callable, namespace: str | None = None  # type: ignore[type-arg]
    ) -> None:
        """Register ``func`` under the ``(namespace, name)`` identifier."""
        self._functions[(namespace, name)] = func
        # Compiled expressions bind the functions they were compiled with.
        self._cache.clear()

    def unregister(self, name: str, namespace: str | None = None) -> None:
        """Remove the function registered under ``(namespace, name)``."""
        del self._functions[(namespace, name)]
        self._cache.clear()

    def get_evaluator(
        self, query: str, namespaces: dict[str, str], smart_strings: bool
    ) -> etree.XPath:
        """Return a compiled expression bound to this session's functions.

        Compiled expressions are cached per session so that repeated queries
        are consistent, and the cache is dropped when the session ends so
        that no reference to the session functions outlives it.
        """
        key = (query, tuple(sorted(namespaces.items())), smart_strings)
        try:
            return self._cache[key]
        except KeyError:
            evaluator = etree.XPath(
                query,
                namespaces=namespaces,
                extensions=dict(self._functions),
                smart_strings=smart_strings,
            )
            self._cache[key] = evaluator
            return evaluator

    def __enter__(self) -> Self:
        self._token = _xpath_function_session.set(self)
        return self

    def __exit__(self, *exc_info: object) -> None:
        token, self._token = self._token, None
        if token is not None:
            _xpath_function_session.reset(token)
        self._cache.clear()
        self._functions.clear()


def setup() -> None:
    set_xpathfunc("has-class", has_class)


def has_class(context: Any, *classes: str) -> bool:
    """has-class function.

    Return True if all ``classes`` are present in element's class attr.

    """
    if not context.eval_context.get("args_checked"):
        if not classes:
            raise ValueError("XPath error: has-class must have at least 1 argument")
        for c in classes:
            if not isinstance(c, str):
                raise ValueError("XPath error: has-class arguments must be strings")  # noqa: TRY004
        context.eval_context["args_checked"] = True

    node_cls = context.context_node.get("class")
    if node_cls is None:
        return False
    node_cls = " " + node_cls + " "
    node_cls = replace_html5_whitespaces(" ", node_cls)
    return all(" " + cls + " " in node_cls for cls in classes)
