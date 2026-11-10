"""Optional node-level provenance for CSS and XPath selections.

The objects in this module are only created when provenance mode is enabled
with the ``provenance`` argument of :class:`~parsel.Selector`. In that mode
every selected value keeps a pointer to the exact node of the *parsed* lxml
tree it came from instead of relying on a reverse string search in the source
markup.

Provenance describes the parsed tree, not the raw document. Paths are logical
structural locations, never byte offsets, and they are not guaranteed to stay
valid after the tree is mutated.

Identity model
--------------
The registry entry always holds the document root directly: this also keeps
the implicit ``<html>/<body>`` wrappers of an HTML fragment alive while a
selector only references a sub-element. Entries are released through
:func:`release_document`, wired to the owning :class:`~parsel.Selector`
finalizer, so a collected selector never leaks its tree. With provenance
disabled nothing is registered at all, so the default path retains no extra
tree reference.

A :class:`Provenance` never relies on a single path string: every resolution
also checks the document's mutation generation and, for non-element nodes,
the captured value. A node dropped and replaced by another node at the same
path therefore never resolves silently to the replacement.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from lxml import etree

_DOCUMENT_COUNTER = itertools.count(1)

# One entry per parsed tree. Entries own the document root (which also keeps
# implicit HTML fragment wrappers alive) and are removed by the owning
# Selector finalizer once every reference to that document is released.
_registry: dict[etree._Element, DocumentProvenance] = {}


class NodeKind:
    """The kind of tree node a :class:`Provenance` points to.

    Elements, attributes and the two kinds of lxml text nodes (an element's
    ``text`` and the ``tail`` of one of its children) are tracked separately.
    ``DERIVED`` is reserved for scalar XPath results that are computed rather
    than read from a single node.
    """

    ELEMENT = "element"
    ATTRIBUTE = "attribute"
    TEXT = "text"
    TAIL = "tail"
    DERIVED = "derived"


class StaleProvenanceError(RuntimeError):
    """Raised when provenance cannot locate its node anymore.

    This happens after the node was dropped, the document tree was released
    or garbage collected, or the tree was mutated in a way that invalidated
    previously captured paths, such as
    :meth:`~parsel.Selector.remove_namespaces`.
    """


@dataclass(frozen=True, slots=True)
class LocatedNode:
    """The node resolved from a still-valid :class:`Provenance`."""

    kind: str
    value: Any
    element: etree._Element | None = None
    attribute: str | None = None


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a single selected value comes from.

    For text, tail and attribute nodes the exact node is identified by
    ``kind``, ``attribute`` and ``child_index`` relative to the element at
    ``path``. The captured ``value`` is compared on resolution so a different
    node that reuses the same path is never mistaken for the original.

    For :data:`NodeKind.DERIVED` results no node is tracked: the value is a
    computed scalar with no unique source node, so none is invented.
    """

    document: "DocumentProvenance"
    kind: str
    path: str
    value: Any
    attribute: str | None
    child_index: int | None
    generation: int

    @property
    def is_stale(self) -> bool:
        """Return whether the captured location can no longer be resolved."""
        if self.generation != self.document.generation:
            return True
        if self.kind == NodeKind.DERIVED:
            return False
        try:
            self._resolve()
        except StaleProvenanceError:
            return True
        return False

    def locate(self) -> LocatedNode:
        """Resolve the provenance back to a node of the parsed tree.

        Raises :class:`StaleProvenanceError` instead of silently returning a
        different node that happens to share the captured path.
        """
        if self.generation != self.document.generation:
            raise StaleProvenanceError(
                f"Provenance at {self.path!r} was invalidated by a change to "
                f"document {self.document.name!r}."
            )
        if self.kind == NodeKind.DERIVED:
            return LocatedNode(NodeKind.DERIVED, self.value)
        return self._resolve()

    def _resolve(self) -> LocatedNode:
        element = self.document.element_at(self.path)
        if self.kind == NodeKind.ELEMENT:
            return LocatedNode(NodeKind.ELEMENT, element, element=element)
        if self.kind == NodeKind.ATTRIBUTE:
            assert self.attribute is not None  # nosec
            if self.attribute not in element.attrib:
                raise StaleProvenanceError(
                    f"Attribute {self.attribute!r} at {self.path!r} was "
                    f"removed from document {self.document.name!r}."
                )
            value = element.attrib[self.attribute]
            if value != self.value:
                raise StaleProvenanceError(
                    f"Attribute {self.attribute!r} at {self.path!r} changed "
                    "since the provenance was captured."
                )
            return LocatedNode(
                NodeKind.ATTRIBUTE,
                value,
                element=element,
                attribute=self.attribute,
            )
        if self.kind == NodeKind.TEXT:
            if element.text is None or element.text != self.value:
                raise StaleProvenanceError(
                    f"Text node at {self.path!r} changed or was removed in "
                    f"document {self.document.name!r}."
                )
            return LocatedNode(NodeKind.TEXT, element.text, element=element)
        assert self.child_index is not None  # nosec
        children = list(element)
        if self.child_index >= len(children):
            raise StaleProvenanceError(
                f"Tail node at {self.path!r} lost its preceding child in "
                f"document {self.document.name!r}."
            )
        child = children[self.child_index]
        if child.tail is None or child.tail != self.value:
            raise StaleProvenanceError(
                f"Tail node at {self.path!r} changed or was removed in "
                f"document {self.document.name!r}."
            )
        return LocatedNode(NodeKind.TAIL, child.tail, element=element)


class DocumentProvenance:
    """Identity and mutation bookkeeping for one parsed tree."""

    __slots__ = ("_root", "_references", "generation", "name")

    def __init__(self, root: etree._Element, name: str) -> None:
        self.name = name
        self.generation = 0
        self._root: etree._Element | None = root
        self._references = 0

    @property
    def root(self) -> etree._Element | None:
        return self._root

    def bump(self) -> None:
        """Mark every provenance captured before this call as stale."""
        self.generation += 1

    def retain(self) -> None:
        """Record one owning selector keeping this document alive."""
        self._references += 1

    def release(self) -> None:
        """Release one owner; drop the tree entry after the last one."""
        self._references -= 1
        if self._references <= 0:
            release_document(self)

    def element_at(self, path: str) -> etree._Element:
        element_path = _element_path_of(path)
        root = self.root
        if root is None:
            raise StaleProvenanceError(
                f"Document {self.name!r} has been released."
            )
        # ElementTree.find() warns on and mishandles paths that select the
        # root itself; an XPath evaluation resolves every getpath() result,
        # including the root, consistently.
        element = root.xpath(element_path)
        if isinstance(element, list):
            element = element[0] if element else None
        if element is None:
            raise StaleProvenanceError(
                f"No element at {path!r} in document {self.name!r}."
            )
        return element

    def element_path(self, element: etree._Element) -> str:
        root = self.root
        if root is None:
            raise StaleProvenanceError(
                f"Document {self.name!r} has been released."
            )
        return etree.ElementTree(root).getpath(element)

    def provenance_for(self, result: Any, value: Any) -> Provenance:
        """Build provenance for one raw lxml XPath result value."""
        if isinstance(result, etree._Element):
            return self._make(
                NodeKind.ELEMENT, self.element_path(result), value
            )
        if isinstance(result, etree._ElementUnicodeResult):
            parent = result.getparent()
            if parent is None:
                # Smart strings without a parent are computed scalar results
                # such as string() or concat(), not real text nodes.
                return self.derived(value)
            child_index = None
            if result.is_tail:
                # For a tail node lxml reports the child whose tail it is as
                # the parent; the container is one level up.
                child = parent
                parent = child.getparent()
                if parent is None:
                    raise StaleProvenanceError(
                        "Tail node has no containing element."
                    )
                child_index = list(parent).index(child)
            base = self.element_path(parent)
            if result.attrname is not None:
                attribute = result.attrname
                return self._make(
                    NodeKind.ATTRIBUTE,
                    f"{base}/@{attribute}",
                    value,
                    attribute=attribute,
                )
            if result.is_tail:
                assert child_index is not None  # nosec
                return self._make(
                    NodeKind.TAIL,
                    f"{base}/tail()[{child_index + 1}]",
                    value,
                    child_index=child_index,
                )
            return self._make(
                NodeKind.TEXT, f"{base}/text()[1]", value
            )
        return self.derived(value)

    def derived(self, value: Any) -> Provenance:
        return self._make(NodeKind.DERIVED, "<derived>", value)

    def _make(
        self,
        kind: str,
        path: str,
        value: Any,
        *,
        attribute: str | None = None,
        child_index: int | None = None,
    ) -> Provenance:
        return Provenance(
            document=self,
            kind=kind,
            path=path,
            value=value,
            attribute=attribute,
            child_index=child_index,
            generation=self.generation,
        )


def get_document_provenance(
    root: etree._Element, name: str | bool | None
) -> DocumentProvenance:
    """Return the shared document identity for *root*.

    Selectors derived from the same tree share one identity so that chains
    and flattened :class:`~parsel.SelectorList` results never mix sources.
    Selectors built from different trees get different identities even when
    their markup is identical.
    """
    tree_root = _tree_root(root)
    existing = _registry.get(tree_root)
    if existing is not None:
        existing.retain()
        return existing
    document_name = name if isinstance(name, str) and name else _default_name()
    document = DocumentProvenance(tree_root, document_name)
    _registry[tree_root] = document
    document.retain()
    return document


def release_document(document: DocumentProvenance) -> None:
    """Drop the registry entry so the document tree can be collected."""
    root = document.root
    if root is not None:
        _registry.pop(root, None)
        document._root = None


def refresh(provenance: Provenance, value: Any) -> Provenance:
    """Recompute a captured location against the current tree.

    Used after an intentional in-place mutation such as namespace removal.
    Raises :class:`StaleProvenanceError` if the anchor element is gone.

    The returned provenance is stamped with the document's current
    generation, so it remains valid while locations captured before the
    mutation keep reporting the stale generation.
    """
    document = provenance.document
    if provenance.kind == NodeKind.DERIVED:
        refreshed = document.derived(value)
        return _stamp(refreshed, document.generation)
    element = document.element_at(provenance.path)
    if provenance.kind == NodeKind.ELEMENT:
        refreshed = document._make(
            NodeKind.ELEMENT, document.element_path(element), value
        )
        return _stamp(refreshed, document.generation)
    base = document.element_path(element)
    if provenance.kind == NodeKind.ATTRIBUTE:
        assert provenance.attribute is not None  # nosec
        if provenance.attribute not in element.attrib:
            raise StaleProvenanceError(
                f"Attribute {provenance.attribute!r} was removed and cannot "
                "be refreshed."
            )
        refreshed = document._make(
            NodeKind.ATTRIBUTE,
            f"{base}/@{provenance.attribute}",
            value,
            attribute=provenance.attribute,
        )
        return _stamp(refreshed, document.generation)
    if provenance.kind == NodeKind.TEXT:
        refreshed = document._make(
            NodeKind.TEXT, f"{base}/text()[1]", value
        )
        return _stamp(refreshed, document.generation)
    assert provenance.child_index is not None  # nosec
    child_index = provenance.child_index
    refreshed = document._make(
        NodeKind.TAIL,
        f"{base}/tail()[{child_index + 1}]",
        value,
        child_index=child_index,
    )
    return _stamp(refreshed, document.generation)


def _stamp(provenance: Provenance, generation: int) -> Provenance:
    return Provenance(
        document=provenance.document,
        kind=provenance.kind,
        path=provenance.path,
        value=provenance.value,
        attribute=provenance.attribute,
        child_index=provenance.child_index,
        generation=generation,
    )


def _default_name() -> str:
    return f"document-{next(_DOCUMENT_COUNTER)}"


def _tree_root(element: etree._Element) -> etree._Element:
    node = element
    while node.getparent() is not None:
        node = node.getparent()
    return node


def _element_path_of(path: str) -> str:
    """Return the element-anchoring part of a possibly text/tail/attr path."""
    for suffix in ("/text()", "/tail()"):
        index = path.find(suffix)
        if index != -1:
            return path[:index]
    attribute_index = path.rfind("/@")
    if attribute_index != -1:
        return path[:attribute_index]
    return path
