from __future__ import annotations

import gc

import pytest
from lxml import html

from parsel import (
    AbsoluteXPathWarning,
    LocatedNode,
    NodeKind,
    Provenance,
    Selector,
    SelectorList,
    StaleProvenanceError,
)
from parsel.provenance import _registry
from parsel.selector import (
    CannotRemoveElementWithoutParent,
    CannotRemoveElementWithoutRoot,
)


HTML_DOC = """
<html><body>
  <p class="dup">same</p>
  <p class="dup">same</p>
  <p>before <b>bold</b> middle <i>it</i> after</p>
</body></html>
"""


class TestProvenanceModes:
    def test_disabled_by_default_keeps_nothing(self) -> None:
        before = len(_registry)
        sel = Selector(text=HTML_DOC)
        sel.xpath("//p").css("::text")
        assert sel._provenance is None
        assert all(child._provenance is None for child in sel.xpath("//p"))
        with pytest.raises(ValueError):
            sel.provenance
        with pytest.raises(ValueError):
            sel.locate()
        assert len(_registry) == before

    def test_enabled_assigns_document_identity(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        provenance = sel.provenance
        assert isinstance(provenance, Provenance)
        assert provenance.kind == NodeKind.ELEMENT
        assert provenance.path == "/html"
        assert provenance.document.name.startswith("document-")
        assert provenance.generation == 0
        assert provenance.is_stale is False

    def test_custom_document_name(self) -> None:
        sel = Selector(text=HTML_DOC, provenance="page-1")
        assert sel.provenance.document.name == "page-1"
        # chains reuse the same document identity
        child = sel.xpath("//p")[0]
        assert child.provenance.document is sel.provenance.document

    @pytest.mark.parametrize("type_", ["json", "text"])
    def test_unsupported_types_are_rejected(self, type_: str) -> None:
        text = '{"a": 1}' if type_ == "json" else "plain text"
        with pytest.raises(ValueError, match="Provenance"):
            Selector(text=text, type=type_, provenance=True)


class TestNodeKinds:
    def test_elements_are_tracked(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        paragraphs = sel.xpath("//p[@class='dup']")
        paths = [p.provenance.path for p in paragraphs]
        assert paths == ["/html/body/p[1]", "/html/body/p[2]"]
        for paragraph in paragraphs:
            located = paragraph.locate()
            assert isinstance(located, LocatedNode)
            assert located.kind is NodeKind.ELEMENT or located.kind == "element"
            assert located.element is paragraph.root
            assert located.attribute is None

    def test_duplicate_text_keeps_distinct_locations(self) -> None:
        # The two paragraphs serialize to the same string; provenance must not
        # be derivable from a reverse string search.
        sel = Selector(text=HTML_DOC, provenance=True)
        texts = sel.css("p.dup::text")
        provenance = [text.provenance for text in texts]
        assert all(p.kind == NodeKind.TEXT for p in provenance)
        assert all(p.value == "same" for p in provenance)
        assert [p.path for p in provenance] == [
            "/html/body/p[1]/text()[1]",
            "/html/body/p[2]/text()[1]",
        ]
        anchors = [p.locate().element for p in provenance]
        assert anchors[0] is not anchors[1]

    def test_text_and_adjacent_tails_are_distinguished(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        nodes = sel.xpath(
            "//p[3]/text() | //p[3]/b/following-sibling::text() "
            "| //p[3]/i/following-sibling::text()"
        )
        kinds = [node.provenance.kind for node in nodes]
        paths = [node.provenance.path for node in nodes]
        assert kinds == [
            NodeKind.TEXT,
            NodeKind.TAIL,
            NodeKind.TAIL,
        ]
        assert paths == [
            "/html/body/p[3]/text()[1]",
            "/html/body/p[3]/tail()[1]",
            "/html/body/p[3]/tail()[2]",
        ]
        located = [node.locate() for node in nodes]
        assert [node.value for node in located] == [
            "before ",
            " middle ",
            " after",
        ]

    def test_same_named_attributes_are_distinguished(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        attributes = sel.xpath("//p/@class")
        provenance = [attribute.provenance for attribute in attributes]
        assert all(p.kind == NodeKind.ATTRIBUTE for p in provenance)
        assert all(p.attribute == "class" for p in provenance)
        assert [p.path for p in provenance] == [
            "/html/body/p[1]/@class",
            "/html/body/p[2]/@class",
        ]
        located = [p.locate() for p in provenance]
        assert all(node.attribute == "class" for node in located)
        assert all(node.value == "dup" for node in located)
        assert located[0].element is not located[1].element

    @pytest.mark.parametrize(
        ("query", "expected_kind"),
        [
            ("count(//p)", NodeKind.DERIVED),
            ("concat('a', 'b')", NodeKind.DERIVED),
            ("string(//p[1])", NodeKind.DERIVED),
            ("true()", NodeKind.DERIVED),
            ("1 + 1", NodeKind.DERIVED),
        ],
    )
    def test_computed_scalars_are_derived(
        self, query: str, expected_kind: str
    ) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        result = sel.xpath(query)
        provenance = result[0].provenance
        assert provenance.kind == expected_kind
        assert provenance.path == "<derived>"
        # A derived result invents no node and never goes stale.
        assert provenance.is_stale is False
        located = provenance.locate()
        assert located.kind == NodeKind.DERIVED
        assert located.element is None


class TestChainingAndDocuments:
    def test_selector_list_chain_preserves_identity(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        chained = sel.css("body").xpath(".//p").xpath("text()")
        assert isinstance(chained, SelectorList)
        documents = {p.document for p in chained.provenance}
        assert documents == {sel.provenance.document}
        assert [node.kind for node in chained.locate()] == [
            NodeKind.TEXT,
            NodeKind.TEXT,
            NodeKind.TEXT,
            NodeKind.TAIL,
            NodeKind.TAIL,
        ]

    def test_two_documents_never_share_identity(self) -> None:
        first = Selector(text=HTML_DOC, provenance="doc-a")
        second = Selector(text=HTML_DOC, provenance="doc-b")
        first_prov = first.xpath("//p[1]")[0].provenance
        second_prov = second.xpath("//p[1]")[0].provenance
        assert first_prov.path == second_prov.path
        assert first_prov.document is not second_prov.document
        assert first_prov.document.name == "doc-a"
        assert second_prov.document.name == "doc-b"

    def test_combined_list_keeps_each_source(self) -> None:
        first = Selector(text=HTML_DOC, provenance="doc-a")
        second = Selector(text=HTML_DOC, provenance="doc-b")
        combined = SelectorList(first.xpath("//p") + second.xpath("//p"))
        names = [p.document.name for p in combined.provenance]
        assert names == ["doc-a"] * 3 + ["doc-b"] * 3
        located = combined.locate()
        assert all(node.kind == NodeKind.ELEMENT for node in located)

    def test_identical_trees_get_distinct_auto_names(self) -> None:
        first = Selector(text=HTML_DOC, provenance=True)
        second = Selector(text=HTML_DOC, provenance=True)
        assert first.provenance.document is not second.provenance.document
        assert first.provenance.document.name != second.provenance.document.name


class TestStaleness:
    def test_dropped_element_does_not_point_to_replacement(self) -> None:
        sel = Selector(
            text="<html><body><ul><li>a</li><li>b</li><li>c</li></ul></body></html>",
            provenance=True,
        )
        original = sel.xpath("//li[1]")[0].provenance
        sel.xpath("//li[1]")[0].drop()
        # A new node now occupies the exact same structural path.
        replacement = sel.xpath("//li[1]")[0]
        assert replacement.provenance.path == original.path
        assert replacement.get() == "<li>b</li>"
        assert original.is_stale is True
        with pytest.raises(StaleProvenanceError):
            original.locate()

    def test_dropped_element_invalidates_its_text_provenance(self) -> None:
        sel = Selector(
            text="<html><body><ul><li>a</li><li>b</li></ul></body></html>",
            provenance=True,
        )
        text_provenance = sel.xpath("//li[1]/text()")[0].provenance
        sel.xpath("//li[1]")[0].drop()
        assert text_provenance.is_stale is True
        with pytest.raises(StaleProvenanceError):
            text_provenance.locate()

    def test_attribute_change_is_detected(self) -> None:
        sel = Selector(
            text="<html><body><p id='1'>x</p></body></html>", provenance=True
        )
        attribute = sel.xpath("//p/@id")[0].provenance
        sel.root.find(".//p").set("id", "9")
        assert attribute.is_stale is True
        with pytest.raises(StaleProvenanceError):
            attribute.locate()

    def test_tail_change_is_detected(self) -> None:
        sel = Selector(
            text="<html><body><p>a <b>x</b>tail</p></body></html>",
            provenance=True,
        )
        tail = sel.xpath("//b/following::text()[1]")[0].provenance
        assert tail.kind == NodeKind.TAIL
        sel.root.find(".//b").tail = "changed"
        assert tail.is_stale is True
        with pytest.raises(StaleProvenanceError):
            tail.locate()

    def test_pseudo_element_drop_keeps_original_error(self) -> None:
        sel = Selector(
            text="<html><body><ul><li>1</li></ul></body></html>",
            provenance=True,
        )
        with pytest.raises(CannotRemoveElementWithoutRoot):
            sel.css("li::text")[0].drop()
        with pytest.raises(CannotRemoveElementWithoutParent):
            sel.drop()


class TestNamespaceRemoval:
    XML = (
        '<?xml version="1.0"?>'
        '<feed xmlns="http://x"><link id="1">v</link></feed>'
    )

    def test_old_paths_are_stale_after_remove_namespaces(self) -> None:
        sel = Selector(text=self.XML, type="xml", provenance=True)
        element = sel.xpath(
            "//x:link", namespaces={"x": "http://x"}
        )[0].provenance
        text = sel.xpath(
            "//x:link/text()", namespaces={"x": "http://x"}
        )[0].provenance
        attribute = sel.xpath(
            "//x:link/@id", namespaces={"x": "http://x"}
        )[0].provenance
        sel.remove_namespaces()
        assert element.is_stale is True
        assert text.is_stale is True
        assert attribute.is_stale is True
        for old in (element, text, attribute):
            with pytest.raises(StaleProvenanceError):
                old.locate()

    def test_new_paths_resolve_after_remove_namespaces(self) -> None:
        sel = Selector(text=self.XML, type="xml", provenance=True)
        sel.remove_namespaces()
        assert sel.provenance.is_stale is False
        link = sel.xpath("//link")[0].provenance
        text = sel.xpath("//link/text()")[0].provenance
        attribute = sel.xpath("//link/@id")[0].provenance
        assert link.path == "/feed/link"
        assert link.locate().element is sel.root.find("link")
        assert text.locate().value == "v"
        assert attribute.locate().value == "1"


class TestLifecycle:
    def test_html_document_is_released_with_selector(self) -> None:
        sel = Selector(text=HTML_DOC, provenance=True)
        provenance = sel.xpath("//p[1]")[0].provenance
        assert len(_registry) >= 1
        del sel
        gc.collect()
        assert provenance.is_stale is True
        with pytest.raises(StaleProvenanceError):
            provenance.locate()

    def test_fragment_wrapper_is_kept_then_released(self) -> None:
        fragment = html.fragment_fromstring("<div><p>x</p></div>")
        sel = Selector(root=fragment, type="html", provenance=True)
        with pytest.warns(AbsoluteXPathWarning):
            provenance = sel.xpath("//p")[0].provenance
        assert provenance.path == "/html/body/div/p"
        assert provenance.locate().element is fragment.find("p")
        del sel, fragment
        gc.collect()
        assert provenance.is_stale is True

    def test_xml_document_is_released_with_selector(self) -> None:
        sel = Selector(
            text='<?xml version="1.0"?><root><a/></root>',
            type="xml",
            provenance=True,
        )
        assert len(_registry) >= 1
        del sel
        gc.collect()
        # The finalizer must not leak the XML tree.
        assert _registry == {}

    def test_no_registry_entries_when_disabled(self) -> None:
        Selector(text=HTML_DOC).xpath("//p")
        Selector(
            text='<?xml version="1.0"?><root/>', type="xml"
        ).xpath("//root")
        assert _registry == {}

    def test_shared_document_survives_until_last_owner(self) -> None:
        first = Selector(text=HTML_DOC, provenance=True)
        second = Selector(root=first.root, provenance=True)
        document = first.provenance.document
        assert second.provenance.document is document
        assert document._references == 2
        child = first.xpath("//p[1]")[0].provenance
        del first
        gc.collect()
        # The second selector still owns the document, so child provenance
        # must remain resolvable.
        assert child.is_stale is False
        assert child.locate().element is not None
        del second
        gc.collect()
        assert _registry == {}
        assert child.is_stale is True
