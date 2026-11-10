from __future__ import annotations

import json

import pytest

from parsel import Selector
from parsel.records import (
    MISSING,
    NULL,
    Field,
    RecordExtractionError,
    RecordPlan,
    Source,
)

# Three articles: the middle one has no link, the last one has two links
# and a broken JSON payload. Zipping full-page title/link lists would
# misalign everything after the first article.
HTML = """
<html><body>
  <article>
    <h2>First post</h2>
    <a href="/posts/1">Read</a>
    <script type="application/json" class="data">
      {"author": {"name": "Alice"}, "tags": ["a", "b"], "views": 10}
    </script>
  </article>
  <article>
    <h2>Second post</h2>
    <script type="application/json" class="data">
      {"author": {"name": "Bob"}, "tags": [], "views": null}
    </script>
  </article>
  <article>
    <h2>Third post</h2>
    <a href="/posts/3a">Read A</a>
    <a href="/posts/3b">Read B</a>
    <script type="application/json" class="data">{broken json</script>
  </article>
</body></html>
"""


def make_plan() -> RecordPlan:
    return RecordPlan(
        record=Source.css("article"),
        fields=[
            Field.css("title", "h2::text"),
            Field.css("link", "a::attr(href)", cardinality="optional"),
            Field.jmespath(
                "author",
                "author.name",
                source=Source.css("script.data::text"),
                cardinality="optional",
            ),
            Field.jmespath(
                "views",
                "views",
                source=Source.css("script.data::text"),
                cardinality="optional",
            ),
        ],
    )


def test_lenient_mode_keeps_records_aligned() -> None:
    result = make_plan().compile().extract(Selector(HTML), strict=False)
    assert len(result.records) == 3
    assert not result.ok

    first, second, third = result.records

    assert first.ok
    assert first.data == {
        "title": "First post",
        "link": "/posts/1",
        "author": "Alice",
        "views": 10,
    }

    # The missing link of the middle record is MISSING, not the link of
    # a neighboring record, and it is not an error for an optional field.
    assert second.ok
    assert second.data["link"] is MISSING
    assert second.data["link"] != ""
    assert second.data["author"] == "Bob"
    # JSON null is represented by its own sentinel, distinct from MISSING.
    assert second.data["views"] is NULL
    assert second.data["views"] is not MISSING

    # The broken JSON of the last record does not leak into earlier
    # records, and its duplicate links are a conflict, not "first wins".
    assert not third.ok
    assert third.data == {"title": "Third post"}
    assert {(e.field, e.stage, e.kind) for e in third.errors} == {
        ("link", "select", "conflict"),
        ("author", "json", "json-parse"),
        ("views", "json", "json-parse"),
    }
    assert all(e.record_index == 2 for e in third.errors)
    assert "link" not in third.data


def test_strict_mode_raises_with_all_errors() -> None:
    with pytest.raises(RecordExtractionError) as excinfo:
        make_plan().compile().extract(Selector(HTML))
    errors = excinfo.value.errors
    assert len(errors) == 3
    assert all(e.record_index == 2 for e in errors)
    assert {e.field for e in errors} == {"link", "author", "views"}
    assert "record #2" in str(excinfo.value)


def test_extraction_does_not_modify_the_tree() -> None:
    selector = Selector(HTML)
    before = selector.get()
    plan = make_plan().compile()
    with pytest.raises(RecordExtractionError):
        plan.extract(selector)
    plan.extract(selector, strict=False)
    assert selector.get() == before


def test_json_is_parsed_once_per_record(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    real_loads = json.loads

    def counting_loads(text: str, *args: object, **kwargs: object) -> object:
        calls.append(text)
        return real_loads(text)

    selector = Selector(HTML)  # built before patching: type detection parses
    monkeypatch.setattr(json, "loads", counting_loads)
    make_plan().compile().extract(selector, strict=False)
    # Two JMESPath fields share the same source per record: 3 records
    # (including the broken one) mean exactly 3 parse attempts, not 6.
    assert len(calls) == 3


def test_one_cardinality_missing_is_an_error() -> None:
    plan = RecordPlan(
        record=Source.css("article"),
        fields=[Field.css("price", "span.price::text")],
    )
    result = plan.compile().extract(Selector(HTML), strict=False)
    assert [r.data for r in result.records] == [{}, {}, {}]
    assert all(
        (e.field, e.stage, e.kind) == ("price", "select", "missing")
        for r in result.records
        for e in r.errors
    )


def test_empty_string_is_distinct_from_missing() -> None:
    html = '<article><a href="">x</a></article><article></article>'
    plan = RecordPlan(
        record=Source.css("article"),
        fields=[Field.css("link", "a::attr(href)", cardinality="optional")],
    )
    result = plan.compile().extract(Selector(html), strict=False)
    assert result.ok
    assert result.records[0].data["link"] == ""
    assert result.records[0].data["link"] is not MISSING
    assert result.records[1].data["link"] is MISSING


def test_many_cardinality() -> None:
    plan = RecordPlan(
        record=Source.xpath("//article"),
        fields=[
            Field.css("links", "a::attr(href)", cardinality="many"),
            Field.jmespath(
                "tags",
                "tags",
                source=Source.css("script.data::text"),
                cardinality="many",
            ),
        ],
    )
    result = plan.compile().extract(Selector(HTML), strict=False)
    assert [r.data["links"] for r in result.records] == [
        ["/posts/1"],
        [],
        ["/posts/3a", "/posts/3b"],
    ]
    assert result.records[0].data["tags"] == ["a", "b"]
    assert result.records[1].data["tags"] == []
    # Broken JSON in the last record is still reported for a many field.
    assert result.records[2].errors[0].kind == "json-parse"


def test_jmespath_null_list_items_become_null_sentinel() -> None:
    html = """
    <article>
      <script type="application/json" class="data">{"tags": ["x", null]}</script>
    </article>
    """
    plan = RecordPlan(
        record=Source.css("article"),
        fields=[
            Field.jmespath(
                "tags",
                "tags",
                source=Source.css("script.data::text"),
                cardinality="many",
            )
        ],
    )
    result = plan.compile().extract(Selector(html), strict=False)
    assert result.ok
    assert result.records[0].data["tags"] == ["x", NULL]


def test_jmespath_many_rejects_scalar() -> None:
    html = """
    <article>
      <script type="application/json" class="data">{"tags": "nope"}</script>
    </article>
    """
    plan = RecordPlan(
        record=Source.css("article"),
        fields=[
            Field.jmespath(
                "tags",
                "tags",
                source=Source.css("script.data::text"),
                cardinality="many",
            )
        ],
    )
    result = plan.compile().extract(Selector(html), strict=False)
    (error,) = result.errors
    assert (error.stage, error.kind) == ("jmespath", "not-a-list")


def test_jmespath_list_result_conflicts_with_one() -> None:
    plan = RecordPlan(
        record=Source.css("article"),
        fields=[
            Field.jmespath(
                "tags", "tags", source=Source.css("script.data::text")
            )
        ],
    )
    result = plan.compile().extract(Selector(HTML), strict=False)
    (error,) = result.records[0].errors
    assert (error.stage, error.kind) == ("jmespath", "conflict")


def test_xpath_field_and_record_source() -> None:
    plan = RecordPlan(
        record=Source.xpath("//article"),
        fields=[Field.xpath("title", ".//h2/text()")],
    )
    result = plan.compile().extract(Selector(HTML), strict=False)
    assert result.ok
    assert [r.data["title"] for r in result.records] == [
        "First post",
        "Second post",
        "Third post",
    ]


def test_compile_validates_queries() -> None:
    with pytest.raises(ValueError, match="Invalid CSS"):
        RecordPlan(
            record=Source.css("article"),
            fields=[Field.css("title", "h2:__bad__")],
        ).compile()
    with pytest.raises(ValueError, match="Invalid XPath"):
        RecordPlan(
            record=Source.xpath("//article["),
            fields=[Field.css("title", "h2::text")],
        ).compile()
    with pytest.raises(ValueError, match="Invalid JMESPath"):
        RecordPlan(
            record=Source.css("article"),
            fields=[
                Field.jmespath(
                    "x", "foo[", source=Source.css("script::text")
                )
            ],
        ).compile()


def test_field_declaration_validation() -> None:
    with pytest.raises(ValueError, match="cardinality"):
        Field.css("x", "a", cardinality="few")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="source selector"):
        Field("x", jmespath_query="a.b")
    with pytest.raises(ValueError, match="exactly one"):
        Field("x")
