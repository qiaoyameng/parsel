"""Declarative record extraction plans.

When a page contains several records (e.g. articles on a listing page),
extracting a full-page list of titles and a full-page list of links and
then zipping them together misaligns every record after the first one
that is missing a value.  This module provides a compilable plan that
first locates one root node per record and then runs every field query
scoped to that root, so a missing or duplicated value in one record
never shifts the values of the following records.

Field values can come from relative CSS or XPath queries, or from a
JMESPath query applied to JSON parsed from text selected within the
same record.  The same JSON text is parsed only once per record, no
matter how many fields read from it.

Extraction never modifies the tree of the input selector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import jmespath
from lxml import etree

from parsel.csstranslator import css2xpath
from parsel.selector import Selector

__all__ = [
    "MISSING",
    "NULL",
    "CompiledPlan",
    "ExtractionResult",
    "Field",
    "FieldError",
    "RecordExtractionError",
    "RecordPlan",
    "RecordResult",
    "Source",
]

Cardinality = Literal["one", "optional", "many"]
_CARDINALITIES = ("one", "optional", "many")


class _Sentinel:
    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name

    def __bool__(self) -> bool:
        return False


#: Value of an ``optional`` field that had no match in its record.
#: Distinct from ``""`` (a matched but empty value) and from ``NULL``.
MISSING = _Sentinel("MISSING")

#: Value of a field whose JSON source evaluated to a JSON ``null``.
#: Distinct from ``MISSING`` (no match) and from ``""``.
NULL = _Sentinel("NULL")


@dataclass(frozen=True)
class Source:
    """A CSS or XPath query, validated and pre-translated at compile time."""

    kind: Literal["css", "xpath"]
    query: str

    @classmethod
    def css(cls, query: str) -> Source:
        return cls("css", query)

    @classmethod
    def xpath(cls, query: str) -> Source:
        return cls("xpath", query)

    def _compile(self) -> str:
        """Return the XPath expression for this source, validating the query."""
        if self.kind == "css":
            try:
                return css2xpath(self.query)
            except Exception as exc:
                raise ValueError(f"Invalid CSS query {self.query!r}: {exc}") from exc
        try:
            etree.XPath(self.query)
        except etree.XPathError as exc:
            raise ValueError(f"Invalid XPath query {self.query!r}: {exc}") from exc
        return self.query


@dataclass(frozen=True)
class Field:
    """A single named field of a record.

    Use :meth:`Field.css`, :meth:`Field.xpath` or :meth:`Field.jmespath`
    to create instances.

    ``cardinality`` is one of:

    - ``"one"``: exactly one value is required; zero matches is a
      ``missing`` error, more than one is a ``conflict`` error.
    - ``"optional"``: zero matches produce the ``MISSING`` sentinel,
      more than one is a ``conflict`` error.
    - ``"many"``: any number of values is accepted, returned as a list.

    Conflicts are always reported as errors; the first match is never
    silently used.
    """

    name: str
    cardinality: Cardinality = "one"
    selector: Source | None = None
    source: Source | None = None
    jmespath_query: str | None = None

    def __post_init__(self) -> None:
        if self.cardinality not in _CARDINALITIES:
            raise ValueError(
                f"Invalid cardinality {self.cardinality!r} for field {self.name!r}, "
                f"expected one of {_CARDINALITIES}"
            )
        if (self.selector is None) == (self.jmespath_query is None):
            raise ValueError(
                f"Field {self.name!r} needs exactly one of a CSS/XPath selector "
                f"or a JMESPath query"
            )
        if self.jmespath_query is not None and self.source is None:
            raise ValueError(
                f"JMESPath field {self.name!r} needs a source selector "
                f"for the JSON text"
            )
        if self.selector is not None and self.source is not None:
            raise ValueError(
                f"Field {self.name!r} cannot have both a selector and a source"
            )

    @classmethod
    def css(cls, name: str, query: str, *, cardinality: Cardinality = "one") -> Field:
        """A field extracted with a (relative) CSS query."""
        return cls(name, cardinality=cardinality, selector=Source.css(query))

    @classmethod
    def xpath(cls, name: str, query: str, *, cardinality: Cardinality = "one") -> Field:
        """A field extracted with a (relative) XPath query."""
        return cls(name, cardinality=cardinality, selector=Source.xpath(query))

    @classmethod
    def jmespath(
        cls,
        name: str,
        query: str,
        *,
        source: Source,
        cardinality: Cardinality = "one",
    ) -> Field:
        """A field extracted with a JMESPath query.

        ``source`` selects the node within the record whose text is a JSON
        document.  The JSON is parsed once per record and shared by all
        JMESPath fields reading the same text.
        """
        return cls(
            name, cardinality=cardinality, source=source, jmespath_query=query
        )


@dataclass(frozen=True)
class FieldError:
    """A single field failure, located by record, field and query stage.

    ``stage`` is ``"select"`` (CSS/XPath evaluation), ``"json"`` (JSON
    parsing of the selected text) or ``"jmespath"`` (JMESPath evaluation
    and result cardinality).

    ``kind`` is ``"missing"``, ``"conflict"``, ``"json-parse"``,
    ``"jmespath-eval"`` or ``"not-a-list"``.
    """

    record_index: int
    field: str
    stage: str
    kind: str
    message: str

    def __str__(self) -> str:
        return (
            f"record #{self.record_index}, field {self.field!r} "
            f"[{self.stage}/{self.kind}]: {self.message}"
        )


class RecordExtractionError(Exception):
    """Raised by :meth:`CompiledPlan.extract` in strict mode.

    Carries every :class:`FieldError` found across the whole batch in
    ``errors``.
    """

    def __init__(self, errors: list[FieldError]) -> None:
        self.errors = errors
        details = "; ".join(str(error) for error in errors)
        super().__init__(
            f"Record extraction failed with {len(errors)} error(s): {details}"
        )


@dataclass(frozen=True)
class RecordResult:
    """Extraction outcome for a single record.

    ``data`` maps field names to values; fields that failed are absent
    from ``data`` and reported in ``errors`` instead.
    """

    index: int
    data: dict[str, Any]
    errors: tuple[FieldError, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class ExtractionResult:
    """Outcome of a non-strict :meth:`CompiledPlan.extract` call."""

    records: tuple[RecordResult, ...]

    @property
    def errors(self) -> list[FieldError]:
        return [error for record in self.records for error in record.errors]

    @property
    def ok(self) -> bool:
        return all(record.ok for record in self.records)


@dataclass(frozen=True)
class _CompiledField:
    name: str
    cardinality: str
    xpath: str | None = None
    source_xpath: str | None = None
    jmespath_expr: Any = None


class _JsonParseFailure:
    __slots__ = ("message",)

    def __init__(self, message: str) -> None:
        self.message = message


@dataclass(frozen=True)
class RecordPlan:
    """A declarative extraction plan: a record root query plus field queries.

    ``record`` locates one root node per record; every field query then
    runs scoped to that root.  Call :meth:`compile` to validate and
    pre-compile all queries once, before running the plan on any number
    of documents.
    """

    record: Source
    fields: tuple[Field, ...] | list[Field]

    def compile(self) -> CompiledPlan:
        compiled_fields = []
        for field in self.fields:
            if field.selector is not None:
                compiled_fields.append(
                    _CompiledField(
                        name=field.name,
                        cardinality=field.cardinality,
                        xpath=field.selector._compile(),
                    )
                )
            else:
                assert field.jmespath_query is not None
                assert field.source is not None
                try:
                    jmespath_expr = jmespath.compile(field.jmespath_query)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid JMESPath query {field.jmespath_query!r} "
                        f"for field {field.name!r}: {exc}"
                    ) from exc
                compiled_fields.append(
                    _CompiledField(
                        name=field.name,
                        cardinality=field.cardinality,
                        source_xpath=field.source._compile(),
                        jmespath_expr=jmespath_expr,
                    )
                )
        return CompiledPlan(
            record_xpath=self.record._compile(), fields=compiled_fields
        )


def _missing_or_error(
    field: _CompiledField, record_index: int, stage: str
) -> tuple[Any, FieldError | None]:
    if field.cardinality == "many":
        return [], None
    if field.cardinality == "optional":
        return MISSING, None
    return None, FieldError(
        record_index, field.name, stage, "missing", "no match found"
    )


def _resolve_strings(
    field: _CompiledField, record_index: int, values: list[str], stage: str
) -> tuple[Any, FieldError | None]:
    if not values:
        return _missing_or_error(field, record_index, stage)
    if len(values) > 1 and field.cardinality != "many":
        return None, FieldError(
            record_index,
            field.name,
            stage,
            "conflict",
            f"expected at most 1 match, got {len(values)}",
        )
    if field.cardinality == "many":
        return list(values), None
    return values[0], None


class CompiledPlan:
    """A :class:`RecordPlan` with all queries validated and pre-compiled."""

    def __init__(self, record_xpath: str, fields: list[_CompiledField]) -> None:
        self.record_xpath = record_xpath
        self.fields = fields

    def extract(self, selector: Selector, *, strict: bool = True) -> ExtractionResult:
        """Run the plan on ``selector`` without modifying its tree.

        In strict mode (the default) every record is processed and a
        :class:`RecordExtractionError` listing all field errors is raised
        if any record failed.  With ``strict=False`` an
        :class:`ExtractionResult` is returned where each record carries
        its own errors.
        """
        roots = selector.xpath(self.record_xpath)
        records = []
        errors: list[FieldError] = []
        for index, root in enumerate(roots):
            record = self._extract_record(root, index)
            records.append(record)
            errors.extend(record.errors)
        if strict and errors:
            raise RecordExtractionError(errors)
        return ExtractionResult(records=tuple(records))

    def _extract_record(self, root: Selector, index: int) -> RecordResult:
        data: dict[str, Any] = {}
        errors: list[FieldError] = []
        json_cache: dict[str, Any] = {}
        for field in self.fields:
            value, error = self._run_field(field, root, index, json_cache)
            if error is not None:
                errors.append(error)
            else:
                data[field.name] = value
        return RecordResult(index=index, data=data, errors=tuple(errors))

    def _run_field(
        self,
        field: _CompiledField,
        root: Selector,
        index: int,
        json_cache: dict[str, Any],
    ) -> tuple[Any, FieldError | None]:
        if field.xpath is not None:
            values = root.xpath(field.xpath).getall()
            return _resolve_strings(field, index, values, "select")
        return self._run_jmespath_field(field, root, index, json_cache)

    def _run_jmespath_field(
        self,
        field: _CompiledField,
        root: Selector,
        index: int,
        json_cache: dict[str, Any],
    ) -> tuple[Any, FieldError | None]:
        assert field.source_xpath is not None
        texts = root.xpath(field.source_xpath).getall()
        if not texts:
            return _missing_or_error(field, index, "select")
        if len(texts) > 1:
            return None, FieldError(
                index,
                field.name,
                "select",
                "conflict",
                f"JSON source matched {len(texts)} nodes, expected 1",
            )
        text = texts[0]
        if text not in json_cache:
            try:
                json_cache[text] = json.loads(text)
            except ValueError as exc:
                json_cache[text] = _JsonParseFailure(str(exc))
        parsed = json_cache[text]
        if isinstance(parsed, _JsonParseFailure):
            return None, FieldError(
                index, field.name, "json", "json-parse", parsed.message
            )
        try:
            result = field.jmespath_expr.search(parsed)
        except Exception as exc:
            return None, FieldError(
                index, field.name, "jmespath", "jmespath-eval", str(exc)
            )
        if isinstance(result, list):
            if field.cardinality == "many":
                return [NULL if item is None else item for item in result], None
            return None, FieldError(
                index,
                field.name,
                "jmespath",
                "conflict",
                f"expected a single value, JMESPath query returned "
                f"{len(result)} values",
            )
        if field.cardinality == "many":
            if result is None:
                return [], None
            return None, FieldError(
                index,
                field.name,
                "jmespath",
                "not-a-list",
                f"expected a JSON array, got {type(result).__name__}",
            )
        return (NULL if result is None else result), None
