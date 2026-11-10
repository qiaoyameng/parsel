"""Responsive image candidate resolution for HTML ``img``/``picture``.

Pure-markup implementation of a subset of the HTML standard "responsive
image" selection algorithm.  It never executes JavaScript and never
performs network requests; the caller explicitly supplies the viewport
width, the device pixel ratio (DPR) and the set of supported MIME types.

Supported in this first version:

- ``media``: the ``screen``/``all`` media types, ``min-width`` and
  ``max-width`` features (in ``px``) combined with ``and``, and
  comma-separated media query lists.  Anything else is reported as
  *undetermined* and is never treated as a match.
- ``sizes``: lengths in ``px`` and ``vw`` plus a final default length.
  Anything else (``calc()``, ``em``, ...) is *undetermined*.
- ``srcset``: ``x`` and ``w`` descriptors, each handled by its own
  rules.  Mixing descriptor kinds or repeating a descriptor value is
  reported in the result issues.
"""

from __future__ import annotations

import re
import typing
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

if typing.TYPE_CHECKING:
    from collections.abc import Collection

# Tri-state used by media/sizes evaluation: True / False / None (undetermined).
_TriState = bool | None

_LENGTH_RE = re.compile(r"^(\d+(?:\.\d+)?)(px|vw)$")
_WIDTH_CONDITION_RE = re.compile(
    r"^\(\s*(min|max)-width\s*:\s*(\d+(?:\.\d+)?)px\s*\)$"
)
_X_DESCRIPTOR_RE = re.compile(r"^(\d+(?:\.\d+)?)x$")
_W_DESCRIPTOR_RE = re.compile(r"^(\d+)w$")


@dataclass
class ImageCandidate:
    """A single ``srcset`` candidate with its computed density."""

    url: str
    descriptor: str  # "w", "x" or "none"
    descriptor_value: float  # width in px for "w", density for "x" (1.0 if none)
    density: float | None  # effective density, None when undetermined


@dataclass
class ImageResolution:
    """Outcome of resolving one ``img`` element.

    ``img`` and ``source`` hold the associated elements (lxml elements
    when produced by :func:`resolve_image_for_element`, or Selector
    objects when produced through the Selector API).  ``issues``
    collects reports such as mixed/duplicate descriptors and
    undetermined ``media``/``sizes`` expressions.
    """

    url: str | None
    reason: str
    img: Any = None
    source: Any = None
    density: float | None = None
    candidates: list[ImageCandidate] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)


def _eval_media_query(query: str, viewport_width: float) -> _TriState:
    """Evaluate one media query (no commas) against the viewport width."""
    query = query.strip()
    if not query:
        return None
    parts = [p.strip() for p in re.split(r"\band\b", query)]
    saw_undetermined = False
    for index, part in enumerate(parts):
        if not part:
            return None
        condition = _WIDTH_CONDITION_RE.match(part)
        if condition is not None:
            limit = float(condition.group(2))
            if condition.group(1) == "min":
                if viewport_width < limit:
                    return False
            elif viewport_width > limit:
                return False
            continue
        if index == 0 and part in ("screen", "all"):
            continue
        if index == 0 and part in ("print", "speech"):
            return False
        # "only"/"not", unknown media types, unsupported features or units.
        saw_undetermined = True
    return None if saw_undetermined else True


def eval_media(media: str, viewport_width: float) -> _TriState:
    """Evaluate a media query list; ``None`` means undetermined."""
    saw_undetermined = False
    for query in media.split(","):
        result = _eval_media_query(query, viewport_width)
        if result is True:
            return True
        if result is None:
            saw_undetermined = True
    return None if saw_undetermined else False


def _parse_length(token: str, viewport_width: float) -> float | None:
    match = _LENGTH_RE.match(token.strip())
    if match is None:
        return None
    value = float(match.group(1))
    if match.group(2) == "vw":
        return value * viewport_width / 100.0
    return value


def eval_sizes(
    sizes: str, viewport_width: float
) -> tuple[float | None, str | None]:
    """Evaluate a ``sizes`` attribute to a length in CSS pixels.

    Returns ``(length, None)`` on success or ``(None, reason)`` when the
    value cannot be determined.
    """
    parts = [p.strip() for p in sizes.split(",") if p.strip()]
    if not parts:
        return None, "empty sizes attribute"
    for index, part in enumerate(parts):
        is_last = index == len(parts) - 1
        condition_match = re.match(r"^\((.*)\)\s+(\S+)$", part)
        if condition_match is not None and not is_last:
            media_result = eval_media(
                f"({condition_match.group(1)})", viewport_width
            )
            if media_result is None:
                return (
                    None,
                    f"undetermined media condition in sizes: {part!r}",
                )
            if media_result is False:
                continue
            length = _parse_length(condition_match.group(2), viewport_width)
            if length is None:
                return None, f"unsupported length in sizes: {part!r}"
            return length, None
        if is_last and condition_match is None:
            length = _parse_length(part, viewport_width)
            if length is None:
                return None, f"unsupported default length in sizes: {part!r}"
            return length, None
        return None, f"unsupported sizes component: {part!r}"
    return None, "no matching sizes condition"


def _parse_srcset(
    srcset: str, sizes_px: float | None, issues: list[str]
) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    seen: set[tuple[str, float]] = set()
    kinds: set[str] = set()
    for raw_part in srcset.split(","):
        tokens = raw_part.split()
        if not tokens:
            continue
        url, descriptors = tokens[0], tokens[1:]
        kind = "none"
        value = 1.0
        if len(descriptors) > 1:
            issues.append(
                f"multiple descriptors {descriptors!r} for candidate {url!r}; "
                "only the first is used"
            )
        if descriptors:
            descriptor = descriptors[0]
            width_match = _W_DESCRIPTOR_RE.match(descriptor)
            density_match = _X_DESCRIPTOR_RE.match(descriptor)
            if width_match is not None:
                kind, value = "w", float(width_match.group(1))
            elif density_match is not None:
                kind, value = "x", float(density_match.group(1))
            else:
                issues.append(
                    f"unsupported descriptor {descriptor!r} for candidate "
                    f"{url!r}; candidate ignored"
                )
                continue
        kinds.add(kind)
        key = (kind, value)
        if kind != "none" and key in seen:
            issues.append(
                f"duplicate descriptor {value:g}{kind} for candidate {url!r}; "
                "first occurrence wins"
            )
            continue
        seen.add(key)
        if kind == "w":
            density = value / sizes_px if sizes_px else None
        else:
            density = value
        candidates.append(
            ImageCandidate(
                url=url,
                descriptor=kind,
                descriptor_value=value,
                density=density,
            )
        )
    if "w" in kinds and kinds != {"w"}:
        issues.append(
            "mixed 'w' and 'x'/bare descriptors in one srcset; each "
            "candidate is still handled by its own rule"
        )
    return candidates


def _select_candidate(
    candidates: list[ImageCandidate], dpr: float
) -> ImageCandidate | None:
    usable = [c for c in candidates if c.density is not None]
    if not usable:
        return None
    above = [c for c in usable if c.density >= dpr]
    if above:
        return min(above, key=lambda c: c.density or 0.0)
    return max(usable, key=lambda c: c.density or 0.0)


def _resolve_srcset(
    srcset: str,
    sizes: str | None,
    viewport_width: float,
    dpr: float,
    base_url: str | None,
    issues: list[str],
) -> tuple[ImageCandidate | None, list[ImageCandidate]]:
    sizes_px: float | None = None
    has_w = any(
        _W_DESCRIPTOR_RE.match(tok)
        for part in srcset.split(",")
        for tok in part.split()[1:]
    )
    if has_w:
        sizes_px, sizes_error = eval_sizes(sizes or "100vw", viewport_width)
        if sizes_error is not None:
            issues.append(sizes_error)
    candidates = _parse_srcset(srcset, sizes_px, issues)
    for candidate in candidates:
        candidate.url = urljoin(base_url, candidate.url) if base_url else candidate.url
    return _select_candidate(candidates, dpr), candidates


def _iter_picture_sources(img_el: Any) -> list[Any]:
    """Return the ``source`` elements preceding ``img_el`` in its picture.

    libxml2 does not treat ``source`` as a void element and may nest the
    elements instead of keeping them siblings, so walk up to the
    ``picture`` ancestor and collect ``source`` elements in document
    order instead of relying on the exact tree shape.
    """
    picture = img_el.getparent()
    while picture is not None and picture.tag != "picture":
        picture = picture.getparent()
    if picture is None:
        return []
    sources = []
    for el in picture.iter():
        if el is img_el:
            break
        if el.tag == "source":
            sources.append(el)
    return sources


def resolve_image_for_element(
    img_el: Any,
    *,
    viewport_width: float,
    dpr: float,
    supported_mime_types: Collection[str],
) -> ImageResolution:
    """Resolve the image URL a browser would pick for one ``img`` element.

    Works purely on the parsed markup: no JavaScript is executed and no
    image is requested.  ``viewport_width`` is in CSS pixels, ``dpr`` is
    the device pixel ratio and ``supported_mime_types`` is the set of
    MIME types the caller can decode.
    """
    if viewport_width <= 0:
        raise ValueError("viewport_width must be positive")
    if dpr <= 0:
        raise ValueError("dpr must be positive")
    supported = {mime.strip().lower() for mime in supported_mime_types}
    base_url = img_el.base
    issues: list[str] = []

    for source_el in _iter_picture_sources(img_el):
        mime = source_el.get("type")
        if mime is not None and mime.strip().lower() not in supported:
            issues.append(f"skipped <source>: unsupported MIME type {mime!r}")
            continue
        media = source_el.get("media")
        if media is not None:
            media_result = eval_media(media, viewport_width)
            if media_result is None:
                issues.append(
                    f"skipped <source>: undetermined media expression {media!r}"
                )
                continue
            if media_result is False:
                issues.append(f"skipped <source>: media {media!r} does not match")
                continue
        srcset = source_el.get("srcset")
        if not srcset or not srcset.strip():
            issues.append("skipped <source>: missing srcset")
            continue
        candidate, candidates = _resolve_srcset(
            srcset,
            source_el.get("sizes"),
            viewport_width,
            dpr,
            base_url,
            issues,
        )
        if candidate is None:
            issues.append("skipped <source>: no usable srcset candidate")
            continue
        return ImageResolution(
            url=candidate.url,
            reason=(
                f"first <source> with matching type/media; srcset candidate "
                f"{candidate.descriptor_value:g}{candidate.descriptor} "
                f"has effective density {candidate.density:g}x for DPR {dpr:g}"
            ),
            img=img_el,
            source=source_el,
            density=candidate.density,
            candidates=candidates,
            issues=issues,
        )

    srcset = img_el.get("srcset")
    if srcset and srcset.strip():
        candidate, candidates = _resolve_srcset(
            srcset,
            img_el.get("sizes"),
            viewport_width,
            dpr,
            base_url,
            issues,
        )
        if candidate is not None:
            return ImageResolution(
                url=candidate.url,
                reason=(
                    f"img srcset candidate "
                    f"{candidate.descriptor_value:g}{candidate.descriptor} "
                    f"has effective density {candidate.density:g}x for DPR {dpr:g}"
                ),
                img=img_el,
                source=None,
                density=candidate.density,
                candidates=candidates,
                issues=issues,
            )
        issues.append("img srcset yielded no usable candidate")

    src = img_el.get("src")
    if src:
        return ImageResolution(
            url=urljoin(base_url, src) if base_url else src,
            reason="fallback to img src attribute",
            img=img_el,
            source=None,
            density=None,
            candidates=[],
            issues=issues,
        )
    return ImageResolution(
        url=None,
        reason="no matching <source>, no usable srcset and no src attribute",
        img=img_el,
        source=None,
        density=None,
        candidates=[],
        issues=issues,
    )
