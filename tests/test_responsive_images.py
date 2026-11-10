"""Tests for parsel.responsive and Selector.resolve_image."""

import pytest

from parsel import Selector
from parsel.responsive import eval_media, eval_sizes

PICTURE_HTML = """
<html><head><base href="https://cdn.example.com/assets/"></head><body>
<picture>
  <source type="image/avif" srcset="a-400.avif 400w, a-800.avif 800w" sizes="50vw">
  <source type="image/webp" media="screen and (min-width: 600px)"
          srcset="w-400.webp 400w, w-800.webp 800w, w-1600.webp 1600w"
          sizes="(min-width: 1000px) 50vw, 100vw">
  <img src="fallback.jpg" srcset="f-1x.jpg 1x, f-2x.jpg 2x">
</picture>
</body></html>
"""

MIME = {"image/webp", "image/jpeg"}


def make_selector(html=PICTURE_HTML, base_url="https://example.com/page/"):
    return Selector(text=html, base_url=base_url)


class TestMedia:
    @pytest.mark.parametrize(
        ("media", "vw", "expected"),
        [
            ("screen", 800, True),
            ("all", 800, True),
            ("print", 800, False),
            ("(min-width: 600px)", 800, True),
            ("(min-width: 600px)", 500, False),
            ("(max-width: 600px)", 500, True),
            ("screen and (min-width: 600px)", 800, True),
            ("screen and (min-width: 600px)", 500, False),
            ("(min-width: 400px) and (max-width: 600px)", 500, True),
            ("(min-width: 400px) and (max-width: 600px)", 700, False),
            ("(max-width: 600px), (min-width: 900px)", 1000, True),
            ("(max-width: 600px), (min-width: 900px)", 700, False),
            # unsupported expressions are undetermined, never a match
            ("(orientation: portrait)", 800, None),
            ("only screen and (min-width: 600px)", 800, None),
            ("not screen", 800, None),
            ("(min-width: 60em)", 800, None),
            ("(min-width: 600px) or (max-width: 300px)", 800, None),
            ("(max-width: 600px), (orientation: portrait)", 700, None),
        ],
    )
    def test_eval_media(self, media, vw, expected):
        assert eval_media(media, vw) is expected


class TestSizes:
    @pytest.mark.parametrize(
        ("sizes", "vw", "expected"),
        [
            ("100vw", 800, 800.0),
            ("50vw", 800, 400.0),
            ("300px", 800, 300.0),
            ("(min-width: 1000px) 50vw, 100vw", 1200, 600.0),
            ("(min-width: 1000px) 50vw, 100vw", 800, 800.0),
            ("(max-width: 600px) 100vw, 33vw", 500, 500.0),
        ],
    )
    def test_supported(self, sizes, vw, expected):
        length, error = eval_sizes(sizes, vw)
        assert error is None
        assert length == pytest.approx(expected)

    @pytest.mark.parametrize(
        "sizes",
        [
            "calc(100vw - 20px)",
            "10em",
            "(orientation: portrait) 50vw, 100vw",
            "(min-width: 600px)",
            "",
        ],
    )
    def test_undetermined(self, sizes):
        length, error = eval_sizes(sizes, 800)
        assert length is None
        assert error is not None


class TestResolveImage:
    def test_same_picture_different_viewport_and_dpr(self):
        img = make_selector().css("img")[0]
        desktop = img.resolve_image(
            viewport_width=1200, dpr=1.0, supported_mime_types=MIME
        )
        hidpi = img.resolve_image(
            viewport_width=1200, dpr=2.0, supported_mime_types=MIME
        )
        mobile = img.resolve_image(
            viewport_width=360, dpr=2.0, supported_mime_types=MIME
        )
        assert desktop.url == "https://cdn.example.com/assets/w-800.webp"
        assert hidpi.url == "https://cdn.example.com/assets/w-1600.webp"
        # media no longer matches: falls back to the img srcset x candidates
        assert mobile.url == "https://cdn.example.com/assets/f-2x.jpg"
        assert desktop.url != hidpi.url != mobile.url
        assert desktop.source is not None
        assert desktop.source.root.tag == "source"
        assert mobile.source is None
        for resolution in (desktop, hidpi, mobile):
            assert resolution.img.root.tag == "img"
            assert resolution.reason

    def test_unsupported_mime_skipped_and_reported(self):
        img = make_selector().css("img")[0]
        resolution = img.resolve_image(
            viewport_width=1200, dpr=1.0, supported_mime_types=MIME
        )
        assert any("image/avif" in issue for issue in resolution.issues)
        assert "avif" not in resolution.url

    def test_all_mime_unsupported_falls_back(self):
        img = make_selector().css("img")[0]
        resolution = img.resolve_image(
            viewport_width=1200, dpr=1.0, supported_mime_types={"image/png"}
        )
        assert resolution.url == "https://cdn.example.com/assets/f-1x.jpg"
        assert resolution.source is None

    def test_no_matching_source_falls_back_to_src(self):
        sel = make_selector(
            '<picture>'
            '<source type="image/avif" srcset="a.avif 1x">'
            '<source media="(min-width: 2000px)" srcset="big.jpg 1x">'
            '<img src="plain.jpg">'
            "</picture>"
        )
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://example.com/page/plain.jpg"
        assert "src" in resolution.reason

    def test_relative_url_uses_document_base(self):
        sel = make_selector('<img src="img/pic.png">')
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://example.com/page/img/pic.png"

    def test_base_href_takes_precedence(self):
        img = make_selector().css("img")[0]
        resolution = img.resolve_image(
            viewport_width=360, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://cdn.example.com/assets/f-1x.jpg"

    def test_undetermined_media_is_not_a_match(self):
        sel = make_selector(
            '<picture>'
            '<source media="(orientation: portrait)" srcset="portrait.jpg 1x">'
            '<img src="plain.jpg">'
            "</picture>"
        )
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://example.com/page/plain.jpg"
        assert any("undetermined media" in issue for issue in resolution.issues)

    def test_undetermined_sizes_is_not_a_match(self):
        sel = make_selector(
            '<picture>'
            '<source srcset="a.jpg 400w, b.jpg 800w" sizes="calc(100vw - 10px)">'
            '<img src="plain.jpg">'
            "</picture>"
        )
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://example.com/page/plain.jpg"
        assert any("sizes" in issue for issue in resolution.issues)

    def test_mixed_and_duplicate_descriptors_reported(self):
        sel = make_selector(
            '<img src="x.jpg" srcset="a.jpg 400w, b.jpg 1x, c.jpg 2x, d.jpg 2x">'
        )
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.5, supported_mime_types=MIME
        )
        assert any("mixed" in issue for issue in resolution.issues)
        assert any("duplicate" in issue for issue in resolution.issues)
        # w candidates use the default 100vw sizes, x candidates their density
        assert resolution.url == "https://example.com/page/c.jpg"
        assert resolution.density == 2.0

    def test_x_descriptor_selection(self):
        sel = make_selector('<img src="x.jpg" srcset="a.jpg 1x, b.jpg 2x, c.jpg 3x">')
        img = sel.css("img")[0]
        low = img.resolve_image(viewport_width=800, dpr=1.0, supported_mime_types=MIME)
        high = img.resolve_image(viewport_width=800, dpr=2.5, supported_mime_types=MIME)
        assert low.url.endswith("/a.jpg")
        assert high.url.endswith("/c.jpg")

    def test_selectorlist_matches_single_selector(self):
        sel = make_selector(
            '<img src="a.jpg">'
            '<picture><source srcset="b-400.jpg 400w, b-800.jpg 800w" sizes="100vw">'
            '<img src="b.jpg"></picture>'
        )
        imgs = sel.css("img")
        kwargs = {"viewport_width": 800, "dpr": 1.0, "supported_mime_types": MIME}
        listed = imgs.resolve_image(**kwargs)
        singles = [img.resolve_image(**kwargs) for img in imgs]
        assert len(listed) == 2
        assert [r.url for r in listed] == [r.url for r in singles]
        assert listed[0].url == "https://example.com/page/a.jpg"
        assert listed[1].url == "https://example.com/page/b-800.jpg"

    def test_picture_root_selector(self):
        sel = make_selector()
        resolution = sel.css("picture")[0].resolve_image(
            viewport_width=1200, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url == "https://cdn.example.com/assets/w-800.webp"
        assert resolution.img.root.tag == "img"

    def test_no_img_raises(self):
        sel = make_selector("<div><span>text</span></div>")
        with pytest.raises(ValueError, match="no <img>"):
            sel.css("div")[0].resolve_image(
                viewport_width=800, dpr=1.0, supported_mime_types=MIME
            )

    def test_non_html_type_raises(self):
        sel = Selector(text='{"a": 1}')
        with pytest.raises(ValueError, match="html"):
            sel.resolve_image(
                viewport_width=800, dpr=1.0, supported_mime_types=MIME
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"viewport_width": 0, "dpr": 1.0},
            {"viewport_width": 800, "dpr": 0},
        ],
    )
    def test_invalid_environment(self, kwargs):
        img = make_selector("<img src='a.jpg'>").css("img")[0]
        with pytest.raises(ValueError):
            img.resolve_image(supported_mime_types=MIME, **kwargs)

    def test_no_url_at_all(self):
        sel = make_selector("<img>")
        resolution = sel.css("img")[0].resolve_image(
            viewport_width=800, dpr=1.0, supported_mime_types=MIME
        )
        assert resolution.url is None
        assert resolution.reason
