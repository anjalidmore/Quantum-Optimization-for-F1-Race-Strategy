"""
The study guide (``scripts/build_study_guide.py`` -> ``docs/study_guide.html``).

Builds the guide into ``tmp_path`` — never over the committed file — and checks
what the page promises: every document is in it, every link resolves, all ten
practicals are covered section by section, and no placeholder or code
reference is left unresolved.

``markdown`` is a dev-only dependency (requirements-dev.txt). Locally this
module skips without it; under CI (``CI`` set, as GitHub Actions does) a
missing ``markdown`` is an error, so the tests can never silently skip there.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

if os.environ.get("CI"):
    import markdown  # noqa: F401  - fail loudly: CI installs requirements-dev.txt
else:
    pytest.importorskip("markdown")

from scripts import build_study_guide as guide  # noqa: E402
from scripts.study_guide.content import PRACTICALS, SLIDES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("what", "how", "tech", "funcs", "code", "out", "apps")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("guide") / "docs" / "study_guide.html"
    report = guide.build(out, live=False)
    return out, out.read_text(encoding="utf-8"), report


def test_docs_folder_holds_no_markdown():
    # docs/ is HTML only. At the root, README.md and DESIGN.md stay Markdown:
    # DESIGN.md is read there by the design tooling in the DESIGN.md format.
    assert list((REPO_ROOT / "docs").rglob("*.md")) == []
    assert sorted(p.name for p in REPO_ROOT.glob("*.md")) == ["DESIGN.md", "README.md"]


def test_every_artifact_report_is_in_the_reports_tab(built):
    _, page, report = built
    expected = sorted(p.relative_to(REPO_ROOT).as_posix() for p in (REPO_ROOT / "artifacts").rglob("*.md"))
    assert expected, "no reports found"
    assert report.docs == len(expected)
    reports_tab = page[page.index('id="tab-reports"'):page.index('id="tab-showcase"')]
    for rel in expected:
        slug = guide.doc_slug(rel)
        assert f'id="{slug}"' in reports_tab and f'href="#{slug}"' in reports_tab, rel


def test_removed_sections_are_gone_from_the_study_guide(built):
    _, page, _ = built
    hub = page[page.index('id="tab-guide"'):page.index('id="tab-diagrams"')]
    for heading in ("Start here", "Requirements &amp; design", "Practical notes", "Running &amp; deployment",
                    "Testing &amp; status", "Reports · Practical"):
        assert heading not in page, heading
    # No hand-written docs page is rendered anywhere, and the hub holds no reports.
    for doc in (REPO_ROOT / "docs").glob("*.html"):
        if doc.name != "study_guide.html":
            assert f'id="doc-{doc.stem.lower()}' not in page and f"<code>docs/{doc.name}</code></a></p>" not in page
    assert 'data-doc="r-' not in hub and "doc-nav" not in hub


def test_report_listing_is_discovered_not_hard_coded(tmp_path, monkeypatch):
    """A report folder the categories table has never heard of still appears."""
    art = tmp_path / "artifacts"
    (art / "brand_new_stage").mkdir(parents=True)
    (art / "brand_new_stage" / "fresh_report.md").write_text("# Fresh report\n\nWritten by a new pipeline.\n")
    (art / "xai").mkdir()
    (art / "xai" / "SHAP_Report.md").write_text("# SHAP\n\n_Generated 2026-01-02 03:04 UTC._\n\n| a | b |\n|---|---|\n| 1 | 2 |\n")
    monkeypatch.setattr(guide, "REPO_ROOT", tmp_path)
    found = guide.discover_reports()
    assert [d.rel for d in found] == ["artifacts/xai/SHAP_Report.md", "artifacts/brand_new_stage/fresh_report.md"]
    assert guide._category_meta("brand_new_stage")[0] == "Brand new stage"
    assert guide._summary_and_date((art / "xai" / "SHAP_Report.md").read_text()) == ("A table of 1 rows.", "2026-01-02")


def test_tabs_back_bar_and_views(built):
    _, page, _ = built
    for tab in ("guide", "diagrams", "code", "reports", "showcase"):
        assert f'id="tabbtn-{tab}"' in page and f'id="tab-{tab}"' in page
    assert 'id="back-btn"' in page and 'aria-label="Back"' in page and 'id="breadcrumb"' in page
    # Every view knows its parent, so Back works on a direct page load too.
    views = re.findall(r'<(?:article|div)\b[^>]*\sid="([^"]+)"[^>]*\sdata-view\b([^>]*)>', page)
    ids = {v for v, _ in views}
    assert {"home", "diagrams", "p-home", "reports-home", "deck"} <= ids
    for vid, attrs in views:
        if vid != "home":
            parent = re.search(r'data-parent="([^"]+)"', attrs)
            assert parent and parent.group(1) in ids, vid


def test_diagrams_page_is_drawn_from_the_implementation(built):
    _, page, _ = built
    from app.intelligence.ml import pipeline
    from app.intelligence.ml.data_contract import load_and_validate
    from scripts.study_guide import diagrams

    tab = page[page.index('id="tab-diagrams"'):page.index('id="tab-code"')]
    for sid in ("arch", "build", "validation", "request", "decision", "inference", "deploy"):
        assert f'id="dg-{sid}"' in tab, sid
    _, _, folds, *_ = pipeline._split(load_and_validate(), "target_laptime")
    for f in folds:
        assert f">train {len(f.train_index)}<" in tab and f">val {len(f.val_index)}<" in tab
    stages = diagrams.build_stages()
    assert stages and all(html.escape(re.sub(r"\s*\(Task \d+\)", "", s)) in tab for s in stages)
    assert "localhost:" in tab and "/api/health" in tab


def test_viva_sections_render(built):
    _, page, _ = built
    for p in PRACTICALS:
        if p.get("viva"):
            n = p["number"]
            assert f'id="p-{n}-viva"' in page
            assert page.count('<details class="qa">', page.index(f'id="p-{n}-viva"')) >= len(p["viva"])


def test_in_page_anchors_all_resolve(built):
    _, page, report = built
    assert report.broken_anchors == []
    ids = set(re.findall(r'\sid="([^"]+)"', page))
    assert {h for h in re.findall(r'href="#([^"]+)"', page)} <= ids


def test_relative_links_and_images_point_at_real_files(built):
    out, page, _ = built
    urls = re.findall(r'<(?:a|img)\b[^>]*?\s(?:href|src)="([^"#][^"]*)"', page)
    relative = [u for u in urls if not re.match(r"^[a-z][a-z0-9+.-]*:", u)]
    assert relative
    missing = [u for u in relative if not (out.parent / unquote(urlparse(u).path)).exists()]
    assert missing == []


def test_broken_source_links_are_reported_not_emitted(built):
    _, page, report = built
    for entry in report.broken_links:
        target = entry.split(": ", 1)[1]
        assert f'data-broken="{target}"' in page


def test_all_ten_practicals_are_covered_section_by_section(built):
    _, page, _ = built
    assert [p["number"] for p in PRACTICALS] == list(range(1, 11))
    for p in PRACTICALS:
        n = p["number"]
        assert f'id="p-{n}"' in page
        for section in SECTIONS:
            assert f'id="p-{n}-{section}"' in page, (n, section)
        assert len(p["steps"]) >= 5 and len(p["functions"]) >= 5 and len(p["snippets"]) >= 2


def test_code_references_resolve_to_real_source(built):
    for p in PRACTICALS:
        for path, qual, _ in p["functions"]:
            ref = guide.extract_source(path, qual)
            assert ref.def_line >= 1 and ref.code.strip()
        for spec in p["snippets"]:
            ref = guide.extract_source(spec[0], spec[1], spec[3] if len(spec) > 3 else None)
            source_lines = (REPO_ROOT / spec[0]).read_text().splitlines()
            # The excerpt is the file's own text at the line numbers shown.
            assert ref.code.splitlines()[0].strip() == source_lines[ref.start - 1].strip()


def test_declared_outputs_exist():
    missing = [path for p in PRACTICALS for path, _ in p["outputs"] if not (REPO_ROOT / path).exists()]
    assert missing == []


def test_no_placeholder_survives_and_facts_come_from_artifacts(built):
    _, page, _ = built
    assert re.findall(r"\[\[[a-z0-9_]+\]\]", page) == []   # ``Callable[[Node], float]`` is code, not a placeholder
    facts = guide.collect_facts()
    assert "n/a" not in facts.values()
    from app.intelligence.expert_system.rule_base import build_rule_base
    assert facts["es_rules"] == str(len(build_rule_base()))


def test_showcase_has_every_slide_and_controls(built):
    _, page, _ = built
    for i in range(1, len(SLIDES) + 1):
        assert f'id="slide-{i}"' in page
    for control in ("prev-slide", "next-slide", "slide-jump", "fullscreen", "progress-fill"):
        assert f'id="{control}"' in page


def test_extract_source_fails_loudly_on_a_stale_reference():
    with pytest.raises(LookupError):
        guide.extract_source("app/intelligence/search/algorithms.py", "no_such_function")
    with pytest.raises(LookupError):
        guide.extract_source("app/intelligence/search/algorithms.py", "astar_search", ("not there", "x"))


def test_github_heading_slugs():
    assert guide.gh_slug("8. Retraining on a different race") == "8-retraining-on-a-different-race"
    assert guide.gh_slug('What "Not available" means when it appears') == "what-not-available-means-when-it-appears"
