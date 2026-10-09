"""Clinical hardening: the audit log's chain, de-identification, sign-off, the report.

What has to hold: the log is a hash chain, so an entry edited, removed or
slipped in afterwards is found, and a redacted one is reported as redacted;
identifying text anywhere a person could have typed it is found (dates,
record numbers, e-mail, phone, the names given) and redacted in place, with
the names never written anywhere; no report until the work is done and the
checks pass; a sign-off is for the content it saw and is superseded when any
of it changes; reports are numbered, never overwritten, and carry a manifest
of checksums; the report's statements of what the methods were measured to
do match the committed study tables; and the window does all of it, down to
a PDF.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from onset_hfo.case import annotations as marks
from onset_hfo.case import bridges, deid
from onset_hfo.case.model import Case

ROOT = Path(__file__).resolve().parent.parent
SF = 512.0
NAMES = [f"LA{i}" for i in range(1, 5)] + [f"LB{i}" for i in range(1, 5)]
ONSETS = (50.0, 150.0)


# -- the log ----------------------------------------------------------------------------------
def test_the_log_is_a_chain_that_shows_tampering(tmp_path):
    case = Case.create(tmp_path / "case", "P60")
    for i in range(4):
        case.record("did something", f"step {i}")
    assert case.verify_log()["ok"] and case.verify_log()["checked"] == 5
    again = Case.open(case.root)
    assert again.verify_log()["ok"], "the chain survives saving and opening"

    edited = Case.open(case.root)
    edited.log[2]["detail"] = "something else"
    result = edited.verify_log()
    assert not result["ok"] and result["broken_at"] == 2 and "changed" in result["summary"]

    removed = Case.open(case.root)
    del removed.log[2]
    assert removed.verify_log()["broken_at"] == 2

    inserted = Case.open(case.root)
    inserted.log.insert(1, dict(inserted.log[1]))
    assert not inserted.verify_log()["ok"]

    redacted = Case.open(case.root)
    redacted.log[2]["detail"] = "[removed]"
    redacted.log[2]["redacted"] = True
    result = redacted.verify_log()
    assert result["ok"] and result["redacted"] == [2] and "1 redacted" in result["summary"]

    old = Case.open(case.root)
    old.log = [{"at": "x", "by": "y", "action": "from before", "detail": ""}] + old.log
    old.log[1]["prev"] = ""
    assert old.verify_log()["before_chain"] == 1


# -- de-identification --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def seizures_edf(tmp_path_factory):
    rng = np.random.default_rng(4)
    t = np.arange(int(200 * SF)) / SF
    data = rng.normal(0, 1.0, (len(NAMES), len(t)))
    data += 4.0 * np.sin(2 * np.pi * 6.0 * t + rng.uniform(0, 6, (len(NAMES), 1)))
    for onset in ONSETS:
        early = (t >= onset + 1.0) & (t < onset + 31.0)
        data[1, early] += 6.0 * np.sin(2 * np.pi * 40.0 * t[early])
    raw = mne.io.RawArray(data * 1e-5, mne.create_info(NAMES, SF, "seeg"), verbose="ERROR")
    path = tmp_path_factory.mktemp("edf") / "SMITH_JOHN_2024-03-02.edf"
    mne.export.export_raw(path, raw, fmt="edf", overwrite=True, verbose="ERROR")
    return path


@pytest.fixture
def case(seizures_edf, tmp_path, monkeypatch):
    monkeypatch.setenv("ONSET_REVIEW_READS", str(tmp_path / "reads"))
    monkeypatch.setenv("ONSET_ATLAS_DIR", str(tmp_path / "no-atlas"))
    case = Case.create(tmp_path / "case", "P61", note="MRN 12345678, call 555-123-4567")
    bridges.convert(seizures_edf, case, channel_types={n: "seeg" for n in NAMES})
    case.save_marks("01", pd.DataFrame([
        marks.new_mark(ONSETS[0], "seizure-onset"), marks.new_mark(ONSETS[1], "seizure-onset"),
        marks.new_mark(20.0, "note", value="John Smith pushed the button")]))
    return case


def _text_of(folder: Path) -> str:
    return "\n".join(p.read_text(encoding="utf-8", errors="ignore").lower()
                     for p in folder.rglob("*") if p.is_file() and p.suffix in
                     (".json", ".tsv", ".md", ".html", ".vhdr", ".vmrk"))


def test_identifying_text_is_found_and_redacted_and_names_are_not_kept(case):
    found = deid.check_case(case, names=["Smith", "John"])
    kinds = {(f.where.split(" ")[0], f.what) for f in found}
    assert ("the", "a long number (a record number?)") in kinds        # the case's note
    assert ("the", "a phone number") in kinds
    assert ("run", "a date") in kinds and ("run", "a name you gave") in kinds
    assert any(f.where.startswith("mark at 20") and f.what == "a name you gave" for f in found)
    assert any(f.where.startswith("log entry") for f in found), "the log copied the file name"
    assert all("smith" not in f.excerpt.lower() for f in found), "excerpts are masked"
    changed = deid.redact(case, found, names=["Smith", "John"], by="dr test")
    assert changed >= 5
    assert deid.check_case(case, names=["Smith", "John"]) == []
    text = _text_of(case.root)
    assert "smith" not in text and "12345678" not in text and "2024-03-02" not in text
    log = case.verify_log()
    assert log["ok"] and log["redacted"], "redacted entries show as redacted, chain intact"
    assert case.log[-1]["action"] == "redacted identifying text"
    assert "smith" not in json.dumps(case.log).lower()


# -- the report ---------------------------------------------------------------------------------
def _ready(case):
    from onset_review.caseictal import run_ictal

    case.mark_step("channels", True, by="dr test")
    run_ictal(case)
    deid.redact(case, deid.check_case(case, names=["Smith", "John"]), names=["Smith", "John"])
    case.record("checked de-identification", "0 finding(s); 2 name(s) looked for")


def test_no_report_before_the_work_and_the_checks(case):
    from onset_review import casereport

    rows = dict((w, ok) for w, ok, _d in casereport.readiness(case))
    assert not rows["An analysis run"] and not rows["Channels checked"]
    assert not rows["De-identification checked"] and rows["Audit log intact"]
    with pytest.raises(ValueError, match="not ready"):
        casereport.produce(case, "dr test")
    _ready(case)
    assert all(ok for _w, ok, _d in casereport.readiness(case))
    case.log[3]["detail"] = "tampered"
    assert dict((w, ok) for w, ok, _d in casereport.readiness(case))["Audit log intact"] is False


def test_signoffs_are_for_content_and_versions_are_kept(case):
    from onset_review import casereport

    _ready(case)
    first = casereport.produce(case, "dr test")
    assert first.version == 1 and not first.signed and first.pdf is None
    text = first.html.read_text()
    assert "Not signed off for this content" in text and "Case P61" in text
    assert "LA1-LA2" in text and "smith" not in text.lower()
    with pytest.raises(ValueError):
        casereport.sign_off(case, "  ", "role")
    casereport.sign_off(case, "dr test", "neurophysiologist")
    assert casereport.signoff_status(case)[0]["current"]
    with pytest.raises(ValueError, match="already"):
        casereport.sign_off(case, "dr test", "neurophysiologist")
    second = casereport.produce(case, "dr test")
    assert second.version == 2 and second.signed and case.step_done("report")
    assert "Signed off for this content" in second.html.read_text()
    manifest = json.loads((second.folder / "manifest.json").read_text())
    assert manifest["fingerprint"] == casereport.fingerprint(case)
    assert manifest["files"]["report.html"] == casereport._sha(second.html)
    assert (first.folder / "report.html").read_text() == text, "v1 is never overwritten"
    case.set_zone(["LA1"], by="dr test")
    assert not casereport.signoff_status(case)[0]["current"], "content changed: superseded"
    third = casereport.produce(case, "dr test")
    assert not third.signed and [v["version"] for v in casereport.report_versions(case)] == \
        [1, 2, 3]
    assert [e["action"] for e in case.log].count("produced a report") == 3


def test_the_validation_statements_match_the_study_tables():
    from onset_review.casereport import VALIDATION

    ictal = json.loads((ROOT / "data" / "ictal" / "summary.json").read_text())
    control = json.loads((ROOT / "data" / "ictal" / "control_summary.json").read_text())
    template = json.loads((ROOT / "data" / "template" / "summary.json").read_text())
    block = ictal["ei_all"]
    assert (f"{ictal['patients']} patients: median per-patient AUC {block['median']:.2f} "
            f"(95% interval {block['ci_low']:.2f}–{block['ci_high']:.2f})") in VALIDATION["ictal"]
    assert f"{control['ei_per_seizure']['median']:.2f} on the same seizures" in \
        VALIDATION["ictal"]
    assert f"in {round(ictal['top_in_zone_patients'] * 28)} of 28" in VALIDATION["ictal"]
    variants = template["variants"]
    assert f"median {variants['own']['median_mm']:.1f} mm off" in VALIDATION["template"]
    assert f"{variants['3.5mm']['median_mm']:.1f} mm at a 3.5 mm pitch" in VALIDATION["template"]
    assert f"{template['patients']} patients" in VALIDATION["template"]
    assert f"{template['real_labels']['in'] * 100:.0f}% of" in VALIDATION["template"]


# -- the window --------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    pytest.importorskip("mne_qt_browser", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_the_window_checks_signs_and_produces_a_pdf(qapp, case):
    from onset_review.caseictal import run_ictal
    from onset_review.casewindow import CaseWindow

    case.mark_step("channels", True, by="dr test")
    run_ictal(case)
    window = CaseWindow(case, reader="dr test")
    try:
        assert window.show_step("report")
        page = window.report_page
        assert not page.produce_button.isEnabled()
        assert "De-identification checked" in page.status.text()
        page.names.setText("Smith, John")
        found = page.check()
        assert found and page.redact_button.isEnabled() and page.found.rowCount() == len(found)
        page.redact()
        assert page.findings == [] and page.found.rowCount() == 0
        assert page.produce_button.isEnabled(), page.status.text()
        page.role.setText("neurophysiologist")
        page.sign()
        assert page.signoffs.item(0, 3).text() == "yes"
        result = page.produce()
        assert result.signed and result.pdf.exists() and result.pdf.stat().st_size > 5000
        assert result.pdf.read_bytes()[:5] == b"%PDF-"
        assert page.versions.rowCount() == 1 and page.versions.item(0, 3).text() == "yes"
        assert case.step_done("report")
        assert "smith" not in _text_of(case.root)
    finally:
        window.close()
