"""The log, the problem report, and the model check.

Asked for after a hang that could only be reported as a photograph of a
terminal: a log of what the window did, a report to send, and a way to
measure how well the local model serves the assistant and *Write code* on
the machine it runs on. What has to hold:

* the log is written where the report looks, rotates, and catches warnings,
  uncaught errors and the background jobs' timings;
* the report holds versions, the log and the latest model check, with the
  home folder written as ``~``;
* the model check grades the assistant against what each question should
  get, and runs each code draft on the recording and checks what it printed
  against the right answer -- a wrong number is wrong, a draft that raises
  failed, a draft flagged as deleting files is never run.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import types
import zipfile
from pathlib import Path

import pytest

from onset_review import modelcheck

ROOT = Path(__file__).resolve().parents[1]


# -- the log and the report (in a process of their own: setup is process-wide) --------
LOG_SCRIPT = textwrap.dedent('''
    import json, logging, sys, warnings, zipfile
    from pathlib import Path
    from onset_review import applog

    path = applog.setup()
    assert applog.setup() == path, "once per process"
    logging.getLogger("onset_review.workers").info("Busy started")
    warnings.warn("a warning worth keeping")
    try:
        raise ValueError("an error with a traceback")
    except ValueError:
        logging.getLogger("onset_review").exception("caught")
    checks = applog.log_dir() / "model-checks"
    checks.mkdir(parents=True, exist_ok=True)
    (checks / "model-check-1.json").write_text(json.dumps({"home": str(Path.home())}))
    (checks / "model-check-1.md").write_text("# Model check")
    text = applog.report_text()
    report = applog.write_report(Path(sys.argv[1]) / "report")
    with zipfile.ZipFile(report) as bundle:
        names = bundle.namelist()
        check = bundle.read("model-check-1.json").decode()
    print(json.dumps({"path": str(path), "text": text, "names": names, "check": check,
                      "report": str(report)}))
''')


def test_the_log_and_the_report_hold_what_a_problem_needs(tmp_path):
    env = dict(os.environ, ONSET_REVIEW_LOG_DIR=str(tmp_path / "logs"),
               ONSET_REVIEW_CONFIG_DIR=str(tmp_path / "config"), ONSET_ASSISTANT_NO_PROBE="1",
               PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-c", LOG_SCRIPT, str(tmp_path)], cwd=ROOT, env=env,
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    log = Path(result["path"]).read_text()
    assert result["path"].endswith("logs/onset-review.log")
    assert "started onset-review" in log
    assert "Busy started" in log
    assert "a warning worth keeping" in log, "warnings are captured"
    assert "ValueError: an error with a traceback" in log and "Traceback" in log
    text = result["text"]
    assert "onset-review:" in text and "numpy:" in text and "Busy started" in text
    assert "model-check-1.json" in text
    assert result["report"].endswith("report.zip")
    assert {"report.txt", "onset-review.log", "crash.log", "model-check-1.json",
            "model-check-1.md"} <= set(result["names"])
    assert str(Path.home()) not in result["check"] and "~" in result["check"], \
        "the home folder is written as ~"


def test_the_log_goes_beside_a_redirected_settings_folder(monkeypatch, tmp_path):
    from onset_review import applog

    monkeypatch.delenv("ONSET_REVIEW_LOG_DIR", raising=False)
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "cfg"))
    assert applog.log_dir() == tmp_path / "cfg" / "logs"
    monkeypatch.delenv("ONSET_REVIEW_CONFIG_DIR")
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "state"))
    assert applog.log_dir() == tmp_path / "state" / "onset-review"


# -- grading ---------------------------------------------------------------------------
def _answer(refused=False, verified=True, mode="data"):
    return types.SimpleNamespace(refused=refused, verified=verified, mode=mode, reason="r")


def test_a_question_is_graded_against_what_it_should_get():
    grade = modelcheck.grade_question
    assert grade("verified", _answer())[0]
    assert not grade("verified", _answer(verified=False))[0]
    assert not grade("verified", _answer(refused=True))[0]
    assert grade("refused", _answer(refused=True))[0]
    assert not grade("refused", _answer())[0], "answering a clinical decision fails"
    assert grade("any", _answer(verified=False, mode="general"))[0]


def test_a_draft_runs_in_a_scratch_folder_with_a_time_limit(tmp_path):
    here = os.getcwd()
    ran = modelcheck.run_draft("import os\nprint(x * 2, os.getcwd())\n"
                               "open('made.txt', 'w').write('1')", {"x": 21})
    assert ran["stdout"].startswith("42 ") and not ran["error"]
    assert os.getcwd() == here and not (Path(here) / "made.txt").exists()
    assert "onset-model-check-" in ran["stdout"]
    failed = modelcheck.run_draft("print(missing_name)", {})
    assert failed["error"].startswith("NameError")
    drew = modelcheck.run_draft("plt.figure(); plt.plot([1, 2])",
                                {"plt": __import__("matplotlib.pyplot").pyplot})
    assert drew["figures"] == 1
    slow = modelcheck.run_draft("while True:\n    pass", {}, seconds=1)
    assert slow["error"] == "stopped after 1 s"


# -- the whole check, against a stand-in model -------------------------------------------
ORACLE = {
    "Print how many events were accepted.":
        "print(sum(1 for e in events if e.accepted))",
    "Print the name of the channel with the highest rate_per_min in findings.":
        "print(findings.sort_values('rate_per_min', ascending=False).iloc[0]['channel'])",
    # Wrong on purpose: one too many.
    "Print how many channels the recording has.": "print(len(channels) + 1)",
    "Print the sampling rate in Hz.": "print(sfreq)",
    "Print how many accepted events each detector found.":
        "from collections import Counter\n"
        "print(Counter(e.detector for e in events if e.accepted))",
    # Raises: the commonest way a small model's draft fails.
    "Print the mean duration of the accepted ripple events in milliseconds.":
        "print(np.mean(durations_ms))",
    "Plot the first two seconds of the busiest channel's signal in microvolts.":
        "i = channels.index(findings.iloc[0]['channel'])\n"
        "plt.figure(); plt.plot(times[:4000], signal[i, :4000] * 1e6)",
    # Flagged: never run.
    "Plot the power spectrum of the busiest channel from 1 to 500 Hz.":
        "import os\nos.remove('x')",
    "Compute each channel's mean power between 80 and 250 Hz and print its Spearman "
    "correlation with rate_per_min.":
        "from scipy.signal import welch\nfrom scipy.stats import spearmanr\n"
        "f, p = welch(signal, fs=sfreq, nperseg=int(sfreq))\n"
        "power = pd.Series(p[:, (f >= 80) & (f <= 250)].mean(axis=1), index=channels)\n"
        "rate = findings.set_index('channel')['rate_per_min'].reindex(channels)\n"
        "print(round(spearmanr(power, rate).statistic, 3))",
}


class _Oracle:
    name = "oracle"
    is_language_model = True

    def describe(self):
        return "a stand-in model that writes the drafts above"

    def abort(self):
        pass

    def chat(self, messages, tools):
        request = messages[-1]["content"].split("REQUEST\n", 1)[1].strip()
        return types.SimpleNamespace(content=f"```python\n{ORACLE[request]}\n```", tool_calls=[])


@pytest.fixture
def synthetic(recording, monkeypatch, tmp_path):
    from onset_review.session import ReviewRequest, session_from_recording

    session = session_from_recording(
        recording, ReviewRequest(t_start=0.0, t_stop=float(recording.duration),
                                 detectors=("rms", "line_length")))
    monkeypatch.setattr(modelcheck, "_session", lambda log: (session, "synthetic, test"))
    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    return session


def test_every_draft_is_run_and_graded_against_the_right_answer(synthetic, tmp_path,
                                                                 monkeypatch, capsys):
    from onset_agent import backends

    monkeypatch.setattr(backends, "make_backend", lambda *a, **k: _Oracle())
    out = tmp_path / "check.json"
    assert modelcheck.main(["--only", "code", "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    by = {row["key"]: row for row in report["code"]}
    assert by["accepted"]["ok"] and by["leader"]["ok"] and by["sfreq"]["ok"]
    assert by["per_detector"]["ok"] and by["plot"]["ok"] and by["correlation"]["ok"]
    assert not by["channels"]["ok"] and by["channels"]["note"].startswith("expected ")
    assert by["duration"]["run_error"].startswith("NameError")
    assert "durations_ms" in by["duration"]["unknown"]
    assert not by["spectrum"].get("ran") and by["spectrum"]["note"].startswith("flagged")
    summary = report["summary"]
    assert (summary["drafts"], summary["drafts_correct"], summary["drafts_flagged"]) == (9, 6, 1)
    assert summary["drafts_with_unknown_names"] == 1
    assert "| channels | ✓ | ✓ | ✗ |" in out.with_suffix(".md").read_text()
    assert "Write code: 6 of 9 right" in capsys.readouterr().out


def test_the_assistant_questions_are_graded_without_a_model(synthetic, tmp_path):
    out = tmp_path / "assistant.json"
    assert modelcheck.main(["--backend", "scripted", "--only", "assistant",
                            "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert [q["key"] for q in report["questions"]] == [k for k, _, _ in modelcheck.QUESTIONS]
    overstep = next(q for q in report["questions"] if q["key"] == "overstep")
    assert overstep["expect"] == "refused" and overstep["ok"], "a clinical decision is refused"
    assert report["summary"]["questions_as_expected"] == len(modelcheck.QUESTIONS)


# -- in the window ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    yield widgets.QApplication.instance() or widgets.QApplication([])


def _help_entries(host) -> list[str]:
    for action in host.menuBar().actions():
        if action.text().replace("&", "") == "Help":
            return [a.text() for a in action.menu().actions() if a.text()]
    return []


def test_every_window_offers_the_check_and_the_report(qapp, tmp_path, monkeypatch):
    import pandas as pd

    from onset_review import window as window_module

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    start = window_module.decorate_start(cached=lambda: pd.DataFrame())
    try:
        entries = _help_entries(start)
        assert "Test the local model…" in entries and "Report a problem…" in entries
    finally:
        start.close()


def test_the_report_dialog_shows_and_saves_the_report(qapp, tmp_path, monkeypatch):
    from onset_review.helpdialogs import ReportDialog

    monkeypatch.setenv("ONSET_REVIEW_LOG_DIR", str(tmp_path / "logs"))
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "onset-review.log").write_text("2026-10-08 INFO onset_review: hello\n")
    dialog = ReportDialog()
    try:
        assert "hello" in dialog.text.toPlainText() and "onset-review:" in dialog.text.toPlainText()
        saved = dialog.save(tmp_path / "sent.zip")
        assert saved == tmp_path / "sent.zip"
        assert "report.txt" in zipfile.ZipFile(saved).namelist()
        assert "Send this file" in dialog.status.text()
    finally:
        dialog.close()


def test_the_model_check_runs_from_the_window_in_a_process_of_its_own(qapp, tmp_path,
                                                                       monkeypatch):
    from qtpy.QtCore import QThread

    from onset_review.helpdialogs import ModelCheckDialog

    monkeypatch.setenv("ONSET_REVIEW_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("ONSET_ASSISTANT_NO_PROBE", "1")
    dialog = ModelCheckDialog(arguments=["--backend", "scripted", "--only", "assistant"])
    try:
        assert str(dialog.out).startswith(str(tmp_path / "config"))
        dialog.start()
        assert dialog.running()
        for _ in range(3000):
            qapp.processEvents()
            if not dialog.running():
                break
            QThread.msleep(100)
        qapp.processEvents()
        assert not dialog.running(), "the check finished"
        assert dialog.status.text().startswith("Done."), dialog.output.toPlainText()[-800:]
        assert "Assistant: 8 of 8 as expected" in dialog.output.toPlainText()
        assert dialog.out.with_suffix(".md").exists() and dialog.open_button.isEnabled()
    finally:
        dialog.close()
