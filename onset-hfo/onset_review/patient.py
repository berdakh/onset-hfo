"""The Patient panel: the archive's record, drawn.

Everything this panel is *allowed to say* lives in `onset_review.record`, which
imports no Qt and is where the claims are tested. This file renders them.

The outcome is shown last and behind a deliberate click, because it is the one
field that can turn a review into a self-fulfilling one: a reviewer who knows
the patient became seizure-free reads the same rate table differently.
"""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from onset_review.record import CHART_FIELDS, ILAE_CLASSES, patient_record
from onset_review.theme import SPACING, card, current, muted, scrolled, section_label

__all__ = ["PatientPanel", "patient_record", "CHART_FIELDS", "ILAE_CLASSES"]


class PatientPanel(QWidget):
    """Who this recording belongs to, as far as the archive says."""

    def __init__(self, session, data_dir: Path | None = None, parent=None):
        super().__init__(parent)
        self.palette_tokens = current()
        self._session = session
        self.record = patient_record(str(session.request.subject), data_dir)
        #: An imported file has no accession and no cohort row, so the two
        #: paragraphs below that describe the *archive's* record would be
        #: answering a question nobody asked.
        self.imported = bool(getattr(session.request, "imported", False))

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(SPACING + 4, SPACING, SPACING + 4, SPACING)
        column.setSpacing(2)

        column.addWidget(_title(session.request.subject))
        where = (f"imported from {session.request.path.name}"
                 if self.imported else session.request.dataset)
        column.addWidget(muted(
            f"{where} · run {session.request.run} · "
            f"{session.request.t_start:g}–{session.request.t_stop:g} s of this "
            f"recording", self.palette_tokens))

        if not self.record.get("available"):
            column.addWidget(section_label("Record", self.palette_tokens))
            unavailable = QLabel(
                "This window was imported from a file, so there is no "
                "participant record, no resected zone and no expert marking to "
                "compare against. Nothing in this panel is missing by accident."
                if self.imported else
                "No participant record is committed for this subject. "
                "`data/outcome/participants.csv` carries the cohort this "
                "software was evaluated on; a recording from elsewhere will "
                "not appear in it.")
            unavailable.setWordWrap(True)
            unavailable.setStyleSheet(card("warn", self.palette_tokens))
            column.addWidget(unavailable)
        else:
            column.addWidget(section_label("Epilepsy", self.palette_tokens))
            column.addWidget(_facts(self._epilepsy_rows(), self.palette_tokens))

            column.addWidget(section_label("This implantation",
                                           self.palette_tokens))
            column.addWidget(_facts(self._implant_rows(), self.palette_tokens))
            coverage = self._coverage_warning()
            if coverage:
                note = QLabel(coverage)
                note.setWordWrap(True)
                note.setStyleSheet(card("warn", self.palette_tokens))
                column.addWidget(note)

            column.addWidget(section_label("Surgical outcome",
                                           self.palette_tokens))
            self.outcome_shown = QCheckBox("Show what happened after surgery")
            self.outcome_shown.setToolTip(
                "Hidden by default. Knowing the patient became seizure-free "
                "changes how the same rate table reads, and this software is "
                "for forming an impression from the signal.")
            self.outcome = QLabel()
            self.outcome.setWordWrap(True)
            self.outcome.setStyleSheet(card("info", self.palette_tokens))
            self.outcome.setVisible(False)
            self.outcome_shown.toggled.connect(self._reveal)
            column.addWidget(self.outcome_shown)
            column.addWidget(self.outcome)

        column.addWidget(section_label("Clinical record", self.palette_tokens))
        empty = QLabel(
            "<b>Nothing connected to this software carries the following.</b> "
            "They are listed so you can see where a site's own record would "
            "appear, and they are deliberately blank rather than filled with "
            "an example — an invented history in a clinical tool is "
            "indistinguishable from a real one.")
        empty.setWordWrap(True)
        empty.setStyleSheet(card("warn", self.palette_tokens))
        column.addWidget(empty)
        column.addWidget(_facts([(name, "— no source connected", why)
                                 for name, why in CHART_FIELDS],
                                self.palette_tokens))

        if not self.imported:
            column.addWidget(section_label("Deliberately not published",
                                           self.palette_tokens))
            column.addWidget(_withheld(self.palette_tokens))
        column.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(body))

    # -- the rows ----------------------------------------------------------
    def _epilepsy_rows(self):
        record = self.record
        kind = {"TLE": "Temporal lobe", "ETE": "Extratemporal"}.get(
            record.get("epilepsy"), record.get("epilepsy") or "—")
        lesion = {1: "Lesion found on imaging", 0: "No lesion found"}.get(
            record.get("lesion"), "Not recorded")
        nights = record.get("nights")
        return [
            ("Epilepsy", kind, "as classified in the archive's participants table"),
            ("Imaging", lesion,
             "whether a structural lesion was identified; the report itself is "
             "not in the archive"),
            ("Nights recorded", f"{nights}" if nights else "—",
             "this review is one window of one of them"),
        ]

    def _implant_rows(self):
        record = self.record
        total, reviewed = record.get("n_channels"), record.get("n_reviewed")
        rows = [
            ("Channels", f"{total}" if total else "—",
             "bipolar channels in the analysed run"),
            ("Reviewed by annotators", f"{reviewed}" if reviewed else "—",
             "only these can be scored; a detection elsewhere is unjudged, "
             "not wrong"),
            ("Expert HFO marks", f"{record.get('n_expert_events') or '—'}",
             "across the whole run, not just this window"),
            ("Inside the resection",
             f"{record.get('n_resected')} channels"
             if record.get("n_resected") is not None else "—",
             "both contacts removed; "
             f"{record.get('n_partial', '—')} more had one contact removed"),
        ]
        eloquent = record.get("n_eloquent")
        if eloquent:
            rows.append(("Eloquent cortex", f"{eloquent} channels",
                         "excluded from resection because stimulation evoked a "
                         "motor or language response"))
        return rows

    def _coverage_warning(self) -> str:
        """The most important sentence on this panel, when it applies.

        In five of the twenty patients only a quarter of the contacts the
        surgeon removed appear in the recording at all. Their "share inside the
        resection" describes a quarter of their resection, and a reviewer who
        does not know that will read the number as though it described the
        whole.
        """
        coverage = self.record.get("rz_coverage")
        if coverage is None or coverage >= 0.999:
            return ""
        missing = self.record.get("missing") or []
        text = (f"<b>Only {coverage:.0%} of the contacts the surgeon removed "
                f"were recorded.</b> Anything this software says about "
                f"&ldquo;inside the resection&rdquo; for this patient describes "
                f"that {coverage:.0%}, not the resection.")
        if missing:
            text += (f" Not recorded: {', '.join(missing[:12])}"
                     + ("…" if len(missing) > 12 else "") + ".")
        return text

    def _reveal(self, shown: bool) -> None:
        record = self.record
        ilae = record.get("ilae")
        free = record.get("outcome") == "S"
        months = record.get("months")
        self.outcome.setText(
            f"<b>{'Seizure-free' if free else 'Seizures returned'}</b> "
            f"(ILAE class {ilae}: {ILAE_CLASSES.get(ilae, 'not recorded')})"
            + (f", at {months} months of follow-up." if months else ".")
            + "<br><br>This is the reference standard the cohort study uses, "
              "and it is the only one in epilepsy surgery that is not another "
              "opinion. It is also only twenty patients: see "
              "<code>docs/OUTCOME.md</code> before reading a single case as "
              "evidence of anything.")
        self.outcome.setVisible(shown)


def _withheld(tokens) -> QLabel:
    """Why three fields the archive publishes are absent from this repository.

    A blank beside pathology and outcome reads as an oversight unless it says
    otherwise, so it says otherwise. Only shown for an archive window: an
    imported file has no cohort for the argument to be about.
    """
    label = QLabel(
        "The archive's <code>participants.tsv</code> also carries <b>age, "
        "sex and handedness</b>. They are not committed here: three "
        "demographic fields published beside pathology, surgical extent and "
        "outcome narrow a cohort of twenty considerably, and no analysis in "
        "this project uses any of them. This is a blank by choice, not a "
        "gap in the data — see <code>data/outcome/README.md</code>.")
    label.setWordWrap(True)
    label.setStyleSheet(card("plain", tokens))
    return label


def _title(subject: str) -> QLabel:
    label = QLabel(str(subject))
    label.setStyleSheet(
        f"font-size:16pt;font-weight:600;color:{current().text};")
    return label


def _facts(rows, palette) -> QWidget:
    """A two-column list: label, value, and a line of why it matters.

    A grid rather than a table because these are not data to sort; they are a
    handful of facts to read once, and a `QTableView` makes them look like
    something to scan.
    """
    widget = QWidget()
    grid = QGridLayout(widget)
    grid.setContentsMargins(0, 2, 0, SPACING)
    grid.setHorizontalSpacing(SPACING + 4)
    grid.setVerticalSpacing(SPACING // 2)
    grid.setColumnStretch(1, 1)
    for index, (name, value, why) in enumerate(rows):
        key = QLabel(name)
        key.setStyleSheet(f"color:{palette.text_muted};font-size:9pt;")
        key.setAlignment(Qt.AlignRight | Qt.AlignTop)
        key.setMinimumWidth(130)
        body = QLabel(f"<b>{value}</b><br>"
                      f"<span style='color:{palette.text_muted};font-size:8pt;'>"
                      f"{why}</span>")
        body.setWordWrap(True)
        body.setTextFormat(Qt.RichText)
        grid.addWidget(key, index, 0)
        grid.addWidget(body, index, 1)
    return widget
