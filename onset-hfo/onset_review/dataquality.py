"""The Data quality panel: which contacts and which seconds were analysed.

Every number on this screen is computed in :mod:`onset_hfo.quality`, which
imports no Qt and is where the claims are tested. This file renders them and
gives the reviewer the two decisions that are theirs.

The panel exists because of an asymmetry. A reviewer reading the Findings table
can see that a channel is busy; they cannot see *why*. A contact with a noisy
amplifier tops the ranking with a tight confidence interval and agrees with
itself across every other panel, because its noise is genuinely oscillatory and
every stage downstream is working correctly on it. Nothing else in this window
would tell them. So the measurement behind each verdict is shown as a number,
not just the verdict: a reviewer who disagrees can see what the software was
looking at.

There are two kinds of verdict here and the difference is the point. A contact
is **set aside** only for a fault no physiology produces -- flat, clipped at
the rail, swamped by mains, or with too little surviving time to rate. It is
**flagged** when a measurement is unusual in a way that is as consistent with
the finding as with a fault, and then it is analysed normally and the reviewer
is asked to look at it.

That line was drawn by a recording rather than by taste. The band-power check
flagged `TR1-TR2` and `TR2-TR3` of sub-13; the archive's own annotators had
marked 91, 102 and 164 ripples on those three contacts. A contact full of real
ripples has high band power *because the ripples are in the band*, and nothing
here can tell that from a noisy amplifier. Setting them aside would have
deleted the finding and reported a cleaner table.

Two controls, and no more. **Check data quality** turns the stage off, which
reproduces the numbers this project measured before it existed. **Reinstate**
puts one set-aside contact back after the reviewer has looked at it. There is
no control for excluding a contact the checks passed, deliberately: that is
`PreprocessConfig.exclude` on the Preprocessing panel, where it is recorded as
the reviewer's own choice rather than as an override of a verdict nobody made.

Like the preprocessing panel, this one decides nothing and re-runs nothing. It
emits the choice; `onset_review.app` owns the reload, because setting a channel
aside changes every number in every other panel and rebuilding them all is the
only honest response.
"""

from __future__ import annotations

import pandas as pd
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from onset_hfo.quality import REASONS, SET_ASIDE, quality_summary
from onset_review import compact
from onset_review.theme import SPACING, card, current, muted, scrolled, section_label

__all__ = ["QualityPanel", "COLUMNS", "REASONS", "SET_ASIDE"]

#: What is shown, in the order a reviewer reads it: the verdict, then the
#: measurement that produced it, then what it cost in analysed time.
COLUMNS: list[tuple[str, str, str]] = [
    ("channel", "Channel", ""),
    ("verdict", "Verdict", "Whether this contact was analysed, and why not"),
    ("amplitude_uv", "Amplitude µV",
     "Robust amplitude (1.4826 × MAD). Near zero is a dead contact"),
    ("hf_ratio_sd", "HF outlier",
     "How far this contact's in-band share of power sits above the rest of "
     "the montage, in robust SDs. High on a noisy amplifier — and high on a "
     "contact full of real ripples, because the ripples are in the band"),
    ("burstiness", "Burstiness",
     "How that band power arrives: the 99th percentile of the ripple-band "
     "envelope over its 10th. 6.6 is the value for pure noise, whatever the "
     "amplitude — a contact sitting there has no events in it. Well above "
     "means discrete bursts, which is what ripples look like (and what a "
     "repeating artifact looks like)"),
    ("line_fraction", "Mains share",
     "Share of power at the mains frequency and its harmonics. The 4th and "
     "5th harmonics of 50 Hz land at 200 and 250 Hz, inside the ripple band"),
    ("clipped_fraction", "Clipped",
     "Share of the window pinned at the amplifier's limit"),
    ("bad_segment_fraction", "Seconds lost",
     "Share of this contact's seconds rejected as discontinuous or flat"),
    ("analysed_s", "Analysed s",
     "The seconds this contact's rate was actually divided by. Blank means "
     "it was set aside and no rate is claimed for it"),
]


class QualityPanel(QWidget):
    """The verdicts, the measurements behind them, and the two overrides."""

    #: Emitted on Apply, with (check_quality, keep_channels).
    applied = Signal(bool, tuple)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.tokens = current()
        self._session = session
        request = session.request
        self._started_on = (bool(getattr(request, "check_quality", True)),
                            tuple(getattr(request, "keep_channels", ())))

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(SPACING + 4, SPACING, SPACING + 4, SPACING)
        column.setSpacing(SPACING // 2)

        self.enabled = QCheckBox("Check data quality")
        self.enabled.setChecked(self._started_on[0])
        self.enabled.setToolTip(
            "Set aside contacts and seconds that are not fit to analyse, and "
            "divide each contact's rate by the time that survived. Turning it "
            "off reproduces the numbers this project measured before the "
            "stage existed — including, on a faulty contact, a confident rate "
            "for an amplifier.")
        column.addWidget(self.enabled)

        self.summary = QLabel(self._summary_text())
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet(self._summary_style())
        column.addWidget(self.summary)

        column.addWidget(section_label("Contacts", self.tokens))
        # One chip per contact, the ones worth a look first. A reader takes
        # forty verdicts in at a glance here where the table below says the
        # same thing in nine columns; a click on a chip selects its row.
        self.strip = QLabel()
        self.strip.setObjectName("onset_quality_strip")
        self.strip.setWordWrap(True)
        self.strip.setTextFormat(Qt.RichText)
        self.strip.setOpenExternalLinks(False)
        self.strip.linkActivated.connect(self._chip_clicked)
        column.addWidget(self.strip)
        self.legend = muted("", self.tokens)
        column.addWidget(self.legend)

        self.explain = QCheckBox("Why nothing is repaired, and what a flag means")
        self.explain.setObjectName("onset_quality_explain")
        self.explain.setChecked(False)
        column.addWidget(self.explain)
        self._explanations = []
        self._explanations.append(muted(
            "Nothing is repaired. A standard EEG cleaner interpolates a bad "
            "channel from its neighbours; here the whole output is a "
            "per-contact rate, so an interpolated contact's rate would be "
            "borrowed from the ones beside it and read as a finding about it.",
            self.tokens))
        self._explanations.append(muted(
            "And little is removed. A contact is only set aside for a fault "
            "no physiology produces. Where a measurement is odd in a way that "
            "could equally be the finding — far more band power than its "
            "neighbours, say — it is analysed and flagged instead, and the "
            "judgement is yours: open it on the trace.", self.tokens))
        self._explanations.append(muted(
            "Burstiness says which way a band-power flag leans. 6.6 is the "
            "value a contact carrying no events at all takes, whatever its "
            "amplitude — it falls out of the algebra, not out of this "
            "cohort. Far above it means the energy arrives in bursts.",
            self.tokens))
        for label in self._explanations:
            label.setVisible(False)
            column.addWidget(label)
        self.explain.toggled.connect(
            lambda on: [label.setVisible(bool(on)) for label in self._explanations])

        self.show_table = QCheckBox("Show the measurements")
        self.show_table.setObjectName("onset_quality_table")
        self.show_table.setToolTip("Amplitude, band-power outlier, burstiness, mains "
                                   "share, clipping and the seconds analysed, per contact")
        column.addWidget(self.show_table)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels([label for _, label, _ in COLUMNS])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        for index, (_, _, tip) in enumerate(COLUMNS):
            if tip:
                self.table.horizontalHeaderItem(index).setToolTip(tip)
        self.table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.table.setMinimumHeight(220)
        column.addWidget(self.table, 1)
        # The table opens shown only when there is something in it to act on:
        # a reviewer who opens this panel because a contact was set aside
        # should not have to ask for the row.
        self._compact = True
        self.show_table.toggled.connect(lambda on: self.set_compact(not on))

        self.reinstate = QPushButton("Reinstate this contact")
        self.reinstate.setEnabled(False)
        self.reinstate.setToolTip(
            "Put a contact the checks set aside back into the analysis, "
            "having looked at it. To remove one they passed, use Exclude on "
            "the Preprocessing panel — there it is recorded as your choice "
            "rather than as an override.")
        self.reinstate.clicked.connect(self._reinstate)

        self.apply = QPushButton("Apply and re-analyse")
        self.apply.setProperty("primary", True)
        self.apply.setEnabled(False)
        self.apply.setToolTip(
            "Re-run this window. Setting a contact aside or putting one back "
            "changes every rate, interval and rank, so all of them are "
            "rebuilt rather than some refreshed.")
        self.apply.clicked.connect(self._emit)

        self.reset = QPushButton("Reset")
        self.reset.setToolTip("Back to the verdicts as the software made them.")
        self.reset.clicked.connect(self._reset)

        buttons = QHBoxLayout()
        buttons.addWidget(self.reinstate)
        buttons.addStretch(1)
        buttons.addWidget(self.reset)
        buttons.addWidget(self.apply)
        column.addLayout(buttons)
        # With the table folded, the spare height goes here rather than into
        # gaps between the lines above: an expanding spacer takes it before
        # any label does, and the table, when shown, still takes it all.
        column.addStretch(0)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(body))

        self._kept = set(self._started_on[1])
        self._fill()
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.enabled.toggled.connect(self._changed)
        self.set_compact(not self._something_to_act_on())

    # -- compact or not --------------------------------------------------------
    @property
    def compact(self) -> bool:
        return self._compact

    def set_compact(self, on: bool) -> None:
        on = bool(on)
        self._compact = on
        self.show_table.blockSignals(True)
        self.show_table.setChecked(not on)
        self.show_table.blockSignals(False)
        self.table.setVisible(not on)

    def _something_to_act_on(self) -> bool:
        quality = self._session.quality
        if quality is None or quality.empty:
            return False
        return bool((~quality["good"]).any() or quality["flagged"].any())

    def chips(self) -> list[dict]:
        return compact.quality_chips(self._session.quality, kept=tuple(self._kept))

    def _draw_strip(self) -> None:
        chips = self.chips()
        if not chips:
            self.strip.setText("")
            self.legend.setText("")
            return
        # Colour is spent on the exceptions. A contact that passed is a
        # neutral chip; forty green chips said nothing forty times and made
        # the one red one harder to find.
        colours = {"bad": (self.tokens.bad, self.tokens.accent_text),
                   "warn": (self.tokens.warn, self.tokens.accent_text),
                   "accent": (self.tokens.accent, self.tokens.accent_text),
                   "good": (self.tokens.surface_alt, self.tokens.text)}
        kinds = {kind: (label, token) for kind, label, token in compact.QUALITY_KINDS}
        parts = []
        for chip in chips:
            label, token = kinds[chip["kind"]]
            reason = REASONS.get(chip["reason"], chip["reason"]) if chip["reason"] else label
            back, fore = colours[token]
            parts.append(
                f"<a href='{chip['channel']}' title='{label}: {reason}' "
                f"style='text-decoration:none;color:{self.tokens.text};'>"
                f"<span style='background:{back};color:{fore};"
                f"border-radius:3px;padding:1px 5px;font-size:8pt;'>&nbsp;{chip['channel']}"
                f"&nbsp;</span></a>")
        self.strip.setText(" ".join(parts))
        counts = compact.counts_by_kind(chips)
        said = [f"{counts[kind]} {label}" for kind, label, _t in compact.QUALITY_KINDS
                if counts.get(kind)]
        self.legend.setText("Each chip is one contact, the ones to look at first: "
                            + ", ".join(said) + ". Click one to select it.")

    def _chip_clicked(self, href: str) -> None:
        channel = str(href)
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is not None and item.text() == channel:
                self.table.selectRow(row)
                self.table.scrollToItem(item)
                return

    # -- what the table says -----------------------------------------------
    def rows(self) -> pd.DataFrame:
        """The quality table, with the two columns this panel adds."""
        quality = self._session.quality
        if quality is None or quality.empty:
            return pd.DataFrame(columns=[name for name, _, _ in COLUMNS])
        out = quality.copy()
        seconds = self._session.clean_seconds or {}
        out["analysed_s"] = [float(seconds.get(str(c), float("nan")))
                             for c in out["channel"]]
        out["verdict"] = [_verdict(row, str(row.channel) in self._kept)
                          for row in out.itertuples()]
        # Set aside first, then flagged, then the rest: a reviewer opens this
        # panel because something is wrong, not to read forty rows that say
        # "analysed".
        out["_order"] = [0 if not row.good else (1 if row.flagged else 2)
                         for row in out.itertuples()]
        return (out.sort_values(["_order", "channel"])
                .drop(columns="_order").reset_index(drop=True))

    def _fill(self) -> None:
        frame = self.rows()
        self.table.setRowCount(len(frame))
        for row, record in enumerate(frame.to_dict("records")):
            for index, (name, _, _) in enumerate(COLUMNS):
                item = QTableWidgetItem(_format(name, record.get(name)))
                if name in ("channel", "verdict"):
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                    # The verdict is a sentence and the column elides it; the
                    # whole of it is the part worth reading, so it is also the
                    # tooltip rather than only a truncated phrase.
                    item.setToolTip(str(record.get("verdict", "")))
                else:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tint = _tint(record, str(record["channel"]) in self._kept)
                if tint:
                    item.setForeground(_brush(self.tokens, tint))
                self.table.setItem(row, index, item)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.summary.setText(self._summary_text())
        self.summary.setStyleSheet(self._summary_style())
        self._draw_strip()
        # Refill the button state too. Rewriting the rows does not change the
        # selection when the row count is the same, so `itemSelectionChanged`
        # does not fire -- and Reinstate would stay lit on a contact that has
        # just been reinstated.
        self._selection_changed()

    def _summary_text(self) -> str:
        session = self._session
        if session.quality is None or session.quality.empty:
            return ("The quality checks did not run on this window, so every "
                    "contact was analysed and every rate was divided by the "
                    "whole window — including any contact that was not fit to "
                    "be in the ranking.")
        return quality_summary(session.quality, session.segments,
                               kept=tuple(getattr(self, "_kept", ()) or ()))

    def _summary_style(self) -> str:
        """A box only when there is something in it to act on. "All passed"
        is one quiet line; a box the size of a paragraph saying nothing is
        wrong reads as if something were."""
        kind = self._summary_kind()
        if kind == "info":
            return (f"color:{self.tokens.good};font-size:9pt;"
                    f"padding:2px 0 {SPACING}px 0;")
        return card(kind, self.tokens)

    def _summary_kind(self) -> str:
        session = self._session
        quality = session.quality
        if quality is None or quality.empty:
            return "warn"
        if not quality["good"].all():
            return "bad"
        return "warn" if quality.get("flagged", pd.Series(dtype=bool)).any() else "info"

    # -- the two overrides -------------------------------------------------
    def selected_channel(self) -> str | None:
        rows = {item.row() for item in self.table.selectedItems()}
        if not rows:
            return None
        item = self.table.item(next(iter(rows)), 0)
        return item.text() if item is not None else None

    def _selection_changed(self) -> None:
        """Only a *set-aside* contact can be reinstated; a flagged one is in.

        The button staying grey on a flagged row is the panel saying so: there
        is nothing to put back, and the thing to do with a flag is look at the
        trace.
        """
        channel = self.selected_channel()
        frame = self.rows()
        set_aside = (channel is not None and not frame.empty
                     and not bool(frame.loc[frame["channel"] == channel,
                                            "good"].iloc[0]))
        self.reinstate.setEnabled(bool(set_aside) and channel not in self._kept)

    def _reinstate(self) -> None:
        channel = self.selected_channel()
        if channel:
            self._kept.add(channel)
            self._changed()      # which refills the table and the buttons

    def _reset(self) -> None:
        self._kept = set(self._started_on[1])
        self.enabled.setChecked(self._started_on[0])
        self._fill()
        self._changed()

    def _changed(self) -> None:
        self.apply.setEnabled(self.choice() != self._started_on)
        self._fill()

    def choice(self) -> tuple[bool, tuple[str, ...]]:
        return (bool(self.enabled.isChecked()), tuple(sorted(self._kept)))

    def _emit(self) -> None:
        enabled, kept = self.choice()
        self.applied.emit(enabled, kept)


def _format(name: str, value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if name in ("channel", "verdict"):
        return str(value)
    if name in ("line_fraction", "clipped_fraction", "bad_segment_fraction"):
        return f"{float(value):.1%}"
    if name == "burstiness":
        return f"{float(value):.1f}"
    if name == "analysed_s":
        return "—" if float(value) <= 0 else f"{float(value):.0f}"
    return f"{float(value):.2f}"


def _verdict(row, reinstated: bool) -> str:
    sentence = REASONS.get(row.reason, row.reason)
    if not row.reason:
        return "analysed"
    if row.flagged:
        return f"analysed, flagged — {sentence}"
    if reinstated:
        return f"reinstated by you — {sentence}"
    return f"set aside — {sentence}"


def _tint(record, reinstated: bool) -> str:
    """Which palette token a row is drawn in, or "" for the ordinary ones."""
    if not record.get("good", True):
        return "text_muted" if reinstated else "bad"
    if record.get("flagged", False):
        return "warn"
    return ""


def _brush(tokens, token: str):
    from qtpy.QtGui import QBrush, QColor

    return QBrush(QColor(getattr(tokens, token)))
