"""The event detail panel: three stacked axes and the numbers under them.

Drawing only. Everything shown here is computed by `onset_review.detail`,
which imports no Qt and is where the choices -- the window length, the
wavelet lengths, what the colours are relative to -- are argued and tested.

The layout is fixed and vertical because the three pictures only work read
together, on a shared time axis: the spike in the wideband trace, the burst in
the filtered one and the column of energy in the time-frequency plot have to
line up under the reviewer's eye without them doing the alignment themselves.
A tabbed version of this panel would be three views that cannot be compared,
which is the same as no view at all.
"""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from onset_review import detail, theme
from onset_review.theme import card

__all__ = ["EventDetailPanel"]

#: Colour range of the time-frequency plot, in dB above the surrounding
#: background. Fixed rather than auto-scaled: an auto-scaled colour map makes
#: every event look equally striking, including the ones that are not, and the
#: reviewer comparing two events needs the colours to mean the same thing in
#: both. 0 dB is "no more energy here than either side of it".
DB_RANGE = (0.0, 20.0)


class EventDetailPanel(QWidget):
    """One event, wideband and filtered and in time-frequency.

    Driven by whatever the reviewer selects in the events list. It holds no
    selection of its own, because two places in the window disagreeing about
    which event is under discussion is worse than one place not having an
    opinion.
    """

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self._snapshot = None

        tokens = theme.current()
        # Short by default for the same reason the 3D view is: `figsize` sets
        # the canvas's size *hint*, which is the height the scroll area lays
        # the panel out at, and a tall hint pushes the caption -- the part that
        # says how to read the pictures -- below the fold in a docked column.
        # The figure takes the whole dock when there is room.
        self.figure = Figure(figsize=(4.4, 2.1), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumWidth(240)
        self.canvas.setMinimumHeight(150)
        # Three rows, the time-frequency plot given the most height because it
        # is the one carrying the argument; shared x so the three line up.
        self.axes = self.figure.subplots(
            3, 1, sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.5], "hspace": 0.12,
                         "left": 0.14, "right": 0.99, "top": 0.97,
                         "bottom": 0.17})
        for axis in self.axes:
            axis.set_facecolor(tokens.surface)
        self._mesh = None

        self.headline = QLabel()
        self.headline.setWordWrap(True)
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.headline.setStyleSheet(
            f"font-size:9pt;padding:1px 4px;color:{tokens.text};")

        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.caption.setStyleSheet(card("info"))
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addWidget(self.canvas)
        box.addWidget(self.headline)
        box.addWidget(self.caption)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))

        self.clear("Select an event — in the list, the trend or the trace — "
                   "to see it close up.")

    # -- what to draw ------------------------------------------------------

    @property
    def _unit(self) -> str:
        """The unit this window's amplitudes are in: µV, or fT/cm or fT for MEG."""
        return getattr(self._session, "unit", "µV")
    def show_key(self, key: str) -> bool:
        """Draw the event a verdict key names. False when there is no such event."""
        event = detail.find_event(self._session, str(key))
        if event is None:
            self.clear("That event is not in this analysis.")
            return False
        self.show_snapshot(detail.snapshot(self._session, event))
        return True

    def show_snapshot(self, snap) -> None:
        self._snapshot = snap
        if not snap.available:
            self.clear("The signal around this event is not in the loaded "
                       "window, so there is nothing to draw.")
            return
        self._draw(snap)
        self.canvas.setVisible(True)
        self.headline.setText(self._headline(snap))
        self.caption.setText(self._caption(snap))
        self.caption.setStyleSheet(card("warn" if snap.notes else "info"))

    def clear(self, message: str) -> None:
        self._snapshot = None
        for axis in self.axes:
            axis.clear()
            axis.set_facecolor(theme.current().surface)
            axis.set_xticks([])
            axis.set_yticks([])
        self._mesh = None
        self.canvas.draw_idle()
        # Three empty frames above a sentence at the bottom of a tall column
        # read as a panel that failed to draw. With nothing to draw, the
        # sentence is the panel.
        self.canvas.setVisible(False)
        self.headline.setText("")
        self.caption.setText(message)
        self.caption.setStyleSheet(card("info"))

    # -- drawing -----------------------------------------------------------
    def _draw(self, snap) -> None:
        tokens = theme.current()
        wide, band, spectrum = self.axes
        for axis in self.axes:
            axis.clear()
            axis.set_facecolor(tokens.surface)
            for spine in ("top", "right"):
                axis.spines[spine].set_visible(False)
            axis.tick_params(labelsize=7, colors=tokens.text_muted)
            for spine in axis.spines.values():
                spine.set_color(tokens.separator)

        milliseconds = snap.times * 1000.0
        on, off = snap.onset * 1000.0, snap.offset * 1000.0

        wide.plot(milliseconds, snap.wideband, lw=0.7, color=tokens.text)
        wide.set_ylabel(f"wideband\n{self._unit}", fontsize=7, color=tokens.text_muted)
        band.plot(milliseconds, snap.band_passed, lw=0.8, color=tokens.accent)
        band.set_ylabel(f"{snap.band[0]:g}–{snap.band[1]:g} Hz\n{self._unit}",
                        fontsize=7, color=tokens.text_muted)

        for axis in (wide, band):
            # The event's own span, on both traces. Shaded rather than drawn
            # as two lines, because what is being judged is an interval.
            axis.axvspan(on, off, color=tokens.accent, alpha=0.13, lw=0)

        if snap.power_db.size:
            self._mesh = spectrum.pcolormesh(
                milliseconds, snap.freqs, snap.power_db, shading="nearest",
                cmap="magma", vmin=DB_RANGE[0], vmax=DB_RANGE[1])
            spectrum.set_yscale("log")
            spectrum.set_ylabel("Hz", fontsize=7, color=tokens.text_muted)
            # The detector's band marked on the frequency axis, so "inside the
            # band" is something the reviewer can see rather than estimate.
            for edge in snap.band:
                if snap.freqs.size and snap.freqs[0] <= edge <= snap.freqs[-1]:
                    spectrum.axhline(edge, color="white", lw=0.6, alpha=0.45,
                                     ls="--")
            spectrum.axvline(on, color="white", lw=0.6, alpha=0.5)
            spectrum.axvline(off, color="white", lw=0.6, alpha=0.5)
        else:
            spectrum.text(0.5, 0.5, "no time-frequency plot for this event",
                          ha="center", va="center", fontsize=7,
                          color=tokens.text_muted, transform=spectrum.transAxes)
            spectrum.set_yticks([])
        spectrum.set_xlabel("ms from the start of the event", fontsize=7,
                            color=tokens.text_muted)
        self.canvas.draw_idle()

    # -- words -------------------------------------------------------------
    def _headline(self, snap) -> str:
        m = snap.measurements
        bits = [f"<b>{snap.channel}</b> — {snap.kind}",
                f"{m['duration_ms']:.0f} ms", f"{m['amplitude_uv']:.0f} {self._unit}"]
        if np.isfinite(m["frequency_hz"]):
            bits.append(f"{m['frequency_hz']:.0f} Hz")
        if np.isfinite(m["prominence_db"]):
            bits.append(f"{m['prominence_db']:.1f} dB above background")
        if np.isfinite(m["n_cycles"]):
            bits.append(f"{m['n_cycles']:.1f} cycles")
        if m["with_spike"]:
            bits.append("on a discharge")
        if not m["accepted"]:
            bits.append(f"<b>rejected</b>: {m['reject_reason'] or 'artifact'}")
        return " · ".join(bits)

    def _caption(self, snap) -> str:
        """What the pictures mean, in the words the judgement is made in.

        Kept on screen rather than in the documentation because this is the
        one view whose whole purpose is to be read correctly by someone who
        has not read the documentation.
        """
        text = (
            "Top: the signal as the detector saw it — high-passed, notched and "
            "re-referenced, not unprocessed. Middle: the same signal through "
            "the detector's band. Bottom: energy against the 0.4 s either side "
            "of the event, white lines at the band edges.<br>"
            "<b>A real oscillation is an island</b> — energy confined in "
            "frequency and lasting several cycles. <b>Filter ringing is a "
            "column</b>: a sharp transient is broadband, so its energy runs "
            "top to bottom at one instant and the middle trace is an artifact "
            "of the filter, not a finding.")
        if snap.notes:
            text += "<br><br>" + " ".join(snap.notes)
        return text
