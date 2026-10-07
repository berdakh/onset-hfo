"""The Average panel: a channel's events aligned and averaged, three ways.

Drawing only; :mod:`onset_review.average` computes. The same three rows as
the single-event view -- wideband, band-passed, time-frequency -- on the same
time axis, so the mean can be read with the same eye: a spindle with an
island under it, or a transient with a column under it.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from onset_review import average, theme
from onset_review.eventview import DB_RANGE
from onset_review.theme import card

__all__ = ["AveragePanel"]


class AveragePanel(QWidget):
    """Every accepted event on one channel, averaged at its peak."""

    channelPicked = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._cache: dict[str, object] = {}
        self._current: str | None = None

        self.channel = QComboBox()
        self.channel.setObjectName("onset_average_channel")
        for name, count in average.channels_with_events(session):
            self.channel.addItem(f"{name} ({count})", name)
        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("Channel"))
        bar.addWidget(self.channel, 1)

        tokens = theme.current()
        self.figure = Figure(figsize=(4.4, 2.4), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumWidth(240)
        self.canvas.setMinimumHeight(150)
        self.axes = self.figure.subplots(
            3, 1, sharex=True,
            gridspec_kw={"height_ratios": [1.0, 1.0, 1.5], "hspace": 0.12,
                         "left": 0.14, "right": 0.99, "top": 0.97, "bottom": 0.17})
        for axis in self.axes:
            axis.set_facecolor(tokens.surface)

        self.headline = QLabel("")
        self.headline.setObjectName("onset_average_headline")
        self.headline.setWordWrap(True)
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.headline.setStyleSheet(f"font-size:9pt;padding:1px 4px;color:{tokens.text};")
        self.caption = QLabel("")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.caption.setStyleSheet(card("info"))
        self.caption.setTextInteractionFlags(Qt.TextSelectableByMouse)

        body = QWidget()
        box = QVBoxLayout(body)
        box.setContentsMargins(4, 4, 4, 4)
        box.setSpacing(4)
        box.addLayout(bar)
        box.addWidget(self.canvas, 1)
        box.addWidget(self.headline)
        box.addWidget(self.caption)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(theme.scrolled(body))

        self.channel.currentIndexChanged.connect(self._chosen)
        if self.channel.count():
            self._chosen(0)
        else:
            self._clear("No accepted events on any channel, so there is nothing to average.")

    # -- what to draw ---------------------------------------------------------
    def highlight(self, channel: str) -> None:
        """Follow the channel chosen elsewhere, when it has events to average."""
        index = self.channel.findData(str(channel))
        if index < 0 or index == self.channel.currentIndex():
            return
        self.channel.setCurrentIndex(index)

    def average(self, channel: str | None = None):
        """The computed average for `channel` (the shown one by default), kept."""
        channel = channel or self._current
        if channel is None:
            return None
        if channel not in self._cache:
            self._cache[channel] = average.compute(self._session, channel)
        return self._cache[channel]

    def _chosen(self, index: int) -> None:
        channel = self.channel.itemData(index)
        if not channel:
            return
        self._current = str(channel)
        avg = self.average(self._current)
        if not avg.available:
            self._clear(" ".join(avg.notes) or "Nothing to average on this channel.")
            return
        self._draw(avg)
        self.canvas.setVisible(True)
        self.headline.setText(average.describe(avg))
        self.caption.setText(self._caption(avg))
        self.caption.setStyleSheet(card("warn" if avg.reading == "column" else "info"))
        self.channelPicked.emit(self._current)

    def _clear(self, message: str) -> None:
        for axis in self.axes:
            axis.clear()
            axis.set_xticks([])
            axis.set_yticks([])
        self.canvas.draw_idle()
        self.canvas.setVisible(False)
        self.headline.setText("")
        self.caption.setText(message)
        self.caption.setStyleSheet(card("info"))

    # -- drawing --------------------------------------------------------------
    def _draw(self, avg) -> None:
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
        ms = avg.times * 1000.0
        wide.fill_between(ms, avg.mean_wideband - avg.sd_wideband,
                          avg.mean_wideband + avg.sd_wideband,
                          color=tokens.text_muted, alpha=0.18, lw=0)
        wide.plot(ms, avg.mean_wideband, lw=0.9, color=tokens.text)
        wide.set_ylabel("wideband\nµV", fontsize=7, color=tokens.text_muted)
        band.fill_between(ms, avg.mean_band - avg.sd_band, avg.mean_band + avg.sd_band,
                          color=tokens.accent, alpha=0.18, lw=0)
        band.plot(ms, avg.mean_band, lw=0.9, color=tokens.accent)
        band.set_ylabel(f"{avg.band[0]:g}–{avg.band[1]:g} Hz\nµV", fontsize=7,
                        color=tokens.text_muted)
        half = avg.mean_duration_ms / 2.0 if np.isfinite(avg.mean_duration_ms) else 0.0
        for axis in (wide, band):
            axis.axvspan(-half, half, color=tokens.accent, alpha=0.10, lw=0)
            axis.axvline(0.0, color=tokens.text_muted, lw=0.5, alpha=0.6)
        if avg.mean_power_db.size:
            keep = np.abs(avg.tfr_times) <= float(avg.times.max()) + 1e-9
            spectrum.pcolormesh(avg.tfr_times[keep] * 1000.0, avg.freqs,
                                avg.mean_power_db[:, keep], shading="nearest",
                                cmap="magma", vmin=DB_RANGE[0], vmax=DB_RANGE[1])
            spectrum.set_yscale("log")
            spectrum.set_ylabel("Hz", fontsize=7, color=tokens.text_muted)
            for edge in avg.band:
                if avg.freqs.size and avg.freqs[0] <= edge <= avg.freqs[-1]:
                    spectrum.axhline(edge, color="white", lw=0.6, alpha=0.45, ls="--")
            spectrum.axvline(-half, color="white", lw=0.6, alpha=0.5)
            spectrum.axvline(half, color="white", lw=0.6, alpha=0.5)
        else:
            spectrum.text(0.5, 0.5, "no time-frequency mean", ha="center", va="center",
                          fontsize=7, color=tokens.text_muted, transform=spectrum.transAxes)
            spectrum.set_yticks([])
        spectrum.set_xlabel("ms from the aligned peak", fontsize=7, color=tokens.text_muted)
        spectrum.set_xlim(float(ms.min()), float(ms.max()))
        self.canvas.draw_idle()

    # -- words ----------------------------------------------------------------
    @staticmethod
    def _caption(avg) -> str:
        text = (
            f"The mean of {avg.n} events on this channel, each aligned at the peak of its "
            "band-passed trace; the shading is one standard deviation across events. "
            "Bottom: the mean of each event's own time-frequency picture, in dB above "
            "its own surroundings, white lines at the band edges and at the mean "
            "event's span.<br><b>An island</b> under a spindle-shaped mean is what a "
            "channel of oscillations averages to; <b>a column</b> under a sharp mean "
            "is what a channel of transients and filter ringing averages to, whatever "
            "its rate.")
        if avg.why:
            # Not `capitalize()`: that lowercases the Hz and dB in the sentence.
            text += f"<br><br>{avg.why[0].upper()}{avg.why[1:]}."
        if avg.notes:
            text += "<br><br>" + " ".join(avg.notes)
        return text
