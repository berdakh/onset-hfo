"""The Spectrum panel: every channel's power spectrum, the chosen one in front.

A log-log plot with the band being analysed shaded, the mains lines dashed,
every channel faint and the chosen channel drawn over them with its slope
line, so "is this contact noisy or busy" is answered by looking rather than
by trusting a flag. The headline says the numbers the picture shows.

Only draws; the numbers come from :mod:`onset_review.spectrum`.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onset_review import spectrum, theme
from onset_review.theme import card

__all__ = ["SpectrumPanel"]


class SpectrumPanel(QWidget):
    """Power against frequency, per channel, with the chosen channel in front."""

    channelPicked = Signal(str)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._session = session
        self._spectrum = None
        self._highlighted: str | None = None
        self._lines: dict[str, object] = {}

        self.show_mode = QComboBox()
        self.show_mode.setObjectName("onset_spectrum_show")
        self.show_mode.addItem("Chosen channel over all", "all")
        self.show_mode.addItem("Chosen channel only", "one")
        self.show_mode.addItem("Leading five", "leading")
        self.slope = QCheckBox("Slope line")
        self.slope.setChecked(True)
        self.slope.setToolTip("The straight line fitted to log power against log "
                              "frequency away from the mains lines; its slope is "
                              "the number in the headline")

        bar = QHBoxLayout()
        bar.setSpacing(6)
        bar.addWidget(QLabel("Show"))
        bar.addWidget(self.show_mode)
        bar.addWidget(self.slope)
        bar.addStretch(1)

        tokens = theme.current()
        self.figure = Figure(figsize=(4.4, 3.0), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumHeight(150)
        self.canvas.setMinimumWidth(240)
        self.axes = self.figure.add_axes((0.14, 0.15, 0.83, 0.80))
        self.canvas.mpl_connect("pick_event", self._picked)

        self.headline = QLabel("")
        self.headline.setObjectName("onset_spectrum_headline")
        self.headline.setWordWrap(True)
        self.headline.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.headline.setStyleSheet(f"font-size:10pt;padding:1px 4px;color:{tokens.text};")
        self.caption = QLabel(
            "Welch's estimate of the signal as the detectors saw it: high-passed, "
            "notched, re-referenced. The shaded band is the one being analysed; "
            "dashed lines are the mains and its harmonics. A steep fall-off is the "
            "brain's; a flat carpet of high-frequency power is what a noisy or "
            "poorly coupled contact lays down, and a comb of peaks is mains. "
            "Click a line to choose that channel.")
        self.caption.setWordWrap(True)
        self.caption.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Maximum)
        self.caption.setStyleSheet(card("info"))

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

        self.show_mode.currentIndexChanged.connect(lambda _i: self.redraw())
        self.slope.stateChanged.connect(lambda _s: self.redraw())
        self.redraw()

    # -- data ---------------------------------------------------------------

    @property
    def _unit(self) -> str:
        """The unit this window's amplitudes are in: µV, or fT/cm or fT for MEG."""
        return getattr(self._session, "unit", "µV")
    def spectrum(self):
        """Computed once per session and kept: a minute of 43 channels is a
        fraction of a second, but not one to spend on every click."""
        if self._spectrum is None:
            self._spectrum = spectrum.compute(self._session)
        return self._spectrum

    def highlight(self, channel: str) -> None:
        if not channel or channel == self._highlighted:
            return
        self._highlighted = str(channel)
        self.redraw()

    # -- drawing --------------------------------------------------------------
    def redraw(self) -> None:
        tokens = theme.current()
        spec = self.spectrum()
        self.axes.clear()
        self.axes.set_facecolor(tokens.surface)
        self._lines = {}
        if not spec.available:
            self.axes.text(0.5, 0.5, "No signal to take a spectrum of", ha="center",
                           va="center", transform=self.axes.transAxes,
                           color=tokens.text_muted)
            self.headline.setText("")
            self.canvas.draw_idle()
            return

        mode = str(self.show_mode.currentData() or "all")
        chosen = self._highlighted if self._highlighted in spec.channels else None
        if chosen is None and spec.channels:
            chosen = self._leading()[0] if self._leading() else spec.channels[0]
        if mode == "one":
            shown = [chosen]
        elif mode == "leading":
            shown = self._leading()[:5] or spec.channels[:5]
            if chosen not in shown:
                shown.append(chosen)
        else:
            shown = list(spec.channels)

        lo, hi = spec.band
        self.axes.axvspan(lo, hi, color=tokens.accent, alpha=0.08, zorder=0)
        for line in spectrum.mains_lines(spec):
            self.axes.axvline(line, color=tokens.text_muted, linestyle=(0, (2, 4)),
                              linewidth=0.6, alpha=0.6, zorder=1)
        positive = spec.freqs > 0
        for channel in shown:
            i = spec.index(channel)
            power = np.maximum(spec.power[i][positive], np.finfo(float).tiny)
            is_chosen = channel == chosen
            (artist,) = self.axes.loglog(
                spec.freqs[positive], power,
                color=tokens.accent if is_chosen else tokens.text_muted,
                linewidth=1.8 if is_chosen else 0.6,
                alpha=1.0 if is_chosen else 0.35, zorder=5 if is_chosen else 2,
                picker=4, label=channel)
            self._lines[channel] = artist
        if chosen and self.slope.isChecked():
            slope, intercept = spectrum.slope_fit(spec, chosen)
            if not np.isnan(slope):
                f = np.array(spectrum.SLOPE_RANGE)
                self.axes.loglog(f, 10 ** (intercept + slope * np.log10(f)),
                                 color=tokens.warn, linewidth=1.4, linestyle="--", zorder=6)
        self.axes.set_xlabel("frequency (Hz)", fontsize=8, color=tokens.text_muted)
        self.axes.set_ylabel(f"power ({self._unit}²/Hz)", fontsize=8, color=tokens.text_muted)
        self.axes.tick_params(labelsize=7, colors=tokens.text_muted)
        for side in self.axes.spines.values():
            side.set_color(tokens.separator)
        self.axes.set_xlim(max(0.5, float(spec.freqs[positive].min())), float(spec.freqs.max()))
        self.headline.setText(spectrum.describe(spec, chosen) if chosen else "")
        self.canvas.draw_idle()

    def _leading(self) -> list[str]:
        findings = getattr(self._session, "findings", None)
        if findings is None or findings.empty or "channel" not in findings.columns:
            return []
        spec = self.spectrum()
        return [str(c) for c in findings.sort_values("rank")["channel"]
                if str(c) in spec.channels]

    # -- interaction ----------------------------------------------------------
    def _picked(self, event) -> None:
        for channel, artist in self._lines.items():
            if event.artist is artist:
                self.highlight(channel)
                self.channelPicked.emit(channel)
                return
