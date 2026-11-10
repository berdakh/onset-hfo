"""Two seconds of signal before and after the filters, as they are being set.

Changing a filter on the Signal page re-runs the whole window, which is right
-- every number depends on it -- and slow enough that nobody tries a setting
just to see what it does. This shows the effect first: two seconds of the
busiest channel as recorded, and the same two seconds through the high-pass,
low-pass and notch the page is set to, redrawn as the settings change.

Only the filters are previewed. The reference, the artefact marking,
regression and ICA act across channels or over the whole window, and a
two-second picture of them would be a guess; the caption says so.

The signal comes from the recording as it was read, before anything was done
to it, so the "before" trace is the file and not the last analysis. The two
seconds are filtered with three seconds either side and then cut, so the
filter's own start-up does not show as an artefact at the edges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from onset_hfo.preprocess import notch_frequencies

__all__ = ["Preview", "preview_signal", "apply_filters", "PreviewPanel", "SECONDS", "PAD"]

#: Seconds shown.
SECONDS = 2.0
#: Seconds filtered either side and then cut away.
PAD = 3.0


@dataclass
class Preview:
    """The stretch of signal the preview shows, before any filter."""

    label: str            # the channel, as the ranking names it
    t0: float             # where the shown two seconds start, in recording seconds
    sfreq: float
    signal: np.ndarray    # SECONDS + 2 * PAD of it, in volts
    line_freq: float | None

    @property
    def shown(self) -> slice:
        pad = int(round(PAD * self.sfreq))
        return slice(pad, pad + int(round(SECONDS * self.sfreq)))

    @property
    def times(self) -> np.ndarray:
        n = self.shown.stop - self.shown.start
        return self.t0 + np.arange(n) / self.sfreq


def _source_raw(session):
    return getattr(getattr(session, "recording", None), "raw", None)


def _channel_signal(raw, name: str):
    """`name` as the analysis names it: a bipolar pair "A-B" is A minus B
    from the recording; a single contact is itself."""
    names = list(raw.ch_names)
    if name in names:
        return raw.get_data(picks=[name])[0], name
    if "-" in name:
        first, second = name.split("-", 1)
        if first in names and second in names:
            data = raw.get_data(picks=[first, second])
            return data[0] - data[1], name
    return None, ""


def preview_signal(session, channel: str | None = None) -> Preview | None:
    """The busiest channel (or `channel`), two seconds around its first event,
    from the recording as read. None when the recording is not at hand
    or shorter than the two seconds."""
    raw = _source_raw(session)
    if raw is None or not len(raw.ch_names):
        return None
    leader = channel or (session.leader.get("leader")
                         if getattr(session, "leader", {}).get("available") else None)
    data, label = (None, "")
    for name in [n for n in (leader,) if n] + list(session.findings.get("channel", [])):
        data, label = _channel_signal(raw, str(name))
        if data is not None:
            break
    if data is None:
        from onset_hfo.preprocess import _is_brain_channel

        brain = [n for n, kind in zip(raw.ch_names, raw.get_channel_types(), strict=True)
                 if _is_brain_channel(n, kind)]
        if not brain:
            return None
        label = brain[0]
        data = raw.get_data(picks=[label])[0]
    sfreq = float(raw.info["sfreq"])
    # Two seconds starting a little before the channel's first accepted event,
    # so there is something to look at; the start of the window otherwise.
    t_first = None
    for event in getattr(session, "events", []):
        if getattr(event, "accepted", True) and getattr(event, "channel", "") == label:
            t_first = float(event.start)
            break
    offset = float(getattr(session, "t_offset", 0.0) or 0.0)
    shown = int(round(SECONDS * sfreq))
    pad = int(round(PAD * sfreq))
    if len(data) < shown:
        return None
    start = ((t_first - offset - 0.5) if t_first is not None else 0.0)
    first = int(round(min(max(start, 0.0), len(data) / sfreq - SECONDS) * sfreq))
    # The padding is the recording where there is recording, and its mirror
    # image past either end, so two seconds at the very start can be shown.
    lo, hi = first - pad, first + shown + pad
    piece = np.asarray(data[max(lo, 0):min(hi, len(data))], dtype=float)
    piece = np.pad(piece, (max(0, -lo), max(0, hi - len(data))), mode="reflect")
    line = getattr(getattr(session, "recording", None), "line_freq", None)
    return Preview(label=label, t0=offset + first / sfreq, sfreq=sfreq, signal=piece,
                   line_freq=float(line) if line else None)


def apply_filters(preview: Preview, cfg) -> tuple[np.ndarray, list[str]]:
    """The preview's signal through `cfg`'s high-pass, low-pass and notch, as
    `onset_hfo.preprocess.prepare` applies them. Returns the shown two
    seconds and what was done, in words."""
    import mne

    x = preview.signal.copy()
    done: list[str] = []
    sfreq = preview.sfreq
    nyquist = sfreq / 2.0
    method = getattr(cfg, "filter_method", "fir") or "fir"
    kwargs = ({"method": "iir", "iir_params": {"order": int(cfg.iir_order), "ftype": "butter"}}
              if method == "iir" else
              {"method": "fir", "phase": "minimum" if cfg.filter_phase == "minimum" else "zero"})
    high = float(cfg.highpass or 0.0) or None
    low = float(cfg.lowpass) if cfg.lowpass else None
    if low is not None and low >= nyquist:
        low = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if high or low:
            if method == "fir" and cfg.transition_bandwidth:
                if high:
                    kwargs["l_trans_bandwidth"] = float(cfg.transition_bandwidth)
                if low:
                    kwargs["h_trans_bandwidth"] = float(cfg.transition_bandwidth)
            x = mne.filter.filter_data(x[np.newaxis], sfreq, high, low, verbose="ERROR",
                                       **kwargs)[0]
            done.append(" and ".join(part for part in (
                f"high-pass {high:g} Hz" if high else "",
                f"low-pass {low:g} Hz" if low else "") if part))
        line = cfg.line_freq or preview.line_freq
        if cfg.notch and line:
            freqs = notch_frequencies(sfreq, line, cfg.notch_harmonics)
            if len(freqs):
                x = mne.filter.notch_filter(x[np.newaxis], sfreq, freqs,
                                            notch_widths=float(cfg.notch_width),
                                            verbose="ERROR")[0]
                done.append(f"notch at {line:g} Hz" + (" and its harmonics"
                                                       if cfg.notch_harmonics else ""))
        elif cfg.notch:
            done.append("no notch: the mains frequency is not known")
    return x[preview.shown], done


try:  # The widget needs Qt; the two functions above do not.
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from matplotlib.figure import Figure
    from qtpy.QtCore import QTimer
    from qtpy.QtWidgets import QVBoxLayout, QWidget
except Exception:          # pragma: no cover - the review extra is not installed
    QWidget = object       # type: ignore[assignment,misc]


class PreviewPanel(QWidget):
    """The two seconds, before and after, redrawn a moment after the settings
    stop changing."""

    def __init__(self, session, parent=None):
        super().__init__(parent)
        from onset_review import theme

        tokens = theme.current()
        self._preview = preview_signal(session)
        self._unit_scale = 1e6
        self._cfg = None
        self.figure = Figure(figsize=(5.0, 1.9), facecolor=tokens.surface)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(170)
        self.axes = self.figure.subplots(
            2, 1, sharex=True, gridspec_kw={"hspace": 0.15, "left": 0.12, "right": 0.96,
                                            "top": 0.92, "bottom": 0.2})
        self.caption = theme.muted("", size=9)
        self.caption.setObjectName("onset_preview_caption")
        self.caption.setWordWrap(True)
        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(self.canvas)
        box.addWidget(self.caption)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._draw)
        if self._preview is None:
            self.canvas.setVisible(False)
            self.caption.setText("No preview: the recording as read is not at hand in "
                                 "this window.")

    @property
    def available(self) -> bool:
        return self._preview is not None

    def show_config(self, cfg, now: bool = False) -> None:
        """Preview `cfg`; drawn after a short pause, or `now`."""
        self._cfg = cfg
        if self._preview is None:
            return
        if now:
            self._draw()
        else:
            self._timer.start()

    def _draw(self) -> None:
        from onset_review import theme

        if self._preview is None or self._cfg is None:
            return
        tokens = theme.current()
        before = self._preview.signal[self._preview.shown] * self._unit_scale
        after, done = apply_filters(self._preview, self._cfg)
        after = after * self._unit_scale
        times = self._preview.times
        for axis, trace, title, colour in ((self.axes[0], before, "recorded", tokens.text),
                                          (self.axes[1], after, "filtered", tokens.accent)):
            axis.clear()
            axis.set_facecolor(tokens.surface)
            axis.plot(times, trace, color=colour, linewidth=0.7)
            axis.set_ylabel(f"{title}\nµV", fontsize=7, color=tokens.text_muted)
            axis.tick_params(labelsize=7, colors=tokens.text_muted)
            for side in ("top", "right"):
                axis.spines[side].set_visible(False)
        self.axes[1].set_xlabel("seconds in the recording", fontsize=7,
                                color=tokens.text_muted)
        self.canvas.draw_idle()
        self.caption.setText(
            f"{self._preview.label}, {times[0]:.1f}–{times[-1]:.1f} s: "
            + ("; ".join(done) if done else "no filter") + ". Filters only: the "
            "reference, the artefact marking and ICA take effect when you apply.")
