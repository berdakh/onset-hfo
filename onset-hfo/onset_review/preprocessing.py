"""Preprocessing, with a mouse: the MNE steps a reviewer would otherwise type.

Everything this panel does, `onset_hfo.preprocess.prepare` already did -- it
calls MNE's `filter`, `notch_filter`, `resample`, `pick` and `drop_channels`,
and builds the bipolar montage. What it did not have was a way to change any of
it without editing a config in Python, which means that in practice nobody
changed it and nobody checked whether the defaults suited their recording.

So this is not new signal processing. It is the existing chain, exposed, with
three rules:

1. **Nothing is applied silently.** Every control writes a line into
   `Prepared.steps`, which is already what the provenance panel and the
   exported report print. A reviewer who widens the notch to 6 Hz gets a report
   that says so.
2. **Settings that would produce numbers rather than a measurement are
   refused**, not applied. A low-pass below the band being analysed removes the
   signal and leaves a rate; `prepare` and `session_from_recording` raise, and
   this panel says so before the reviewer clicks anything.
3. **The defaults are the measured ones.** Opening the panel and pressing Apply
   without touching it changes nothing. "Reset" returns to exactly that.

The panel builds a `PreprocessConfig` and emits it. It does not re-run anything
itself: `onset_review.window` owns the reload, because changing the filtering
changes every number in every other panel and the only honest response is to
rebuild them all.
"""

from __future__ import annotations

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onset_hfo.config import BANDS, PreprocessConfig

__all__ = ["PreprocessPanel", "RESAMPLE_CHOICES", "describe"]

#: Offered downsampling rates. Upsampling is not offered: it adds no
#: information and the only thing it can do is mislead someone reading the
#: sampling rate off a report.
RESAMPLE_CHOICES = [("Leave as recorded", None), ("2000 Hz", 2000.0),
                    ("1000 Hz", 1000.0), ("500 Hz", 500.0), ("250 Hz", 250.0)]

#: Mains options. "From the dataset" is the default and the right answer:
#: notching the wrong mains frequency leaves the interference in place *and*
#: carves a hole where there was none, and it never announces itself.
MAINS_CHOICES = [("From the dataset", None), ("50 Hz", 50.0), ("60 Hz", 60.0)]


def describe(cfg: PreprocessConfig, band: tuple[float, float],
             sfreq: float) -> tuple[str, list[str]]:
    """Plain English for a config, plus whatever is wrong with it.

    Pure, so the sentence a reviewer reads before clicking Apply is testable
    without a display -- and so the warnings cannot drift away from the
    refusals in `prepare`, which this mirrors.
    """
    rate = float(cfg.resample) if cfg.resample else float(sfreq)
    lines = []
    if cfg.highpass:
        lines.append(f"high-pass at {cfg.highpass:g} Hz")
    if cfg.lowpass:
        lines.append(f"low-pass at {cfg.lowpass:g} Hz")
    if cfg.notch:
        mains = f"{cfg.line_freq:g} Hz" if cfg.line_freq else "the mains frequency"
        lines.append(f"notch {mains}"
                     + (" and its harmonics" if cfg.notch_harmonics else " only")
                     + f", {cfg.notch_width:g} Hz wide")
    if cfg.resample:
        lines.append(f"resample to {cfg.resample:g} Hz")
    lines.append("re-reference to neighbouring contacts (bipolar)" if cfg.bipolar
                 else "re-reference to the common average" if cfg.average_reference
                 else "leave the recording's own reference")
    if cfg.drop_bads:
        lines.append("drop channels the dataset flagged bad")
    if cfg.exclude:
        lines.append(f"drop {len(cfg.exclude)} channel(s) you marked")

    warnings = []
    if cfg.highpass and cfg.lowpass and cfg.lowpass <= cfg.highpass:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) is at or below the "
                        f"high-pass ({cfg.highpass:g} Hz). That passes nothing.")
    if cfg.lowpass and cfg.lowpass >= rate / 2:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) is at or above the "
                        f"Nyquist frequency of {rate / 2:g} Hz.")
    elif cfg.lowpass and cfg.lowpass < band[1]:
        warnings.append(f"The low-pass ({cfg.lowpass:g} Hz) cuts into the band "
                        f"being analysed ({band[0]:.0f}–{band[1]:.0f} Hz). The "
                        f"detector would still run and its rates would mean "
                        f"nothing.")
    if not BANDS.usable(rate, band):
        warnings.append(f"At {rate:g} Hz this recording cannot carry the "
                        f"{band[1]:.0f} Hz top of the band being analysed; it "
                        f"needs more than {2 * band[1]:.0f} Hz.")
    if cfg.highpass and cfg.highpass > band[0]:
        warnings.append(f"The high-pass ({cfg.highpass:g} Hz) is inside the band "
                        f"being analysed, which starts at {band[0]:.0f} Hz.")
    if not cfg.notch:
        warnings.append("With the notch off, mains harmonics sit inside the HFO "
                        "bands and are detected as oscillations.")
    if cfg.notch_width <= 0:
        warnings.append(f"A notch width of {cfg.notch_width:g} Hz is not a "
                        f"filter. It must be positive.")
    if cfg.resample is not None and float(cfg.resample) <= 0:
        warnings.append(f"A sampling rate of {float(cfg.resample):g} Hz is not "
                        f"a rate. It must be positive.")
    if cfg.notch_width > 4.0:
        warnings.append(f"A {cfg.notch_width:g} Hz notch is wide; its harmonics "
                        f"carve visible holes in the band being analysed.")
    if not cfg.bipolar and not cfg.average_reference:
        warnings.append("Without re-referencing, a shared reference puts the "
                        "same noise on every channel, which reads as HFOs "
                        "appearing everywhere at once.")
    return "; ".join(lines) + ".", warnings


class PreprocessPanel(QWidget):
    """The MNE chain as controls, with what it will do written underneath."""

    #: Emitted on Apply, with the config to re-run the window under.
    applied = Signal(object)

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self._session = session
        self._band = session.request.band_hz
        self._sfreq = float(session.sfreq)
        self._default = PreprocessConfig()
        start = session.request.preprocess or self._default

        # -- filters -------------------------------------------------------
        self.highpass = _spin(0.0, 100.0, 0.5, " Hz", start.highpass or 0.0,
                              "Removes drift. 0 turns it off. The HFO bands "
                              "start at 80 Hz, so anything up to that is free.")
        self.highpass.setSpecialValueText("off")
        self.lowpass = _spin(0.0, 5000.0, 10.0, " Hz", start.lowpass or 0.0,
                             "0 — the default — means none, which is what an "
                             "HFO analysis wants: the band runs to 500 Hz and a "
                             "low-pass below that removes the signal.")
        self.lowpass.setSpecialValueText("off")

        self.notch = QCheckBox("Notch the mains")
        self.notch.setChecked(start.notch)
        self.mains = _combo(MAINS_CHOICES, start.line_freq)
        self.mains.setToolTip(
            "The dataset carries its own (50 Hz in Zurich, 60 Hz in the US). "
            "Overriding it wrongly leaves the interference in and carves a hole "
            "where there was none.")
        self.notch_width = _spin(0.5, 10.0, 0.5, " Hz wide", start.notch_width,
                                 "Narrow on purpose: harmonics at 180 and 240 Hz "
                                 "sit inside the ripple band.")
        self.harmonics = QCheckBox("…and its harmonics")
        self.harmonics.setChecked(start.notch_harmonics)

        filters = QGroupBox("Filtering")
        form = QFormLayout(filters)
        form.addRow("High-pass", self.highpass)
        form.addRow("Low-pass", self.lowpass)
        form.addRow(self.notch, self.mains)
        form.addRow("", self.harmonics)
        form.addRow("Notch width", self.notch_width)

        # -- reference -----------------------------------------------------
        self.bipolar = QRadioButton("Bipolar — each contact minus its neighbour")
        self.average = QRadioButton("Common average across all channels")
        self.monopolar = QRadioButton("None — keep the recording's reference")
        self.bipolar.setToolTip(
            "Standard for HFO work: a common reference shares its noise with "
            "every channel and produces HFOs that appear everywhere at once.")
        self.average.setToolTip(
            "Standard elsewhere in EEG. For HFOs it re-introduces exactly the "
            "shared noise the bipolar montage exists to suppress.")
        (self.bipolar if start.bipolar
         else self.average if start.average_reference
         else self.monopolar).setChecked(True)

        reference = QGroupBox("Re-referencing")
        inner = QVBoxLayout(reference)
        for button in (self.bipolar, self.average, self.monopolar):
            inner.addWidget(button)

        # -- rate ----------------------------------------------------------
        self.resample = _combo(RESAMPLE_CHOICES, start.resample)
        self.resample.setToolTip(
            "Downsampling is the fastest way to make an HFO analysis "
            "meaningless: 1000 Hz cannot carry the fast-ripple band at all.")
        rate = QGroupBox("Sampling rate")
        rate_form = QFormLayout(rate)
        rate_form.addRow(f"Recorded at {self._sfreq:g} Hz", self.resample)

        # -- channels ------------------------------------------------------
        self.drop_bads = QCheckBox("Drop channels the dataset flagged bad")
        self.drop_bads.setChecked(start.drop_bads)
        self.channels = QListWidget()
        self.channels.setSelectionMode(QListWidget.NoSelection)
        self.channels.setMaximumHeight(130)
        self.channels.setToolTip(
            "Tick a contact to leave it out. Bipolar pairs are built after "
            "this, so excluding one contact removes both of the pairs it was "
            "part of.")
        excluded = set(start.exclude)
        for name in _contacts(session):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in excluded else Qt.Unchecked)
            self.channels.addItem(item)

        channels = QGroupBox("Channels")
        channel_box = QVBoxLayout(channels)
        channel_box.addWidget(self.drop_bads)
        channel_box.addWidget(QLabel("Exclude:"))
        channel_box.addWidget(self.channels)

        # -- summary and buttons -------------------------------------------
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.summary.setStyleSheet(
            "padding:5px;background:#eef4fb;border:1px solid #cfe0f0;font-size:11px;")
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.warnings.setStyleSheet(
            "padding:5px;background:#fdf0f0;border:1px solid #e0b0b0;"
            "color:#8a2020;font-size:11px;")

        self.apply = QPushButton("Apply and re-analyse")
        self.apply.setToolTip(
            "Re-runs this window from the signal up. Everything on screen "
            "changes, because every number depends on these settings.")
        self.reset = QPushButton("Reset to the measured defaults")
        buttons = QHBoxLayout()
        buttons.addWidget(self.reset)
        buttons.addStretch(1)
        buttons.addWidget(self.apply)

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(4, 4, 4, 4)
        column.setSpacing(6)
        for group in (filters, reference, rate, channels):
            column.addWidget(group)
        column.addWidget(self.summary)
        column.addWidget(self.warnings)
        column.addLayout(buttons)
        column.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        for widget in (self.highpass, self.lowpass, self.notch_width):
            widget.valueChanged.connect(self.refresh)
        for widget in (self.notch, self.harmonics, self.drop_bads):
            widget.stateChanged.connect(self.refresh)
        for widget in (self.bipolar, self.average, self.monopolar):
            widget.toggled.connect(self.refresh)
        for widget in (self.mains, self.resample):
            widget.currentIndexChanged.connect(self.refresh)
        self.channels.itemChanged.connect(self.refresh)
        self.apply.clicked.connect(self._apply)
        self.reset.clicked.connect(self.reset_to_defaults)
        self.refresh()

    # -- the config this panel describes -----------------------------------
    def config(self) -> PreprocessConfig:
        return PreprocessConfig(
            line_freq=self.mains.currentData(),
            notch=self.notch.isChecked(),
            notch_width=float(self.notch_width.value()),
            notch_harmonics=self.harmonics.isChecked(),
            bipolar=self.bipolar.isChecked(),
            average_reference=self.average.isChecked(),
            highpass=float(self.highpass.value()) or 0.0,
            lowpass=float(self.lowpass.value()) or None,
            resample=self.resample.currentData(),
            drop_bads=self.drop_bads.isChecked(),
            exclude=tuple(self._checked()),
        )

    def _checked(self) -> list[str]:
        return [self.channels.item(i).text()
                for i in range(self.channels.count())
                if self.channels.item(i).checkState() == Qt.Checked]

    def refresh(self) -> None:
        """Rewrite the summary and the warnings for the current settings."""
        summary, warnings = describe(self.config(), self._band, self._sfreq)
        self.summary.setText("This will " + summary)
        self.warnings.setText("\n".join("• " + w for w in warnings))
        self.warnings.setVisible(bool(warnings))
        self.apply.setEnabled(self.config() != (
            self._session.request.preprocess or self._default))

    def reset_to_defaults(self) -> None:
        cfg = self._default
        self.highpass.setValue(cfg.highpass or 0.0)
        self.lowpass.setValue(cfg.lowpass or 0.0)
        self.notch.setChecked(cfg.notch)
        self.harmonics.setChecked(cfg.notch_harmonics)
        self.notch_width.setValue(cfg.notch_width)
        _select(self.mains, cfg.line_freq)
        _select(self.resample, cfg.resample)
        self.bipolar.setChecked(cfg.bipolar)
        self.drop_bads.setChecked(cfg.drop_bads)
        for index in range(self.channels.count()):
            self.channels.item(index).setCheckState(Qt.Unchecked)
        self.refresh()

    def _apply(self) -> None:
        """Hand the config up. Refused settings never get this far.

        The panel declines rather than letting `prepare` raise after a reload
        has already been started: a reviewer should learn that a 150 Hz
        low-pass is wrong from the red text under the control, not from a
        dialog thirty seconds later.
        """
        cfg = self.config()
        _, warnings = describe(cfg, self._band, self._sfreq)
        blocking = [w for w in warnings if "would still run" in w
                    or "passes nothing" in w or "Nyquist" in w
                    or "cannot carry" in w]
        if blocking:
            self.warnings.setText("\n".join("• " + w for w in blocking)
                                  + "\n\nFix this before applying.")
            self.warnings.setVisible(True)
            return
        self.applied.emit(cfg)


def _contacts(session) -> list[str]:
    """Every physical contact behind the window's channels, in order.

    Contacts rather than channels, because exclusion happens before the bipolar
    montage is built -- that is what makes it a *preprocessing* choice and not
    a filter on the results.
    """
    findings = getattr(session, "findings", None)
    if findings is None or "channel" not in getattr(findings, "columns", []):
        return []
    seen: dict[str, None] = {}
    for channel in findings["channel"]:
        for contact in str(channel).split("-"):
            if contact:
                seen.setdefault(contact, None)
    return list(seen)


def _spin(low: float, high: float, step: float, suffix: str, value: float,
          tooltip: str) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(low, high)
    box.setSingleStep(step)
    box.setDecimals(1)
    box.setSuffix(suffix)
    box.setValue(value)
    box.setToolTip(tooltip)
    return box


def _combo(choices, value) -> QComboBox:
    box = QComboBox()
    for label, data in choices:
        box.addItem(label, data)
    _select(box, value)
    return box


def _select(box: QComboBox, value) -> None:
    for index in range(box.count()):
        if box.itemData(index) == value:
            box.setCurrentIndex(index)
            return
    box.setCurrentIndex(0)
