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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from onset_hfo.config import PreprocessConfig
from onset_hfo.preprocess import (
    describe,
    effective_reference,
    filter_description,
    ica_methods_available,
)
from onset_review.theme import card, current, muted, scrolled

#: Re-exported: the sentence and the warnings a reviewer reads under these
#: controls are computed in `onset_hfo.preprocess`, beside the refusals they
#: mirror, so the two cannot drift. Nothing in this file decides anything.
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

        # -- filter design --------------------------------------------------
        self.method = QComboBox()
        self.method.addItem("Windowed FIR (MNE's default)", "fir")
        self.method.addItem("Butterworth IIR", "iir")
        self.method.setToolTip(
            "HFO detection is filter-sensitive: a long FIR rings at a sharp "
            "discharge and the ringing looks like a ripple. The design is "
            "yours; the line below says what MNE actually builds.")
        _select(self.method, start.filter_method)
        self.iir_order = QSpinBox()
        self.iir_order.setRange(1, 16)
        self.iir_order.setValue(int(start.iir_order))
        self.iir_order.setToolTip("Butterworth order; applied twice for zero phase, "
                                  "so the effective order is double")
        self.phase = QComboBox()
        self.phase.addItem("Zero phase (no delay)", "zero")
        self.phase.addItem("Minimum phase (causal, delayed)", "minimum")
        _select(self.phase, start.filter_phase)
        self.transition = _spin(0.0, 50.0, 0.25, " Hz", start.transition_bandwidth or 0.0,
                                "FIR transition bandwidth. Narrower is sharper and "
                                "longer; longer rings more. 0 lets MNE choose.")
        self.transition.setSpecialValueText("auto")
        self.design = QLabel()
        self.design.setWordWrap(True)
        self.design.setStyleSheet(f"color:{current().text_muted};font-size:9pt;")

        filters = QGroupBox("Filtering")
        form = QFormLayout(filters)
        form.addRow("High-pass", self.highpass)
        form.addRow("Low-pass", self.lowpass)
        form.addRow(self.notch, self.mains)
        form.addRow("", self.harmonics)
        form.addRow("Notch width", self.notch_width)
        form.addRow("Design", self.method)
        form.addRow("IIR order", self.iir_order)
        form.addRow("Phase", self.phase)
        form.addRow("Transition", self.transition)
        form.addRow("", self.design)

        # -- reference -----------------------------------------------------
        self.bipolar = QRadioButton("Bipolar — each contact minus its neighbour")
        self.shaft = QRadioButton("Per-shaft average — each contact minus its electrode's mean")
        self.median = QRadioButton("Common median across all channels")
        self.average = QRadioButton("Common average across all channels")
        self.monopolar = QRadioButton("None — keep the recording's reference")
        self.bipolar.setToolTip(
            "Standard for HFO work: a common reference shares its noise with "
            "every channel and produces HFOs that appear everywhere at once.")
        self.shaft.setToolTip(
            "The usual SEEG choice after bipolar: removes what a whole shaft "
            "shares, keeps each contact's own signal at its own place.")
        self.median.setToolTip(
            "The average's robust cousin: one faulty contact cannot drag it. "
            "Still shares the common noise.")
        self.average.setToolTip(
            "Standard elsewhere in EEG. For HFOs it re-introduces exactly the "
            "shared noise the bipolar montage exists to suppress.")
        self._reference_buttons = {"bipolar": self.bipolar, "shaft": self.shaft,
                                   "median": self.median, "average": self.average,
                                   "none": self.monopolar}
        self._reference_buttons[effective_reference(start)].setChecked(True)

        reference = QGroupBox("Re-referencing")
        inner = QVBoxLayout(reference)
        for button in (self.bipolar, self.shaft, self.median, self.average, self.monopolar):
            inner.addWidget(button)

        # -- artifact annotation ---------------------------------------------
        self.muscle = QCheckBox("Mark muscle and movement bursts")
        self.muscle.setChecked(bool(start.annotate_muscle))
        self.muscle.setToolTip(
            "MNE's muscle annotator: seconds where broadband 110–140 Hz power "
            "rises across the whole montage at once. Set aside by the quality "
            "stage, never deleted.")
        self.muscle_z = _spin(1.0, 10.0, 0.5, " z", float(start.muscle_z),
                              "How far above the montage's usual high-frequency "
                              "power a second must rise to be marked")
        self.amplitude = QCheckBox("Mark seconds above a peak-to-peak ceiling")
        self.amplitude.setChecked(bool(start.annotate_amplitude))
        self.amplitude.setToolTip(
            "MNE's amplitude annotator, per contact. Off by default: an "
            "amplitude ceiling removes the loudest seconds, and on an "
            "epileptic contact those are the discharges.")
        self.ptp = _spin(0.0, 20000.0, 50.0, " µV", float(start.amplitude_ptp_uv or 0.0),
                         "The ceiling. 0 learns one per contact from the data by "
                         "cross-validation, as autoreject does for its global "
                         "threshold. Nothing is interpolated.")
        self.ptp.setSpecialValueText("learn from the data")

        artifacts = QGroupBox("Artifact annotation")
        artifact_form = QFormLayout(artifacts)
        artifact_form.addRow(self.muscle, self.muscle_z)
        artifact_form.addRow(self.amplitude, self.ptp)
        artifact_form.addRow("", muted(
            "Marked seconds are set aside by the data-quality stage and shown "
            "there with their reason; the detectors never see them. Nothing is "
            "repaired or deleted.", current()))

        # -- regression and ICA, experimental --------------------------------
        self.regress = QListWidget()
        self.regress.setObjectName("onset_regress_channels")
        self.regress.setSelectionMode(QListWidget.NoSelection)
        self.regress.setMaximumHeight(70)
        self.regress.setToolTip(
            "Tick a channel to regress its signal out of every brain channel before "
            "the montage by least squares: an ECG lead, a reference or a "
            "ground channel the export carries. The channel itself is not analysed.")
        regressing = set(start.regress_channels)
        for name in _regressable(session):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if name in regressing else Qt.Unchecked)
            self.regress.addItem(item)
        self.ica = QCheckBox("Fit ICA (experimental)")
        self.ica.setObjectName("onset_ica")
        self.ica.setChecked(bool(start.ica))
        self.ica.setToolTip(
            "Fits MNE's ICA on the filtered channels and shows the components on "
            "the Components panel, scored for muscle and ECG. Removes nothing until "
            "you choose components there. Seconds to tens of seconds.")
        self.ica_method = QComboBox()
        for name in ica_methods_available():
            self.ica_method.addItem(name, name)
        index = self.ica_method.findData(start.ica_method)
        self.ica_method.setCurrentIndex(index if index >= 0 else 0)
        self.ica_components = QSpinBox()
        self.ica_components.setRange(0, 200)
        self.ica_components.setSpecialValueText("up to 20")
        self.ica_components.setValue(int(start.ica_n_components or 0))
        self.ica_components.setToolTip("Components to fit; 'up to 20' lets the stage choose")
        self._ica_exclude: tuple[int, ...] = tuple(int(i) for i in start.ica_exclude)
        self.ica_removed = muted("", current())
        self.ica_removed.setObjectName("onset_ica_removed")
        experimental = QGroupBox("Regression and ICA (experimental)")
        experimental_form = QFormLayout(experimental)
        experimental_form.addRow("Regress out", self.regress)
        if not self.regress.count():
            experimental_form.addRow("", muted(
                "This recording carries no reference or ECG channel to regress out.",
                current()))
        experimental_form.addRow(self.ica, self.ica_method)
        experimental_form.addRow("Components", self.ica_components)
        experimental_form.addRow("", self.ica_removed)
        experimental_form.addRow("", muted(
            "ICA can take real HFO energy out with the artefact. Nothing is removed "
            "until you choose components on the Components panel; the report names "
            "what was removed.", current()))

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
        self.summary.setStyleSheet(card("info"))
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        self.warnings.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Minimum)
        self.warnings.setStyleSheet(card("bad"))

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
        for group in (filters, reference, artifacts, experimental, rate, channels):
            column.addWidget(group)
        column.addWidget(self.summary)
        column.addWidget(self.warnings)
        column.addLayout(buttons)
        column.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scrolled(body))

        for widget in (self.highpass, self.lowpass, self.notch_width, self.transition,
                       self.muscle_z, self.ptp):
            widget.valueChanged.connect(self.refresh)
        self.iir_order.valueChanged.connect(self.refresh)
        for widget in (self.notch, self.harmonics, self.drop_bads, self.muscle,
                       self.amplitude):
            widget.stateChanged.connect(self.refresh)
        for widget in self._reference_buttons.values():
            widget.toggled.connect(self.refresh)
        for widget in (self.mains, self.resample, self.method, self.phase):
            widget.currentIndexChanged.connect(self.refresh)
        self.channels.itemChanged.connect(self.refresh)
        self.apply.clicked.connect(self._apply)
        self.reset.clicked.connect(self.reset_to_defaults)
        self.refresh()

    # -- the config this panel describes -----------------------------------
    def reference(self) -> str:
        for name, button in self._reference_buttons.items():
            if button.isChecked():
                return name
        return "none"

    def config(self) -> PreprocessConfig:
        scheme = self.reference()
        return PreprocessConfig(
            line_freq=self.mains.currentData(),
            notch=self.notch.isChecked(),
            notch_width=float(self.notch_width.value()),
            notch_harmonics=self.harmonics.isChecked(),
            bipolar=scheme == "bipolar",
            average_reference=scheme == "average",
            # Named only when the two older flags cannot say it, so a panel on
            # the defaults equals the defaults and Apply stays grey.
            reference=scheme if scheme in ("median", "shaft") else None,
            highpass=float(self.highpass.value()) or 0.0,
            lowpass=float(self.lowpass.value()) or None,
            resample=self.resample.currentData(),
            drop_bads=self.drop_bads.isChecked(),
            exclude=tuple(self._checked()),
            filter_method=str(self.method.currentData() or "fir"),
            iir_order=int(self.iir_order.value()),
            filter_phase=str(self.phase.currentData() or "zero"),
            transition_bandwidth=float(self.transition.value()) or None,
            annotate_muscle=self.muscle.isChecked(),
            muscle_z=float(self.muscle_z.value()),
            annotate_amplitude=self.amplitude.isChecked(),
            amplitude_ptp_uv=float(self.ptp.value()) or None,
            regress_channels=tuple(self._regressing()),
            ica=self.ica.isChecked(),
            ica_method=str(self.ica_method.currentData() or "fastica"),
            ica_n_components=int(self.ica_components.value()) or None,
            ica_exclude=tuple(self._ica_exclude) if self.ica.isChecked() else (),
        )

    def _regressing(self) -> list[str]:
        return [self.regress.item(i).text() for i in range(self.regress.count())
                if self.regress.item(i).checkState() == Qt.Checked]

    def set_ica_exclude(self, indices) -> None:
        """The Components panel's choice, carried into the next Apply."""
        self._ica_exclude = tuple(sorted({int(i) for i in indices}))
        self.refresh()

    def _checked(self) -> list[str]:
        return [self.channels.item(i).text()
                for i in range(self.channels.count())
                if self.channels.item(i).checkState() == Qt.Checked]

    def refresh(self) -> None:
        """Rewrite the summary and the warnings for the current settings."""
        cfg = self.config()
        summary, warnings = describe(cfg, self._band, self._sfreq)
        self.summary.setText("This will " + summary)
        is_iir = cfg.filter_method == "iir"
        self.iir_order.setEnabled(is_iir)
        self.phase.setEnabled(not is_iir)
        self.transition.setEnabled(not is_iir)
        self.muscle_z.setEnabled(cfg.annotate_muscle)
        self.ptp.setEnabled(cfg.annotate_amplitude)
        self.ica_method.setEnabled(cfg.ica)
        self.ica_components.setEnabled(cfg.ica)
        self.ica_removed.setText(
            ("Components to remove: " + ", ".join(str(i) for i in self._ica_exclude)
             + " (chosen on the Components panel)") if cfg.ica and self._ica_exclude
            else ("No component removed; choose on the Components panel after Apply."
                  if cfg.ica else ""))
        self.design.setText("MNE builds: " + filter_description(cfg, self._sfreq))
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
        self._reference_buttons[effective_reference(cfg)].setChecked(True)
        _select(self.method, cfg.filter_method)
        self.iir_order.setValue(int(cfg.iir_order))
        _select(self.phase, cfg.filter_phase)
        self.transition.setValue(cfg.transition_bandwidth or 0.0)
        self.muscle.setChecked(bool(cfg.annotate_muscle))
        self.muscle_z.setValue(float(cfg.muscle_z))
        self.amplitude.setChecked(bool(cfg.annotate_amplitude))
        self.ptp.setValue(float(cfg.amplitude_ptp_uv or 0.0))
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
                    or "cannot carry" in w or "below 0 Hz" in w
                    or "must be FIR or IIR" in w or "solver is not installed" in w]
        if blocking:
            self.warnings.setText("\n".join("• " + w for w in blocking)
                                  + "\n\nFix this before applying.")
            self.warnings.setVisible(True)
            return
        self.applied.emit(cfg)


def _regressable(session) -> list[str]:
    """The recording's channels that are not brain channels: what there is
    to regress out. An ECG lead, a reference, a ground; never a contact."""
    from onset_hfo.preprocess import _is_brain_channel

    raw = getattr(getattr(session, "recording", None), "raw", None)
    if raw is None:
        return []
    out = []
    for name, kind in zip(raw.ch_names, raw.get_channel_types(), strict=False):
        if not _is_brain_channel(name, kind) and not name.upper().startswith(
                ("DC", "TRIG", "STIM", "EVENT", "MARK")):
            out.append(name)
    return out


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
