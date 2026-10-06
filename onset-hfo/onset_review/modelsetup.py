"""Get a local Qwen onto this machine from inside the window.

The command line already does this (``onset-agent --backend auto``) and so
does the installer (``--with-assistant``). What neither gives a reviewer is a
way to do it from the application they are sitting in, with a bar that moves
while four gigabytes come down. This box is that way.

It does three things, each on a worker thread so the window keeps painting:

1. **Look.** Probe the machine, run the same chooser the CLI runs
   (:func:`onset_agent.hardware.choose` on the Ollama route), and ask the
   Ollama server which tags it already holds.
2. **Pull.** Stream ``/api/pull`` for the chosen tag, drawing progress.
3. **Use.** Write the assistant's defaults so the next window opens on this
   model, and tell the panel to switch to it now.

What it cannot do is install Ollama: that is a daemon, not a package. A
machine without one gets the download address and the two commands, and a
*Check again* button for after they have been run.

**The size is a choice.** Every catalogue model the machine can serve is in a
box, smallest first, and the box opens on the chooser's pick for a person
waiting at a window: on a CPU that is the 3B, which answers a briefed question
in well under a minute where the 7B takes several. A reviewer who finds it
careless goes one bigger; one who finds it slow goes one smaller. Each entry
says what it costs to pull and what to expect of it.
"""

from __future__ import annotations

import html

from qtpy.QtCore import Qt, QThread, Signal
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from onset_review import theme

__all__ = ["ModelSetupBox", "Look"]


class Look:
    """What the probe found, in one plain object the widget reads from."""

    def __init__(self) -> None:
        self.up: bool = False
        self.names: list[str] = []
        self.tag: str = ""
        self.model_id: str = ""
        self.download_gb: float = 0.0
        self.fits: bool = False
        self.warnings: list[str] = []
        self.machine: str = ""
        self.base_url: str = ""
        self.error: str = ""
        #: Every size this machine can serve: (tag, model_id, pull_gb, note),
        #: smallest first. `tag` is the one to open on.
        self.options: list[tuple[str, str, float, str]] = []

    @property
    def present(self) -> bool:
        return self.has(self.tag)

    def has(self, tag: str) -> bool:
        from onset_agent.hardware import _has_tag

        return bool(tag) and _has_tag(tag, self.names)

    def option(self, tag: str) -> tuple[str, str, float, str] | None:
        for entry in self.options:
            if entry[0] == tag:
                return entry
        return None


class _LookWorker(QThread):
    """Probe, choose and ask the server, off the GUI thread.

    `probe` may import torch, which can take seconds on a cold disk; the
    server check waits up to two seconds on a machine with nothing listening.
    Neither belongs on the thread that paints.
    """

    def __init__(self, base_url: str | None = None, parent=None):
        super().__init__(parent)
        self.base_url = base_url
        self.look = Look()

    def run(self) -> None:
        look = self.look
        try:
            from onset_agent.hardware import (
                choose,
                ollama_status,
                ollama_url,
                probe,
                served_options,
            )

            look.base_url = self.base_url or ollama_url()
            machine = probe()
            look.machine = (f"{machine.accelerator}, {machine.ram_gb:.0f} GB RAM"
                            + (f", {machine.gpus[0].vram_gb:.0f} GB VRAM"
                               if machine.gpus else ""))
            fitting, default = served_options(machine)
            look.options = [(c.ollama_tag, c.model_id, float(c.download_gb), c.model.note)
                            for c in fitting]
            choice = choose(machine, prefer=default, route="ollama")
            look.tag = choice.ollama_tag
            look.model_id = choice.model_id
            look.download_gb = float(choice.download_gb)
            look.fits = bool(choice.fits)
            look.warnings = list(choice.warnings)
            look.up, look.names = ollama_status(look.base_url)
        except Exception as error:      # noqa: BLE001 - reported, not raised
            look.error = str(error)


class _PullWorker(QThread):
    """Stream one pull, reporting every status line Ollama sends."""

    progress = Signal(str, int, int)

    def __init__(self, tag: str, base_url: str, parent=None):
        super().__init__(parent)
        self.tag = tag
        self.base_url = base_url
        self.status = ""
        self.error = ""

    def run(self) -> None:
        from onset_agent.hardware import ollama_pull_stream

        try:
            self.status = ollama_pull_stream(
                self.tag, self.base_url,
                on_progress=lambda s, done, total: self.progress.emit(s, done, total))
        except Exception as error:      # noqa: BLE001 - reported, not raised
            self.error = str(error)


class ModelSetupBox(QWidget):
    """Status, the recommended tag, and one button that does the right thing."""

    #: (tag, base_url) once a model is pulled and the defaults are written.
    modelReady = Signal(str, str)

    def __init__(self, base_url: str | None = None, parent=None):
        super().__init__(parent)
        self._base_url = base_url
        self._look: Look | None = None
        self._looker: _LookWorker | None = None
        self._puller: _PullWorker | None = None

        self.status = QLabel("Looking at this machine…")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.RichText)
        self.status.setOpenExternalLinks(True)
        self.status.setObjectName("onset_model_status")

        # The size, once the probe has said what fits. Hidden until then:
        # an empty box is a question with no options.
        self.size = QComboBox()
        self.size.setObjectName("onset_model_size")
        self.size.setToolTip("Which size to run. Smaller answers sooner; larger "
                             "reads a table more carefully.")
        self.size.setVisible(False)
        self.size.currentIndexChanged.connect(lambda _i: self._refresh())

        self.action = QPushButton("Download and use")
        self.action.setEnabled(False)
        self.action.setObjectName("onset_model_action")
        self.action.clicked.connect(self._act)
        self.again = QPushButton("Check again")
        self.again.setObjectName("onset_model_again")
        self.again.clicked.connect(self.look)
        for button in (self.action, self.again):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(True)
        self.bar.setVisible(False)
        self.bar.setObjectName("onset_model_bar")

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.size)
        buttons.addWidget(self.action)
        buttons.addWidget(self.again)
        buttons.addStretch(1)

        box = QVBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(4)
        box.addWidget(theme.section_label("Local model on this machine"))
        box.addWidget(self.status)
        box.addWidget(self.bar)
        box.addLayout(buttons)

    # -- looking ------------------------------------------------------------
    def idle(self) -> None:
        """Say that nothing has been checked yet, rather than 'Looking…'
        for ever when the probe is switched off."""
        self.status.setText("Not checked yet. <i>Check again</i> looks at this "
                            "machine and at Ollama.")
        self.action.setEnabled(False)
        self.again.setEnabled(True)

    def look(self) -> None:
        """Probe and report. Asynchronous; `looked` is set when it lands."""
        if self._looker is not None and self._looker.isRunning():
            return
        self.status.setText("Looking at this machine…")
        self.action.setEnabled(False)
        self.again.setEnabled(False)
        self._looker = _LookWorker(self._base_url, self)
        self._looker.finished.connect(self._looked)
        self._looker.start()

    def wait(self, ms: int = 30000) -> None:
        """Block until the current worker has finished. For tests and the
        screenshot path; the window itself never calls this."""
        from qtpy.QtWidgets import QApplication

        for worker in (self._looker, self._puller):
            if worker is not None and worker.isRunning():
                worker.wait(ms)
        # The workers' signals are queued onto this thread; a few turns of
        # the loop deliver them, including `finished` and whatever it runs.
        for _ in range(5):
            QApplication.processEvents()

    def _looked(self) -> None:
        self._look = look = self._looker.look
        self.again.setEnabled(True)
        if look.error:
            self.status.setText(
                f"Could not look at this machine: {html.escape(look.error)}")
            return
        self.size.blockSignals(True)
        self.size.clear()
        for tag, _model_id, gb, note in look.options:
            self.size.addItem(f"{tag} · {gb:.1f} GB", tag)
            self.size.setItemData(self.size.count() - 1, note, Qt.ToolTipRole)
        index = self.size.findData(look.tag)
        if index >= 0:
            self.size.setCurrentIndex(index)
        self.size.blockSignals(False)
        self.size.setVisible(self.size.count() > 0)
        self._refresh()

    def selected(self) -> str:
        """The tag the size box is on, or the chooser's pick before it is filled."""
        tag = self.size.currentData()
        return str(tag) if tag else (self._look.tag if self._look else "")

    def _refresh(self) -> None:
        """Say where things stand for the size that is selected."""
        look = self._look
        if look is None or look.error:
            return
        tag = self.selected()
        entry = look.option(tag)
        model_id = entry[1] if entry else look.model_id
        gb = entry[2] if entry else look.download_gb
        note = entry[3] if entry else ""
        is_pick = tag == look.tag
        which = ("the pick for this machine" if is_pick else "your pick")
        about = f" {html.escape(note[0].upper() + note[1:])}." if note else ""
        pull = f"<code>ollama pull {html.escape(tag)}</code>"
        if not look.up:
            self.status.setText(
                f"No Ollama server answers at {html.escape(look.base_url)}. "
                f"Install it from <a href='https://ollama.com/download'>"
                f"ollama.com/download</a>, start it with <code>ollama serve</code>, "
                f"then check again. For this machine ({html.escape(look.machine)}) "
                f"the chooser would pull <b>{html.escape(look.tag)}</b>, about "
                f"{look.download_gb:.1f} GB.")
            self.action.setText("Download and use")
            self.action.setEnabled(False)
            return
        if look.has(tag):
            self.status.setText(
                f"Ollama is running and already holds <b>{html.escape(tag)}</b> "
                f"({html.escape(model_id)}), {which} ({html.escape(look.machine)})."
                + about)
            self.action.setText("Use this model")
            self.action.setEnabled(True)
            return
        warnings = ("<br>" + " ".join(html.escape(w) for w in look.warnings)
                    if look.warnings and is_pick else "")
        self.status.setText(
            f"Ollama is running at {html.escape(look.base_url)} but has not pulled "
            f"<b>{html.escape(tag)}</b> ({html.escape(model_id)}), {which} "
            f"({html.escape(look.machine)}): about {gb:.1f} GB, once. Or run {pull} "
            f"yourself.{about}" + warnings)
        self.action.setText("Download and use")
        self.action.setEnabled(True)

    # -- acting -------------------------------------------------------------
    def _act(self) -> None:
        look = self._look
        if look is None or not look.up:
            return
        tag = self.selected()
        if look.has(tag):
            self._ready(tag, look.base_url)
            return
        if self._puller is not None and self._puller.isRunning():
            return
        self.action.setEnabled(False)
        self.again.setEnabled(False)
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)         # busy until the first sized line
        self.size.setEnabled(False)
        self.status.setText(f"Pulling <b>{html.escape(tag)}</b>…")
        self._puller = _PullWorker(tag, look.base_url, self)
        self._puller.progress.connect(self._progress)
        self._puller.finished.connect(self._pulled)
        self._puller.start()

    def _progress(self, status: str, done: int, total: int) -> None:
        if total > 0:
            self.bar.setRange(0, 100)
            self.bar.setValue(int(100 * done / total))
            self.bar.setFormat(f"{status} — {done / 1e9:.2f} of {total / 1e9:.2f} GB")
        else:
            self.bar.setFormat(status)

    def _pulled(self) -> None:
        puller = self._puller
        self.bar.setVisible(False)
        self.again.setEnabled(True)
        self.size.setEnabled(True)
        if puller.error:
            self.status.setText(
                f"The pull failed: {html.escape(puller.error)}. The server is "
                f"still there; check again, or run <code>ollama pull "
                f"{html.escape(puller.tag)}</code> in a terminal to see why.")
            self.action.setEnabled(True)
            return
        if self._look is not None:
            self._look.names.append(puller.tag)
        self._ready(puller.tag, puller.base_url)

    def _ready(self, tag: str, base_url: str) -> None:
        """Write the defaults, then tell whoever is listening."""
        from onset_review.assistant_config import write_defaults

        served = f"{base_url.rstrip('/')}/v1"
        try:
            path = write_defaults("ollama", tag, served)
            where = f" Saved as the default in {html.escape(str(path))}."
        except OSError as error:
            where = (f" The default could not be saved ({html.escape(str(error))}); "
                     f"this window uses it anyway.")
        self.status.setText(
            f"<b>{html.escape(tag)}</b> is ready on Ollama at "
            f"{html.escape(base_url)}.{where}")
        self.action.setText("Use this model")
        self.action.setEnabled(True)
        self.modelReady.emit(tag, served)
