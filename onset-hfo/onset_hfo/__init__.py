"""Onset-HFO: an AI-assisted research tool for intracranial EEG -- detecting
high-frequency oscillations and interictal epileptiform discharges, and finding
where seizures start, in public recordings or your own.

Two deliberately separate halves:

* ``onset_hfo``  -- the signal-processing pipeline. Deterministic, testable,
  no language model anywhere near the numbers.
* ``onset_agent`` -- an agent built on an open-weight LLM that can only read
  what the pipeline produced, must cite it, and refuses clinical advice.

Start with ``docs/README.md`` or ``notebooks/01_hfo_detection_quickstart.ipynb``.
"""

from onset_hfo.config import BANDS, PIPELINE_VERSION, PipelineConfig  # noqa: F401

__version__ = PIPELINE_VERSION
__author__ = "Berdakh Abibullaev"
__all__ = ["BANDS", "PipelineConfig", "PIPELINE_VERSION"]
