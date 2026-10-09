"""A patient's monitoring as a case: BIDS-iEEG, bridges from clinical
formats, marks, and an overview. Qt-free.

* `bids` -- the layout, written and read;
* `bridges` -- clinical files into a case, without what names the patient;
* `annotations` -- the vocabulary marks are read in;
* `model` -- the case itself: recordings, steps, audit log;
* `overview` -- hours at a glance, sampled.

``python -m onset_hfo.case`` converts from the terminal.
"""
