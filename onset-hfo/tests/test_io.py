"""Reading a recording off this machine, checked without a display.

`onset_hfo.io` is the answer to "should this project invent its own file
format": no, because `Recording.raw` is already an `mne.io.BaseRaw` and MNE
already ships thirty readers, several of which (Persyst, Nihon Kohden, Nicolet,
Blackrock, MEF3) are exactly the clinical systems an epilepsy unit exports
from. So this module is a table of extensions and a function that crops before
it loads, and these tests are about the three things such a module can still
get wrong:

* **picking the wrong reader** -- `.eeg` is BrainVision's data file *and* Nihon
  Kohden's whole recording, and guessing wrong gives a confident error about
  the wrong format;
* **believing the file about its channels** -- a clinical export declares
  everything `eeg`, this pipeline analyses anything typed eeg/ecog/seeg, and
  the result is rates for a scalp montage with nothing anywhere saying so;
* **losing where the window came from** -- every time this software reports is
  a time in the original recording, so a cropped import has to carry its
  offset or the event list points at the wrong minute of a night.

Everything runs on files written in `tmp_path`, so none of it needs a cached
slice, a network or a clinical recording. FIF is used for the round trips
because MNE can always write it; the export-only formats are checked for
detection, which is all that can be checked without their writer installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from onset_hfo.io import (
    FORMATS,
    channel_overview,
    detect_format,
    file_filter,
    open_recording,
    recording_info,
    recording_root,
    supported_suffixes,
)

SFREQ = 2000.0


@pytest.fixture(scope="module")
def fif(tmp_path_factory):
    """A 20 s, 6-channel file on disk, declared `eeg` as a clinical export is."""
    import mne

    rng = np.random.default_rng(3)
    names = ["AR1", "AR2", "AR3", "HL1", "EKG", "DC1"]
    data = rng.normal(0.0, 2e-5, size=(len(names), int(20 * SFREQ)))
    info = mne.create_info(names, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    path = tmp_path_factory.mktemp("recordings") / "study-001_raw.fif"
    raw.save(path, overwrite=True, verbose="ERROR")
    return path


# -- which reader --------------------------------------------------------

def test_every_format_in_the_table_names_a_reader_that_exists():
    """A typo in the table is a format that fails only when someone uses it."""
    for fmt in FORMATS:
        assert fmt.function is not None, fmt.name
        assert fmt.suffixes, fmt.name
        assert all(s.startswith(".") for s in fmt.suffixes), fmt.name


def test_the_file_dialog_offers_exactly_what_the_importer_handles():
    """Built from the table, so the dialog cannot drift from the readers."""
    filters = file_filter()
    for suffix in supported_suffixes():
        assert f"*{suffix}" in filters
    assert len(supported_suffixes()) == len(set(supported_suffixes()))


@pytest.mark.parametrize("name,reader", [
    ("a.vhdr", "read_raw_brainvision"),
    ("a.edf", "read_raw_edf"),
    ("a.bdf", "read_raw_bdf"),
    ("a.lay", "read_raw_persyst"),
    ("a.data", "read_raw_nicolet"),
    ("a.dat", "read_raw_curry"),
    ("a.ns3", "read_raw_nsx"),
    ("a.fif", "read_raw_fif"),
    ("a.fif.gz", "read_raw_fif"),
])
def test_the_extension_picks_the_reader(name, reader):
    fmt = detect_format(name)
    assert fmt is not None and fmt.reader == reader


def test_a_bare_eeg_file_is_nihon_kohden_and_one_beside_a_header_is_brainvision(
        tmp_path):
    """The one genuine ambiguity in the table, and the only one worth code.

    BrainVision writes the signal to `.eeg` beside a `.vhdr` header; Nihon
    Kohden writes the whole recording to `.eeg`. Reading a BrainVision data
    file with the Nihon reader does not produce a subtly wrong recording, it
    produces a confident error about the wrong format -- which is what sends a
    reviewer looking for a corrupt file.
    """
    lonely = tmp_path / "nihon.eeg"
    lonely.write_bytes(b"")
    assert detect_format(lonely).reader == "read_raw_nihon"

    paired = tmp_path / "brainvision.eeg"
    paired.write_bytes(b"")
    (tmp_path / "brainvision.vhdr").write_text("Brain Vision Data Exchange\n")
    assert detect_format(paired).reader == "read_raw_brainvision"


def test_a_recording_that_is_a_folder_is_recognised_as_one(tmp_path):
    """Two formats are directories, and they announce it differently.

    MEF3 names the folder `.mefd`, so the extension rule finds it. Neuralynx
    writes one `.ncs` per channel into a folder named whatever the recording
    was called, so it is recognised by what is inside; and a reviewer who
    picks one channel file in a file dialog has picked the recording, so that
    resolves to the folder too.
    """
    neuralynx = tmp_path / "2026-10-03_night"
    neuralynx.mkdir()
    (neuralynx / "CSC1.ncs").write_bytes(b"")
    assert detect_format(neuralynx).reader == "read_raw_neuralynx"
    assert detect_format(neuralynx / "CSC1.ncs").reader == "read_raw_neuralynx"

    mef = tmp_path / "study-001.mefd"
    (mef / "a.timd").mkdir(parents=True)
    assert detect_format(mef).reader == "read_raw_mef"
    # And anything picked from inside the bundle resolves back up to it.
    assert recording_root(mef / "a.timd") == mef
    assert detect_format(mef / "a.timd").reader == "read_raw_mef"

    ordinary = tmp_path / "an_ordinary_folder"
    ordinary.mkdir()
    assert detect_format(ordinary) is None
    assert recording_root(tmp_path / "plain.edf") == tmp_path / "plain.edf"


def test_an_unreadable_extension_says_what_is_readable(tmp_path):
    """The error is the only documentation most people will read."""
    bad = tmp_path / "recording.xyz"
    bad.write_bytes(b"")
    assert detect_format(bad) is None
    with pytest.raises(ValueError, match="not a format this software reads"):
        open_recording(bad)
    with pytest.raises(ValueError, match="BrainVision"):
        channel_overview(bad)


def test_a_format_needing_more_than_a_filename_says_so(tmp_path):
    """Nicolet cannot be read without being told the channel type.

    Its header does not carry one. Raising with the reason beats MNE's own
    `TypeError: missing argument`, which in a clinic reads as a bug in this
    software.
    """
    nicolet = tmp_path / "recording.data"
    nicolet.write_bytes(b"")
    with pytest.raises(ValueError, match="ch_type"):
        open_recording(nicolet)


# -- what the file says about itself -------------------------------------

def test_the_overview_shows_the_file_s_own_answer_and_the_suggestion(fif):
    """Both columns, because the whole screen is about the gap between them."""
    frame = channel_overview(fif)
    assert list(frame.columns) == ["name", "declared", "suggested", "analysed"]
    assert len(frame) == 6
    assert set(frame["declared"]) == {"eeg"}        # as a clinical export does
    assert set(frame["suggested"]) == {"seeg"}      # what this software assumes
    assert frame.loc[frame["name"] == "EKG", "analysed"].item() is False


def test_the_header_gives_the_rate_and_length_without_reading_signal(fif):
    info = recording_info(fif)
    assert info["sfreq"] == SFREQ
    assert info["n_channels"] == 6
    assert info["duration"] == pytest.approx(20.0, abs=0.01)
    assert info["reader"] == "read_raw_fif"


# -- opening a window ----------------------------------------------------

def test_a_window_is_cropped_and_carries_its_offset(fif):
    """Every time this software reports is a time in the original recording.

    An import that forgot its offset would list an event at 3 s when it is at
    8 s of the night, which is worse than refusing to import at all: the
    numbers still look right.
    """
    record = open_recording(fif, t_start=5.0, t_stop=12.0, subject="study-001")
    assert record.t_offset == 5.0
    assert record.raw.n_times == int(7 * SFREQ)
    assert record.subject == "study-001"
    assert record.source.endswith("_raw.fif")
    assert record.dataset_id == ""          # no accession; it is a local file


def test_the_reviewer_s_channel_types_are_applied_before_anything_sees_them(fif):
    record = open_recording(
        fif, t_stop=5.0,
        channel_types={"AR1": "seeg", "AR2": "seeg", "AR3": "seeg",
                       "HL1": "seeg", "EKG": "ecg", "DC1": "misc"})
    types = dict(zip(record.raw.ch_names, record.raw.get_channel_types(),
                     strict=True))
    assert types["AR1"] == "seeg" and types["EKG"] == "ecg"
    assert types["DC1"] == "misc"


def test_a_name_that_is_not_in_the_file_is_ignored_rather_than_fatal(fif):
    """A saved type map outliving the montage it was made for is not a crash."""
    record = open_recording(fif, t_stop=5.0,
                            channel_types={"AR1": "seeg", "NOT-A-CHANNEL": "seeg"})
    assert "NOT-A-CHANNEL" not in record.raw.ch_names
    assert record.raw.get_channel_types()[0] == "seeg"


def test_the_mains_frequency_is_recorded_as_a_choice_not_a_reading(fif):
    """Notching the wrong mains leaves the interference *and* cuts a hole."""
    record = open_recording(fif, t_stop=5.0, line_freq=60.0)
    assert record.line_freq == 60.0
    assert record.raw.info["line_freq"] == 60.0
    assert any("not read from the file" in note for note in record.notes)


def test_an_empty_window_is_refused_with_the_recording_s_length(fif):
    with pytest.raises(ValueError, match="20 s long"):
        open_recording(fif, t_start=40.0, t_stop=60.0)


def test_asking_past_the_end_takes_what_there_is(fif):
    """A reviewer typing 0–60 on a 20 s file wants the 20 s, not an error."""
    record = open_recording(fif, t_start=0.0, t_stop=600.0)
    assert record.raw.n_times == int(20 * SFREQ)


def test_a_window_with_nothing_intracranial_says_so_in_its_own_provenance(fif):
    """The failure this whole module is arranged around.

    Nothing downstream can tell a scalp montage from a depth electrode, so if
    the types are wrong the review is wrong in a way that looks entirely
    normal. The warning travels in the recording's notes, which means it
    reaches the exported report rather than only a dialog nobody kept.
    """
    record = open_recording(fif, t_stop=5.0,
                            channel_types={n: "eeg" for n in
                                           ("AR1", "AR2", "AR3", "HL1",
                                            "EKG", "DC1")})
    said = [note for note in record.notes if note.startswith("no channel in this window")]
    assert said and "analysed as scalp EEG" in said[0]
    assert "If it is in fact intracranial" in said[0], "the scalp reading is said, not assumed"

    good = open_recording(fif, t_stop=5.0, channel_types={"AR1": "seeg"})
    assert not any(note.startswith("no channel in this window") for note in good.notes)


def test_the_provenance_says_an_imported_file_brings_no_ground_truth(fif):
    record = open_recording(fif, t_stop=5.0)
    joined = " ".join(record.notes)
    assert "no expert markings" in joined
    assert "mne.io.read_raw_fif" in joined


# -- and all the way through the pipeline --------------------------------

def test_an_imported_window_runs_the_same_pipeline_as_an_archive_one(fif):
    """The point of not inventing a format: nothing downstream is different.

    `session_from_recording` is the same call the archive loader makes, so if
    this produces a session with findings then preprocessing, the detectors,
    the montage and the report all accepted the imported recording unchanged.
    """
    from onset_review.session import ReviewRequest, session_from_recording

    record = open_recording(
        fif, t_start=2.0, t_stop=18.0, subject="study-001",
        channel_types={n: "seeg" for n in ("AR1", "AR2", "AR3", "HL1")})
    request = ReviewRequest(dataset="", subject="study-001", path=fif,
                            t_start=2.0, t_stop=18.0)
    review = session_from_recording(record, request)

    assert review.request.imported is True
    assert not review.findings.empty
    assert review.resection is None          # no archive, so no resected zone
    assert review.has_expert is False
    assert fif.name in review.request.label()


def test_the_report_names_the_file_rather_than_an_empty_accession(fif, tmp_path):
    """An empty Dataset cell reads as a missing value, not an absent one."""
    from onset_review.report import review_markdown
    from onset_review.session import ReviewRequest, session_from_recording

    record = open_recording(fif, t_stop=10.0, subject="study-001",
                            channel_types={n: "seeg" for n in
                                           ("AR1", "AR2", "AR3", "HL1")})
    review = session_from_recording(
        record, ReviewRequest(dataset="", subject="study-001", path=fif,
                              t_start=0.0, t_stop=10.0))
    text = review_markdown(review)
    assert f"local file `{fif.name}`" in text
    assert "no expert markings" in text


@pytest.mark.parametrize("module", ["onset_hfo.io"])
def test_the_importer_core_needs_no_qt(module):
    """`onset_hfo.io` is imported by the export path, which runs without Qt."""
    import ast
    import pathlib

    import onset_hfo.io as target

    tree = ast.parse(pathlib.Path(target.__file__).read_text())
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        for name in names:
            assert name.split(".")[0] not in {"qtpy", "PySide6", "PyQt5",
                                              "PyQt6", "onset_review"}, name
