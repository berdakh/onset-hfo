"""File → Open from OpenNeuro…: list a dataset, download a window, confirm it.

What has to hold, against the small archive `test_openneuro` serves from
memory: listing fills the table and says what the dataset is and how to cite
it; a bad id says so and offers nothing to download; downloading a window
hands over the fetched recording; and the confirmation that follows is the
import dialog with the dataset's channel types, mains frequency and window
already in it, so the request it makes reads exactly that window.
"""

from __future__ import annotations

import os

import pytest

widgets = pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from test_openneuro import DS, serve_archive  # noqa: E402

from onset_review.openneurodialog import OpenNeuroDialog, import_dialog_for  # noqa: E402


@pytest.fixture
def archive(tmp_path, monkeypatch):
    return serve_archive(tmp_path, monkeypatch)


@pytest.fixture
def qapp():
    yield widgets.QApplication.instance() or widgets.QApplication([])


def test_listing_fills_the_table_and_says_what_the_dataset_is(qapp, archive):
    dialog = OpenNeuroDialog(data_home=archive["home"], dataset=DS)
    assert not dialog.fetch_button.isEnabled()
    assert dialog.list_recordings()
    assert dialog.table.rowCount() == 3
    assert [dialog.table.item(r, 6).text() for r in range(3)] == ["BrainVision", "EDF", "FIF"]
    assert "A test archive" in dialog.about.text() and "licence CC0" in dialog.about.text()
    assert "doi:10.0/test" in dialog.about.text()
    assert dialog.fetch_button.isEnabled() and dialog.selected()["subject"] == "sub-01"
    dialog.table.selectRow(2)
    assert "comes whole" in dialog.status.text()


def test_a_bad_id_says_so_and_offers_nothing(qapp, archive):
    dialog = OpenNeuroDialog(data_home=archive["home"], dataset="nope")
    assert not dialog.list_recordings()
    assert "not an OpenNeuro dataset id" in dialog.about.text()
    assert dialog.table.rowCount() == 0 and not dialog.fetch_button.isEnabled()


def test_a_window_is_downloaded_and_confirmed_with_the_datasets_own_settings(qapp, archive):
    dialog = OpenNeuroDialog(data_home=archive["home"], dataset=DS)
    dialog.list_recordings()
    dialog.t_start.setValue(2.0)
    dialog.t_stop.setValue(5.0)
    assert dialog.fetch()
    bunch = dialog.bunch
    assert bunch.subject == "sub-01" and bunch.t_start == 2.0 and bunch.t_stop == 5.0

    confirm = import_dialog_for(bunch)
    # A2 is marked bad in channels.tsv, so it is not analysed.
    assert confirm.channel_types() == {"A1": "seeg", "A2": "misc", "EKG": "misc"}
    assert confirm.line_freq.currentData() == 60.0
    assert confirm.subject.text() == f"{DS} sub-01 ictal from 2 s"
    request = confirm.request()
    assert request.path == bunch.local_path
    # The window file starts where the window does, so it is read whole.
    assert (request.t_start, request.t_stop) == (0.0, 3.0)
    assert request.line_freq == 60.0


def test_an_edf_window_is_offset_within_its_records(qapp, archive):
    dialog = OpenNeuroDialog(data_home=archive["home"], dataset=DS)
    dialog.list_recordings()
    dialog.table.selectRow(1)
    dialog.t_start.setValue(2.5)
    dialog.t_stop.setValue(4.5)
    assert dialog.fetch()
    request = import_dialog_for(dialog.bunch).request()
    # The file holds whole records from 2 s, so the window is 0.5–2.5 s into it.
    assert (request.t_start, request.t_stop) == (0.5, 2.5)


def test_the_catalogue_is_searched_as_you_type_and_chooses_the_dataset(qapp, archive):
    from onset_hfo import openneuro

    openneuro.build_catalogue(out=archive["home"] / "openneuro_catalogue.csv", workers=1,
                              progress=lambda *_: None)
    dialog = OpenNeuroDialog(data_home=archive["home"])
    assert dialog.found.rowCount() == 1            # iEEG first, and the test archive is iEEG
    assert dialog.found.item(0, 1).text() == "iEEG"
    assert "1 of 1 OpenNeuro datasets" in dialog.found_count.text()
    dialog.search.setText("no such words")
    assert dialog.found.rowCount() == 0
    dialog.search.setText("test archive")
    dialog.found.selectRow(0)
    assert dialog.dataset.text() == DS
    assert dialog.list_recordings() and dialog.table.rowCount() == 3


def test_listing_reaches_the_network_even_when_the_window_is_offline(qapp, archive,
                                                                     monkeypatch):
    from onset_hfo import openneuro
    from onset_hfo.datasets import OFFLINE_ENV, offline

    monkeypatch.setenv(OFFLINE_ENV, "1")
    seen = []
    real = openneuro.describe_openneuro
    monkeypatch.setattr(openneuro, "describe_openneuro",
                        lambda *a, **k: seen.append(offline()) or real(*a, **k))
    dialog = OpenNeuroDialog(data_home=archive["home"], dataset=DS)
    assert dialog.list_recordings()
    assert seen == [False], "the asking lifts the refusal for its own calls"
    assert offline(), "and puts it back afterwards"
