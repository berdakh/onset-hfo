"""Fetching any OpenNeuro recording, as scikit-learn fetches a dataset.

What has to hold, against a small archive served from memory: the listing
finds every continuous recording and says which can be windowed; the
description carries the name, licence and citation; a window of a BrainVision
or EDF recording is fetched by byte range and is exactly the samples asked
for, in original-recording seconds; a format that cannot be windowed comes
whole, and is refused over `max_mb` without leaving anything behind; the
sidecars come by BIDS inheritance (a mains frequency at the dataset's top
level counts); a non-ictal task's "start" is not read as a seizure; and a
second fetch, or one with `download_if_missing=False`, needs no network.
"""

from __future__ import annotations

import json
import re
import struct
import urllib.parse

import mne
import numpy as np
import pytest

from onset_hfo import openneuro

DS = "ds000001"
SFREQ_BV, N_BV = 1000.0, 10_000
SFREQ_EDF, N_RECORDS = 100, 10


def _brainvision(stem: str) -> dict[str, bytes]:
    name = stem.rsplit("/", 1)[-1]
    header = (
        "Brain Vision Data Exchange Header File Version 1.0\n\n[Common Infos]\n"
        f"Codepage=UTF-8\nDataFile={name}.eeg\nMarkerFile={name}.vmrk\n"
        "DataFormat=BINARY\nDataOrientation=MULTIPLEXED\nNumberOfChannels=3\n"
        f"SamplingInterval={1e6 / SFREQ_BV:g}\n\n[Binary Infos]\nBinaryFormat=IEEE_FLOAT_32\n\n"
        "[Channel Infos]\nCh1=A1,,1,µV\nCh2=A2,,1,µV\nCh3=EKG,,1,µV\n")
    k = np.arange(N_BV, dtype=np.float32)
    # Each sample says where it is: channel c holds k + 100000 c (µV).
    data = np.stack([k, k + 1e5, k + 2e5], axis=1).astype("<f4")
    return {f"{stem}.vhdr": header.encode(), f"{stem}.eeg": data.tobytes(),
            f"{stem}.vmrk": b"Brain Vision Data Exchange Marker File, Version 1.0\n"}


def _edf() -> bytes:
    """Two signals at 100 Hz in 1 s records; sample k of channel c is k + 1000 c (0.1 µV)."""
    def field(value, width):
        return f"{value:<{width}}"[:width].encode()

    n_signals, header_bytes = 2, 256 * 3
    head = (field("0", 8) + field("X X X X", 80) + field("Startdate X X X X", 80)
            + field("01.01.20", 8) + field("00.00.00", 8) + field(header_bytes, 8)
            + field("", 44) + field(N_RECORDS, 8) + field(1, 8) + field(n_signals, 4))
    labels = ["B1", "B2"]
    per = b"".join(field(x, 16) for x in labels) + field("", 80) * 2 + field("uV", 8) * 2
    per += field("-3276.8", 8) * 2 + field("3276.7", 8) * 2
    per += field("-32768", 8) * 2 + field("32767", 8) * 2 + field("", 80) * 2
    per += field(SFREQ_EDF, 8) * 2 + field("", 32) * 2
    body = b""
    for record in range(N_RECORDS):
        for c in range(n_signals):
            k = np.arange(record * SFREQ_EDF, (record + 1) * SFREQ_EDF) + 1000 * c
            body += struct.pack(f"<{SFREQ_EDF}h", *k.astype(int))
    return head + per + body


def _fif(tmp_path) -> bytes:
    info = mne.create_info(["C1", "C2"], 200.0, "seeg")
    raw = mne.io.RawArray(np.arange(2 * 2000, dtype=float).reshape(2, 2000) * 1e-6, info,
                          verbose="ERROR")
    path = tmp_path / "x_ieeg.fif"
    raw.save(path, verbose="ERROR")
    return path.read_bytes()


@pytest.fixture
def archive(tmp_path, monkeypatch):
    return serve_archive(tmp_path, monkeypatch)


def serve_archive(tmp_path, monkeypatch) -> dict:
    """Serve a small BIDS archive from memory through the package's one HTTP call."""
    bv = f"{DS}/sub-01/ses-1/ieeg/sub-01_ses-1_task-ictal_run-01_ieeg"
    edf = f"{DS}/sub-02/ieeg/sub-02_task-rest_ieeg"
    files = {
        **_brainvision(bv),
        f"{edf}.edf": _edf(),
        f"{DS}/sub-03/ieeg/sub-03_task-rest_ieeg.fif": _fif(tmp_path),
        f"{DS}/dataset_description.json": json.dumps({
            "Name": "A test archive", "License": "CC0", "Authors": ["A. Author", "B. Author"],
            "DatasetDOI": "doi:10.0/test"}).encode(),
        f"{DS}/README": b"Recordings for a test.",
        f"{DS}/participants.tsv": b"participant_id\tage\nsub-01\t30\nsub-02\t40\n",
        # The mains frequency at the top level, for every ictal recording.
        f"{DS}/task-ictal_ieeg.json": json.dumps({"PowerLineFrequency": 60}).encode(),
        f"{bv}".replace("_ieeg", "_channels.tsv"):
            b"name\ttype\tstatus\nA1\tSEEG\tgood\nA2\tSEEG\tbad\nEKG\tECG\tgood\n",
        f"{bv}".replace("_ieeg", "_events.tsv"):
            b"onset\tduration\ttrial_type\n3.0\t0\tsz onset\n8.0\t0\tsz offset\n",
        f"{DS}/sub-01/ses-1/ieeg/sub-01_ses-1_space-ACPC_electrodes.tsv":
            b"name\tx\ty\tz\nA1\t1\t2\t3\nA2\t4\t5\t6\n",
        f"{edf}".replace("_ieeg", "_events.tsv"):
            b"onset\tduration\ttrial_type\n1.0\t0\tstart task\n",
        f"{edf}.json".replace("_ieeg.json", "_ieeg.json"):
            json.dumps({"PowerLineFrequency": 50}).encode(),
    }
    asked = []

    def serve(url, byte_range=None, **_):
        asked.append((url, byte_range))
        if "?" in url:
            query = urllib.parse.parse_qs(url.split("?", 1)[1])
            prefix = query.get("prefix", [""])[0]
            keys = [k for k in sorted(files) if k.startswith(prefix)]
            if query.get("delimiter"):     # S3's folders: one level, then the files
                folders = sorted({prefix + k[len(prefix):].split("/", 1)[0] + "/"
                                  for k in keys if "/" in k[len(prefix):]})
                keys = [k for k in keys if "/" not in k[len(prefix):]]
                body = "".join(f"<CommonPrefixes><Prefix>{f}</Prefix></CommonPrefixes>"
                               for f in folders)
            else:
                body = ""
            body += "".join(f"<Contents><Key>{k}</Key><Size>{len(files[k])}</Size></Contents>"
                            for k in keys)
            return f"<ListBucketResult>{body}</ListBucketResult>".encode()
        key = urllib.parse.unquote(url.split("openneuro.org/", 1)[1])
        if key not in files:
            raise RuntimeError(f"HTTP 404 for {url}")
        blob = files[key]
        return blob[byte_range[0]: byte_range[1] + 1] if byte_range else blob

    monkeypatch.setattr("onset_hfo.datasets._http_get", serve)
    return {"home": tmp_path / "home", "asked": asked, "files": files}


def test_the_listing_finds_every_recording_and_what_can_be_windowed(archive):
    table = openneuro.list_openneuro(DS, data_home=archive["home"])
    assert list(table["subject"]) == ["sub-01", "sub-02", "sub-03"]
    assert list(table["format"]) == ["BrainVision", "EDF", "FIF"]
    assert list(table["windowed"]) == [True, True, False]
    assert table.loc[0, "task"] == "ictal" and table.loc[0, "session"] == "ses-1"
    assert table.loc[0, "size_mb"] == round(N_BV * 3 * 4 / 1e6, 1)   # the .eeg, not the header
    assert len(openneuro.list_openneuro(DS, subject="02", data_home=archive["home"])) == 1


def test_the_description_says_what_the_dataset_is_and_how_to_cite_it(archive):
    about = openneuro.describe_openneuro(DS, data_home=archive["home"])
    assert about.name == "A test archive" and about.license == "CC0"
    assert about.citation == "A. Author, B. Author. A test archive. OpenNeuro ds000001. " \
                             "doi:10.0/test"
    assert about.subjects == ["sub-01", "sub-02", "sub-03"]
    assert "Recordings for a test." in about.DESCR and "Licence: CC0" in about.DESCR


def test_a_brainvision_window_is_fetched_by_byte_range_and_is_exact(archive):
    bunch = openneuro.fetch_openneuro(DS, "01", t_start=2.0, t_stop=5.0,
                                      data_home=archive["home"], verbose=False)
    first, last = int(2.0 * SFREQ_BV), int(5.0 * SFREQ_BV)
    assert bunch.data.shape == (3, last - first)
    assert np.allclose(bunch.data[0] * 1e6, np.arange(first, last))
    assert np.allclose(bunch.data[2, :3] * 1e6, 2e5 + np.arange(first, first + 3))
    assert bunch.times[0] == pytest.approx(2.0) and bunch.recording.t_offset == 2.0
    signal = [r for u, r in archive["asked"] if u.endswith(".eeg")]
    assert signal == [(first * 12, last * 12 - 1)], "only the window's bytes"
    record = bunch.recording
    assert record.raw.get_channel_types() == ["seeg", "seeg", "ecg"]
    assert record.bads == ["A2"] and bunch.line_freq == 60.0   # from the top-level sidecar
    assert record.seizure == (3.0, 8.0) and record.source == f"openneuro:{DS}"
    assert list(bunch.electrodes["name"]) == ["A1", "A2"] and bunch.participant["age"] == 30
    assert bunch.citation in " ".join(record.notes)


def test_an_edf_window_takes_only_its_records_and_is_cut_exactly(archive):
    bunch = openneuro.fetch_openneuro(DS, subject="sub-02", t_start=2.5, t_stop=4.5,
                                      data_home=archive["home"], verbose=False)
    assert bunch.data.shape == (2, 200)
    assert np.allclose(bunch.data[0] * 1e6, 0.1 * np.arange(250, 450), atol=1e-3)
    assert bunch.times[0] == pytest.approx(2.5)
    record_bytes = 2 * 2 * SFREQ_EDF
    ranges = [r for u, r in archive["asked"] if u.endswith(".edf") and r and r[0] >= 768]
    assert ranges == [(768 + 2 * record_bytes, 768 + 5 * record_bytes - 1)]
    assert bunch.line_freq == 50.0
    assert bunch.recording.seizure == (None, None), "a task's start is not a seizure"


def test_a_format_that_cannot_be_windowed_comes_whole_within_max_mb(archive):
    with pytest.raises(ValueError, match="downloaded whole"):
        openneuro.fetch_openneuro(DS, "03", t_start=1, t_stop=4, max_mb=0.001,
                                  data_home=archive["home"], verbose=False)
    assert not any((archive["home"] / DS).glob("sub-03*/*")), "nothing half-written is kept"
    bunch = openneuro.fetch_openneuro(DS, "03", t_start=1, t_stop=4,
                                      data_home=archive["home"], verbose=False)
    assert bunch.data.shape == (2, 600) and bunch.times[0] == pytest.approx(1.0)
    assert bunch.line_freq is None and "50 Hz assumed" in " ".join(bunch.recording.notes)


def test_the_data_home_serves_a_second_fetch_without_the_network(archive):
    kwargs = dict(t_start=1.0, t_stop=2.0, data_home=archive["home"], verbose=False)
    first = openneuro.fetch_openneuro(DS, "01", **kwargs)
    archive["asked"].clear()
    again = openneuro.fetch_openneuro(DS, "01", download_if_missing=False, **kwargs)
    assert archive["asked"] == [] and np.array_equal(first.data, again.data)
    assert again.raw is again.recording.raw and again.ch_names == ["A1", "A2", "EKG"]
    openneuro.clear_data_home(archive["home"])
    with pytest.raises(OSError, match="download_if_missing"):
        openneuro.fetch_openneuro(DS, "01", download_if_missing=False, **kwargs)


def test_asking_for_what_is_not_there_names_what_is(archive):
    with pytest.raises(ValueError, match=r"subject='sub-09'.*sub-01, sub-02, sub-03"):
        openneuro.fetch_openneuro(DS, "09", data_home=archive["home"], verbose=False)
    with pytest.raises(ValueError, match="not an OpenNeuro dataset id"):
        openneuro.list_openneuro("dataset1", data_home=archive["home"])
    with pytest.raises(ValueError, match="t_stop must be greater"):
        openneuro.fetch_openneuro(DS, "01", t_start=5, t_stop=5, data_home=archive["home"])


def test_a_sidecar_is_found_by_bids_inheritance():
    keys = {"task-ictal_ieeg.json": 1, "sub-01/sub-01_task-ictal_ieeg.json": 1,
            "sub-01/ses-1/ieeg/sub-01_ses-1_task-rest_ieeg.json": 1}
    stem = "sub-01/ses-1/ieeg/sub-01_ses-1_task-ictal_run-01_ieeg"
    assert openneuro._sidecar_key(keys, stem, "ieeg.json") == \
        "sub-01/sub-01_task-ictal_ieeg.json", "the most specific that applies"
    del keys["sub-01/sub-01_task-ictal_ieeg.json"]
    assert openneuro._sidecar_key(keys, stem, "ieeg.json") == "task-ictal_ieeg.json"
    assert re.search("rest", str(openneuro._sidecar_key(keys, stem, "ieeg.json"))) is None


def test_a_bunch_is_a_dict_with_attributes():
    bunch = openneuro.Bunch(a=1)
    bunch.b = 2
    assert bunch.a == 1 and bunch["b"] == 2 and set(dir(bunch)) == {"a", "b"}
    with pytest.raises(AttributeError):
        _ = bunch.c


def test_the_command_line_lists_describes_and_fetches(archive, capsys):
    from onset_hfo.cli import main

    home = str(archive["home"])
    assert main(["openneuro", "list", DS, "--data-home", home]) == 0
    listed = capsys.readouterr().out
    assert "BrainVision" in listed and "sub-03" in listed
    assert main(["openneuro", "describe", DS, "--data-home", home]) == 0
    assert "A test archive" in capsys.readouterr().out
    assert main(["openneuro", "fetch", DS, "--subject", "01", "--t-start", "1",
                 "--t-stop", "2", "--data-home", home]) == 0
    assert capsys.readouterr().out.strip().endswith("_ieeg.vhdr")
    assert main(["openneuro", "list", "nope", "--data-home", home]) == 1


def test_the_catalogue_finds_datasets_by_name_and_kind(archive, monkeypatch):
    table = openneuro.build_catalogue(out=archive["home"] / "openneuro_catalogue.csv",
                                      workers=2, progress=lambda *_: None)
    assert table.to_dict("records") == [{
        "dataset_id": DS, "name": "A test archive", "modalities": "ieeg", "subjects": 3,
        "license": "CC0", "doi": "10.0/test"}]
    # The refreshed copy in the data home is preferred to the bundled one.
    assert list(openneuro.catalogue(data_home=archive["home"])["dataset_id"]) == [DS]
    assert len(openneuro.catalogue(search="test ARCHIVE", data_home=archive["home"])) == 1
    assert len(openneuro.catalogue(search="test sleep", data_home=archive["home"])) == 0
    assert len(openneuro.catalogue(modality="eeg", data_home=archive["home"])) == 0


def test_the_bundled_catalogue_holds_the_archives_this_project_uses(tmp_path):
    bundled = openneuro.catalogue(data_home=tmp_path)
    assert len(bundled) > 100
    assert set(bundled.columns) == set(openneuro.CATALOGUE_COLUMNS)
    assert bundled["dataset_id"].is_unique
    for dataset in ("ds003029", "ds003498", "ds004100", "ds003688"):
        row = bundled[bundled["dataset_id"] == dataset]
        assert len(row) == 1 and "ieeg" in row.iloc[0]["modalities"], dataset
    assert set(openneuro.catalogue(modality="ieeg", data_home=tmp_path)["modalities"]
               .str.contains("ieeg")) == {True}
    assert "HUP" in openneuro.catalogue(search="hup", data_home=tmp_path).iloc[0]["name"]


def test_the_network_is_allowed_only_inside_the_asking(monkeypatch):
    from onset_hfo.datasets import OFFLINE_ENV, offline

    monkeypatch.setenv(OFFLINE_ENV, "1")
    with openneuro.allow_network():
        assert not offline()
    assert offline()
    monkeypatch.delenv(OFFLINE_ENV)
    with openneuro.allow_network():
        assert not offline()
    assert not offline()


def test_the_command_line_searches_the_catalogue(archive, capsys):
    from onset_hfo.cli import main

    openneuro.build_catalogue(out=archive["home"] / "openneuro_catalogue.csv", workers=1,
                              progress=lambda *_: None)
    home = str(archive["home"])
    assert main(["openneuro", "catalogue", "--search", "test", "--data-home", home]) == 0
    assert DS in capsys.readouterr().out
    assert main(["openneuro", "list", "--data-home", home]) == 1
    assert "needs a dataset id" in capsys.readouterr().err
