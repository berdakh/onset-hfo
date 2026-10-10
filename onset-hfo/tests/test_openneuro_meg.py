"""MEG on OpenNeuro: split FIF recordings, CTF folders and KIT files.

What has to hold, against a small archive served from memory: a FIF recording
split across files is listed once, sized as a whole, and downloaded whole so
MNE reads on from its first part, with the dataset's MEG channel types; a CTF
recording, which is a folder, is listed once by its folder and downloaded as
that folder; a KIT file is listed and downloaded whole; and none of them can
be windowed by byte range.
"""

from __future__ import annotations

import json
import urllib.parse

import pytest

from onset_hfo import openneuro
from onset_hfo.synthetic import as_modality, make_synthetic_recording

DS = "ds999998"
MEG = f"{DS}/sub-01/meg/sub-01_task-rest"
CTF = f"{DS}/sub-02/meg/sub-02_task-rest_meg.ds"


def _split_fif(tmp_path) -> dict[str, bytes]:
    raw = as_modality(make_synthetic_recording(verbose=False, duration_s=30),
                      "meg_grad").raw
    out = tmp_path / "src" / "sub-01_task-rest_meg.fif"
    out.parent.mkdir()
    raw.save(out, split_size="2MB", split_naming="bids", overwrite=True, verbose="ERROR")
    parts = sorted(out.parent.glob("*_split-*_meg.fif"))
    assert len(parts) >= 2, "the recording should need more than one file"
    return {f"{DS}/sub-01/meg/{p.name}": p.read_bytes() for p in parts}


@pytest.fixture
def archive(tmp_path, monkeypatch):
    files = {
        **_split_fif(tmp_path),
        f"{MEG}_channels.tsv": ("name\ttype\tstatus\n" + "".join(
            f"MEG{1000 + 10 * i + 2:04d}\tMEGGRADPLANAR\tgood\n" for i in range(18))).encode(),
        f"{MEG}_meg.json": json.dumps({"PowerLineFrequency": 50}).encode(),
        f"{CTF}/sub-02_task-rest_meg.meg4": b"x" * 3000,
        f"{CTF}/sub-02_task-rest_meg.res4": b"y" * 1000,
        f"{CTF}/sub-02_task-rest_meg.infods": b"z" * 10,
        f"{DS}/sub-03/meg/sub-03_task-rest_meg.con": b"k" * 2000,
        f"{DS}/dataset_description.json": json.dumps({"Name": "MEG archive",
                                                      "License": "CC0"}).encode(),
    }

    def serve(url, byte_range=None, **_):
        if "?" in url:
            prefix = urllib.parse.parse_qs(url.split("?", 1)[1]).get("prefix", [""])[0]
            body = "".join(f"<Contents><Key>{k}</Key><Size>{len(files[k])}</Size></Contents>"
                           for k in sorted(files) if k.startswith(prefix))
            return f"<ListBucketResult>{body}</ListBucketResult>".encode()
        key = urllib.parse.unquote(url.split("openneuro.org/", 1)[1])
        if key not in files:
            raise RuntimeError(f"HTTP 404 for {url}")
        blob = files[key]
        return blob[byte_range[0]: byte_range[1] + 1] if byte_range else blob

    monkeypatch.setattr("onset_hfo.datasets._http_get", serve)
    return {"home": tmp_path / "home", "files": files}


def test_split_fif_ctf_and_kit_are_each_listed_once(archive):
    table = openneuro.list_openneuro(DS, data_home=archive["home"])
    assert list(table["format"]) == ["FIF", "CTF", "KIT"]
    assert set(table["modality"]) == {"meg"} and not table["windowed"].any()
    splits = [k for k in archive["files"] if "_split-" in k]
    assert table.loc[0, "size_mb"] == round(sum(len(archive["files"][k])
                                                for k in splits) / 1e6, 1)
    assert table.loc[1, "path"] == CTF.split("/", 1)[1]
    assert table.loc[1, "size_mb"] == round(4010 / 1e6, 1)


def test_a_split_fif_recording_is_downloaded_whole_and_read_on(archive):
    bunch = openneuro.fetch_openneuro(DS, "01", t_start=5.0, t_stop=20.0,
                                      data_home=archive["home"], verbose=False)
    local = sorted(p.name for p in bunch.local_path.parent.glob("*_meg.fif"))
    assert len(local) == len([k for k in archive["files"] if "_split-" in k])
    assert bunch.modality == "meg" and bunch.line_freq == 50.0
    assert set(bunch.raw.get_channel_types()) == {"grad"}
    assert (bunch.t_start, bunch.t_stop) == (5.0, 20.0), "the window, from the whole file"


def test_a_ctf_recording_comes_down_as_its_folder(archive):
    keys = openneuro._keys(DS, archive["home"])
    folder = archive["home"] / "window"
    folder.mkdir(parents=True)
    path = CTF.split("/", 1)[1]
    local, offset, _ = openneuro._whole_file(DS, path, ".ds", keys, folder, 500.0,
                                             openneuro._Transfer(None, None))
    assert local == folder / "sub-02_task-rest_meg.ds" and local.is_dir()
    assert sorted(p.name for p in local.iterdir()) == [
        "sub-02_task-rest_meg.infods", "sub-02_task-rest_meg.meg4",
        "sub-02_task-rest_meg.res4"]
    assert (local / "sub-02_task-rest_meg.meg4").stat().st_size == 3000


def test_a_file_with_no_unit_is_read_in_the_unit_channels_tsv_gives():
    import mne
    import numpy as np
    import pandas as pd

    raw = mne.io.RawArray(np.full((2, 100), 50.0), mne.create_info(["Fp1", "Fp2"], 100.0, "eeg"),
                          verbose="ERROR")
    raw._orig_units = {"Fp1": "n/a", "Fp2": "uV"}        # Fp2's header did say
    channels = pd.DataFrame({"name": ["Fp1", "Fp2"], "units": ["uV", "uV"]})
    assert openneuro._units_from_channels_tsv(raw, channels) == {"Fp1": "uV"}
    assert raw.get_data()[0, 0] == pytest.approx(50e-6), "50 µV, not 50 V"
    assert raw.get_data()[1, 0] == 50.0, "a channel whose file states its unit is left alone"


def test_a_refused_file_says_why(monkeypatch):
    def refuse(url, byte_range=None, **_):
        raise RuntimeError(f"HTTP 403 for {url}")

    monkeypatch.setattr("onset_hfo.datasets._http_get", refuse)
    with pytest.raises(RuntimeError, match="not public on its S3 bucket"):
        openneuro._get(f"{CTF}/sub-02_task-rest_meg.res4")


def test_an_empty_file_in_a_ctf_folder_is_written_not_fetched(archive, monkeypatch):
    archive["files"][f"{CTF}/BadChannels"] = b""
    keys = openneuro._keys(DS, archive["home"], refresh=True)
    real = openneuro._get

    def no_empty(path, byte_range=None):
        assert not path.endswith("BadChannels"), "S3 refuses some empty files"
        return real(path, byte_range)

    monkeypatch.setattr(openneuro, "_get", no_empty)
    folder = archive["home"] / "window2"
    folder.mkdir(parents=True)
    local, _, _ = openneuro._whole_file(DS, CTF.split("/", 1)[1], ".ds", keys, folder, 500.0,
                                        openneuro._Transfer(None, None))
    assert (local / "BadChannels").exists() and (local / "BadChannels").stat().st_size == 0
