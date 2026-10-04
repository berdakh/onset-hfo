"""Shared fixtures. Everything here is offline: no download, no model, no GPU.

"Offline" is enforced, not intended. :func:`_no_network` sets
``ONSET_HFO_OFFLINE`` for the whole session, and the single choke point in
``onset_hfo.datasets`` refuses every outbound request. Before that guard
existed the suite quietly downloaded the clinical spreadsheet on every run --
harmless in itself, but it made a green CI depend on S3 being reachable, and
it made the workflow's "never downloads data" comment untrue.
"""

from __future__ import annotations

import os

import pytest

from onset_hfo.datasets import OFFLINE_ENV


@pytest.fixture(scope="session", autouse=True)
def _no_network():
    """Refuse every outbound request for the whole test session.

    Autouse and session-scoped on purpose: a future test that reaches for the
    archive should fail loudly here rather than pass on a machine with
    network and hang on one without.
    """
    previous = os.environ.get(OFFLINE_ENV)
    os.environ[OFFLINE_ENV] = "1"
    yield
    if previous is None:
        os.environ.pop(OFFLINE_ENV, None)
    else:
        os.environ[OFFLINE_ENV] = previous

from onset_hfo.pipeline import run_pipeline
from onset_hfo.preprocess import prepare
from onset_hfo.synthetic import make_synthetic_recording


@pytest.fixture(scope="session", autouse=True)
def _reads_go_to_a_temporary_directory(tmp_path_factory):
    """Keep the suite's verdicts out of the project's artifacts directory.

    The reviewer's read autosaves on every verdict -- that is the point of it
    -- so a test that judges an event writes a file. Without this it would
    write into `artifacts/reads/`, where it would be indistinguishable from a
    real read of the same window.
    """
    previous = os.environ.get("ONSET_REVIEW_READS")
    os.environ["ONSET_REVIEW_READS"] = str(tmp_path_factory.mktemp("reads"))
    yield
    if previous is None:
        os.environ.pop("ONSET_REVIEW_READS", None)
    else:
        os.environ["ONSET_REVIEW_READS"] = previous


@pytest.fixture(scope="session")
def recording():
    """A short labelled synthetic recording (seed fixed, so tests are stable)."""
    return make_synthetic_recording(duration_s=30, seed=7, verbose=False)


@pytest.fixture(scope="session")
def prepared(recording):
    return prepare(recording, verbose=False)


@pytest.fixture(scope="session")
def result(recording):
    return run_pipeline(recording, verbose=False)


@pytest.fixture(scope="session")
def store(result, tmp_path_factory):
    from onset_hfo.store import ResultStore

    return ResultStore(result.save(tmp_path_factory.mktemp("results")))


@pytest.fixture(scope="session")
def session(recording):
    """A live analysis session over the synthetic recording.

    Session-scoped so the montage and band-pass are computed once; tests that
    care about cost call ``reset_memo`` themselves.
    """
    from onset_agent.analysis import AnalysisSession

    return AnalysisSession(recording, verbose=False)


@pytest.fixture
def registry(session):
    from onset_agent.analysis import build_registry

    session.reset_memo()
    return build_registry(session)
