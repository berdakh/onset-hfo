"""The developer is named wherever the software speaks for itself: the
package, `--version`, Help → About, the problem report's facts, and every
exported report."""

from __future__ import annotations

import pytest


def test_the_developer_is_named():
    import onset_hfo
    import onset_review
    from onset_review import applog, credit
    from onset_review.app import build_parser

    assert onset_review.DEVELOPER == "Berdakh Abibullaev"
    assert onset_review.__author__ == onset_hfo.__author__ == onset_review.DEVELOPER
    assert credit() == f"Onset Review {onset_review.__version__} · developed by Berdakh Abibullaev"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--version"])
    assert applog.about()["developer"] == "Berdakh Abibullaev"


def test_help_about_names_the_developer():
    pytest.importorskip("qtpy.QtWidgets", reason="the review extra is not installed")
    from onset_review.window import about_text

    text = about_text()
    assert "Developed by <b>Berdakh Abibullaev</b>" in text
    assert "not a medical device" in text
