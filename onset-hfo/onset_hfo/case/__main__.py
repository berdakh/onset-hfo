"""Cases from the terminal: make one, convert files into it, list it.

    python -m onset_hfo.case new  /data/cases/P017 P017
    python -m onset_hfo.case convert /data/cases/P017 night1.edf night2.edf --all-as seeg --mains 50
    python -m onset_hfo.case list /data/cases/P017

`convert` types every channel the file calls EEG, SEEG or ECoG as `--all-as`
(names like ECG*, EMG*, EOG*, DC*, TRIG* keep the batch rule's types), the
way the window's one-click button does. Check one file in the window first.
"""

from __future__ import annotations

import argparse
import sys

from onset_hfo.case import bridges
from onset_hfo.case.model import Case


def _types(info: dict, all_as: str) -> dict:
    import fnmatch

    exceptions = (("ECG*", "ecg"), ("EKG*", "ecg"), ("EMG*", "emg"), ("EOG*", "eog"),
                  ("DC*", "misc"), ("TRIG*", "stim"), ("STI*", "stim"))
    out = {}
    for row in info["channels"].itertuples():
        kind = all_as if row.declared in ("eeg", "seeg", "ecog") else row.declared
        for pattern, other in exceptions:
            if fnmatch.fnmatch(str(row.name).upper(), pattern):
                kind = other
                break
        out[str(row.name)] = kind
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m onset_hfo.case")
    sub = parser.add_subparsers(dest="command", required=True)
    new = sub.add_parser("new", help="make an empty case")
    new.add_argument("folder")
    new.add_argument("pseudonym")
    conv = sub.add_parser("convert", help="convert clinical files into a case")
    conv.add_argument("folder")
    conv.add_argument("files", nargs="+")
    conv.add_argument("--all-as", choices=("seeg", "ecog"), required=True)
    conv.add_argument("--mains", type=float, choices=(50.0, 60.0), required=True)
    conv.add_argument("--bad", default="", help="comma-separated contacts known bad")
    listing = sub.add_parser("list", help="what a case holds")
    listing.add_argument("folder")
    args = parser.parse_args(argv)

    if args.command == "new":
        case = Case.create(args.folder, args.pseudonym)
        print(f"case {case.case_id} at {case.root}")
        return 0
    case = Case.open(args.folder)
    if args.command == "list":
        print(f"case {case.case_id}: {len(case.recordings)} recording(s)")
        for rec in case.recordings:
            print(f"  run {rec.run}: {rec.source_name} ({rec.source_format}), "
                  f"{rec.n_channels} ch, {rec.sfreq:g} Hz, {rec.duration_s / 3600:.2f} h")
        done = [k for k, v in case.steps.items() if v.get("done")]
        print(f"  steps done: {', '.join(done) or 'none'}")
        return 0
    failed = 0
    bad = [b.strip() for b in args.bad.split(",") if b.strip()]
    for path in args.files:
        try:
            info = bridges.inspect(path)
            report = bridges.convert(path, case, channel_types=_types(info, args.all_as),
                                     bad=bad, line_freq=args.mains)
            print(f"{path}: run {report.run}, {report.n_channels} ch at {report.sfreq:g} Hz, "
                  f"{report.duration_s / 3600:.2f} h; left out: "
                  f"{', '.join(report.left_out) or 'nothing identifying'}")
            for warning in report.warnings:
                print(f"  warning: {warning}")
        except Exception as error:      # noqa: BLE001 - one bad file does not stop the rest
            failed += 1
            print(f"{path}: not converted: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
