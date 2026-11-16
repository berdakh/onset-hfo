"""Problem reports for the pipeline and command line, without a desktop runtime.

The bundle holds the exact redacted text shown before saving. The desktop
logger adds its own facts and log files to the same bundle and ZIP writer;
this module never probes the window, its settings or its logs.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

__all__ = ["Bundle", "about", "redact", "report_text", "build_bundle", "write_bundle",
           "main", "run", "build_parser", "add_arguments"]


def about() -> dict:
    """Software versions and machine facts; no optional package is imported."""
    from onset_hfo import __author__
    from onset_hfo.config import PIPELINE_VERSION

    # The import package's __version__ tracks the pipeline, not the release.
    try:
        package_version = version("onset-hfo")
    except PackageNotFoundError:
        package_version = "unknown (package metadata unavailable)"
    out = {
        "onset-hfo": package_version, "pipeline": PIPELINE_VERSION,
        "developer": __author__,
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "machine": platform.machine(),
        "cpus": os.cpu_count(),
    }
    for package in ("mne", "numpy", "scipy", "pandas", "matplotlib", "scikit-learn"):
        try:
            out[package] = version(package)
        except PackageNotFoundError:
            out[package] = "not installed"
    return out


def redact(text: str) -> str:
    """Write the home folder as ``~`` in every report file."""
    home = str(Path.home())
    return text.replace(home, "~") if home and home != "/" else text


def report_text(description: str = "") -> str:
    """The base report: versions, machine and the reporter's own words."""
    parts = ["# Onset HFO problem report", time.strftime("%Y-%m-%d %H:%M:%S"), ""]
    if description.strip():
        parts += ["## What happened, in the reporter's words", description.strip(), ""]
    parts.append("## Versions and machine")
    parts += [f"{key}: {value}" for key, value in about().items()]
    return redact("\n".join(parts)) + "\n"


@dataclass
class Bundle:
    """Exact text of each file, shared by the preview and the saved ZIP.

    A desktop caller supplies its report renderer when it adds window facts
    and logs, so editing the description preserves that report's context.
    """

    files: dict[str, str] = field(default_factory=dict)
    with_logs: bool = False
    render_report: Callable[[str], str] = field(default=report_text, repr=False, compare=False)

    def manifest(self) -> list[dict]:
        """Each file's name, lines and UTF-8 size, before deciding to save."""
        return [{"name": name, "lines": len(text.splitlines()),
                 "bytes": len(text.encode("utf-8"))}
                for name, text in self.files.items()]

    def with_description(self, description: str) -> Bundle:
        """Replace the report's description, carrying other files untouched."""
        files = dict(self.files)
        files["report.txt"] = self.render_report(description)
        return Bundle(files, self.with_logs, self.render_report)

    def only(self, names) -> Bundle:
        """Restrict the bundle to selected names, in the original order."""
        keep = set(names)
        return Bundle({n: t for n, t in self.files.items() if n in keep},
                      self.with_logs, self.render_report)


def build_bundle(description: str = "") -> Bundle:
    """Build the command line report without looking for any window logs."""
    return Bundle({"report.txt": report_text(description)})


def write_bundle(bundle: Bundle, path: str | Path) -> Path:
    """Write exactly the previewed bundle as one ZIP, and nothing else."""
    path = Path(path)
    if path.suffix.lower() != ".zip":
        path = path.with_suffix(".zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, text in bundle.files.items():
            zipped.writestr(name, text)
    return path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The options of the report, on any parser: this module's own and the
    `onset-hfo report-problem` subcommand's, so they cannot drift apart."""
    parser.add_argument("--note", default="",
                        help="what you did, what you expected, and what happened instead")
    parser.add_argument("--out", type=Path, default=None, metavar="FILE.zip",
                        help="where to save it (default: ./onset-hfo-report-<time>.zip)")
    parser.add_argument("--yes", action="store_true",
                        help="save without asking; the text is still printed first")


def build_parser(prog: str = "onset-hfo report-problem") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Show a problem report for you to read, and save it as a zip if you "
                    "agree. It holds versions, this machine's description and your note: "
                    "no signal, no recording names, no log. Nothing is sent anywhere.")
    add_arguments(parser)
    return parser


def run(args: argparse.Namespace) -> int:
    """Show the report, then save it if the person agrees (or said `--yes`)."""
    bundle = build_bundle(args.note)
    out = args.out or Path(f"onset-hfo-report-{time.strftime('%Y%m%d-%H%M')}.zip")

    print("This report would contain:")
    for row in bundle.manifest():
        print(f"  {row['name']}  ({row['lines']} lines, {row['bytes']} bytes)")
    print("\n" + "-" * 60)
    for name, text in bundle.files.items():
        print(f"[{name}]\n{text}")
    print("-" * 60)

    if not args.yes:
        try:
            answer = input(f"Save this to {out}? [y/N] ").strip().lower()
        except EOFError:       # no terminal to ask: the safe answer is no
            answer = ""
        if answer not in ("y", "yes"):
            print("Nothing was saved.")
            return 0
    written = write_bundle(bundle, out)
    print(f"Saved to {written}. Send this file; nothing has been sent.")
    return 0


def main(argv: list[str] | None = None, prog: str = "onset-hfo report-problem") -> int:
    return run(build_parser(prog).parse_args(argv))


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
