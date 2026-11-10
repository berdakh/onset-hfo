"""Compatibility entry point for the core command line problem report."""

from __future__ import annotations

import sys

from onset_hfo.problemreport import add_arguments, build_parser, main, run

__all__ = ["main", "run", "build_parser", "add_arguments"]


if __name__ == "__main__":      # pragma: no cover
    sys.exit(main())
