"""Repo-root import shim for running `python -m distributed_sgd.*` without install."""

from pathlib import Path

_src_pkg = Path(__file__).resolve().parent.parent / "src" / "distributed_sgd"
if _src_pkg.exists():
    __path__.append(str(_src_pkg))

__version__ = "0.1.0"
