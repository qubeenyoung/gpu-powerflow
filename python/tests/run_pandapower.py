"""Compatibility shim for the renamed pypower baseline runner."""
from __future__ import annotations

from .run_pypower import main, run_pypower

__all__ = ["main", "run_pypower"]


if __name__ == "__main__":
    main()
