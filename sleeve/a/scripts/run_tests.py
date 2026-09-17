#!/usr/bin/env python3
"""Run the sleeve/a test suite under the 3 GB watchdog (scripts/pyrun.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    raise SystemExit(pytest.main([str(ROOT / "tests"), "-q", *sys.argv[1:]]))
