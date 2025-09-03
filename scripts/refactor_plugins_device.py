#!/usr/bin/env python3
"""Automates device-aware refactor across all plugins and runs tests.

This script scans every plugin under ``marble/plugins`` and replaces
any explicit ``.to("cpu")`` casts with ``.to(owner._device)`` so tensors
follow the owning object's active device.  It also ensures that detach
is only used when converting to Python scalars by keeping ``detach``
when an ``.item()`` call is present on the same expression.

After modifying plugins, the script executes each test module in
``tests/`` individually, respecting repository testing policy.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
PLUGIN_DIR = ROOT / "marble" / "plugins"
TEST_DIR = ROOT / "tests"

CPU_CAST_RE = re.compile(r"\.to\(\s*['\"]cpu['\"]\s*\)")
DETACH_CLEAN_RE = re.compile(r"\.detach\(\)\.to\(owner._device\)(?!\.item\()")


def patch_plugin(path: pathlib.Path) -> bool:
    text = path.read_text()
    original = text
    text = CPU_CAST_RE.sub(".to(owner._device)", text)
    # Remove detach() when not followed by .item(); keep for scalar extraction
    text = DETACH_CLEAN_RE.sub(".to(owner._device)", text)
    if text != original:
        path.write_text(text)
        return True
    return False


def run_tests() -> None:
    tests = sorted(
        p for p in TEST_DIR.glob("test_*.py") if "3d_printer_sim" not in p.name
    )
    for t in tests:
        mod = f"tests.{t.stem}"
        print(f"Running {mod}")
        subprocess.run([sys.executable, "-m", "unittest", "-v", mod], check=True)


def main() -> None:
    changed = False
    for py in PLUGIN_DIR.glob("*.py"):
        if patch_plugin(py):
            print(f"Patched {py.relative_to(ROOT)}")
            changed = True
    if not changed:
        print("No plugins required patching.")
    run_tests()


if __name__ == "__main__":
    main()
