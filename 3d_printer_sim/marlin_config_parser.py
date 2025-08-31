"""Utility to enumerate configuration items from Marlin's default
Configuration.h and Configuration_adv.h files.

The parser downloads the two configuration headers from the Marlin
repository and extracts all ``#define`` identifiers. This helps keep the
simulator's configuration coverage in sync with upstream firmware.

Only the Python standard library is used so the module remains
self-contained within ``3d_printer_sim``.
"""
from __future__ import annotations

import re
import urllib.request
from typing import Iterable, List

MARLIN_BASE = (
    "https://raw.githubusercontent.com/MarlinFirmware/Marlin/2.1.x/Marlin"
)
FILES = ("Configuration.h", "Configuration_adv.h")


def fetch_marlin_files(filenames: Iterable[str]) -> List[str]:
    """Download Marlin configuration files and return their contents."""
    contents: List[str] = []
    for name in filenames:
        url = f"{MARLIN_BASE}/{name}"
        with urllib.request.urlopen(url) as resp:  # nosec: B310 - controlled URL
            contents.append(resp.read().decode("utf-8", errors="ignore"))
    return contents


def parse_config_items(text: str) -> List[str]:
    """Return all ``#define`` identifiers found in *text*."""
    pattern = re.compile(r"^\s*#define\s+(\w+)", re.MULTILINE)
    return [m.group(1) for m in pattern.finditer(text)]


def enumerate_config_items() -> List[str]:
    """Enumerate all configuration items from Marlin's default headers."""
    items: set[str] = set()
    for text in fetch_marlin_files(FILES):
        items.update(parse_config_items(text))
    return sorted(items)


if __name__ == "__main__":  # pragma: no cover - manual usage
    for name in enumerate_config_items():
        print(name)
