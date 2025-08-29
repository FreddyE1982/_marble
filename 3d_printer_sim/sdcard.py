from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class VirtualSDCard:
    """In-memory representation of an SD card."""

    mounted: bool = False
    files: Dict[str, bytes] = field(default_factory=dict)

    def mount(self) -> None:
        self.mounted = True

    def unmount(self) -> None:
        self.mounted = False
        self.files.clear()

    def write_file(self, path: str, data: bytes) -> None:
        if not self.mounted:
            raise RuntimeError("SD card not mounted")
        self.files[path] = bytes(data)

    def read_file(self, path: str) -> Optional[bytes]:
        if not self.mounted:
            raise RuntimeError("SD card not mounted")
        return self.files.get(path)

    def list_files(self) -> List[str]:
        if not self.mounted:
            raise RuntimeError("SD card not mounted")
        return sorted(self.files.keys())

    def delete_file(self, path: str) -> None:
        if not self.mounted:
            raise RuntimeError("SD card not mounted")
        self.files.pop(path, None)
