from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VirtualUSB:
    """Simple virtual USB communication interface."""

    connected: bool = False
    host_buffer: List[bytes] = field(default_factory=list)
    device_buffer: List[bytes] = field(default_factory=list)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.host_buffer.clear()
        self.device_buffer.clear()

    def send_from_host(self, data: bytes) -> None:
        if not self.connected:
            raise RuntimeError("USB not connected")
        self.host_buffer.append(bytes(data))

    def read_from_host(self) -> Optional[bytes]:
        if not self.connected or not self.host_buffer:
            return None
        return self.host_buffer.pop(0)

    def send_from_device(self, data: bytes) -> None:
        if not self.connected:
            raise RuntimeError("USB not connected")
        self.device_buffer.append(bytes(data))

    def read_from_device(self) -> Optional[bytes]:
        if not self.connected or not self.device_buffer:
            return None
        return self.device_buffer.pop(0)
