from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class VirtualUSB:
    """Simple virtual USB communication interface."""

    connected: bool = False
    host_buffer: List[bytes] = field(default_factory=list)
    device_buffer: List[bytes] = field(default_factory=list)
    _last_host_read: Optional[bytes] = None
    _last_device_read: Optional[bytes] = None

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.host_buffer.clear()
        self.device_buffer.clear()
        self._last_host_read = None
        self._last_device_read = None

    def send_from_host(self, data: bytes) -> None:
        if not self.connected:
            raise RuntimeError("USB not connected")
        self.host_buffer.append(bytes(data))

    def read_from_host(self) -> Optional[bytes]:
        if not self.connected:
            return None
        if self.host_buffer:
            self._last_host_read = self.host_buffer.pop(0)
        return self._last_host_read

    def send_from_device(self, data: bytes) -> None:
        if not self.connected:
            raise RuntimeError("USB not connected")
        self.device_buffer.append(bytes(data))

    def read_from_device(self) -> Optional[bytes]:
        if not self.connected:
            return None
        if self.device_buffer:
            self._last_device_read = self.device_buffer.pop(0)
        return self._last_device_read
