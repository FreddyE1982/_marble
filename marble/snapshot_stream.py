"""Incremental snapshot stream helpers.

This module provides a crash-resistant append-only stream format used by
``Brain.save_snapshot``.  Each snapshot write appends a self-describing frame
containing the full serialized state so that readers can reconstruct the most
recent brain layout even while training is still running.

The stream format is intentionally simple:

* 8-byte magic header ``MBSTREAM`` followed by a 1-byte version and a 1-byte
  compression level hint.  Readers ignore the hint but it documents the
  intended zlib level.
* Repeated frames, each encoded as ``[type(4s)][length(8)][crc(4)][payload]
  [length(4)]``.  ``type`` is ASCII (``FULL`` for full-state frames),
  ``length`` is the payload size in bytes and is repeated at the end for quick
  sanity checks.  ``crc`` stores the zlib CRC32 of the compressed payload.
* ``payload`` is ``zlib.compress`` of a pickled mapping
  ``{"time": float, "reason": str, "state": Mapping}``.

The framing guarantees that incomplete writes (e.g. power failure) are simply
ignored by readers—the last fully written frame remains accessible.  Writers
flush and fsync after every append so the file is always readable by
``snapshot_to_image`` while training continues.
"""

from __future__ import annotations

import os
import pickle
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional


MAGIC = b"MBSTREAM"
VERSION = 1
HEADER_STRUCT = struct.Struct(">8sBB")
FRAME_HEADER_STRUCT = struct.Struct(">4sQI")
FRAME_FOOTER_STRUCT = struct.Struct(">I")
FRAME_TYPE_FULL = b"FULL"


@dataclass(frozen=True)
class SnapshotFrame:
    """In-memory representation of a decoded stream frame."""

    frame_type: str
    timestamp: float
    reason: str
    state: Dict[str, Any]


class SnapshotStreamError(RuntimeError):
    """Raised when the on-disk stream is irrecoverably malformed."""


class SnapshotStreamWriter:
    """Append-only writer that produces crash-safe snapshot streams."""

    def __init__(self, path: str, *, compress_level: int = 2) -> None:
        self.path = str(path)
        self.compress_level = int(max(1, min(9, compress_level)))
        self._lock = threading.Lock()
        self._file = self._open_file()
        self._frame_index = self._scan_existing_frames()

    # -- private helpers -------------------------------------------------
    def _open_file(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        is_new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        mode = "r+b" if not is_new else "w+b"
        fh = open(self.path, mode)
        if is_new:
            fh.write(HEADER_STRUCT.pack(MAGIC, VERSION, self.compress_level))
            fh.flush()
            os.fsync(fh.fileno())
        else:
            header = fh.read(HEADER_STRUCT.size)
            if len(header) != HEADER_STRUCT.size:
                raise SnapshotStreamError("Snapshot stream header is truncated")
            magic, version, stored_level = HEADER_STRUCT.unpack(header)
            if magic != MAGIC:
                raise SnapshotStreamError("Not a marble snapshot stream")
            if version != VERSION:
                raise SnapshotStreamError(
                    f"Unsupported snapshot stream version: {version}"
                )
            # If the stored compression level differs we keep using the original
            # setting to avoid mixing payload encodings mid-stream.
            self.compress_level = int(stored_level) or self.compress_level
        fh.seek(0, os.SEEK_END)
        return fh

    def _scan_existing_frames(self) -> int:
        count = 0
        for _ in iterate_snapshot_frames(self.path):
            count += 1
        return count

    # -- public API ------------------------------------------------------
    def append_state(self, state: Dict[str, Any], *, reason: str = "manual") -> None:
        payload = {
            "time": float(time.time()),
            "reason": str(reason),
            "state": dict(state),
        }
        raw = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
        compressed = zlib.compress(raw, self.compress_level)
        length = len(compressed)
        crc = zlib.crc32(compressed) & 0xFFFFFFFF
        header = FRAME_HEADER_STRUCT.pack(FRAME_TYPE_FULL, length, crc)
        footer = FRAME_FOOTER_STRUCT.pack(length)
        with self._lock:
            self._file.write(header)
            self._file.write(compressed)
            self._file.write(footer)
            self._file.flush()
            os.fsync(self._file.fileno())
            self._frame_index += 1

    def close(self) -> None:
        with self._lock:
            try:
                self._file.flush()
                os.fsync(self._file.fileno())
            except Exception:
                pass
            self._file.close()


def iterate_snapshot_frames(path: str) -> Iterator[SnapshotFrame]:
    """Yield decoded frames from *path* until the stream terminates."""

    if not os.path.exists(path):
        return
    with open(path, "rb") as fh:
        header = fh.read(HEADER_STRUCT.size)
        if len(header) != HEADER_STRUCT.size:
            return
        magic, version, _level = HEADER_STRUCT.unpack(header)
        if magic != MAGIC or version != VERSION:
            raise SnapshotStreamError("Unsupported snapshot stream header")
        while True:
            prefix = fh.read(FRAME_HEADER_STRUCT.size)
            if len(prefix) < FRAME_HEADER_STRUCT.size:
                break
            frame_type, length, crc = FRAME_HEADER_STRUCT.unpack(prefix)
            payload = fh.read(length)
            if len(payload) < length:
                break
            suffix = fh.read(FRAME_FOOTER_STRUCT.size)
            if len(suffix) < FRAME_FOOTER_STRUCT.size:
                break
            (length_footer,) = FRAME_FOOTER_STRUCT.unpack(suffix)
            if length_footer != length:
                break
            if zlib.crc32(payload) & 0xFFFFFFFF != crc:
                break
            try:
                decoded = pickle.loads(zlib.decompress(payload))
            except Exception:
                break
            state = decoded.get("state") or decoded.get("data")
            if not isinstance(state, dict):
                continue
            timestamp = float(decoded.get("time", 0.0))
            reason = str(decoded.get("reason", ""))
            try:
                frame_name = frame_type.decode("ascii", "ignore")
            except Exception:
                frame_name = "FULL"
            yield SnapshotFrame(frame_name, timestamp, reason, state)


def read_latest_state(path: str) -> Optional[Dict[str, Any]]:
    """Return the most recent state dictionary from *path* if available."""

    last_state: Optional[Dict[str, Any]] = None
    for frame in iterate_snapshot_frames(path):
        if frame.frame_type == "FULL":
            last_state = frame.state
    return last_state


def append_state(path: str, state: Dict[str, Any], *, reason: str = "manual", compress_level: int = 2) -> None:
    """Convenience helper mirroring :meth:`SnapshotStreamWriter.append_state`."""

    writer = SnapshotStreamWriter(path, compress_level=compress_level)
    try:
        writer.append_state(state, reason=reason)
    finally:
        writer.close()
