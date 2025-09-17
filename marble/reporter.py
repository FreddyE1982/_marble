from __future__ import annotations

import atexit
import json
import os
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.tensorboard import SummaryWriter


def _load_tensorboard_config() -> Tuple[bool, Optional[str], int]:
    """Read TensorBoard settings from ``config.yaml``."""

    try:
        import yaml  # type: ignore

        base = os.path.dirname(os.path.dirname(__file__))
        with open(os.path.join(base, "config.yaml"), "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        reporter = data.get("reporter", {}) if isinstance(data, dict) else {}
        tb = reporter.get("tensorboard", {}) if isinstance(reporter, dict) else {}
        enabled = bool(tb.get("enabled", True))
        log_dir = tb.get("log_dir")
        if isinstance(log_dir, str) and log_dir.strip():
            log_dir = os.path.expanduser(log_dir.strip())
        else:
            log_dir = None
        flush_interval = tb.get("flush_interval_steps", 5)
        try:
            flush_interval_int = max(1, int(flush_interval))
        except Exception:
            flush_interval_int = 5
        return enabled, log_dir, flush_interval_int
    except Exception:
        return True, None, 5


_TB_ENABLED, _TB_LOGDIR, _TB_FLUSH_INTERVAL = _load_tensorboard_config()


class _TensorBoardAdapter:
    """Mirror reporter updates into TensorBoard."""

    def __init__(self, enabled: bool, log_dir: Optional[str], flush_interval: int) -> None:
        self._enabled = enabled
        self._log_dir = log_dir
        self._flush_interval = max(1, int(flush_interval))
        self._writer: Optional[SummaryWriter] = None
        self._step_counters: Dict[str, int] = defaultdict(int)
        self._pending_flush = 0
        self._lock = threading.RLock()

    @property
    def logdir(self) -> Optional[str]:
        with self._lock:
            if self._writer is not None:
                return self._writer.log_dir
            return self._log_dir

    def ensure_writer(self) -> Optional[SummaryWriter]:
        if not self._enabled:
            return None
        with self._lock:
            if self._writer is None:
                try:
                    self._writer = SummaryWriter(log_dir=self._log_dir)
                    self._log_dir = self._writer.log_dir
                    atexit.register(self.close)
                except Exception:
                    self._enabled = False
                    self._writer = None
                    return None
            return self._writer

    def _next_step(self, tag: str) -> int:
        step = self._step_counters[tag]
        self._step_counters[tag] = step + 1
        return step

    def log(self, group_path: Tuple[str, ...], itemname: str, value: Any) -> None:
        writer = self.ensure_writer()
        if writer is None:
            return
        tag_root = "/".join(str(x) for x in (*group_path, itemname))
        self._log_value(writer, tag_root, value)

    def _log_value(self, writer: SummaryWriter, tag: str, value: Any) -> None:
        try:
            if isinstance(value, Mapping):
                for key, subvalue in value.items():
                    self._log_value(writer, f"{tag}/{key}", subvalue)
                self._after_log(writer)
                return

            if isinstance(value, torch.Tensor):
                tensor = value.detach().to("cpu")
                if tensor.numel() == 1:
                    writer.add_scalar(tag, float(tensor.item()), self._next_step(tag))
                else:
                    writer.add_histogram(tag, tensor, self._next_step(tag))
                self._after_log(writer)
                return

            if isinstance(value, Number):
                writer.add_scalar(tag, float(value), self._next_step(tag))
                self._after_log(writer)
                return

            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                seq = list(value)
                if seq and all(isinstance(x, Number) for x in seq):
                    tensor = torch.tensor(seq, dtype=torch.float32)
                    writer.add_histogram(tag, tensor, self._next_step(tag))
                    self._after_log(writer)
                    return

            writer.add_text(tag, repr(value), self._next_step(tag))
            self._after_log(writer)
        except Exception:
            return

    def _after_log(self, writer: SummaryWriter) -> None:
        self._pending_flush += 1
        if self._pending_flush >= self._flush_interval:
            try:
                writer.flush()
            except Exception:
                pass
            self._pending_flush = 0

    def flush(self) -> None:
        with self._lock:
            if self._writer is None:
                return
            try:
                self._writer.flush()
            except Exception:
                pass
            self._pending_flush = 0

    def close(self) -> None:
        with self._lock:
            if self._writer is None:
                return
            try:
                self._writer.flush()
                self._writer.close()
            except Exception:
                pass
            self._writer = None


class _ReporterItemAccessor:
    def __init__(self, owner: "Reporter") -> None:
        self._owner = owner

    def __setitem__(self, key, value) -> None:
        if not isinstance(key, tuple) or len(key) < 2:
            raise KeyError("Key must be a tuple of (itemname, groupname[, subgroups...])")
        itemname = key[0]
        path = tuple(str(x) for x in key[1:])
        self._owner._set_item(path, str(itemname), value)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) < 2:
            raise KeyError("Key must be a tuple of (itemname, groupname[, subgroups...])")
        itemname = key[0]
        path = tuple(str(x) for x in key[1:])
        return self._owner.get_item(path, str(itemname))

    def __call__(self, itemname: str, groupname: str, *subgroups: str):
        return self._owner.get_item((groupname,) + tuple(str(s) for s in subgroups), itemname)


class Reporter:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._groups: Dict[str, Dict[str, Any]] = {}
        self.item = _ReporterItemAccessor(self)
        self._tensorboard = _TensorBoardAdapter(_TB_ENABLED, _TB_LOGDIR, _TB_FLUSH_INTERVAL)

    def registergroup(self, groupname: str, *subgroups: str) -> None:
        with self._lock:
            self._ensure_group_path((str(groupname),) + tuple(str(s) for s in subgroups))

    def _ensure_group_path(self, path: Tuple[str, ...]) -> Dict[str, Any]:
        node: Dict[str, Any]
        node = {"_items": {}, "_subgroups": self._groups}
        for name in path:
            subs = node["_subgroups"]
            if name not in subs or not isinstance(subs[name], dict):
                subs[name] = {"_items": {}, "_subgroups": {}}
            node = subs[name]
        return node

    def _set_item(self, group_path: Tuple[str, ...], itemname: str, value: Any) -> None:
        with self._lock:
            node = self._ensure_group_path(group_path)
            key = str(itemname)
            existing = node["_items"].get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                # merge dictionaries in place instead of replacing
                existing.update(value)
            else:
                node["_items"][key] = value
            try:
                self._tensorboard.log(group_path, key, value)
            except Exception:
                pass

    def get_item(self, group_path: Tuple[str, ...], itemname: str) -> Any:
        with self._lock:
            node = self._find_group_node(group_path)
            return None if node is None else node["_items"].get(str(itemname), None)

    def group(self, groupname: str, *subgroups: str) -> Dict[str, Any]:
        path = (str(groupname),) + tuple(str(s) for s in subgroups)
        with self._lock:
            node = self._find_group_node(path)
            return {} if node is None else dict(node["_items"])  # shallow copy

    def dirgroups(self) -> List[str]:
        with self._lock:
            return list(self._groups.keys())

    def dirtree(self, groupname: Optional[str] = None, *subgroups: str) -> Dict[str, Any]:
        with self._lock:
            if groupname is None:
                return {k: self._summarize(self._groups[k]) for k in self._groups}
            path = (str(groupname),) + tuple(str(s) for s in subgroups)
            node = self._find_group_node(path)
            return {} if node is None else self._summarize(node)

    def clear_group(self, groupname: str, *subgroups: str) -> None:
        path = (str(groupname),) + tuple(str(s) for s in subgroups)
        with self._lock:
            parent: Dict[str, Any] = {"_subgroups": self._groups}
            cur: Optional[Dict[str, Any]] = parent
            for name in path[:-1]:
                if cur is None:
                    return
                subs = cur.get("_subgroups", {})
                cur = subs.get(str(name))
                if not isinstance(cur, dict):
                    return
            subs = cur.get("_subgroups", {}) if cur else {}
            subs.pop(str(path[-1]), None)

    def _find_group_node(self, path: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
        node: Dict[str, Any] = {"_items": {}, "_subgroups": self._groups}
        cur: Optional[Dict[str, Any]] = node
        for name in path:
            if cur is None:
                return None
            subs = cur.get("_subgroups", {})
            cur = subs.get(str(name))
            if not isinstance(cur, dict):
                return None
        return cur

    def _summarize(self, node: Dict[str, Any]) -> Dict[str, Any]:
        items = list(node.get("_items", {}).keys())
        subs: Dict[str, Any] = {}
        for name, sub in node.get("_subgroups", {}).items():
            if isinstance(sub, dict):
                subs[name] = self._summarize(sub)
        return {"items": items, "subgroups": subs}

    def tensorboard_logdir(self) -> Optional[str]:
        """Return the TensorBoard log directory used by the reporter."""

        try:
            writer = self._tensorboard.ensure_writer()
            if writer is not None:
                return writer.log_dir
            return self._tensorboard.logdir
        except Exception:
            return None

    def flush_tensorboard(self) -> None:
        """Force a flush of buffered TensorBoard events."""

        try:
            self._tensorboard.flush()
        except Exception:
            pass


REPORTER = Reporter()


def report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    REPORTER.item[(itemname,) + (groupname,) + tuple(subgroups)] = data


def report_group(groupname: str, *subgroups: str) -> Dict[str, Any]:
    return REPORTER.group(groupname, *subgroups)


def report_dir(groupname: Optional[str] = None, *subgroups: str) -> Dict[str, Any]:
    return REPORTER.dirtree(groupname, *subgroups)


def clear_report_group(groupname: str, *subgroups: str) -> None:
    REPORTER.clear_group(groupname, *subgroups)


def export_wanderer_steps_to_jsonl(path: str, *, groupname: str = "wanderer_steps", subgroups: Sequence[str] = ("logs",)) -> None:
    try:
        items = REPORTER.group(groupname, *subgroups)
    except Exception:
        items = {}

    def keyfn(k: str) -> Tuple[int, int]:
        try:
            parts = k.split("_")
            nums = [int(p) for p in parts if p.isdigit()]
            if not nums:
                return (1 << 30, 0)
            return (nums[0], nums[-1])
        except Exception:
            return (1 << 30, 0)

    keys = sorted(items.keys(), key=keyfn)
    with open(path, "w", encoding="utf-8") as f:
        for k in keys:
            rec = items[k]
            try:
                json.dump({"item": k, **(rec if isinstance(rec, dict) else {"value": rec})}, f)
            except Exception:
                json.dump({"item": k, "value": str(rec)}, f)
            f.write("\n")
    try:
        report("logs", "export_wanderer_steps", {"path": path, "count": len(keys)}, "files")
    except Exception:
        pass


__all__ = [
    "Reporter",
    "REPORTER",
    "report",
    "report_group",
    "report_dir",
    "clear_report_group",
    "export_wanderer_steps_to_jsonl",
]

