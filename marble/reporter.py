from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
            node["_items"][str(itemname)] = value

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


REPORTER = Reporter()


def report(groupname: str, itemname: str, data: Any, *subgroups: str) -> None:
    REPORTER.item[(itemname,) + (groupname,) + tuple(subgroups)] = data


def report_group(groupname: str, *subgroups: str) -> Dict[str, Any]:
    return REPORTER.group(groupname, *subgroups)


def report_dir(groupname: Optional[str] = None, *subgroups: str) -> Dict[str, Any]:
    return REPORTER.dirtree(groupname, *subgroups)


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
    "export_wanderer_steps_to_jsonl",
]

