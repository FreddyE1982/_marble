from __future__ import annotations

import atexit
import json
import os
import threading
from collections import defaultdict
from collections.abc import Mapping, Sequence
from numbers import Number
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import hashlib
import time

import torch
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from .marblemain import Brain


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
        self._last_graph_signature: Optional[str] = None
        self._graph_step = 0

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
        if group_path and str(group_path[0]) == "wanderer_steps":
            return
        if self._log_training_walk(writer, group_path, itemname, value):
            return
        if self._log_training_datapair(writer, group_path, itemname, value):
            return
        tag_root = "/".join(str(x) for x in (*group_path, itemname))
        self._log_value(writer, tag_root, value)

    # --- Brain graph logging -------------------------------------------------

    def log_brain_graph(self, brain: "Brain", *, tag: str = "brain_topology") -> None:
        """Emit the current brain topology to TensorBoard's graph tab."""

        writer = self.ensure_writer()
        if writer is None:
            return

        build = self._build_brain_graph(brain, tag)
        if build is None:
            return
        graph_def, signature = build
        with self._lock:
            if signature == self._last_graph_signature:
                return
            self._last_graph_signature = signature
            self._graph_step += 1
        self._write_graph(writer, graph_def)

    def _build_brain_graph(self, brain: "Brain", tag: str) -> Optional[Tuple["graph_pb2.GraphDef", str]]:
        try:
            from tensorboard.compat.proto import graph_pb2
        except Exception:
            return None

        graph_def = graph_pb2.GraphDef()

        try:
            neuron_items = getattr(brain, "neurons", {})
            synapses = list(getattr(brain, "synapses", []) or [])
        except Exception:
            return None

        node_defs: Dict[str, Any] = {}
        entity_names: Dict[Tuple[str, int], str] = {}
        summary_bits: List[str] = []

        brain_node = graph_def.node.add()
        brain_node.name = f"brain/{getattr(brain, 'mode', 'unknown')}"
        brain_node.op = "Brain"
        brain_dims = getattr(brain, "n", None)
        try:
            brain_node.attr["dimensions"].i = int(brain_dims) if brain_dims is not None else 0
        except Exception:
            brain_node.attr["dimensions"].s = str(brain_dims).encode("utf-8")
        try:
            brain_node.attr["tag"].s = str(tag).encode("utf-8")
        except Exception:
            pass
        node_defs[brain_node.name] = brain_node
        summary_bits.append(f"B|{brain_node.name}|{brain_dims}|{tag}")

        neuron_names: List[str] = []
        for key, neuron in neuron_items.items():
            node = graph_def.node.add()
            name = self._format_neuron_name(key, neuron)
            node.name = name
            node.op = "Neuron"
            ntype = getattr(neuron, "type_name", None)
            if ntype:
                node.attr["type"].s = str(ntype).encode("utf-8")
            try:
                node.attr["weight"].f = float(getattr(neuron, "weight", 0.0))
            except Exception:
                pass
            try:
                node.attr["bias"].f = float(getattr(neuron, "bias", 0.0))
            except Exception:
                pass
            node_defs[name] = node
            entity_names[("Neuron", id(neuron))] = name
            neuron_names.append(name)
            summary_bits.append(
                "N|{}|{}|{}|{}".format(
                    name,
                    ntype,
                    getattr(neuron, "weight", 0.0),
                    getattr(neuron, "bias", 0.0),
                )
            )

        synapse_links: List[Tuple[str, Any, Any]] = []
        for idx, synapse in enumerate(synapses):
            node = graph_def.node.add()
            name = f"synapse/{idx}_{id(synapse)}"
            node.name = name
            node.op = "Synapse"
            direction = getattr(synapse, "direction", "uni")
            node.attr["direction"].s = str(direction).encode("utf-8")
            try:
                node.attr["weight"].f = float(getattr(synapse, "weight", 0.0))
            except Exception:
                pass
            try:
                node.attr["bias"].f = float(getattr(synapse, "bias", 0.0))
            except Exception:
                pass
            node_defs[name] = node
            entity_names[("Synapse", id(synapse))] = name
            synapse_links.append((name, getattr(synapse, "source", None), getattr(synapse, "target", None)))
            summary_bits.append(
                "S|{}|{}|{}|{}".format(
                    name,
                    direction,
                    getattr(synapse, "weight", 0.0),
                    getattr(synapse, "bias", 0.0),
                )
            )

        edges: List[str] = []
        for syn_name, source, target in synapse_links:
            src_name = self._resolve_entity_name(source, entity_names)
            dst_name = self._resolve_entity_name(target, entity_names)
            if src_name and syn_name in node_defs:
                if src_name not in node_defs[syn_name].input:
                    node_defs[syn_name].input.append(src_name)
                edges.append(f"{src_name}->{syn_name}")
            if dst_name and dst_name in node_defs:
                if syn_name not in node_defs[dst_name].input:
                    node_defs[dst_name].input.append(syn_name)
                edges.append(f"{syn_name}->{dst_name}")

        for neuron_name in neuron_names:
            if neuron_name in node_defs and brain_node.name not in node_defs[neuron_name].input:
                node_defs[neuron_name].input.append(brain_node.name)
                edges.append(f"{brain_node.name}->{neuron_name}")

        summary_bits.extend(edges)
        signature = hashlib.sha1("|".join(sorted(str(bit) for bit in summary_bits)).encode("utf-8")).hexdigest()
        return graph_def, signature

    def _format_neuron_name(self, key: Any, neuron: Any) -> str:
        try:
            if isinstance(key, (list, tuple)):
                coords = ",".join(str(v) for v in key)
                return f"neuron/[{coords}]"
        except Exception:
            pass
        pos = getattr(neuron, "position", None)
        if isinstance(pos, (list, tuple)):
            coords = ",".join(str(v) for v in pos)
            return f"neuron/[{coords}]"
        return f"neuron/{id(neuron)}"

    def _resolve_entity_name(self, entity: Any, mapping: Dict[Tuple[str, int], str]) -> Optional[str]:
        if entity is None:
            return None
        cls_name = getattr(entity.__class__, "__name__", "")
        key = (cls_name, id(entity))
        return mapping.get(key)

    def _write_graph(self, writer: SummaryWriter, graph_def: Any) -> None:
        try:
            file_writer = writer._get_file_writer()
        except Exception:
            file_writer = None

        if file_writer is not None:
            try:
                file_writer.add_graph(graph_def)
                return
            except Exception:
                pass

        try:
            writer.add_graph(graph_def=graph_def)
            return
        except Exception:
            pass

        if file_writer is not None:
            try:
                from tensorboard.compat.proto import event_pb2

                event = event_pb2.Event(
                    wall_time=time.time(),
                    step=self._graph_step,
                    graph_def=graph_def.SerializeToString(),
                )
                file_writer.add_event(event)
            except Exception:
                pass

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

    def _log_training_datapair(
        self,
        writer: SummaryWriter,
        group_path: Tuple[str, ...],
        itemname: str,
        value: Any,
    ) -> bool:
        if (
            len(group_path) >= 2
            and group_path[0] == "training"
            and group_path[1] == "datapair"
            and isinstance(value, Mapping)
            and str(itemname).startswith("pair_")
        ):
            for key, subvalue in value.items():
                tag = "/".join(str(x) for x in (*group_path, key))
                self._log_value(writer, tag, subvalue)
            return True
        return False

    def _log_training_walk(
        self,
        writer: SummaryWriter,
        group_path: Tuple[str, ...],
        itemname: str,
        value: Any,
    ) -> bool:
        if not (
            len(group_path) >= 2
            and str(group_path[0]) == "training"
            and isinstance(value, Mapping)
            and str(itemname).startswith("walk_")
        ):
            return False

        walks_index: Optional[int] = None
        for idx, segment in enumerate(group_path):
            if str(segment) == "walks":
                walks_index = idx
                break

        if walks_index is None:
            return False

        tag = "/".join(str(part) for part in group_path[: walks_index + 1])
        if not tag:
            return False

        self._log_value(writer, tag, value)
        return True

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

    def log_brain_graph(self, brain: "Brain", *, tag: str = "brain_topology") -> None:
        """Mirror the brain topology into TensorBoard's graph view."""

        try:
            self._tensorboard.log_brain_graph(brain, tag=tag)
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

