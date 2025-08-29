from __future__ import annotations

from typing import List

from ..reporter import report
from .conv1d import Conv1DNeuronPlugin as _Conv1DNeuronPlugin


class MaxPool1DNeuronPlugin(_Conv1DNeuronPlugin):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool1D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool1D requires 3 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size")
        ls = lstore.get("stride")
        lp = lstore.get("padding")
        try:
            ksize = int(max(1, round(float(lk.detach().to("cpu").view(-1)[0].item())))) if hasattr(lk, "detach") else int(max(1, round(base_ks)))
        except Exception:
            ksize = int(max(1, round(base_ks)))
        try:
            stride = int(max(1, round(float(ls.detach().to("cpu").view(-1)[0].item())))) if hasattr(ls, "detach") else int(max(1, round(base_st)))
        except Exception:
            stride = int(max(1, round(base_st)))
        try:
            padding = int(max(0, round(float(lp.detach().to("cpu").view(-1)[0].item())))) if hasattr(lp, "detach") else int(max(0, round(base_pd)))
        except Exception:
            padding = int(max(0, round(base_pd)))

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            x1: List[float] = []
            for s in data_incs:
                x1 += self._to_list1d(getattr(s.source, "tensor", []))
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x1 = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None:
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                y = torch.nn.functional.max_pool1d(xt, kernel_size=ksize, stride=stride, padding=padding)
                y = y.view(-1)
                try:
                    report("neuron", "maxpool1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if padding > 0:
            x1 = ([float("-inf")] * padding) + x1 + ([float("-inf")] * padding)
        n = len(x1)
        out_len = 0 if n < ksize else 1 + (n - ksize) // stride
        y_list = []
        for t in range(out_len):
            base = t * stride
            y_list.append(max(x1[base: base + ksize]))
        try:
            report("neuron", "maxpool1d", {"in": len(x1), "out": len(y_list), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


__all__ = ["MaxPool1DNeuronPlugin"]

# Remove base class alias to keep plugin discovery focused on MaxPool1DNeuronPlugin
del _Conv1DNeuronPlugin

