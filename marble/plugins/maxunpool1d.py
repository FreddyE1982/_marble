from __future__ import annotations

from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class MaxUnpool1DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxUnpool1D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxunpool1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src = [s.source for s in param_incs[:3]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        data_incs.sort(key=self._key_src)
        vals = self._to_list1d(getattr(data_incs[0].source, "tensor", [])) if len(data_incs) >= 1 else self._to_list1d(input_value)
        idxs = self._to_list1d(getattr(data_incs[1].source, "tensor", [])) if len(data_incs) >= 2 else []

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and vals and str(device) == "cuda":
            try:
                vt = torch.tensor(vals, dtype=torch.float32, device=device).view(1, 1, -1)
                it = torch.tensor([int(i) for i in idxs[: len(vals)]], dtype=torch.long, device=device).view(1, 1, -1)
                out_len = (vt.shape[-1] - 1) * stride - 2 * padding + ksize
                y = torch.nn.functional.max_unpool1d(vt, it, kernel_size=ksize, stride=stride, padding=padding, output_size=(1, 1, int(out_len)))
                y = y.view(-1)
                try:
                    report("neuron", "maxunpool1d", {"in": int(vt.numel()), "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass
        return neuron._ensure_tensor(vals)


__all__ = ["MaxUnpool1DNeuronPlugin"]

