from __future__ import annotations

from typing import List

from ..reporter import report
from .conv1d import Conv1DNeuronPlugin


class ConvTranspose1DNeuronPlugin(Conv1DNeuronPlugin):
    def on_init(self, neuron: "Neuron") -> None:
        # Reuse strict rule: 5 PARAM + 1 outgoing
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"ConvTranspose1D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "convtranspose1d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        # Gather params and data using Conv1D helpers
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("ConvTranspose1D requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]
        kernel = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

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
        if torch is not None and str(device) == "cuda":
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                wt = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel, dtype=torch.float32, device=device)).view(1, 1, -1)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv_transpose1d(xt, wt, bias=bt, stride=stride, padding=padding, dilation=dilation)
                y = y.view(-1)
                try:
                    report("neuron", "convtranspose1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": int(wt.numel()), "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure-Python fallback
        n = len(x1)
        klen = len(kernel)
        out_len = (n - 1) * stride - 2 * padding + (klen - 1) * dilation + 1
        y_list = [0.0] * max(0, out_len)
        if hasattr(learn_kernel, "detach"):
            try:
                kernel = list(learn_kernel.detach().to("cpu").view(-1).tolist())
                klen = len(kernel)
            except Exception:
                pass
        if hasattr(learn_bias, "detach"):
            try:
                bias = float(learn_bias.detach().to("cpu").view(-1)[0].item())
            except Exception:
                try:
                    bias = float(learn_bias)
                except Exception:
                    pass
        for t in range(n):
            base = t * stride
            for i in range(klen):
                oi = base + i * dilation - padding
                if 0 <= oi < len(y_list):
                    y_list[oi] += kernel[i] * x1[t]
        y_list = [v + bias for v in y_list]
        try:
            report("neuron", "convtranspose1d", {"in": n, "out": len(y_list), "k": klen, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


__all__ = ["ConvTranspose1DNeuronPlugin"]

