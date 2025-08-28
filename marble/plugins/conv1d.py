from __future__ import annotations

from typing import Any, List, Optional

from ..reporter import report


class Conv1DNeuronPlugin:
    """Pure-Python 1D convolution whose parameters are driven by connected neurons."""

    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"Conv1D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "conv1d_init", {"incoming": len(inc), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def _key_src(self, syn: "Synapse"):
        src = syn.source
        pos = getattr(src, "position", None)
        if isinstance(pos, tuple):
            return (0, tuple(pos))
        return (1, id(src))

    def _to_list1d(self, x) -> list:
        try:
            if hasattr(x, "detach") and hasattr(x, "tolist"):
                lst = x.detach().to("cpu").view(-1).tolist()
            elif isinstance(x, (list, tuple)):
                lst = list(x)
            else:
                lst = [x]
        except Exception:
            lst = []
        out = []
        for v in lst:
            try:
                out.append(float(v))
            except Exception:
                out.append(0.0)
        return out

    def _first_scalar(self, x, *, default: float = 0.0, min_val: Optional[float] = None) -> float:
        vals = self._to_list1d(x)
        v = float(vals[0]) if vals else float(default)
        if min_val is not None and v < min_val:
            v = float(min_val)
        return v

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) >= 5:
            param_incs.sort(key=self._key_src)
            sel = param_incs[:5]
        else:
            if len(incoming) < 5:
                raise ValueError("Conv1D plugin requires 5 incoming synapses (kernel, stride, padding, dilation, bias)")
            incoming.sort(key=self._key_src)
            sel = incoming[:5]
        k_src = sel[0].source
        s_src = sel[1].source
        p_src = sel[2].source
        d_src = sel[3].source
        b_src = sel[4].source

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
            x_list_all: List[float] = []
            for s in data_incs:
                x_list_all += self._to_list1d(getattr(s.source, "tensor", []))
            x1 = x_list_all
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x1 = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and str(device) == "cuda":
            try:
                xt = torch.tensor(x1, dtype=torch.float32, device=device).view(1, 1, -1)
                wt = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel, dtype=torch.float32, device=device)).view(1, 1, -1)
                bt_val = float(bias)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bt_val], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv1d(xt, wt, bias=bt, stride=stride, padding=padding, dilation=dilation)
                y = y.view(-1)
                try:
                    report("neuron", "conv1d", {"in": int(xt.numel()), "out": int(y.numel()), "k": int(wt.numel()), "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if padding > 0:
            x1 = ([0.0] * padding) + x1 + ([0.0] * padding)
        n = len(x1)
        klen = len(kernel)
        span = (klen - 1) * dilation + 1
        out_len = 0
        if n >= span:
            out_len = 1 + (n - span) // stride
        y_list: List[float] = []
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
        for t in range(out_len):
            base = t * stride
            acc = 0.0
            for i in range(klen):
                xi = base + i * dilation
                acc += kernel[i] * x1[xi]
            y_list.append(acc + bias)
        try:
            report("neuron", "conv1d", {"in": len(x1), "out": len(y_list), "k": klen, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)  # type: ignore[attr-defined]
        except Exception:
            return y_list


__all__ = ["Conv1DNeuronPlugin"]

