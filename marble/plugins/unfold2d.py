from __future__ import annotations

import math
from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class Unfold2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 4 or len(out) != 1:
            raise ValueError(
                f"Unfold2D neuron requires exactly 4 incoming PARAM synapses (kernel,stride,padding,dilation) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "unfold2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 4:
            raise ValueError("Unfold2D requires 4 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src = [s.source for s in param_incs[:4]]
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        base_dl = self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        lk = lstore.get("kernel_size")
        ls = lstore.get("stride")
        lp = lstore.get("padding")
        ld = lstore.get("dilation")
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        ksize = to_int(lk, base_ks, 1)
        stride = to_int(ls, base_st, 1)
        padding = to_int(lp, base_pd, 0)
        dilation = to_int(ld, base_dl, 1)

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            rows = [self._to_list1d(getattr(s.source, "tensor", [])) for s in data_incs]
            width = min((len(r) for r in rows if r), default=0)
            if width <= 0:
                x_vals: List[float] = []
                H = W = 0
            else:
                rows = [r[:width] for r in rows]
                H = len(rows)
                W = width
                x_vals = [v for r in rows for v in r]
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            x_vals = self._to_list1d(x)
            N = max(1, len(x_vals))
            r = int(math.isqrt(N))
            if r * r == N:
                H = W = r
            else:
                H, W = N, 1

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                y = torch.nn.functional.unfold(xt, kernel_size=(ksize, ksize), dilation=(dilation, dilation), padding=(padding, padding), stride=(stride, stride))
                y = y.view(-1)
                try:
                    report("neuron", "unfold2d", {"inHW": [H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if H <= 0 or W <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        if padding > 0:
            padded = []
            zero_row = [0.0] * (W + 2 * padding)
            for _ in range(padding):
                padded.append(list(zero_row))
            for r_ in range(H):
                row = [0.0] * padding + x_vals[r_ * W:(r_ + 1) * W] + [0.0] * padding
                padded.append(row)
            for _ in range(padding):
                padded.append(list(zero_row))
            H2, W2 = len(padded), len(padded[0])
        else:
            padded = [x_vals[r_ * W:(r_ + 1) * W] for r_ in range(H)]
            H2, W2 = H, W
        out_h = 0 if H2 < ((ksize - 1) * dilation + 1) else 1 + (H2 - ((ksize - 1) * dilation + 1)) // stride
        out_w = 0 if W2 < ((ksize - 1) * dilation + 1) else 1 + (W2 - ((ksize - 1) * dilation + 1)) // stride
        cols: List[float] = []
        for oy in range(out_h):
            base_y = oy * stride
            for ox in range(out_w):
                base_x = ox * stride
                for ky in range(ksize):
                    for kx in range(ksize):
                        iy = base_y + ky * dilation
                        ix = base_x + kx * dilation
                        cols.append(padded[iy][ix])
        try:
            report("neuron", "unfold2d", {"inHW": [H, W], "out": len(cols), "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        return neuron._ensure_tensor(cols)


__all__ = ["Unfold2DNeuronPlugin"]

