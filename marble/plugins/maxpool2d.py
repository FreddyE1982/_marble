from __future__ import annotations

import math
from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class MaxPool2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool2D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool2D requires 3 incoming PARAM synapses")
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
            rh = int(math.isqrt(N))
            if rh * rh == N:
                H = W = rh
            else:
                H, W = N, 1

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                y = torch.nn.functional.max_pool2d(xt, kernel_size=(ksize, ksize), stride=(stride, stride), padding=(padding, padding))
                y = y.view(-1)
                try:
                    report("neuron", "maxpool2d", {"inHW": [H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if H <= 0 or W <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        if padding > 0:
            padded: List[List[float]] = []
            zero_row = [float("-inf")] * (W + 2 * padding)
            for _ in range(padding):
                padded.append(list(zero_row))
            for r_ in range(H):
                row = [float("-inf")] * padding + x_vals[r_ * W:(r_ + 1) * W] + [float("-inf")] * padding
                padded.append(row)
            for _ in range(padding):
                padded.append(list(zero_row))
            H2, W2 = len(padded), len(padded[0])
        else:
            padded = [x_vals[r_ * W:(r_ + 1) * W] for r_ in range(H)]
            H2, W2 = H, W
        out_h = 0 if H2 < ksize else 1 + (H2 - ksize) // stride
        out_w = 0 if W2 < ksize else 1 + (W2 - ksize) // stride
        y2 = []
        for oy in range(out_h):
            base_y = oy * stride
            for ox in range(out_w):
                base_x = ox * stride
                m = float("-inf")
                for ky in range(ksize):
                    for kx in range(ksize):
                        vy = padded[base_y + ky][base_x + kx]
                        if vy > m:
                            m = vy
                y2.append(m)
        try:
            report("neuron", "maxpool2d", {"inHW": [H, W], "out": len(y2), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y2)
        except Exception:
            return y2


__all__ = ["MaxPool2DNeuronPlugin"]

