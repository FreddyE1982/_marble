from __future__ import annotations

import math
from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class ConvTranspose2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"ConvTranspose2D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "convtranspose2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("ConvTranspose2D requires 5 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        k_src, s_src, p_src, d_src, b_src = [s.source for s in param_incs[:5]]
        kernel_1d = self._to_list1d(getattr(k_src, "tensor", [])) or [1.0]
        stride = int(self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0))
        padding = int(max(0.0, self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)))
        dilation = int(self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0))
        bias = float(self._first_scalar(getattr(b_src, "tensor", 0.0), default=0.0))
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        learn_kernel = lstore.get("kernel")
        learn_bias = lstore.get("conv_bias")

        # Build 2D input from DATA rows
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
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, H, W)
                L = max(1, len(kernel_1d))
                r = int(math.isqrt(L))
                if r * r == L:
                    kh = kw = r
                else:
                    kh, kw = L, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv_transpose2d(
                    xt, wt, bias=bt, stride=(stride, stride), padding=(padding, padding), dilation=(dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "convtranspose2d", {"inHW": [H, W], "out": int(y.numel()), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        # Pure Python fallback (simplified)
        if hasattr(learn_kernel, "detach"):
            try:
                kernel_1d = list(learn_kernel.detach().to("cpu").view(-1).tolist())
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
        L = max(1, len(kernel_1d))
        r = int(math.isqrt(L))
        if r * r == L:
            kh = kw = r
        else:
            kh, kw = L, 1
        out_h = (H - 1) * stride - 2 * padding + (kh - 1) * dilation + 1
        out_w = (W - 1) * stride - 2 * padding + (kw - 1) * dilation + 1
        y2 = [[0.0 for _ in range(max(0, out_w))] for _ in range(max(0, out_h))]
        for iy in range(H):
            for ix in range(W):
                base_y = iy * stride
                base_x = ix * stride
                val = x_vals[iy * W + ix]
                for ky in range(kh):
                    for kx in range(kw):
                        oy = base_y + ky * dilation - padding
                        ox = base_x + kx * dilation - padding
                        if 0 <= oy < len(y2) and 0 <= ox < len(y2[0]):
                            y2[oy][ox] += kernel_1d[ky * kw + kx] * val
        y_list = [v + bias for row in y2 for v in row]
        try:
            report("neuron", "convtranspose2d", {"inHW": [H, W], "out": len(y_list), "kHW": [kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


__all__ = ["ConvTranspose2DNeuronPlugin"]

