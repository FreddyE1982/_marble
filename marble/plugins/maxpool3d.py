from __future__ import annotations

import math
from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class MaxPool3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 3 or len(out) != 1:
            raise ValueError(
                f"MaxPool3D neuron requires exactly 3 incoming PARAM synapses (kernel,stride,padding) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "maxpool3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 3:
            raise ValueError("MaxPool3D requires 3 incoming PARAM synapses")
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
            slices = []
            dims = []
            for s in data_incs:
                vals = self._to_list1d(getattr(s.source, "tensor", []))
                N = max(1, len(vals))
                r = int(math.isqrt(N))
                if r * r == N:
                    H = W = r
                else:
                    H, W = N, 1
                dims.append((H, W))
                slices.append(vals[: H * W])
            Hmin = min((h for h, _ in dims), default=0)
            Wmin = min((w for _, w in dims), default=0)
            x_vals: List[float] = []
            D = len(slices)
            for sl, (h, w) in zip(slices, dims):
                for rr in range(Hmin):
                    row = sl[rr * w:(rr + 1) * w]
                    x_vals.extend(row[:Wmin])
            H, W = Hmin, Wmin
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            vals = self._to_list1d(x)
            N = max(1, len(vals))
            r3 = round(N ** (1.0 / 3.0))
            if r3 > 0 and (r3 * r3 * r3) == N:
                D = H = W = int(r3)
                x_vals = vals[: D * H * W]
            else:
                D, H, W = N, 1, 1
                x_vals = vals

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        if torch is not None and H > 0 and W > 0:
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                y = torch.nn.functional.max_pool3d(xt, kernel_size=(ksize, ksize, ksize), stride=(stride, stride, stride), padding=(padding, padding, padding))
                y = y.view(-1)
                try:
                    report("neuron", "maxpool3d", {"inDHW": [D, H, W], "out": int(y.numel()), "k": ksize, "stride": stride, "pad": padding}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if padding > 0 and H > 0 and W > 0:
            D2, H2, W2 = D + 2 * padding, H + 2 * padding, W + 2 * padding
            padval = float("-inf")
            padded = [padval] * (D2 * H2 * W2)
            for dz in range(D):
                for yy in range(H):
                    for xx in range(W):
                        pd = dz + padding
                        py = yy + padding
                        px = xx + padding
                        padded[pd * (H2 * W2) + py * W2 + px] = x_vals[dz * (H * W) + yy * W + xx]
        else:
            padded = list(x_vals)
            D2, H2, W2 = D, H, W
        out_d = 0 if D2 < ksize else 1 + (D2 - ksize) // stride
        out_h = 0 if H2 < ksize else 1 + (H2 - ksize) // stride
        out_w = 0 if W2 < ksize else 1 + (W2 - ksize) // stride
        y_list = []
        for od in range(out_d):
            base_d = od * stride
            for oy in range(out_h):
                base_y = oy * stride
                for ox in range(out_w):
                    base_x = ox * stride
                    m = float("-inf")
                    for kz in range(ksize):
                        for ky in range(ksize):
                            for kx in range(ksize):
                                val = padded[(base_d + kz) * (H2 * W2) + (base_y + ky) * W2 + (base_x + kx)]
                                if val > m:
                                    m = val
                    y_list.append(m)
        try:
            report("neuron", "maxpool3d", {"inDHW": [D, H, W], "out": len(y_list), "k": ksize, "stride": stride, "pad": padding}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


__all__ = ["MaxPool3DNeuronPlugin"]

