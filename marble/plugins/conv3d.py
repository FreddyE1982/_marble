from __future__ import annotations

import math
from typing import List, Tuple

from ..reporter import report
from .conv_common import _ConvNDCommon


class Conv3DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 5 or len(out) != 1:
            raise ValueError(
                f"Conv3D neuron requires exactly 5 incoming PARAM synapses and exactly 1 outgoing synapse; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "conv3d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 5:
            raise ValueError("Conv3D plugin requires 5 incoming PARAM synapses")
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

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            slices: List[List[float]] = []
            dims: List[Tuple[int, int]] = []
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
            if dims:
                Hmin = min(h for h, _ in dims)
                Wmin = min(w for _, w in dims)
            else:
                Hmin = Wmin = 0
            x_vals = []
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
        if torch is not None and H > 0 and W > 0 and str(device) == "cuda":
            try:
                xt = torch.tensor(x_vals, dtype=torch.float32, device=device).view(1, 1, D, H, W)
                L = max(1, len(kernel_1d))
                r = round(L ** (1.0 / 3.0))
                if r > 0 and (r * r * r) == L:
                    kd = kh = kw = int(r)
                else:
                    kd, kh, kw = L, 1, 1
                base_w = (learn_kernel.to(device) if hasattr(learn_kernel, "to") else torch.tensor(kernel_1d, dtype=torch.float32, device=device))
                wt = base_w.view(1, 1, kd, kh, kw)
                bt = (learn_bias.to(device).view(-1) if hasattr(learn_bias, "to") else torch.tensor([bias], dtype=torch.float32, device=device))
                y = torch.nn.functional.conv3d(
                    xt, wt, bias=bt, stride=(stride, stride, stride), padding=(padding, padding, padding), dilation=(dilation, dilation, dilation)
                )
                y = y.view(-1)
                try:
                    report("neuron", "conv3d", {"inDHW": [D, H, W], "out": int(y.numel()), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if padding > 0 and H > 0 and W > 0:
            D2, H2, W2 = D + 2 * padding, H + 2 * padding, W + 2 * padding
            padded = [0.0] * (D2 * H2 * W2)
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
        r = round(L ** (1.0 / 3.0))
        if r > 0 and (r * r * r) == L:
            kd = kh = kw = int(r)
        else:
            kd, kh, kw = L, 1, 1
        def idx3(a, d, h, w, H_, W_):
            return a[d * (H_ * W_) + h * W_ + w]
        out_d = 0 if D2 < kd else 1 + (D2 - ((kd - 1) * dilation + 1)) // stride
        out_h = 0 if H2 < kh else 1 + (H2 - ((kh - 1) * dilation + 1)) // stride
        out_w = 0 if W2 < kw else 1 + (W2 - ((kw - 1) * dilation + 1)) // stride
        y_list: List[float] = []
        for od in range(out_d):
            base_d = od * stride
            for oy in range(out_h):
                base_y = oy * stride
                for ox in range(out_w):
                    base_x = ox * stride
                    acc = 0.0
                    for kz in range(kd):
                        for ky in range(kh):
                            for kx in range(kw):
                                iz = base_d + kz * dilation
                                iy = base_y + ky * dilation
                                ix = base_x + kx * dilation
                                acc += kernel_1d[(kz * kh + ky) * kw + kx] * idx3(padded, iz, iy, ix, H2, W2)
                    y_list.append(acc + bias)
        try:
            report("neuron", "conv3d", {"inDHW": [D, H, W], "out": len(y_list), "kDHW": [kd, kh, kw], "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        try:
            return neuron._ensure_tensor(y_list)
        except Exception:
            return y_list


__all__ = ["Conv3DNeuronPlugin"]

