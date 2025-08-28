from __future__ import annotations

from typing import List

from ..reporter import report
from .conv_common import _ConvNDCommon


class Fold2DNeuronPlugin(_ConvNDCommon):
    def on_init(self, neuron: "Neuron") -> None:
        inc = list(getattr(neuron, "incoming", []) or [])
        out = list(getattr(neuron, "outgoing", []) or [])
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in inc if is_param(s)]
        if len(param_incs) != 6 or len(out) != 1:
            raise ValueError(
                f"Fold2D neuron requires exactly 6 incoming PARAM synapses (out_h,out_w,kernel,stride,padding,dilation) and exactly 1 outgoing; got params={len(param_incs)} out={len(out)}"
            )
        try:
            report("neuron", "fold2d_init", {"incoming_params": len(param_incs), "outgoing": len(out)}, "plugins")
        except Exception:
            pass

    def forward(self, neuron: "Neuron", input_value=None):
        incoming = list(getattr(neuron, "incoming", []))
        def is_param(s):
            t = getattr(s, "type_name", None)
            return isinstance(t, str) and t.startswith("param")
        param_incs = [s for s in incoming if is_param(s)]
        if len(param_incs) < 6:
            raise ValueError("Fold2D requires 6 incoming PARAM synapses")
        param_incs.sort(key=self._key_src)
        oh_src, ow_src, k_src, s_src, p_src, d_src = [s.source for s in param_incs[:6]]
        base_oh = self._first_scalar(getattr(oh_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_ow = self._first_scalar(getattr(ow_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_ks = self._first_scalar(getattr(k_src, "tensor", 2.0), default=2.0, min_val=1.0)
        base_st = self._first_scalar(getattr(s_src, "tensor", 1.0), default=1.0, min_val=1.0)
        base_pd = self._first_scalar(getattr(p_src, "tensor", 0.0), default=0.0)
        base_dl = self._first_scalar(getattr(d_src, "tensor", 1.0), default=1.0, min_val=1.0)
        lstore = getattr(neuron, "_plugin_state", {}).get("learnable_params", {})
        def to_int(val, default, minv):
            try:
                return int(max(minv, round(float(val.detach().to("cpu").view(-1)[0].item())))) if hasattr(val, "detach") else int(max(minv, round(default)))
            except Exception:
                return int(max(minv, round(default)))
        out_h = to_int(lstore.get("out_h"), base_oh, 1)
        out_w = to_int(lstore.get("out_w"), base_ow, 1)
        ksize = to_int(lstore.get("kernel_size"), base_ks, 1)
        stride = to_int(lstore.get("stride"), base_st, 1)
        padding = to_int(lstore.get("padding"), base_pd, 0)
        dilation = to_int(lstore.get("dilation"), base_dl, 1)

        data_incs = [s for s in incoming if getattr(s, "type_name", None) == "data"]
        if data_incs:
            data_incs.sort(key=self._key_src)
            cols: List[float] = []
            for s in data_incs:
                cols += self._to_list1d(getattr(s.source, "tensor", []))
        else:
            x = input_value if input_value is not None else getattr(neuron, "tensor", [])
            cols = self._to_list1d(x)

        torch = getattr(neuron, "_torch", None)
        device = getattr(neuron, "_device", "cpu")
        kk = max(1, ksize * ksize)
        L = len(cols) // kk if len(cols) >= kk else 0
        if torch is not None and L > 0:
            try:
                ct = torch.tensor(cols[: L * kk], dtype=torch.float32, device=device).view(1, 1 * kk, L)
                y = torch.nn.functional.fold(ct, output_size=(out_h, out_w), kernel_size=(ksize, ksize), dilation=(dilation, dilation), padding=(padding, padding), stride=(stride, stride))
                y = y.view(-1)
                try:
                    report("neuron", "fold2d", {"outHW": [out_h, out_w], "in_cols": L, "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
                except Exception:
                    pass
                return y
            except Exception:
                pass

        if L <= 0:
            return neuron._ensure_tensor([]) if hasattr(neuron, "_ensure_tensor") else []
        out = [[0.0 for _ in range(out_w)] for _ in range(out_h)]
        idx = 0
        def npos(size):
            cnt = 0
            pos = 0
            span = (ksize - 1) * dilation + 1
            while pos + span <= size:
                cnt += 1
                pos += stride
            return cnt
        nh = npos(out_h)
        nw = npos(out_w)
        total = nh * nw
        used = min(L, total)
        for oy in range(nh):
            base_y = oy * stride
            for ox in range(nw):
                if idx >= used:
                    break
                base_x = ox * stride
                for ky in range(ksize):
                    for kx in range(ksize):
                        iy = base_y + ky * dilation
                        ix = base_x + kx * dilation
                        if 0 <= iy < out_h and 0 <= ix < out_w:
                            out[iy][ix] += cols[idx * kk + ky * ksize + kx]
                idx += 1
        flat = [v for row in out for v in row]
        try:
            report("neuron", "fold2d", {"outHW": [out_h, out_w], "in_cols": used, "k": ksize, "stride": stride, "pad": padding, "dil": dilation}, "plugins")
        except Exception:
            pass
        return neuron._ensure_tensor(flat)


__all__ = ["Fold2DNeuronPlugin"]

