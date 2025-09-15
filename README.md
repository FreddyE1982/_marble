# _marble

## Tracking custom tensors

The resource allocator plugin can juggle not only graph tensors but also any
custom buffers you allocate in your scripts. Wrap assignments in
`track_tensor` so the allocator can offload them when VRAM gets tight.

```python
from marble.plugins import wanderer_resource_allocator as resource_allocator
import torch

class Holder:
    buf: torch.Tensor | None = None

holder = Holder()
with resource_allocator.track_tensor(holder, "buf"):
    holder.buf = torch.zeros(
        1024, device="cuda" if torch.cuda.is_available() else "cpu"
    )
```

After the context exits the buffer is registered and may be moved between GPU,
CPU or even disk automatically.

## Visualising brain snapshots

You can turn a ``.marble`` snapshot into a quick topology image using
``snapshot_to_image``.

```python
from marble import snapshot_to_image

png_path = snapshot_to_image("snapshot_123.marble", "topology.png")
print("saved image to", png_path)
```

The helper arranges neurons in a circle and draws synapses as straight lines,
which is handy for debugging and inspecting small brains.
