from marble.marblemain import run_wine_hello_world
from marble.plugins import wanderer_resource_allocator as resource_allocator
import torch


def main():
    # Demonstrate tracking a custom tensor so the resource allocator can move
    # it between devices when needed.
    class _Buf:
        buf = None

    holder = _Buf()
    with resource_allocator.track_tensor(holder, "buf"):
        holder.buf = torch.zeros(
            128, device="cuda" if torch.cuda.is_available() else "cpu"
        )

    res = run_wine_hello_world(
        log_path="wanderer_steps.jsonl",
        num_pairs=50,
        steps_per_pair=5,
        lr=5e-3,
        seed=42,
    )
    print("wine hello world final_loss:", res.get("final_loss"))


if __name__ == "__main__":
    main()
