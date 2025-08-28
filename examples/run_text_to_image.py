import tarfile, io, urllib.request
from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs, make_datapair, report


def _download_and_extract(url: str, *, member_predicate=None, timeout: float = 60.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    by = io.BytesIO(data)
    out = {}
    with tarfile.open(fileobj=by, mode="r:gz") as tar:
        for m in tar.getmembers():
            if not m.isfile():
                continue
            name = m.name
            if member_predicate is not None and not member_predicate(name):
                continue
            f = tar.extractfile(m)
            if f is None:
                continue
            out[name] = f.read()
    return out


def _build_text_to_image_pairs_from_mnistpng(n: int = 200):
    url = "https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz"
    files = _download_and_extract(url, member_predicate=lambda s: ("training" in s or "testing" in s) and s.lower().endswith(".png"))
    pairs = []
    for path, by in files.items():
        parts = path.replace("\\", "/").split("/")
        label = parts[-2] if len(parts) >= 2 else "unknown"
        pairs.append(make_datapair(label, by))
        if len(pairs) >= n:
            break
    try:
        report("examples", "t2i_pairs", {"count": len(pairs)}, "mnist_png")
    except Exception:
        pass
    return pairs


def main():
    pairs = _build_text_to_image_pairs_from_mnistpng(n=200)
    brain = Brain(2, size=(10, 10))
    codec = UniversalTensorCodec()
    res = run_training_with_datapairs(
        brain,
        pairs,
        codec,
        steps_per_pair=3,
        lr=1e-2,
        seed=0,
    )
    print("text->image final_loss:", res.get("final_loss"))


if __name__ == "__main__":
    main()
