import urllib.request
from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs
from marble.marblemain import make_datapair, report


def _download_text(url: str, timeout: float = 15.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = resp.read()
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin1", errors="ignore")


def _build_text_prediction_pairs_from_corpus(text: str, *, window: int = 32, max_pairs: int = 500):
    pairs = []
    n = len(text)
    if n <= window + 1:
        return pairs
    step = max(1, (n - window - 1) // max(1, max_pairs))
    idx = 0
    while idx + window + 1 <= n and len(pairs) < max_pairs:
        left = text[idx : idx + window]
        right = text[idx + window]
        pairs.append(make_datapair(left, right))
        idx += step
    try:
        report("examples", "text_pairs", {"count": len(pairs), "window": window}, "text")
    except Exception:
        pass
    return pairs


def main():
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = _download_text(url)
    pairs = _build_text_prediction_pairs_from_corpus(text, window=64, max_pairs=500)
    brain = Brain(2, size=(8, 8))
    codec = UniversalTensorCodec()
    res = run_training_with_datapairs(
        brain,
        pairs,
        codec,
        steps_per_pair=3,
        lr=1e-2,
        seed=0,
    )
    print("text prediction final_loss:", res.get("final_loss"))


if __name__ == "__main__":
    main()
