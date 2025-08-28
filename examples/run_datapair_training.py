"""
Example: Training with DataPairs and the Wanderer.

This script demonstrates how to:
- Create a simple Brain
- Build a few DataPair examples (left=input context, right=target)
- Run training via run_training_with_datapairs
- Print summary and inspect REPORTER logs

Note: The library’s internal rule is that only marble/marblemain.py performs
imports. This script is outside the package and may import from the package.
"""

from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    make_datapair,
    run_training_with_datapairs,
    REPORTER,
)


def main():
    b = Brain(2, size=(6, 6))
    codec = UniversalTensorCodec()

    pairs = [
        make_datapair({"label": 0}, 0.0),
        make_datapair({"label": 1}, 1.0),
        make_datapair({"label": 2}, 0.5),
    ]

    result = run_training_with_datapairs(
        b,
        pairs,
        codec,
        steps_per_pair=3,
        lr=5e-3,
        loss="nn.MSELoss",
    )

    print("example: datapair training final_loss=", result["final_loss"], "count=", result["count"])

    summary = REPORTER.item("datapair_summary", "training", "datapair")
    print("example: reporter summary:", summary)


if __name__ == "__main__":
    main()

