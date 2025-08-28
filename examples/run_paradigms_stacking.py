from marble.marblemain import (
    Brain,
    UniversalTensorCodec,
    run_training_with_datapairs,
)


def main():
    # Build a tiny dataset of pairs
    data = [("foo", "bar"), ("baz", "qux"), ("lorem", "ipsum")]
    codec = UniversalTensorCodec()

    # Create brain and stack multiple paradigms
    b = Brain(2, size=(8, 8))
    b.load_paradigm("contrastive", {"contrastive_tau": 0.1, "contrastive_lambda": 0.5})
    b.load_paradigm("reinforcement", {"rl_epsilon": 0.1, "rl_alpha": 0.2, "rl_gamma": 0.95})
    b.load_paradigm("student_teacher", {"distill_lambda": 0.05, "teacher_momentum": 0.9})
    b.load_paradigm("hebbian", {"hebbian_eta": 0.02, "hebbian_decay": 0.001})

    # Run a short training pass; paradigms are applied automatically
    res = run_training_with_datapairs(
        b,
        data,
        codec,
        steps_per_pair=3,
        lr=1e-2,
        wanderer_type="l2_weight_penalty,wanderalongsynapseweights",
    )
    print("stacking final_loss:", res.get("final_loss"), "count:", res.get("count"))


if __name__ == "__main__":
    main()

