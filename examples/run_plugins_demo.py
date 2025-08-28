from marble.marblemain import Brain, UniversalTensorCodec, run_training_with_datapairs


def main():
    b = Brain(2, size=(10, 10))
    # Load paradigms: supervised conv insertion, epsilon-greedy exploration, and evolutionary path growth
    b.load_paradigm("supervised_conv", {"period": 5, "eval_after": 3})
    b.load_paradigm("epsilon_greedy", {"epsilongreedy_epsilon": 0.2})
    b.load_paradigm(
        "evolutionary_paths",
        {"altpaths_min_len": 2, "altpaths_max_len": 3, "altpaths_max_paths_per_walk": 1, "mutate_prob": 0.3, "mutate_scale": 0.2},
    )

    data = [("foo", "bar"), ("baz", "qux"), ("lorem", "ipsum")]
    codec = UniversalTensorCodec()
    res = run_training_with_datapairs(b, data, codec, steps_per_pair=3, lr=1e-2, wanderer_type="bestlosspath,wanderalongsynapseweights")
    print("demo final_loss:", res["final_loss"], "count:", res["count"])


if __name__ == "__main__":
    main()

