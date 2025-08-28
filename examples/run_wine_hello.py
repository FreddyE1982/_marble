from marble.marblemain import run_wine_hello_world


def main():
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
