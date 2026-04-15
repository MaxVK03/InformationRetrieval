import pandas as pd

import reccomender.config as config
from dataLoad.loader import load_preprocessed
from experiment.runner import run_all


def main():
    print("Loading data...")
    source_df, target_df = load_preprocessed(config.SOURCE_CSV, config.TARGET_CSV)
<<<<<<< HEAD

    source_df = source_df.rename(columns={"timestamp": "time"})
    target_df = target_df.rename(columns={"timestamp": "time"})
=======
>>>>>>> origin/main

    print(f"  {config.SOURCE_DOMAIN}: {len(source_df):,} interactions")
    print(f"  {config.TARGET_DOMAIN}: {len(target_df):,} interactions")

<<<<<<< HEAD
    print("  Source columns:", source_df.columns.tolist())
    print("  Target columns:", target_df.columns.tolist())

=======
>>>>>>> origin/main
    results_df = run_all(source_df, target_df)

    print("\n\n=== Final Results ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results_df.to_string(index=False))

    results_df.to_csv("results.csv", index=False)
    print("\nSaved to results.csv")


if __name__ == "__main__":
    main()