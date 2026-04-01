import pandas as pd

import reccomender.config as config
from dataLoad.loader import load_domain
from dataLoad.preproccesing import filter_shared_users
from experiment.runner import run_all


def main():
    print("Loading data...")
    source_df = load_domain(config.BOOKS_PATH,  config.SOURCE_DOMAIN, config.CHUNK_SIZE)
    target_df = load_domain(config.MOVIES_PATH, config.TARGET_DOMAIN, config.CHUNK_SIZE)

    print(f"  {config.SOURCE_DOMAIN}: {len(source_df):,} interactions")
    print(f"  {config.TARGET_DOMAIN}: {len(target_df):,} interactions")

    source_df, target_df = filter_shared_users(source_df, target_df)

    print(f"\nAfter shared-user filter:")
    print(f"  {config.SOURCE_DOMAIN}: {len(source_df):,} interactions")
    print(f"  {config.TARGET_DOMAIN}: {len(target_df):,} interactions")

    results_df = run_all(source_df, target_df)

    print("\n\n=== Final Results ===")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(results_df.to_string(index=False))

    results_df.to_csv("results.csv", index=False)
    print("\nSaved to results.csv")


if __name__ == "__main__":
    main()