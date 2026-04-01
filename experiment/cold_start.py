import pandas as pd
from pathlib import Path


def _load_split(split_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    def read(name):
        return (
            pd.read_csv(split_dir / name)
            .rename(columns={"user_id": "user", "item_id": "item"})
        )
    return read("support.csv"), read("query.csv")


def build_single_domain_frames(cold_k: int, splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    return _load_split(splits_dir / f"target_k_{cold_k}")


def build_cross_domain_frames(source_df: pd.DataFrame, cold_k: int, splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    support, query = _load_split(splits_dir / f"target_k_{cold_k}")
    train = pd.concat([source_df, support], ignore_index=True)
    return train, query