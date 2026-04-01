# experiment/cold_start.py
import pandas as pd
from pathlib import Path


def build_single_domain_frames(cold_k: int, splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    split_dir = splits_dir / f"target_k_{cold_k}"
    support = pd.read_csv(split_dir / "support.csv")
    query   = pd.read_csv(split_dir / "query.csv")
    return support, query


def build_cross_domain_frames(source_df: pd.DataFrame, cold_k: int, splits_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    support, query = build_single_domain_frames(cold_k, splits_dir)
    train = pd.concat([source_df, support], ignore_index=True)
    return train, query