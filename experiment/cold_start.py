import pandas as pd

from dataLoad.preproccesing import (
    filter_users_for_cold_start,
    make_target_cold_start_split,
    filter_test_seen_in_train,
)


def _count_source_after_query_time(
    source_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> tuple[int, int]:
    query_times = query_df[["user", "time"]].rename(columns={"time": "query_time"})
    merged = source_df.merge(query_times, on="user", how="inner")
    mask = merged["time"] > merged["query_time"]
    return int(mask.sum()), int(merged.loc[mask, "user"].nunique())


def _trim_source_to_query_time(
    source_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only source interactions that occur on/before each user's
    target query timestamp to avoid temporal leakage.
    """
    query_times = query_df[["user", "time"]].rename(columns={"time": "query_time"})
    merged = source_df.merge(query_times, on="user", how="inner")
    trimmed = merged.loc[merged["time"] <= merged["query_time"]].drop(columns=["query_time"])
    return trimmed.reset_index(drop=True)


def build_single_domain_frames(
    target_df: pd.DataFrame,
    source_df: pd.DataFrame,
    cold_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, target_filtered, _ = filter_users_for_cold_start(source_df, target_df, cold_k)
    target_train, target_test = make_target_cold_start_split(target_filtered, cold_k)
    target_test = filter_test_seen_in_train(target_train, target_test)
    return target_train, target_test

def build_cross_domain_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cold_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_filtered, target_filtered, _ = filter_users_for_cold_start(source_df, target_df, cold_k)
    target_train, target_test = make_target_cold_start_split(target_filtered, cold_k)

    # Remove per-user source events that happen after the held-out target query.
    source_filtered = _trim_source_to_query_time(source_filtered, target_test)
    leak_rows, leak_users = _count_source_after_query_time(source_filtered, target_test)
    if leak_rows != 0 or leak_users != 0:
        raise RuntimeError("Temporal leakage check failed after source trimming.")

    train = pd.concat([source_filtered, target_train], ignore_index=True)
    target_test = filter_test_seen_in_train(train, target_test)
    return train, target_test
