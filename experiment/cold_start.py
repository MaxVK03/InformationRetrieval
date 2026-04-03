import pandas as pd

from dataLoad.preproccesing import (
    filter_users_for_cold_start,
    make_target_cold_start_split,
    filter_test_seen_in_train,
)

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
    train = pd.concat([source_filtered, target_train], ignore_index=True)
    target_test = filter_test_seen_in_train(train, target_test)
    return train, target_test
