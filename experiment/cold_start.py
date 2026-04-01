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
    """
    Single-domain baseline: train on target interactions only.

    Returns (train_df, test_df) filtered so that:
        - each user has exactly cold_k target train interactions
        - test item is always seen in train
    """
    _, target_f, _ = filter_users_for_cold_start(source_df, target_df, cold_k)
    target_train, target_test = make_target_cold_start_split(target_f, cold_k)
    test = filter_test_seen_in_train(target_train, target_test)
    return target_train, test


def build_cross_domain_frames(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cold_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cross-domain model: train on all source + cold_k target interactions.

    Returns (train_df, test_df) filtered so that:
        - each user has all source interactions + cold_k target train interactions
        - test item is always seen in train
    """
    source_f, target_f, _ = filter_users_for_cold_start(source_df, target_df, cold_k)
    target_train, target_test = make_target_cold_start_split(target_f, cold_k)

    train = pd.concat(
        [
            source_f[["user", "item", "time"]],
            target_train[["user", "item", "time"]],
        ],
        ignore_index=True,
    )
    test = filter_test_seen_in_train(train, target_test)
    return train, test