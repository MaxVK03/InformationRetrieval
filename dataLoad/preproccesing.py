import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# Shared-user filtering

def filter_shared_users(source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep only users that appear in both domains."""
    shared = set(source_df["user"]) & set(target_df["user"])
    return (
        source_df[source_df["user"].isin(shared)].copy(),
        target_df[target_df["user"].isin(shared)].copy(),
    )


# Train / test split

def leave_one_out_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sort by (user, time) and hold out each user's last interaction as test.
    Returns (train_df, test_df).
    """
    df = df.sort_values(["user", "time"]).copy()
    test  = df.groupby("user").tail(1).copy()
    train = df.drop(test.index).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


# Cold-start filtering

def filter_users_for_cold_start(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cold_k: int,
    min_source: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    Keep users that have:
        - at least `min_source` interactions in source
        - at least `cold_k + 1` interactions in target
          (cold_k for train, 1 held out for test)
    """
    source_counts = source_df["user"].value_counts()
    target_counts = target_df["user"].value_counts()

    valid = (
        set(source_counts[source_counts >= min_source].index)
        & set(target_counts[target_counts >= cold_k + 1].index)
    )

    return (
        source_df[source_df["user"].isin(valid)].copy(),
        target_df[target_df["user"].isin(valid)].copy(),
        valid,
    )


def make_target_cold_start_split(
    target_df: pd.DataFrame, cold_k: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each user in target_df:
        - last interaction  → test
        - first cold_k earlier interactions → train
    Returns (target_train, target_test).
    """
    target_df = target_df.sort_values(["user", "time"]).copy()
    test      = target_df.groupby("user").tail(1).copy()
    remaining = target_df.drop(test.index).copy()
    train     = remaining.groupby("user", group_keys=False).head(cold_k).copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)


# Test filtering

def filter_test_seen_in_train(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Drop test rows whose user or item does not appear in train.
    Models cannot rank completely unseen entities.
    """
    train_users = set(train_df["user"].unique())
    train_items = set(train_df["item"].unique())
    mask = test_df["user"].isin(train_users) & test_df["item"].isin(train_items)
    return test_df[mask].reset_index(drop=True)


# Sparse matrix builder (used by BPR models)

def build_sparse_matrix(
    train_df: pd.DataFrame,
) -> tuple[pd.DataFrame, csr_matrix, dict, dict, dict]:
    """
    Tightly remap user/item string IDs to contiguous integers and build a
    CSR user-item interaction matrix.

    Returns:
        train_mapped  – DataFrame with added integer user_id / item_id columns
        user_items    – csr_matrix (num_users × num_items)
        user2id       – {user_str: int}
        item2id       – {item_str: int}
        id2item       – {int: item_str}
    """
    df = train_df.copy()

    user_ids = sorted(df["user"].unique())
    item_ids = sorted(df["item"].unique())

    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {it: i for i, it in enumerate(item_ids)}
    id2item = {i: it for it, i in item2id.items()}

    df["user_id"] = df["user"].map(user2id).astype(int)
    df["item_id"] = df["item"].map(item2id).astype(int)

    n_users = df["user_id"].max() + 1
    n_items = df["item_id"].max() + 1

    user_items = csr_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (df["user_id"], df["item_id"]),
        ),
        shape=(n_users, n_items),
    )

    return df, user_items, user2id, item2id, id2item