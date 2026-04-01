import pandas as pd

from dataLoad.loader import load_target_split
from dataLoad.preproccesing import (
    filter_users_for_cold_start,
    make_target_cold_start_split,
    filter_test_seen_in_train,
)

def build_single_domain_frames(cold_k, splits_dir):
    support, query = load_target_split(splits_dir, cold_k)
    return support, query

def build_cross_domain_frames(source_df, cold_k, splits_dir):
    support, query = load_target_split(splits_dir, cold_k)
    train = pd.concat([source_df, support], ignore_index=True)
    return train, query