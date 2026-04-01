import numpy as np
import pandas as pd

from evaluation.metrics import (
    recall_at_k,
    precision_at_k,
    ndcg_at_k,
    mean_average_precision,
    mean_reciprocal_rank,
)

def evaluate(model, test_df: pd.DataFrame, k: int = 10) -> dict:
    """
    Generic evaluation loop.

    Every model must implement:
        model.recommend(user, k) -> list[item]

    Parameters
    ----------
    model   : any model with a .recommend(user, k) method
    test_df : DataFrame with columns ["user", "item"]
    k       : cutoff for all metrics

    Returns
    -------
    dict with keys:
        users_evaluated, Precision@k, Recall@k, NDCG@k, MAP@k, MRR@k
    """
    recalls, precisions, ndcgs = [], [], []
    all_recs: dict = {}
    ground_truths: dict = {}

    for _, row in test_df.iterrows():
        user    = row["user"]
        gt_item = row["item"]

        recs = model.recommend(user, k=k)

        all_recs[user]      = recs
        ground_truths[user] = [gt_item]

        recalls.append(recall_at_k(recs, gt_item))
        precisions.append(precision_at_k(recs, gt_item, k))
        ndcgs.append(ndcg_at_k(recs, gt_item))

    n = len(recalls)
    return {
        "users_evaluated": n,
        f"Precision@{k}":  float(np.mean(precisions)) if n else 0.0,
        f"Recall@{k}":     float(np.mean(recalls))    if n else 0.0,
        f"NDCG@{k}":       float(np.mean(ndcgs))      if n else 0.0,
        f"MAP@{k}":        mean_average_precision(all_recs, ground_truths, k),
        f"MRR@{k}":        mean_reciprocal_rank(all_recs, ground_truths, k),
    }