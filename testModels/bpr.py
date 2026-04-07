import numpy as np
import pandas as pd
from implicit.bpr import BayesianPersonalizedRanking

from dataLoad.preproccesing import build_sparse_matrix
import reccomender.config as config


class BPRRecommender:
    """
    Wrapper around `implicit` BayesianPersonalizedRanking.

    Works for both single-domain and joint (cross-domain) training:
    pass whatever train_df you want (target only, or source + target).
    Item IDs are kept as strings so the caller can filter by domain prefix.
    """

    def __init__(
        self,
        factors: int    = config.BPR_FACTORS,
        lr: float       = config.BPR_LR,
        reg: float      = config.BPR_REG,
        iterations: int = config.BPR_ITERATIONS,
    ):
        self.factors    = factors
        self.lr         = lr
        self.reg        = reg
        self.iterations = iterations

        self._model      = None
        self._user_items = None
        self._user2id: dict[str, int] = {}
        self._item2id: dict[str, int] = {}
        self._id2item: dict[int, str] = {}

    def fit(self, train_df: pd.DataFrame) -> "BPRRecommender":
        _, self._user_items, self._user2id, self._item2id, self._id2item = (
            build_sparse_matrix(train_df)
        )

        self._model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.lr,
            regularization=self.reg,
            iterations=self.iterations,
            random_state=42,
        )
        # implicit expects item-user matrix
        self._model.fit(self._user_items, show_progress=True)
        return self

    def recommend(
        self,
        user: str,
        k: int = 10,
        target_prefix: str | None = None,
    ) -> list[str]:
        if user not in self._user2id:
            return []

        user_id  = self._user2id[user]
        n_items  = self._user_items.shape[1]

        # Request more candidates when filtering to a domain prefix
        n_candidates = min(200, n_items) if target_prefix else k

        rec_ids, _ = self._model.recommend(
            userid=user_id,
            user_items=self._user_items[user_id],
            N=n_candidates,
            filter_already_liked_items=True,
        )

        items = [self._id2item[i] for i in rec_ids.tolist()]

        if target_prefix:
            items = [it for it in items if it.startswith(target_prefix)]

        return items[:k]