import numpy as np
import pandas as pd
from libreco.data import DatasetPure
from libreco.algorithms import LightGCN

import reccomender.config as config


class LightGCNRecommender:
    """
    Wrapper around LibRecommender's LightGCN.

    Expects train_df with columns: user, item, label (1.0), time.
    Works for both single-domain and cross-domain training.
    """

    def __init__(
        self,
        embed_size: int  = config.LGCN_EMBED_SIZE,
        n_epochs: int    = config.LGCN_N_EPOCHS,
        lr: float        = config.LGCN_LR,
        batch_size: int  = config.LGCN_BATCH_SIZE,
        num_neg: int     = config.LGCN_NUM_NEG,
        device: str      = config.LGCN_DEVICE,
        seed: int        = config.LGCN_SEED,
    ):
        self.embed_size = embed_size
        self.n_epochs   = n_epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.num_neg    = num_neg
        self.device     = device
        self.seed       = seed

        self._model     = None
        self._data_info = None

    def fit(self, train_df: pd.DataFrame) -> "LightGCNRecommender":
        # LibRecommender expects a "label" column
        df = train_df.copy()
        if "label" not in df.columns:
            df["label"] = 1.0

        train_data, self._data_info = DatasetPure.build_trainset(df)

        self._model = LightGCN(
            task="ranking",
            data_info=self._data_info,
            loss_type="bpr",
            embed_size=self.embed_size,
            n_epochs=self.n_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            num_neg=self.num_neg,
            device=self.device,
            seed=self.seed,
        )
        self._model.fit(
            train_data,
            neg_sampling=True,
            verbose=2,
            shuffle=True,
        )
        return self

    def recommend(
        self,
        user: str,
        k: int = 10,
        target_prefix: str | None = None,
    ) -> list[str]:
        # Request extra candidates when we need to filter by domain
        n_req = 100 if target_prefix else k

        rec_dict = self._model.recommend_user(
            user=user,
            n_rec=n_req,
            cold_start="popular",
            inner_id=False,
            filter_consumed=True,
            random_rec=False,
        )

        items = rec_dict.get(user, [])
        if isinstance(items, np.ndarray):
            items = items.tolist()

        if target_prefix:
            items = [it for it in items if it.startswith(target_prefix)]

        return items[:k]