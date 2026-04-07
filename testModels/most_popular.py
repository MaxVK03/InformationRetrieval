from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


class MostPopularRecommender:
    """Global popularity recommender with per-user seen-item filtering."""

    def __init__(self) -> None:
        self.ranked_items: list[str] = []
        self.user_seen: dict[str, set[str]] = defaultdict(set)

    def fit(self, train_df: pd.DataFrame) -> None:
        counts = Counter(train_df["item"].astype(str).tolist())
        self.ranked_items = [item for item, _ in counts.most_common()]

        seen_map: dict[str, set[str]] = defaultdict(set)
        for _, row in train_df.iterrows():
            seen_map[str(row["user"])].add(str(row["item"]))
        self.user_seen = seen_map

    def recommend(self, user: str, k: int = 10, target_prefix: str | None = None) -> list[str]:
        user = str(user)
        seen = self.user_seen.get(user, set())
        out: list[str] = []

        for item in self.ranked_items:
            if item in seen:
                continue
            if target_prefix and not item.startswith(target_prefix):
                continue
            out.append(item)
            if len(out) >= k:
                break

        return out
