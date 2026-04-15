from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


class BPRRecommender:
    def __init__(self) -> None:
        self.user_seen: dict[str, set[str]] = defaultdict(set)
        self.item_co_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.global_popularity: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        item_counts = Counter(train_df["item"].astype(str).tolist())
        self.global_popularity = [item for item, _ in item_counts.most_common()]

        user_items: dict[str, list[str]] = defaultdict(list)
        for _, row in train_df.iterrows():
            user = str(row["user"])
            item = str(row["item"])
            user_items[user].append(item)

        self.user_seen = {u: set(items) for u, items in user_items.items()}

        co_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for items in user_items.values():
            unique_items = list(set(items))
            for i in unique_items:
                for j in unique_items:
                    if i != j:
                        co_counts[i][j] += 1
        self.item_co_counts = co_counts

    def recommend(self, user: str, k: int = 10, target_prefix: str | None = None) -> list[str]:
        user = str(user)
        seen = self.user_seen.get(user, set())
        candidate_scores: Counter[str] = Counter()

        for item in seen:
            for other, score in self.item_co_counts.get(item, {}).items():
                if other not in seen:
                    candidate_scores[other] += score

        ranked = [item for item, _ in candidate_scores.most_common()]

        out: list[str] = []
        for item in ranked:
            if target_prefix and not item.startswith(target_prefix):
                continue
            out.append(item)
            if len(out) >= k:
                return out

        for item in self.global_popularity:
            if item in seen or item in out:
                continue
            if target_prefix and not item.startswith(target_prefix):
                continue
            out.append(item)
            if len(out) >= k:
                break

        return out
