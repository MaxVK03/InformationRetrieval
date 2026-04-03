from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd


class LightGCNRecommender:
    """
    Lightweight graph-style recommender based on 2-hop user-item propagation.
    This avoids heavy dependencies while preserving the expected API.
    """

    def __init__(self) -> None:
        self.user_seen: dict[str, set[str]] = defaultdict(set)
        self.item_users: dict[str, set[str]] = defaultdict(set)
        self.global_popularity: list[str] = []

    def fit(self, train_df: pd.DataFrame) -> None:
        item_counts = Counter(train_df["item"].astype(str).tolist())
        self.global_popularity = [item for item, _ in item_counts.most_common()]

        user_seen: dict[str, set[str]] = defaultdict(set)
        item_users: dict[str, set[str]] = defaultdict(set)

        for _, row in train_df.iterrows():
            user = str(row["user"])
            item = str(row["item"])
            user_seen[user].add(item)
            item_users[item].add(user)

        self.user_seen = user_seen
        self.item_users = item_users

    def recommend(self, user: str, k: int = 10, target_prefix: str | None = None) -> list[str]:
        user = str(user)
        seen = self.user_seen.get(user, set())
        scores: Counter[str] = Counter()

        # 2-hop: user -> interacted items -> neighbor users -> their items
        for item in seen:
            neighbors = self.item_users.get(item, set())
            if not neighbors:
                continue

            for neighbor in neighbors:
                if neighbor == user:
                    continue
                neighbor_items = self.user_seen.get(neighbor, set())
                if not neighbor_items:
                    continue
                weight = 1.0 / (1.0 + len(neighbor_items))
                for candidate in neighbor_items:
                    if candidate not in seen:
                        scores[candidate] += weight

        ranked = [item for item, _ in scores.most_common()]
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
