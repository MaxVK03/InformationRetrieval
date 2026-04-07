import pandas as pd


class MostPopularRecommender:
    """
    Non-personalised baseline. Recommends the most interacted-with items
    in the training set, excluding items the user has already seen.
    """

    def __init__(self):
        self._popular_items: list = []
        self._user_seen: dict[str, set] = {}

    def fit(self, train_df: pd.DataFrame) -> "MostPopularRecommender":
        popularity = (
            train_df.groupby("item")
            .size()
            .sort_values(ascending=False)
        )
        self._popular_items = popularity.index.tolist()
        self._user_seen = (
            train_df.groupby("user")["item"].apply(set).to_dict()
        )
        return self

    def recommend(
        self,
        user: str,
        k: int = 10,
        target_prefix: str | None = None,
    ) -> list[str]:
        seen = self._user_seen.get(user, set())
        recs = []
        for item in self._popular_items:
            if item in seen:
                continue
            if target_prefix and not item.startswith(target_prefix):
                continue
            recs.append(item)
            if len(recs) == k:
                break
        return recs