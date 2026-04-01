import math


def recall_at_k(recommended: list, ground_truth: object) -> float:
    return 1.0 if ground_truth in recommended else 0.0


def precision_at_k(recommended: list, ground_truth: object, k: int) -> float:
    return 1.0 / k if ground_truth in recommended else 0.0


def ndcg_at_k(recommended: list, ground_truth: object) -> float:
    if ground_truth in recommended:
        rank = recommended.index(ground_truth) + 1
        return 1.0 / math.log2(rank + 1)
    return 0.0


def mean_average_precision(
    predictions: dict[str, list],
    ground_truth: dict[str, list],
    k: int | None = None,
) -> float:
    """
    predictions : {user: [item, ...]}  (ranked list)
    ground_truth: {user: [item, ...]}  (relevant items)
    """
    total_ap, n_users = 0.0, 0

    for user, ranked in predictions.items():
        relevant = ground_truth.get(user)
        if not relevant:
            continue

        ranked = ranked[:k] if k else ranked
        hits, sum_prec = 0, 0.0

        for i, item in enumerate(ranked, start=1):
            if item in relevant:
                hits += 1
                sum_prec += hits / i

        total_ap += sum_prec / len(relevant)
        n_users  += 1

    return total_ap / n_users if n_users > 0 else 0.0


def mean_reciprocal_rank(
    predictions: dict[str, list],
    ground_truth: dict[str, list],
    k: int | None = None,
) -> float:
    total, n_users = 0.0, 0

    for user, ranked in predictions.items():
        relevant = ground_truth.get(user)
        if not relevant:
            continue

        ranked = ranked[:k] if k else ranked

        for rank, item in enumerate(ranked, start=1):
            if item in relevant:
                total += 1.0 / rank
                break

        n_users += 1

    return total / n_users if n_users > 0 else 0.0