import pandas as pd

from testModels import most_popular, bpr, lightgcn
from evaluation.evaluator import evaluate
from experiment.cold_start import build_single_domain_frames, build_cross_domain_frames
import reccomender.config as config
from testModels.bpr import BPRRecommender
from testModels.lightgcn import LightGCNRecommender
from testModels.most_popular import MostPopularRecommender


def _make_model_configs(target_prefix: str) -> list[dict]:
    """
    Each entry defines one (model_name, training_mode, model_factory) triple.

    training_mode: "single" | "cross"
        single → trained on target-domain data only
        cross  → trained on source + target data
    """
    return [
        {
            "name":  "MostPopular (single)",
            "mode":  "single",
            "model": MostPopularRecommender,
            "kwargs": {},
        },
        {
            "name":  "BPR (single)",
            "mode":  "single",
            "model": BPRRecommender,
            "kwargs": {},
        },
        {
            "name":  "BPR (cross)",
            "mode":  "cross",
            "model": BPRRecommender,
            "kwargs": {},
        },
        {
            "name":  "LightGCN (single)",
            "mode":  "single",
            "model": LightGCNRecommender,
            "kwargs": {},
        },
        {
            "name":  "LightGCN (cross)",
            "mode":  "cross",
            "model": LightGCNRecommender,
            "kwargs": {},
        },
    ]

def run_all(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    cold_levels: list[int] = config.COLD_LEVELS,
    k_eval: int            = config.K_EVAL,
    target_domain: str     = config.TARGET_DOMAIN,
) -> pd.DataFrame:
    target_prefix = f"{target_domain}::"
    model_configs = _make_model_configs(target_prefix)
    splits_dir    = config.TARGET_SPLITS_DIR
    rows = []

    for cold_k in cold_levels:
        print(f"\n{'='*60}")
        print(f"Cold-start level: {cold_k} target train interactions/user")
        print(f"{'='*60}")

        single_train, single_test = build_single_domain_frames(cold_k, splits_dir)
        cross_train,  cross_test  = build_cross_domain_frames(source_df, cold_k, splits_dir)

        for cfg in model_configs:
            print(f"\n  → {cfg['name']}")
            train = single_train if cfg["mode"] == "single" else cross_train
            test  = single_test  if cfg["mode"] == "single" else cross_test

            model = cfg["model"](**cfg["kwargs"])
            model.fit(train)

            results = evaluate(model, test, k=k_eval)
            rows.append({"cold_k": cold_k, "model": cfg["name"], **results})

    return pd.DataFrame(rows)