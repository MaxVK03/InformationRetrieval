from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DomainConfig:
    name: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a cross-domain Amazon Reviews 5-core dataset using "
            "Books as source and Movies_and_TV as target."
        )
    )
    parser.add_argument("--books-path", type=Path, required=True)
    parser.add_argument("--movies-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-rating", type=float, default=4.0)
    parser.add_argument("--target-ks", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--min-users", type=int, default=10_000)
    parser.add_argument("--max-users", type=int, default=30_000)
    parser.add_argument("--desired-users", type=int, default=20_000)
    parser.add_argument("--min-source-interactions", type=int, default=50_000)
    parser.add_argument("--max-source-interactions", type=int, default=100_000)
    parser.add_argument("--min-target-interactions", type=int, default=50_000)
    parser.add_argument("--max-target-interactions", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    yield from pd.read_json(
        path,
        lines=True,
        chunksize=chunksize,
        compression="infer",
    )


def build_user_set(domain: DomainConfig, min_rating: float, chunksize: int) -> set[str]:
    users: set[str] = set()
    for chunk in iter_chunks(domain.path, chunksize):
        filtered = chunk.loc[chunk["overall"] >= min_rating, ["reviewerID"]]
        users.update(filtered["reviewerID"].astype(str).tolist())
    return users


def load_domain_interactions(
    domain: DomainConfig,
    overlap_users: set[str],
    min_rating: float,
    chunksize: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    columns = ["reviewerID", "asin", "unixReviewTime", "overall"]
    for chunk in iter_chunks(domain.path, chunksize):
        filtered = chunk.loc[
            (chunk["overall"] >= min_rating) & chunk["reviewerID"].isin(overlap_users),
            columns,
        ].copy()
        if filtered.empty:
            continue
        filtered["reviewerID"] = filtered["reviewerID"].astype(str)
        filtered["asin"] = filtered["asin"].astype(str)
        frames.append(filtered)

    if not frames:
        raise ValueError(f"No interactions left for domain {domain.name}.")

    df = pd.concat(frames, ignore_index=True)
    df = (
        df.rename(
            columns={
                "reviewerID": "raw_user_id",
                "asin": "raw_item_id",
                "unixReviewTime": "timestamp",
            }
        )
        .sort_values(["raw_user_id", "timestamp", "raw_item_id"])
        .drop_duplicates(["raw_user_id", "raw_item_id"], keep="first")
        .reset_index(drop=True)
    )
    df["interaction"] = 1
    return df[["raw_user_id", "raw_item_id", "timestamp", "interaction"]]


def choose_users(
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    ks: list[int],
    min_users: int,
    max_users: int,
    desired_users: int,
    min_source_interactions: int,
    max_source_interactions: int,
    min_target_interactions: int,
    max_target_interactions: int,
    seed: int,
) -> pd.DataFrame:
    source_counts = source_df.groupby("raw_user_id").size().rename("source_count")
    target_counts = target_df.groupby("raw_user_id").size().rename("target_count")
    user_stats = pd.concat([source_counts, target_counts], axis=1, join="inner").reset_index()

    required_target_interactions = max(ks) + 1
    eligible = user_stats.loc[
        (user_stats["source_count"] >= 1)
        & (user_stats["target_count"] >= required_target_interactions)
    ].copy()
    if eligible.empty:
        raise ValueError(
            "No users satisfy the overlap and cold-start requirements. "
            f"Need target_count >= {required_target_interactions}."
        )

    eligible = eligible.sample(frac=1, random_state=seed).reset_index(drop=True)
    eligible["cum_source"] = eligible["source_count"].cumsum()
    eligible["cum_target"] = eligible["target_count"].cumsum()
    eligible["num_users"] = pd.RangeIndex(1, len(eligible) + 1)

    candidates = eligible.loc[
        (eligible["num_users"] >= min_users)
        & (eligible["num_users"] <= max_users)
        & (eligible["cum_source"] >= min_source_interactions)
        & (eligible["cum_source"] <= max_source_interactions)
        & (eligible["cum_target"] >= min_target_interactions)
        & (eligible["cum_target"] <= max_target_interactions)
    ].copy()

    if candidates.empty:
        stats = {
            "eligible_users": int(len(eligible)),
            "source_interactions_at_min_users": int(eligible.iloc[min(min_users, len(eligible)) - 1]["cum_source"])
            if len(eligible) >= 1
            else 0,
            "target_interactions_at_min_users": int(eligible.iloc[min(min_users, len(eligible)) - 1]["cum_target"])
            if len(eligible) >= 1
            else 0,
            "source_interactions_at_max_users": int(eligible.iloc[min(max_users, len(eligible)) - 1]["cum_source"])
            if len(eligible) >= 1
            else 0,
            "target_interactions_at_max_users": int(eligible.iloc[min(max_users, len(eligible)) - 1]["cum_target"])
            if len(eligible) >= 1
            else 0,
        }
        raise ValueError(
            "Could not find a user subset matching the requested size targets. "
            f"Observed stats: {json.dumps(stats, indent=2)}"
        )

    choice = candidates.iloc[(candidates["num_users"] - desired_users).abs().argmin()]
    return eligible.iloc[: int(choice["num_users"])].copy()


def remap_ids(source_df: pd.DataFrame, target_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    user_order = sorted(set(source_df["raw_user_id"]).intersection(target_df["raw_user_id"]))
    user_map = pd.DataFrame(
        {"raw_user_id": user_order, "user_id": pd.RangeIndex(len(user_order))}
    )

    def map_domain(df: pd.DataFrame) -> pd.DataFrame:
        item_order = pd.Index(sorted(df["raw_item_id"].unique()), name="raw_item_id")
        item_map = pd.DataFrame(
            {"raw_item_id": item_order, "item_id": pd.RangeIndex(len(item_order))}
        )
        mapped = (
            df.merge(user_map, on="raw_user_id", how="inner")
            .merge(item_map, on="raw_item_id", how="inner")
            .sort_values(["user_id", "timestamp", "item_id"])
            .reset_index(drop=True)
        )
        return mapped[["user_id", "item_id", "timestamp", "interaction", "raw_user_id", "raw_item_id"]]

    return map_domain(source_df), map_domain(target_df), user_map


def build_target_splits(target_df: pd.DataFrame, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_df = target_df.sort_values(["user_id", "timestamp", "item_id"]).reset_index(drop=True)
    order = target_df.groupby("user_id").cumcount()
    counts = target_df.groupby("user_id")["item_id"].transform("size")

    support = target_df.loc[order < k].copy()
    query = target_df.loc[(order >= k) & (counts > k)].copy()
    return support.reset_index(drop=True), query.reset_index(drop=True)


def write_outputs(
    output_dir: Path,
    source_df: pd.DataFrame,
    target_df: pd.DataFrame,
    user_map: pd.DataFrame,
    selected_users: pd.DataFrame,
    ks: list[int],
    args: argparse.Namespace,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_df.to_csv(output_dir / "source_books.csv", index=False)
    target_df.to_csv(output_dir / "target_movies_and_tv.csv", index=False)
    user_map.to_csv(output_dir / "user_mapping.csv", index=False)
    selected_users.to_csv(output_dir / "selected_users.csv", index=False)

    for k in ks:
        split_dir = output_dir / f"target_k_{k}"
        split_dir.mkdir(parents=True, exist_ok=True)
        support, query = build_target_splits(target_df, k)
        support.to_csv(split_dir / "support.csv", index=False)
        query.to_csv(split_dir / "query.csv", index=False)

    metadata = {
        "books_path": str(args.books_path),
        "movies_path": str(args.movies_path),
        "min_rating": args.min_rating,
        "target_ks": ks,
        "min_users": args.min_users,
        "max_users": args.max_users,
        "desired_users": args.desired_users,
        "selected_users": int(selected_users.shape[0]),
        "source_interactions": int(source_df.shape[0]),
        "target_interactions": int(target_df.shape[0]),
        "unique_source_items": int(source_df["item_id"].nunique()),
        "unique_target_items": int(target_df["item_id"].nunique()),
        "seed": args.seed,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def main() -> None:
    args = parse_args()
    ks = sorted(set(args.target_ks))
    if ks[0] <= 0:
        raise ValueError("All target ks must be positive integers.")

    books = DomainConfig(name="Books", path=args.books_path)
    movies = DomainConfig(name="Movies_and_TV", path=args.movies_path)

    books_users = build_user_set(books, args.min_rating, args.chunksize)
    movies_users = build_user_set(movies, args.min_rating, args.chunksize)
    overlap_users = books_users & movies_users
    if not overlap_users:
        raise ValueError("No overlapping users found after rating filtering.")

    source_df = load_domain_interactions(books, overlap_users, args.min_rating, args.chunksize)
    target_df = load_domain_interactions(movies, overlap_users, args.min_rating, args.chunksize)

    selected_users = choose_users(
        source_df=source_df,
        target_df=target_df,
        ks=ks,
        min_users=args.min_users,
        max_users=args.max_users,
        desired_users=args.desired_users,
        min_source_interactions=args.min_source_interactions,
        max_source_interactions=args.max_source_interactions,
        min_target_interactions=args.min_target_interactions,
        max_target_interactions=args.max_target_interactions,
        seed=args.seed,
    )
    selected_user_ids = set(selected_users["raw_user_id"])
    source_df = source_df.loc[source_df["raw_user_id"].isin(selected_user_ids)].copy()
    target_df = target_df.loc[target_df["raw_user_id"].isin(selected_user_ids)].copy()

    source_df, target_df, user_map = remap_ids(source_df, target_df)
    write_outputs(args.output_dir, source_df, target_df, user_map, selected_users, ks, args)

    print(
        json.dumps(
            {
                "selected_users": int(selected_users.shape[0]),
                "source_interactions": int(source_df.shape[0]),
                "target_interactions": int(target_df.shape[0]),
                "target_ks": ks,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
