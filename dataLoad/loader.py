import pandas as pd
import numpy as np
from pathlib import Path


def load_domain(path: str | Path, domain_name: str, chunksize: int = 100_000) -> pd.DataFrame:
  """
  Load the first chunk of a raw Amazon review JSON file.

  Returns a DataFrame with columns:
      user, item, time, domain
  where item is domain-prefixed to avoid collisions across domains.
  """
  chunks = pd.read_json(path, lines=True, chunksize=chunksize)
  raw = next(chunks)

  df = raw[["reviewerID", "asin", "unixReviewTime"]].dropna().copy()
  df["domain"] = domain_name

  out = pd.DataFrame({
    "user": df["reviewerID"].astype(str),
    "item": df["domain"] + "::" + df["asin"].astype(str),
    "time": df["unixReviewTime"].astype(np.int64),
    "domain": df["domain"].astype(str),
  })

  return out


def load_target_split(splits_dir: Path, cold_k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
  split_dir = splits_dir / f"target_k_{cold_k}"
  support = pd.read_csv(split_dir / "support.csv")
  query = pd.read_csv(split_dir / "query.csv")
  return support, query

def load_preprocessed(source_path, target_path):
  def read(path):
    df = pd.read_csv(path)
    return df.rename(columns={"user_id": "user", "item_id": "item"})

  return read(source_path), read(target_path)