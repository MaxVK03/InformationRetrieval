# Information Retrieval

## Amazon Reviews 5-core preprocessing

This repo includes a preprocessing script for a small cross-domain Amazon Reviews setup:

- `Books` as the source domain
- `Movies_and_TV` as the target domain
- ratings converted to implicit feedback with `rating >= 4 -> interaction = 1`
- users restricted to the intersection of both domains
- target-domain cold-start splits for `k in {1, 3, 5, 10}`

### Cold-start split semantics

For each selected user in the target domain:

- `support.csv` keeps the first `k` interactions in timestamp order
- `query.csv` keeps the remaining interactions after those `k`

The script only keeps users with at least `max(k) + 1` target interactions so that every split has both support and query data.

### Expected raw files

Download the Amazon Reviews 5-core JSON Lines files and point the script at them, for example:

- `data/raw/Books_5.json.gz`
- `data/raw/Movies_and_TV_5.json.gz`

Both plain `.json` and `.json.gz` are supported.

### Example

```bash
python scripts/preprocess_amazon_5core.py \
  --books-path data/raw/Books_5.json.gz \
  --movies-path data/raw/Movies_and_TV_5.json.gz \
  --output-dir data/processed/amazon_books_to_movies
```

Defaults are tuned for the requested reduced dataset size:

- `~10k-30k` users
- `~50k-100k` interactions in each domain

If the default bounds do not fit the raw data distribution, adjust:

- `--min-users` / `--max-users` / `--desired-users`
- `--min-source-interactions` / `--max-source-interactions`
- `--min-target-interactions` / `--max-target-interactions`
