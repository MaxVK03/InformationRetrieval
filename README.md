# Information Retrieval Project

## Quick start

1. Create or reuse a Python 3.11 virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Put raw Amazon review files in:

`data/raw/Books_5.json`

`data/raw/Movies_and_TV_5.json`

4. Run:

```bash
python main.py
```

The script will print evaluation metrics and write `results.csv`.

## Notes

- The current pipeline loads the first chunk (`CHUNK_SIZE`) from each JSON file.
- File paths and experiment hyperparameters are defined in `reccomender/config.py`.
