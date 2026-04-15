from pathlib import Path

# Paths
DATA_DIR = Path("data/raw")
BOOKS_PATH   = DATA_DIR / "Books_5.json"
MOVIES_PATH  = DATA_DIR / "Movies_and_TV_5.json"

# Domain names
SOURCE_DOMAIN = "Books"
TARGET_DOMAIN = "Movies"

# Data loading
CHUNK_SIZE = 100_000

# Experiment
COLD_LEVELS = [1, 2, 5] # number of target-domain train interactions per user
K_EVAL      = 10 # cutoff for all ranking metrics

# BPR hyperparameters
BPR_FACTORS    = 64
BPR_LR         = 0.05
BPR_REG        = 0.01
BPR_ITERATIONS = 100

# LightGCN hyperparameters
LGCN_EMBED_SIZE  = 64
LGCN_N_EPOCHS    = 20
LGCN_LR          = 1e-3
LGCN_BATCH_SIZE  = 2048
LGCN_NUM_NEG     = 1
LGCN_DEVICE      = "cuda"
LGCN_SEED        = 42


PROCESSED_DIR        = Path("data/processed")
SOURCE_CSV           = PROCESSED_DIR / "source_books.csv"
TARGET_CSV           = PROCESSED_DIR / "target_movies_and_tv.csv"
TARGET_SPLITS_DIR = Path("data/processed")
