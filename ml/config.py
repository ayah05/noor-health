from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"


# ============================================================
# MODEL
# ============================================================

BASE_MODEL = "Qwen/Qwen3-0.6B"

MAX_LENGTH = 512


# ============================================================
# DATASETS
# ============================================================

TRAIN_FILE = PROCESSED_DATA_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DATA_DIR / "validation.jsonl"
TEST_FILE = PROCESSED_DATA_DIR / "test.jsonl"


# ============================================================
# REPRODUCIBILITY
# ============================================================

RANDOM_SEED = 42