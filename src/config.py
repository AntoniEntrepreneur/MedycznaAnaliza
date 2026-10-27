"""Configuration paths, remote dataset URLs, and feature definitions."""

from pathlib import Path

# Repository paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_FILENAME = "healthcare-dataset-stroke-data.csv"
LOCAL_RAW_DATA_PATH = RAW_DATA_DIR / RAW_DATA_FILENAME

# Remote fallback URLs for Google Colab and remote environments
PRIMARY_REMOTE_URL = (
    "https://raw.githubusercontent.com/AntoniEntrepreneur/MedycznaAnaliza/main/"
    "data/raw/healthcare-dataset-stroke-data.csv"
)
MIRROR_REMOTE_URL = (
    "https://raw.githubusercontent.com/ray-project/raydp/master/tutorials/dataset/"
    "healthcare-dataset-stroke-data.csv"
)

# Column and feature definitions
TARGET_COL = "stroke"
ID_COL = "id"

CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

BINARY_COLS = [
    "hypertension",
    "heart_disease",
]

NUMERIC_CONTINUOUS_COLS = [
    "age",
    "avg_glucose_level",
    "bmi",
]
