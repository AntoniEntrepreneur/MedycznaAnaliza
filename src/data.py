"""Load and validate the tracked stroke prediction dataset."""

from pathlib import Path
from typing import NamedTuple, Optional, Union
import pandas as pd

from src.config import (
    LOCAL_RAW_DATA_PATH,
    RAW_DATA_FILENAME,
    TARGET_COL,
    ID_COL,
    CATEGORICAL_COLS,
    BINARY_COLS,
    NUMERIC_CONTINUOUS_COLS,
)

REQUIRED_COLUMNS = (
    [ID_COL, TARGET_COL]
    + NUMERIC_CONTINUOUS_COLS
    + BINARY_COLS
    + CATEGORICAL_COLS
)


class DatasetValidationError(ValueError):
    """Raised when the tracked stroke CSV cannot support the analysis."""


def validate_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the cohort and feature columns used by the modeling pipeline."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise DatasetValidationError(
            "Stroke CSV is missing required columns: "
            + ", ".join(missing)
            + ". Restore data/raw/healthcare-dataset-stroke-data.csv from the repository."
        )
    if df.empty:
        raise DatasetValidationError("Stroke CSV is empty; provide the tracked cohort CSV.")
    if df[ID_COL].duplicated().any():
        raise DatasetValidationError("Stroke CSV contains duplicate patient IDs.")
    if not set(df[TARGET_COL].dropna().unique()).issubset({0, 1}):
        raise DatasetValidationError("The stroke target must contain only 0/1 values.")
    if df[TARGET_COL].isna().any():
        raise DatasetValidationError("The stroke target contains missing values.")
    if not (df["age"].dropna() >= 0).all():
        raise DatasetValidationError("Age must be non-negative.")
    if int((df["age"] >= 18).sum()) == 0 or int((df["age"] < 18).sum()) == 0:
        raise DatasetValidationError("CSV must contain both adult and paediatric cohorts.")
    return df


def load_data(
    path_or_url: Optional[Union[str, Path]] = None,
    auto_download: bool = False,
    cache: bool = False,
) -> pd.DataFrame:
    """Load only the tracked CSV; never download or silently return an empty frame."""
    if isinstance(path_or_url, str) and "://" in path_or_url:
        raise ValueError(
            "Remote dataset URLs are not supported. Use the tracked CSV at "
            "data/raw/healthcare-dataset-stroke-data.csv."
        )
    target_file = Path(path_or_url) if path_or_url is not None else LOCAL_RAW_DATA_PATH
    if not target_file.is_file():
        raise FileNotFoundError(
            f"Required stroke CSV not found at {target_file}. "
            "Check out data/raw/healthcare-dataset-stroke-data.csv from the repository."
        )
    try:
        frame = pd.read_csv(target_file, na_values=["N/A", "NA", ""])
    except (OSError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Unable to read required stroke CSV at {target_file}: {exc}") from exc
    return validate_dataset(frame)


class CuratedCohorts(NamedTuple):
    """Curated dataset partitioned into adult modeling and paediatric exploratory cohorts."""

    adult: pd.DataFrame
    paediatric: pd.DataFrame


def curate_stroke_data(
    df: Optional[pd.DataFrame] = None,
    remove_other_gender: bool = True,
    age_threshold: float = 18.0,
) -> CuratedCohorts:
    """Curate raw stroke data into adult modeling and paediatric exploratory cohorts.

    Removes singleton records (gender == 'Other', id = 56156), preserves
    smoking_status == 'Unknown' as an explicit clinical category, and segregates
    the adult modeling cohort (age >= 18) from the paediatric sub-cohort (age < 18).

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Input raw stroke dataset. If None, loaded via load_data().
    remove_other_gender : bool, default True
        Whether to remove the single non-binary record (gender == 'Other', id = 56156).
    age_threshold : float, default 18.0
        Cutoff age separating paediatric (< threshold) and adult (>= threshold) cohorts.

    Returns
    -------
    CuratedCohorts
        NamedTuple with adult (N=4,253 when gender == 'Other' removed, 247 stroke cases;
        N=4,254 before removal) and paediatric (N=856, 2 stroke cases).
        Supports tuple unpacking: adult_df, paediatric_df = curate_stroke_data().
    """
    if df is None:
        df = load_data()
    else:
        df = df.copy()

    # Remove singleton non-binary record if requested
    if remove_other_gender:
        df = df[df["gender"] != "Other"].copy()

    # Segregate adult modeling and paediatric exploratory cohorts
    adult_df = df[df["age"] >= age_threshold].copy().reset_index(drop=True)
    paediatric_df = df[df["age"] < age_threshold].copy().reset_index(drop=True)

    return CuratedCohorts(adult=adult_df, paediatric=paediatric_df)


def get_curated_cohorts(
    path_or_url: Optional[Union[str, Path]] = None,
    remove_other_gender: bool = True,
    age_threshold: float = 18.0,
) -> CuratedCohorts:
    """Load dataset from path/url/fallback and return curated cohorts."""
    raw_df = load_data(path_or_url=path_or_url)
    return curate_stroke_data(
        raw_df,
        remove_other_gender=remove_other_gender,
        age_threshold=age_threshold,
    )


def validate_missing_bmi_patterns(
    df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Validate and summarize missing BMI patterns across clinical cohorts and stroke strata."""
    if df is None:
        df = load_data()

    adult_df = df[df["age"] >= 18.0]
    paed_df = df[df["age"] < 18.0]

    definitions = [
        ("Overall Dataset", df),
        ("Adults Overall (age >= 18)", adult_df),
        ("Adults Non-Stroke (stroke = 0)", adult_df[adult_df["stroke"] == 0]),
        ("Adults Stroke (stroke = 1)", adult_df[adult_df["stroke"] == 1]),
        ("Paediatric Overall (age < 18)", paed_df),
        ("Paediatric Non-Stroke (stroke = 0)", paed_df[paed_df["stroke"] == 0]),
        ("Paediatric Stroke (stroke = 1)", paed_df[paed_df["stroke"] == 1]),
    ]

    records = []
    for name, subset in definitions:
        total_n = len(subset)
        missing_n = int(subset["bmi"].isna().sum()) if total_n > 0 else 0
        missing_pct = (missing_n / total_n * 100.0) if total_n > 0 else 0.0
        records.append({
            "Cohort": name,
            "Total (N)": total_n,
            "Missing BMI (n)": missing_n,
            "Missing BMI (%)": f"{missing_pct:.2f}%",
            "Observed BMI (n)": total_n - missing_n,
        })

    return pd.DataFrame(records)

