"""Automated data acquisition pipeline for stroke prediction dataset."""

import io
import ssl
from pathlib import Path
from typing import NamedTuple, Optional, Union
import urllib.request
import pandas as pd

from src.config import (
    LOCAL_RAW_DATA_PATH,
    PRIMARY_REMOTE_URL,
    MIRROR_REMOTE_URL,
    RAW_DATA_FILENAME,
)


def _fetch_remote_bytes(url: str) -> bytes:
    """Fetch bytes from remote URL with SSL fallback."""
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "MedycznaAnaliza/1.0 (academic stroke analysis)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            return response.read()
    except urllib.error.URLError:
        # Fallback for environments with unconfigured SSL root certificates (e.g. macOS dev setups)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, context=ctx, timeout=15) as response:
            return response.read()


def find_local_dataset() -> Optional[Path]:
    """Search for the dataset CSV in standard project directories."""
    candidate_paths = [
        LOCAL_RAW_DATA_PATH,
        Path("data/raw") / RAW_DATA_FILENAME,
        Path("../data/raw") / RAW_DATA_FILENAME,
        Path("../../data/raw") / RAW_DATA_FILENAME,
        Path(RAW_DATA_FILENAME),
    ]
    for path in candidate_paths:
        if path.exists() and path.is_file():
            return path.resolve()
    return None


def download_dataset(
    dest_path: Optional[Union[str, Path]] = None,
    primary_url: str = PRIMARY_REMOTE_URL,
    mirror_url: str = MIRROR_REMOTE_URL,
) -> Path:
    """Download the raw dataset from remote repository fallback URLs."""
    target_path = Path(dest_path) if dest_path else LOCAL_RAW_DATA_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)

    errors = []
    for url in [primary_url, mirror_url]:
        try:
            data = _fetch_remote_bytes(url)
            with open(target_path, "wb") as f:
                f.write(data)
            return target_path
        except Exception as exc:
            errors.append(f"{url} ({exc})")

    raise RuntimeError(
        f"Failed to download stroke dataset from all sources: {'; '.join(errors)}"
    )


def load_data(
    path_or_url: Optional[Union[str, Path]] = None,
    auto_download: bool = True,
    cache: bool = True,
) -> pd.DataFrame:
    """Load stroke dataset from local path, custom URL, or automated fallback."""
    # Explicit URL supplied
    if isinstance(path_or_url, str) and (
        path_or_url.startswith("http://") or path_or_url.startswith("https://")
    ):
        raw_bytes = _fetch_remote_bytes(path_or_url)
        return pd.read_csv(io.BytesIO(raw_bytes), na_values=["N/A", "NA", ""])

    # Explicit local path supplied
    if path_or_url is not None:
        target_file = Path(path_or_url)
        if not target_file.exists():
            raise FileNotFoundError(f"Specified dataset file not found: {path_or_url}")
        return pd.read_csv(target_file, na_values=["N/A", "NA", ""])

    # Automatic discovery
    local_path = find_local_dataset()
    if local_path is not None:
        return pd.read_csv(local_path, na_values=["N/A", "NA", ""])

    # Remote fallback
    if auto_download:
        save_target = LOCAL_RAW_DATA_PATH if cache else None
        try:
            downloaded_path = download_dataset(dest_path=save_target)
            return pd.read_csv(downloaded_path, na_values=["N/A", "NA", ""])
        except Exception:
            # Direct in-memory load if local write fails
            raw_bytes = _fetch_remote_bytes(PRIMARY_REMOTE_URL)
            return pd.read_csv(io.BytesIO(raw_bytes), na_values=["N/A", "NA", ""])

    raise FileNotFoundError(
        "Stroke dataset not found locally and auto_download is disabled."
    )


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

