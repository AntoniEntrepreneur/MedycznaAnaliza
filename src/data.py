"""Automated data acquisition pipeline for stroke prediction dataset."""

import io
import ssl
from pathlib import Path
from typing import Optional, Union
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
