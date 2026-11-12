"""Shared preprocessing, model comparison, and patient prediction pipeline."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    NUMERIC_CONTINUOUS_COLS,
    TARGET_COL,
)
from src.data import get_curated_cohorts

DEFAULT_NUMERIC_COLS = list(NUMERIC_CONTINUOUS_COLS)
DEFAULT_BINARY_COLS = list(BINARY_COLS)
DEFAULT_CATEGORICAL_COLS = list(CATEGORICAL_COLS)
DEFAULT_FEATURE_COLS = DEFAULT_NUMERIC_COLS + DEFAULT_BINARY_COLS + DEFAULT_CATEGORICAL_COLS


def build_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build the shared median/mode imputation and one-hot encoding transformer."""
    numeric = numeric_cols or DEFAULT_NUMERIC_COLS
    binary = binary_cols or DEFAULT_BINARY_COLS
    categorical = categorical_cols or DEFAULT_CATEGORICAL_COLS
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), categorical),
            ("binary", Pipeline([("imputer", SimpleImputer(strategy="most_frequent"))]), binary),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def get_model_catalog(random_state: int = 42) -> Dict[str, Any]:
    """Return the two classifiers evaluated on exactly the same split."""
    return {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=random_state
        ),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> tuple[float, float]:
    """Select the probability threshold with the maximum F1 score."""
    candidates = thresholds if thresholds is not None else np.linspace(0.01, 0.99, 99)
    scores = [
        f1_score(y_true, y_probs >= threshold, zero_division=0)
        for threshold in candidates
    ]
    best_index = int(np.argmax(scores))
    return float(candidates[best_index]), float(scores[best_index])


def compute_classification_metrics(
    y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5
) -> Dict[str, float]:
    """Compute F1, precision, and recall at a probability threshold."""
    predictions = y_probs >= threshold
    return {
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
    }


def classify_risk_tier(prob: float) -> Dict[str, Any]:
    """Return the dashboard's clinical risk tier for a probability."""
    if prob < 0.10:
        return {"tier": "Niskie", "tier_en": "Low", "color": "green"}
    if prob < 0.25:
        return {"tier": "Umiarkowane", "tier_en": "Moderate", "color": "amber"}
    return {"tier": "Wysokie", "tier_en": "High", "color": "red"}


def run_ml_benchmark(
    df_adult: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_state: int = 42,
    **_: Any,
) -> Dict[str, Any]:
    """Compare Logistic Regression and Random Forest on one stratified split."""
    df = df_adult.copy() if df_adult is not None else get_curated_cohorts().adult.copy()
    df = df[df["age"] >= 18].reset_index(drop=True)
    features = [column for column in DEFAULT_FEATURE_COLS if column in df]
    X, y = df[features], df[TARGET_COL].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    models: Dict[str, Pipeline] = {}
    rows = []
    for name, classifier in get_model_catalog(random_state).items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        threshold, _ = optimize_threshold(y_test, probabilities)
        metrics = compute_classification_metrics(y_test, probabilities, threshold)
        models[name] = pipeline
        rows.append({
            "model": name,
            "threshold": threshold,
            **metrics,
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
        })

    results = pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
    best_name = str(results.loc[0, "model"])
    best_row = results.iloc[0].to_dict()
    return {
        "models": models,
        "results": results,
        "cv_results_df": results,
        "best_model_name": best_name,
        "pipeline": models[best_name],
        "production_pipeline": models[best_name],
        "optimal_threshold": float(best_row["threshold"]),
        "calibrated_optimal_threshold": float(best_row["threshold"]),
        "feature_names": features,
        "metrics": best_row,
        "train_test_split": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "test_size": test_size,
            "random_state": random_state,
            "stratified": True,
        },
    }


def run_logistic_regression_pipeline(
    df_adult: Optional[pd.DataFrame] = None,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train the shared Logistic Regression pipeline and select its F1 threshold."""
    df = df_adult.copy() if df_adult is not None else get_curated_cohorts().adult.copy()
    df = df[df["age"] >= 18].reset_index(drop=True)
    features = [column for column in DEFAULT_FEATURE_COLS if column in df]
    X, y = df[features], df[TARGET_COL].astype(int).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor(scale_numeric=True)),
        ("classifier", get_model_catalog(random_state)["Logistic Regression"]),
    ])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    threshold, f1 = optimize_threshold(y_test, probabilities)
    metrics = compute_classification_metrics(y_test, probabilities, threshold)
    metrics["threshold"] = threshold
    metrics["threshold_f1"] = f1
    return {
        "pipeline": pipeline,
        "production_pipeline": pipeline,
        "optimal_threshold": threshold,
        "calibrated_optimal_threshold": threshold,
        "feature_names": features,
        "metrics": metrics,
        "train_test_split": {"n_train": len(X_train), "n_test": len(X_test)},
    }


def predict_stroke_risk(
    pipeline_or_result: Any,
    patient_data: Union[pd.DataFrame, Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Return one patient's probability and threshold-based classification."""
    result = pipeline_or_result if isinstance(pipeline_or_result, dict) else {}
    pipeline = result.get("production_pipeline", result.get("pipeline", pipeline_or_result))
    cutoff = threshold if threshold is not None else result.get("optimal_threshold", 0.5)
    frame = pd.DataFrame([patient_data]) if isinstance(patient_data, dict) else patient_data.copy()
    probability = float(pipeline.predict_proba(frame)[:, 1][0])
    return {
        "probability": probability,
        "probability_percent": probability * 100,
        "decision_threshold": float(cutoff),
        "screening_positive": bool(probability >= cutoff),
        "risk_tier": classify_risk_tier(probability)["tier"],
        "risk_tier_en": classify_risk_tier(probability)["tier_en"],
    }


def load_production_pipeline(*_: Any, **__: Any) -> Dict[str, Any]:
    """Train the in-memory Logistic Regression pipeline; no artifact is loaded."""
    return run_logistic_regression_pipeline()


def save_production_pipeline(*_: Any, **__: Any) -> None:
    """Serialization is intentionally unsupported for this shared pipeline."""
    raise RuntimeError("Serialized model artifacts are not used; train run_ml_benchmark() instead.")
