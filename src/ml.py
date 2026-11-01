"""Machine learning benchmark suite and clinical decision threshold optimization for stroke risk."""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.config import (
    BINARY_COLS,
    CATEGORICAL_COLS,
    NUMERIC_CONTINUOUS_COLS,
    PROJECT_ROOT,
    TARGET_COL,
)
from src.data import get_curated_cohorts

DEFAULT_NUMERIC_COLS = list(NUMERIC_CONTINUOUS_COLS)
DEFAULT_BINARY_COLS = list(BINARY_COLS)
DEFAULT_CATEGORICAL_COLS = list(CATEGORICAL_COLS)
DEFAULT_FEATURE_COLS = (
    DEFAULT_NUMERIC_COLS + DEFAULT_BINARY_COLS + DEFAULT_CATEGORICAL_COLS
)
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "stroke_model.joblib"


def build_preprocessor(
    numeric_cols: Optional[List[str]] = None,
    binary_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    n_neighbors: int = 5,
) -> ColumnTransformer:
    """Build leak-free preprocessing ColumnTransformer with KNN imputation."""
    num_cols = numeric_cols if numeric_cols is not None else DEFAULT_NUMERIC_COLS
    bin_cols = binary_cols if binary_cols is not None else DEFAULT_BINARY_COLS
    cat_cols = (
        categorical_cols if categorical_cols is not None else DEFAULT_CATEGORICAL_COLS
    )

    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", KNNImputer(n_neighbors=n_neighbors)),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = OneHotEncoder(
        drop="first",
        handle_unknown="ignore",
        sparse_output=False,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
            ("bin", "passthrough", bin_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )


def get_model_catalog(
    scale_pos_weight: float = 1.0,
    random_state: int = 42,
) -> Dict[str, Dict[str, BaseEstimator]]:
    """Return catalog of 4 candidate classifiers under Cost-Sensitive and SMOTE strategies."""
    return {
        "Logistic Regression": {
            "Cost-Sensitive": LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            ),
            "SMOTE": LogisticRegression(
                max_iter=1000,
                random_state=random_state,
            ),
        },
        "Random Forest": {
            "Cost-Sensitive": RandomForestClassifier_fn(
                n_estimators=100,
                class_weight="balanced",
                random_state=random_state,
            ),
            "SMOTE": RandomForestClassifier_fn(
                n_estimators=100,
                random_state=random_state,
            ),
        },
        "XGBoost": {
            "Cost-Sensitive": XGBClassifier(
                eval_metric="logloss",
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
            ),
            "SMOTE": XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
            ),
        },
        "LightGBM": {
            "Cost-Sensitive": LGBMClassifier(
                scale_pos_weight=scale_pos_weight,
                random_state=random_state,
                verbose=-1,
            ),
            "SMOTE": LGBMClassifier(
                random_state=random_state,
                verbose=-1,
            ),
        },
    }


def RandomForestClassifier_fn(*args: Any, **kwargs: Any) -> BaseEstimator:
    """Instantiate RandomForestClassifier with dynamic import."""
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(*args, **kwargs)


def optimize_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    beta: float = 2.0,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[float, float]:
    """Find decision threshold maximizing F-beta score on predictions."""
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_threshold = 0.50
    best_fbeta = -1.0

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        score = fbeta_score(y_true, preds, beta=beta, zero_division=0)
        # In case of tie, prioritize lower threshold for clinical screening sensitivity
        if score > best_fbeta:
            best_fbeta = float(score)
            best_threshold = float(t)

    return best_threshold, best_fbeta


def compute_classification_metrics(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float = 0.50,
    beta: float = 2.0,
) -> Dict[str, float]:
    """Compute precision, recall, and F-beta at a given probability cutoff."""
    preds = (y_probs >= threshold).astype(int)
    return {
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "f_score": float(fbeta_score(y_true, preds, beta=beta, zero_division=0)),
    }


def classify_risk_tier(prob: float) -> Dict[str, Any]:
    """Classify predicted stroke probability into 3-tier clinical risk category.

    Tiers:
    - Low Risk (< 10%): "Niskie" (color: green, #16a34a)
    - Moderate Risk (10% - 25%): "Umiarkowane" (color: amber, #ca8a04)
    - High Risk (>= 25%): "Wysokie" (color: red, #dc2626)
    """
    if prob < 0.10:
        return {
            "tier": "Niskie",
            "tier_en": "Low",
            "color": "green",
            "hex_color": "#16a34a",
            "bg_color": "#dcfce7",
            "border_color": "#86efac",
            "cutoff_range": "< 10%",
            "clinical_recommendation": (
                "Rutynowa profilaktyka i nadzór stylu życia. "
                "Kolejna ocena ryzyka podczas planowej wizyty rocznej."
            ),
        }
    elif prob < 0.25:
        return {
            "tier": "Umiarkowane",
            "tier_en": "Moderate",
            "color": "amber",
            "hex_color": "#ca8a04",
            "bg_color": "#fef9c3",
            "border_color": "#fde047",
            "cutoff_range": "10% - 25%",
            "clinical_recommendation": (
                "Zalecana kontrola kliniczna za 6 miesięcy. "
                "Wdrożenie niefarmakologicznej modyfikacji czynników ryzyka "
                "(dieta, aktywność fizyczna, monitorowanie ciśnienia tętniczego i glikemii)."
            ),
        }
    else:
        return {
            "tier": "Wysokie",
            "tier_en": "High",
            "color": "red",
            "hex_color": "#dc2626",
            "bg_color": "#fee2e2",
            "border_color": "#fca5a5",
            "cutoff_range": ">= 25%",
            "clinical_recommendation": (
                "Wskazana pilna konsultacja specjalistyczna (neurologiczna/kardiologiczna). "
                "Rozważenie agresywnej farmakoterapii przeciwnadciśnieniowej i hipolipemizującej."
            ),
        }


def save_production_pipeline(
    pipeline: Any,
    metadata: Dict[str, Any],
    file_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Path:
    """Save production pipeline artifact with model metadata to disk."""
    dest_path = Path(file_path).resolve()
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "pipeline": pipeline,
        "metadata": metadata,
        "optimal_threshold": metadata.get("optimal_threshold", 0.50),
        "feature_names": metadata.get("feature_names", DEFAULT_FEATURE_COLS),
        "model_name": metadata.get("model_name", "Logistic Regression"),
        "strategy": metadata.get("strategy", "Cost-Sensitive"),
        "metrics": metadata.get("metrics", {}),
        "risk_tiers": {
            "low": "< 10%",
            "moderate": "10% - 25%",
            "high": ">= 25%",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, dest_path)
    return dest_path


def load_production_pipeline(
    file_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Load production pipeline artifact from disk."""
    source_path = Path(file_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(
            f"Production model artifact not found at {source_path}. "
            "Please run run_ml_benchmark() to generate it."
        )
    return joblib.load(source_path)


def predict_stroke_risk(
    pipeline_or_artifact: Any,
    patient_data: Union[pd.DataFrame, Dict[str, Any]],
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Predict stroke risk probability, screening flag, and clinical tier for patient data."""
    if isinstance(pipeline_or_artifact, dict) and "pipeline" in pipeline_or_artifact:
        pipeline = pipeline_or_artifact["pipeline"]
        cutoff = (
            threshold
            if threshold is not None
            else pipeline_or_artifact.get("optimal_threshold", 0.50)
        )
    else:
        pipeline = pipeline_or_artifact
        cutoff = threshold if threshold is not None else 0.50

    if isinstance(patient_data, dict):
        df_input = pd.DataFrame([patient_data])
    else:
        df_input = patient_data.copy()

    probs = pipeline.predict_proba(df_input)[:, 1]
    prob = float(probs[0])
    tier_info = classify_risk_tier(prob)

    return {
        "probability": prob,
        "probability_percent": prob * 100.0,
        "screening_positive": bool(prob >= cutoff),
        "decision_threshold": cutoff,
        "risk_tier": tier_info["tier"],
        "risk_tier_en": tier_info["tier_en"],
        "color": tier_info["color"],
        "hex_color": tier_info["hex_color"],
        "clinical_recommendation": tier_info["clinical_recommendation"],
    }


def run_ml_benchmark(
    df_adult: Optional[pd.DataFrame] = None,
    cv_folds: int = 5,
    test_size: float = 0.20,
    random_state: int = 42,
    export_production: bool = True,
    model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
) -> Dict[str, Any]:
    """Execute leak-free 5-fold CV benchmark across 4 models x 2 strategies and calibrate winner."""
    if df_adult is None:
        cohorts = get_curated_cohorts()
        df = cohorts.adult.copy()
    else:
        df = df_adult.copy()

    # Filter adult cohort and exclude rare gender classes if present
    if "gender" in df.columns:
        df = df[df["gender"] != "Other"].copy()
    if "age" in df.columns:
        df = df[df["age"] >= 18.0].copy()

    df = df.reset_index(drop=True)

    feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)

    neg_count = int(np.sum(y_train == 0))
    pos_count = int(np.sum(y_train == 1))
    scale_pos_weight = neg_count / max(pos_count, 1)

    catalog = get_model_catalog(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    strategies = ["Cost-Sensitive", "SMOTE"]

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    cv_results = []
    oof_predictions: Dict[str, np.ndarray] = {}

    for model_name, strat_dict in catalog.items():
        for strategy in strategies:
            config_key = f"{model_name} ({strategy})"
            classifier = strat_dict[strategy]
            oof_probs = np.zeros(len(X_train), dtype=float)

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, y_tr = X_train.iloc[train_idx], y_train[train_idx]
                X_va = X_train.iloc[val_idx]

                # Leak-free preprocessor constructed and fitted strictly on fold train split
                preprocessor = build_preprocessor()

                steps: List[Tuple[str, Any]] = [("preprocessor", preprocessor)]
                if strategy == "SMOTE":
                    steps.append(("smote", SMOTE(random_state=random_state)))
                steps.append(("classifier", classifier))

                pipeline = ImbPipeline(steps=steps)
                pipeline.fit(X_tr, y_tr)
                oof_probs[val_idx] = pipeline.predict_proba(X_va)[:, 1]

            oof_predictions[config_key] = oof_probs

            # Global out-of-fold metrics
            roc_auc = float(roc_auc_score(y_train, oof_probs))
            pr_auc = float(average_precision_score(y_train, oof_probs))

            # Default p=0.50 metrics
            m_default = compute_classification_metrics(y_train, oof_probs, threshold=0.50, beta=2.0)

            # Optimal p* threshold maximizing F2
            p_star, f2_star = optimize_threshold(y_train, oof_probs, beta=2.0)
            m_star = compute_classification_metrics(y_train, oof_probs, threshold=p_star, beta=2.0)

            cv_results.append({
                "Model": model_name,
                "Strategy": strategy,
                "PR-AUC": round(pr_auc, 4),
                "ROC-AUC": round(roc_auc, 4),
                "F2 (p=0.50)": round(m_default["f_score"], 4),
                "Recall (p=0.50)": round(m_default["recall"], 4),
                "Precision (p=0.50)": round(m_default["precision"], 4),
                "Optimal Cutoff p*": round(p_star, 2),
                "F2 (p*)": round(m_star["f_score"], 4),
                "Recall (p*)": round(m_star["recall"], 4),
                "Precision (p*)": round(m_star["precision"], 4),
            })

    results_df = pd.DataFrame(cv_results)
    # Sort by PR-AUC descending, then F2 (p*) descending
    results_df = results_df.sort_values(
        by=["PR-AUC", "F2 (p*)"],
        ascending=[False, False],
    ).reset_index(drop=True)

    # Select best model based on PR-AUC
    best_row = results_df.iloc[0]
    best_model_name = str(best_row["Model"])
    best_strategy = str(best_row["Strategy"])

    # Train and calibrate winning production pipeline on full training split
    raw_best_classifier = catalog[best_model_name][best_strategy]
    production_preprocessor = build_preprocessor()

    # Wrap the classifier in CalibratedClassifierCV using sigmoid (Platt) scaling
    calibrated_classifier = CalibratedClassifierCV(
        estimator=raw_best_classifier,
        method="sigmoid",
        cv=cv_folds,
    )

    prod_steps: List[Tuple[str, Any]] = [("preprocessor", production_preprocessor)]
    if best_strategy == "SMOTE":
        prod_steps.append(("smote", SMOTE(random_state=random_state)))
    prod_steps.append(("classifier", calibrated_classifier))

    production_pipeline = ImbPipeline(steps=prod_steps)
    production_pipeline.fit(X_train, y_train)

    # Determine calibrated decision threshold on training split
    train_cal_probs = production_pipeline.predict_proba(X_train)[:, 1]
    optimal_cal_threshold, _ = optimize_threshold(y_train, train_cal_probs, beta=2.0)

    # Evaluate on held-out 20% test partition
    test_cal_probs = production_pipeline.predict_proba(X_test)[:, 1]
    test_roc_auc = float(roc_auc_score(y_test, test_cal_probs))
    test_pr_auc = float(average_precision_score(y_test, test_cal_probs))
    test_m_default = compute_classification_metrics(y_test, test_cal_probs, threshold=0.50, beta=2.0)
    test_m_opt = compute_classification_metrics(
        y_test, test_cal_probs, threshold=optimal_cal_threshold, beta=2.0
    )

    test_metrics = {
        "test_roc_auc": round(test_roc_auc, 4),
        "test_pr_auc": round(test_pr_auc, 4),
        "test_optimal_cutoff": round(optimal_cal_threshold, 2),
        "test_f2_default": round(test_m_default["f_score"], 4),
        "test_recall_default": round(test_m_default["recall"], 4),
        "test_precision_default": round(test_m_default["precision"], 4),
        "test_f2_optimal": round(test_m_opt["f_score"], 4),
        "test_recall_optimal": round(test_m_opt["recall"], 4),
        "test_precision_optimal": round(test_m_opt["precision"], 4),
    }

    metadata = {
        "model_name": best_model_name,
        "strategy": best_strategy,
        "optimal_threshold": round(optimal_cal_threshold, 2),
        "uncalibrated_cv_optimal_threshold": float(best_row["Optimal Cutoff p*"]),
        "feature_names": feature_cols,
        "categorical_cols": DEFAULT_CATEGORICAL_COLS,
        "binary_cols": DEFAULT_BINARY_COLS,
        "numeric_cols": DEFAULT_NUMERIC_COLS,
        "metrics": {
            "cv_best_row": best_row.to_dict(),
            "test_evaluation": test_metrics,
        },
    }

    saved_path = None
    if export_production:
        saved_path = save_production_pipeline(
            pipeline=production_pipeline,
            metadata=metadata,
            file_path=model_path,
        )

    return {
        "cv_results_df": results_df,
        "oof_predictions": oof_predictions,
        "best_model_name": best_model_name,
        "best_strategy": best_strategy,
        "best_cv_metrics": best_row.to_dict(),
        "calibrated_optimal_threshold": round(optimal_cal_threshold, 2),
        "production_pipeline": production_pipeline,
        "test_metrics": test_metrics,
        "saved_artifact_path": str(saved_path) if saved_path else None,
        "train_test_split": {
            "n_train": len(X_train),
            "n_test": len(X_test),
            "pos_train": pos_count,
            "neg_train": neg_count,
        },
    }


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING MACHINE LEARNING BENCHMARK SUITE (TICKET #24)")
    print("=" * 80)
    results = run_ml_benchmark(export_production=True)
    print("\nCROSS-VALIDATION BENCHMARK RESULTS (5-Fold Stratified CV):")
    print(results["cv_results_df"].to_string(index=False))
    print("\nBEST PRODUCTION CONFIGURATION:")
    print(f"Model: {results['best_model_name']} ({results['best_strategy']})")
    print(f"Optimal Calibrated Screening Cutoff (p*): {results['calibrated_optimal_threshold']}")
    print("\nHELD-OUT 20% TEST SET EVALUATION:")
    for k, v in results["test_metrics"].items():
        print(f"  {k}: {v}")
    if results["saved_artifact_path"]:
        print(f"\nSaved production pipeline artifact to: {results['saved_artifact_path']}")
