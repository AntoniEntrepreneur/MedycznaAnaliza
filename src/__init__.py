"""Shared data and modeling pipeline for the notebook and Streamlit app."""

from src.data import (
    CuratedCohorts,
    DatasetValidationError,
    curate_stroke_data,
    get_curated_cohorts,
    load_data,
    validate_dataset,
    validate_missing_bmi_patterns,
)
from src.ml import (
    build_preprocessor,
    classify_risk_tier,
    compute_classification_metrics,
    get_model_catalog,
    optimize_threshold,
    predict_stroke_risk,
    run_logistic_regression_pipeline,
    run_ml_benchmark,
    train_production_pipeline,
)

__all__ = [
    "CuratedCohorts",
    "DatasetValidationError",
    "build_preprocessor",
    "classify_risk_tier",
    "compute_classification_metrics",
    "curate_stroke_data",
    "get_curated_cohorts",
    "get_model_catalog",
    "load_data",
    "optimize_threshold",
    "predict_stroke_risk",
    "run_logistic_regression_pipeline",
    "run_ml_benchmark",
    "train_production_pipeline",
    "validate_dataset",
    "validate_missing_bmi_patterns",
]
