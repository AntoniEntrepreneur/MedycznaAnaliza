"""Predictive clinical risk and SHAP attribution service."""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
import streamlit as st
from src.data import load_data

FEATURE_ORDER = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

CATEGORICAL_COLS = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]


@st.cache_resource(show_spinner="Initializing clinical risk model and SHAP explainer...")
def get_model_and_explainer():
    """Train and cache calibrated baseline model with SHAP TreeExplainer."""
    df = load_data()
    df_clean = df.dropna(subset=["bmi"]).copy() if "bmi" in df.columns else df.copy()
    if df_clean["bmi"].isna().any():
        df_clean["bmi"] = df_clean["bmi"].fillna(df_clean["bmi"].median())

    X = df_clean[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    y = df_clean["stroke"].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    return model, explainer


def predict_patient_risk(patient: Dict[str, Any]) -> Tuple[float, pd.DataFrame]:
    """Calculate 5-year stroke risk probability and SHAP attribution for patient."""
    model, explainer = get_model_and_explainer()

    row_df = pd.DataFrame([patient])[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        row_df[col] = row_df[col].astype("category")

    prob = float(model.predict_proba(row_df)[0, 1])

    shap_explanation = explainer(row_df)
    shap_values = shap_explanation.values[0]
    base_val = float(shap_explanation.base_values[0]) if hasattr(shap_explanation, "base_values") else 0.05

    contributions_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "value": [patient[col] for col in FEATURE_ORDER],
        "shap_value": shap_values,
        "abs_impact": np.abs(shap_values),
    }).sort_values(by="abs_impact", ascending=False)

    contributions_df.attrs["base_value"] = base_val
    contributions_df.attrs["explanation"] = shap_explanation[0]

    return prob, contributions_df


@st.cache_data(show_spinner="Pobieranie globalnej ważności cech klinicznych...")
def get_cached_global_importance() -> pd.DataFrame:
    """Retrieve global feature importance DataFrame for the clinical LightGBM model."""
    from src.xai import get_feature_importances
    model, _ = get_model_and_explainer()
    return get_feature_importances(model, FEATURE_ORDER)


@st.cache_data(show_spinner="Ładowanie klinicznych studiów przypadków...")
def get_cached_archetype_case_studies() -> Dict[str, Dict[str, Any]]:
    """Retrieve 3 clinical archetypes with precomputed SHAP explanations."""
    from src.xai import get_clinical_archetypes
    model, explainer = get_model_and_explainer()
    archetypes = get_clinical_archetypes(pipeline_or_model=model)

    for _, data in archetypes.items():
        p = data["patient"]
        row_df = pd.DataFrame([p])[FEATURE_ORDER].copy()
        for col in CATEGORICAL_COLS:
            row_df[col] = row_df[col].astype("category")
        exp = explainer(row_df)[0]
        data["shap_values"] = exp.values.tolist()
        data["base_value"] = float(exp.base_values)
        data["feature_names"] = FEATURE_ORDER
        data["data"] = [p[col] for col in FEATURE_ORDER]

    return archetypes

