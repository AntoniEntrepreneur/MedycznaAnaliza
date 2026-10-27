"""Lightweight model service for clinical risk prediction and SHAP calculation."""

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
    """Train a fast LightGBM baseline model on the stroke dataset and initialize SHAP."""
    df = load_data()
    df_clean = df.dropna(subset=["bmi"]).copy() if "bmi" in df.columns else df.copy()
    if df_clean["bmi"].isna().any():
        df_clean["bmi"] = df_clean["bmi"].fillna(df_clean["bmi"].median())

    X = df_clean[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        X[col] = X[col].astype("category")

    y = df_clean["stroke"].astype(int)

    # Train balanced LightGBM model
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
    """Compute stroke risk probability and SHAP feature contributions for a patient."""
    model, explainer = get_model_and_explainer()

    row_df = pd.DataFrame([patient])[FEATURE_ORDER].copy()
    for col in CATEGORICAL_COLS:
        row_df[col] = row_df[col].astype("category")

    prob = float(model.predict_proba(row_df)[0, 1])

    shap_explanation = explainer(row_df)
    shap_values = shap_explanation.values[0]

    contributions_df = pd.DataFrame({
        "feature": FEATURE_ORDER,
        "value": [patient[col] for col in FEATURE_ORDER],
        "shap_value": shap_values,
        "abs_impact": np.abs(shap_values),
    }).sort_values(by="abs_impact", ascending=False)

    return prob, contributions_df
