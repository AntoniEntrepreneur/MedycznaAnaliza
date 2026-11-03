"""Vascular age model inversion engine for stroke risk estimation."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

from src.ml import load_production_pipeline, predict_stroke_risk


def calculate_vascular_age(
    patient_risk: float,
    pipeline_artifact: Any = None,
    min_age: float = 18.0,
    max_age: float = 85.0,
    chronological_age: Optional[float] = None,
    patient_gender: Optional[str] = None,
    patient_profile: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Calculate vascular age by inverting stroke risk model against healthy baseline.

    Solves for the chronological age at which an idealized healthy individual would
    have the same predicted stroke probability as the patient. Clamps outputs strictly
    to [min_age, max_age] and computes framing metrics with positive reinforcement.
    """
    # Resolve optional patient profile / kwargs
    if patient_profile is not None:
        if chronological_age is None and "age" in patient_profile:
            chronological_age = float(patient_profile["age"])
        if patient_gender is None and "gender" in patient_profile:
            patient_gender = str(patient_profile["gender"])

    if "patient" in kwargs and isinstance(kwargs["patient"], dict):
        p_kw = kwargs["patient"]
        if chronological_age is None and "age" in p_kw:
            chronological_age = float(p_kw["age"])
        if patient_gender is None and "gender" in p_kw:
            patient_gender = str(p_kw["gender"])

    if chronological_age is None:
        if "age" in kwargs:
            chronological_age = float(kwargs["age"])
        else:
            chronological_age = 50.0

    if patient_gender is None:
        if "gender" in kwargs:
            patient_gender = str(kwargs["gender"])
        else:
            patient_gender = "Female"

    # Normalize risk probability if passed as a percentage (e.g., 15.0 instead of 0.15)
    risk_val = float(patient_risk)
    if risk_val > 1.0:
        risk_val = risk_val / 100.0
    risk_val = float(np.clip(risk_val, 0.0, 1.0))

    # Resolve pipeline artifact
    if pipeline_artifact is None:
        pipeline_artifact = load_production_pipeline()
    elif isinstance(pipeline_artifact, (str, Path)):
        pipeline_artifact = load_production_pipeline(pipeline_artifact)

    # Extract pipeline estimator
    if isinstance(pipeline_artifact, dict) and "pipeline" in pipeline_artifact:
        pipeline = pipeline_artifact["pipeline"]
    else:
        pipeline = pipeline_artifact

    # Generate fine grid of ages for healthy baseline
    n_points = 671
    grid_ages = np.linspace(min_age, max_age, n_points)

    baseline_records = [
        {
            "gender": patient_gender,
            "age": float(a),
            "hypertension": 0,
            "heart_disease": 0,
            "ever_married": "No",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 85.0,
            "bmi": 22.5,
            "smoking_status": "never smoked",
        }
        for a in grid_ages
    ]
    baseline_df = pd.DataFrame(baseline_records)

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(baseline_df)[:, 1]
    else:
        probs = np.array([
            predict_stroke_risk(pipeline_artifact, rec)["probability"]
            for rec in baseline_records
        ])

    probs = np.asarray(probs, dtype=float)

    # Inversion via monotonic interpolation or boundary clamping
    min_risk = float(probs[0])
    max_risk = float(probs[-1])

    if risk_val <= min_risk:
        raw_vascular_age = float(min_age)
    elif risk_val >= max_risk:
        raw_vascular_age = float(max_age)
    else:
        interpolator = interp1d(
            probs,
            grid_ages,
            kind="linear",
            bounds_error=False,
            fill_value=(min_age, max_age),
        )
        raw_vascular_age = float(interpolator(risk_val))

    # Strict clamping to [min_age, max_age]
    vascular_age = float(np.clip(raw_vascular_age, min_age, max_age))
    delta = float(vascular_age - chronological_age)

    # Positive reinforcement framing
    if delta > 0:
        rejuvenation_potential = delta
        rejuv_str = f"{rejuvenation_potential:.0f}" if float(rejuvenation_potential).is_integer() else f"{rejuvenation_potential:.1f}"
        framing_message = f"Potencjał Odmłodzenia: -{rejuv_str} lat przy optymalizacji czynników ryzyka"
    else:
        rejuvenation_potential = 0.0
        abs_delta = abs(delta)
        delta_str = f"{abs_delta:.0f}" if float(abs_delta).is_integer() else f"{abs_delta:.1f}"
        framing_message = f"Wiek naczyniowy optymalny / młodszy o {delta_str} lat"

    return {
        "vascular_age": round(vascular_age, 1),
        "chronological_age": round(chronological_age, 1),
        "delta": round(delta, 1),
        "rejuvenation_potential": round(rejuvenation_potential, 1),
        "framing_message": framing_message,
        "description": framing_message,
        "min_age": min_age,
        "max_age": max_age,
        "baseline_min_risk": min_risk,
        "baseline_max_risk": max_risk,
    }
