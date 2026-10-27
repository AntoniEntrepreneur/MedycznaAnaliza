"""Variant B: Bento Clinical Dashboard (Executive EHR Workstation)."""

from typing import Dict, Any
import streamlit as st
import pandas as pd
from src.prototype.presets import PATIENT_PRESETS, CATEGORICAL_OPTIONS
from src.prototype.model_service import predict_patient_risk
from src.prototype.components import (
    render_risk_gauge,
    render_shap_waterfall,
    render_state_inspector,
    get_risk_tier,
)


def render_variant_b():
    """Render Variant B: High-density bento grid dashboard with interactive sensitivity simulation."""
    st.title("📊 Bento Clinical EHR Workstation (Variant B)")
    st.caption("All-in-one dense executive dashboard with sidebar parameters and real-time sensitivity analysis.")

    # Sidebar: presets and full clinical parameter controls
    st.sidebar.header("⚡ Patient Presets & Inputs")

    preset_names = list(PATIENT_PRESETS.keys())
    selected_preset = st.sidebar.selectbox(
        "Load Clinical Archetype",
        preset_names,
        index=0,
        key="var_b_preset",
    )
    preset_data = PATIENT_PRESETS[selected_preset]

    if "patient_b" not in st.session_state or st.session_state.get("last_preset_b") != selected_preset:
        st.session_state["patient_b"] = preset_data.copy()
        st.session_state["last_preset_b"] = selected_preset

    patient = st.session_state["patient_b"]

    with st.sidebar.expander("👤 Patient Demographics", expanded=True):
        patient["gender"] = st.selectbox(
            "Sex",
            CATEGORICAL_OPTIONS["gender"],
            index=CATEGORICAL_OPTIONS["gender"].index(patient["gender"]),
            key="vb_gender",
        )
        patient["age"] = float(st.slider(
            "Age",
            min_value=1.0,
            max_value=90.0,
            value=float(patient["age"]),
            step=1.0,
            key="vb_age",
        ))
        patient["ever_married"] = st.selectbox(
            "Married",
            CATEGORICAL_OPTIONS["ever_married"],
            index=CATEGORICAL_OPTIONS["ever_married"].index(patient["ever_married"]),
            key="vb_married",
        )
        patient["work_type"] = st.selectbox(
            "Work Type",
            CATEGORICAL_OPTIONS["work_type"],
            index=CATEGORICAL_OPTIONS["work_type"].index(patient["work_type"]),
            key="vb_work",
        )
        patient["Residence_type"] = st.selectbox(
            "Residence",
            CATEGORICAL_OPTIONS["Residence_type"],
            index=CATEGORICAL_OPTIONS["Residence_type"].index(patient["Residence_type"]),
            key="vb_residence",
        )

    with st.sidebar.expander("🩺 Vitals & Laboratory", expanded=True):
        htn_bool = patient["hypertension"] == 1
        new_htn = st.checkbox("Hypertension", value=htn_bool, key="vb_htn")
        patient["hypertension"] = 1 if new_htn else 0

        hd_bool = patient["heart_disease"] == 1
        new_hd = st.checkbox("Heart Disease", value=hd_bool, key="vb_hd")
        patient["heart_disease"] = 1 if new_hd else 0

        patient["avg_glucose_level"] = float(st.slider(
            "Average Glucose (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=float(patient["avg_glucose_level"]),
            step=1.0,
            key="vb_glucose",
        ))
        patient["bmi"] = float(st.slider(
            "BMI (kg/m²)",
            min_value=12.0,
            max_value=60.0,
            value=float(patient["bmi"]),
            step=0.2,
            key="vb_bmi",
        ))
        patient["smoking_status"] = st.selectbox(
            "Smoking Status",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=CATEGORICAL_OPTIONS["smoking_status"].index(patient["smoking_status"]),
            key="vb_smoking",
        )

    # Base prediction
    prob, contributions = predict_patient_risk(patient)
    tier = get_risk_tier(prob)
    baseline_pop_rate = 0.0487  # ~4.87% empirical stroke rate in dataset
    relative_risk = prob / baseline_pop_rate

    # Top summary metrics strip
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("5-Year Predicted Stroke Risk", f"{prob * 100:.1f}%")
    with kpi2:
        st.metric("Relative Risk vs Population", f"{relative_risk:.1f}x", delta=f"{relative_risk - 1.0:+.1f}x", delta_color="inverse")
    with kpi3:
        st.metric("Clinical Risk Category", tier["tier"])
    with kpi4:
        st.metric("Blood Sugar Status", "Diabetic" if patient["avg_glucose_level"] >= 140 else "Normal")

    st.markdown("---")

    # Bento Grid Middle: Gauge on left, SHAP Waterfall on right
    c_left, c_right = st.columns([1, 1])

    with c_left:
        st.markdown("### 🎯 Predictive Risk Meter")
        render_risk_gauge(prob, title="Calibrated 5-Year Stroke Risk")

    with c_right:
        st.markdown("### 🔬 Explainable SHAP Attribution")
        st.caption("Biomarkers pushing risk up (Red) vs protective factors (Green)")
        render_shap_waterfall(contributions, max_features=7)

    st.markdown("---")

    # Bottom Bento Cell: Interactive "What-If" Sensitivity Simulator
    st.markdown("### 🎛️ Real-Time 'What-If' Intervention Sensitivity")
    st.caption("Test how aggressive clinical targets impact the patient's predicted risk without leaving this view.")

    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        sim_glucose = st.slider(
            "Simulated Target Glucose (mg/dL)",
            min_value=70.0,
            max_value=250.0,
            value=float(patient["avg_glucose_level"]),
            step=5.0,
            key="vb_sim_glucose",
        )
    with sim2:
        sim_bmi = st.slider(
            "Simulated Target BMI",
            min_value=18.0,
            max_value=45.0,
            value=float(patient["bmi"]),
            step=0.5,
            key="vb_sim_bmi",
        )
    with sim3:
        sim_smoking = st.selectbox(
            "Simulated Smoking Target",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=0,  # "never smoked"
            key="vb_sim_smoke",
        )

    simulated_patient = patient.copy()
    simulated_patient["avg_glucose_level"] = sim_glucose
    simulated_patient["bmi"] = sim_bmi
    simulated_patient["smoking_status"] = sim_smoking

    sim_prob, _ = predict_patient_risk(simulated_patient)
    risk_reduction = prob - sim_prob

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.metric(
            "Simulated Projected Risk",
            f"{sim_prob * 100:.1f}%",
            delta=f"{-risk_reduction * 100:.1f}%",
            delta_color="normal",
        )
    with res_col2:
        if risk_reduction > 0.005:
            st.success(
                f"✅ **Intervention Impact**: Reducing glucose to {sim_glucose:.0f} mg/dL, BMI to {sim_bmi:.1f}, "
                f"and setting smoking to '{sim_smoking}' yields an estimated absolute risk reduction of "
                f"**{risk_reduction * 100:.1f}%** (relative reduction of **{(risk_reduction / prob) * 100:.0f}%**)."
            )
        else:
            st.info("Adjust the sliders above to model targeted clinical risk reduction.")

    # Surface state (Rule 5)
    render_state_inspector(patient)
