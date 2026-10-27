"""Unified Winning Variant: Clinician-Tailored EHR Workstation.

Synthesized from the user design interview:
1. Persistent categorized sidebar with quick archetype presets at the top.
2. Top KPI summary metric cards (Risk %, Relative Risk vs Baseline, Risk Category, Glycemic Status).
3. Bento center: Semicircular dial gauge meter (left) + SHAP factor impact waterfall (right).
4. Bottom actionability: Real-time 'What-If' sensitivity simulator with dynamic ARR/RRR and counseling script.
5. Collapsible copy-ready formatted EHR consultation note.
"""

from typing import Dict, Any
import streamlit as st
from src.prototype.presets import PATIENT_PRESETS, CATEGORICAL_OPTIONS
from src.prototype.model_service import predict_patient_risk
from src.prototype.components import (
    render_risk_gauge,
    render_shap_waterfall,
    render_state_inspector,
    get_risk_tier,
)


def render_winning_variant():
    """Render the synthesized winning variant for clinical risk calculation."""
    st.title("🩺 Clinical Stroke Risk Calculator (Validated Design)")
    st.caption("EHR Decision Support Workstation • Calibrated LightGBM & SHAP Attribution")

    # --- SIDEBAR: Presets & Categorized Inputs ---
    st.sidebar.header("⚡ Patient Profile & Inputs")

    preset_names = list(PATIENT_PRESETS.keys())
    selected_preset = st.sidebar.selectbox(
        "Load Clinical Archetype:",
        preset_names,
        index=0,
        key="win_preset",
    )
    preset_data = PATIENT_PRESETS[selected_preset]

    if "patient_win" not in st.session_state or st.session_state.get("last_preset_win") != selected_preset:
        st.session_state["patient_win"] = preset_data.copy()
        st.session_state["last_preset_win"] = selected_preset

    patient = st.session_state["patient_win"]

    with st.sidebar.expander("👤 Demographics & Social History", expanded=True):
        patient["gender"] = st.selectbox(
            "Sex",
            CATEGORICAL_OPTIONS["gender"],
            index=CATEGORICAL_OPTIONS["gender"].index(patient["gender"]),
            key="win_gender",
        )
        patient["age"] = float(st.slider(
            "Age (years)",
            min_value=1.0,
            max_value=90.0,
            value=float(patient["age"]),
            step=1.0,
            key="win_age",
        ))
        patient["ever_married"] = st.selectbox(
            "Marital Status",
            CATEGORICAL_OPTIONS["ever_married"],
            index=CATEGORICAL_OPTIONS["ever_married"].index(patient["ever_married"]),
            key="win_married",
        )
        patient["work_type"] = st.selectbox(
            "Work Type",
            CATEGORICAL_OPTIONS["work_type"],
            index=CATEGORICAL_OPTIONS["work_type"].index(patient["work_type"]),
            key="win_work",
        )
        patient["Residence_type"] = st.selectbox(
            "Residence",
            CATEGORICAL_OPTIONS["Residence_type"],
            index=CATEGORICAL_OPTIONS["Residence_type"].index(patient["Residence_type"]),
            key="win_residence",
        )

    with st.sidebar.expander("🩺 Vitals & Laboratory Biomarkers", expanded=True):
        htn_bool = patient["hypertension"] == 1
        new_htn = st.checkbox("Diagnosed Hypertension", value=htn_bool, key="win_htn")
        patient["hypertension"] = 1 if new_htn else 0

        hd_bool = patient["heart_disease"] == 1
        new_hd = st.checkbox("Pre-existing Heart Disease", value=hd_bool, key="win_hd")
        patient["heart_disease"] = 1 if new_hd else 0

        patient["avg_glucose_level"] = float(st.slider(
            "Average Blood Glucose (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=float(patient["avg_glucose_level"]),
            step=1.0,
            key="win_glucose",
        ))
        patient["bmi"] = float(st.slider(
            "Body Mass Index (BMI, kg/m²)",
            min_value=12.0,
            max_value=60.0,
            value=float(patient["bmi"]),
            step=0.2,
            key="win_bmi",
        ))
        patient["smoking_status"] = st.selectbox(
            "Smoking Status",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=CATEGORICAL_OPTIONS["smoking_status"].index(patient["smoking_status"]),
            key="win_smoking",
        )

    # Model evaluation
    prob, contributions = predict_patient_risk(patient)
    tier = get_risk_tier(prob)
    baseline_pop_rate = 0.0487  # ~4.87% population stroke rate in dataset
    relative_risk = prob / baseline_pop_rate

    # --- TOP KPI METRIC CARDS STRIP ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("5-Year Predicted Risk", f"{prob * 100:.1f}%")
    with k2:
        st.metric(
            "Relative Risk vs Population",
            f"{relative_risk:.1f}x",
            delta=f"{relative_risk - 1.0:+.1f}x",
            delta_color="inverse",
        )
    with k3:
        st.metric("Clinical Risk Category", tier["tier"])
    with k4:
        st.metric("Glycemic Status", "Diabetic (≥140)" if patient["avg_glucose_level"] >= 140 else "Normal / Pre-diabetic")

    st.markdown("---")

    # --- CENTER BENTO: Gauge + SHAP Waterfall ---
    c_gauge, c_shap = st.columns([1, 1])
    with c_gauge:
        st.markdown("### 🎯 Predictive Risk Dial")
        render_risk_gauge(prob, title="5-Year Stroke Risk")

    with c_shap:
        st.markdown("### 🔬 Feature Impact Breakdown (SHAP)")
        st.caption("Directional impact on prediction: Risk Escalators (Red) vs Protective Factors (Green)")
        render_shap_waterfall(contributions, max_features=7)

    st.markdown("---")

    # --- BOTTOM SECTION: Real-Time 'What-If' Sensitivity & Actionability ---
    st.markdown("### 🎛️ Clinical Intervention Sensitivity Simulator")
    st.caption("Adjust modifiable risk targets to evaluate projected risk reduction for patient counseling.")

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_glucose = st.slider(
            "Target Fasting Glucose (mg/dL)",
            min_value=70.0,
            max_value=250.0,
            value=float(min(patient["avg_glucose_level"], 100.0)),
            step=5.0,
            key="win_sim_glucose",
        )
    with sim_c2:
        sim_bmi = st.slider(
            "Target BMI (kg/m²)",
            min_value=18.5,
            max_value=40.0,
            value=float(min(patient["bmi"], 24.9)),
            step=0.5,
            key="win_sim_bmi",
        )
    with sim_c3:
        current_smk = patient["smoking_status"]
        default_idx = 0 if current_smk == "smokes" else CATEGORICAL_OPTIONS["smoking_status"].index(current_smk)
        sim_smoking = st.selectbox(
            "Smoking Cessation Goal",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=default_idx,
            key="win_sim_smoke",
        )

    # Simulated patient prediction
    sim_patient = patient.copy()
    sim_patient["avg_glucose_level"] = sim_glucose
    sim_patient["bmi"] = sim_bmi
    sim_patient["smoking_status"] = sim_smoking

    sim_prob, _ = predict_patient_risk(sim_patient)
    arr = (prob - sim_prob) * 100
    rrr = ((prob - sim_prob) / prob) * 100 if prob > 0 else 0

    res1, res2 = st.columns([1, 2])
    with res1:
        st.metric(
            "Simulated Post-Goal Risk",
            f"{sim_prob * 100:.1f}%",
            delta=f"{-arr:.1f}% (ARR)",
            delta_color="normal",
        )
    with res2:
        if arr > 0.5:
            st.success(
                f"🗣️ **Patient Counseling Script**: *'Achieving a target blood sugar of {sim_glucose:.0f} mg/dL, "
                f"BMI of {sim_bmi:.1f}, and {sim_smoking} reduces your 5-year stroke probability from "
                f"**{prob * 100:.1f}% down to {sim_prob * 100:.1f}%** (an absolute risk reduction of **{arr:.1f}%**, "
                f"representing a **{rrr:.0f}% relative risk reduction**).'*"
            )
        else:
            st.info("Biomarkers are already within the optimal target range.")

    # --- COLLAPSIBLE FORMATTED EHR NOTE ---
    with st.expander("📋 Exportable Clinical Consultation Note (EHR Ready)", expanded=False):
        top_driver = contributions.iloc[0]
        ehr_body = (
            f"STROKE RISK STRATIFICATION & INTERVENTION NOTE\n"
            f"---------------------------------------------------\n"
            f"Demographics: Age {patient['age']:.0f} | Sex: {patient['gender']} | Residence: {patient['Residence_type']}\n"
            f"Hypertension: {'Yes' if patient['hypertension'] else 'No'} | "
            f"Heart Disease: {'Yes' if patient['heart_disease'] else 'No'} | "
            f"Smoking: {patient['smoking_status']}\n"
            f"Glucose: {patient['avg_glucose_level']:.1f} mg/dL | BMI: {patient['bmi']:.1f} kg/m²\n\n"
            f"FINDINGS:\n"
            f"- Model Predicted 5-Year Stroke Risk: {prob * 100:.1f}%\n"
            f"- Relative Risk vs Population Baseline: {relative_risk:.1f}x\n"
            f"- Clinical Risk Category: {tier['tier'].upper()}\n"
            f"- Dominant Risk Contributor: {top_driver['feature']} ({top_driver['value']}) with SHAP impact +{top_driver['shap_value']:.2f}\n\n"
            f"INTERVENTION GOALS & PROJECTED RISK REDUCTION:\n"
            f"- Target Glucose: {sim_glucose:.0f} mg/dL | Target BMI: {sim_bmi:.1f} | Smoking: {sim_smoking}\n"
            f"- Projected Post-Intervention Risk: {sim_prob * 100:.1f}%\n"
            f"- Absolute Risk Reduction (ARR): {arr:.1f}% | Relative Risk Reduction (RRR): {rrr:.1f}%\n"
            f"- Clinical Pathway: {tier['recommendation']}\n"
        )
        st.text_area("Copy into Clinical Health Record:", ehr_body, height=220)

    # Surface state (Rule 5)
    render_state_inspector({"patient": patient, "simulated_target": sim_patient})
