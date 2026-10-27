"""Variant A: Guided Intake & Clinical Triage Wizard."""

from typing import Dict, Any
import streamlit as st
from src.prototype.presets import PATIENT_PRESETS, DEFAULT_PATIENT, CATEGORICAL_OPTIONS
from src.prototype.model_service import predict_patient_risk
from src.prototype.components import (
    render_risk_gauge,
    render_shap_waterfall,
    render_state_inspector,
    get_risk_tier,
)


def render_variant_a():
    """Render Variant A: Step-by-step guided intake with immediate triage output."""
    st.title("📋 Guided Clinical Intake & Stroke Triage (Variant A)")
    st.caption("A sequential step-by-step workflow with progressive disclosure and clinical triage notes.")

    # Preset selection toolbar
    st.markdown("##### ⚡ Quick Clinical Presets")
    preset_names = list(PATIENT_PRESETS.keys())
    selected_preset = st.selectbox(
        "Choose a representative patient archetype:",
        preset_names,
        index=0,
        key="var_a_preset",
    )
    preset_data = PATIENT_PRESETS[selected_preset]

    # Initialize or populate session state for inputs
    if "patient_a" not in st.session_state or st.session_state.get("last_preset_a") != selected_preset:
        st.session_state["patient_a"] = preset_data.copy()
        st.session_state["last_preset_a"] = selected_preset

    patient = st.session_state["patient_a"]

    # Stepped tabbed workflow
    tab1, tab2, tab3 = st.tabs([
        "1. Demographics & Lifestyle",
        "2. Clinical Vitals & Lab Biomarkers",
        "3. Triage Assessment & EHR Export",
    ])

    with tab1:
        st.subheader("Patient Background & Social History")
        c1, c2 = st.columns(2)
        with c1:
            patient["gender"] = st.selectbox(
                "Biological Sex",
                CATEGORICAL_OPTIONS["gender"],
                index=CATEGORICAL_OPTIONS["gender"].index(patient["gender"]),
                key="va_gender",
            )
            patient["age"] = float(st.slider(
                "Age (years)",
                min_value=1.0,
                max_value=90.0,
                value=float(patient["age"]),
                step=1.0,
                key="va_age",
            ))
            patient["ever_married"] = st.selectbox(
                "Marital Status",
                CATEGORICAL_OPTIONS["ever_married"],
                index=CATEGORICAL_OPTIONS["ever_married"].index(patient["ever_married"]),
                key="va_married",
            )
        with c2:
            patient["work_type"] = st.selectbox(
                "Occupation / Work Type",
                CATEGORICAL_OPTIONS["work_type"],
                index=CATEGORICAL_OPTIONS["work_type"].index(patient["work_type"]),
                key="va_work",
            )
            patient["Residence_type"] = st.selectbox(
                "Residence Environment",
                CATEGORICAL_OPTIONS["Residence_type"],
                index=CATEGORICAL_OPTIONS["Residence_type"].index(patient["Residence_type"]),
                key="va_residence",
            )
            patient["smoking_status"] = st.selectbox(
                "Smoking Status",
                CATEGORICAL_OPTIONS["smoking_status"],
                index=CATEGORICAL_OPTIONS["smoking_status"].index(patient["smoking_status"]),
                key="va_smoking",
            )

    with tab2:
        st.subheader("Cardiovascular & Metabolic Biomarkers")
        c1, c2 = st.columns(2)
        with c1:
            htn_bool = patient["hypertension"] == 1
            new_htn = st.checkbox("Diagnosed Hypertension", value=htn_bool, key="va_htn")
            patient["hypertension"] = 1 if new_htn else 0

            hd_bool = patient["heart_disease"] == 1
            new_hd = st.checkbox("Pre-existing Heart Disease", value=hd_bool, key="va_hd")
            patient["heart_disease"] = 1 if new_hd else 0

        with c2:
            patient["avg_glucose_level"] = float(st.number_input(
                "Average Glucose Level (mg/dL)",
                min_value=50.0,
                max_value=320.0,
                value=float(patient["avg_glucose_level"]),
                step=1.0,
                key="va_glucose",
            ))
            patient["bmi"] = float(st.number_input(
                "Body Mass Index (BMI, kg/m²)",
                min_value=10.0,
                max_value=70.0,
                value=float(patient["bmi"]),
                step=0.5,
                key="va_bmi",
            ))

    # Evaluate model
    prob, contributions = predict_patient_risk(patient)
    tier = get_risk_tier(prob)

    with tab3:
        st.subheader("Predictive Risk Stratification & Clinical Guidance")
        col_gauge, col_shap = st.columns([1, 1])

        with col_gauge:
            st.markdown("#### Clinical Risk Gauge")
            render_risk_gauge(prob, title="Estimated 5-Year Stroke Risk")

        with col_shap:
            st.markdown("#### Key Biomarker Contributions (SHAP)")
            render_shap_waterfall(contributions, max_features=6)

        st.markdown("---")
        st.markdown("#### 📄 Standard Clinical Consultation Note")
        ehr_text = (
            f"STROKE RISK ASSESSMENT NOTE\n"
            f"Patient Age: {patient['age']:.0f} | Sex: {patient['gender']} | BMI: {patient['bmi']:.1f}\n"
            f"Hypertension: {'Yes' if patient['hypertension'] else 'No'} | "
            f"Heart Disease: {'Yes' if patient['heart_disease'] else 'No'} | "
            f"Glucose: {patient['avg_glucose_level']:.1f} mg/dL\n"
            f"Smoking Status: {patient['smoking_status']}\n\n"
            f"ASSESSMENT:\n"
            f"- Model Predicted 5-Year Stroke Probability: {prob * 100:.1f}%\n"
            f"- Risk Tier: {tier['tier'].upper()}\n"
            f"- Primary Driving Factor: {contributions.iloc[0]['feature']} ({contributions.iloc[0]['value']})\n"
            f"- Recommended Pathway: {tier['recommendation']}\n"
        )
        st.text_area("Formatted EHR Note (Ready to copy):", ehr_text, height=160)

    # Surface state (Rule 5)
    render_state_inspector(patient)
