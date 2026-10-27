"""Streamlit Clinical Risk Calculator — Medical Data Analysis (Stroke Prediction).

Clinician decision support workstation validated in Issue #7.
"""

from typing import Dict, Any
import streamlit as st
from src.app.presets import PATIENT_PRESETS, CATEGORICAL_OPTIONS
from src.app.model import predict_patient_risk
from src.app.components import (
    render_risk_gauge,
    render_shap_waterfall,
    get_risk_tier,
)

# Page configuration
st.set_page_config(
    page_title="Clinical Stroke Risk Calculator",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom validated styling (24px radius, #caf0fe ice-sky background)
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3.5rem;
        max-width: 1200px;
    }
    .clinic-card {
        background-color: #caf0fe;
        border-radius: 24px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.06);
        padding: 20px;
        border: 1px solid #bae6fd;
        margin-bottom: 16px;
    }
    .clinic-accent {
        color: #2563eb !important;
    }
    .ehr-note-area textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
        font-size: 13px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Render clinical risk calculator workstation."""
    st.title("🩺 Clinical Stroke Risk Calculator")
    st.caption("EHR Decision Support Workstation • Calibrated LightGBM & SHAP Attribution")

    # --- SIDEBAR: Presets & Categorized Inputs ---
    st.sidebar.header("⚡ Patient Profile & Inputs")

    preset_names = list(PATIENT_PRESETS.keys())
    selected_preset = st.sidebar.selectbox(
        "Load Clinical Archetype:",
        preset_names,
        index=0,
        key="app_preset",
    )
    preset_data = PATIENT_PRESETS[selected_preset]

    if "patient" not in st.session_state or st.session_state.get("last_preset") != selected_preset:
        st.session_state["patient"] = preset_data.copy()
        st.session_state["last_preset"] = selected_preset

    patient = st.session_state["patient"]

    with st.sidebar.expander("👤 Demographics & Social History", expanded=True):
        patient["gender"] = st.selectbox(
            "Biological Sex",
            CATEGORICAL_OPTIONS["gender"],
            index=CATEGORICAL_OPTIONS["gender"].index(patient["gender"]),
            key="in_gender",
        )
        patient["age"] = float(st.slider(
            "Age (years)",
            min_value=1.0,
            max_value=90.0,
            value=float(patient["age"]),
            step=1.0,
            key="in_age",
        ))
        patient["ever_married"] = st.selectbox(
            "Marital Status",
            CATEGORICAL_OPTIONS["ever_married"],
            index=CATEGORICAL_OPTIONS["ever_married"].index(patient["ever_married"]),
            key="in_married",
        )
        patient["work_type"] = st.selectbox(
            "Occupation Type",
            CATEGORICAL_OPTIONS["work_type"],
            index=CATEGORICAL_OPTIONS["work_type"].index(patient["work_type"]),
            key="in_work",
        )
        patient["Residence_type"] = st.selectbox(
            "Residence",
            CATEGORICAL_OPTIONS["Residence_type"],
            index=CATEGORICAL_OPTIONS["Residence_type"].index(patient["Residence_type"]),
            key="in_residence",
        )

    with st.sidebar.expander("🩺 Vitals & Laboratory Biomarkers", expanded=True):
        htn_bool = patient["hypertension"] == 1
        new_htn = st.checkbox("Diagnosed Hypertension", value=htn_bool, key="in_htn")
        patient["hypertension"] = 1 if new_htn else 0

        hd_bool = patient["heart_disease"] == 1
        new_hd = st.checkbox("Pre-existing Heart Disease", value=hd_bool, key="in_hd")
        patient["heart_disease"] = 1 if new_hd else 0

        patient["avg_glucose_level"] = float(st.slider(
            "Average Blood Glucose (mg/dL)",
            min_value=50.0,
            max_value=300.0,
            value=float(patient["avg_glucose_level"]),
            step=1.0,
            key="in_glucose",
        ))
        patient["bmi"] = float(st.slider(
            "Body Mass Index (BMI, kg/m²)",
            min_value=12.0,
            max_value=60.0,
            value=float(patient["bmi"]),
            step=0.2,
            key="in_bmi",
        ))
        patient["smoking_status"] = st.selectbox(
            "Smoking Status",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=CATEGORICAL_OPTIONS["smoking_status"].index(patient["smoking_status"]),
            key="in_smoking",
        )

    # Compute risk and attribution
    prob, contributions = predict_patient_risk(patient)
    tier = get_risk_tier(prob)
    baseline_pop_rate = 0.0487  # ~4.87% population stroke rate in Kaggle dataset
    relative_risk = prob / baseline_pop_rate

    # --- TOP KPI METRIC CARDS STRIP ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700;">5-YEAR STROKE RISK</div>
                <div class="clinic-accent" style="font-size: 1.9rem; font-weight: 800; margin-top: 4px;">{prob * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700;">RELATIVE RISK</div>
                <div style="font-size: 1.9rem; font-weight: 800; color: #0f172a; margin-top: 4px;">
                    {relative_risk:.1f}x
                </div>
                <div style="font-size: 0.78rem; color: #e11d48; font-weight: 700;">
                    {relative_risk - 1.0:+.1f}x vs baseline
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700;">CLINICAL TIER</div>
                <div style="display: inline-block; margin-top: 6px; padding: 4px 14px; border-radius: 999px; background:{tier['bg']}; color:{tier['color']}; font-weight:800; font-size:1.1rem; border: 1px solid {tier['border']};">
                    {tier['tier']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        glycemic = "Diabetic (≥140)" if patient["avg_glucose_level"] >= 140 else "Normal / Pre-DM"
        gly_color = "#dc2626" if patient["avg_glucose_level"] >= 140 else "#16a34a"
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700;">GLYCEMIC STATUS</div>
                <div style="font-size: 1.35rem; font-weight: 700; color: {gly_color}; margin-top: 8px;">
                    {glycemic}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- CENTER BENTO: Gauge + SHAP Waterfall ---
    c_gauge, c_shap = st.columns([1, 1])
    with c_gauge:
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; color: #0f172a;">
                    🎯 Predictive Risk Dial
                </div>
            """,
            unsafe_allow_html=True,
        )
        render_risk_gauge(prob, title="5-Year Stroke Risk")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_shap:
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; color: #0f172a;">
                    🔬 Feature Impact Breakdown (SHAP)
                </div>
                <div style="font-size: 0.82rem; color: #475569; margin-bottom: 10px;">
                    Directional impact: Red = Risk Escalators, Green = Protective Factors
                </div>
            """,
            unsafe_allow_html=True,
        )
        render_shap_waterfall(contributions, max_features=7)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- BOTTOM SECTION: Real-Time 'What-If' Sensitivity & Actionability ---
    st.markdown(
        """
        <div class="clinic-card">
            <div style="font-weight: 700; font-size: 1.15rem; color: #0f172a; margin-bottom: 4px;">
                🎛️ Clinical Intervention Sensitivity Simulator
            </div>
            <div style="font-size: 0.85rem; color: #475569; margin-bottom: 14px;">
                Adjust modifiable clinical targets to evaluate projected stroke risk reduction for patient counseling.
            </div>
        """,
        unsafe_allow_html=True,
    )

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_glucose = st.slider(
            "Target Fasting Glucose (mg/dL)",
            min_value=70.0,
            max_value=250.0,
            value=float(min(patient["avg_glucose_level"], 100.0)),
            step=5.0,
            key="sim_glucose",
        )
    with sim_c2:
        sim_bmi = st.slider(
            "Target BMI (kg/m²)",
            min_value=18.5,
            max_value=40.0,
            value=float(min(patient["bmi"], 24.9)),
            step=0.5,
            key="sim_bmi",
        )
    with sim_c3:
        current_smk = patient["smoking_status"]
        default_idx = 0 if current_smk == "smokes" else CATEGORICAL_OPTIONS["smoking_status"].index(current_smk)
        sim_smoking = st.selectbox(
            "Smoking Cessation Goal",
            CATEGORICAL_OPTIONS["smoking_status"],
            index=default_idx,
            key="sim_smoke",
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

    st.markdown("</div>", unsafe_allow_html=True)

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
        st.markdown('<div class="ehr-note-area">', unsafe_allow_html=True)
        st.text_area("Copy into Clinical Health Record:", ehr_body, height=220)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
