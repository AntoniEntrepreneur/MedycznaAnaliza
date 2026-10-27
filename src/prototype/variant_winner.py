"""Unified Winning Variant: Clinician-Tailored EHR Workstation (Tunable)."""

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
from src.prototype.tuner_component import render_tuner_panel


def render_winning_variant():
    """Render the synthesized winning variant with tunable CSS variables."""
    # Mount floating knob panel
    render_tuner_panel()

    # Inject CSS variables styling
    st.markdown(
        """
        <style>
        .tunable-card {
            background-color: var(--tune-card-bg, #caf0fe);
            border-radius: var(--tune-card-radius, 24px);
            box-shadow: var(--tune-card-shadow, 0 4px 12px rgba(0,0,0,0.08));
            padding: var(--tune-container-padding, 18px);
            border: 1px solid #bae6fd;
            margin-bottom: 16px;
            transition: all 0.2s ease;
        }
        .tunable-accent {
            color: var(--tune-accent-color, #2563eb) !important;
        }
        .pop-badge-wrapper {
            display: var(--tune-show-pop-badge, inline-block);
        }
        .dial-container {
            max-width: var(--tune-dial-size, 320px);
            margin: 0 auto;
        }
        .ehr-note-area textarea {
            font-family: var(--tune-ehr-font, ui-monospace, monospace) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🩺 Clinical Stroke Risk Calculator")
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
        st.markdown(
            f"""
            <div class="tunable-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">5-YEAR STROKE RISK</div>
                <div class="tunable-accent" style="font-size: 1.85rem; font-weight: 800; margin-top: 4px;">{prob * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        st.markdown(
            f"""
            <div class="tunable-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">RELATIVE RISK</div>
                <div style="font-size: 1.85rem; font-weight: 800; color: #0f172a; margin-top: 4px;">
                    {relative_risk:.1f}x
                </div>
                <div class="pop-badge-wrapper" style="font-size: 0.75rem; color: #e11d48; font-weight: 700;">
                    {relative_risk - 1.0:+.1f}x vs baseline
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="tunable-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">CLINICAL TIER</div>
                <div style="display: inline-block; margin-top: 6px; padding: 4px 12px; border-radius: 999px; background:{tier['bg']}; color:{tier['color']}; font-weight:800; font-size:1.1rem;">
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
            <div class="tunable-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #64748b; font-weight: 600;">GLYCEMIC STATUS</div>
                <div style="font-size: 1.3rem; font-weight: 700; color: {gly_color}; margin-top: 8px;">
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
            <div class="tunable-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 8px; color: #1e293b;">
                    🎯 Predictive Risk Dial
                </div>
                <div class="dial-container">
            """,
            unsafe_allow_html=True,
        )
        render_risk_gauge(prob, title="5-Year Stroke Risk")
        st.markdown("</div></div>", unsafe_allow_html=True)

    with c_shap:
        st.markdown(
            """
            <div class="tunable-card">
                <div style="font-weight: 700; font-size: 1.05rem; margin-bottom: 2px; color: #1e293b;">
                    🔬 Feature Impact Breakdown (SHAP)
                </div>
                <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 10px;">
                    Directional impact: Red = Escalators, Green = Protective
                </div>
            """,
            unsafe_allow_html=True,
        )
        render_shap_waterfall(contributions, max_features=7)
        st.markdown("</div>", unsafe_allow_html=True)

    # --- BOTTOM SECTION: Real-Time 'What-If' Sensitivity & Actionability ---
    st.markdown(
        """
        <div class="tunable-card">
            <div style="font-weight: 700; font-size: 1.15rem; color: #1e293b; margin-bottom: 4px;">
                🎛️ Clinical Intervention Sensitivity Simulator
            </div>
            <div style="font-size: 0.85rem; color: #64748b; margin-bottom: 14px;">
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
            st.info("Biomarkers are already within optimal target range.")

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

    # Surface state (Rule 5)
    render_state_inspector({"patient": patient, "simulated_target": sim_patient})
