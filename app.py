"""Streamlit Clinical Risk Calculator — Medical Data Analysis (Stroke Prediction).

Clinician decision support workstation validated in Issue #7.
"""

from typing import Dict, Any, Optional
import streamlit as st
from src.app.presets import PATIENT_PRESETS, CATEGORICAL_OPTIONS
from src.app.model import predict_patient_risk
from src.app.components import (
    render_risk_gauge,
    render_shap_waterfall,
    render_population_bento_card,
    get_risk_tier,
)
from src.population import calculate_patient_percentiles
from src.ml import load_production_pipeline, predict_stroke_risk
from src.vascular_age import calculate_vascular_age

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


@st.cache_resource(show_spinner=False)
def get_production_pipeline_artifact():
    """Load and cache calibrated production model pipeline artifact."""
    try:
        return load_production_pipeline()
    except Exception:
        return None


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
    pipeline_artifact = get_production_pipeline_artifact()
    if pipeline_artifact is not None:
        risk_res = predict_stroke_risk(pipeline_artifact, patient)
        prob = risk_res["probability"]
        optimal_th = pipeline_artifact.get("optimal_threshold", 0.07)
    else:
        prob, _ = predict_patient_risk(patient)
        optimal_th = 0.07

    _, contributions = predict_patient_risk(patient)
    tier = get_risk_tier(prob)
    baseline_pop_rate = 0.0487  # ~4.87% population stroke rate in Kaggle dataset
    relative_risk = prob / baseline_pop_rate

    # Calculate vascular age via model inversion
    vasc_res = calculate_vascular_age(
        patient_risk=prob,
        pipeline_artifact=pipeline_artifact,
        chronological_age=float(patient["age"]),
        patient_gender=patient["gender"],
    )
    vasc_age = vasc_res["vascular_age"]
    delta_age = vasc_res["delta"]
    rejuv_potential = vasc_res["rejuvenation_potential"]

    # --- TOP 4-KPI SUMMARY STRIP ---
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">WIEK CHRONOLOGICZNY</div>
                <div style="font-size: 2.0rem; font-weight: 800; color: #0f172a; margin-top: 4px;">{patient['age']:.0f} lat</div>
                <div style="font-size: 0.78rem; color: #64748b; font-weight: 600; margin-top: 2px;">Metryka pacjenta ({patient['gender']})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k2:
        threshold_tag = (
            f"<span style='color: #dc2626;'>⚠️ > p* ({optimal_th * 100:.0f}%)</span>"
            if prob >= optimal_th
            else f"<span style='color: #16a34a;'>✅ ≤ p* ({optimal_th * 100:.0f}%)</span>"
        )
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">RYZYKO UDARU (5 LAT)</div>
                <div class="clinic-accent" style="font-size: 2.0rem; font-weight: 800; margin-top: 4px;">{prob * 100:.1f}%</div>
                <div style="font-size: 0.78rem; font-weight: 700; margin-top: 2px;">{threshold_tag}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k3:
        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">KATEGORIA RYZYKA</div>
                <div style="margin-top: 6px;">
                    <span style="display: inline-block; padding: 4px 14px; border-radius: 999px; background:{tier['bg']}; color:{tier['color']}; font-weight:800; font-size:1.1rem; border: 1px solid {tier['border']};">
                        {tier['tier']}
                    </span>
                </div>
                <div style="font-size: 0.76rem; color: #64748b; font-weight: 600; margin-top: 4px;">Przedział: {tier['cutoff_range']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with k4:
        if delta_age > 0:
            delta_badge = f"""<span style="display: inline-block; padding: 2px 7px; border-radius: 6px; background: #fee2e2; color: #dc2626; font-weight: 800; font-size: 0.82rem; border: 1px solid #fca5a5;">+{delta_age:.1f} lat</span>"""
            framing_subtext = f"""<div style="font-size: 0.76rem; color: #2563eb; font-weight: 700; margin-top: 4px;">Potencjał Odmłodzenia: -{rejuv_potential:.1f} lat</div>"""
        else:
            delta_badge = f"""<span style="display: inline-block; padding: 2px 7px; border-radius: 6px; background: #dcfce7; color: #16a34a; font-weight: 800; font-size: 0.82rem; border: 1px solid #86efac;">{delta_age:+.1f} lat</span>"""
            framing_subtext = f"""<div style="font-size: 0.76rem; color: #16a34a; font-weight: 700; margin-top: 4px;">Wiek naczyniowy optymalny</div>"""

        st.markdown(
            f"""
            <div class="clinic-card" style="text-align: center;">
                <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">WIEK NACZYNIOWY</div>
                <div style="display: flex; align-items: baseline; justify-content: center; gap: 6px; margin-top: 4px;">
                    <span style="font-size: 2.0rem; font-weight: 800; color: #0f172a;">{vasc_age:.0f} lat</span>
                    {delta_badge}
                </div>
                {framing_subtext}
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
        render_risk_gauge(
            prob,
            title="5-Year Stroke Risk",
            optimal_threshold=optimal_th,
            default_threshold=0.50,
        )
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

    # --- POPULATION BENCHMARK BENTO CARD (TICKET #27) ---
    render_population_bento_card(patient_risk=prob, patient_age=patient["age"])

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

    if pipeline_artifact is not None:
        sim_prob = predict_stroke_risk(pipeline_artifact, sim_patient)["probability"]
    else:
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
        pop_benchmarks = calculate_patient_percentiles(prob, patient["age"])
        ehr_body = (
            f"STROKE RISK STRATIFICATION & CLINICAL DECISION SUPPORT NOTE\n"
            f"---------------------------------------------------\n"
            f"Demographics: Age {patient['age']:.0f} | Sex: {patient['gender']} | Residence: {patient['Residence_type']}\n"
            f"Hypertension: {'Yes' if patient['hypertension'] else 'No'} | "
            f"Heart Disease: {'Yes' if patient['heart_disease'] else 'No'} | "
            f"Smoking: {patient['smoking_status']}\n"
            f"Glucose: {patient['avg_glucose_level']:.1f} mg/dL | BMI: {patient['bmi']:.1f} kg/m²\n\n"
            f"ASSESSMENT & MODEL FINDINGS:\n"
            f"- Predicted 5-Year Stroke Risk: {prob * 100:.1f}%\n"
            f"- Clinical Risk Category: {tier['tier'].upper()} ({tier['cutoff_range']})\n"
            f"- Screening Status: {'ABOVE OPTIMAL THRESHOLD (Positive)' if prob >= optimal_th else 'Below screening cutoff (Negative)'} (p* = {optimal_th * 100:.1f}%)\n"
            f"- Vascular Age: {vasc_age:.1f} lat (Delta: {delta_age:+.1f} lat)\n"
            f"- Framing / Guidance: {vasc_res['framing_message']}\n"
            f"- Relative Risk vs Baseline: {relative_risk:.1f}x\n"
            f"- Population Percentile Benchmarks: {pop_benchmarks['decade_percentile']:.1f}% ({pop_benchmarks['decade_label']}) | {pop_benchmarks['overall_percentile']:.1f}% (Adult Cohort Overall)\n"
            f"- Primary Risk Contributor: {top_driver['feature']} ({top_driver['value']}) with SHAP impact +{top_driver['shap_value']:.2f}\n\n"
            f"INTERVENTION GOALS & PROJECTED RISK REDUCTION:\n"
            f"- Target Glucose: {sim_glucose:.0f} mg/dL | Target BMI: {sim_bmi:.1f} | Smoking: {sim_smoking}\n"
            f"- Projected Post-Intervention Risk: {sim_prob * 100:.1f}%\n"
            f"- Absolute Risk Reduction (ARR): {arr:.1f}% | Relative Risk Reduction (RRR): {rrr:.1f}%\n"
            f"- Clinical Recommendation: {tier['recommendation']}\n"
        )
        st.markdown('<div class="ehr-note-area">', unsafe_allow_html=True)
        st.text_area("Copy into Clinical Health Record:", ehr_body, height=220)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
