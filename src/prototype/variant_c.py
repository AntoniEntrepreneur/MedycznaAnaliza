"""Variant C: Comparative Intervention Studio (Shared Decision Making)."""

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


def render_variant_c():
    """Render Variant C: Side-by-side comparative intervention studio for shared decision making."""
    st.title("⚖️ Comparative Intervention Studio (Variant C)")
    st.caption("Side-by-side comparison between Current Patient State and Target Clinical Goals for shared decision making.")

    # Top archetype loader
    col_preset, col_info = st.columns([2, 3])
    with col_preset:
        preset_names = list(PATIENT_PRESETS.keys())
        selected_preset = st.selectbox(
            "Load Patient Scenario:",
            preset_names,
            index=0,
            key="var_c_preset",
        )
    with col_info:
        st.info("💡 **Clinical Goal**: Compare baseline risk directly against realistic therapeutic targets to motivate patient adherence.")

    preset_data = PATIENT_PRESETS[selected_preset]
    if "patient_c_base" not in st.session_state or st.session_state.get("last_preset_c") != selected_preset:
        st.session_state["patient_c_base"] = preset_data.copy()
        target_default = preset_data.copy()
        # Smart target defaults
        target_default["avg_glucose_level"] = min(preset_data["avg_glucose_level"], 95.0)
        target_default["smoking_status"] = "never smoked" if preset_data["smoking_status"] == "smokes" else preset_data["smoking_status"]
        target_default["bmi"] = min(preset_data["bmi"], 24.5)
        st.session_state["patient_c_target"] = target_default
        st.session_state["last_preset_c"] = selected_preset

    base_p = st.session_state["patient_c_base"]
    target_p = st.session_state["patient_c_target"]

    # Two prominent side-by-side columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 🔴 1. Current Baseline State")
        st.caption("Patient's current measured biomarkers")

        base_p["age"] = float(st.slider("Age", 1.0, 90.0, float(base_p["age"]), 1.0, key="vc_base_age"))
        base_p["gender"] = st.selectbox("Sex", CATEGORICAL_OPTIONS["gender"], index=CATEGORICAL_OPTIONS["gender"].index(base_p["gender"]), key="vc_base_gender")

        c1, c2 = st.columns(2)
        with c1:
            base_htn = st.checkbox("Hypertension", value=base_p["hypertension"] == 1, key="vc_base_htn")
            base_p["hypertension"] = 1 if base_htn else 0
        with c2:
            base_hd = st.checkbox("Heart Disease", value=base_p["heart_disease"] == 1, key="vc_base_hd")
            base_p["heart_disease"] = 1 if base_hd else 0

        base_p["avg_glucose_level"] = float(st.number_input("Current Glucose (mg/dL)", 50.0, 320.0, float(base_p["avg_glucose_level"]), 1.0, key="vc_base_glu"))
        base_p["bmi"] = float(st.number_input("Current BMI", 10.0, 60.0, float(base_p["bmi"]), 0.5, key="vc_base_bmi"))
        base_p["smoking_status"] = st.selectbox("Smoking Status", CATEGORICAL_OPTIONS["smoking_status"], index=CATEGORICAL_OPTIONS["smoking_status"].index(base_p["smoking_status"]), key="vc_base_smoke")

        # Evaluate base
        base_prob, base_contribs = predict_patient_risk(base_p)
        render_risk_gauge(base_prob, title="Baseline Stroke Risk")

    with col_right:
        st.markdown("### 🟢 2. Post-Intervention Goal")
        st.caption("Simulated outcome under clinical adherence")

        # Target demographic stays identical to base
        target_p["age"] = base_p["age"]
        target_p["gender"] = base_p["gender"]
        target_p["ever_married"] = base_p["ever_married"]
        target_p["work_type"] = base_p["work_type"]
        target_p["Residence_type"] = base_p["Residence_type"]

        c1, c2 = st.columns(2)
        with c1:
            target_htn = st.checkbox("Controlled BP (Normotensive)", value=target_p["hypertension"] == 1, key="vc_target_htn")
            target_p["hypertension"] = 1 if target_htn else 0
        with c2:
            target_hd = st.checkbox("Cardiology Management", value=target_p["heart_disease"] == 1, key="vc_target_hd")
            target_p["heart_disease"] = 1 if target_hd else 0

        target_p["avg_glucose_level"] = float(st.number_input("Target Glucose (mg/dL)", 50.0, 320.0, float(target_p["avg_glucose_level"]), 1.0, key="vc_target_glu"))
        target_p["bmi"] = float(st.number_input("Target BMI", 10.0, 60.0, float(target_p["bmi"]), 0.5, key="vc_target_bmi"))
        target_p["smoking_status"] = st.selectbox("Target Smoking Status", CATEGORICAL_OPTIONS["smoking_status"], index=CATEGORICAL_OPTIONS["smoking_status"].index(target_p["smoking_status"]), key="vc_target_smoke")

        # Evaluate target
        target_prob, target_contribs = predict_patient_risk(target_p)
        render_risk_gauge(target_prob, title="Post-Intervention Risk")

    # Center Comparison Banner
    arr = (base_prob - target_prob) * 100
    rrr = ((base_prob - target_prob) / base_prob) * 100 if base_prob > 0 else 0

    st.markdown("---")
    st.markdown("### 🎯 Shared Decision Making Summary")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("Baseline Stroke Probability", f"{base_prob * 100:.1f}%")
    with m2:
        st.metric("Goal Stroke Probability", f"{target_prob * 100:.1f}%", delta=f"{-arr:.1f}%", delta_color="normal")
    with m3:
        st.metric("Relative Risk Reduction (RRR)", f"{rrr:.1f}%", delta="Benefit", delta_color="normal")

    st.success(
        f"🗣️ **Patient Counseling Script**: *'By bringing your average blood sugar to {target_p['avg_glucose_level']:.0f} mg/dL, "
        f"lowering BMI to {target_p['bmi']:.1f}, and adhering to {'tobacco cessation' if target_p['smoking_status'] != 'smokes' else 'lifestyle goals'}, "
        f"we cut your 5-year stroke probability from **{base_prob * 100:.1f}% down to {target_prob * 100:.1f}%** "
        f"(a **{rrr:.0f}% relative risk reduction**).'*"
    )

    # Detailed SHAP Comparison Accordion
    with st.expander("📊 Compare Detailed Biomarker Explanations (Baseline vs Goal)", expanded=False):
        c_shap_base, c_shap_target = st.columns(2)
        with c_shap_base:
            st.markdown("**Baseline SHAP Factors**")
            render_shap_waterfall(base_contribs, max_features=5)
        with c_shap_target:
            st.markdown("**Target SHAP Factors**")
            render_shap_waterfall(target_contribs, max_features=5)

    # Surface state (Rule 5)
    render_state_inspector({"baseline": base_p, "intervention_goal": target_p})
