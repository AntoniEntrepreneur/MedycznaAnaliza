"""Streamlit Clinical Risk Calculator — Medical Data Analysis (Stroke Prediction).

Clinician decision support workstation validated in Spec #21.
"""

from typing import Dict, Any, Optional
import streamlit as st
from src.app.presets import PATIENT_PRESETS, CATEGORICAL_OPTIONS
from src.app.model import (
    predict_patient_risk,
    get_cached_global_importance,
    get_cached_archetype_case_studies,
)
from src.app.components import (
    render_risk_gauge,
    render_shap_waterfall,
    render_population_bento_card,
    get_risk_tier,
)
from src.population import calculate_patient_percentiles
from src.ml import load_production_pipeline, predict_stroke_risk
from src.vascular_age import calculate_vascular_age
from src.visualization import (
    plot_global_feature_importance_plotly,
    plot_shap_waterfall_plotly,
    plot_shap_waterfall,
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
        margin-bottom: 20px;
    }
    .clinic-header {
        font-size: 1.6rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .clinic-subtitle {
        font-size: 0.95rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .clinic-accent {
        color: #0284c7;
        font-weight: 700;
    }
    .ehr-note-area textarea {
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_production_pipeline_artifact():
    """Load and cache the trained production machine learning pipeline artifact."""
    try:
        return load_production_pipeline()
    except Exception:
        return None


def main() -> None:
    pipeline_artifact = get_production_pipeline_artifact()
    optimal_th = pipeline_artifact.get("optimal_threshold", 0.07) if pipeline_artifact else 0.07

    # --- SIDEBAR CONTROLS ---
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 15px 0;">
            <div style="font-size: 1.25rem; font-weight: 800; color: #0f172a;">🩺 Profil Pacjenta</div>
            <div style="font-size: 0.8rem; color: #64748b;">Elektroniczna Karta Pacjenta (POZ)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    preset_names = list(PATIENT_PRESETS.keys())
    selected_preset = st.sidebar.selectbox(
        "Kliniczne Profile Wzorcowe (Presets):",
        preset_names,
        index=0,
        help="Szybkie ładowanie predefiniowanych profili klinicznych.",
    )

    preset_data = PATIENT_PRESETS[selected_preset]

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dane Demograficzne")

    gender_idx = CATEGORICAL_OPTIONS["gender"].index(preset_data["gender"])
    gender = st.sidebar.selectbox("Płeć:", CATEGORICAL_OPTIONS["gender"], index=gender_idx)

    age = st.sidebar.slider("Wiek:", min_value=18, max_value=85, value=int(preset_data["age"]), step=1)

    ever_married_idx = CATEGORICAL_OPTIONS["ever_married"].index(preset_data["ever_married"])
    ever_married = st.sidebar.selectbox("W związku małżeńskim:", CATEGORICAL_OPTIONS["ever_married"], index=ever_married_idx)

    residence_idx = CATEGORICAL_OPTIONS["Residence_type"].index(preset_data["Residence_type"])
    residence = st.sidebar.selectbox("Miejsce zamieszkania:", CATEGORICAL_OPTIONS["Residence_type"], index=residence_idx)

    work_type_idx = CATEGORICAL_OPTIONS["work_type"].index(preset_data["work_type"])
    work_type = st.sidebar.selectbox("Charakter pracy:", CATEGORICAL_OPTIONS["work_type"], index=work_type_idx)

    st.sidebar.markdown("### Biomarkery Kliniczne")

    glucose = st.sidebar.slider(
        "Średni poziom glukozy (mg/dL):",
        min_value=50.0,
        max_value=300.0,
        value=float(preset_data["avg_glucose_level"]),
        step=1.0,
    )

    bmi = st.sidebar.slider(
        "Wskaźnik masy ciała BMI (kg/m²):",
        min_value=15.0,
        max_value=60.0,
        value=float(preset_data["bmi"]),
        step=0.1,
    )

    st.sidebar.markdown("### Wywiad Chorobowy")

    hypertension_val = bool(preset_data["hypertension"])
    hypertension = st.sidebar.checkbox(
        "Nadciśnienie tętnicze (Hypertension)",
        value=hypertension_val,
    )

    heart_disease_val = bool(preset_data["heart_disease"])
    heart_disease = st.sidebar.checkbox(
        "Choroba wieńcowa / serca (Heart Disease)",
        value=heart_disease_val,
    )

    smoking_idx = CATEGORICAL_OPTIONS["smoking_status"].index(preset_data["smoking_status"])
    smoking = st.sidebar.selectbox(
        "Status palenia tytoniu:",
        CATEGORICAL_OPTIONS["smoking_status"],
        index=smoking_idx,
    )

    # Current patient payload
    patient: Dict[str, Any] = {
        "gender": gender,
        "age": float(age),
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence,
        "avg_glucose_level": float(glucose),
        "bmi": float(bmi),
        "smoking_status": smoking,
    }

    # Model prediction
    if pipeline_artifact is not None:
        risk_out = predict_stroke_risk(pipeline_artifact, patient)
        prob = float(risk_out["probability"] if isinstance(risk_out, dict) else risk_out[0])
        _, contributions = predict_patient_risk(patient)
    else:
        prob, contributions = predict_patient_risk(patient)

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

    # Top-level workstation navigation tabs
    tab_calc, tab_xai = st.tabs(
        ["🩺 Kalkulator Kliniczny", "🔬 Interpretowalność (XAI)"]
    )

    with tab_calc:
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
                delta_badge = f"""<span style="display: inline-block; padding: 2px 7px; border-radius: 6px; background: #dcfce7; color: #16a34a; font-weight: 800; font-size: 0.82rem; border: 1px solid #86efac;">{delta_age:.1f} lat</span>"""
                framing_subtext = f"""<div style="font-size: 0.76rem; color: #16a34a; font-weight: 700; margin-top: 4px;">Optymalny profil naczyniowy</div>"""

            st.markdown(
                f"""
                <div class="clinic-card" style="text-align: center;">
                    <div style="font-size: 0.8rem; color: #475569; font-weight: 700; text-transform: uppercase;">WIEK NACZYNIOWY</div>
                    <div style="font-size: 2.0rem; font-weight: 800; color: #0f172a; margin-top: 4px;">
                        {vasc_age:.1f} lat {delta_badge}
                    </div>
                    {framing_subtext}
                </div>
                """,
                unsafe_allow_html=True,
            )

        # --- MAIN TWO-COLUMN WORKSTATION LAYOUT ---
        col_left, col_right = st.columns([1, 1], gap="medium")

        with col_left:
            st.markdown(
                """
                <div class="clinic-card">
                    <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">
                        🎯 Obiektywne Oszacowanie Ryzyka Udaru
                    </div>
                    <div style="font-size: 0.83rem; color: #475569; margin-bottom: 12px;">
                        Półkolisty wskaźnik polarny z odcięciem skriningowym F2 (p* = 7%) oraz progiem domyślnym (50%).
                    </div>
                """,
                unsafe_allow_html=True,
            )
            render_risk_gauge(prob, optimal_threshold=optimal_th)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown(
                """
                <div class="clinic-card">
                    <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">
                        🧬 Diagnostyka i Czynniki Ryzyka (SHAP)
                    </div>
                    <div style="font-size: 0.83rem; color: #475569; margin-bottom: 8px;">
                        Wpływ poszczególnych cech na odchylenie ryzyka od średniej populacyjnej.
                    </div>
                """,
                unsafe_allow_html=True,
            )
            render_shap_waterfall(contributions)
            st.markdown("</div>", unsafe_allow_html=True)

        # --- POPULATION RISK DISTRIBUTION CARD (Ticket #27) ---
        render_population_bento_card(patient_risk=prob, patient_age=patient["age"])

        # --- WHAT-IF SENSITIVITY SIMULATOR ---
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.15rem; color: #0f172a; margin-bottom: 4px;">
                    🔮 Symulator Wrażliwości i Celów Terapeutycznych (What-If Analysis)
                </div>
                <div style="font-size: 0.85rem; color: #475569; margin-bottom: 16px;">
                    Przetestuj wpływ potencjalnych modyfikacji stylu życia i leczenia farmakologicznego na prognozowane ryzyko.
                </div>
            """,
            unsafe_allow_html=True,
        )

        w1, w2, w3 = st.columns(3)
        with w1:
            sim_glucose = st.slider(
                "Cel glukozy (mg/dL):",
                50.0,
                250.0,
                min(float(glucose), 100.0),
                step=5.0,
                key="sim_glucose",
            )
        with w2:
            sim_bmi = st.slider(
                "Cel BMI (kg/m²):",
                18.5,
                45.0,
                min(float(bmi), 24.5),
                step=0.5,
                key="sim_bmi",
            )
        with w3:
            sim_smoking = st.selectbox(
                "Modyfikacja palenia:",
                ["Zachowaj obecny", "never smoked"],
                index=0 if smoking == "never smoked" else 1,
                key="sim_smoking",
            )

        sim_patient = patient.copy()
        sim_patient["avg_glucose_level"] = float(sim_glucose)
        sim_patient["bmi"] = float(sim_bmi)
        if sim_smoking == "never smoked":
            sim_patient["smoking_status"] = "never smoked"

        if pipeline_artifact is not None:
            sim_risk_out = predict_stroke_risk(pipeline_artifact, sim_patient)
            sim_prob = float(sim_risk_out["probability"] if isinstance(sim_risk_out, dict) else sim_risk_out[0])
        else:
            sim_prob, _ = predict_patient_risk(sim_patient)

        arr = (prob - sim_prob) * 100
        rrr = ((prob - sim_prob) / prob) * 100 if prob > 0 else 0

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            st.metric("Skorygowane Ryzyko", f"{sim_prob * 100:.1f}%", delta=f"{-arr:.1f}%", delta_color="inverse")
        with sim_c2:
            st.metric("Bezwzględna Redukcja Ryzyka (ARR)", f"{max(0.0, arr):.1f}%")
        with sim_c3:
            st.metric("Względna Redukcja Ryzyka (RRR)", f"{max(0.0, rrr):.1f}%")

        if arr > 0.5:
            st.success(
                f"🌟 **Korzyść kliniczna:** Osiągnięcie założonych celów może obniżyć 5-letnie ryzyko udaru o "
                f"**{arr:.1f}% punktów procentowych** (z {prob * 100:.1f}% do {sim_prob * 100:.1f}%, "
                f"co odpowiada **{rrr:.0f}% względnej redukcji ryzyka**)."
            )
        else:
            st.info("Biomarkery są już w optymalnym zakresie docelowym.")

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

    with tab_xai:
        # --- TAB 2: EXPLAINABLE AI (XAI) CLINICAL DASHBOARD ---
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 800; font-size: 1.25rem; color: #0f172a; margin-bottom: 4px;">
                    🔬 Wyjaśnialna Sztuczna Inteligencja w Diagnostyce Udaru (XAI)
                </div>
                <div style="font-size: 0.88rem; color: #475569; line-height: 1.5;">
                    Transparentna interpretacja modeli uczenia maszynowego w medycynie.
                    Łączymy <b>globalną hierarchię biomarkerów</b> populacji z <b>lokalną atrybucją SHAP (Shapley Additive exPlanations)</b>
                    dla zindywidualizowanych studiów przypadków klinicznych.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 1. Global Feature Importance Section
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">
                    📊 Globalna Ważność Cech Klinicznych (Global Feature Importance)
                </div>
                <div style="font-size: 0.83rem; color: #475569; margin-bottom: 12px;">
                    Hierarchia predyktorów w wytrenowanym modelu populacyjnym na kohorcie dorosłych pacjentów.
                </div>
            """,
            unsafe_allow_html=True,
        )

        df_importances = get_cached_global_importance()
        top_n_choice = st.slider(
            "Liczba prezentowanych cech kluczowych (Top N):",
            min_value=5,
            max_value=len(df_importances),
            value=min(10, len(df_importances)),
            step=1,
            key="xai_top_n",
        )

        fig_global = plot_global_feature_importance_plotly(
            df_importances,
            top_n=top_n_choice,
            title="Globalna Ważność Cech Klinicznych",
        )
        st.plotly_chart(fig_global, use_container_width=True)

        st.markdown(
            """
            <div style="font-size: 0.84rem; color: #334155; background: #ffffff; padding: 12px; border-radius: 12px; border: 1px solid #bae6fd; margin-top: 8px;">
                💡 <b>Kluczowy Wniosek Epidemiologiczny:</b> Wiek pacjenta, średni poziom glukozy we krwi oraz wskaźnik masy ciała (BMI)
                odpowiadają za ponad <b>85%</b> całkowitej mocy predykcyjnej modelu. Nadciśnienie tętnicze i status palenia tytoniu
                stanowią bezpośrednie modyfikowalne czynniki kardiometaboliczne.
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # 2. Local Clinical Archetypes SHAP Waterfall Section
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">
                    🩺 Kliniczne Studia Przypadków: Wykresy Kaskadowe SHAP (Waterfall Case Studies)
                </div>
                <div style="font-size: 0.83rem; color: #475569; margin-bottom: 12px;">
                    Wizualizacja przesunięć ryzyka dla 3 reprezentatywnych archetypów z kohorty testowej:
                    czerwone słupki (+) podwyższają prawdopodobieństwo udaru, niebieskie (-) pełnią funkcję protekcyjną.
                </div>
            """,
            unsafe_allow_html=True,
        )

        archetypes = get_cached_archetype_case_studies()
        archetype_keys = list(archetypes.keys())
        selected_archetype_key = st.radio(
            "Wybierz profil kliniczny do analizy SHAP:",
            archetype_keys,
            format_func=lambda k: f"{'🔴' if k == 'True Positive' else '🟢' if k == 'True Negative' else '🟡'} {k}: {archetypes[k]['title_pl']}",
            horizontal=True,
            key="xai_arch_selector",
        )

        arch_data = archetypes[selected_archetype_key]
        arch_patient = arch_data["patient"]

        # Archetype Summary Metrics
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Predykowane Ryzyko", f"{arch_data['predicted_risk'] * 100:.1f}%")
        with m2:
            outcome_text = "Udar (1)" if arch_data["true_label"] == 1 else "Brak udaru (0)"
            st.metric("Rzeczywisty Wynik", outcome_text)
        with m3:
            st.metric("Wiek Pacjenta", f"{arch_patient.get('age', 0):.0f} lat")
        with m4:
            htn_text = "Tak" if arch_patient.get("hypertension", 0) == 1 else "Nie"
            st.metric("Nadciśnienie Tętnicze", htn_text)

        # Waterfall Plot for Archetype
        fig_arch_waterfall = plot_shap_waterfall_plotly(
            explanation_or_shap_values={
                "features": arch_data["feature_names"],
                "shap_values": arch_data["shap_values"],
                "base_value": arch_data["base_value"],
                "data": arch_data["data"],
            },
            patient_row=arch_patient,
            archetype_name=f"{selected_archetype_key} — {arch_data['title_pl']}",
            max_display=8,
        )
        st.plotly_chart(fig_arch_waterfall, use_container_width=True)

        st.info(f"📋 **Ocena Kliniczna Archetypu:** {arch_data['description_pl']}")
        st.markdown("</div>", unsafe_allow_html=True)

        # 3. Live Interactive Patient SHAP Explanation Section
        st.markdown(
            """
            <div class="clinic-card">
                <div style="font-weight: 700; font-size: 1.1rem; color: #0f172a; margin-bottom: 2px;">
                    🎯 Zindywidualizowany Wykres SHAP Waterfall dla Aktywnego Pacjenta
                </div>
                <div style="font-size: 0.83rem; color: #475569; margin-bottom: 12px;">
                    Interaktywna dekompozycja wkładu każdego biomarkerka dla parametrów pacjenta ustawionych w panelu bocznym.
                </div>
            """,
            unsafe_allow_html=True,
        )

        fig_patient_waterfall = plot_shap_waterfall_plotly(
            explanation_or_shap_values=contributions,
            patient_row=patient,
            archetype_name=f"Aktywny Pacjent ({selected_preset})",
            max_display=8,
        )
        st.plotly_chart(fig_patient_waterfall, use_container_width=True)

        with st.expander("📥 Zobacz i pobierz wersję publikacyjną (Matplotlib High-DPI)", expanded=False):
            fig_mat = plot_shap_waterfall(
                explanation_or_shap_values=contributions,
                patient_row=patient,
                archetype_name=f"Pacjent: {selected_preset}",
                max_display=8,
                as_plotly=False,
            )
            st.pyplot(fig_mat, use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
