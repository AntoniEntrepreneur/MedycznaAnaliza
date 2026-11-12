"""Small two-view Streamlit application for the academic stroke-risk model."""

from typing import Any, Dict

import numpy as np
import streamlit as st

from src.config import CATEGORICAL_COLS, NUMERIC_CONTINUOUS_COLS
from src.ml import load_production_pipeline, predict_stroke_risk


CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male", "Other"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
}


@st.cache_resource(show_spinner="Trenowanie modelu na śledzonym zbiorze danych…")
def get_model_result() -> Dict[str, Any]:
    """Train and cache the in-memory shared Logistic Regression pipeline."""
    return load_production_pipeline()


def validate_patient(patient: Dict[str, Any], feature_names: list[str]) -> list[str]:
    """Return actionable validation messages for one patient profile."""
    errors = [
        f"Brak wartości dla cechy „{feature}”." for feature in feature_names
        if feature not in patient or patient[feature] in (None, "")
    ]
    for feature in NUMERIC_CONTINUOUS_COLS:
        value = patient.get(feature)
        if value is None or not np.isfinite(float(value)):
            errors.append(f"Cecha „{feature}” musi być skończoną liczbą.")
    if float(patient.get("age", 0)) < 18:
        errors.append("Wiek musi wynosić co najmniej 18 lat, ponieważ model korzysta z kohorty dorosłych.")
    if float(patient.get("avg_glucose_level", 0)) <= 0:
        errors.append("Średni poziom glukozy musi być większy od 0.")
    if float(patient.get("bmi", 0)) <= 0:
        errors.append("BMI musi być większe od 0.")
    for feature in CATEGORICAL_COLS:
        if patient.get(feature) not in CATEGORICAL_OPTIONS[feature]:
            errors.append(f"Nieprawidłowa wartość cechy „{feature}”.")
    return errors


def render_calculator() -> None:
    """Render the patient calculator and its threshold-based result."""
    st.title("Kalkulator ryzyka udaru")
    st.caption("Model Logistic Regression jest trenowany w pamięci przy pierwszym użyciu.")
    st.info(
        "To wyłącznie akademicki szacunek statystyczny, a nie diagnoza medyczna "
        "ani rekomendacja leczenia."
    )

    try:
        model_result = get_model_result()
    except Exception as exc:
        st.error(
            "Nie można przygotować modelu. Sprawdź, czy plik "
            "`data/raw/healthcare-dataset-stroke-data.csv` istnieje i zawiera wymagane kolumny. "
            f"Szczegóły: {exc}"
        )
        return

    with st.form("patient-risk-form"):
        st.subheader("Profil pacjenta")
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Płeć", CATEGORICAL_OPTIONS["gender"])
            age = st.number_input("Wiek (lata)", min_value=18.0, max_value=120.0, value=50.0)
            hypertension = st.checkbox("Nadciśnienie tętnicze")
            heart_disease = st.checkbox("Choroba serca")
            ever_married = st.selectbox("Stan cywilny", CATEGORICAL_OPTIONS["ever_married"])
        with col2:
            work_type = st.selectbox("Rodzaj pracy", CATEGORICAL_OPTIONS["work_type"])
            residence = st.selectbox("Miejsce zamieszkania", CATEGORICAL_OPTIONS["Residence_type"])
            glucose = st.number_input(
                "Średni poziom glukozy (mg/dL)", min_value=0.1, value=100.0
            )
            bmi = st.number_input("BMI (kg/m²)", min_value=0.1, value=25.0)
            smoking = st.selectbox("Status palenia", CATEGORICAL_OPTIONS["smoking_status"])
        submitted = st.form_submit_button("Oceń ryzyko")

    if not submitted:
        st.write("Wprowadź profil pacjenta i wybierz **Oceń ryzyko**.")
        return

    patient = {
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
    errors = validate_patient(patient, model_result["feature_names"])
    if errors:
        st.error("Nie można ocenić profilu:")
        for error in errors:
            st.write(f"- {error}")
        return

    try:
        result = predict_stroke_risk(model_result, patient)
    except Exception as exc:
        st.error(f"Nie można obliczyć wyniku. Sprawdź dane wejściowe. Szczegóły: {exc}")
        return

    probability = result["probability"]
    threshold = result["decision_threshold"]
    st.subheader("Wynik")
    st.metric("Prawdopodobieństwo udaru w danych modelu", f"{probability:.1%}")
    classification = "powyżej" if result["screening_positive"] else "poniżej"
    st.write(
        f"Klasyfikacja względem progu maksimum F1: **{classification} progu** "
        f"({threshold:.3f})."
    )
    st.caption(
        "Reguła: wynik jest dodatni, gdy prawdopodobieństwo ≥ próg wybrany jako maksimum F1 "
        "na wydzielonej części danych."
    )


def render_about() -> None:
    """Render the short model and limitation view."""
    st.title("O modelu")
    st.write(
        "Aplikacja wykorzystuje wspólny pipeline z regresją logistyczną, imputacją braków, "
        "skalowaniem zmiennych liczbowych i kodowaniem cech kategorycznych."
    )
    st.write(
        "Pipeline jest trenowany w czasie działania z pliku "
        "`data/raw/healthcare-dataset-stroke-data.csv`. Nie jest odczytywany żaden "
        "serializowany artefakt modelu."
    )
    st.write(
        "Próg klasyfikacji jest wybierany przez wspólną funkcję optymalizacji maksimum F1 "
        "na tej samej walidacji, co wynik modelu."
    )
    st.warning(
        "Wynik ma charakter akademicki i nie stanowi diagnozy, porady medycznej "
        "ani rekomendacji leczenia."
    )


def main() -> None:
    """Run one of the application's two views."""
    st.set_page_config(page_title="Akademicki kalkulator ryzyka", page_icon="🩺")
    view = st.sidebar.radio("Widok", ["Kalkulator pacjenta", "O modelu"])
    if view == "Kalkulator pacjenta":
        render_calculator()
    else:
        render_about()


if __name__ == "__main__":
    main()
