"""Small Streamlit application for the shared stroke-analysis dataset."""

from typing import Any, Dict

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import LOCAL_RAW_DATA_PATH
from src.data import DatasetValidationError, curate_stroke_data, load_data
from src.ml import load_production_pipeline, predict_stroke_risk


@st.cache_data
def load_population() -> pd.DataFrame:
    """Load the validated adult cohort used by the notebook and application."""
    raw = load_data()
    return curate_stroke_data(raw).adult


@st.cache_resource
def train_pipeline(adult: pd.DataFrame) -> Dict[str, Any]:
    """Train the runtime model from the same curated cohort."""
    return load_production_pipeline(df_adult=adult)


def render_dataset_error(error: Exception) -> None:
    """Show an actionable dataset error instead of rendering an empty dashboard."""
    st.error(f"Nie można uruchomić analizy: {error}")
    st.info(
        "Napraw dane i uruchom aplikację ponownie. Wymagany jest prawidłowy plik "
        f"`{LOCAL_RAW_DATA_PATH}` z repozytorium."
    )


def validate_patient(patient: Dict[str, Any], feature_names: list[str]) -> list[str]:
    """Return actionable validation messages for the submitted patient profile."""
    errors = [
        f"Brak wartości dla cechy „{feature}”."
        for feature in feature_names
        if feature not in patient or pd.isna(patient[feature])
    ]
    for feature in NUMERIC_CONTINUOUS_COLS:
        value = patient.get(feature)
        try:
            valid_number = value is not None and pd.notna(value) and float(value) == float(value)
        except (TypeError, ValueError):
            valid_number = False
        if not valid_number:
            errors.append(f"Cecha „{feature}” musi być skończoną liczbą.")
    if float(patient.get("age", 0)) < 18:
        errors.append("Wiek musi wynosić co najmniej 18 lat, ponieważ model korzysta z kohorty dorosłych.")
    if float(patient.get("avg_glucose_level", 0)) <= 0:
        errors.append("Średni poziom glukozy musi być większy od 0.")
    if float(patient.get("bmi", 0)) <= 0:
        errors.append("BMI musi być większe od 0.")
    return errors


def _prevalence_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Calculate stroke prevalence for a categorical or binary feature."""
    result = (
        df.groupby(column, dropna=False)["stroke"]
        .agg(prevalence="mean", cohort_size="size")
        .reset_index()
    )
    result["label"] = result[column].fillna("Brak danych").astype(str)
    result["prevalence_percent"] = result["prevalence"] * 100
    return result


def render_population_view(adult: pd.DataFrame) -> None:
    """Render a concise overview of the adult study population."""
    st.header("Populacja badania")
    st.caption(
        "Opisowa analiza dorosłej kohorty po tej samej walidacji i kuracji danych, "
        "które wykorzystuje notebook."
    )

    prevalence = float(adult["stroke"].mean())
    stroke_group = adult[adult["stroke"] == 1]
    control_group = adult[adult["stroke"] == 0]
    hypertension = _prevalence_by(adult, "hypertension")

    metric_1, metric_2, metric_3 = st.columns(3)
    metric_1.metric("Liczebność kohorty", f"{len(adult):,}".replace(",", " "))
    metric_2.metric("Przypadki udaru", f"{int(adult['stroke'].sum()):,}".replace(",", " "))
    metric_3.metric("Prewalencja udaru", f"{prevalence:.2%}")

    st.subheader("Najważniejsze ustalenia")
    st.markdown(
        "\n".join(
            [
                f"- Udar wystąpił u **{prevalence:.2%}** osób w kohorcie.",
                f"- Mediana wieku wynosi **{stroke_group['age'].median():.1f} lat** "
                f"w grupie udaru i **{control_group['age'].median():.1f} lat** w grupie bez udaru.",
                f"- Prewalencja wśród osób z nadciśnieniem wynosi "
                f"**{hypertension.loc[hypertension['hypertension'] == 1, 'prevalence_percent'].iloc[0]:.1f}%**.",
            ]
        )
    )

    left, right = st.columns(2)
    with left:
        age_plot = px.histogram(
            adult,
            x="age",
            color=adult["stroke"].map({0: "Bez udaru", 1: "Udar"}),
            nbins=24,
            barmode="overlay",
            opacity=0.75,
            labels={"age": "Wiek", "count": "Liczba osób", "color": "Wynik"},
            title="Rozkład wieku według wyniku",
        )
        age_plot.update_layout(legend_title_text="Wynik")
        st.plotly_chart(age_plot, use_container_width=True)

    with right:
        age_bins = pd.cut(
            adult["age"],
            bins=[17, 40, 50, 60, 70, 120],
            labels=["18–39", "40–49", "50–59", "60–69", "70+"],
        )
        age_prevalence = (
            adult.assign(grupa_wiekowa=age_bins)
            .groupby("grupa_wiekowa", observed=False)["stroke"]
            .mean()
            .mul(100)
            .rename("prewalencja")
            .reset_index()
        )
        age_plot = px.bar(
            age_prevalence,
            x="grupa_wiekowa",
            y="prewalencja",
            labels={"grupa_wiekowa": "Grupa wieku", "prewalencja": "Udar (%)"},
            title="Prewalencja udaru według wieku",
            text_auto=".1f",
        )
        st.plotly_chart(age_plot, use_container_width=True)

    risk_factors = pd.concat(
        [
            _prevalence_by(adult, "hypertension").assign(czynnik="Nadciśnienie"),
            _prevalence_by(adult, "heart_disease").assign(czynnik="Choroba serca"),
        ],
        ignore_index=True,
    )
    risk_factors["poziom"] = risk_factors["label"].map({"0": "Nie", "1": "Tak"}).fillna(risk_factors["label"])
    factor_plot = px.bar(
        risk_factors,
        x="czynnik",
        y="prevalence_percent",
        color="poziom",
        barmode="group",
        labels={"czynnik": "Czynnik", "prevalence_percent": "Udar (%)", "poziom": "Obecność"},
        title="Prewalencja udaru według czynników chorobowych",
        text_auto=".1f",
    )
    st.plotly_chart(factor_plot, use_container_width=True)


def render_patient_view(adult: pd.DataFrame, result: Dict[str, Any]) -> None:
    """Render the second, deliberately small patient risk view."""
    st.header("Kalkulator ryzyka pacjenta")
    st.caption(
        "To akademicki szacunek statystyczny, a nie diagnoza medyczna ani rekomendacja leczenia."
    )

    with st.form("patient_form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Wiek", min_value=18.0, max_value=110.0, value=60.0)
            glucose = st.number_input("Średni poziom glukozy", min_value=0.0, value=100.0)
            bmi = st.number_input("BMI", min_value=0.0, value=28.0)
            gender = st.selectbox("Płeć", sorted(adult["gender"].dropna().unique()))
            smoking = st.selectbox("Status palenia", sorted(adult["smoking_status"].dropna().unique()))
        with right:
            hypertension = st.checkbox("Nadciśnienie")
            heart_disease = st.checkbox("Choroba serca")
            married = st.selectbox("W związku małżeńskim", sorted(adult["ever_married"].dropna().unique()))
            work_type = st.selectbox("Charakter pracy", sorted(adult["work_type"].dropna().unique()))
            residence = st.selectbox("Miejsce zamieszkania", sorted(adult["Residence_type"].dropna().unique()))
        submitted = st.form_submit_button("Oceń profil")

    if submitted:
        patient = {
            "age": age,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "gender": gender,
            "smoking_status": smoking,
            "hypertension": int(hypertension),
            "heart_disease": int(heart_disease),
            "ever_married": married,
            "work_type": work_type,
            "Residence_type": residence,
        }
        errors = validate_patient(patient, result["feature_names"])
        if errors:
            st.error("Nie można ocenić profilu:")
            for error in errors:
                st.write(f"- {error}")
            return
        try:
            prediction = predict_stroke_risk(result, patient)
        except (TypeError, ValueError, KeyError) as error:
            st.error(f"Nie można obliczyć wyniku. Sprawdź dane wejściowe: {error}")
            return
        st.metric("Szacowane prawdopodobieństwo udaru", f"{prediction['probability_percent']:.1f}%")
        st.write(
            f"Reguła: prawdopodobieństwo **≥ próg** oznacza klasyfikację dodatnią. "
            f"Próg maksimum F1: **{prediction['decision_threshold']:.3f}**. "
            f"Klasyfikacja: **{'powyżej' if prediction['screening_positive'] else 'poniżej'} progu**."
        )


def main() -> None:
    """Start the two-view population and patient application."""
    st.set_page_config(page_title="Analiza populacji udaru", page_icon="🧪", layout="wide")
    st.title("Analiza danych o udarze")
    try:
        adult = load_population()
        model_result = train_pipeline(adult)
    except (DatasetValidationError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        render_dataset_error(error)
        st.stop()

    view = st.sidebar.radio("Widok", ["Populacja", "Pacjent"])
    if view == "Populacja":
        render_population_view(adult)
    else:
        render_patient_view(adult, model_result)


if __name__ == "__main__":
    main()
