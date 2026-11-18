"""Aplikacja Streamlit: przegląd populacji z udarem i kalkulator ryzyka pacjenta."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path(__file__).parent / "data" / "raw" / "healthcare-dataset-stroke-data.csv"
RANDOM_STATE = 42
NUMERIC_CONTINUOUS_COLS = ["age", "avg_glucose_level", "bmi"]
BINARY_COLS = ["hypertension", "heart_disease"]
CATEGORICAL_COLS = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
FEATURES = NUMERIC_CONTINUOUS_COLS + BINARY_COLS + CATEGORICAL_COLS

CATEGORICAL_OPTIONS = {
    "gender": ["Female", "Male"],
    "ever_married": ["No", "Yes"],
    "work_type": ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
    "Residence_type": ["Urban", "Rural"],
    "smoking_status": ["never smoked", "formerly smoked", "smokes", "Unknown"],
}

# Polskie etykiety wyświetlane użytkownikowi (wartości przekazywane do modelu pozostają oryginalne)
VALUE_LABELS = {
    "Female": "Kobieta", "Male": "Mężczyzna", "Other": "Inna",
    "No": "Nie", "Yes": "Tak",
    "Private": "Sektor prywatny", "Self-employed": "Samozatrudnienie",
    "Govt_job": "Sektor publiczny", "children": "Dziecko", "Never_worked": "Nigdy nie pracował(a)",
    "Urban": "Miasto", "Rural": "Wieś",
    "never smoked": "Nigdy nie palił(a)", "formerly smoked": "Palił(a) dawniej",
    "smokes": "Pali", "Unknown": "Brak danych",
}

# Polskie nazwy cech używane w komunikatach walidacji
FEATURE_LABELS = {
    "age": "wiek", "avg_glucose_level": "średni poziom glukozy", "bmi": "BMI",
    "hypertension": "nadciśnienie", "heart_disease": "choroba serca",
    "gender": "płeć", "ever_married": "stan cywilny", "work_type": "rodzaj pracy",
    "Residence_type": "miejsce zamieszkania", "smoking_status": "status palenia",
}


@st.cache_data
def load_population() -> pd.DataFrame:
    """Wczytuje kohortę dorosłych (wiek >= 18, bez płci „Other”)."""
    df = pd.read_csv(DATA_PATH)
    return df[(df["gender"] != "Other") & (df["age"] >= 18)].reset_index(drop=True)


@st.cache_resource(show_spinner="Trenowanie modelu…")
def get_model_result() -> Dict[str, Any]:
    """Trenuje regresję logistyczną i wybiera próg maksymalizujący F1 (walidacja krzyżowa)."""
    adult = load_population()
    X, y = adult[FEATURES], adult["stroke"].astype(int)
    # Preprocessing: cechy liczbowe, kategoryczne i binarne
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUMERIC_CONTINUOUS_COLS),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_COLS),
        ("binary", SimpleImputer(strategy="most_frequent"), BINARY_COLS),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
    ])
    # Predykcje out-of-fold do doboru progu
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    probs = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
    candidates = np.unique(probs)
    scores = [f1_score(y, probs >= t, zero_division=0) for t in candidates]
    pipeline.fit(X, y)
    return {"pipeline": pipeline, "threshold": float(candidates[int(np.argmax(scores))])}


def render_dataset_error(error: Exception) -> None:
    """Pokazuje błąd danych zamiast pustego panelu."""
    st.error(f"Nie można uruchomić analizy: {error}")
    st.info(f"Wymagany jest plik `{DATA_PATH}`.")


def validate_patient(patient: Dict[str, Any], feature_names: list[str]) -> list[str]:
    """Zwraca komunikaty walidacji dla profilu pacjenta."""
    errors = [
        f"Brak wartości dla cechy „{FEATURE_LABELS.get(feature, feature)}”."
        for feature in feature_names
        if feature not in patient or patient[feature] in (None, "")
    ]
    for feature in NUMERIC_CONTINUOUS_COLS:
        value = patient.get(feature)
        try:
            valid_number = value is not None and np.isfinite(float(value))
        except (TypeError, ValueError):
            valid_number = False
        if not valid_number:
            errors.append(f"Cecha „{FEATURE_LABELS.get(feature, feature)}” musi być skończoną liczbą.")
    try:
        if float(patient.get("age", 0)) < 18:
            errors.append(
                "Wiek musi wynosić co najmniej 18 lat, ponieważ model korzysta z kohorty dorosłych."
            )
        if float(patient.get("avg_glucose_level", 0)) <= 0:
            errors.append("Średni poziom glukozy musi być większy od 0.")
        if float(patient.get("bmi", 0)) <= 0:
            errors.append("BMI musi być większe od 0.")
    except (TypeError, ValueError):
        pass
    for feature in CATEGORICAL_COLS:
        if patient.get(feature) not in CATEGORICAL_OPTIONS[feature]:
            errors.append(f"Nieprawidłowa wartość cechy „{FEATURE_LABELS.get(feature, feature)}”.")
    return errors


def _prevalence_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Oblicza prewalencję udaru dla cechy kategorycznej lub binarnej."""
    result = (
        df.groupby(column, dropna=False)["stroke"]
        .agg(prevalence="mean", cohort_size="size")
        .reset_index()
    )
    result["label"] = result[column].fillna("Brak danych").astype(str)
    result["prevalence_percent"] = result["prevalence"] * 100
    return result


def render_population_view(adult: pd.DataFrame) -> None:
    """Wyświetla zwięzły przegląd dorosłej populacji badania."""
    st.header("Populacja badania")
    st.caption(
        "Opisowa analiza wyselekcjonowanej kohorty dorosłych (wiek >= 18, bez rekordu "
        "`gender == Other`). Notebook dodatkowo opisuje cały zwalidowany zbiór danych."
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
        st.plotly_chart(age_plot, width="stretch")

    with right:
        # Podział na grupy wiekowe
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
        prevalence_plot = px.bar(
            age_prevalence,
            x="grupa_wiekowa",
            y="prewalencja",
            labels={"grupa_wiekowa": "Grupa wieku", "prewalencja": "Udar (%)"},
            title="Prewalencja udaru według wieku",
            text_auto=".1f",
        )
        st.plotly_chart(prevalence_plot, width="stretch")

    # Prewalencja według czynników chorobowych
    risk_factors = pd.concat(
        [
            _prevalence_by(adult, "hypertension").assign(czynnik="Nadciśnienie"),
            _prevalence_by(adult, "heart_disease").assign(czynnik="Choroba serca"),
        ],
        ignore_index=True,
    )
    risk_factors["poziom"] = risk_factors["label"].map({"0": "Nie", "1": "Tak"}).fillna(
        risk_factors["label"]
    )
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
    st.plotly_chart(factor_plot, width="stretch")


def render_calculator() -> None:
    """Wyświetla kalkulator pacjenta i wynik względem progu."""
    st.header("Kalkulator ryzyka pacjenta")
    st.caption(
        "To akademicki szacunek statystyczny, a nie diagnoza medyczna ani rekomendacja leczenia."
    )

    try:
        model_result = get_model_result()
    except Exception as exc:
        st.error(
            "Nie można przygotować modelu. Sprawdź, czy plik "
            f"`{DATA_PATH}` istnieje i zawiera wymagane kolumny. Szczegóły: {exc}"
        )
        return

    with st.form("patient-risk-form"):
        left, right = st.columns(2)
        with left:
            age = st.number_input("Wiek (lata)", min_value=18.0, max_value=120.0, value=50.0)
            glucose = st.number_input(
                "Średni poziom glukozy (mg/dL)", min_value=0.1, value=100.0
            )
            bmi = st.number_input("BMI (kg/m²)", min_value=0.1, value=25.0)
            gender = st.selectbox("Płeć", CATEGORICAL_OPTIONS["gender"], format_func=VALUE_LABELS.get)
            smoking = st.selectbox("Status palenia", CATEGORICAL_OPTIONS["smoking_status"], format_func=VALUE_LABELS.get)
        with right:
            hypertension = st.checkbox("Nadciśnienie tętnicze")
            heart_disease = st.checkbox("Choroba serca")
            married = st.selectbox("Stan cywilny", CATEGORICAL_OPTIONS["ever_married"], format_func=VALUE_LABELS.get)
            work_type = st.selectbox("Rodzaj pracy", CATEGORICAL_OPTIONS["work_type"], format_func=VALUE_LABELS.get)
            residence = st.selectbox("Miejsce zamieszkania", CATEGORICAL_OPTIONS["Residence_type"], format_func=VALUE_LABELS.get)
        submitted = st.form_submit_button("Oceń ryzyko")

    if not submitted:
        st.write("Wprowadź profil pacjenta i wybierz **Oceń ryzyko**.")
        return

    # Profil pacjenta w formacie oczekiwanym przez model
    patient = {
        "age": float(age),
        "avg_glucose_level": float(glucose),
        "bmi": float(bmi),
        "gender": gender,
        "smoking_status": smoking,
        "hypertension": int(hypertension),
        "heart_disease": int(heart_disease),
        "ever_married": married,
        "work_type": work_type,
        "Residence_type": residence,
    }
    # Walidacja danych wejściowych przed predykcją
    errors = validate_patient(patient, FEATURES)
    if errors:
        st.error("Nie można ocenić profilu:")
        for error in errors:
            st.write(f"- {error}")
        return

    # Predykcja prawdopodobieństwa udaru
    try:
        probability = float(model_result["pipeline"].predict_proba(pd.DataFrame([patient]))[:, 1][0])
    except Exception as exc:
        st.error(f"Nie można obliczyć wyniku. Sprawdź dane wejściowe. Szczegóły: {exc}")
        return

    st.metric("Szacowane prawdopodobieństwo udaru", f"{probability:.1%}")
    st.write(
        f"Reguła: prawdopodobieństwo **≥ próg** oznacza klasyfikację dodatnią. "
        f"Próg maksimum F1: **{model_result['threshold']:.3f}**. "
        f"Klasyfikacja: **{'powyżej' if probability >= model_result['threshold'] else 'poniżej'} progu**."
    )
    st.caption(
        "Prawdopodobieństwo i klasyfikacja względem progu służą wyłącznie do "
        "akademickiego przesiewu; nie są kliniczną kategorią ryzyka."
    )


def main() -> None:
    """Uruchamia aplikację z dwoma widokami: populacja i pacjent."""
    st.set_page_config(page_title="Analiza populacji udaru", page_icon="🩺", layout="wide")
    st.title("Analiza danych o udarze")
    view = st.sidebar.radio("Widok", ["Populacja", "Kalkulator pacjenta"])
    if view == "Populacja":
        try:
            render_population_view(load_population())
        except (FileNotFoundError, OSError, ValueError) as error:
            render_dataset_error(error)
    else:
        render_calculator()


if __name__ == "__main__":
    main()
