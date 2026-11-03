"""Population stroke risk distribution and percentile benchmarking service."""

from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from src.data import get_curated_cohorts
from src.ml import load_production_pipeline, DEFAULT_FEATURE_COLS

DECADE_BINS = [18.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 150.0]
DECADE_CODES = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-79", "80+"]
DECADE_PL_LABELS = {
    "18-29": "18–29 lat",
    "30-39": "30–39 lat",
    "40-49": "40–49 lat",
    "50-59": "50–59 lat",
    "60-69": "60–69 lat",
    "70-79": "70–79 lat",
    "80+": "80+ lat",
}

# Module-level memory cache for baseline population risk DataFrame
_CACHED_BASELINE_DF: Optional[pd.DataFrame] = None


def get_age_decade_label(age: float) -> Tuple[str, str]:
    """Map patient age to decade classification code and Polish clinical label."""
    if age < 30.0:
        code = "18-29"
    elif age < 40.0:
        code = "30-39"
    elif age < 50.0:
        code = "40-49"
    elif age < 60.0:
        code = "50-59"
    elif age < 70.0:
        code = "60-69"
    elif age < 80.0:
        code = "70-79"
    else:
        code = "80+"
    label = DECADE_PL_LABELS.get(code, f"{code} lat")
    return code, label


def get_population_baseline_risks(
    pipeline_artifact: Optional[Any] = None,
    df_adult: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Precompute and cache stroke risk probabilities and age decades for adult cohort (N = 4,253)."""
    global _CACHED_BASELINE_DF

    if pipeline_artifact is None and df_adult is None and _CACHED_BASELINE_DF is not None:
        return _CACHED_BASELINE_DF.copy()

    # Retrieve curated adult cohort (N=4,253)
    if df_adult is None:
        cohorts = get_curated_cohorts()
        df = cohorts.adult.copy()
    else:
        df = df_adult.copy()

    # Load production pipeline if not explicitly provided
    if pipeline_artifact is None:
        pipeline_artifact = load_production_pipeline()

    if isinstance(pipeline_artifact, dict) and "pipeline" in pipeline_artifact:
        pipeline = pipeline_artifact["pipeline"]
        feature_names = pipeline_artifact.get("feature_names", DEFAULT_FEATURE_COLS)
    else:
        pipeline = pipeline_artifact
        feature_names = getattr(pipeline_artifact, "feature_names_in_", DEFAULT_FEATURE_COLS)

    features = [c for c in feature_names if c in df.columns]
    probs = pipeline.predict_proba(df[features])[:, 1]

    # Assign age decades
    decades = pd.cut(df["age"], bins=DECADE_BINS, labels=DECADE_CODES, right=False)
    # Fallback for any unexpected missing categories (e.g. out of range)
    if decades.isna().any():
        decades = decades.astype(str).replace({"nan": "18-29"})

    decade_labels = [DECADE_PL_LABELS.get(str(d), f"{d} lat") for d in decades]

    baseline_df = pd.DataFrame({
        "age": df["age"].values,
        "stroke_risk": probs,
        "decade": decades.astype(str).values,
        "age_decade": decades.astype(str).values,
        "decade_label": decade_labels,
    })

    if pipeline_artifact is None or (isinstance(pipeline_artifact, dict) and "pipeline" in pipeline_artifact):
        _CACHED_BASELINE_DF = baseline_df.copy()

    return baseline_df


def calculate_patient_percentiles(
    patient_risk: float,
    patient_age: float,
    baseline_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute empirical percentile ranks for patient risk vs age decade cohort and overall adult population."""
    if baseline_df is None:
        baseline_df = get_population_baseline_risks()

    # Normalize percentage risk to fraction if passed in 0-100 scale
    norm_risk = patient_risk / 100.0 if patient_risk > 1.0 and baseline_df["stroke_risk"].max() <= 1.0 else patient_risk
    norm_risk = float(np.clip(norm_risk, 0.0, 1.0))

    decade_code, decade_label = get_age_decade_label(float(patient_age))

    decade_mask = (baseline_df["decade"] == decade_code) | (baseline_df["age_decade"] == decade_code)
    decade_risks = baseline_df.loc[decade_mask, "stroke_risk"].values
    if len(decade_risks) == 0:
        decade_risks = baseline_df["stroke_risk"].values

    decade_pct = float((decade_risks <= norm_risk).mean() * 100.0)
    overall_pct = float((baseline_df["stroke_risk"] <= norm_risk).mean() * 100.0)

    # Polish clinical interpretation
    if decade_pct >= 90.0:
        interp = (
            f"Bardzo wysokie ryzyko w grupie rówieśniczej: szacowane 5-letnie ryzyko udaru jest wyższe "
            f"niż u {decade_pct:.1f}% osób w Twoim wieku ({decade_label}) oraz u {overall_pct:.1f}% całej populacji dorosłych."
        )
    elif decade_pct >= 60.0:
        interp = (
            f"Podwyższone ryzyko: szacowane ryzyko udaru jest wyższe niż u {decade_pct:.1f}% osób "
            f"w Twojej grupie wiekowej ({decade_label}) i u {overall_pct:.1f}% dorosłych ogółem."
        )
    elif decade_pct >= 40.0:
        interp = (
            f"Typowe ryzyko populacyjne: wynik plasuje się w pobliżu mediany rówieśników ({decade_pct:.1f}. centyl "
            f"dla {decade_label}, {overall_pct:.1f}. centyl ogółu dorosłych)."
        )
    else:
        interp = (
            f"Korzystny profil ryzyka: oszacowane ryzyko jest niższe niż u {100.0 - decade_pct:.1f}% osób "
            f"w Twoim wieku ({decade_label}) i niższe niż u {100.0 - overall_pct:.1f}% dorosłych ogółem."
        )

    return {
        "decade_percentile": round(decade_pct, 1),
        "overall_percentile": round(overall_pct, 1),
        "decade_label": decade_label,
        "decade": decade_code,
        "interpretation": interp,
        "patient_risk": norm_risk,
        "patient_age": float(patient_age),
    }


def plot_population_risk_distribution(
    patient_risk: float,
    baseline_df: Optional[pd.DataFrame] = None,
    patient_age: Optional[float] = None,
    title: str = "Rozkład Ryzyka w Populacji Dorosłych",
) -> go.Figure:
    """Render interactive hybrid KDE distribution with 3 clinical risk zones and patient percentile marker."""
    if baseline_df is None:
        baseline_df = get_population_baseline_risks()

    # Normalize risk
    norm_risk = patient_risk / 100.0 if patient_risk > 1.0 and baseline_df["stroke_risk"].max() <= 1.0 else patient_risk
    norm_risk = float(np.clip(norm_risk, 0.0, 1.0))

    pop_risks = baseline_df["stroke_risk"].values
    kde = gaussian_kde(pop_risks)

    # Dynamic x range spanning adult distribution and patient risk
    x_max = max(0.40, float(np.percentile(pop_risks, 99.8)) * 1.25, norm_risk * 1.20)
    x_max = min(1.0, max(0.35, x_max))
    x_grid = np.linspace(0.0, x_max, 400)
    y_kde = kde(x_grid)

    percentiles = calculate_patient_percentiles(
        patient_risk=norm_risk,
        patient_age=patient_age if patient_age is not None else 50.0,
        baseline_df=baseline_df,
    )

    fig = go.Figure()

    # 3 Background risk zones (<10% green, 10-25% amber, >=25% red)
    fig.add_vrect(
        x0=0.0,
        x1=0.10,
        fillcolor="rgba(22, 163, 74, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Niskie (<10%)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#15803d", family="sans-serif"),
    )
    fig.add_vrect(
        x0=0.10,
        x1=0.25,
        fillcolor="rgba(202, 138, 4, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Umiarkowane (10–25%)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#a16207", family="sans-serif"),
    )
    fig.add_vrect(
        x0=0.25,
        x1=x_max,
        fillcolor="rgba(220, 38, 38, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Wysokie (≥25%)",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#b91c1c", family="sans-serif"),
    )

    # Shaded left-tail area under the curve up to patient's score (ECDF representation)
    tail_mask = x_grid <= norm_risk
    x_tail = list(x_grid[tail_mask])
    y_tail = list(y_kde[tail_mask])
    if not x_tail or x_tail[-1] < norm_risk:
        x_tail.append(norm_risk)
        y_tail.append(float(kde(norm_risk)[0]))

    fig.add_trace(go.Scatter(
        x=x_tail,
        y=y_tail,
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.22)",
        line=dict(color="rgba(37, 99, 235, 0)", width=0),
        mode="lines",
        name=f"Skumulowany odsetek (ECDF: {percentiles['overall_percentile']:.1f}%)",
        hoverinfo="skip",
    ))

    # Overall Adult Population KDE Curve
    fig.add_trace(go.Scatter(
        x=x_grid,
        y=y_kde,
        mode="lines",
        line=dict(color="#1d4ed8", width=2.8),
        name="Populacja dorosłych (N = 4 253)",
        hovertemplate="Ryzyko: %{x:.1%}<br>Gęstość: %{y:.2f}<extra></extra>",
    ))

    # Age Decade KDE Curve if patient_age provided
    max_density = float(np.max(y_kde))
    if patient_age is not None:
        decade_code, decade_label = get_age_decade_label(float(patient_age))
        decade_risks = baseline_df[baseline_df["decade"] == decade_code]["stroke_risk"].values
        if len(decade_risks) >= 15:
            kde_decade = gaussian_kde(decade_risks)
            y_decade = kde_decade(x_grid)
            max_density = max(max_density, float(np.max(y_decade)))
            fig.add_trace(go.Scatter(
                x=x_grid,
                y=y_decade,
                mode="lines",
                line=dict(color="#7c3aed", width=2.2, dash="dash"),
                name=f"Grupa wieku: {decade_label} (N = {len(decade_risks)})",
                hovertemplate=f"Ryzyko: %{{x:.1%}}<br>Gęstość ({decade_label}): %{{y:.2f}}<extra></extra>",
            ))

    # Distinct vertical dashed line for active patient
    fig.add_vline(
        x=norm_risk,
        line_dash="dash",
        line_color="#0f172a",
        line_width=2.5,
    )

    # Distinct vertical marker annotation for active patient's predicted risk
    if patient_age is not None:
        marker_text = (
            f"<b>Wynik pacjenta: {norm_risk * 100:.1f}%</b><br>"
            f"Centyl w dekadzie: <b>{percentiles['decade_percentile']:.1f}%</b><br>"
            f"Centyl ogółem: <b>{percentiles['overall_percentile']:.1f}%</b>"
        )
    else:
        marker_text = (
            f"<b>Wynik pacjenta: {norm_risk * 100:.1f}%</b><br>"
            f"Centyl w populacji: <b>{percentiles['overall_percentile']:.1f}%</b>"
        )

    ax_shift = 55 if norm_risk < (x_max * 0.65) else -55
    fig.add_annotation(
        x=norm_risk,
        y=max_density * 0.78,
        text=marker_text,
        showarrow=True,
        arrowhead=2,
        arrowsize=1.2,
        arrowwidth=1.8,
        arrowcolor="#0f172a",
        ax=ax_shift,
        ay=-45,
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#0f172a",
        borderwidth=1.5,
        borderpad=6,
        font=dict(size=11, color="#0f172a", family="sans-serif"),
    )

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0f172a", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="Szacowane 5-letnie Ryzyko Udaru",
            tickformat=".0%",
            range=[0.0, x_max],
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(
            title="Gęstość Prawdopodobieństwa (KDE)",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=11),
        ),
        margin=dict(l=50, r=30, t=65, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.85)",
        height=390,
    )

    return fig
