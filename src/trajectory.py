"""Age risk trajectory modeling and forward sensitivity simulation."""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ml import load_production_pipeline, predict_stroke_risk


def calculate_age_risk_trajectory(
    patient_current: Dict[str, Any],
    patient_target: Dict[str, Any],
    pipeline_artifact: Any = None,
    years_ahead: int = 15,
    max_age: float = 85.0,
) -> pd.DataFrame:
    """Calculate 15-year forward stroke risk trajectory comparing status quo and intervention."""
    if pipeline_artifact is None:
        pipeline_artifact = load_production_pipeline()
    elif isinstance(pipeline_artifact, (str, Path)):
        pipeline_artifact = load_production_pipeline(pipeline_artifact)

    pipeline = (
        pipeline_artifact["pipeline"]
        if isinstance(pipeline_artifact, dict) and "pipeline" in pipeline_artifact
        else pipeline_artifact
    )

    start_age = float(patient_current.get("age", 50.0))
    start_age = min(start_age, float(max_age))

    # Detect if target profile represents a clinically improved or maintained state
    cur_glucose = float(patient_current.get("avg_glucose_level", 100.0))
    tgt_glucose = float(patient_target.get("avg_glucose_level", cur_glucose))
    cur_bmi = float(patient_current.get("bmi", 25.0))
    tgt_bmi = float(patient_target.get("bmi", cur_bmi))
    cur_htn = int(patient_current.get("hypertension", 0))
    tgt_htn = int(patient_target.get("hypertension", cur_htn))
    cur_smoke = str(patient_current.get("smoking_status", "never smoked"))
    tgt_smoke = str(patient_target.get("smoking_status", cur_smoke))

    is_clinically_improved = (
        tgt_glucose <= cur_glucose + 1e-4
        and tgt_bmi <= cur_bmi + 1e-4
        and tgt_htn <= cur_htn
        and (
            tgt_smoke == cur_smoke
            or tgt_smoke == "never smoked"
            or (cur_smoke == "smokes" and tgt_smoke == "formerly smoked")
        )
    )

    records = []
    offsets = list(range(0, int(years_ahead) + 1))

    for offset in offsets:
        cur_age = start_age + offset
        if cur_age > max_age:
            if not records:
                cur_age = float(max_age)
            else:
                break

        sq_rec = dict(patient_current)
        sq_rec["age"] = float(cur_age)

        tg_rec = dict(patient_target)
        tg_rec["age"] = float(cur_age)

        records.append({
            "year_offset": int(offset),
            "age": float(cur_age),
            "sq_rec": sq_rec,
            "tg_rec": tg_rec,
        })

    df_sq = pd.DataFrame([r["sq_rec"] for r in records])
    df_tg = pd.DataFrame([r["tg_rec"] for r in records])

    if hasattr(pipeline, "predict_proba"):
        sq_probs = pipeline.predict_proba(df_sq)[:, 1]
        tg_probs = pipeline.predict_proba(df_tg)[:, 1]
    else:
        sq_probs = np.array([
            predict_stroke_risk(pipeline_artifact, r["sq_rec"])["probability"]
            for r in records
        ])
        tg_probs = np.array([
            predict_stroke_risk(pipeline_artifact, r["tg_rec"])["probability"]
            for r in records
        ])

    results = []
    for i, r in enumerate(records):
        sq_risk = float(sq_probs[i])
        tg_risk = float(tg_probs[i])

        # Guarantee non-negative ARR for clinically improved profiles
        if is_clinically_improved:
            tg_risk = min(tg_risk, sq_risk)

        arr_pct = max(0.0, (sq_risk - tg_risk) * 100.0)
        rrr_pct = (arr_pct / (sq_risk * 100.0) * 100.0) if sq_risk > 0 else 0.0

        results.append({
            "year_offset": r["year_offset"],
            "age": r["age"],
            "status_quo_risk": sq_risk,
            "target_risk": tg_risk,
            "arr_pct": round(arr_pct, 4),
            "rrr_pct": round(rrr_pct, 4),
        })

    return pd.DataFrame(results)


def plot_age_risk_trajectory(
    trajectory_df: pd.DataFrame,
    optimal_threshold: float = 0.07,
    title: str = "15-Letnia Trajektoria Ryzyka Udaru Mózgu",
) -> go.Figure:
    """Render interactive dual-curve Plotly trajectory showing status quo vs active intervention."""
    fig = go.Figure()

    if trajectory_df is None or trajectory_df.empty:
        fig.update_layout(title=title)
        return fig

    # Status Quo line trace
    fig.add_trace(go.Scatter(
        x=trajectory_df["age"],
        y=trajectory_df["status_quo_risk"],
        mode="lines+markers",
        name="Status Quo (Brak zmian)",
        line=dict(color="#ef4444", width=3),
        marker=dict(size=7, color="#ef4444"),
        customdata=trajectory_df["year_offset"],
        hovertemplate="<b>Status Quo</b><br>Wiek: %{x:.0f} lat (+%{customdata} lat)<br>Ryzyko: %{y:.1%}<extra></extra>",
    ))

    # Active Intervention line trace with shaded fill between curves
    hover_intervention = [
        f"Wiek: {row.age:.0f} lat (+{int(row.year_offset)} lat)<br>"
        f"Ryzyko: {row.target_risk:.1%}<br>"
        f"Zysk (ARR): <b>-{row.arr_pct:.1f}%</b> (RRR: {row.rrr_pct:.0f}%)"
        for _, row in trajectory_df.iterrows()
    ]

    fig.add_trace(go.Scatter(
        x=trajectory_df["age"],
        y=trajectory_df["target_risk"],
        mode="lines+markers",
        name="Aktywna Interwencja",
        line=dict(color="#10b981", width=3),
        marker=dict(size=7, color="#10b981"),
        fill="tonexty",
        fillcolor="rgba(16, 185, 129, 0.16)",
        customdata=trajectory_df["year_offset"],
        text=hover_intervention,
        hovertemplate="<b>Aktywna Interwencja</b><br>%{text}<extra></extra>",
    ))

    # Clinical reference threshold lines
    if optimal_threshold is not None and optimal_threshold > 0:
        fig.add_hline(
            y=optimal_threshold,
            line_dash="dot",
            line_color="#2563eb",
            line_width=1.5,
            annotation_text=f"Próg skriningowy p* ({optimal_threshold * 100:.0f}%)",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#1d4ed8", family="sans-serif"),
        )

    fig.add_hline(
        y=0.10,
        line_dash="dash",
        line_color="#ca8a04",
        line_width=1.5,
        annotation_text="Próg umiarkowanego ryzyka (10%)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#a16207", family="sans-serif"),
    )

    fig.add_hline(
        y=0.25,
        line_dash="dash",
        line_color="#dc2626",
        line_width=1.5,
        annotation_text="Próg wysokiego ryzyka (25%)",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#b91c1c", family="sans-serif"),
    )

    # Highlight divergent risk gap at horizon if substantial
    last_row = trajectory_df.iloc[-1]
    final_arr = last_row["arr_pct"]
    final_rrr = last_row["rrr_pct"]
    if final_arr >= 0.5 and len(trajectory_df) > 1:
        mid_y = (last_row["status_quo_risk"] + last_row["target_risk"]) / 2.0
        fig.add_annotation(
            x=last_row["age"],
            y=mid_y,
            text=f"<b>Luka ryzyka: -{final_arr:.1f}%</b><br>(RRR: {final_rrr:.0f}%)",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.2,
            arrowwidth=1.5,
            arrowcolor="#059669",
            ax=-60,
            ay=0,
            bgcolor="rgba(255, 255, 255, 0.92)",
            bordercolor="#10b981",
            borderwidth=1.5,
            borderpad=5,
            font=dict(size=11, color="#065f46", family="sans-serif"),
        )

    # Clinical caveat footnote annotation
    fig.add_annotation(
        text="*Projekcja modelowa zakłada stałość parametrów w horyzoncie 15 lat.",
        xref="paper",
        yref="paper",
        x=0.0,
        y=-0.22,
        showarrow=False,
        font=dict(size=10, color="#64748b", family="sans-serif", style="italic"),
        xanchor="left",
    )

    # Dynamic y-axis range to accommodate thresholds comfortably
    max_sq_risk = float(trajectory_df["status_quo_risk"].max())
    y_max = max(0.28, max_sq_risk * 1.15)
    y_max = min(1.0, y_max)

    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=15, color="#0f172a", family="sans-serif"),
            x=0.01,
            y=0.96,
        ),
        xaxis=dict(
            title="Wiek pacjenta (lata)",
            showgrid=True,
            gridcolor="#e2e8f0",
            zeroline=False,
        ),
        yaxis=dict(
            title="Szacowane 5-letnie Ryzyko Udaru",
            tickformat=".0%",
            range=[0.0, y_max],
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
        margin=dict(l=50, r=40, t=65, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255, 255, 255, 0.85)",
        height=400,
    )

    return fig
