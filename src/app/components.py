"""Reusable visual components for clinical risk dashboard."""

from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.visualization import create_risk_gauge


def get_risk_tier(prob: float) -> Dict[str, str]:
    """Return risk classification tier, styling colors, and clinical guidance."""
    pct = prob * 100 if prob <= 1.0 else prob
    if pct < 10.0:
        return {
            "tier": "Low Risk",
            "tier_pl": "Niskie",
            "color": "#16a34a",
            "bg": "#dcfce7",
            "border": "#86efac",
            "cutoff_range": "< 10%",
            "recommendation": "Rutynowa profilaktyka i nadzór stylu życia. Kolejna ocena ryzyka podczas planowej wizyty rocznej.",
        }
    elif pct < 25.0:
        return {
            "tier": "Moderate Risk",
            "tier_pl": "Umiarkowane",
            "color": "#ca8a04",
            "bg": "#fef9c3",
            "border": "#fde047",
            "cutoff_range": "10% - 25%",
            "recommendation": "Zalecana kontrola kliniczna za 6 miesięcy. Wdrożenie niefarmakologicznej modyfikacji czynników ryzyka (dieta, aktywność fizyczna, monitorowanie ciśnienia tętniczego i glikemii).",
        }
    else:
        return {
            "tier": "High Risk",
            "tier_pl": "Wysokie",
            "color": "#dc2626",
            "bg": "#fee2e2",
            "border": "#fca5a5",
            "cutoff_range": "≥ 25%",
            "recommendation": "Wskazana pilna konsultacja specjalistyczna (neurologiczna/kardiologiczna). Rozważenie agresywnej farmakoterapii przeciwnadciśnieniowej i hipolipemizującej.",
        }


def render_risk_gauge(
    prob: float,
    title: str = "5-Year Stroke Risk Estimate",
    optimal_threshold: float = 0.07,
    default_threshold: float = 0.50,
):
    """Render a semicircular polar gauge meter showing colored risk zones and dual thresholds."""
    fig = create_risk_gauge(
        risk_prob=prob,
        optimal_threshold=optimal_threshold,
        default_threshold=default_threshold,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_shap_waterfall(contributions_df: pd.DataFrame, max_features: int = 7):
    """Render horizontal SHAP attribution chart showing directional risk drivers."""
    top_df = contributions_df.head(max_features).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("#caf0fe")
    ax.set_facecolor("#caf0fe")

    labels = [
        f"{row.feature.replace('_', ' ').title()} = {row.value}"
        for _, row in top_df.iterrows()
    ]
    vals = top_df["shap_value"].values
    colors = ["#ef4444" if v >= 0 else "#22c55e" for v in vals]

    bars = ax.barh(labels, vals, color=colors, height=0.55, alpha=0.9)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)

    for bar, val in zip(bars, vals):
        sign = "+" if val >= 0 else ""
        offset = 0.04 if val >= 0 else -0.04
        ha = "left" if val >= 0 else "right"
        ax.text(
            val + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{sign}{val:.2f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94a3b8")
    ax.spines["bottom"].set_color("#94a3b8")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("SHAP Impact on Stroke Log-Odds", fontsize=9, color="#334155")
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
