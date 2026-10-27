"""Reusable visual components for clinical risk dashboard."""

from typing import Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


def get_risk_tier(prob: float) -> Dict[str, str]:
    """Return risk classification tier, styling colors, and clinical guidance."""
    pct = prob * 100
    if pct < 10.0:
        return {
            "tier": "Low Risk",
            "color": "#16a34a",
            "bg": "#dcfce7",
            "border": "#86efac",
            "recommendation": "Routine preventive care; monitor lifestyle factors at next regular annual visit.",
        }
    elif pct < 25.0:
        return {
            "tier": "Moderate Risk",
            "color": "#ca8a04",
            "bg": "#fef9c3",
            "border": "#fde047",
            "recommendation": "Schedule 6-month clinical review. Initiate targeted diet, exercise, and lipid monitoring.",
        }
    elif pct < 50.0:
        return {
            "tier": "High Risk",
            "color": "#ea580c",
            "bg": "#ffedd5",
            "border": "#fdba74",
            "recommendation": "Expedited clinical evaluation. Consider aggressive blood pressure and glucose pharmacotherapy.",
        }
    else:
        return {
            "tier": "Critical Risk",
            "color": "#dc2626",
            "bg": "#fee2e2",
            "border": "#fca5a5",
            "recommendation": "Urgent neurological and cardiovascular assessment. Immediate comprehensive risk reduction protocol.",
        }


def render_risk_gauge(prob: float, title: str = "5-Year Stroke Risk Estimate"):
    """Render a semicircular polar gauge meter showing colored risk zones and indicator needle."""
    pct = np.clip(prob * 100, 0, 100)
    tier_info = get_risk_tier(prob)

    fig, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#caf0fe")
    ax.set_facecolor("#caf0fe")

    categories = [
        ("Low", 0, 10, "#22c55e"),
        ("Moderate", 10, 25, "#eab308"),
        ("High", 25, 50, "#f97316"),
        ("Critical", 50, 100, "#ef4444"),
    ]

    for _, start_pct, end_pct, color in categories:
        theta_start = np.pi - (start_pct / 100.0) * np.pi
        theta_end = np.pi - (end_pct / 100.0) * np.pi
        theta = np.linspace(theta_start, theta_end, 50)
        ax.fill_between(theta, 0.65, 0.95, color=color, alpha=0.88, zorder=2)

    needle_theta = np.pi - (pct / 100.0) * np.pi
    ax.annotate(
        "",
        xy=(needle_theta, 0.9),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#0f172a",
            lw=3.5,
            mutation_scale=18,
        ),
        zorder=5,
    )

    ax.plot(0, 0, marker="o", markersize=14, color="#0f172a", zorder=6)
    ax.plot(0, 0, marker="o", markersize=8, color="#ffffff", zorder=7)

    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: -12px; margin-bottom: 8px;">
            <span style="font-size: 2.2rem; font-weight: 800; color: {tier_info['color']};">
                {pct:.1f}%
            </span>
            <div style="margin-top: 4px;">
                <span style="display: inline-block; padding: 4px 14px; border-radius: 9999px; 
                             background-color: {tier_info['bg']}; color: {tier_info['color']}; 
                             border: 1px solid {tier_info['border']}; font-weight: 700; font-size: 0.9rem;">
                    {tier_info['tier']}
                </span>
            </div>
            <p style="font-size: 0.84rem; color: #475569; margin-top: 6px; font-weight: 500;">
                {tier_info['recommendation']}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
