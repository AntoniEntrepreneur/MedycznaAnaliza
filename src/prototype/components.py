"""Shared UI components for the Streamlit clinical risk calculator prototype."""

import html
import json
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

VARIANTS = [
    ("A", "Guided Intake & Triage Wizard"),
    ("B", "Bento Clinical Dashboard"),
    ("C", "Comparative Intervention Studio"),
]

VARIANT_KEYS = [k for k, _ in VARIANTS]
VARIANT_MAP = dict(VARIANTS)


def get_risk_tier(prob: float) -> Dict[str, str]:
    """Return risk classification, badge color, and clinical guidance."""
    pct = prob * 100
    if pct < 10.0:
        return {
            "tier": "Low Risk",
            "color": "#16a34a",
            "bg": "#dcfce7",
            "border": "#86efac",
            "recommendation": "Routine preventive care; monitor lifestyle factors at next regular visit.",
        }
    elif pct < 25.0:
        return {
            "tier": "Moderate Risk",
            "color": "#ca8a04",
            "bg": "#fef9c3",
            "border": "#fde047",
            "recommendation": "Schedule 6-month clinical review. Initiate targeted lifestyle interventions.",
        }
    elif pct < 50.0:
        return {
            "tier": "High Risk",
            "color": "#ea580c",
            "bg": "#ffedd5",
            "border": "#fdba74",
            "recommendation": "Expedited clinical evaluation. Consider pharmacotherapy for blood pressure and lipids.",
        }
    else:
        return {
            "tier": "Critical Risk",
            "color": "#dc2626",
            "bg": "#fee2e2",
            "border": "#fca5a5",
            "recommendation": "Urgent neurological and cardiovascular workup. Immediate comprehensive risk reduction protocol.",
        }


def render_risk_gauge(prob: float, title: str = "5-Year Stroke Risk Estimate"):
    """Render an aesthetic semicircular clinical gauge meter."""
    pct = np.clip(prob * 100, 0, 100)
    tier_info = get_risk_tier(prob)

    fig, ax = plt.subplots(figsize=(6, 3.2), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Arc angle range: np.pi (left / 0%) to 0 (right / 100%)
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
        ax.fill_between(theta, 0.65, 0.95, color=color, alpha=0.85, zorder=2)

    # Needle calculation
    needle_theta = np.pi - (pct / 100.0) * np.pi
    ax.annotate(
        "",
        xy=(needle_theta, 0.9),
        xytext=(0, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="#1e293b",
            lw=3.5,
            mutation_scale=18,
        ),
        zorder=5,
    )

    # Center pivot circle
    ax.plot(0, 0, marker="o", markersize=14, color="#1e293b", zorder=6)
    ax.plot(0, 0, marker="o", markersize=8, color="#ffffff", zorder=7)

    # Dial limits
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
        <div style="text-align: center; margin-top: -15px; margin-bottom: 12px;">
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
            <p style="font-size: 0.82rem; color: #64748b; margin-top: 6px;">{tier_info['recommendation']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_shap_waterfall(contributions_df: pd.DataFrame, max_features: int = 7):
    """Render a clean horizontal SHAP factor contribution chart."""
    top_df = contributions_df.head(max_features).iloc[::-1]

    fig, ax = plt.subplots(figsize=(6, 3.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    labels = [
        f"{row.feature.replace('_', ' ').title()} = {row.value}"
        for _, row in top_df.iterrows()
    ]
    vals = top_df["shap_value"].values
    colors = ["#ef4444" if v >= 0 else "#22c55e" for v in vals]

    bars = ax.barh(labels, vals, color=colors, height=0.55, alpha=0.9)
    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1.2, alpha=0.7)

    # Value labels on bars
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
            color="#334155",
        )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.set_xlabel("SHAP Impact on Stroke Odds (log-odds)", fontsize=9, color="#475569")
    plt.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_state_inspector(state: Dict[str, Any]):
    """Surface full active state to satisfy prototype rule 5."""
    with st.expander("🔍 Surface State (Prototype Data Inspector)", expanded=False):
        st.caption("Active in-memory patient payload evaluated by model:")
        st.code(json.dumps(state, indent=2), language="json")


def render_prototype_switcher(current_variant: str):
    """Render the floating switcher bar adhering to the prototype skill specification."""
    if current_variant not in VARIANT_KEYS:
        current_variant = "A"

    curr_idx = VARIANT_KEYS.index(current_variant)
    prev_variant = VARIANT_KEYS[(curr_idx - 1) % len(VARIANT_KEYS)]
    next_variant = VARIANT_KEYS[(curr_idx + 1) % len(VARIANT_KEYS)]
    curr_label = f"{current_variant} ({VARIANT_MAP[current_variant]})"

    # Inject floating CSS styling
    st.markdown(
        """
        <style>
        .prototype-floating-bar {
            position: fixed;
            bottom: 24px;
            left: 50%;
            transform: translateX(-50%);
            z-index: 999999;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            color: #ffffff;
            padding: 8px 18px;
            border-radius: 9999px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
            display: flex;
            align-items: center;
            gap: 16px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            font-size: 13px;
        }
        .prototype-pill-tag {
            background: #2563eb;
            color: #ffffff;
            font-size: 10px;
            font-weight: 800;
            padding: 2px 7px;
            border-radius: 6px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Bottom sticky switcher using columns inside a styled container
    st.write("---")
    cols = st.columns([1, 1, 4, 1, 1])

    with cols[1]:
        if st.button(f"◀ Prev ({prev_variant})", key="proto_nav_prev", use_container_width=True):
            st.query_params["variant"] = prev_variant
            st.rerun()

    with cols[2]:
        st.markdown(
            f"""
            <div style="text-align: center; padding-top: 6px;">
                <span class="prototype-pill-tag">Prototype Switcher</span>
                <span style="font-weight: 700; font-size: 14px; margin-left: 8px; color: #1e293b;">
                    Variant {curr_label}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:
        if st.button(f"Next ({next_variant}) ▶", key="proto_nav_next", use_container_width=True):
            st.query_params["variant"] = next_variant
            st.rerun()
