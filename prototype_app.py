# Three variants of the Streamlit clinical risk calculator UI, switchable via ?variant=, on the prototype app route.
"""
Streamlit Clinical Risk Calculator UI Prototype (Issue #7).

Question:
How should the Streamlit interface (sidebar clinical inputs, risk gauge meter,
SHAP plot placement, and patient scenario presets) look and behave to provide
an intuitive workflow for clinical decision support?

Variants:
- Variant A: Guided Intake & Clinical Triage Wizard (progressive disclosure)
- Variant B: Bento Clinical Dashboard (executive EHR workstation with real-time sensitivity)
- Variant C: Comparative Intervention Studio (side-by-side baseline vs goal for shared decision making)
"""

import streamlit as st
from src.prototype.components import render_prototype_switcher, VARIANT_KEYS, VARIANT_MAP
from src.prototype.variant_a import render_variant_a
from src.prototype.variant_b import render_variant_b
from src.prototype.variant_c import render_variant_c

# Configure page
st.set_page_config(
    page_title="Stroke Risk Calculator Prototype",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 5rem;
    }
    .reportview-container .main .block-container {
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    """Main routing and switcher entrypoint for the UI prototype."""
    # Query parameter resolution
    query_variant = st.query_params.get("variant", "A")
    if isinstance(query_variant, list):
        query_variant = query_variant[0] if query_variant else "A"

    if query_variant not in VARIANT_KEYS:
        query_variant = "A"
        st.query_params["variant"] = "A"

    # Top banner indicating prototype mode and variant selection
    top_col1, top_col2 = st.columns([3, 1])
    with top_col1:
        st.caption(f"Prototype Mode • Evaluating UI Design Alternatives for Issue #7")
    with top_col2:
        # Quick dropdown fallback in addition to floating switcher
        direct_selection = st.selectbox(
            "Switch Variant",
            VARIANT_KEYS,
            index=VARIANT_KEYS.index(query_variant),
            format_func=lambda k: f"{k}: {VARIANT_MAP[k]}",
            label_visibility="collapsed",
            key="direct_var_select",
        )
        if direct_selection != query_variant:
            st.query_params["variant"] = direct_selection
            st.rerun()

    # Route to appropriate variant
    if query_variant == "A":
        render_variant_a()
    elif query_variant == "B":
        render_variant_b()
    elif query_variant == "C":
        render_variant_c()

    # Sticky floating navigation switcher at bottom center
    render_prototype_switcher(query_variant)


if __name__ == "__main__":
    main()
