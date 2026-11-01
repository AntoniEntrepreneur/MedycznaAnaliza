"""Data visualization components for stroke risk analysis and biostatistics."""

from pathlib import Path
from typing import Optional, Union
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _format_p_val_display(p: float) -> str:
    """Format exact p-value for display in clinical forest plot table."""
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def plot_forest_matplotlib(
    forest_df: pd.DataFrame,
    title: str = "Wielowymiarowa Regresja Logistyczna Czynników Ryzyka Udaru",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Render publication-grade static Forest Plot of adjusted Odds Ratios with data table.

    Displays covariate effect sizes on a logarithmic axis with a dashed reference line
    at OR = 1.0, sorted descending by aOR magnitude. Covariates are colored with two-tone
    clinical styling (#1E3A8A navy for p < 0.05, #94A3B8 slate for non-significant).
    A right-hand aligned table presents Polish covariate labels, aOR [95% CI], and p-values.

    Parameters
    ----------
    forest_df : pd.DataFrame
        DataFrame containing multivariable logistic regression results with columns:
        variable, label_pl, or, ci_lower, ci_upper, p_value, is_significant.
    title : str, default 'Wielowymiarowa Regresja Logistyczna Czynników Ryzyka Udaru'
        Plot title.
    save_path : Optional[Union[str, Path]], default None
        File path to save the generated figure.

    Returns
    -------
    matplotlib.figure.Figure
        Rendered figure ready for display or embedding.
    """
    df = forest_df.sort_values(by="or", ascending=False).reset_index(drop=True)
    n = len(df)
    y_pos = np.arange(n)[::-1]

    fig = plt.figure(figsize=(14, max(5.5, n * 0.55 + 1.2)), dpi=150)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.15], wspace=0.16)
    ax_plot = fig.add_subplot(gs[0, 0])
    ax_table = fig.add_subplot(gs[0, 1], sharey=ax_plot)

    # Alternating row shading across both panels
    for i in range(n):
        if i % 2 == 1:
            y = y_pos[i]
            ax_plot.axhspan(y - 0.45, y + 0.45, color="#F8FAFC", zorder=0)
            ax_table.axhspan(y - 0.45, y + 0.45, color="#F8FAFC", zorder=0)

    # Plot forest markers and 95% Wald CI error bars
    for i, (_, row) in enumerate(df.iterrows()):
        y = y_pos[i]
        color = "#1E3A8A" if row["is_significant"] else "#94A3B8"
        ax_plot.plot(
            [row["ci_lower"], row["ci_upper"]],
            [y, y],
            color=color,
            linewidth=2.2,
            zorder=3,
        )
        ax_plot.scatter(
            row["or"],
            y,
            color=color,
            s=75,
            marker="s",
            zorder=4,
            edgecolors=color,
        )

    # Dashed reference line at OR = 1.0 (no effect)
    ax_plot.axvline(
        1.0, color="#64748B", linestyle="--", linewidth=1.2, zorder=2
    )

    # Logarithmic horizontal axis with clinical tick intervals
    ax_plot.set_xscale("log")
    ax_plot.set_xlim(0.18, 5.5)
    ax_plot.set_xticks([0.2, 0.5, 1.0, 2.0, 5.0])
    ax_plot.set_xticklabels(["0.2", "0.5", "1.0", "2.0", "5.0"], fontsize=10)
    ax_plot.set_yticks(y_pos)
    ax_plot.set_yticklabels(df["label_pl"], fontsize=10.5)
    ax_plot.set_ylim(-0.6, n - 0.3)
    ax_plot.set_xlabel(
        "Skorygowany Wskaźnik Szans (aOR, skala logarytmiczna)",
        fontsize=10.5,
        labelpad=8,
    )

    # Styling axes and grid
    ax_plot.spines[["top", "right"]].set_visible(False)
    ax_plot.spines["left"].set_color("#CBD5E1")
    ax_plot.spines["bottom"].set_color("#CBD5E1")
    ax_plot.grid(axis="x", linestyle=":", color="#E2E8F0", alpha=0.8, zorder=1)

    # Two-tone significance legend
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Istotny statystycznie (p < 0.05)",
            markerfacecolor="#1E3A8A",
            markersize=8,
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            label="Nieistotny (p ≥ 0.05)",
            markerfacecolor="#94A3B8",
            markersize=8,
        ),
    ]
    ax_plot.legend(
        handles=legend_elements,
        loc="lower left",
        frameon=True,
        facecolor="white",
        edgecolor="#E2E8F0",
        fontsize=9.5,
    )

    # Right-hand aligned data table
    ax_table.axis("off")
    ax_table.set_xlim(0, 1)

    header_y = n - 0.4
    ax_table.text(
        0.04,
        header_y,
        "Zmienna",
        fontweight="bold",
        fontsize=10.5,
        color="#0F172A",
        va="center",
    )
    ax_table.text(
        0.53,
        header_y,
        "aOR [95% CI]",
        fontweight="bold",
        fontsize=10.5,
        color="#0F172A",
        va="center",
    )
    ax_table.text(
        0.86,
        header_y,
        "Wartość p",
        fontweight="bold",
        fontsize=10.5,
        color="#0F172A",
        va="center",
    )
    ax_table.axhline(header_y - 0.35, color="#94A3B8", linewidth=1.2)

    for i, (_, row) in enumerate(df.iterrows()):
        y = y_pos[i]
        text_color = "#1E3A8A" if row["is_significant"] else "#475569"
        weight = "bold" if row["is_significant"] else "normal"
        p_str = _format_p_val_display(row["p_value"])
        ci_str = f"{row['or']:.2f} [{row['ci_lower']:.2f}, {row['ci_upper']:.2f}]"

        ax_table.text(
            0.04,
            y,
            row["label_pl"],
            fontsize=10,
            color=text_color,
            fontweight=weight,
            va="center",
        )
        ax_table.text(
            0.53,
            y,
            ci_str,
            fontsize=10,
            color=text_color,
            fontweight=weight,
            va="center",
        )
        ax_table.text(
            0.86,
            y,
            p_str,
            fontsize=10,
            color=text_color,
            fontweight=weight,
            va="center",
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    if save_path:
        out_p = Path(save_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, bbox_inches="tight")

    return fig


def plot_forest_plotly(
    forest_df: pd.DataFrame,
    title: str = "Wielowymiarowa Regresja Logistyczna Czynników Ryzyka Udaru",
) -> go.Figure:
    """Render interactive dual-engine Plotly Forest Plot with embedded data table.

    Displays covariate adjusted Odds Ratios and 95% Wald confidence intervals on a
    logarithmic horizontal axis (ticks at 0.2, 0.5, 1.0, 2.0, 5.0) with reference
    line at OR = 1.0, sorted descending by aOR. Features two-tone clinical styling
    (#1E3A8A navy for p < 0.05, #94A3B8 slate for non-significant) and interactive hover.

    Parameters
    ----------
    forest_df : pd.DataFrame
        DataFrame containing multivariable logistic regression results with columns:
        variable, label_pl, or, ci_lower, ci_upper, p_value, is_significant.
    title : str, default 'Wielowymiarowa Regresja Logistyczna Czynników Ryzyka Udaru'
        Plot title.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive Plotly figure.
    """
    df = forest_df.sort_values(by="or", ascending=False).reset_index(drop=True)

    fig = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.58, 0.42],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        subplot_titles=[
            "Wskaźnik Szans (OR, 95% CI)",
            "Tabela Parametrów Modelu",
        ],
        horizontal_spacing=0.08,
    )

    # Two traces for two-tone significance and interactive legend
    for sig, color, name in [
        (True, "#1E3A8A", "Istotny statystycznie (p < 0.05)"),
        (False, "#94A3B8", "Nieistotny (p ≥ 0.05)"),
    ]:
        sub = df[df["is_significant"] == sig]
        if len(sub) == 0:
            continue

        fig.add_trace(
            go.Scatter(
                x=sub["or"],
                y=sub["label_pl"],
                mode="markers",
                marker=dict(size=9, color=color, symbol="square"),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=sub["ci_upper"] - sub["or"],
                    arrayminus=sub["or"] - sub["ci_lower"],
                    color=color,
                    thickness=2,
                    width=4,
                ),
                name=name,
                customdata=np.stack(
                    [sub["ci_lower"], sub["ci_upper"], sub["p_value"]],
                    axis=-1,
                ),
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    "Skorygowany OR: %{x:.2f}<br>"
                    "95% CI: [%{customdata[0]:.2f}, %{customdata[1]:.2f}]<br>"
                    "Wartość p: %{customdata[2]:.4f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    # Dashed reference line at OR = 1.0
    fig.add_vline(
        x=1.0,
        line_dash="dash",
        line_color="#64748B",
        line_width=1.2,
        row=1,
        col=1,
    )

    # Right-hand aligned data table
    table_text_colors = [
        "#1E3A8A" if s else "#475569" for s in df["is_significant"]
    ]
    ci_formatted = [
        f"{r['or']:.2f} [{r['ci_lower']:.2f}, {r['ci_upper']:.2f}]"
        for _, r in df.iterrows()
    ]
    p_formatted = [_format_p_val_display(p) for p in df["p_value"]]

    cell_fill = [
        "#FFFFFF" if i % 2 == 0 else "#F8FAFC" for i in range(len(df))
    ]

    fig.add_trace(
        go.Table(
            header=dict(
                values=[
                    "<b>Zmienna</b>",
                    "<b>aOR [95% CI]</b>",
                    "<b>Wartość p</b>",
                ],
                fill_color="#F1F5F9",
                align=["left", "center", "right"],
                font=dict(size=12, color="#0F172A"),
                line_color="#CBD5E1",
                height=32,
            ),
            cells=dict(
                values=[df["label_pl"].tolist(), ci_formatted, p_formatted],
                fill_color=[cell_fill, cell_fill, cell_fill],
                align=["left", "center", "right"],
                font=dict(
                    size=11,
                    color=[
                        table_text_colors,
                        table_text_colors,
                        table_text_colors,
                    ],
                ),
                line_color="#E2E8F0",
                height=28,
            ),
        ),
        row=1,
        col=2,
    )

    # Configure axes: log scale, specific ticks, descending aOR y-axis
    fig.update_xaxes(
        type="log",
        tickvals=[0.2, 0.5, 1.0, 2.0, 5.0],
        ticktext=["0.2", "0.5", "1.0", "2.0", "5.0"],
        range=[float(np.log10(0.18)), float(np.log10(5.5))],
        gridcolor="#E2E8F0",
        zeroline=False,
        row=1,
        col=1,
    )

    # Plotly categorical axis displays first element at the bottom;
    # reversed categoryarray places the highest aOR at the top.
    fig.update_yaxes(
        categoryorder="array",
        categoryarray=df["label_pl"].iloc[::-1].tolist(),
        gridcolor="#F1F5F9",
        row=1,
        col=1,
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#0F172A")),
        template="plotly_white",
        height=max(480, len(df) * 38 + 140),
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=10.5),
        ),
    )

    return fig
