"""Biostatistical hypothesis testing and descriptive Table 1 generator."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

DEFAULT_CONTINUOUS_VARS = ["age", "avg_glucose_level", "bmi"]
DEFAULT_CATEGORICAL_VARS = [
    "hypertension",
    "heart_disease",
    "gender",
    "smoking_status",
    "ever_married",
    "work_type",
    "Residence_type",
]


def _format_p_value(p: Optional[float]) -> str:
    """Format p-value with standard medical publication threshold (<0.001)."""
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def generate_table_one(
    df: pd.DataFrame,
    stratify_by: Optional[str] = "stroke",
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """Generate baseline Table 1 with descriptive metrics and hypothesis tests.

    For continuous variables:
      - Mean +/- SD
      - Median [IQR: Q25 - Q75]
      - Hypothesis testing: Student's or Welch's t-test (selected via Levene's test, alpha=0.05)

    For categorical variables:
      - Counts and percentages: n (%)
      - Hypothesis testing: Pearson chi-square test (alpha=0.05)

    Parameters
    ----------
    df : pd.DataFrame
        Cohort dataset to summarize.
    stratify_by : Optional[str], default 'stroke'
        Binary stratification column (e.g. 'stroke'). If None, overall stats only.
    continuous_vars : Optional[List[str]]
        List of continuous numeric features. Defaults to ['age', 'avg_glucose_level', 'bmi'].
    categorical_vars : Optional[List[str]]
        List of categorical features. Defaults to clinical risk indicators.
    decimals : int, default 2
        Decimal places for continuous metrics.

    Returns
    -------
    pd.DataFrame
        Formatted descriptive Table 1 ready for publication or presentation.
    """
    if continuous_vars is None:
        cont_vars = [c for c in DEFAULT_CONTINUOUS_VARS if c in df.columns]
    else:
        cont_vars = [c for c in continuous_vars if c in df.columns]

    if categorical_vars is None:
        cat_vars = [c for c in DEFAULT_CATEGORICAL_VARS if c in df.columns]
    else:
        cat_vars = [c for c in categorical_vars if c in df.columns]

    total_n = len(df)
    col_overall = f"Overall (N={total_n:,})"

    is_stratified = stratify_by is not None and stratify_by in df.columns
    if is_stratified:
        strata_vals = sorted(df[stratify_by].dropna().unique())
        if len(strata_vals) == 2 and set(strata_vals) == {0, 1} and stratify_by == "stroke":
            col_s0 = f"No Stroke (N={(df[stratify_by] == 0).sum():,})"
            col_s1 = f"Stroke (N={(df[stratify_by] == 1).sum():,})"
        else:
            col_s0 = f"{stratify_by}={strata_vals[0]} (N={(df[stratify_by] == strata_vals[0]).sum():,})"
            col_s1 = f"{stratify_by}={strata_vals[1]} (N={(df[stratify_by] == strata_vals[1]).sum():,})" if len(strata_vals) > 1 else "N/A"
    else:
        strata_vals = []
        col_s0 = None
        col_s1 = None

    rows: List[Dict[str, Any]] = []

    # Process Continuous Variables
    for var in cont_vars:
        s_all = df[var].dropna()
        if len(s_all) == 0:
            continue

        mean_all = s_all.mean()
        std_all = s_all.std()
        med_all = s_all.median()
        q25_all, q75_all = np.percentile(s_all, 25), np.percentile(s_all, 75)

        p_str = ""
        test_name = ""

        if is_stratified and len(strata_vals) == 2:
            s0 = df[df[stratify_by] == strata_vals[0]][var].dropna()
            s1 = df[df[stratify_by] == strata_vals[1]][var].dropna()

            mean_s0 = s0.mean() if len(s0) > 0 else np.nan
            std_s0 = s0.std() if len(s0) > 0 else np.nan
            med_s0 = s0.median() if len(s0) > 0 else np.nan
            q25_s0, q75_s0 = (np.percentile(s0, 25), np.percentile(s0, 75)) if len(s0) > 0 else (np.nan, np.nan)

            mean_s1 = s1.mean() if len(s1) > 0 else np.nan
            std_s1 = s1.std() if len(s1) > 0 else np.nan
            med_s1 = s1.median() if len(s1) > 0 else np.nan
            q25_s1, q75_s1 = (np.percentile(s1, 25), np.percentile(s1, 75)) if len(s1) > 0 else (np.nan, np.nan)

            # Hypothesis testing
            if len(s0) >= 2 and len(s1) >= 2 and s0.nunique() > 1 and s1.nunique() > 1:
                try:
                    _, lev_p = stats.levene(s0, s1)
                    equal_var = bool(lev_p >= 0.05)
                    _, tt_p = stats.ttest_ind(s0, s1, equal_var=equal_var)
                    test_name = "Student's t-test" if equal_var else "Welch's t-test"
                    p_str = _format_p_value(tt_p)
                except Exception:
                    p_str = "N/A"
                    test_name = "N/A"
            else:
                p_str = "N/A"
                test_name = "N/A"

            mean_s0_str = (
                f"{mean_s0:.{decimals}f} ± {std_s0:.{decimals}f}"
                if not np.isnan(std_s0)
                else (f"{mean_s0:.{decimals}f}" if not np.isnan(mean_s0) else "N/A")
            )
            mean_s1_str = (
                f"{mean_s1:.{decimals}f} ± {std_s1:.{decimals}f}"
                if not np.isnan(std_s1)
                else (f"{mean_s1:.{decimals}f}" if not np.isnan(mean_s1) else "N/A")
            )

            row_mean = {
                "Variable": var,
                "Metric / Level": "Mean ± SD",
                col_overall: f"{mean_all:.{decimals}f} ± {std_all:.{decimals}f}" if not np.isnan(std_all) else f"{mean_all:.{decimals}f}",
                col_s0: mean_s0_str,
                col_s1: mean_s1_str,
                "p-value": p_str,
                "Test": test_name,
            }
            row_med = {
                "Variable": var,
                "Metric / Level": "Median [IQR]",
                col_overall: f"{med_all:.{decimals}f} [{q25_all:.{decimals}f} - {q75_all:.{decimals}f}]",
                col_s0: f"{med_s0:.{decimals}f} [{q25_s0:.{decimals}f} - {q75_s0:.{decimals}f}]" if not np.isnan(med_s0) else "N/A",
                col_s1: f"{med_s1:.{decimals}f} [{q25_s1:.{decimals}f} - {q75_s1:.{decimals}f}]" if not np.isnan(med_s1) else "N/A",
                "p-value": "",
                "Test": "",
            }
        else:
            row_mean = {
                "Variable": var,
                "Metric / Level": "Mean ± SD",
                col_overall: f"{mean_all:.{decimals}f} ± {std_all:.{decimals}f}",
            }
            row_med = {
                "Variable": var,
                "Metric / Level": "Median [IQR]",
                col_overall: f"{med_all:.{decimals}f} [{q25_all:.{decimals}f} - {q75_all:.{decimals}f}]",
            }

        rows.append(row_mean)
        rows.append(row_med)

    # Process Categorical Variables
    for var in cat_vars:
        if var not in df.columns:
            continue

        raw_series = df[var].dropna()
        if hasattr(df[var], "cat") and hasattr(df[var].cat, "categories"):
            categories = list(df[var].cat.categories)
        elif set(raw_series.unique()).issubset({0, 1}):
            categories = [0, 1]
        else:
            categories = sorted(raw_series.unique(), key=lambda x: str(x))

        chi2_p_str = ""
        chi2_test_name = ""

        if is_stratified and len(strata_vals) == 2:
            tab = pd.crosstab(df[var], df[stratify_by])
            if (
                tab.shape[0] >= 2
                and tab.shape[1] >= 2
                and (tab.sum(axis=0) > 0).all()
                and (tab.sum(axis=1) > 0).all()
            ):
                try:
                    chi2_res = stats.chi2_contingency(tab)
                    chi2_p_str = _format_p_value(chi2_res.pvalue)
                    chi2_test_name = "Pearson Chi-square"
                except Exception:
                    chi2_p_str = "N/A"
                    chi2_test_name = "N/A"
            else:
                chi2_p_str = "N/A"
                chi2_test_name = "N/A"

            denom_s0 = (df[stratify_by] == strata_vals[0]).sum()
            denom_s1 = (df[stratify_by] == strata_vals[1]).sum()

        for idx, cat in enumerate(categories):
            n_all = (df[var] == cat).sum()
            pct_all = (n_all / total_n * 100.0) if total_n > 0 else 0.0

            if is_stratified and len(strata_vals) == 2:
                n_s0 = ((df[stratify_by] == strata_vals[0]) & (df[var] == cat)).sum()
                pct_s0 = (n_s0 / denom_s0 * 100.0) if denom_s0 > 0 else 0.0

                n_s1 = ((df[stratify_by] == strata_vals[1]) & (df[var] == cat)).sum()
                pct_s1 = (n_s1 / denom_s1 * 100.0) if denom_s1 > 0 else 0.0

                row_cat = {
                    "Variable": var,
                    "Metric / Level": str(cat),
                    col_overall: f"{n_all:,} ({pct_all:.1f}%)",
                    col_s0: f"{n_s0:,} ({pct_s0:.1f}%)",
                    col_s1: f"{n_s1:,} ({pct_s1:.1f}%)",
                    "p-value": chi2_p_str if idx == 0 else "",
                    "Test": chi2_test_name if idx == 0 else "",
                }
            else:
                row_cat = {
                    "Variable": var,
                    "Metric / Level": str(cat),
                    col_overall: f"{n_all:,} ({pct_all:.1f}%)",
                }
            rows.append(row_cat)

    return pd.DataFrame(rows)


def generate_paediatric_table_one(
    df_paediatric: pd.DataFrame,
    continuous_vars: Optional[List[str]] = None,
    categorical_vars: Optional[List[str]] = None,
    decimals: int = 2,
) -> pd.DataFrame:
    """Generate exploratory descriptive Table 1 for the paediatric cohort (age < 18).

    Handles low cell counts (stroke = 2, non-stroke = 854) gracefully without statistical
    convergence errors.
    """
    return generate_table_one(
        df_paediatric,
        stratify_by="stroke",
        continuous_vars=continuous_vars,
        categorical_vars=categorical_vars,
        decimals=decimals,
    )


def table_one_to_markdown(
    table_df: pd.DataFrame,
    group_variables: bool = False,
) -> str:
    """Export Table 1 DataFrame to formatted GitHub-Flavored Markdown table.

    Parameters
    ----------
    table_df : pd.DataFrame
        Table 1 DataFrame generated by generate_table_one().
    group_variables : bool, default False
        If True, suppresses repeating variable names on consecutive rows for readability.
    """
    df_export = table_df.copy()
    if group_variables and "Variable" in df_export.columns:
        last_var = None
        grouped_vars = []
        for v in df_export["Variable"]:
            if v == last_var:
                grouped_vars.append("")
            else:
                grouped_vars.append(f"**{v}**")
                last_var = v
        df_export["Variable"] = grouped_vars

    headers = [str(c) for c in df_export.columns]
    rows = [
        [str(cell) if not pd.isna(cell) else "" for cell in row]
        for row in df_export.values
    ]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    header_line = (
        "| "
        + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        + " |"
    )
    sep_line = (
        "| "
        + " | ".join(":" + "-" * max(col_widths[i] - 1, 2) for i in range(len(headers)))
        + " |"
    )
    row_lines = [
        "| "
        + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row))
        + " |"
        for row in rows
    ]

    return "\n".join([header_line, sep_line] + row_lines)


def table_one_to_html(
    table_df: pd.DataFrame,
    title: Optional[str] = "Table 1: Baseline Clinical Characteristics",
    group_variables: bool = False,
) -> str:
    """Export Table 1 DataFrame to publication-grade styled HTML table."""
    df_export = table_df.copy()
    if group_variables and "Variable" in df_export.columns:
        last_var = None
        grouped_vars = []
        for v in df_export["Variable"]:
            if v == last_var:
                grouped_vars.append("")
            else:
                grouped_vars.append(f"<strong>{v}</strong>")
                last_var = v
        df_export["Variable"] = grouped_vars

    html_table = df_export.to_html(index=False, escape=False, classes="table-one")

    css = """<style>
.table-one-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 1rem 0;
    overflow-x: auto;
}
.table-one-title {
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: #1a1a1a;
}
table.table-one {
    border-collapse: collapse;
    width: 100%;
    font-size: 0.88rem;
    border-top: 2px solid #222222;
    border-bottom: 2px solid #222222;
}
table.table-one th {
    border-bottom: 1.5px solid #222222;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    background-color: #f8fafc;
}
table.table-one td {
    padding: 6px 12px;
    border-bottom: 1px solid #e2e8f0;
    text-align: left;
}
table.table-one tr:hover {
    background-color: #f1f5f9;
}
</style>"""

    title_html = f'<div class="table-one-title">{title}</div>\n' if title else ""
    return f"{css}\n<div class=\"table-one-container\">\n{title_html}{html_table}\n</div>"


def print_table_one(table_df: pd.DataFrame, title: Optional[str] = None) -> None:
    """Print formatted Table 1 to console."""
    if title:
        print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    print(table_one_to_markdown(table_df, group_variables=True))


# Multivariable Logistic Regression and Forest Plot Data Preparation

COVARIATE_LABELS_PL: Dict[str, str] = {
    "hypertension": "Nadciśnienie tętnicze",
    "heart_disease": "Choroba serca",
    "smoking_status_smokes": "Palenie: Aktywni palacze",
    "smoking_status_formerly_smoked": "Palenie: Byli palacze",
    "smoking_status_Unknown": "Palenie: Nieznane",
    "gender_Male": "Płeć (Mężczyzna)",
    "age": "Wiek (lata)",
    "avg_glucose_level": "Poziom glukozy (mg/dl)",
    "bmi": "BMI (kg/m²)",
}

PARAM_NAME_MAP: Dict[str, Tuple[str, str]] = {
    'C(gender, Treatment(reference="Female"))[T.Male]': (
        "gender_Male",
        "Płeć (Mężczyzna)",
    ),
    'C(smoking_status, Treatment(reference="never smoked"))[T.smokes]': (
        "smoking_status_smokes",
        "Palenie: Aktywni palacze",
    ),
    'C(smoking_status, Treatment(reference="never smoked"))[T.formerly smoked]': (
        "smoking_status_formerly_smoked",
        "Palenie: Byli palacze",
    ),
    'C(smoking_status, Treatment(reference="never smoked"))[T.Unknown]': (
        "smoking_status_Unknown",
        "Palenie: Nieznane",
    ),
    "hypertension": ("hypertension", "Nadciśnienie tętnicze"),
    "heart_disease": ("heart_disease", "Choroba serca"),
    "age": ("age", "Wiek (lata)"),
    "avg_glucose_level": ("avg_glucose_level", "Poziom glukozy (mg/dl)"),
    "bmi": ("bmi", "BMI (kg/m²)"),
}


def _map_param_name(param: str) -> Tuple[str, str]:
    """Map statsmodels formula term name to clean variable identifier and Polish label."""
    if param in PARAM_NAME_MAP:
        return PARAM_NAME_MAP[param]
    if "gender" in param and "Male" in param:
        return ("gender_Male", "Płeć (Mężczyzna)")
    if "smoking_status" in param and "formerly" in param:
        return ("smoking_status_formerly_smoked", "Palenie: Byli palacze")
    if "smoking_status" in param and "smokes" in param:
        return ("smoking_status_smokes", "Palenie: Aktywni palacze")
    if "smoking_status" in param and "Unknown" in param:
        return ("smoking_status_Unknown", "Palenie: Nieznane")
    clean_var = (
        param.replace("C(", "")
        .replace(")", "")
        .replace("[T.", "_")
        .replace("]", "")
    )
    return (clean_var, COVARIATE_LABELS_PL.get(clean_var, clean_var))


def fit_multivariable_logistic_regression(
    df: Optional[pd.DataFrame] = None,
    formula: Optional[str] = None,
    bmi_imputation: str = "median",
    alpha: float = 0.05,
    filter_adults: bool = True,
    remove_other_gender: bool = True,
    return_model: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Any]]:
    """Fit multivariable logistic regression on stroke risk cohort.

    Evaluates core biological risk factors (age, avg_glucose_level, bmi,
    hypertension, heart_disease, gender, smoking_status) while excluding
    demographic proxies (work_type, Residence_type, ever_married) to prevent
    collinearity.

    Missing BMI values are handled via cohort median imputation by default
    (avoiding loss of 40 adult stroke cases) or dropna.

    Parameters
    ----------
    df : Optional[pd.DataFrame]
        Cohort dataset. If None, loaded via get_curated_cohorts().
    formula : Optional[str]
        Statsmodels formula. Defaults to biological covariate model.
    bmi_imputation : str, default 'median'
        Strategy for missing BMI: 'median' or 'drop'.
    alpha : float, default 0.05
        Significance threshold for Wald confidence intervals and p-value flag.
    filter_adults : bool, default True
        Whether to filter for adults (age >= 18).
    remove_other_gender : bool, default True
        Whether to filter out singleton 'Other' gender.
    return_model : bool, default False
        Whether to return tuple of (results_df, fitted_model).

    Returns
    -------
    pd.DataFrame or Tuple[pd.DataFrame, Any]
        DataFrame with columns: variable, label_pl, coef, std_err, z_value,
        p_value, or, ci_lower, ci_upper, is_significant.
    """
    if df is None:
        from src.data import get_curated_cohorts

        cohorts = get_curated_cohorts()
        working_df = cohorts.adult.copy()
    else:
        working_df = df.copy()

    if filter_adults and "age" in working_df.columns:
        working_df = working_df[working_df["age"] >= 18.0].copy()

    if remove_other_gender and "gender" in working_df.columns:
        working_df = working_df[working_df["gender"] != "Other"].copy()

    if "bmi" in working_df.columns:
        if bmi_imputation == "median":
            median_bmi = float(working_df["bmi"].median())
            working_df["bmi"] = working_df["bmi"].fillna(median_bmi)
        elif bmi_imputation == "drop":
            working_df = working_df.dropna(subset=["bmi"]).copy()
        else:
            raise ValueError(
                f"Unsupported bmi_imputation: {bmi_imputation}. Use 'median' or 'drop'."
            )

    if formula is None:
        formula = (
            "stroke ~ age + avg_glucose_level + bmi + hypertension + heart_disease "
            '+ C(gender, Treatment(reference="Female")) '
            '+ C(smoking_status, Treatment(reference="never smoked"))'
        )

    model = smf.logit(formula=formula, data=working_df).fit(disp=False)
    conf_int = model.conf_int(alpha=alpha)
    params = model.params
    bse = model.bse
    tvalues = model.tvalues
    pvalues = model.pvalues

    records: List[Dict[str, Any]] = []
    for term in params.index:
        if term == "Intercept":
            continue
        var_name, label_pl = _map_param_name(term)
        coef = float(params[term])
        std_err = float(bse[term])
        z_val = float(tvalues[term])
        p_val = float(pvalues[term])
        odds_ratio = float(np.exp(coef))
        ci_l = float(np.exp(conf_int.loc[term, 0]))
        ci_u = float(np.exp(conf_int.loc[term, 1]))
        is_sig = bool(p_val < alpha)

        records.append({
            "variable": var_name,
            "label_pl": label_pl,
            "coef": coef,
            "std_err": std_err,
            "z_value": z_val,
            "p_value": p_val,
            "or": odds_ratio,
            "ci_lower": ci_l,
            "ci_upper": ci_u,
            "is_significant": is_sig,
        })

    results_df = (
        pd.DataFrame(records)
        .sort_values(by="or", ascending=False)
        .reset_index(drop=True)
    )
    results_df.attrs["model"] = model
    results_df.attrs["formula"] = formula

    if return_model:
        return results_df, model
    return results_df
