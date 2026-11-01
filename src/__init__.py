"""Stroke prediction medical data analysis package."""

from src.data import (
    load_data,
    curate_stroke_data,
    get_curated_cohorts,
    validate_missing_bmi_patterns,
    CuratedCohorts,
)
from src.biostats import (
    generate_table_one,
    generate_paediatric_table_one,
    table_one_to_markdown,
    table_one_to_html,
    print_table_one,
    fit_multivariable_logistic_regression,
    COVARIATE_LABELS_PL,
)
from src.visualization import (
    plot_forest_matplotlib,
    plot_forest_plotly,
    create_risk_gauge,
    plot_global_feature_importance,
    plot_global_feature_importance_plotly,
    plot_shap_waterfall,
    plot_shap_waterfall_plotly,
)
from src.xai import (
    get_feature_importances,
    get_tree_explainer,
    compute_shap_values,
    get_clinical_archetypes,
    FEATURE_LABELS_PL,
    format_feature_label_pl,
)
from src.ml import (
    build_preprocessor,
    get_model_catalog,
    optimize_threshold,
    compute_classification_metrics,
    classify_risk_tier,
    run_ml_benchmark,
    save_production_pipeline,
    load_production_pipeline,
    predict_stroke_risk,
)
from src.population import (
    get_population_baseline_risks,
    calculate_patient_percentiles,
    plot_population_risk_distribution,
    get_age_decade_label,
)
from src.vascular_age import (
    calculate_vascular_age,
)
from src.trajectory import (
    calculate_age_risk_trajectory,
    plot_age_risk_trajectory,
)

__all__ = [
    "load_data",
    "curate_stroke_data",
    "get_curated_cohorts",
    "validate_missing_bmi_patterns",
    "CuratedCohorts",
    "generate_table_one",
    "generate_paediatric_table_one",
    "table_one_to_markdown",
    "table_one_to_html",
    "print_table_one",
    "fit_multivariable_logistic_regression",
    "COVARIATE_LABELS_PL",
    "plot_forest_matplotlib",
    "plot_forest_plotly",
    "create_risk_gauge",
    "plot_global_feature_importance",
    "plot_global_feature_importance_plotly",
    "plot_shap_waterfall",
    "plot_shap_waterfall_plotly",
    "get_feature_importances",
    "get_tree_explainer",
    "compute_shap_values",
    "get_clinical_archetypes",
    "FEATURE_LABELS_PL",
    "format_feature_label_pl",
    "build_preprocessor",
    "get_model_catalog",
    "optimize_threshold",
    "compute_classification_metrics",
    "classify_risk_tier",
    "run_ml_benchmark",
    "save_production_pipeline",
    "load_production_pipeline",
    "predict_stroke_risk",
    "get_population_baseline_risks",
    "calculate_patient_percentiles",
    "plot_population_risk_distribution",
    "get_age_decade_label",
    "calculate_vascular_age",
    "calculate_age_risk_trajectory",
    "plot_age_risk_trajectory",
]
