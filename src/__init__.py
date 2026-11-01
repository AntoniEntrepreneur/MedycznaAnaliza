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
]
