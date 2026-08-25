"""Reproducible churn modeling pipeline for the Maven Analytics telecom data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".matplotlib")
)

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42
TARGET_COLUMN = "churned"

# These columns either reveal the outcome, identify a customer, or create an
# unnecessarily high-cardinality/geographically precise feature.
EXCLUDED_FEATURES = [
    "Customer ID",
    "Customer Status",
    "Churn Category",
    "Churn Reason",
    "City",
    "Zip Code",
    "Latitude",
    "Longitude",
]


def load_source_data(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the customer and ZIP population tables."""
    customers = pd.read_csv(data_dir / "telecom_customer_churn.csv")
    population = pd.read_csv(data_dir / "telecom_zipcode_population.csv")
    return customers, population


def prepare_dataset(
    customers: pd.DataFrame, population: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Create a leakage-safe binary churn data set.

    Customers who joined during the measured quarter are excluded because they
    have not had the same opportunity to churn as established customers.
    """
    required = {"Customer Status", "Zip Code"}
    missing = required.difference(customers.columns)
    if missing:
        raise ValueError(f"Missing required customer columns: {sorted(missing)}")
    if not {"Zip Code", "Population"}.issubset(population.columns):
        raise ValueError("The population table must contain Zip Code and Population.")

    frame = customers.copy()
    frame["Zip Code"] = pd.to_numeric(frame["Zip Code"], errors="coerce").astype("Int64")

    zip_population = population[["Zip Code", "Population"]].copy()
    zip_population["Zip Code"] = pd.to_numeric(
        zip_population["Zip Code"], errors="coerce"
    ).astype("Int64")
    zip_population["Population"] = pd.to_numeric(
        zip_population["Population"], errors="coerce"
    )

    frame = frame.merge(zip_population, on="Zip Code", how="left", validate="many_to_one")
    frame = frame.loc[frame["Customer Status"].isin(["Stayed", "Churned"])].copy()
    frame[TARGET_COLUMN] = frame["Customer Status"].eq("Churned").astype("int8")

    y = frame[TARGET_COLUMN]
    X = frame.drop(columns=[TARGET_COLUMN, *EXCLUDED_FEATURES], errors="ignore")

    if y.nunique() != 2:
        raise ValueError("The prepared target is not binary.")
    leaked = set(EXCLUDED_FEATURES).intersection(X.columns)
    if leaked:
        raise ValueError(f"Outcome leakage columns remain: {sorted(leaked)}")

    return X, y, frame


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing that is fitted only on training folds."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def candidate_models(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Return comparable end-to-end model pipelines."""
    estimators = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2_000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    return {
        name: Pipeline(
            steps=[("preprocessor", build_preprocessor(X)), ("model", estimator)]
        )
        for name, estimator in estimators.items()
    }


def score_predictions(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate classification metrics with churn as the positive class."""
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred, zero_division=0),
        "test_f1": f1_score(y_true, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_true, y_prob),
        "test_average_precision": average_precision_score(y_true, y_prob),
    }


def create_eda_plot(analysis_frame: pd.DataFrame, report_dir: Path) -> None:
    """Plot churn rate and customer count by contract."""
    contract = (
        analysis_frame.groupby("Contract", observed=True)[TARGET_COLUMN]
        .agg(churn_rate="mean", customers="size")
        .sort_values("churn_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(contract.index, contract["churn_rate"] * 100, color="#1976D2")
    ax.set(title="Churn rate by contract", ylabel="Churn rate (%)", xlabel="")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, max(50, float(contract["churn_rate"].max() * 115)))
    fig.tight_layout()
    fig.savefig(report_dir / "churn_by_contract.png", dpi=180)
    plt.close(fig)


def train_and_report(
    data_dir: Path = DEFAULT_DATA_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> pd.DataFrame:
    """Train candidates, select by cross-validation, and write test artifacts."""
    report_dir.mkdir(parents=True, exist_ok=True)
    customers, population = load_source_data(data_dir)
    X, y, analysis_frame = prepare_dataset(customers, population)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fitted: dict[str, Pipeline] = {}
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, float | str]] = []

    for name, pipeline in candidate_models(X_train).items():
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        fitted[name] = pipeline
        predictions[name] = (y_pred, y_prob)
        rows.append(
            {
                "model": name,
                "cv_roc_auc_mean": float(cv_scores.mean()),
                "cv_roc_auc_std": float(cv_scores.std()),
                **score_predictions(y_test, y_pred, y_prob),
            }
        )

    metrics = pd.DataFrame(rows).sort_values("cv_roc_auc_mean", ascending=False)
    best_name = str(metrics.iloc[0]["model"])
    best_pipeline = fitted[best_name]
    best_pred, _ = predictions[best_name]

    metrics.to_csv(report_dir / "metrics.csv", index=False, float_format="%.6f")
    joblib.dump(best_pipeline, report_dir / "churn_model.joblib")

    importance = permutation_importance(
        best_pipeline,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    feature_importance = (
        pd.DataFrame(
            {
                "feature": X_test.columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    feature_importance.to_csv(
        report_dir / "feature_importance.csv", index=False, float_format="%.6f"
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (_, y_prob) in predictions.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves on the hold-out test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / "roc_curves.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        best_pred,
        display_labels=["Stayed", "Churned"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    top = feature_importance.head(12).sort_values("importance_mean")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#00897B")
    ax.set(title=f"Permutation importance — {best_name}", xlabel="Mean decrease in ROC-AUC")
    fig.tight_layout()
    fig.savefig(report_dir / "feature_importance.png", dpi=180)
    plt.close(fig)

    create_eda_plot(analysis_frame, report_dir)

    summary = {
        "source_rows": int(len(customers)),
        "modeling_rows": int(len(X)),
        "churned_customers": int(y.sum()),
        "churn_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_columns": int(X.shape[1]),
        "selected_model": best_name,
        "selection_rule": "highest mean 5-fold ROC-AUC on the training split",
    }
    (report_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = train_and_report(arguments.data_dir, arguments.report_dir)
    print(result.to_string(index=False))
