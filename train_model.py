"""Train and export the best imbalance-aware fraud model for the Streamlit app.

The source notebook explored several classifiers. This production-facing
workflow compares three candidate estimators inside the same preprocessing and
SMOTE pipeline, selects the best estimator with stratified cross-validation using
average precision, refits that estimator on the complete training split, and
saves the fitted pipeline for model-agnostic Streamlit inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

REQUIRED_FEATURES = ["Time", *[f"V{i}" for i in range(1, 29)], "Amount"]
TARGET = "Class"


def validate_frame(df: pd.DataFrame) -> None:
    required = set(REQUIRED_FEATURES + [TARGET])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if not bool(df[TARGET].dropna().isin([0, 1]).all()):
        raise ValueError("Class must contain only 0 (legitimate) and 1 (fraud) labels.")
    if df[TARGET].nunique() < 2:
        raise ValueError("The uploaded dataset must contain both classes.")


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("robust_scale_amount_time", RobustScaler(), ["Time", "Amount"]),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def build_search_pipeline(smote_ratio: float, random_state: int) -> ImbPipeline:
    """Build the common pipeline; the classifier is selected by GridSearchCV."""
    return ImbPipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("smote", SMOTE(sampling_strategy=smote_ratio, random_state=random_state)),
            ("classifier", LogisticRegression(solver="liblinear", max_iter=1500, random_state=random_state)),
        ]
    )


def build_model_grid(random_state: int) -> list[dict[str, list[Any]]]:
    """Return a compact, strong candidate set for practical local training."""
    return [
        {
            "classifier": [
                LogisticRegression(
                    C=0.5,
                    solver="liblinear",
                    max_iter=1500,
                    random_state=random_state,
                )
            ]
        },
        {
            "classifier": [
                RandomForestClassifier(
                    n_estimators=160,
                    max_depth=14,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                    random_state=random_state,
                )
            ]
        },
        {
            "classifier": [
                HistGradientBoostingClassifier(
                    max_iter=180,
                    learning_rate=0.08,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=random_state,
                )
            ]
        },
        {
            "classifier": [
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    n_jobs=1,
                    random_state=random_state,
                )
            ]
        },
        {
            "classifier": [
                XGBClassifier(
                    n_estimators=220,
                    max_depth=5,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.85,
                    min_child_weight=3,
                    reg_lambda=2.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    n_jobs=1,
                    random_state=random_state,
                    verbosity=0,
                )
            ]
        },
    ]


def _downsample(values: np.ndarray, max_points: int = 600) -> list[float]:
    values = np.asarray(values)
    if len(values) <= max_points:
        return values.tolist()
    indices = np.linspace(0, len(values) - 1, max_points, dtype=int)
    return values[indices].tolist()


def evaluate(y_true: pd.Series, probabilities: np.ndarray, model_name: str) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    fpr, tpr, roc_thresholds = roc_curve(y_true, probabilities)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, probabilities)
    return {
        "model_name": model_name,
        "threshold": 0.5,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": cm.tolist(),
        "roc_curve": {
            "fpr": _downsample(fpr),
            "tpr": _downsample(tpr),
            "thresholds": [None if not np.isfinite(x) else float(x) for x in _downsample(roc_thresholds)],
        },
        "precision_recall_curve": {
            "precision": _downsample(precision),
            "recall": _downsample(recall),
            "thresholds": _downsample(pr_thresholds),
        },
    }


def train(
    data_path: Path,
    output_dir: Path,
    test_size: float,
    random_state: int,
    smote_ratio: float,
    search_rows: int,
    cv_folds: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_path)
    validate_frame(df)

    X = df[REQUIRED_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Searching on a stratified subset keeps local retraining practical while
    # retaining all classes. The chosen estimator is refit on all X_train below.
    search_size = min(search_rows, len(X_train))
    X_search, _, y_search, _ = train_test_split(
        X_train,
        y_train,
        train_size=search_size,
        stratify=y_train,
        random_state=random_state,
    )
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    search = GridSearchCV(
        estimator=build_search_pipeline(smote_ratio=smote_ratio, random_state=random_state),
        param_grid=build_model_grid(random_state=random_state),
        scoring={"average_precision": "average_precision", "roc_auc": "roc_auc"},
        refit="average_precision",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
        error_score="raise",
    )
    search.fit(X_search, y_search)

    # best_estimator_ is the selected model from the imbalance-aware CV search.
    selected_model = clone(search.best_estimator_)
    selected_model.fit(X_train, y_train)
    classifier = selected_model.named_steps["classifier"]
    selected_model_name = type(classifier).__name__
    probabilities = selected_model.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, probabilities, selected_model_name)

    comparison = pd.DataFrame(
        {
            "model": [type(params["classifier"]).__name__ for params in search.cv_results_["params"]],
            "mean_average_precision": search.cv_results_["mean_test_average_precision"],
            "std_average_precision": search.cv_results_["std_test_average_precision"],
            "mean_roc_auc": search.cv_results_["mean_test_roc_auc"],
            "rank_average_precision": search.cv_results_["rank_test_average_precision"],
        }
    ).sort_values(["rank_average_precision", "mean_average_precision"], ascending=[True, False])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    predictions = (probabilities >= 0.5).astype(int)
    pd.DataFrame(
        {
            "actual_class": y_test.to_numpy(),
            "fraud_probability": probabilities,
            "predicted_class": predictions,
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    metadata = {
        "model_name": selected_model_name,
        "model_selection": "GridSearchCV best_estimator_ selected by mean average precision",
        "candidate_models": sorted(comparison["model"].unique().tolist()),
        "selection_metric": "mean_average_precision",
        "feature_columns": REQUIRED_FEATURES,
        "target_column": TARGET,
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(REQUIRED_FEATURES)),
        "fraud_count": int(y.sum()),
        "legitimate_count": int((y == 0).sum()),
        "fraud_rate": float(y.mean()),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "search_rows": int(len(X_search)),
        "cv_folds": cv_folds,
        "test_size": test_size,
        "random_state": random_state,
        "smote_sampling_strategy": smote_ratio,
        "threshold_default": 0.5,
        "notes": [
            "Time and Amount are robust-scaled inside the fitted pipeline.",
            "SMOTE is fit only on each training fold during model selection and on X_train during final refit.",
            "The test split remains untouched until final evaluation.",
            "The displayed probability is a model score, not a calibrated financial risk estimate.",
        ],
    }

    joblib.dump(selected_model, output_dir / "fraud_pipeline.joblib", compress=3)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"metadata": metadata, "metrics": metrics, "best_params": search.best_params_}, indent=2, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/creditcard.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--smote-ratio", type=float, default=0.25)
    parser.add_argument(
        "--search-rows",
        type=int,
        default=120000,
        help="Stratified rows used for CV model selection before refitting the winner on all training rows.",
    )
    parser.add_argument("--cv-folds", type=int, default=3)
    args = parser.parse_args()
    train(
        data_path=args.data,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        smote_ratio=args.smote_ratio,
        search_rows=args.search_rows,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
