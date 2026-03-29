"""
Model validation script for binary prediction projects.
Runs end-to-end checks on a trained model using a labeled CSV dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from prediction import ChurnPredictor


def _resolve_target_info(preprocessing_info: dict) -> tuple[str, str, str]:
    """Resolve target metadata with backward-compatible fallbacks."""
    target_info = preprocessing_info.get("target_info", {}) or {}
    target_col = str(target_info.get("target_column", "Churn"))
    positive_label = str(target_info.get("positive_label", "Yes"))
    negative_label = str(target_info.get("negative_label", "No"))
    return target_col, positive_label, negative_label


def evaluate_model(
    data_path: str,
    model_path: str,
    preprocessing_path: str,
    min_auc: float,
    min_accuracy: float,
) -> int:
    """Evaluate trained model and print pass/fail summary."""
    predictor = ChurnPredictor(model_path=model_path, preprocessing_path=preprocessing_path)
    target_col, positive_label, negative_label = _resolve_target_info(predictor.preprocessing_info)

    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in dataset. Available columns: {list(df.columns)}"
        )

    y_raw = df[target_col].astype(str)
    if y_raw.nunique() != 2:
        raise ValueError(f"Expected binary target, found classes: {sorted(y_raw.unique().tolist())}")

    X = df.drop(columns=[target_col])

    original_df, X_scaled = predictor.preprocess_input(X)
    y_true = (y_raw == positive_label).astype(int).to_numpy()

    y_pred = predictor.model.predict(X_scaled.values)
    y_proba = predictor.model.predict_proba(X_scaled.values)[:, 1]

    accuracy = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    auc = float(roc_auc_score(y_true, y_proba))

    passed = (accuracy >= min_accuracy) and (auc >= min_auc)

    summary = {
        "data_path": data_path,
        "model_path": model_path,
        "preprocessing_path": preprocessing_path,
        "target_column": target_col,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "rows_evaluated": int(len(df)),
        "metrics": {
            "accuracy": round(accuracy, 4),
            "f1": round(f1, 4),
            "roc_auc": round(auc, 4),
        },
        "thresholds": {
            "min_accuracy": min_accuracy,
            "min_auc": min_auc,
        },
        "status": "PASS" if passed else "FAIL",
    }

    print("\n" + "=" * 70)
    print("MODEL VALIDATION REPORT")
    print("=" * 70)
    print(json.dumps(summary, indent=2))
    print("=" * 70 + "\n")

    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate trained binary prediction model")
    parser.add_argument(
        "--data-path",
        default="data/raw/customer_data.csv",
        help="Path to labeled CSV used for validation",
    )
    parser.add_argument(
        "--model-path",
        default="models/churn_model_best.pkl",
        help="Path to trained model artifact",
    )
    parser.add_argument(
        "--preprocessing-path",
        default="models/churn_model_best_preprocessing.pkl",
        help="Path to preprocessing artifact",
    )
    parser.add_argument(
        "--min-auc",
        type=float,
        default=0.75,
        help="Minimum acceptable ROC-AUC score",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.70,
        help="Minimum acceptable accuracy score",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in [args.data_path, args.model_path, args.preprocessing_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    raise SystemExit(
        evaluate_model(
            data_path=args.data_path,
            model_path=args.model_path,
            preprocessing_path=args.preprocessing_path,
            min_auc=args.min_auc,
            min_accuracy=args.min_accuracy,
        )
    )
