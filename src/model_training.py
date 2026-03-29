"""
Machine Learning Model Training for Binary Prediction
Trains multiple models and saves the best one
"""

import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def _infer_positive_label(unique_labels):
    """Infer positive class label from common binary naming conventions."""
    ranked_tokens = ["yes", "true", "1", "churn", "positive", "fraud", "default"]
    normalized = {str(label).strip().lower(): label for label in unique_labels}

    for token in ranked_tokens:
        for norm, original in normalized.items():
            if token == norm or token in norm:
                return original

    # Fallback to the second class to keep deterministic binary mapping.
    return list(unique_labels)[1]


def load_and_preprocess_data(filepath, target_col='Churn', positive_label=None):
    """
    Load and preprocess binary classification data
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        tuple: (X, y, feature_names, label_encoders, target_info, numeric_defaults)
    """
    print("[DATA] Loading data...")
    df = pd.read_csv(filepath)
    
    # Separate features and target
    if target_col in df.columns:
        y_raw = df[target_col]
        X = df.drop(target_col, axis=1).copy()
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset")

    unique_labels = list(pd.Series(y_raw).dropna().unique())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Target column '{target_col}' must be binary. Found {len(unique_labels)} classes: {unique_labels}"
        )

    if positive_label is None:
        positive_label = _infer_positive_label(unique_labels)
    elif positive_label not in unique_labels:
        raise ValueError(
            f"Provided positive label '{positive_label}' not found in target classes: {unique_labels}"
        )

    negative_label = [label for label in unique_labels if label != positive_label][0]
    y = (y_raw == positive_label).astype(int)

    target_info = {
        'target_column': target_col,
        'positive_label': positive_label,
        'negative_label': negative_label,
        'class_mapping': {0: negative_label, 1: positive_label},
        'positive_class_index': 1,
    }
    
    print(f"[OK] Data shape: {X.shape}")
    print(f"   Target distribution: {y_raw.value_counts().to_dict()}")
    print(f"   Positive class: {positive_label}")

    # Convert numeric-like object columns to numeric where possible
    for col in X.select_dtypes(include=['object']).columns:
        converted = pd.to_numeric(X[col], errors='coerce')
        if converted.notna().mean() >= 0.8:
            X[col] = converted

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    numeric_defaults = {}
    for col in numeric_cols:
        median_val = float(X[col].median()) if X[col].notna().any() else 0.0
        X[col] = X[col].fillna(median_val)
        numeric_defaults[col] = median_val
    
    # Handle categorical variables (Label Encoding)
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    
    print(f"\n[LABEL] Encoding categorical features...")
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"   [CHECK] {col}: {len(le.classes_)} classes")
    
    feature_names = X.columns.tolist()
    
    return X, y, feature_names, label_encoders, target_info, numeric_defaults

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple ML models and compare performance
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        
    Returns:
        dict: Model performance metrics
    """
    
    print("\n" + "="*70)
    print("ML MODEL TRAINING")
    print("="*70)
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n[ML] Training {model_name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[model_name] = {
            'model': model,
            'auc': auc_score,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"   [OK] AUC Score: {auc_score:.4f}")
        print(f"   [OK] Accuracy: {model.score(X_test, y_test):.4f}")
    
    return results

# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_model(model, X_test, y_test, model_name="", class_labels=None):
    """
    Detailed model evaluation
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of model
    """
    print("\n" + "="*70)
    print(f"[EVAL] DETAILED EVALUATION - {model_name}")
    print("="*70)
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    class_labels = class_labels or {0: "Negative", 1: "Positive"}

    print("\n[LIST] Classification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[str(class_labels[0]), str(class_labels[1])],
        )
    )
    
    print("\n[CHART] Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"   True Negatives: {cm[0,0]}")
    print(f"   False Positives: {cm[0,1]}")
    print(f"   False Negatives: {cm[1,0]}")
    print(f"   True Positives: {cm[1,1]}")
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\n[TREND] ROC-AUC Score: {auc_score:.4f}")
    
    return auc_score

# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(model, preprocessing_info, model_name="churn_model"):
    """
    Save trained model and preprocessing info
    
    Args:
        model: Trained model
        preprocessing_info: Dict with encoding info
        model_name: Name for saved model
    """
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    print(f"[OK] Model saved: {model_path}")
    
    # Save preprocessing info
    preprocessing_path = models_dir / f"{model_name}_preprocessing.pkl"
    joblib.dump(preprocessing_info, preprocessing_path)
    print(f"[OK] Preprocessing info saved: {preprocessing_path}")
    
    return str(model_path)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(
    data_path='data/raw/customer_data.csv',
    model_name='churn_model_best',
    target_col='Churn',
    positive_label=None,
):
    """Main training pipeline"""
    
    print("="*70)
    print("CUSTOMER CHURN PREDICTION - ML MODEL TRAINING")
    print("="*70)
    
    # Load and preprocess
    print(f"[DATA] Using dataset: {data_path}")
    X, y, feature_names, label_encoders, target_info, numeric_defaults = load_and_preprocess_data(
        data_path,
        target_col=target_col,
        positive_label=positive_label,
    )
    
    # Train-test split
    print("\n[CHART] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\n[TREND] Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    results = train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['auc'])
    best_model = results[best_model_name]['model']
    best_auc = results[best_model_name]['auc']
    
    print("\n" + "="*70)
    print(f"[CHAMP] BEST MODEL: {best_model_name} (AUC: {best_auc:.4f})")
    print("="*70)
    
    # Detailed evaluation
    evaluate_model(
        best_model,
        X_test_scaled,
        y_test,
        best_model_name,
        class_labels=target_info['class_mapping'],
    )
    
    # Save model
    preprocessing_info = {
        'scaler': scaler,
        'feature_names': feature_names,
        'label_encoders': label_encoders,
        'numeric_defaults': numeric_defaults,
        'target_info': target_info,
        'model_name': best_model_name,
    }
    
    save_model(best_model, preprocessing_info, model_name)
    
    print("\n[OK] Training complete! Model ready for deployment.")
    print("\n[CHART] Feature Importance (Top 10):")
    
    if hasattr(best_model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print(importance_df.to_string(index=False))

def parse_args():
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train customer churn model on a specified CSV dataset."
    )
    parser.add_argument(
        "--data-path",
        default="data/raw/customer_data.csv",
        help="Path to input CSV file (default: data/raw/customer_data.csv)",
    )
    parser.add_argument(
        "--model-name",
        default="churn_model_best",
        help="Base filename for saved model artifacts (default: churn_model_best)",
    )
    parser.add_argument(
        "--target-col",
        default="Churn",
        help="Name of binary target column in the dataset (default: Churn)",
    )
    parser.add_argument(
        "--positive-label",
        default=None,
        help="Optional explicit positive class label in target column (default: auto-infer)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(
        data_path=args.data_path,
        model_name=args.model_name,
        target_col=args.target_col,
        positive_label=args.positive_label,
    )
