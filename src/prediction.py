"""
Prediction Module for Customer Churn
Loads trained model and makes predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Union, Dict, Tuple

class ChurnPredictor:
    """
    Churn prediction model wrapper
    Handles model loading, preprocessing, and predictions
    """
    
    def __init__(self, model_path='models/churn_model_best.pkl', 
                 preprocessing_path='models/churn_model_best_preprocessing.pkl'):
        """
        Initialize predictor with trained model
        
        Args:
            model_path: Path to saved model
            preprocessing_path: Path to preprocessing info
        """
        self.model_path = model_path
        self.preprocessing_path = preprocessing_path
        self.model = None
        self.preprocessing_info = None
        self.load_model()
    
    def load_model(self):
        """Load model and preprocessing information"""
        try:
            self.model = joblib.load(self.model_path)
            self.preprocessing_info = joblib.load(self.preprocessing_path)
            print(f"[OK] Model loaded successfully from {self.model_path}")
        except FileNotFoundError:
            print(f"[ERROR] Model not found at {self.model_path}")
            print("   Please train the model first using: python src/model_training.py")
            raise

    def _get_target_mapping(self):
        """Return target mapping metadata with backward-compatible defaults."""
        target_info = self.preprocessing_info.get('target_info', {})
        class_mapping = target_info.get('class_mapping', {0: 'No', 1: 'Yes'})
        positive_class_index = int(target_info.get('positive_class_index', 1))
        return class_mapping, positive_class_index

    @staticmethod
    def _normalize_text(value) -> str:
        """Normalize text values for robust categorical matching."""
        if pd.isna(value):
            return ""
        return str(value).replace("\u00a0", " ").strip()

    @staticmethod
    def _to_float_if_possible(value):
        """Convert value to float when possible; otherwise return None."""
        try:
            if pd.isna(value):
                return None
            return float(str(value).strip())
        except Exception:
            return None

    @staticmethod
    def _prepare_estimator_input(estimator, frame: pd.DataFrame):
        """Return estimator input in the format expected by the fitted object."""
        return frame if hasattr(estimator, 'feature_names_in_') else frame.values

    def _prepare_encoder_cache(self, encoder):
        """Precompute normalized and numeric lookup structures for an encoder."""
        classes = list(encoder.classes_)
        normalized_to_original = {}
        numeric_classes = []

        for original in classes:
            normalized = self._normalize_text(original)
            if normalized not in normalized_to_original:
                normalized_to_original[normalized] = original

            c_num = self._to_float_if_possible(normalized)
            if c_num is not None:
                numeric_classes.append((c_num, original))

        return {
            "classes": classes,
            "normalized_to_original": normalized_to_original,
            "numeric_classes": numeric_classes,
        }

    def _map_value_to_encoder_class(self, value, encoder_cache):
        """Map raw input value to the closest known encoder class."""
        classes = encoder_cache["classes"]
        if not classes:
            return value

        value_norm = self._normalize_text(value)
        normalized_to_original = encoder_cache["normalized_to_original"]
        if value_norm in normalized_to_original:
            return normalized_to_original[value_norm]

        value_num = self._to_float_if_possible(value_norm)
        numeric_classes = encoder_cache["numeric_classes"]
        if value_num is not None and numeric_classes:
            _, nearest_label = min(numeric_classes, key=lambda item: abs(item[0] - value_num))
            return nearest_label

        return classes[0]
    
    def preprocess_input(self, data: Union[pd.DataFrame, Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess input data for prediction
        
        Args:
            data: Input dataframe or dictionary
            
        Returns:
            tuple: (original_data, preprocessed_data)
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Input must be a DataFrame or dictionary")
        
        # Work with a copy
        df = data.copy()
        original_df = df.copy()

        feature_names = self.preprocessing_info['feature_names']
        label_encoders = self.preprocessing_info['label_encoders']
        numeric_defaults = self.preprocessing_info.get('numeric_defaults', {})

        # Ensure all required columns exist
        for col in feature_names:
            if col in df.columns:
                continue
            if col in label_encoders:
                df[col] = str(label_encoders[col].classes_[0])
            else:
                df[col] = 0

        # Coerce numeric columns and fill missing values
        numeric_cols = [col for col in feature_names if col not in label_encoders]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                default_val = numeric_defaults.get(col)
                if default_val is None:
                    default_val = float(df[col].median()) if df[col].notna().any() else 0.0
                median_val = float(default_val)
                df[col] = df[col].fillna(median_val)
        
        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in df.columns:
                encoder_cache = self._prepare_encoder_cache(encoder)
                mapped_values = df[col].apply(
                    lambda value: self._map_value_to_encoder_class(value, encoder_cache)
                )
                df[col] = encoder.transform(mapped_values)
        
        # Reorder columns to match training data
        df = df[feature_names]
        
        # Scale features
        scaler = self.preprocessing_info['scaler']
        df_scaled = scaler.transform(self._prepare_estimator_input(scaler, df))
        df_scaled = pd.DataFrame(df_scaled, columns=feature_names)
        
        return original_df, df_scaled
    
    def predict(self, data: Union[pd.DataFrame, Dict]) -> Dict:
        """
        Make prediction on input data
        
        Args:
            data: Customer data to predict
            
        Returns:
            dict: Prediction results with probabilities
        """
        original_df, X_scaled = self.preprocess_input(data)
        
        # Get predictions and probabilities
        model_input = self._prepare_estimator_input(self.model, X_scaled)
        predictions = self.model.predict(model_input)
        probabilities = self.model.predict_proba(model_input)

        class_mapping, positive_class_index = self._get_target_mapping()
        
        results = []
        for idx in range(len(X_scaled)):
            pred_value = int(predictions[idx])
            predicted_label = class_mapping.get(pred_value, str(pred_value))
            positive_prob = float(probabilities[idx][positive_class_index])
            
            results.append({
                'prediction': predicted_label,
                'positive_probability': round(positive_prob, 4),
                'negative_probability': round(1 - positive_prob, 4),
                'risk_level': self._get_risk_level(positive_prob)
            })

            # Backward-compatible fields for current app views
            results[-1]['churn_probability'] = results[-1]['positive_probability']
            results[-1]['no_churn_probability'] = results[-1]['negative_probability']
            results[-1]['churn_prediction'] = results[-1]['prediction']
            results[-1]['churn_risk_level'] = results[-1]['risk_level']
        
        return results if len(results) > 1 else results[0]
    
    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """
        Determine risk level based on churn probability
        
        Args:
            probability: Churn probability
            
        Returns:
            str: Risk level (Low, Medium, High, Very High)
        """
        if probability < 0.25:
            return "Low Risk"
        elif probability < 0.5:
            return "Medium Risk"
        elif probability < 0.75:
            return "High Risk"
        else:
            return "Very High Risk"
    
    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make predictions on multiple customers
        
        Args:
            data: DataFrame with multiple customers
            
        Returns:
            DataFrame with predictions and probabilities
        """
        predictions = self.predict(data)
        
        # Convert list of dicts to DataFrame
        if isinstance(predictions, list):
            results_df = pd.DataFrame(predictions)
        else:
            results_df = pd.DataFrame([predictions])
        
        # Add original customer data
        return pd.concat([data.reset_index(drop=True), results_df], axis=1)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model
        
        Returns:
            DataFrame: Feature importance scores
        """
        if hasattr(self.model, 'feature_importances_'):
            feature_names = self.preprocessing_info['feature_names']
            importance = self.model.feature_importances_
            
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            return None


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Example 1: Predict single customer
    print("\n" + "="*70)
    print("[SAMPLE] EXAMPLE: PREDICT SINGLE CUSTOMER")
    print("="*70)
    
    customer_data = {
        'tenure': 24,
        'MonthlyCharges': 65.5,
        'TotalCharges': 1567.4,
        'Contract': 'Month-to-month',
        'InternetService': 'Fiber optic',
        'OnlineSecurity': 'No',
        'TechSupport': 'No',
        'PhoneService': 'Yes',
        'gender': 'Male'
    }
    
    try:
        result = predictor.predict(customer_data)
        print(f"\n[OK] Prediction Result:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Positive Probability: {result['positive_probability']:.2%}")
        print(f"   Risk Level: {result['risk_level']}")
    except Exception as e:
        print(f"Note: Example requires training data structure. Error: {e}")
    
    # Example 2: Feature importance
    print("\n" + "="*70)
    print("[CHART] TOP 10 IMPORTANT FEATURES")
    print("="*70)
    
    importance_df = predictor.get_feature_importance()
    if importance_df is not None:
        print(importance_df.head(10).to_string(index=False))
    else:
        print("Feature importance not available for this model type.")
