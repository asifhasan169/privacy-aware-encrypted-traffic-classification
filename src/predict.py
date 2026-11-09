import json
from pathlib import Path
from typing import Dict, Any, Union, List
import pandas as pd
import numpy as np
import joblib

from src.preprocessing import TrafficPreprocessor

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

class EncryptedTrafficPredictor:
    """
    Inference interface conforming to Section 14 of the Project Specification.
    Validates input schema, applies training-fitted scaling, and outputs predicted
    traffic class and confidence.
    """
    def __init__(self, model_key: str = "rf", config_id: str = "A", task: str = "application"):
        self.model_key = model_key.lower()
        self.config_id = config_id.upper()
        self.task = task.lower()

        model_path = MODELS_DIR / f"{self.model_key}_config_{self.config_id}_{self.task}.joblib"
        prep_path = MODELS_DIR / f"preprocessor_config_{self.config_id}_{self.task}.joblib"

        if not model_path.exists() or not prep_path.exists():
            raise FileNotFoundError(
                f"Model or preprocessor artifact not found. Please train first.\n"
                f"Expected: {model_path.name} and {prep_path.name}"
            )

        self.model = joblib.load(model_path)
        self.preprocessor: TrafficPreprocessor = joblib.load(prep_path)
        self.feature_names = self.preprocessor.feature_names
        self.classes = list(self.preprocessor.label_encoder.classes_)

    def predict(self, input_data: Union[pd.DataFrame, Dict[str, Any], List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Predicts classes for one or more traffic flows.

        Args:
            input_data: DataFrame or dictionary/list of dictionaries containing feature values.

        Returns:
            DataFrame with columns: flow_id, predicted_class, confidence, config_id, model_version
        """
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise TypeError("input_data must be a pandas DataFrame, dict, or list of dicts.")

        # Check required features
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"Input is missing required features for Config {self.config_id}: {missing_features}")

        # Extract and transform features
        X_selected = df[self.feature_names]
        X_scaled = self.preprocessor.scaler.transform(X_selected)

        # Predict
        preds_encoded = self.model.predict(X_scaled)
        pred_labels = self.preprocessor.label_encoder.inverse_transform(preds_encoded)

        # Confidence (if supported)
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_scaled)
            confidence = np.max(probs, axis=1).round(4)
        else:
            # For linear SVM or models without direct predict_proba
            confidence = [None] * len(df)

        flow_ids = df["flow_id"].tolist() if "flow_id" in df.columns else [f"flow_{i+1:04d}" for i in range(len(df))]

        results_df = pd.DataFrame({
            "flow_id": flow_ids,
            "predicted_class": pred_labels,
            "confidence": confidence,
            "config_id": self.config_id,
            "model": self.model_key,
            "task": self.task
        })

        return results_df

if __name__ == "__main__":
    print("[Predictor] Inference module loaded.")

