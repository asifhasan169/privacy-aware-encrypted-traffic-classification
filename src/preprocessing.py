from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

from src.feature_configs import get_feature_list

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

class TrafficPreprocessor:
    """
    Handles feature filtering (Configurations A, B, C), label encoding,
    and leakage-free feature scaling.
    """
    def __init__(self, config_id: str = "A"):
        self.config_id = config_id.upper()
        self.feature_names = get_feature_list(self.config_id)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_fitted = False

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits scaler and label encoder on training set only, and transforms.
        """
        # 1. Feature selection according to configuration
        X_selected = X_train[self.feature_names]

        # 2. Fit and transform scaler
        X_scaled = self.scaler.fit_transform(X_selected)

        # 3. Fit and transform label encoder
        y_encoded = self.label_encoder.fit_transform(y_train)

        self.is_fitted = True
        return X_scaled, y_encoded

    def transform(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Transforms test or inference data using previously fitted parameters.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")

        X_selected = X[self.feature_names]
        X_scaled = self.scaler.transform(X_selected)

        y_encoded = None
        if y is not None:
            y_encoded = self.label_encoder.transform(y)

        return X_scaled, y_encoded

    def save(self, file_path: Optional[str] = None) -> str:
        """Saves fitted preprocessor to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        save_path = Path(file_path) if file_path else MODELS_DIR / f"preprocessor_config_{self.config_id}.joblib"
        joblib.dump(self, save_path)
        print(f"[Preprocessor] Saved to: {save_path}")
        return str(save_path)

    @staticmethod
    def load(file_path: str) -> "TrafficPreprocessor":
        """Loads fitted preprocessor from disk."""
        return joblib.load(file_path)


def prepare_train_test_data(
    X: pd.DataFrame,
    y: pd.Series,
    config_id: str = "A",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrafficPreprocessor]:
    """
    Splits data with stratification, fits preprocessor on train, and transforms test.
    """
    print(f"[PrepareData] Splitting dataset (test_size={test_size}, stratify=y, seed={random_state})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    preprocessor = TrafficPreprocessor(config_id=config_id)
    X_train_proc, y_train_proc = preprocessor.fit_transform(X_train, y_train)
    X_test_proc, y_test_proc = preprocessor.transform(X_test, y_test)

    print(f"[PrepareData] Config {config_id}: Train shape = {X_train_proc.shape}, Test shape = {X_test_proc.shape}")
    return X_train_proc, X_test_proc, y_train_proc, y_test_proc, preprocessor

