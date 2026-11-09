from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from src.preprocessing import TrafficPreprocessor
from src.data_loader import load_vnat_dataset

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"
METRICS_DIR = Path(__file__).resolve().parent.parent / "results" / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

def generate_feature_importance(
    model_key: str = "rf",
    config_id: str = "A",
    task: str = "application",
    top_n: int = 15
):
    """
    Extracts and visualizes tree-based feature importance.
    """
    model_path = MODELS_DIR / f"{model_key}_config_{config_id}_{task}.joblib"
    prep_path = MODELS_DIR / f"preprocessor_config_{config_id}_{task}.joblib"

    if not model_path.exists() or not prep_path.exists():
        raise FileNotFoundError(f"Model or preprocessor artifact not found for {model_key} config {config_id}.")

    model = joblib.load(model_path)
    preprocessor: TrafficPreprocessor = joblib.load(prep_path)
    feature_names = preprocessor.feature_names

    if not hasattr(model, "feature_importances_"):
        print(f"[Explain] Model {model_key} does not have feature_importances_ attribute.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fi_df = pd.DataFrame({
        "feature": [feature_names[i] for i in indices],
        "importance": importances[indices]
    })

    # Save CSV
    csv_path = METRICS_DIR / f"feature_importance_{model_key}_config_{config_id}_{task}.csv"
    fi_df.to_csv(csv_path, index=False)
    print(f"[Explain] Saved feature importance rankings to {csv_path.name}")

    # Plot Top N
    plt.figure(figsize=(10, 6))
    top_df = fi_df.head(top_n).iloc[::-1]
    plt.barh(top_df["feature"], top_df["importance"], color="#2b5c8f")
    plt.xlabel("Gini / Gain Importance")
    plt.title(f"Top {top_n} Most Important Features: {model_key.upper()} (Config {config_id})")
    plt.tight_layout()

    plot_path = FIGURES_DIR / f"feature_importance_{model_key}_config_{config_id}_{task}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Explain] Saved feature importance plot to {plot_path.name}")

def generate_shap_analysis(
    model_key: str = "rf",
    config_id: str = "A",
    task: str = "application",
    sample_size: int = 300
):
    """
    Computes SHAP values using TreeExplainer on a representative background sample.
    """
    model_path = MODELS_DIR / f"{model_key}_config_{config_id}_{task}.joblib"
    prep_path = MODELS_DIR / f"preprocessor_config_{config_id}_{task}.joblib"

    model = joblib.load(model_path)
    preprocessor: TrafficPreprocessor = joblib.load(prep_path)

    print(f"[SHAP] Loading representative sample ({sample_size} flows) for SHAP computation...")
    X, y = load_vnat_dataset(task=task, sample_size=sample_size)
    X_scaled, _ = preprocessor.transform(X)

    feature_names = preprocessor.feature_names
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # For multi-class, shap_values is a list of arrays (or 3D array)
    plt.figure(figsize=(11, 7))
    if isinstance(shap_values, list):
        # Multi-class: plot summary for each class or summary bar
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=feature_names,
            class_names=list(preprocessor.label_encoder.classes_),
            show=False,
            max_display=15
        )
    else:
        shap.summary_plot(
            shap_values,
            X_scaled,
            feature_names=feature_names,
            show=False,
            max_display=15
        )

    plt.title(f"SHAP Summary (Feature Impact): {model_key.upper()} (Config {config_id})", pad=20)
    plt.tight_layout()

    shap_path = FIGURES_DIR / f"shap_summary_{model_key}_config_{config_id}_{task}.png"
    plt.savefig(shap_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved SHAP summary plot to {shap_path.name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Explain trained models")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xgb"])
    parser.add_argument("--config", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--task", type=str, default="application")
    parser.add_argument("--shap", action="store_true", help="Run SHAP analysis")
    args = parser.parse_args()

    generate_feature_importance(model_key=args.model, config_id=args.config, task=args.task)
    if args.shap:
        generate_shap_analysis(model_key=args.model, config_id=args.config, task=args.task)

