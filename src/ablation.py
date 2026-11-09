from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_loader import load_vnat_dataset
from src.preprocessing import prepare_train_test_data
from src.train import get_model
from src.evaluate import evaluate_model
from src.feature_configs import get_feature_list

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Define feature families based on feature_dictionary.csv
TIMING_FEATURES = [
    "FIAT", "BIAT", "MIAT", "Min_FIAT", "Max_FIAT", "Std_FIAT",
    "Var_coeff_FIAT", "Min_BIAT", "Max_BIAT", "Std_BIAT", "Var_coeff_BIAT"
]
PACKET_SIZE_FEATURES = [
    "Forward_abs_mean", "Forward_std_dev", "Backward_abs_mean", "Backward_std_dev",
    "Forward_entropy", "Backward_entropy", "Forward_med", "Forward_mod",
    "Forward_rng", "Forward_iqr", "Forward_var_coeff", "Forward_rms",
    "Backward_med", "Backward_mod", "Backward_rng", "Backward_iqr",
    "Backward_var_coeff", "Backward_rms"
]
WAVELET_FEATURES = [
    "fw_c1", "fw_c2", "fw_c3", "fw_c4", "fw_c5", "fw_c6", "fw_c7",
    "bw_c1", "bw_c2", "bw_c3", "bw_c4", "bw_c5", "bw_c6", "bw_c7"
]

def run_ablation_study(
    model_key: str = "rf",
    task: str = "application",
    sample_size: int = 25000
) -> pd.DataFrame:
    """
    Experiment E5: Evaluates the impact of removing individual feature families
    (Timing, Packet Sizes, DWT Wavelets) on classification performance.
    """
    print("=" * 70)
    print(f"RUNNING FEATURE ABLATION STUDY: Model={model_key} | Task={task}")
    print("=" * 70)

    # 1. Load data
    X_full, y = load_vnat_dataset(task=task, sample_size=sample_size)
    all_features = get_feature_list("A")

    ablation_conditions = {
        "All Features (Baseline Config A)": all_features,
        "Ablate Timing (No Inter-Arrival)": [f for f in all_features if f not in TIMING_FEATURES],
        "Ablate Packet Sizes (No Size Moments)": [f for f in all_features if f not in PACKET_SIZE_FEATURES],
        "Ablate Wavelets (No DWT Sub-bands)": [f for f in all_features if f not in WAVELET_FEATURES],
    }

    results = []

    for condition_name, feature_subset in ablation_conditions.items():
        print(f"\n[Ablation] Testing condition: {condition_name} ({len(feature_subset)} features)...")
        
        # Prepare custom feature subset
        X_sub = X_full[feature_subset]
        
        # Split & scale manually to handle custom subset
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_sub, y, test_size=0.2, random_state=42, stratify=y
        )
        
        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_te_scaled = scaler.transform(X_te)
        
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        y_te_enc = le.transform(y_te)

        # Train model
        model, display_name = get_model(model_key, num_classes=len(le.classes_))
        model.fit(X_tr_scaled, y_tr_enc)

        # Evaluate
        eval_res = evaluate_model(
            model=model,
            X_test=X_te_scaled,
            y_test=y_te_enc,
            class_names=list(le.classes_),
            model_name=f"{display_name} ({condition_name})",
            config_id="Ablation",
            task_name=task
        )

        results.append({
            "Condition": condition_name,
            "Features_Retained": len(feature_subset),
            "Features_Removed": len(all_features) - len(feature_subset),
            "Accuracy": eval_res["accuracy"],
            "Macro_F1": eval_res["macro_f1"],
            "Weighted_F1": eval_res["weighted_f1"],
            "Latency_ms_per_flow": eval_res["latency_per_sample_ms"]
        })

    df_ablation = pd.DataFrame(results)

    # Save CSV
    csv_path = METRICS_DIR / f"ablation_study_{model_key}_{task}.csv"
    df_ablation.to_csv(csv_path, index=False)
    print(f"\n[Ablation] Saved results to: {csv_path.name}")

    # Plot Ablation Comparison
    plt.figure(figsize=(10, 5))
    bars = plt.barh(df_ablation["Condition"], df_ablation["Macro_F1"], color="#1f77b4")
    plt.xlim(0.85, 1.0)
    plt.xlabel("Macro-F1 Score")
    plt.title(f"Feature Family Ablation Study: {model_key.upper()} ({task.capitalize()} Task)")
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2, f"{width:.4f}",
                 va='center', ha='left', fontsize=9, fontweight='bold')
    plt.tight_layout()

    plot_path = FIGURES_DIR / f"ablation_study_{model_key}_{task}.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Ablation] Saved plot to: {plot_path.name}")

    print("\n=== ABLATION RESULTS ===")
    print(df_ablation.to_string(index=False))

    return df_ablation

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Feature family ablation study")
    parser.add_argument("--model", type=str, default="rf", choices=["rf", "xgb", "lr"])
    parser.add_argument("--task", type=str, default="application")
    parser.add_argument("--sample", type=int, default=25000)
    args = parser.parse_args()

    run_ablation_study(model_key=args.model, task=args.task, sample_size=args.sample)

