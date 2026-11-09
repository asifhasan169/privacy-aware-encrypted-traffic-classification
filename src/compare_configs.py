import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.train import run_experiment
from src.feature_configs import load_feature_config

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def run_full_privacy_utility_benchmark(sample_size: int = 40000, task: str = "application"):
    """
    Evaluates models across Configurations A, B, and C to measure the
    Privacy-Utility trade-off (Experiment E1, E2, E3, E4).
    """
    models = ["lr", "rf", "svm", "xgb"]
    configs = ["A", "B", "C"]
    
    records = []

    print("=" * 80)
    print(f"STARTING COMPREHENSIVE PRIVACY-UTILITY BENCHMARK (Task: {task}, Sample: {sample_size})")
    print("=" * 80)

    for cfg in configs:
        cfg_info = load_feature_config(cfg)
        feat_count = cfg_info["feature_count"]

        for m in models:
            res = run_experiment(
                model_key=m,
                config_id=cfg,
                task=task,
                sample_size=sample_size,
                save_model=True
            )
            records.append({
                "Model": res["model_name"],
                "Config": cfg,
                "Features": feat_count,
                "Accuracy": res["accuracy"],
                "Macro_F1": res["macro_f1"],
                "Weighted_F1": res["weighted_f1"],
                "Train_Time_s": res.get("training_time_sec", 0.0),
                "Latency_ms_per_flow": res["latency_per_sample_ms"]
            })

    df_results = pd.DataFrame(records)
    
    # Save CSV
    summary_path = METRICS_DIR / f"privacy_utility_summary_{task}.csv"
    df_results.to_csv(summary_path, index=False)
    
    # Also log to experiment_log.csv
    exp_log_path = DOCS_DIR / "experiment_log.csv"
    df_results.to_csv(exp_log_path, index=False)
    print(f"\n[Benchmark] Saved summary table to: {summary_path}")

    # Plot Privacy-Utility Tradeoff: Macro-F1 vs Retained Feature Count
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("deep", len(models))
    for i, m_name in enumerate(df_results["Model"].unique()):
        sub_df = df_results[df_results["Model"] == m_name].sort_values("Features")
        plt.plot(
            sub_df["Features"],
            sub_df["Macro_F1"],
            marker="o",
            linewidth=2.5,
            markersize=8,
            label=m_name,
            color=palette[i]
        )

    plt.xlabel("Number of Retained Statistical Features (Privacy Proxy)", fontsize=11)
    plt.ylabel("Macro-F1 Score", fontsize=11)
    plt.title(f"Privacy–Utility Trade-off: Macro-F1 vs. Feature Minimization ({task.capitalize()} Task)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(title="Classifier", frameon=True)
    plt.tight_layout()

    tradeoff_plot = FIGURES_DIR / f"privacy_utility_tradeoff_{task}.png"
    plt.savefig(tradeoff_plot, dpi=200)
    plt.close()
    print(f"[Benchmark] Saved trade-off plot to: {tradeoff_plot.name}")

    print("\n=== SUMMARY TABLE ===")
    print(df_results.to_string(index=False))

    return df_results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Privacy-utility trade-off benchmark")
    parser.add_argument("--sample", type=int, default=40000, help="Sample size for benchmark")
    parser.add_argument("--task", type=str, default="application")
    args = parser.parse_args()

    run_full_privacy_utility_benchmark(sample_size=args.sample, task=args.task)

