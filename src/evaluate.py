import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CM_DIR = RESULTS_DIR / "confusion_matrices"

METRICS_DIR.mkdir(parents=True, exist_ok=True)
CM_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str,
    config_id: str,
    task_name: str = "application"
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier, records metrics and latency,
    and generates confusion matrix visualization.
    """
    print(f"\n[Evaluate] Evaluating {model_name} (Config {config_id}, Task: {task_name})...")

    # Measure inference latency
    start_time = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_sec = time.perf_counter() - start_time
    latency_per_sample_ms = (inference_time_sec / len(X_test)) * 1000

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
    macro_precision = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    macro_recall = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Per-class report
    clf_report = classification_report(
        y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model_name": model_name,
        "config_id": config_id,
        "task": task_name,
        "test_samples": len(y_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "total_inference_time_sec": round(inference_time_sec, 4),
        "latency_per_sample_ms": round(latency_per_sample_ms, 4),
        "per_class": {
            cls_name: {
                "precision": round(clf_report[cls_name]["precision"], 4),
                "recall": round(clf_report[cls_name]["recall"], 4),
                "f1_score": round(clf_report[cls_name]["f1-score"], 4),
                "support": clf_report[cls_name]["support"]
            }
            for cls_name in class_names if cls_name in clf_report
        }
    }

    # Save metrics JSON
    metrics_path = METRICS_DIR / f"{model_name.lower().replace(' ', '_')}_config_{config_id}_{task_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Plot & Save Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title(f"Confusion Matrix: {model_name} (Config {config_id})")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = CM_DIR / f"cm_{model_name.lower().replace(' ', '_')}_config_{config_id}_{task_name}.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()

    print(f"[Evaluate] Results: Accuracy={acc:.4f} | Macro-F1={macro_f1:.4f} | Latency={latency_per_sample_ms:.4f} ms/flow")
    print(f"[Evaluate] Confusion matrix saved to: {cm_path.name}")

    return results

