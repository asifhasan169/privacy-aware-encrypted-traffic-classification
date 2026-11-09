import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from src.data_loader import load_vnat_dataset
from src.preprocessing import prepare_train_test_data
from src.evaluate import evaluate_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_model(model_key: str, random_state: int = 42, num_classes: int = 4):
    """
    Factory function for the four project models optimized for CPU execution.
    """
    key = model_key.lower().strip()
    if key in ["lr", "logistic_regression"]:
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ), "Logistic Regression"
    elif key in ["rf", "random_forest"]:
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ), "Random Forest"
    elif key in ["svm", "linearsvc"]:
        return LinearSVC(
            dual=False,
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state
        ), "Linear SVM"
    elif key in ["xgb", "xgboost"]:
        objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
        eval_metric = "logloss" if num_classes == 2 else "mlogloss"
        return XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective=objective,
            eval_metric=eval_metric,
            random_state=random_state,
            n_jobs=-1
        ), "XGBoost"
    else:
        raise ValueError(f"Unknown model key: {model_key}. Choose from 'lr', 'rf', 'svm', 'xgb'.")

def run_experiment(
    model_key: str,
    config_id: str = "A",
    task: str = "application",
    sample_size: Optional[int] = None,
    save_model: bool = True
) -> Dict[str, Any]:
    """
    Executes a complete training and evaluation run for a given model and feature configuration.
    """
    print("=" * 70)
    print(f"RUNNING EXPERIMENT: Model={model_key} | Config={config_id} | Task={task}")
    print("=" * 70)

    # 1. Load data
    X, y = load_vnat_dataset(task=task, sample_size=sample_size)

    # 2. Preprocess & Split
    X_train, X_test, y_train, y_test, preprocessor = prepare_train_test_data(
        X, y, config_id=config_id
    )

    # 3. Instantiate model
    num_classes = len(preprocessor.label_encoder.classes_)
    model, display_name = get_model(model_key, num_classes=num_classes)

    # 4. Train model
    print(f"[Train] Training {display_name} on {len(X_train)} samples...")
    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_sec = time.perf_counter() - start_time
    print(f"[Train] Training completed in {train_time_sec:.2f} seconds.")

    # 5. Evaluate
    class_names = list(preprocessor.label_encoder.classes_)
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        model_name=display_name,
        config_id=config_id,
        task_name=task
    )
    metrics["training_time_sec"] = round(train_time_sec, 2)

    # 6. Save model and preprocessor artifacts
    if save_model:
        model_filename = f"{model_key.lower()}_config_{config_id}_{task}.joblib"
        model_save_path = MODELS_DIR / model_filename
        joblib.dump(model, model_save_path)
        preprocessor.save(MODELS_DIR / f"preprocessor_config_{config_id}_{task}.joblib")
        print(f"[Train] Saved model checkpoint to: {model_save_path.name}")

    return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train encrypted traffic classifier")
    parser.add_argument("--model", type=str, default="rf", choices=["lr", "rf", "svm", "xgb", "all"])
    parser.add_argument("--config", type=str, default="A", choices=["A", "B", "C"])
    parser.add_argument("--task", type=str, default="application", choices=["binary", "application", "granular"])
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for rapid runs")
    args = parser.parse_args()

    models_to_run = ["lr", "rf", "svm", "xgb"] if args.model == "all" else [args.model]
    for m in models_to_run:
        run_experiment(model_key=m, config_id=args.config, task=args.task, sample_size=args.sample)

