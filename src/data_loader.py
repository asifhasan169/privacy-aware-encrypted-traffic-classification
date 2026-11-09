import os
from pathlib import Path
from typing import Tuple, Optional, Literal
import pandas as pd

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "archive" / "ll_mit_tw64_complete.csv"

APPLICATION_MAPPING = {
    "scp": "file_transfer",
    "sftp": "file_transfer",
    "rsync": "file_transfer",
    "netflix": "streaming",
    "vimeo": "streaming",
    "youtube": "streaming",
    "voip": "voip_chat",
    "skype-chat": "voip_chat",
    "rdp": "remote_access",
    "ssh": "remote_access",
}

def extract_clean_app_name(raw_label: str) -> str:
    """Strips vpn_ or nonvpn_ prefix from raw label."""
    if raw_label.startswith("nonvpn_"):
        return raw_label[7:]
    elif raw_label.startswith("vpn_"):
        return raw_label[4:]
    return raw_label

def load_vnat_dataset(
    file_path: Optional[str] = None,
    task: Literal["binary", "application", "granular"] = "application",
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads and prepares the VNAT feature dataset.

    Args:
        file_path: Path to the CSV dataset file. Defaults to archive/ll_mit_tw64_complete.csv.
        task: Classification task:
              - 'binary': 'vpn' vs 'nonvpn'
              - 'application': 4 classes ('file_transfer', 'streaming', 'voip_chat', 'remote_access')
              - 'granular': 20 original classes ('nonvpn_scp', etc.)
        sample_size: Optional integer to take a stratified/random sample for rapid testing.
        random_state: Random seed for sampling reproducibility.

    Returns:
        X: Feature DataFrame with 43 columns.
        y: Target Series corresponding to the selected task.
    """
    data_path = Path(file_path) if file_path else DEFAULT_DATASET_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    print(f"[DataLoader] Loading dataset from: {data_path.name}")
    df = pd.read_csv(data_path)

    if "label" not in df.columns:
        raise ValueError("Expected 'label' column in dataset.")

    # Target assignment
    raw_labels = df["label"]
    if task == "binary":
        y = raw_labels.apply(lambda x: "vpn" if x.startswith("vpn_") else "nonvpn")
    elif task == "application":
        clean_apps = raw_labels.apply(extract_clean_app_name)
        y = clean_apps.map(APPLICATION_MAPPING)
        if y.isnull().any():
            unmapped = clean_apps[y.isnull()].unique()
            raise ValueError(f"Unmapped application labels found: {unmapped}")
    elif task == "granular":
        y = raw_labels.copy()
    else:
        raise ValueError(f"Invalid task: {task}. Choose from 'binary', 'application', 'granular'.")

    X = df.drop(columns=["label"])

    if sample_size is not None and sample_size < len(df):
        print(f"[DataLoader] Sub-sampling {sample_size} rows (stratified by target)...")
        from sklearn.model_selection import train_test_split
        X, _, y, _ = train_test_split(
            X, y, train_size=sample_size, stratify=y, random_state=random_state
        )

    print(f"[DataLoader] Loaded {len(X)} samples with {X.shape[1]} raw features.")
    print(f"[DataLoader] Target distribution ({task}):\n{y.value_counts(normalize=True).round(4) * 100}%")

    return X, y

