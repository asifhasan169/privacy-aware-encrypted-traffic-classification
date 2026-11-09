import json
from pathlib import Path
from typing import List, Dict, Any

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"

def load_feature_config(config_id: str) -> Dict[str, Any]:
    """
    Load feature configuration ('A', 'B', or 'C').
    
    Args:
        config_id: One of 'A', 'B', 'C' (case-insensitive)
        
    Returns:
        dict containing 'name', 'description', 'feature_count', and 'features' (list of str)
    """
    config_id = config_id.upper().strip()
    file_path = CONFIG_DIR / f"features_{config_id}.json"
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)
    return config_data

def get_feature_list(config_id: str) -> List[str]:
    """Returns the list of column names for a given configuration ('A', 'B', 'C')."""
    return load_feature_config(config_id)["features"]

