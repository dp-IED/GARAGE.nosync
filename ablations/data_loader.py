"""
Shared data loader for ablations. Loads train/val/test from the shared dataset
using the same splits as the GDN model.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

DEFAULT_SENSOR_NAMES = [
    "ENGINE_RPM ()",
    "VEHICLE_SPEED ()",
    "THROTTLE ()",
    "ENGINE_LOAD ()",
    "COOLANT_TEMPERATURE ()",
    "INTAKE_MANIFOLD_PRESSURE ()",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()",
]


def load_split(
    data_path: Path,
    split: str,
    limit: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Load a single split (train, val, or test) from the shared dataset.

    Returns dict with: normalized_windows, sensor_labels, window_is_faulty,
    sensor_names, and optionally fault_types if present.
    """
    data_path = Path(data_path)
    npz_path = data_path / f"{split}.npz"
    metadata_path = data_path / f"{split}_metadata.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    normalized_windows = np.array(data["normalized_windows"], dtype=np.float32)
    sensor_labels = np.array(data["sensor_labels"], dtype=np.float32)
    window_is_faulty = np.array(data["window_is_faulty"], dtype=np.float32)

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = DEFAULT_SENSOR_NAMES.copy()

    result = {
        "normalized_windows": normalized_windows,
        "sensor_labels": sensor_labels,
        "window_is_faulty": window_is_faulty,
        "sensor_names": sensor_names,
    }

    if "fault_types" in data:
        result["fault_types"] = data["fault_types"]

    if limit is not None and limit > 0:
        result["normalized_windows"] = result["normalized_windows"][:limit]
        result["sensor_labels"] = result["sensor_labels"][:limit]
        result["window_is_faulty"] = result["window_is_faulty"][:limit]
        if "fault_types" in result:
            result["fault_types"] = result["fault_types"][:limit]

    return result


def load_all_splits(
    data_path: Path,
    limit_test: Optional[int] = None,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load train, val, and test splits. Optionally limit test set size.

    Returns:
        (train_data, val_data, test_data) - each is a dict from load_split
    """
    train_data = load_split(data_path, "train")
    val_data = load_split(data_path, "val")
    test_data = load_split(data_path, "test", limit=limit_test)
    return train_data, val_data, test_data
