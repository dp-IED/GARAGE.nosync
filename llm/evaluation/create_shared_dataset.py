"""
Create shared evaluation dataset for comparing LLM-only vs GDN->KG methods.

1. Loads raw OBD data from CSV files
2. Preprocesses using the SAME pipeline as GDN training (gdn.ipynb)
3. Creates BOTH normalized windows (for GDN->KG) and unnormalized windows (for LLM)
4. Optionally injects faults with sensor-level ground truth labels
5. Computes statistical features for each window
6. Saves standardized dataset for evaluation
7. Runs GDN anomaly detection on windows and loads results into Neo4j

IMPORTANT: Both methods evaluate on IDENTICAL windows with the SAME ground truth,
ensuring fair comparison.

Usage:
    python llm/evaluation/create_shared_dataset.py \
        --raw-data-path data/carOBD/obdiidata \
        --output-dir llm/evaluation/shared_dataset \
        --split test \
        --fault-percentage 0.3 \
        --max-windows 1000 \
        --gdn-model-path anomaly-detection/best_center_loss_gdn.pt \
        --neo4j-uri bolt://127.0.0.1:7687 \
        --neo4j-user neo4j \
        --neo4j-password password

Output files:
    - {split}.npz: Dataset arrays (normalized/unnormalized windows, labels, etc.)
    - {split}_metadata.json: Dataset info, sensor names, window counts

Neo4j Database:
    - Loads windows, sensors, anomaly scores, correlations, and temporal relationships
    - Clears existing data before loading (use --skip-neo4j to disable)

Dataset structure (all in one .npz file):
    - normalized_windows: (N, 300, 8) - normalized [0,1] for GDN
    - unnormalized_windows: (N, 300, 8) - real sensor values for LLM
    - sensor_labels: (N, 8) - ground truth faulty sensors (binary)
    - window_labels: (N,) - window indices (0, 1, 2, ..., N-1)
    - fault_types: (N,) - fault type strings (VSS_DROPOUT, etc.)
    - statistical_features: (N, 8, 9) - statistical features per sensor

Neo4j Three-Layer Architecture:
    - Layer 1: Graph structure (correlations, violations, subsystems)
    - Layer 2: Statistical summaries (mean, std, min, max, variance, num_zeros, trend, quartiles on HAS_READING)
    - Layer 3: Raw time-series (Reading nodes with subsampled values, 15 points per window)
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import torch
from scipy import stats
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from training.train_stage1 import (
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
)
from training.train_stage2_clean import build_clean_windows
from training.fault_injection import inject_faults_with_sensor_labels

FORECAST_HORIZON = 1  # Default forecast horizon

from kg.create_kg import (
    load_gdn_model,
    predict_anomalies,
    extract_sensor_embeddings,
    compute_adjacency_matrix,
    KnowledgeGraph,
)

try:
    from llm.kag.graphdb import Neo4jLoader
except ImportError:
    Neo4jLoader = None


def sensor_labels_to_window_label(sensor_labels: np.ndarray) -> int:
    """
    Convert sensor-level labels to window-level sensor-indexed label.

    Args:
        sensor_labels: (num_sensors,) binary array - which sensors are faulty

    Returns:
        int: 0 if no fault, 1-8 if fault detected (1-indexed sensor index)
             Uses the first faulty sensor index (primary sensor)
    """
    faulty_indices = np.where(sensor_labels > 0)[0]
    if len(faulty_indices) == 0:
        return 0
    # Return first faulty sensor index + 1 (1-indexed)
    return int(faulty_indices[0]) + 1


def compute_statistical_features(window_data: np.ndarray) -> np.ndarray:
    """
    Compute statistical features for each sensor in a window.

    Args:
        window_data: (window_size, num_sensors) array

    Returns:
        (num_sensors, 9) array with features: mean, std, min, max, range,
        median, mode, skewness, kurtosis
    """
    num_sensors = window_data.shape[1]
    features = np.zeros((num_sensors, 9))

    for i in range(num_sensors):
        sensor_values = window_data[:, i]
        features[i, 0] = np.mean(sensor_values)
        features[i, 1] = np.std(sensor_values)
        features[i, 2] = np.min(sensor_values)
        features[i, 3] = np.max(sensor_values)
        features[i, 4] = features[i, 3] - features[i, 2]
        features[i, 5] = np.median(sensor_values)

        try:
            mode_result = stats.mode(np.round(sensor_values, 2))
            features[i, 6] = (
                mode_result.mode[0] if len(mode_result.mode) > 0 else features[i, 0]
            )
        except Exception:
            features[i, 6] = features[i, 0]  # fallback to mean

        if features[i, 1] > 0:
            features[i, 7] = stats.skew(sensor_values)
        else:
            features[i, 7] = 0.0

        if features[i, 1] > 0:
            features[i, 8] = stats.kurtosis(sensor_values)
        else:
            features[i, 8] = 0.0

    return features


def create_shared_evaluation_dataset(
    raw_data_path: str,
    output_dir: str,
    split: str = "test",
    fault_percentage: float = 0.3,
    max_windows: Optional[int] = None,
    random_state: int = 42,
    gdn_model_path: Optional[str] = None,
    neo4j_uri: str = "bolt://127.0.0.1:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "password",
    skip_neo4j: bool = False,
    neo4j_window_subset: Optional[np.ndarray] = None,
    top_k_correlations: int = 10,
) -> Dict:
    """
    Create shared dataset by loading raw OBD data and preprocessing through GDN pipeline.

    Args:
        raw_data_path: Path to directory containing raw OBD CSV files
        output_dir: Directory to save the dataset
        split: 'train', 'val', or 'test'
        fault_percentage: Percentage of windows to inject faults into (0.0-1.0)
        max_windows: Maximum number of windows to include (None for all)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary with dataset info
    """
    print("Creating Shared Evaluation Dataset")
    print(f"Raw data path: {raw_data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Split: {split}")
    print(f"Fault percentage: {fault_percentage}")
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    print("-" * 80)
    print("1. Loading raw OBD data...")
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.exists():
        raise FileNotFoundError(f"Data path not found: {raw_data_path}")

    df_list = []
    csv_files = list(raw_data_path.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files")

    for file in csv_files:
        df = pd.read_csv(file, index_col=False)
        df["drive_id"] = file.stem
        df_list.append(df)
        print(f"  Loaded {file.name}: {df.shape}")

    if len(df_list) == 0:
        raise ValueError(f"No CSV files found in {raw_data_path}")

    data = pd.concat(df_list, ignore_index=True)
    print(f"\nTotal samples: {len(data):,}")
    print(f"Unique drives: {data['drive_id'].nunique()}")

    cols_to_drop = [
        "WARM_UPS_SINCE_CODES_CLEARED ()",
        "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in data.columns]
    if cols_to_drop:
        data = data.drop(columns=cols_to_drop)
    print("-" * 80)
    print("\n2. Preprocessing data...")
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=["drive_id"]
    )
    data = remove_zero_variance_columns(data, exclude_cols=["drive_id"])
    data = downsample(
        data, time_col=TIME_COL, source_file_col="drive_id", downsample_factor=1
    )
    data = filter_long_drives(data, min_length=WINDOW_SIZE + FORECAST_HORIZON)
    data = add_cross_channel_features(data)
    print("  Added cross-channel features")
    data = data.sort_values(["drive_id", TIME_COL]).reset_index(drop=True)
    missing_sensors = [s for s in SENSOR_COLS if s not in data.columns]
    if missing_sensors:
        print(f"  Warning: Missing sensors: {missing_sensors}")
        SENSOR_COLS_AVAILABLE = [s for s in SENSOR_COLS if s in data.columns]
    else:
        SENSOR_COLS_AVAILABLE = SENSOR_COLS

    print(f"  Using {len(SENSOR_COLS_AVAILABLE)} sensors: {SENSOR_COLS_AVAILABLE}")
    print("-" * 80)
    print("3. Creating windows...")
    X_normalized, y_targets, scaler = build_clean_windows(
        data, SENSOR_COLS_AVAILABLE, ID_COL, TIME_COL, WINDOW_SIZE, FORECAST_HORIZON
    )

    print(f"  Created {len(X_normalized)} windows")
    print(f"  Window shape: {X_normalized.shape}")

    X_normalized = X_normalized.numpy()

    print("\n4. Creating unnormalized windows...")
    data_unnorm = data.copy()
    data_unnorm = data_unnorm.sort_values([ID_COL, TIME_COL])

    X_unnormalized_list = []
    for drive_id, group in data_unnorm.groupby(ID_COL):
        values = group[SENSOR_COLS_AVAILABLE].values
        T_, num_sensors = values.shape
        if T_ <= WINDOW_SIZE + FORECAST_HORIZON:
            continue

        for t in range(T_ - WINDOW_SIZE - FORECAST_HORIZON + 1):
            X_window = values[t : t + WINDOW_SIZE]
            X_unnormalized_list.append(X_window)

    X_unnormalized = np.stack(X_unnormalized_list)

    # Ensure same number of windows
    min_windows = min(len(X_normalized), len(X_unnormalized))
    X_normalized = X_normalized[:min_windows]
    X_unnormalized = X_unnormalized[:min_windows]

    print(f"  Created {len(X_unnormalized)} unnormalized windows")

    print("-" * 80)
    if fault_percentage > 0:
        print(f"5. Injecting faults ({fault_percentage * 100:.1f}% of windows)...")
        X_normalized_tensor = torch.tensor(X_normalized, dtype=torch.float32)
        y_targets_tensor = torch.tensor(
            y_targets.numpy()[:min_windows], dtype=torch.float32
        )

        X_faulty, _, sensor_labels, window_labels, fault_types_injected = (
            inject_faults_with_sensor_labels(
                X_normalized_tensor,
                y_targets_tensor,
                SENSOR_COLS_AVAILABLE,
                fault_percentage=fault_percentage,
                random_state=random_state,
                use_stratified=True,
            )
        )

        X_normalized = X_faulty.numpy()
        sensor_labels = sensor_labels.numpy()
        window_labels_binary = window_labels.numpy()

        window_labels = np.arange(len(sensor_labels), dtype=np.int64)

        print(f"  Injected faults into {(window_labels_binary > 0).sum()} windows")
        print(f"  Window indices: 0 to {len(window_labels) - 1}")
    else:
        print("\n5. No faults injected (fault_percentage=0)")
        sensor_labels = np.zeros(
            (len(X_normalized), len(SENSOR_COLS_AVAILABLE)), dtype=np.float32
        )
        window_labels = np.arange(len(X_normalized), dtype=np.int64)

    print("-" * 80)
    print("6. Computing statistical features...")
    print("\n6. Computing statistical features...")
    statistical_features = np.zeros(
        (len(X_unnormalized), len(SENSOR_COLS_AVAILABLE), 9)
    )
    for i in range(len(X_unnormalized)):
        statistical_features[i] = compute_statistical_features(X_unnormalized[i])

    print("-" * 80)
    print("7. Creating fault types...")
    if fault_percentage > 0:
        fault_types = np.array(
            [ft if ft != "normal" else None for ft in fault_types_injected],
            dtype=object,
        )
        window_labels_binary_check = window_labels_binary
    else:
        fault_types = np.array([None] * len(X_normalized), dtype=object)
        window_labels_binary_check = (sensor_labels.sum(axis=1) > 0).astype(np.int64)

    print("-" * 80)
    print("8. Limiting windows if requested...")
    if max_windows is not None and len(X_normalized) > max_windows:
        indices = np.random.choice(len(X_normalized), max_windows, replace=False)
        indices = np.sort(indices)
        X_normalized = X_normalized[indices]
        X_unnormalized = X_unnormalized[indices]
        sensor_labels = sensor_labels[indices]
        window_labels = window_labels[indices]
        fault_types = fault_types[indices]
        statistical_features = statistical_features[indices]
        print(f"\n7. Limited to {max_windows} windows")

    print("-" * 80)
    print("9. Saving dataset...")
    print("\n8. Saving dataset...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{split}.npz"
    np.savez_compressed(
        npz_path,
        normalized_windows=X_normalized.astype(np.float32),
        unnormalized_windows=X_unnormalized.astype(np.float32),
        sensor_labels=sensor_labels.astype(np.float32),
        window_labels=window_labels.astype(np.int64),
        fault_types=fault_types,
        statistical_features=statistical_features.astype(np.float32),
    )
    print(f"  ✓ Saved .npz file: {npz_path}")

    json_path = output_dir / f"{split}_metadata.json"
    metadata = {
        "dataset_info": {
            "name": "shared_evaluation_dataset",
            "split": split,
            "num_windows": int(len(X_normalized)),
            "window_size": int(WINDOW_SIZE),
            "num_sensors": int(len(SENSOR_COLS_AVAILABLE)),
            "sensor_names": SENSOR_COLS_AVAILABLE,
            "fault_percentage": fault_percentage,
            "created_at": datetime.now().isoformat(),
        },
        "sensor_names": SENSOR_COLS_AVAILABLE,
        "metadata": {
            "window_ids": list(range(len(X_normalized))),
            "num_faulty_windows": int((sensor_labels.sum(axis=1) > 0).sum()),
            "num_normal_windows": int((sensor_labels.sum(axis=1) == 0).sum()),
        },
    }
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata JSON: {json_path}")

    print("\n✓ Dataset created successfully!")
    print(f"  Windows: {len(X_normalized)}")
    num_faulty = int((sensor_labels.sum(axis=1) > 0).sum())
    print(f"  Faulty windows: {num_faulty}")
    print(f"  Normal windows: {len(X_normalized) - num_faulty}")

    print("-" * 80)
    print("10. Running GDN anomaly detection and loading to Neo4j...")
    if not skip_neo4j and gdn_model_path is not None:
        print("\n10. Running GDN anomaly detection and loading to Neo4j...")

        try:
            model_path = Path(gdn_model_path)
            if not model_path.is_absolute():
                model_path = project_root / model_path

            if not model_path.exists():
                print(f"  ⚠️  Warning: GDN model not found at {model_path}")
                print("  Skipping Neo4j loading...")
            else:
                print(f"  Loading GDN model from {model_path}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model, metadata = load_gdn_model(str(model_path), device)
                sensor_names = metadata.get("sensor_names", SENSOR_COLS_AVAILABLE)

                print(f"  ✓ GDN model loaded (device: {device})")
                print("  Processing windows through GDN...")
                gdn_predictions = predict_anomalies(
                    model,
                    X_normalized,
                    batch_size=32,
                    device=device,
                )
                sensor_embeddings = extract_sensor_embeddings(model)
                adjacency_matrix = compute_adjacency_matrix(sensor_embeddings)

                print("  ✓ GDN processing completed")

                print("  Building Knowledge Graph...")
                kg = KnowledgeGraph(
                    sensor_names=sensor_names,
                    sensor_embeddings=sensor_embeddings,
                    adjacency_matrix=adjacency_matrix,
                )
                kg.construct(
                    X_windows=X_normalized,
                    gdn_predictions=gdn_predictions,
                    X_windows_unnormalized=X_unnormalized,
                )

                print(
                    f"  ✓ Knowledge Graph built (nodes: {kg.number_of_nodes()}, edges: {kg.number_of_edges()})"
                )

                # Load into Neo4j (optional - requires llm.kag.graphdb)
                if Neo4jLoader is not None:
                    print("  Loading into Neo4j...")
                    try:
                        loader = Neo4jLoader(
                            uri=neo4j_uri, user=neo4j_user, password=neo4j_password
                        )
                        loader.clear_database()
                        loader.create_schema()
                        window_labels_binary = (sensor_labels.sum(axis=1) > 0).astype(
                            np.int64
                        )
                        loader.load_from_kg_builder(
                            kg,
                            window_labels=window_labels_binary,
                            sensor_labels=sensor_labels,
                            window_indices_subset=neo4j_window_subset,
                            top_k_correlations=top_k_correlations,
                        )
                        print("  ✓ Neo4j loading completed")
                    except Exception as e:
                        print(f"  ⚠️  Warning: Failed to load into Neo4j: {e}")
                        print("  Continuing without Neo4j...")
                else:
                    print("  Skipping Neo4j (llm.kag.graphdb not available)")

        except Exception as e:
            print(f"  ⚠️  Warning: Error during GDN/Neo4j processing: {e}")
            print("  Continuing without Neo4j...")
            import traceback

            traceback.print_exc()
    elif skip_neo4j:
        print("\n10. Skipping Neo4j loading (--skip-neo4j flag set)")
    elif gdn_model_path is None:
        print("\n10. Skipping Neo4j loading (no GDN model path provided)")

    return {
        "npz_path": str(npz_path),
        "json_path": str(json_path),
        "num_windows": len(X_normalized),
        "num_faulty": int((sensor_labels.sum(axis=1) > 0).sum()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create shared evaluation dataset from raw OBD data"
    )
    parser.add_argument(
        "--raw-data-path",
        type=str,
        default="data/carOBD/obdiidata",
        help="Path to directory containing raw OBD CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="llm/evaluation/shared_dataset",
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split name",
    )
    parser.add_argument(
        "--fault-percentage",
        type=float,
        default=0.3,
        help="Percentage of windows to inject faults (0.0-1.0)",
    )
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Maximum number of windows to include",
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--gdn-model-path",
        type=str,
        default="anomaly-detection/best_center_loss_gdn.pt",
        help="Path to GDN model checkpoint (default: anomaly-detection/best_center_loss_gdn.pt)",
    )
    parser.add_argument(
        "--neo4j-uri",
        type=str,
        default="bolt://127.0.0.1:7687",
        help="Neo4j connection URI (default: bolt://127.0.0.1:7687)",
    )
    parser.add_argument(
        "--neo4j-user",
        type=str,
        default="neo4j",
        help="Neo4j username (default: neo4j)",
    )
    parser.add_argument(
        "--neo4j-password",
        type=str,
        default="password",
        help="Neo4j password (default: password)",
    )
    parser.add_argument(
        "--skip-neo4j",
        action="store_true",
        help="Skip Neo4j loading (useful when Neo4j is not available)",
    )
    parser.add_argument(
        "--neo4j-window-subset",
        type=str,
        default=None,
        help='Comma-separated list of window indices to load into Neo4j (e.g., "0,1,2,3,4") or "first:50" for first 50 windows. Default: load all windows.',
    )
    parser.add_argument(
        "--top-k-correlations",
        type=int,
        default=10,
        help="Number of top correlations to select per window (default: 10)",
    )

    args = parser.parse_args()

    # Parse window subset if provided
    neo4j_window_subset = None
    if args.neo4j_window_subset:
        if args.neo4j_window_subset.startswith("first:"):
            # Format: "first:50" - load first N windows
            num_windows = int(args.neo4j_window_subset.split(":")[1])
            neo4j_window_subset = list(range(num_windows))
        else:
            # Format: "0,1,2,3,4" - load specific windows
            neo4j_window_subset = [
                int(x.strip()) for x in args.neo4j_window_subset.split(",")
            ]

    create_shared_evaluation_dataset(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        split=args.split,
        fault_percentage=args.fault_percentage,
        max_windows=args.max_windows,
        random_state=args.random_state,
        gdn_model_path=args.gdn_model_path,
        neo4j_uri=args.neo4j_uri,
        neo4j_user=args.neo4j_user,
        neo4j_password=args.neo4j_password,
        skip_neo4j=args.skip_neo4j,
        neo4j_window_subset=neo4j_window_subset,
        top_k_correlations=args.top_k_correlations,
    )
