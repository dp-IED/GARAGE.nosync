"""
Create shared dataset for all pipeline stages:
  - Stage 1 training: clean_normalized_windows (0% faults) + forecast_targets
  - Stage 2 training: normalized_windows (25% faults) + sensor_labels + window_is_faulty
  - Eval:             normalized_windows (30% faults) + unnormalized_windows + full labels

Run once:
    python data/create_shared_dataset.py \
        --raw-data-path data/carOBD/obdiidata \
        --output-dir data/shared_dataset

Outputs:
    data/shared_dataset/train.npz
    data/shared_dataset/val.npz
    data/shared_dataset/test.npz
    data/shared_dataset/{split}_metadata.json

NPZ schema:
    train.npz / val.npz:
        clean_normalized_windows  (N, 300, 8)   Stage 1 input (no faults)
        forecast_targets           (N, 3, 8)     Stage 1 labels (horizons 1,5,10)
        normalized_windows         (N, 300, 8)   Stage 2 input (25% faults injected)
        unnormalized_windows       (N, 300, 8)   inverse_transform of normalized_windows
        sensor_labels              (N, 8)         Stage 2 + eval ground truth
        window_is_faulty           (N,)           binary fault flag
        fault_types                (N,)           strings e.g. "VSS_DROPOUT" / "normal"
        drive_ids                  (N,)

    test.npz:
        normalized_windows         (N, 300, 8)   30% faults; GDN + KG-LLM eval
        unnormalized_windows       (N, 300, 8)   inverse_transform; LLM prompt values
        sensor_labels              (N, 8)
        window_is_faulty           (N,)
        window_labels              (N,)           sequential 0..N-1 (backward compat)
        fault_types                (N,)
        statistical_features       (N, 8, 9)
        reference_reasoning        (N,)
        drive_ids                  (N,)
"""

import numpy as np
import pandas as pd
import json
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from scipy import stats
import torch
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from training.train_stage1 import (
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    remove_zero_variance_columns,
    mean_fill_missing_timestamps_and_remove_duplicates,
    downsample,
    filter_long_drives,
    add_cross_channel_features,
    build_forecast_windows,
    SENSOR_COLS,
    ID_COL,
    TIME_COL,
    WINDOW_SIZE,
    FORECAST_HORIZONS,
)
from training.train_stage2_clean import build_clean_windows
from training.fault_injection import inject_faults_with_sensor_labels

# Fault percentages: train/val 25%, test 30%
TRAIN_VAL_FAULT_PCT = 0.25
TEST_FAULT_PCT = 0.30
MIN_WINDOWS_PER_DRIVE = 30


def compute_statistical_features(window_data: np.ndarray) -> np.ndarray:
    """(window_size, D) -> (D, 9): mean, std, min, max, range, median, mode, skew, kurtosis"""
    D = window_data.shape[1]
    features = np.zeros((D, 9))
    for i in range(D):
        v = window_data[:, i]
        features[i, 0] = np.mean(v)
        features[i, 1] = np.std(v)
        features[i, 2] = np.min(v)
        features[i, 3] = np.max(v)
        features[i, 4] = features[i, 3] - features[i, 2]
        features[i, 5] = np.median(v)
        try:
            m = stats.mode(np.round(v, 2))
            features[i, 6] = m.mode[0] if len(m.mode) > 0 else features[i, 0]
        except Exception:
            features[i, 6] = features[i, 0]
        features[i, 7] = stats.skew(v) if features[i, 1] > 0 else 0.0
        features[i, 8] = stats.kurtosis(v) if features[i, 1] > 0 else 0.0
    return features


def _generate_reference_reasoning(
    fault_type: str, sensor_labels_row, sensor_cols: List[str]
) -> str:
    affected = [sensor_cols[i] for i, v in enumerate(sensor_labels_row) if v > 0]
    affected_str = ", ".join(affected) if affected else "unknown sensors"
    templates = {
        "COOLANT_DROPOUT": f"Coolant temperature signal dropped unexpectedly, affecting {affected_str}.",
        "VSS_DROPOUT": f"Vehicle speed sensor signal lost, causing correlation errors with {affected_str}.",
        "MAF_SCALE_LOW": f"Mass airflow sensor reading skewed low, impacting combustion metrics like {affected_str}.",
        "TPS_STUCK": f"Throttle position sensor is flatlined or stuck, mismatching expected load on {affected_str}.",
        "gradual_drift": f"Gradual sensor drift detected on {affected_str} over time.",
    }
    return templates.get(fault_type, f"Anomalous behavior detected in {affected_str}.")


def _drop_sparse_drives(
    X: np.ndarray,
    drive_ids: np.ndarray,
    extras: Dict[str, np.ndarray],
) -> tuple:
    """Remove windows from drives with fewer than MIN_WINDOWS_PER_DRIVE windows."""
    counts = pd.Series(drive_ids).value_counts()
    valid = counts[counts >= MIN_WINDOWS_PER_DRIVE].index
    mask = np.isin(drive_ids, valid)
    if (~mask).sum() > 0:
        dropped_drives = (counts < MIN_WINDOWS_PER_DRIVE).sum()
        print(
            f"  [Filter] Kept {mask.sum()} windows from {len(valid)} drives "
            f"(dropped {(~mask).sum()} windows from {dropped_drives} drives with <{MIN_WINDOWS_PER_DRIVE} windows)"
        )
    X = X[mask]
    drive_ids = drive_ids[mask]
    filtered_extras = {
        k: v[mask] if isinstance(v, (np.ndarray, torch.Tensor)) else v
        for k, v in extras.items()
    }
    return X, drive_ids, filtered_extras, mask


def _inverse_transform_windows(X_norm: np.ndarray, scaler) -> np.ndarray:
    """Inverse-transform (N, W, D) normalized windows back to raw sensor values."""
    N, W, D = X_norm.shape
    return scaler.inverse_transform(X_norm.reshape(-1, D)).reshape(N, W, D)


def process_split(
    split: str,
    split_data: pd.DataFrame,
    sensor_cols: List[str],
    scaler,  # pre-fitted on train split
    random_state: int,
    output_dir: Path,
    max_windows: int = None,
) -> Dict:
    print(f"\n{'=' * 60}")
    print(f"Processing split: {split.upper()}")
    print(f"{'=' * 60}")
    print(f"  Drives: {split_data[ID_COL].nunique()}, Rows: {len(split_data):,}")

    # ------------------------------------------------------------------ #
    # 1. Build clean normalized windows (Stage 1 needs this for all splits;
    #    also used as the base for fault injection)
    # ------------------------------------------------------------------ #
    print("  Building forecast windows (clean)...")
    # build_forecast_windows returns (X, y_forecast, drive_ids, scaler)
    # X: (N, W, D), y_forecast: (N, num_horizons, D)
    X_clean_norm, y_forecast, drive_ids_arr, _ = build_forecast_windows(
        split_data,
        sensor_cols,
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        horizons=FORECAST_HORIZONS,
        scaler=scaler,  # always transform-only; scaler fitted on train
    )
    X_clean_norm = X_clean_norm.numpy()  # (N, W, D)
    y_forecast = y_forecast.numpy()  # (N, num_horizons, D)
    drive_ids_arr = np.array(drive_ids_arr)

    # Remove drives with too few windows
    X_clean_norm, drive_ids_arr, extras, mask = _drop_sparse_drives(
        X_clean_norm, drive_ids_arr, {"y_forecast": y_forecast}
    )
    y_forecast = extras["y_forecast"]
    N = len(X_clean_norm)
    print(f"  After filter: {N} windows")

    # Optional cap
    if max_windows and N > max_windows:
        rng = np.random.default_rng(random_state)
        idx = np.sort(rng.choice(N, max_windows, replace=False))
        X_clean_norm = X_clean_norm[idx]
        y_forecast = y_forecast[idx]
        drive_ids_arr = drive_ids_arr[idx]
        N = max_windows
        print(f"  Capped to {N} windows")

    # ------------------------------------------------------------------ #
    # 2. Inject faults into normalized clean windows
    # ------------------------------------------------------------------ #
    fault_pct = TEST_FAULT_PCT if split == "test" else TRAIN_VAL_FAULT_PCT
    # Each split uses a distinct RNG stream to avoid shared fault-injection sequences:
    # train=random_state (42), val=43, test=44
    fault_seed = {"val": 43, "test": random_state + 2}.get(split, random_state)

    print(f"  Injecting faults ({fault_pct * 100:.0f}%, seed={fault_seed})...")
    X_clean_tensor = torch.tensor(X_clean_norm, dtype=torch.float32)
    # y_forecast not needed by injector; pass a dummy target matching expected shape
    y_dummy = torch.zeros(N, len(sensor_cols), dtype=torch.float32)

    (
        X_faulty_tensor,
        _,
        sensor_labels_tensor,
        window_is_faulty_tensor,
        fault_types_list,
    ) = inject_faults_with_sensor_labels(
        X_clean_tensor,
        y_dummy,
        sensor_cols,
        fault_percentage=fault_pct,
        random_state=fault_seed,
        use_stratified=True,
    )

    X_faulty_norm = X_faulty_tensor.numpy()  # (N, W, D)
    sensor_labels = sensor_labels_tensor.numpy().astype(np.float32)  # (N, D)
    window_is_faulty = window_is_faulty_tensor.numpy().astype(np.int64)  # (N,)
    fault_types = np.array(fault_types_list, dtype=object)  # (N,) strings

    n_faulty = window_is_faulty.sum()
    print(f"  Injected faults: {n_faulty}/{N} windows ({n_faulty / N * 100:.1f}%)")

    # ------------------------------------------------------------------ #
    # 3. Derive unnormalized windows via inverse_transform
    #    Both normalized_windows (faulty) and unnormalized_windows reflect
    #    the same signal so LLM and GDN evaluate IDENTICAL fault patterns.
    # ------------------------------------------------------------------ #
    print("  Inverse-transforming to get unnormalized_windows...")
    X_unnorm = _inverse_transform_windows(X_faulty_norm, scaler).astype(np.float32)

    # ------------------------------------------------------------------ #
    # 4. Build per-split npz payload
    # ------------------------------------------------------------------ #
    sensor_cols_clean = [s.replace(" ()", "") for s in sensor_cols]

    payload = {
        "normalized_windows": X_faulty_norm.astype(np.float32),
        "unnormalized_windows": X_unnorm,
        "sensor_labels": sensor_labels,
        "window_is_faulty": window_is_faulty,
        "fault_types": fault_types,
        "drive_ids": drive_ids_arr,
        # backward compat key for eval scripts that read window_labels
        "window_labels": np.arange(N, dtype=np.int64),
    }

    if split in ("train", "val"):
        # Stage 1 fields — only needed for training, omit from test to save space
        payload["clean_normalized_windows"] = X_clean_norm.astype(np.float32)
        payload["forecast_targets"] = y_forecast.astype(np.float32)

    if split == "test":
        # Eval-only fields
        print("  Computing statistical features...")
        stat_feats = np.stack(
            [compute_statistical_features(X_unnorm[i]) for i in range(N)]
        )  # (N, D, 9)
        payload["statistical_features"] = stat_feats.astype(np.float32)

        print("  Generating reference_reasoning strings...")
        reference_reasoning = np.array(
            [
                _generate_reference_reasoning(
                    str(fault_types[i]) if fault_types[i] else "normal",
                    sensor_labels[i],
                    sensor_cols_clean,
                )
                if window_is_faulty[i]
                else "Normal operating conditions."
                for i in range(N)
            ],
            dtype=object,
        )
        payload["reference_reasoning"] = reference_reasoning

    # ------------------------------------------------------------------ #
    # 5. Save
    # ------------------------------------------------------------------ #
    npz_path = output_dir / f"{split}.npz"
    np.savez_compressed(npz_path, **payload)
    print(f"  ✓ Saved {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    metadata = {
        "dataset_info": {
            "split": split,
            "num_windows": N,
            "num_faulty_windows": int(n_faulty),
            "num_normal_windows": int(N - n_faulty),
            "fault_percentage": fault_pct,
            "window_size": WINDOW_SIZE,
            "num_sensors": len(sensor_cols),
            "sensor_names": sensor_cols,
            "forecast_horizons": FORECAST_HORIZONS,
            "created_at": datetime.now().isoformat(),
        }
    }
    meta_path = output_dir / f"{split}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return {"split": split, "num_windows": N, "num_faulty": int(n_faulty)}


def create_shared_dataset(
    raw_data_path: str,
    output_dir: str,
    max_windows: int = None,
    random_state: int = 42,
) -> None:
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load + preprocess ALL drives (same pipeline as training scripts)
    # ------------------------------------------------------------------ #
    print("\n1. Loading raw OBD data...")
    raw_data_path = Path(raw_data_path)
    csv_files = sorted(raw_data_path.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {raw_data_path}")

    df_list = []
    for f in csv_files:
        df = pd.read_csv(f, index_col=False)
        df["drive_id"] = f.name
        df_list.append(df)
    print(f"  Loaded {len(csv_files)} CSV files")

    data = pd.concat(df_list, ignore_index=True)
    cols_to_drop = [
        "WARM_UPS_SINCE_CODES_CLEARED ()",
        "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
    ]
    data.drop(columns=[c for c in cols_to_drop if c in data.columns], inplace=True)

    print("\n2. Preprocessing...")
    data = mean_fill_missing_timestamps_and_remove_duplicates(
        data, time_col=TIME_COL, id_cols=["drive_id"]
    )
    data = remove_zero_variance_columns(data, exclude_cols=["drive_id"])
    data = downsample(
        data, time_col=TIME_COL, source_file_col="drive_id", downsample_factor=2
    )
    # Stage 1 needs W + max(FORECAST_HORIZONS) = 310 min length
    data = filter_long_drives(
        data, id_col="drive_id", min_length=WINDOW_SIZE + max(FORECAST_HORIZONS)
    )
    data = add_cross_channel_features(data)
    data = data.sort_values(["drive_id", TIME_COL]).reset_index(drop=True)

    sensor_cols = [s for s in SENSOR_COLS if s in data.columns]
    missing = set(SENSOR_COLS) - set(sensor_cols)
    if missing:
        print(f"  Warning: Missing sensors dropped: {missing}")

    # ------------------------------------------------------------------ #
    # Drive-level split — deterministic, matches training scripts
    # ------------------------------------------------------------------ #
    print("\n3. Splitting drives...")
    unique_drives = np.sort(data["drive_id"].unique())
    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(unique_drives)
    n = len(shuffled)
    t_end = int(TRAIN_RATIO * n)
    v_end = int((TRAIN_RATIO + VAL_RATIO) * n)
    drive_splits = {
        "train": set(shuffled[:t_end]),
        "val": set(shuffled[t_end:v_end]),
        "test": set(shuffled[v_end:]),
    }
    for s, drives in drive_splits.items():
        print(f"  {s}: {len(drives)} drives")

    # ------------------------------------------------------------------ #
    # Fit scaler on TRAIN split only (used for all splits)
    # ------------------------------------------------------------------ #
    print("\n4. Fitting scaler on train split...")
    train_data = data[data["drive_id"].isin(drive_splits["train"])].reset_index(
        drop=True
    )
    # build_forecast_windows with scaler=None fits and returns scaler
    _, _, _, scaler_train = build_forecast_windows(
        train_data,
        sensor_cols,
        ID_COL,
        TIME_COL,
        WINDOW_SIZE,
        horizons=FORECAST_HORIZONS,
        scaler=None,
    )
    print("  ✓ Scaler fitted on train drives")

    # ------------------------------------------------------------------ #
    # Process each split
    # ------------------------------------------------------------------ #
    results = []
    for split in ("train", "val", "test"):
        split_data = data[data["drive_id"].isin(drive_splits[split])].reset_index(
            drop=True
        )
        result = process_split(
            split=split,
            split_data=split_data,
            sensor_cols=sensor_cols,
            scaler=scaler_train,
            random_state=random_state,
            output_dir=output_dir,
            max_windows=max_windows,
        )
        results.append(result)

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("✓ Shared dataset created")
    print("=" * 60)
    for r in results:
        print(
            f"  {r['split']:5s}: {r['num_windows']:5d} windows "
            f"({r['num_faulty']} faulty, {r['num_windows'] - r['num_faulty']} normal)"
        )
    print(f"\nOutput: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create shared dataset for all pipeline stages"
    )
    parser.add_argument("--raw-data-path", default="data/carOBD/obdiidata")
    parser.add_argument("--output-dir", default="data/shared_dataset")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Cap per split (default: no limit)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    create_shared_dataset(
        raw_data_path=args.raw_data_path,
        output_dir=args.output_dir,
        max_windows=args.max_windows,
        random_state=args.random_state,
    )
