"""
Evaluate LLM-only baseline method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Formats unnormalized windows for LLM prompts
3. Runs LLM inference
4. Compares predictions to ground truth
5. Computes evaluation metrics
"""

import numpy as np
import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import time
import sys
from tqdm import tqdm

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.inference import create_client, LMStudioClient
from llm.evaluation.metrics import compute_all_metrics, format_metrics_report
from llm.evaluation.stratified_sampling import (
    stratified_sample_indices,
    validate_fault_types_for_stratification,
)
from llm.evaluation.utils import call_llm_fault_diagnosis
from training.fault_injection import STATE_NAMES

VALID_FAULT_TYPES = sorted(STATE_NAMES)

# Observable signal characteristics for each fault type.
# Descriptions are expressed as what can be seen in the raw sensor signal —
# no injection parameters or implementation details.
FAULT_TYPE_DESCRIPTIONS = {
    "COOLANT_DROPOUT": "coolant temperature shows 2-5 intermittent drops well below normal operating range",
    "MAF_SCALE_LOW": "intake manifold pressure reads consistently ~20% lower than normal throughout the window",
    "TPS_STUCK": "throttle position freezes at a constant value from roughly the midpoint onward (near-zero variability in second half)",
    "VSS_DROPOUT": "vehicle speed drops to near-zero for roughly the middle third of the window then recovers",
    "RPM_SPIKE_DROPOUT": "engine RPM shows a sharp spike (up to ~1.8x normal) or dropout (down to ~0.4x) in the middle half of the window",
    "LOAD_SCALE_LOW": "engine load reads uniformly lower than normal throughout the window (scaled to roughly 25-60% of expected)",
    "STFT_STUCK_HIGH": "short-term fuel trim freezes at an elevated value in the middle 40% of the window",
    "LTFT_DRIFT_HIGH": "long-term fuel trim is shifted consistently higher than normal throughout the entire window",
}

SYSTEM_PROMPT = """You are an automotive OBD-II fault diagnostics system.

Your task for each window of sensor data:
1. Determine whether the window contains a fault (is_faulty).
2. If faulty, identify which sensor(s) are the root cause (faulty_sensors).
3. If faulty, classify the fault type (fault_type).

A window is faulty if any sensor shows an abnormal pattern such as:
- abrupt drops to near-zero or well below its normal operating range
- freezing at a constant value (std ≈ 0 in a portion of the window)
- a consistent scale reduction (mean significantly below normal)
- sustained spikes, dropouts, or drifts compared to typical operation

Respond with a single JSON object. No markdown, no extra prose."""

JSON_INSTRUCTIONS = (
    "Required JSON fields:\n"
    "  - is_faulty: true if a fault is detected, false otherwise\n"
    "  - faulty_sensors: list of sensor names (use exactly from VALID SENSOR NAMES; empty [] if no fault)\n"
    "  - fault_type: exactly one value from VALID FAULT TYPES (use 'normal' if no fault)\n"
    "  - confidence: 'high', 'medium', or 'low'\n"
    "  - reasoning: brief explanation referencing the specific signal pattern observed"
)


def _format_fault_type_list() -> str:
    """Format fault types with descriptions for prompt inclusion."""
    lines = []
    for ft, desc in FAULT_TYPE_DESCRIPTIONS.items():
        lines.append(f"  {ft}: {desc}")
    return "\n".join(lines)


def build_baseline_prompt(
    window_data: np.ndarray,
    sensor_names: List[str],
) -> List[Dict[str, str]]:
    """
    Build prompt with serialised raw unnormalised sensor time series.
    window_data shape: (300, 8) — timesteps × sensors.
    """
    series_dict = {}
    for col_idx, name in enumerate(sensor_names):
        values = window_data[:, col_idx].tolist()
        series_dict[name] = [round(float(v), 3) for v in values]

    lines = []
    lines.append("SENSOR TIME SERIES (300 timesteps per sensor, unnormalized real-world values):")
    lines.append(json.dumps(series_dict))
    lines.append("")
    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors field):")
    lines.append(", ".join(sensor_names))
    lines.append("")
    lines.append("VALID FAULT TYPES (use exactly one; 'normal' if no fault detected):")
    lines.append(_format_fault_type_list())
    lines.append("")
    lines.append(JSON_INSTRUCTIONS)

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def filter_sensor_labels_to_root_only(parsed_result: Dict, sensor_names: List[str]) -> np.ndarray:
    """
    Filter sensor labels to only include root cause sensors.

    Args:
        parsed_result: Dictionary from parsed_to_prediction() containing:
            - 'root_cause_sensors': List of root cause sensor names
            - 'sensor_labels_root_only': Binary array (already filtered)
        sensor_names: List of all sensor names
    
    Returns:
        Binary array (num_sensors,) with only root cause sensors marked as 1
    """
    # If root_only labels already computed, use them
    if "sensor_labels_root_only" in parsed_result:
        return parsed_result["sensor_labels_root_only"].copy()
    
    # Otherwise, extract from root_cause_sensors list
    sensor_labels = np.zeros(len(sensor_names), dtype=np.float32)
    root_cause_sensors = parsed_result.get("root_cause_sensors", [])
    
    for sensor in root_cause_sensors:
        if sensor in sensor_names:
            idx = sensor_names.index(sensor)
            sensor_labels[idx] = 1.0
        else:
            # Try matching without parentheses
            sensor_clean = sensor.replace(" ()", "").strip()
            for i, name in enumerate(sensor_names):
                name_clean = name.replace(" ()", "").strip()
                if sensor_clean == name_clean:
                    sensor_labels[i] = 1.0
                    break
    
    return sensor_labels


def load_llm_model(
    model_repo: str = "granite-4.0-h-micro-GGUF",
    base_url: str = "http://localhost:1234/v1",
    timeout: int = 120,
) -> LMStudioClient:
    """
    Load LLM model via LM Studio HTTP server.

    Args:
        model_repo: Model name in LM Studio (default: granite-4.0-h-micro-GGUF)
        base_url: Base URL for LM Studio HTTP server
        timeout: Request timeout in seconds (default 120; first request may need warmup)

    Returns:
        LMStudioClient instance
    """
    print(f"Connecting to LM Studio for model: {model_repo} (timeout={timeout}s)")
    client = create_client(
        model_name=model_repo,
        base_url=base_url,
        timeout=timeout,
        check_connection=True
    )
    return client


def evaluate_llm_baseline(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    model_repo: Optional[str] = None,
    base_url: Optional[str] = None,
    limit: Optional[int] = None,
    stratify_limit: bool = True,
    seed: int = 42,
    timeout: int = 120,
) -> Dict[str, any]:
    """
    Evaluate LLM baseline on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        output_path: Optional path to save results JSON
        model_repo: Model name in LM Studio
        limit: Optional cap on windows (default: stratified by fault_types when below dataset size)
        stratify_limit: If False with --limit, use the first N windows in file order

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating LLM Baseline")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print()

    # Load LLM model
    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"
    lm_base_url = base_url or "http://localhost:1234/v1"

    try:
        client = load_llm_model(model_repo, base_url=lm_base_url, timeout=timeout)
        print()
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to LM Studio: {e}. "
            f"Please ensure LM Studio is running with the HTTP server enabled."
        )

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)

    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]
    ref_reasoning_full = data["reference_reasoning"].tolist() if "reference_reasoning" in data else None

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
        statistical_features = np.array(metadata.get("statistical_features", []))
    else:
        # Fallback: use default sensor names
        sensor_names = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]
        statistical_features = None

    num_windows = unnormalized_windows.shape[0]
    num_windows_full = num_windows
    fault_types_full = data.get("fault_types", None)
    sample_indices = None

    if limit is not None and limit > 0 and limit < num_windows:
        if stratify_limit:
            if fault_types_full is None:
                raise ValueError(
                    "Stratified limit requires fault_types in the dataset. "
                    "The .npz file must contain a 'fault_types' array."
                )
            validate_fault_types_for_stratification(fault_types_full, sensor_labels_true)
            sample_indices = stratified_sample_indices(
                fault_types_full, limit, random_state=seed
            )
            num_windows = len(sample_indices)
            unnormalized_windows = unnormalized_windows[sample_indices]
            sensor_labels_true = sensor_labels_true[sample_indices]
            window_labels_true = window_labels_true[sample_indices]
            fault_types_full = fault_types_full[sample_indices]
            if ref_reasoning_full is not None:
                ref_reasoning_full = [ref_reasoning_full[j] for j in sample_indices]
            if statistical_features is not None and len(statistical_features) == num_windows_full:
                statistical_features = statistical_features[sample_indices]
            print(f"  ⚠️  LIMIT MODE: Stratified sample of {num_windows} windows (seed={seed})")
        else:
            sample_indices = None
            ix = np.arange(limit, dtype=np.int64)
            num_windows = int(limit)
            unnormalized_windows = unnormalized_windows[ix]
            sensor_labels_true = sensor_labels_true[ix]
            window_labels_true = window_labels_true[ix]
            if fault_types_full is not None:
                fault_types_full = fault_types_full[ix]
            if ref_reasoning_full is not None:
                ref_reasoning_full = ref_reasoning_full[:limit]
            if statistical_features is not None and len(statistical_features) == num_windows_full:
                statistical_features = statistical_features[ix]
            print(
                f"  ⚠️  LIMIT MODE: First {num_windows} windows in file order (--no-stratify-limit)"
            )

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {unnormalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Get model name from the client (already created above)
    model_name = client.config.model_name

    # Run predictions
    print("Running LLM predictions...")
    window_labels_pred = []
    is_faulty_pred = []  # Direct LLM is_faulty boolean
    sensor_labels_pred = []  # Filtered (root-only) predictions
    sensor_labels_pred_raw = []  # Raw (all sensors) predictions
    fault_types_pred = []
    reasoning_list = []
    processing_times = []
    context_lengths = []

    with tqdm(total=num_windows, desc="LLM Inference", unit="window") as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            messages = build_baseline_prompt(
                unnormalized_windows[window_idx], sensor_names
            )
            context_length = sum(len(m.get("content", "").split()) for m in messages)

            prediction = call_llm_fault_diagnosis(client, model_name, messages, sensor_names)
            if window_idx == 0 and "Error:" in prediction.get("reasoning", ""):
                print(f"  ⚠️  LLM error on first window: {prediction['reasoning']}")

            sensor_labels_filtered = filter_sensor_labels_to_root_only(prediction, sensor_names)
            sensor_labels_raw = prediction.get("sensor_labels", sensor_labels_filtered.copy())

            window_labels_pred.append(prediction["window_label"])
            is_faulty_pred.append(int(prediction.get("is_faulty", prediction["window_label"] > 0)))
            sensor_labels_pred.append(sensor_labels_filtered)
            sensor_labels_pred_raw.append(sensor_labels_raw)
            fault_types_pred.append(prediction["fault_type"])
            reasoning_list.append(prediction.get("reasoning", ""))
            processing_times.append(time.time() - start_time)
            context_lengths.append(context_length)

            # Update progress bar with current metrics
            pbar.update(1)
            if (window_idx + 1) % 10 == 0:
                avg_time = (
                    np.mean(processing_times[-10:])
                    if len(processing_times) >= 10
                    else np.mean(processing_times)
                )
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s"})

    window_labels_pred = np.array(window_labels_pred)
    is_faulty_pred = np.array(is_faulty_pred)  # Direct LLM is_faulty boolean (0/1)
    sensor_labels_pred = np.array(sensor_labels_pred)  # Filtered (root-only)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)  # Raw (all sensors)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    avg_context_length = np.mean(context_lengths) if context_lengths else 0

    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total processing time: {total_processing_time:.2f} seconds")
    if avg_context_length > 0:
        print(f"  Average context length: {avg_context_length:.0f} tokens")
    print()

    # Convert window_labels_true to sensor-indexed format (0-8)
    # The dataset stores window_labels as window indices, not sensor-indexed labels
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1  # 1-indexed (sensor 0 -> label 1)
        else:
            window_labels_true_converted[i] = 0  # No fault
    window_labels_true = window_labels_true_converted

    # Build aligned reference reasoning list (same length as predictions)
    n_pred = len(reasoning_list)
    if ref_reasoning_full is not None:
        ref_reasoning_list = list(ref_reasoning_full[:n_pred])
        ref_reasoning_list += [""] * max(0, n_pred - len(ref_reasoning_list))
    else:
        ref_reasoning_list = [""] * n_pred

    # Compute metrics
    print("Computing evaluation metrics...")

    # Compute metrics using filtered (root-only) predictions for precision improvement
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,  # Use filtered (root-only) for main metrics
        sensor_names=sensor_names,
        fault_types=fault_types_full if fault_types_full is not None else None,
        fault_types_pred=fault_types_pred,
        reasoning=reasoning_list,
        reference_reasoning=ref_reasoning_list,
    )

    # Also compute raw metrics for comparison
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred_raw,  # Use raw (all sensors) for comparison
        sensor_names=sensor_names,
        fault_types=fault_types_full if fault_types_full is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]

    # Add efficiency metrics
    metrics["efficiency"] = {
        "avg_processing_time_seconds": float(avg_processing_time),
        "total_processing_time_seconds": float(total_processing_time),
        "windows_per_second": float(num_windows / total_processing_time),
        "avg_context_length_tokens": float(avg_context_length),
    }

    # Print report
    report = format_metrics_report(metrics)
    print(report)

    # Save results
    results = {
        "method": "llm_baseline",
        "dataset": str(dataset_path),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "is_faulty": is_faulty_pred.tolist(),        # Direct LLM is_faulty boolean
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),  # Filtered (root-only)
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),
            "fault_types": fault_types_pred,
            "reasoning": reasoning_list,
        },
    }
    if sample_indices is not None:
        results["sample_indices"] = sample_indices.tolist()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results


def run(
    dataset_path: str,
    lm_url: str,
    limit: Optional[int] = None,
    stratify_limit: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run LLM baseline evaluation and return unified format for compare_methods.

    Returns dict with "results" (per-window list) and "metrics" (unified format).
    """
    from llm.evaluation.metrics import compute_all_metrics_unified

    res = evaluate_llm_baseline(
        dataset_path=Path(dataset_path),
        output_path=None,
        base_url=lm_url,
        limit=limit,
        stratify_limit=stratify_limit,
        seed=seed,
    )
    preds = res["predictions"]
    n_pred = len(preds["window_labels"])
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    fault_types = data.get("fault_types", None)

    sample_indices = res.get("sample_indices")
    if sample_indices is not None:
        sample_indices = np.array(sample_indices)
        sensor_labels_true = sensor_labels_true[sample_indices]
        if fault_types is not None:
            fault_types = fault_types[sample_indices]
    else:
        sensor_labels_true = sensor_labels_true[:n_pred]
        if fault_types is not None:
            fault_types = fault_types[:n_pred]

    ref_reason_raw = data.get("reference_reasoning", None)
    if ref_reason_raw is not None:
        ref_reason = np.asarray(ref_reason_raw)
        if sample_indices is not None:
            ref_reason = ref_reason[sample_indices]
        ref_reason = ref_reason.tolist()
    else:
        ref_reason = [""] * n_pred
    if len(ref_reason) < n_pred:
        ref_reason = list(ref_reason) + [""] * (n_pred - len(ref_reason))
    else:
        ref_reason = ref_reason[:n_pred]

    sensor_names = res.get("sensor_names", [])
    if not sensor_names and "dataset_info" in res:
        sensor_names = res["dataset_info"].get("sensor_names", [])
    metadata_path = Path(dataset_path).parent / f"{Path(dataset_path).stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        sensor_names = meta["dataset_info"]["sensor_names"]

    results = []
    for i in range(len(preds["window_labels"])):
        sl_pred = preds["sensor_labels"][i]
        ft_pred = preds["fault_types"][i]
        reasoning = preds["reasoning"][i] if i < len(preds["reasoning"]) else ""
        sl_true = sensor_labels_true[i].tolist()
        wl_true = 1 if sum(sl_true) > 0 else 0
        # Unified binary window prediction derived from sensor-indexed window_label (0=normal)
        # to stay consistent with the sensor-indexed window_label stored in results/*.json.
        wl_pred_bin = 1 if int(preds["window_labels"][i]) > 0 else 0
        ft_true = fault_types[i] if fault_types is not None and i < len(fault_types) else None
        ft_true_str = "normal" if (ft_true is None or ft_true == "" or wl_true == 0) else str(ft_true)
        ft_pred_str = "normal" if (ft_pred is None or ft_pred == "") else str(ft_pred)
        ref = ref_reason[i] if i < len(ref_reason) else ""
        results.append({
            "window_label_true": int(wl_true),
            "window_label_pred": int(wl_pred_bin),
            "sensor_labels_true": sl_true,
            "sensor_labels_pred": [float(x) for x in sl_pred],
            "fault_type_true": ft_true_str,
            "fault_type_pred": ft_pred_str,
            "reasoning": reasoning or "",
            "reference_reasoning": ref or "",
        })
    metrics = compute_all_metrics_unified(results, sensor_names)
    return {"results": results, "metrics": metrics, "sensor_cols": sensor_names}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LLM baseline on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/shared_dataset/test.npz",
        help="Path to shared dataset .npz file (default: data/shared_dataset/test.npz)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_baseline.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="granite-4.0-h-micro-GGUF",
        help="Model name in LM Studio (default: granite-4.0-h-micro-GGUF)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap windows; default is stratified by fault type (needs fault_types in .npz unless --no-stratify-limit)",
    )
    parser.add_argument(
        "--no-stratify-limit",
        action="store_false",
        dest="stratify_limit",
        help="With --limit, take the first N windows in file order instead of stratified sampling",
    )
    parser.set_defaults(stratify_limit=True)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified --limit (default: 42)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="LLM request timeout in seconds (default: 120; increase if first request times out)",
    )

    args = parser.parse_args()

    evaluate_llm_baseline(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        model_repo=args.model_repo,
        limit=args.limit,
        stratify_limit=args.stratify_limit,
        seed=args.seed,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
