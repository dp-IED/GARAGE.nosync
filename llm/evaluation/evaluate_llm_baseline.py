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

from llm.inference import (
    create_client,
    create_json_schema_response_format,
    LMStudioClient,
)

from llm.evaluation.metrics import compute_all_metrics, format_metrics_report
from llm.evaluation.schemas import FaultDiagnosis
from llm.evaluation.utils import parse_structured_response, parsed_to_prediction


SYSTEM_PROMPT = (
    "You are an automotive fault diagnostics system. You will receive "
    "raw OBD-II sensor time-series data from a vehicle. Identify faulty "
    "sensors and classify the fault type. Only use sensor names from the "
    "provided valid list. Respond with JSON only."
)


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
    lines.append("SENSOR TIME SERIES (300 timesteps per sensor):")
    lines.append(json.dumps(series_dict))
    lines.append("")
    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors field):")
    lines.append(", ".join(sensor_names))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _call_llm_structured(
    client: LMStudioClient,
    model_name: str,
    messages: List[Dict[str, str]],
) -> str:
    """Call LLM with structured JSON schema output."""
    response_format = create_json_schema_response_format(
        schema_name="FaultDiagnosis",
        schema_dict=FaultDiagnosis.model_json_schema(),
        strict=True
    )

    response = client.chat_completions_create(
        model=model_name,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
    )
    return response["choices"][0]["message"]["content"]


INVALID_FAULT_TYPE_LABELS = frozenset({"normal", "unknown", "faulty", ""})


def _validate_fault_types_for_stratification(
    fault_types: np.ndarray,
    sensor_labels: np.ndarray,
) -> None:
    """
    Assert fault_types exists and every faulty window has a valid fault type.
    Raises ValueError if any faulty window lacks a proper fault type.
    """
    faulty_mask = sensor_labels.sum(axis=1) != 0
    faulty_indices = np.where(faulty_mask)[0]
    if len(faulty_indices) == 0:
        return
    for i in faulty_indices:
        ft = fault_types[i]
        ft_str = (str(ft).strip() if ft is not None else "")
        if ft is None or ft_str in INVALID_FAULT_TYPE_LABELS:
            raise ValueError(
                f"Stratified limit requires valid fault_types for every faulty window. "
                f"Window {i} has sensor_labels.sum(axis=1) != 0 but fault_types[{i}] = {ft!r} "
                f"(None, empty, 'normal', 'unknown', or 'faulty' are invalid for faulty windows)."
            )


def _stratified_sample_indices(
    fault_types: np.ndarray,
    limit: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Sample indices stratified by fault type, preserving full-dataset proportions.
    Returns sorted indices for deterministic ordering.
    """
    n = len(fault_types)
    if limit >= n:
        return np.arange(n)

    ft_str = np.array(
        [(str(ft).strip() or "normal") if ft is not None else "normal" for ft in fault_types]
    )
    unique = np.unique(ft_str)
    stratum_indices = {ft: np.where(ft_str == ft)[0] for ft in unique}
    stratum_sizes = {ft: len(idxs) for ft, idxs in stratum_indices.items()}
    total = sum(stratum_sizes.values())

    targets = {}
    for ft in unique:
        raw = limit * stratum_sizes[ft] / total
        targets[ft] = max(0, min(stratum_sizes[ft], int(np.floor(raw))))

    current_sum = sum(targets.values())
    remainder = limit - current_sum
    if remainder > 0:
        fracs = [(limit * stratum_sizes[ft] / total - targets[ft], ft) for ft in unique]
        fracs.sort(key=lambda x: -x[0])
        for _, ft in fracs:
            if remainder <= 0:
                break
            add = min(remainder, stratum_sizes[ft] - targets[ft])
            targets[ft] += add
            remainder -= add

    rng = np.random.default_rng(random_state)
    sampled = []
    for ft in unique:
        idxs = stratum_indices[ft]
        k = min(targets[ft], len(idxs))
        chosen = rng.choice(idxs, size=k, replace=False)
        sampled.extend(chosen.tolist())

    return np.sort(np.array(sampled))


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
    seed: int = 42,
    timeout: int = 120,
) -> Dict[str, any]:
    """
    Evaluate LLM baseline on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        output_path: Optional path to save results JSON
        model_repo: Model name in LM Studio
        limit: Optional limit on number of windows to process (for testing)

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
        if fault_types_full is None:
            raise ValueError(
                "Stratified limit requires fault_types in the dataset. "
                "The .npz file must contain a 'fault_types' array."
            )
        _validate_fault_types_for_stratification(fault_types_full, sensor_labels_true)
        sample_indices = _stratified_sample_indices(
            fault_types_full, limit, random_state=seed
        )
        num_windows = len(sample_indices)
        unnormalized_windows = unnormalized_windows[sample_indices]
        sensor_labels_true = sensor_labels_true[sample_indices]
        window_labels_true = window_labels_true[sample_indices]
        fault_types_full = fault_types_full[sample_indices]
        if statistical_features is not None and len(statistical_features) == num_windows_full:
            statistical_features = statistical_features[sample_indices]
        print(f"  ⚠️  LIMIT MODE: Stratified sample of {num_windows} windows (seed={seed})")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {unnormalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Get model name from the client (already created above)
    model_name = client.config.model_name

    # Run predictions
    print("Running LLM predictions...")
    window_labels_pred = []
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

            try:
                raw_json = _call_llm_structured(client, model_name, messages)
                parsed = parse_structured_response(raw_json, sensor_names)
                prediction = parsed_to_prediction(parsed, sensor_names)
            except Exception as e:
                if window_idx == 0:
                    print(f"  ⚠️  LLM error on first window: {e}")
                empty_labels = np.zeros(len(sensor_names), dtype=np.float32)
                prediction = {
                    "window_label": 0,
                    "sensor_labels": empty_labels,
                    "sensor_labels_root_only": empty_labels.copy(),
                    "fault_type": None,
                    "reasoning": f"Error: {str(e)}",
                }

            sensor_labels_filtered = filter_sensor_labels_to_root_only(prediction, sensor_names)
            sensor_labels_raw = prediction.get("sensor_labels", sensor_labels_filtered.copy())

            window_labels_pred.append(prediction["window_label"])
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
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),  # Filtered (root-only)
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),  # Raw (all sensors)
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


def run(dataset_path: str, lm_url: str, limit: Optional[int] = None, seed: int = 42) -> dict:
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
        seed=seed,
    )
    preds = res["predictions"]
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    fault_types = data.get("fault_types", None)

    sample_indices = res.get("sample_indices")
    if sample_indices is not None:
        sample_indices = np.array(sample_indices)
        sensor_labels_true = sensor_labels_true[sample_indices]
        if fault_types is not None:
            fault_types = fault_types[sample_indices]
    ref_reason = data.get("reference_reasoning", None)
    if ref_reason is not None:
        ref_reason = np.asarray(ref_reason)
        if sample_indices is not None:
            ref_reason = ref_reason[sample_indices]
        ref_reason = ref_reason.tolist()
    else:
        ref_reason = [""] * len(sensor_labels_true)
    sensor_names = res.get("sensor_names", [])
    if not sensor_names and "dataset_info" in res:
        sensor_names = res["dataset_info"].get("sensor_names", [])
    metadata_path = Path(dataset_path).parent / f"{Path(dataset_path).stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        sensor_names = meta["dataset_info"]["sensor_names"]
    ref_reason = data.get("reference_reasoning", None)
    if ref_reason is not None and hasattr(ref_reason, "tolist"):
        ref_reason = ref_reason.tolist()
    elif ref_reason is None:
        ref_reason = [""] * len(sensor_labels_true)

    results = []
    for i in range(len(preds["window_labels"])):
        wl_pred = preds["window_labels"][i]
        sl_pred = preds["sensor_labels"][i]
        ft_pred = preds["fault_types"][i]
        reasoning = preds["reasoning"][i] if i < len(preds["reasoning"]) else ""
        sl_true = sensor_labels_true[i].tolist()
        wl_true = 1 if sum(sl_true) > 0 else 0
        wl_pred_bin = 1 if wl_pred > 0 else 0
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
        help="Limit number of windows to process (stratified by fault type)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified sampling when --limit is used (default: 42)",
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
        seed=args.seed,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
