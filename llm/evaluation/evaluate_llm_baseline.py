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
from openai import OpenAI

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.evaluation.metrics import compute_all_metrics, format_metrics_report
from llm.evaluation.schemas import FaultDiagnosis
from llm.evaluation.utils import parse_structured_response, parsed_to_prediction

@dataclass
class LMStudioConfig:
    """Minimal config for LM Studio HTTP API (model_name, base_url)."""
    model_name: str
    base_url: str


def load_lm_studio_model(
    model_name: str = "granite-4.0-h-micro-GGUF",
    base_url: str = "http://localhost:1234/v1",
) -> LMStudioConfig:
    """Return config for LM Studio HTTP API."""
    return LMStudioConfig(model_name=model_name, base_url=base_url)


SYSTEM_PROMPT = (
    "You are an automotive fault diagnostics system. You will receive "
    "structured sensor graph data from an OBD-II vehicle. Identify faulty "
    "sensors and classify the fault type. Only use sensor names from the "
    "provided valid list. Respond with JSON only."
)


def build_baseline_prompt(
    window_idx: int,
    sensor_scores: Dict[str, float],
    sensor_cols: List[str],
    sensor_threshold: float,
) -> List[Dict[str, str]]:
    above_threshold = [
        (name, score) for name, score in sensor_scores.items()
        if score > sensor_threshold
    ]
    above_threshold.sort(key=lambda x: x[1], reverse=True)

    lines = []
    lines.append(f"ANOMALOUS SENSORS (score > {sensor_threshold:.2f}):")
    if above_threshold:
        for name, score in above_threshold:
            lines.append(f"  {name}: anomaly_score={score:.3f}")
    else:
        lines.append("  (none above threshold)")

    lines.append("")
    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors field):")
    lines.append(", ".join(sensor_cols))

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _call_llm_structured(
    client: OpenAI,
    model_name: str,
    messages: List[Dict[str, str]],
) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "FaultDiagnosis",
                "strict": True,
                "schema": FaultDiagnosis.model_json_schema(),
            },
        },
        temperature=0.0,
        max_tokens=256,
    )
    return response.choices[0].message.content


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
):
    """
    Load LLM model via LM Studio HTTP server.

    Args:
        model_repo: Model name in LM Studio (default: granite-4.0-h-micro-GGUF)
        base_url: Base URL for LM Studio HTTP server

    Returns:
        Tuple of (model, tokenizer) where both are the same LMStudioConfig instance
        for backward compatibility with code expecting (model, tokenizer) tuple
    """
    # Convert old MLX model repo format to LM Studio model name if needed
    if model_repo.startswith("mlx-community/"):
        # Extract model name from MLX format
        model_name = model_repo.replace("mlx-community/", "").replace("-8bit", "").replace("-4bit", "")
        model_name = f"{model_name}-GGUF"
    else:
        model_name = model_repo

    print(f"Connecting to LM Studio for model: {model_name}")
    inference = load_lm_studio_model(model_name=model_name, base_url=base_url)
    
    # Return as tuple for backward compatibility
    # Both model and tokenizer are the same LMStudioConfig instance
    return inference, inference


def evaluate_llm_baseline(
    dataset_path: Path,
    output_path: Optional[Path] = None,
    use_statistical_features: bool = True,
    model_repo: Optional[str] = None,
    model_path: Optional[Path] = None,
    base_url: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, any]:
    """
    Evaluate LLM baseline on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        output_path: Optional path to save results JSON
        use_statistical_features: Unused (kept for API compatibility)
        model_repo: Model name in LM Studio
        model_path: Optional GDN checkpoint path for sensor_scores (fair comparison with KAG)
        limit: Optional limit on number of windows to process (for testing)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating LLM Baseline")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"Use statistical features: {use_statistical_features}")
    print()

    # Load LLM model
    if model_repo is None:
        model_repo = "granite-4.0-h-micro-GGUF"
    lm_base_url = base_url or "http://localhost:1234/v1"

    try:
        model, tokenizer = load_llm_model(model_repo, base_url=lm_base_url)
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
    normalized_windows = data.get("normalized_windows", unnormalized_windows)

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

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {unnormalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")
    print()

    # Get sensor_scores and sensor_threshold (for fair comparison with KAG)
    sensor_scores_per_window = []
    sensor_threshold = 0.30
    if model_path is not None:
        try:
            import torch
            from kg.create_kg import GDNPredictor
            checkpoint = torch.load(model_path, map_location="cpu")
            sensor_threshold = float(checkpoint.get("sensor_threshold", 0.30))
            embed_dim = 32
            if "sensor_embeddings" in checkpoint:
                embed_dim = checkpoint["sensor_embeddings"].shape[1]
            predictor = GDNPredictor(
                model_path=str(model_path),
                sensor_names=sensor_names,
                window_size=300,
                embed_dim=embed_dim,
                top_k=3,
                hidden_dim=32,
                device="cpu",
            )
            gdn_preds = predictor.predict(normalized_windows[:num_windows], batch_size=32)
            for wi in range(gdn_preds.shape[0]):
                sensor_scores_per_window.append({
                    sensor_names[i]: float(gdn_preds[wi, i])
                    for i in range(len(sensor_names))
                })
            print(f"  Loaded GDN sensor scores, sensor_threshold={sensor_threshold:.2f}")
        except Exception as e:
            print(f"  ⚠️  GDN load failed: {e}, using empty sensor_scores")
            sensor_scores_per_window = [{} for _ in range(num_windows)]
    else:
        sensor_scores_per_window = [{} for _ in range(num_windows)]
        print(f"  No model_path: using sensor_threshold={sensor_threshold:.2f}, empty sensor_scores")
    print()

    # Create OpenAI client for structured output (LM Studio compatible)
    client = OpenAI(
        base_url=model.base_url,
        api_key="lm-studio",
    )
    model_name = model.model_name

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
            sensor_scores = sensor_scores_per_window[window_idx]
            messages = build_baseline_prompt(
                window_idx, sensor_scores, sensor_names, sensor_threshold
            )
            context_length = sum(len(m.get("content", "").split()) for m in messages)

            try:
                raw_json = _call_llm_structured(client, model_name, messages)
                parsed = parse_structured_response(raw_json, sensor_names)
                prediction = parsed_to_prediction(parsed, sensor_names)
            except Exception as e:
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
    fault_types_true = data.get("fault_types", None)

    # Compute metrics using filtered (root-only) predictions for precision improvement
    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,  # Use filtered (root-only) for main metrics
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    
    # Also compute raw metrics for comparison
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred_raw,  # Use raw (all sensors) for comparison
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]

    # Add efficiency metrics
    metrics["efficiency"] = {
        "avg_processing_time_seconds": float(avg_processing_time),
        "total_processing_time_seconds": float(total_processing_time),
        "windows_per_second": float(num_windows / total_processing_time),
        "avg_context_length_tokens": float(avg_context_length),
        "use_statistical_features": use_statistical_features,
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

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results


def run(dataset_path: str, model_path: str, lm_url: str, limit: Optional[int] = None) -> dict:
    """
    Run LLM baseline evaluation and return unified format for compare_methods.

    Returns dict with "results" (per-window list) and "metrics" (unified format).
    """
    from llm.evaluation.metrics import compute_all_metrics_unified

    res = evaluate_llm_baseline(
        dataset_path=Path(dataset_path),
        output_path=None,
        model_path=Path(model_path) if model_path else None,
        base_url=lm_url,
        limit=limit,
    )
    preds = res["predictions"]
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    fault_types = data.get("fault_types", None)
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
        "--dataset", type=str, required=True, help="Path to shared dataset .npz file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/llm_baseline.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--use-statistical-features",
        action="store_true",
        default=True,
        help="Include statistical features in LLM prompts",
    )
    parser.add_argument(
        "--model-repo",
        type=str,
        default="granite-4.0-h-micro-GGUF",
        help="Model name in LM Studio (default: granite-4.0-h-micro-GGUF)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional GDN checkpoint for sensor_scores (fair comparison with KAG)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )

    args = parser.parse_args()

    evaluate_llm_baseline(
        dataset_path=Path(args.dataset),
        output_path=Path(args.output),
        use_statistical_features=args.use_statistical_features,
        model_repo=args.model_repo,
        model_path=Path(args.model_path) if args.model_path else None,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
