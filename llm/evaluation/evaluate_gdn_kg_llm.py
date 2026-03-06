#!/usr/bin/env python3
"""
Evaluate Serialised KG->LLM method on shared evaluation dataset.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import sys
import torch
from tqdm import tqdm

# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.inference import (
    create_client,
    create_json_schema_response_format,
    LMStudioClient,
)

from kg.create_kg import (
    KnowledgeGraph,
    EXPECTED_CORRELATIONS,
    SENSOR_SUBSYSTEMS,
    SENSOR_DESCRIPTIONS,
)
from kg.create_kg import GDNPredictor
from llm.evaluation.evaluate_llm_baseline import filter_sensor_labels_to_root_only
from llm.evaluation.schemas import FaultDiagnosis
from llm.evaluation.utils import parse_structured_response, parsed_to_prediction


SYSTEM_PROMPT = """You are an expert automotive fault diagnostics system.
You will receive structured sensor graph data from an OBD-II vehicle.
Your task is to identify faulty sensors and classify the fault type.
Only use sensor names from the provided valid list. Respond with JSON only.
"""


def build_kg_prompt(
    kg_context: Dict[str, Any],
    sensor_scores: Dict[str, float],
    sensor_names: List[str],
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    """Format KG context into LLM messages for fault diagnosis."""
    lines = ["KNOWLEDGE GRAPH CONTEXT (sensor correlations, violations, propagation):", ""]

    lines.append("SENSOR ANOMALY SCORES (GDN model output, 0.0-1.0):")
    for name in sensor_names:
        score = sensor_scores.get(name, 0.0)
        thr = (sensor_thresholds or {}).get(name, sensor_threshold)
        flag = " [ANOMALOUS]" if score > thr else ""
        lines.append(f"  {name}: {score:.3f}{flag}")
    lines.append("")

    violations = kg_context.get("violations", [])
    if violations:
        lines.append("CORRELATION VIOLATIONS (expected vs actual):")
        for v in violations:
            src = v.get("source", "")
            tgt = v.get("target", "")
            exp = v.get("expected_correlation_gdn", 0)
            act = v.get("correlation", 0)
            lines.append(f"  {src} <-> {tgt}: expected={exp:.3f}, actual={act:.3f}")
        lines.append("")

    propagation = kg_context.get("anomaly_propagation", [])
    if propagation:
        lines.append("ANOMALY PROPAGATION:")
        for p in propagation:
            root = p.get("root_sensor", "")
            if root:
                lines.append(f"  Root sensor: {root}")
        lines.append("")

    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors field):")
    lines.append(", ".join(sensor_names))
    content = "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
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
        strict=True,
    )

    response = client.chat_completions_create(
        model=model_name,
        messages=messages,
        response_format=response_format,
        temperature=0.0,
        max_tokens=2048,
    )
    return response["choices"][0]["message"]["content"]


def _load_checkpoint_state(
    path: str, device: str = "cpu"
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Load a Stage 1 checkpoint and return base model state dict plus metadata.

    Stage 1 checkpoints may contain:
    - base_model_state_dict (from GDNWithForecasting wrapper),
    - model_state_dict directly,
    - plain state dict.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    metadata = dict(ckpt) if isinstance(ckpt, dict) else {}

    if isinstance(ckpt, dict):
        if "base_model_state_dict" in ckpt:
            raw_state = ckpt["base_model_state_dict"]
        elif "model_state_dict" in ckpt:
            raw_state = ckpt["model_state_dict"]
        else:
            raw_state = ckpt
    else:
        raw_state = ckpt

    # Remove wrapper prefix if present.
    state_dict = {}
    has_base_prefix = any(k.startswith("base_model.") for k in raw_state.keys())
    if has_base_prefix:
        for k, v in raw_state.items():
            new_key = k[len("base_model.") :] if k.startswith("base_model.") else k
            state_dict[new_key] = v
    else:
        state_dict = dict(raw_state)

    # Global head redesign: old checkpoints use Linear(num_nodes * hidden_dim, 64),
    # new model uses Linear(hidden_dim + num_nodes, 64). Drop old global head so it
    # stays randomly initialized.
    for k in list(state_dict.keys()):
        if k.startswith("global_classifier."):
            state_dict.pop(k)

    return state_dict, metadata


def run_kg_sanity_check(
    dataset_path: Path,
    model_path: Path,
    batch_size: int = 32,
    device: str = "cpu",
    sample_windows: Optional[List[int]] = None,
) -> None:
    """
    Sanity check: verify KG context is non-trivial before full eval.
    Confirms violations/propagation_chain are populated and KG accumulates across windows.
    """
    print("=" * 80)
    print("KG Sanity Check (run before full eval)")
    print("=" * 80)

    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
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

    num_windows = normalized_windows.shape[0]
    num_sensors = len(sensor_names)
    fault_rate = (sensor_labels_true.sum(axis=1) > 0).mean()

    print(f"Dataset: {num_windows} windows, {num_sensors} sensors")
    print(f"Fault rate: {fault_rate:.1%}")
    print()

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        sensor_threshold = float(checkpoint.get("sensor_threshold", 0.30))
        embed_dim = 32
        if "sensor_embeddings" in checkpoint:
            detected_embed_dim = checkpoint["sensor_embeddings"].shape[1]

        predictor = GDNPredictor(
            model_path=str(model_path),
            sensor_names=sensor_names,
            window_size=300,
            embed_dim=detected_embed_dim if "sensor_embeddings" in checkpoint else 32,
            top_k=3,
            hidden_dim=32,
            device=device,
        )

        # Sample windows for analysis
        if sample_windows:
            sample_indices = np.array(sample_windows)
            if sample_indices.max() >= num_windows:
                sample_indices = sample_indices[sample_indices < num_windows]
            normalized_windows = normalized_windows[sample_indices]
            sensor_labels_true = sensor_labels_true[sample_indices]
            window_labels_true = window_labels_true[sample_indices]
            num_windows = len(sample_indices)
            sample_windows = sample_indices.tolist()
        else:
            sample_indices = np.arange(num_windows)
            sample_windows = sample_indices.tolist()

        kg_data = predictor.process_for_kg(
            X_windows=normalized_windows,
            sensor_labels=sensor_labels_true,
            window_labels=window_labels_true,
            batch_size=batch_size,
        )

        kg = KnowledgeGraph(
            sensor_names=kg_data["sensor_names"],
            sensor_embeddings=kg_data["sensor_embeddings"],
            adjacency_matrix=kg_data["adjacency_matrix"],
        )

        kg.construct(
            X_windows=kg_data["X_windows"],
            gdn_predictions=kg_data["gdn_predictions"],
            X_windows_unnormalized=kg_data.get("X_windows_unnormalized"),
            sensor_labels_true=sensor_labels_true,
            window_labels_true=window_labels_true,
            propagation_threshold=kg_data.get("propagation_threshold"),
        )

        n_nodes = kg.kg.number_of_nodes()
        n_edges = kg.kg.number_of_edges()
        n_window_graphs = len(kg.window_graphs)
        n_prop_chains = len(kg.anomaly_propagation_chains)

        print(f"Nodes: {n_nodes}, Edges: {n_edges}")
        print(f"Per-window graphs: {n_window_graphs} (should equal {num_windows})")
        print(f"Anomaly propagation chains: {n_prop_chains}")

        if n_window_graphs > 0 and n_prop_chains > 0:
            print("OK: KG context appears non-trivial (anomaly propagation present).")
        else:
            print("Warning: KG context may be trivial (few or no propagation chains).")
    except Exception as e:
        print(f"Error during KG sanity check: {e}")
    finally:
        print("=" * 80)


def evaluate_gdn_only(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = "cpu",
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate GDN predictions only (no KG, no LLM).

    This is useful for baseline comparison of GDN model's
    anomaly detection performance without downstream components.

    Uses:
    - normalized_windows for GDN input
    - sensor_labels and window_is_faulty as ground truth
    - Calibrated thresholds from checkpoint (if available)
    - fault_types from dataset (no conversion needed)
    """
    from kg.create_kg import GDNPredictor

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_is_faulty_true = data["window_is_faulty"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
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

    num_windows = normalized_windows.shape[0]

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_is_faulty_true = window_is_faulty_true[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")

    # Load GDN checkpoint and detect parameters
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    sensor_threshold = float(checkpoint.get("sensor_threshold", 0.30))
    embed_dim = 32
    if "sensor_embeddings" in checkpoint:
        embed_dim = checkpoint["sensor_embeddings"].shape[1]

    # Initialize GDN predictor
    predictor = GDNPredictor(
        model_path=str(model_path),
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=embed_dim,
        top_k=3,
        hidden_dim=32,
        device=device,
    )

    # Run GDN inference
    print("Running GDN inference...")
    gdn_preds = predictor.predict(normalized_windows[:num_windows], batch_size=32)

    # Threshold predictions
    sensor_labels_pred = (gdn_preds > sensor_threshold).astype(np.float32)
    window_labels_pred = (sensor_labels_pred.sum(axis=1) > 0).astype(int)

    # Compute metrics
    from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

    metrics = compute_all_metrics(
        y_true_window=window_is_faulty_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
    )

    print(format_metrics_report(metrics))

    # Save results
    results = {
        "method": "gdn_only",
        "dataset": str(dataset_path),
        "gdn_model": str(model_path),
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results


def evaluate_gdn_kg_llm(
    dataset_path: Path,
    model_path: Path,
    output_path: Optional[Path] = None,
    batch_size: int = 32,
    device: str = "cpu",
    model_repo: Optional[str] = None,
    base_url: Optional[str] = None,
    max_tokens: Optional[int] = None,  # None = no limit (model supports 128k context)
    temperature: float = 0.7,
    limit: Optional[int] = None,
    use_embeddings: bool = True,
    use_adjacency_matrix: bool = False,  # Use compact adjacency matrix format instead of verbose text
) -> Dict[str, Any]:
    """
    Evaluate Serialised KG->LLM method on shared dataset.

    This script:
    1. Loads shared dataset
    2. Processes normalized windows through GDN->KG pipeline
    3. Extracts KG context for each window
    4. Formats KG-enhanced prompts for LLM
    5. Runs LLM inference with KG context
    6. Compares predictions to ground truth
    7. Computes evaluation metrics

    Args:
        dataset_path: Path to shared dataset (.npz file)
        model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        model_repo: LLM model repository identifier
        max_tokens: Maximum tokens for LLM generation (None = no limit)
        temperature: LLM sampling temperature
        limit: Optional limit on number of windows to process (for testing)

    Returns:
        Dictionary with evaluation results
    """
    print("=" * 80)
    print("Evaluating Serialised KG->LLM Method")
    print("=" * 80)
    print(f"Dataset: {dataset_path}")
    print(f"GDN Model: {model_path}")
    print()

    # Create LM Studio client
    model_name = model_repo or "granite-4.0-h-micro"
    lm_base_url = base_url or "http://localhost:1234/v1"

    try:
        client = create_client(model_name=model_name, base_url=lm_base_url, timeout=60)
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to LM Studio: {e}. "
            f"Please ensure LM Studio is running with the HTTP server enabled."
        )

    # Load dataset
    print("Loading dataset...")
    data = np.load(dataset_path, allow_pickle=True)

    normalized_windows = data["normalized_windows"]
    unnormalized_windows = data["unnormalized_windows"]
    sensor_labels_true = data["sensor_labels"]
    window_labels_true = data["window_labels"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
        statistical_features = data.get("statistical_features", None)
    else:
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

    num_windows = normalized_windows.shape[0]

    # Apply limit if specified
    if limit is not None and limit > 0:
        num_windows = min(num_windows, limit)
        normalized_windows = normalized_windows[:num_windows]
        unnormalized_windows = unnormalized_windows[:num_windows]
        sensor_labels_true = sensor_labels_true[:num_windows]
        window_labels_true = window_labels_true[:num_windows]
        if statistical_features is not None:
            statistical_features = statistical_features[:num_windows]
        print(f"  ⚠️  LIMIT MODE: Processing only {num_windows} windows")
    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")

    # Load checkpoint and detect parameters (prefer stage2 calibrated thresholds)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    calibrated = checkpoint.get("calibrated_thresholds") or {}
    sensor_threshold = float(
        calibrated.get("sensor", checkpoint.get("sensor_threshold", 0.5))
    )
    embed_dim = 32
    if "sensor_embeddings" in checkpoint:
        embed_dim = checkpoint["sensor_embeddings"].shape[1]
        print(f"  Detected embed_dim from checkpoint: {embed_dim}")

    # Initialize GDN predictor
    predictor = GDNPredictor(
        model_path=str(model_path),
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=embed_dim,
        top_k=7,
        hidden_dim=32,
        device=device,
        sensor_threshold=sensor_threshold,
    )

    # Run GDN inference for KG context (process_for_kg returns full KG inputs)
    print("Running GDN inference for KG context...")
    windows_for_kg = normalized_windows[:limit] if limit and limit > 0 else normalized_windows
    unnorm_for_kg = unnormalized_windows[:limit] if limit and limit > 0 else unnormalized_windows
    sl_true_kg = sensor_labels_true[:limit] if limit and limit > 0 else sensor_labels_true
    wl_true_kg = window_labels_true[:limit] if limit and limit > 0 else window_labels_true
    if limit and limit > 0:
        print(f"  Limited to {limit} windows for KG context")

    kg_preds = predictor.process_for_kg(windows_for_kg, batch_size=32)

    # Build knowledge graph
    kg = KnowledgeGraph(
        sensor_names=sensor_names,
        sensor_embeddings=kg_preds["sensor_embeddings"],
        adjacency_matrix=kg_preds["adjacency_matrix"] if use_adjacency_matrix else None,
    )

    kg.construct(
        X_windows=kg_preds["X_windows"],
        gdn_predictions=kg_preds["gdn_predictions"],
        X_windows_unnormalized=unnorm_for_kg,
        sensor_labels_true=sl_true_kg,
        window_labels_true=wl_true_kg,
        propagation_threshold=kg_preds.get("propagation_threshold"),
    )

    n_nodes = kg.kg.number_of_nodes()
    n_edges = kg.kg.number_of_edges()
    n_window_graphs = len(kg.window_graphs)
    n_prop_chains = len(kg.anomaly_propagation_chains)

    print(f"Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"Per-window graphs: {n_window_graphs} (should equal {num_windows})")
    print(f"Anomaly propagation chains: {n_prop_chains}")

    if n_window_graphs > 0 and n_prop_chains > 0:
        print("OK: KG context appears non-trivial (anomaly propagation present).")
    else:
        print("Warning: KG context may be trivial (few or no propagation chains).")
    print()

    # Run LLM predictions with KG context
    from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

    model_name = model_repo or "granite-4.0-h-micro"
    print("Running LLM predictions with KG context...")
    window_labels_pred = []
    sensor_labels_pred = []
    sensor_labels_pred_raw = []
    fault_types_pred = []
    reasoning_list = []
    processing_times = []

    with tqdm(total=num_windows, desc="KG-LLM", unit="window", dynamic_ncols=True) as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            kg_context = kg.get_window_kg(window_idx, temporal_context_windows=2)
            window_stats = kg.window_stats.get(window_idx, {})
            sensor_scores = {
                name: float(stats.anomaly_score)
                for name, stats in window_stats.items()
            }
            sensor_thresholds_dict = None
            messages = build_kg_prompt(
                kg_context,
                sensor_scores,
                sensor_names,
                sensor_threshold,
                sensor_thresholds=sensor_thresholds_dict,
            )
            try:
                raw_json = _call_llm_structured(client, model_name, messages)
                parsed = parse_structured_response(raw_json, sensor_names)
                prediction = parsed_to_prediction(parsed, sensor_names)
            except Exception as e:
                empty_labels = np.zeros(len(sensor_names), dtype=np.float32)
                prediction = {
                    "window_label": 0,
                    "sensor_labels": empty_labels,
                    "sensor_labels_raw": empty_labels.copy(),
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
            pbar.update(1)
            if (window_idx + 1) % 10 == 0:
                avg_time = (
                    np.mean(processing_times[-10:])
                    if len(processing_times) >= 10
                    else np.mean(processing_times)
                )
                pbar.set_postfix({"avg_time": f"{avg_time:.2f}s"})

    window_labels_pred = np.array(window_labels_pred)
    sensor_labels_pred = np.array(sensor_labels_pred)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)
    llm_time = float(np.sum(processing_times))

    print(f"  Average processing time: {np.mean(processing_times):.4f} seconds/window")
    print(f"  Total LLM processing time: {llm_time:.2f} seconds")
    print()

    # Convert window_labels_true to sensor-indexed format for metrics
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = faulty_indices[0] + 1
        else:
            window_labels_true_converted[i] = 0
    window_labels_true = window_labels_true_converted

    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = data.get("fault_types", None)
    gdn_time = 0.0
    kg_time = 0.0

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics_raw = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred_raw,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
    )
    metrics["sensor_level_raw"] = metrics_raw["sensor_level"]
    metrics["efficiency"] = {
        "gdn_processing_time_seconds": float(gdn_time),
        "kg_build_time_seconds": float(kg_time),
        "llm_processing_time_seconds": float(llm_time),
        "total_processing_time_seconds": float(gdn_time + kg_time + llm_time),
        "windows_per_second": float(num_windows / (gdn_time + kg_time + llm_time + 1e-8)),
        "kg_nodes": int(kg.number_of_nodes()),
        "kg_edges": int(kg.number_of_edges()),
    }

    print(format_metrics_report(metrics))

    results = {
        "method": "gdn_kg_llm",
        "dataset": str(dataset_path),
        "gdn_model": str(model_path),
        "llm_model": model_repo,
        "num_windows": int(num_windows),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
            "sensor_labels_raw": sensor_labels_pred_raw.tolist(),
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

    print("=" * 80)
    return results


def run(
    dataset_path: str, model_path: str, lm_url: str, limit: Optional[int] = None
) -> dict:
    """
    Run GDN-KG-LLM evaluation and return unified format for compare_methods.

    Returns dict with "results" (per-window list) and "metrics" (unified format).
    """
    from llm.evaluation.metrics import compute_all_metrics_unified

    res = evaluate_gdn_kg_llm(
        dataset_path=Path(dataset_path),
        model_path=Path(model_path),
        output_path=None,
        base_url=lm_url,
        limit=limit,
    )

    preds = res["predictions"]
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    fault_types = data.get("fault_types", None)
    metadata_path = (
        Path(dataset_path).parent / f"{Path(dataset_path).stem}_metadata.json"
    )
    sensor_names = []
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        sensor_names = meta["dataset_info"]["sensor_names"]
    ref_reason = data.get("reference_reasoning", None)
    if ref_reason is not None and hasattr(ref_reason, "tolist"):
        ref_reason = ref_reason.tolist()
    else:
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
        ft_true = (
            fault_types[i] if fault_types is not None and i < len(fault_types) else None
        )
        ft_true_str = (
            "normal"
            if (ft_true is None or ft_true == "" or wl_true == 0)
            else str(ft_true)
        )
        ft_pred_str = "normal" if (ft_pred is None or ft_pred == "") else str(ft_pred)
        ref = ref_reason[i] if i < len(ref_reason) else ""

        results.append(
            {
                "window_label_true": int(wl_true),
                "window_label_pred": int(wl_pred_bin),
                "sensor_labels_true": sl_true,
                "sensor_labels_pred": [float(x) for x in sl_pred],
                "fault_type_true": ft_true_str,
                "fault_type_pred": ft_pred_str,
                "reasoning": reasoning or "",
                "reference_reasoning": ref or "",
            }
        )
    metrics = compute_all_metrics_unified(results, sensor_names)
    return {"results": results, "metrics": metrics, "sensor_cols": sensor_names}


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Serialised KG->LLM method on shared evaluation dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/shared_dataset/test.npz",
        help="Path to shared dataset .npz file (default: data/shared_dataset/test.npz)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained GDN model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/gdn_kg_llm.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for GDN inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--model-repo", type=str, default=None, help="LLM model repository identifier"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        nargs="?",
        help="Maximum tokens for LLM generation (default: None = no limit)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="LLM sampling temperature"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of windows to process (for testing)",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        default=True,
        help="Enable embedding extraction and similarity computation (default: True)",
    )
    parser.add_argument(
        "--no-embeddings",
        dest="use_embeddings",
        action="store_false",
        help="Disable embedding extraction",
    )
    parser.add_argument(
        "--use-adjacency-matrix",
        action="store_true",
        default=False,
        help="Use compact adjacency matrix format instead of verbose text format for KG representation",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run KG sanity check only (verify violations/propagation non-trivial, then exit)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "gdn_only"],
        default="full",
        help="Evaluation mode: full=KG-LLM pipeline, gdn_only=GDN predictions only",
    )

    args = parser.parse_args()

    if args.sanity_check:
        run_kg_sanity_check(
            dataset_path=Path(args.dataset),
            model_path=Path(args.model_path),
            batch_size=args.batch_size,
            device=args.device,
            sample_windows=[100, 500, 1000],
        )
        return

    if args.mode == "gdn_only":
        evaluate_gdn_only(
            dataset_path=Path(args.dataset),
            model_path=Path(args.model_path),
            output_path=Path(args.output),
            batch_size=args.batch_size,
            device=args.device,
            limit=args.limit,
        )
        return

    evaluate_gdn_kg_llm(
        dataset_path=Path(args.dataset),
        model_path=Path(args.model_path),
        output_path=Path(args.output),
        batch_size=args.batch_size,
        device=args.device,
        model_repo=args.model_repo,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        limit=args.limit,
        use_embeddings=args.use_embeddings,
        use_adjacency_matrix=args.use_adjacency_matrix,
    )


if __name__ == "__main__":
    main()
