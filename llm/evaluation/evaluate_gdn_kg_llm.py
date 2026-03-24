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

from llm.inference import create_client

from kg.create_kg import KnowledgeGraph, require_stage2_per_sensor_thresholds
from kg.create_kg import GDNPredictor
from llm.evaluation.evaluate_llm_baseline import filter_sensor_labels_to_root_only
from llm.evaluation.stratified_sampling import (
    stratified_sample_indices,
    validate_fault_types_for_stratification,
)
from llm.evaluation.utils import call_llm_fault_diagnosis
from llm.evaluation.evaluate_llm_baseline import FAULT_TYPE_DESCRIPTIONS

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

SYSTEM_PROMPT = """You are an automotive OBD-II fault diagnostics assistant working alongside a pre-trained GDN anomaly detection model.

The GDN model's per-sensor thresholds are calibrated. Its verdict is authoritative:

  NORMAL — no sensor exceeded its threshold in this window.
           Set is_faulty=false, faulty_sensors=[], and fault_type="normal". Do not override.
  FAULT  — one or more sensors exceeded their threshold.
           Set is_faulty=true, list the root cause sensor(s) in faulty_sensors, and set fault_type.

FAULT output rules:
- faulty_sensors[0] is the PRIMARY root cause — the [ANOMALOUS] sensor with the highest margin.
- Add secondary sensors only if a listed violation directly implicates them in this window.
- faulty_sensors MUST NOT be empty when is_faulty=true. If uncertain, use the highest-margin [ANOMALOUS] sensor.
- Use only names from VALID SENSOR NAMES.
- fault_type MUST be exactly one value from VALID FAULT TYPES (shown in the user message).
- To choose the correct fault_type, match the signal pattern in ANOMALOUS SENSOR SIGNAL SUMMARY against
  the fault type descriptions. Do not choose based on list position — choose based on evidence.

Output: one JSON object matching the schema (no markdown, no extra prose).
"""


def _sensor_anomalous(
    name: str,
    sensor_scores: Dict[str, float],
    sensor_thresholds: Dict[str, float],
) -> bool:
    return float(sensor_scores.get(name, 0.0)) > float(sensor_thresholds[name])


def _violation_deviation_magnitude(v: Dict[str, Any]) -> float:
    raw = v.get("deviation_from_gdn")
    if raw is not None:
        try:
            f = float(raw)
            if not np.isnan(f):
                return abs(f)
        except (TypeError, ValueError):
            pass
    exp = float(v.get("expected_correlation_gdn", 0.0))
    act = float(v.get("correlation", 0.0))
    return abs(act - exp)


def _format_endpoint_gdn(
    sensor_name: str,
    gdn_score: float,
    sensor_scores: Dict[str, float],
    sensor_thresholds: Dict[str, float],
) -> str:
    thr = float(sensor_thresholds.get(sensor_name, 0.0))
    s = float(gdn_score)
    ann = " [ANOMALOUS]" if s > thr else ""
    return f"{sensor_name}: gdn={s:.3f}{ann}"


def build_kg_prompt(
    kg_context: Dict[str, Any],
    sensor_scores: Dict[str, float],
    sensor_names: List[str],
    sensor_thresholds: Dict[str, float],
    max_violations: Optional[int] = None,
    window_data_unnorm: Optional[np.ndarray] = None,
    sensor_population_stats: Optional[Dict[str, Dict[str, float]]] = None,
) -> List[Dict[str, str]]:
    """
    Format GDN scores and KG context into LLM messages for fault diagnosis.

    Two-branch design:
      NORMAL  — no [ANOMALOUS] sensor and no violations: compact prompt, no candidates.
      FAULT   — one or more anomalous sensors or violations: full prompt with violations,
                candidates, and (when available) GDN-gated signal summaries with population
                baselines for anomalous sensors only.

    window_data_unnorm: (window_size, num_sensors) unnormalized sensor values for this window.
    sensor_population_stats: per-sensor dict with keys mean/std/p5/p95 from normal population.
    """
    missing = [n for n in sensor_names if n not in sensor_thresholds]
    if missing:
        raise KeyError(
            "sensor_thresholds must include every sensor name (Stage-2 per-sensor τ). "
            f"Missing: {missing}"
        )

    violations_all = list(kg_context.get("violations", []))
    violations_sorted = sorted(
        violations_all,
        key=_violation_deviation_magnitude,
        reverse=True,
    )
    violations_display = (
        violations_sorted[:max_violations]
        if (max_violations is not None and max_violations > 0)
        else violations_sorted
    )

    n_anomalous = sum(
        1 for name in sensor_names
        if _sensor_anomalous(name, sensor_scores, sensor_thresholds)
    )
    is_normal = n_anomalous == 0 and len(violations_display) == 0

    lines: List[str] = []

    if is_normal:
        lines.append("GDN VERDICT: NORMAL")
        lines.append("")
        lines.append("SENSOR ANOMALY SCORES (score / threshold / margin):")
        for name in sensor_names:
            score = float(sensor_scores.get(name, 0.0))
            thr = float(sensor_thresholds[name])
            margin = score - thr
            lines.append(f"  {name}: {score:.3f} / {thr:.3f} / {margin:+.3f}")
    else:
        lines.append(f"GDN VERDICT: FAULT DETECTED ({n_anomalous} sensor(s) above threshold)")
        lines.append("")
        lines.append("SENSOR ANOMALY SCORES (score / threshold / margin):")
        for name in sensor_names:
            score = float(sensor_scores.get(name, 0.0))
            thr = float(sensor_thresholds[name])
            margin = score - thr
            flag = " [ANOMALOUS]" if margin > 0 else ""
            lines.append(f"  {name}: {score:.3f} / {thr:.3f} / {margin:+.3f}{flag}")

        if violations_display:
            lines.append("")
            cap_note = (
                f" (top {len(violations_display)} of {len(violations_sorted)} by deviation)"
                if (max_violations is not None and len(violations_sorted) > len(violations_display))
                else ""
            )
            lines.append(f"GDN CORRELATION VIOLATIONS (strongest first){cap_note}:")
            for v in violations_display:
                src = v.get("source", "")
                tgt = v.get("target", "")
                exp = float(v.get("expected_correlation_gdn", 0.0))
                act = float(v.get("correlation", 0.0))
                dev = _violation_deviation_magnitude(v)
                gs = float(v.get("gdn_score_source", sensor_scores.get(src, 0.0)))
                gt = float(v.get("gdn_score_target", sensor_scores.get(tgt, 0.0)))
                left = _format_endpoint_gdn(src, gs, sensor_scores, sensor_thresholds)
                right = _format_endpoint_gdn(tgt, gt, sensor_scores, sensor_thresholds)
                lines.append(f"  {left} <-> {right} | exp={exp:.3f} act={act:.3f} dev={dev:.3f}")

        # Candidates: anomalous sensors + violation endpoints from this window only.
        # Propagation chains are dataset-level artifacts and are intentionally excluded.
        candidate_set: set = set()
        for name in sensor_names:
            if _sensor_anomalous(name, sensor_scores, sensor_thresholds):
                candidate_set.add(name)
        for v in violations_display:
            for key in ("source", "target"):
                s = v.get(key, "")
                if s:
                    candidate_set.add(s)
        if candidate_set:
            lines.append("")
            lines.append("CANDIDATE SENSORS (anomalous + violation endpoints, this window only):")
            lines.append("  " + ", ".join(sorted(candidate_set)))

        # GDN-gated signal summary: only for [ANOMALOUS] sensors, anchored to population baseline.
        # This gives the LLM concrete signal evidence to distinguish between fault types
        # without receiving the full raw time series for all sensors.
        anomalous_sensors = [
            name for name in sensor_names
            if _sensor_anomalous(name, sensor_scores, sensor_thresholds)
        ]
        if anomalous_sensors and window_data_unnorm is not None and sensor_population_stats:
            lines.append("")
            lines.append("ANOMALOUS SENSOR SIGNAL SUMMARY (this window vs. normal population):")
            for name in anomalous_sensors:
                idx = sensor_names.index(name)
                vals = window_data_unnorm[:, idx].astype(float)
                h1, h2 = vals[: len(vals) // 2], vals[len(vals) // 2 :]
                pop = sensor_population_stats.get(name, {})
                lines.append(f"  {name}:")
                lines.append(
                    f"    this window:  mean={vals.mean():.2f}  min={vals.min():.2f}"
                    f"  max={vals.max():.2f}  std={vals.std():.2f}"
                )
                lines.append(
                    f"    first half:   mean={h1.mean():.2f}  std={h1.std():.2f}"
                )
                lines.append(
                    f"    second half:  mean={h2.mean():.2f}  std={h2.std():.2f}"
                )
                if pop:
                    lines.append(
                        f"    normal range: mean={pop['mean']:.2f}  std={pop['std']:.2f}"
                        f"  p5={pop['p5']:.2f}  p95={pop['p95']:.2f}"
                    )

    lines.append("")
    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors):")
    lines.append(", ".join(sensor_names))
    lines.append("")
    lines.append("VALID FAULT TYPES (use exactly one in fault_type; use 'normal' if no fault):")
    for ft, desc in FAULT_TYPE_DESCRIPTIONS.items():
        lines.append(f"  {ft}: {desc}")
    lines.append("  normal: no fault detected in this window")

    content = "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


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
        sensor_names = list(DEFAULT_SENSOR_NAMES)

    num_windows = normalized_windows.shape[0]
    num_sensors = len(sensor_names)
    fault_rate = (sensor_labels_true.sum(axis=1) > 0).mean()

    print(f"Dataset: {num_windows} windows, {num_sensors} sensors")
    print(f"Fault rate: {fault_rate:.1%}")
    print()

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        calibrated = checkpoint.get("calibrated_thresholds") or {}
        per_sensor_thr_list = require_stage2_per_sensor_thresholds(
            calibrated.get("per_sensor"),
            sensor_names,
            context="run_kg_sanity_check: ",
        )
        detected_embed_dim = 32
        if "sensor_embeddings" in checkpoint:
            detected_embed_dim = checkpoint["sensor_embeddings"].shape[1]

        predictor = GDNPredictor(
            model_path=str(model_path),
            sensor_names=sensor_names,
            window_size=300,
            embed_dim=detected_embed_dim,
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
            propagation_per_sensor_thresholds=per_sensor_thr_list,
            X_windows_unnormalized=kg_data.get("X_windows_unnormalized"),
            sensor_labels_true=sensor_labels_true,
            window_labels_true=window_labels_true,
        )

        sanity_contexts = kg.precompute_window_contexts(
            num_windows,
            per_sensor_thr_list,
            temporal_context_windows=2,
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

        violation_breakdown = {"total_violations": 0}
        for idx in range(num_windows):
            ctx = sanity_contexts[idx]
            for _ in ctx.get("violations", []):
                violation_breakdown["total_violations"] += 1
        print("Violation breakdown (GDN correlation violations in prompts):")
        print(f"  total: {violation_breakdown['total_violations']}")
        print()

        # Per-faulty-window diagnostic: are violations and anomaly_propagation populated?
        faulty_mask = (sensor_labels_true.sum(axis=1) > 0)
        n_faulty = int(faulty_mask.sum())
        if n_faulty > 0:
            n_with_violations = 0
            n_with_propagation = 0
            n_with_v_or_p = 0
            n_with_both = 0
            for idx in range(num_windows):
                if not faulty_mask[idx]:
                    continue
                ctx = sanity_contexts[idx]
                has_v = len(ctx.get("violations", [])) > 0
                has_p = len(ctx.get("anomaly_propagation", [])) > 0
                if has_v:
                    n_with_violations += 1
                if has_p:
                    n_with_propagation += 1
                if has_v or has_p:
                    n_with_v_or_p += 1
                if has_v and has_p:
                    n_with_both += 1
            print()
            print("Per-faulty-window KG context (LLM prompt substance):")
            print(f"  Faulty windows: {n_faulty}")
            print(f"  With violations: {n_with_violations} ({100 * n_with_violations / n_faulty:.1f}%)")
            print(f"  With anomaly_propagation: {n_with_propagation} ({100 * n_with_propagation / n_faulty:.1f}%)")
            print(f"  With violations or propagation: {n_with_v_or_p} ({100 * n_with_v_or_p / n_faulty:.1f}%)")
            print(f"  With both: {n_with_both} ({100 * n_with_both / n_faulty:.1f}%)")
            if n_with_v_or_p < n_faulty * 0.5:
                print("  ⚠️  Sparse KG context: many faulty windows lack violations/propagation.")
                print("      LLM gets prompt structure but little substance → poor fault typing.")
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
    stratify_limit: bool = True,
    stratified_limit_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate GDN predictions only (no KG, no LLM).

    This is useful for baseline comparison of GDN model's
    anomaly detection performance without downstream components.

    Uses:
    - normalized_windows for GDN input
    - sensor_labels as ground truth (window metrics use first-faulty sensor index 0..D, same as KG+LLM eval)
    - Calibrated thresholds from checkpoint (if available)
    - fault_types from dataset when present (for per-fault-type metrics)
    """
    from kg.create_kg import GDNPredictor

    # Load dataset
    data = np.load(dataset_path, allow_pickle=True)
    normalized_windows = data["normalized_windows"]
    sensor_labels_true = data["sensor_labels"]

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = list(DEFAULT_SENSOR_NAMES)

    full_num_windows = int(normalized_windows.shape[0])
    fault_types_arr = data.get("fault_types", None)
    fault_types_for_metrics: Optional[np.ndarray] = fault_types_arr
    sample_indices_for_results: Optional[np.ndarray] = None

    if limit is not None and limit > 0:
        if limit < full_num_windows:
            if stratify_limit:
                if fault_types_arr is None:
                    raise ValueError(
                        "Stratified limit requires fault_types in the dataset. "
                        "The .npz file must contain a 'fault_types' array."
                    )
                validate_fault_types_for_stratification(
                    fault_types_arr, sensor_labels_true
                )
                sample_indices_for_results = stratified_sample_indices(
                    fault_types_arr, limit, random_state=stratified_limit_seed
                )
                ix = sample_indices_for_results
                normalized_windows = normalized_windows[ix]
                sensor_labels_true = sensor_labels_true[ix]
                fault_types_for_metrics = fault_types_arr[ix]
                num_windows = int(len(ix))
                print(
                    f"  ⚠️  LIMIT MODE: stratified sample of {num_windows} windows "
                    f"(seed={stratified_limit_seed})"
                )
            else:
                ix = np.arange(limit, dtype=np.int64)
                normalized_windows = normalized_windows[ix]
                sensor_labels_true = sensor_labels_true[ix]
                if fault_types_arr is not None:
                    fault_types_for_metrics = fault_types_arr[ix]
                num_windows = int(limit)
                print(
                    f"  ⚠️  LIMIT MODE: first {num_windows} windows (--no-stratify-limit)"
                )
        else:
            num_windows = full_num_windows
            print(
                f"  LIMIT ({limit}) ≥ dataset size: using all {num_windows} windows"
            )
    else:
        num_windows = full_num_windows

    # Load GDN checkpoint and detect parameters (prefer stage2 calibrated thresholds)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    calibrated = checkpoint.get("calibrated_thresholds") or {}
    per_sensor_thr_list = require_stage2_per_sensor_thresholds(
        calibrated.get("per_sensor"),
        sensor_names,
        context="evaluate_gdn_only: ",
    )
    per_sensor_thr = np.array(per_sensor_thr_list, dtype=np.float32)
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
    gdn_preds = predictor.predict(normalized_windows, batch_size=batch_size)

    sensor_labels_pred = (gdn_preds > per_sensor_thr).astype(np.float32)

    from llm.evaluation.metrics import (
        compute_all_metrics,
        format_metrics_report,
        window_labels_sensor_indexed_from_sensor_binary,
    )

    window_labels_true = window_labels_sensor_indexed_from_sensor_binary(
        sensor_labels_true
    )
    window_labels_pred = window_labels_sensor_indexed_from_sensor_binary(
        sensor_labels_pred
    )

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_for_metrics
        if fault_types_for_metrics is not None
        else None,
    )

    print(format_metrics_report(metrics))

    # Save results
    results = {
        "method": "gdn_only",
        "dataset": str(dataset_path),
        "gdn_model": str(model_path),
        "num_windows": int(num_windows),
        "stratified_limit": sample_indices_for_results is not None,
        "stratified_limit_seed": stratified_limit_seed
        if sample_indices_for_results is not None
        else None,
        "sample_indices": sample_indices_for_results.tolist()
        if sample_indices_for_results is not None
        else None,
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
    temperature: float = 0.1,
    limit: Optional[int] = None,
    stratify_limit: bool = True,
    stratified_limit_seed: int = 42,
    llm_timeout: int = 120,
    max_violations_in_prompt: Optional[int] = None,
    debug_prompt: bool = False,
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
        temperature: LLM sampling temperature (passed through to the API; use ~0.0–0.2 for stable JSON)
        limit: Optional cap on windows; below dataset size defaults to stratified sampling
            by fault_types (set stratify_limit=False for first N rows). Stratified path requires fault_types in .npz.
        stratify_limit: If True and limit is set, subsample proportionally by fault type.
        stratified_limit_seed: RNG seed when stratify_limit is True
        max_violations_in_prompt: If set, only include the top-K violations by deviation in the prompt
        debug_prompt: If True, print the first window's user prompt to stdout for inspection

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
        client = create_client(
            model_name=model_name, base_url=lm_base_url, timeout=llm_timeout
        )
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
    ref_reasoning_full = data["reference_reasoning"].tolist() if "reference_reasoning" in data else None

    # Load metadata
    metadata_path = dataset_path.parent / f"{dataset_path.stem}_metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        sensor_names = metadata["dataset_info"]["sensor_names"]
    else:
        sensor_names = list(DEFAULT_SENSOR_NAMES)

    full_num_windows = int(normalized_windows.shape[0])
    fault_types_for_metrics = data.get("fault_types", None)
    sample_indices_for_results: Optional[np.ndarray] = None

    if limit is not None and limit > 0:
        if limit < full_num_windows:
            if stratify_limit:
                if fault_types_for_metrics is None:
                    raise ValueError(
                        "Stratified limit requires fault_types in the dataset. "
                        "The .npz file must contain a 'fault_types' array."
                    )
                validate_fault_types_for_stratification(
                    fault_types_for_metrics, sensor_labels_true
                )
                sample_indices_for_results = stratified_sample_indices(
                    fault_types_for_metrics, limit, random_state=stratified_limit_seed
                )
                ix = sample_indices_for_results
                normalized_windows = normalized_windows[ix]
                unnormalized_windows = unnormalized_windows[ix]
                sensor_labels_true = sensor_labels_true[ix]
                window_labels_true = window_labels_true[ix]
                fault_types_for_metrics = fault_types_for_metrics[ix]
                if ref_reasoning_full is not None:
                    ref_reasoning_full = [ref_reasoning_full[int(j)] for j in ix]
                num_windows = int(len(ix))
                print(
                    f"  ⚠️  LIMIT MODE: stratified sample of {num_windows} windows "
                    f"(seed={stratified_limit_seed})"
                )
            else:
                ix = np.arange(limit, dtype=np.int64)
                normalized_windows = normalized_windows[ix]
                unnormalized_windows = unnormalized_windows[ix]
                sensor_labels_true = sensor_labels_true[ix]
                window_labels_true = window_labels_true[ix]
                if fault_types_for_metrics is not None:
                    fault_types_for_metrics = fault_types_for_metrics[ix]
                if ref_reasoning_full is not None:
                    ref_reasoning_full = ref_reasoning_full[:limit]
                sample_indices_for_results = None
                num_windows = int(limit)
                print(
                    f"  ⚠️  LIMIT MODE: first {num_windows} windows in file order (--no-stratify-limit)"
                )
        else:
            num_windows = full_num_windows
            print(
                f"  LIMIT ({limit}) ≥ dataset size: using all {num_windows} windows"
            )
    else:
        num_windows = full_num_windows

    print(f"  Loaded {num_windows} windows")
    print(f"  Window size: {normalized_windows.shape[1]}")
    print(f"  Sensors: {len(sensor_names)}")

    # Compute per-sensor population stats from normal windows in the active (sub)set.
    # Used to provide a population baseline alongside anomalous sensor summaries in the prompt.
    # Note: when --limit is active these stats are derived from the subsample only, not the
    # full dataset, so they may be less representative than a full-dataset baseline.
    # This uses ground truth labels only to identify normal windows — not to guide predictions.
    normal_mask_pop = sensor_labels_true.sum(axis=1) == 0
    normal_population = unnormalized_windows[normal_mask_pop]
    if normal_population.shape[0] == 0:
        sensor_population_stats: Dict[str, Dict[str, float]] = {}
    else:
        sensor_population_stats = {
            name: {
                "mean": float(normal_population[:, :, i].mean()),
                "std":  float(normal_population[:, :, i].std()),
                "p5":   float(np.percentile(normal_population[:, :, i], 5)),
                "p95":  float(np.percentile(normal_population[:, :, i], 95)),
            }
            for i, name in enumerate(sensor_names)
        }
    print(
        f"  Normal population baseline computed from {normal_mask_pop.sum()} windows"
        + (" (subsample)" if num_windows < full_num_windows else "")
    )

    # Load checkpoint and detect parameters (prefer stage2 calibrated thresholds)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    calibrated = checkpoint.get("calibrated_thresholds") or {}
    per_sensor_thr_list = require_stage2_per_sensor_thresholds(
        calibrated.get("per_sensor"),
        sensor_names,
        context="evaluate_gdn_kg_llm: ",
    )
    sensor_thresholds_dict = {
        n: t for n, t in zip(sensor_names, per_sensor_thr_list)
    }
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
    )

    # Run GDN inference for KG context (process_for_kg returns full KG inputs)
    print("Running GDN inference for KG context...")
    _gdn_start = time.time()
    kg_preds = predictor.process_for_kg(normalized_windows, batch_size=batch_size)
    gdn_time = time.time() - _gdn_start

    # Build knowledge graph
    _kg_start = time.time()
    kg = KnowledgeGraph(
        sensor_names=sensor_names,
        sensor_embeddings=kg_preds["sensor_embeddings"],
        adjacency_matrix=kg_preds["adjacency_matrix"],
    )

    kg.construct(
        X_windows=kg_preds["X_windows"],
        gdn_predictions=kg_preds["gdn_predictions"],
        propagation_per_sensor_thresholds=per_sensor_thr_list,
        X_windows_unnormalized=unnormalized_windows,
        sensor_labels_true=sensor_labels_true,
        window_labels_true=window_labels_true,
    )
    kg_build_time = time.time() - _kg_start

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

    print("Precomputing per-window KG contexts (single pass, per-sensor Stage-2 τ gated violations)...")
    _precompute_start = time.time()
    precomputed_kg_contexts = kg.precompute_window_contexts(
        num_windows,
        per_sensor_thr_list,
        temporal_context_windows=2,
    )
    kg_precompute_time = time.time() - _precompute_start

    # Per-faulty-window diagnostic: violations and anomaly_propagation
    faulty_mask = (sensor_labels_true.sum(axis=1) > 0)
    n_faulty = int(faulty_mask.sum())
    violation_breakdown = {"total_violations": 0}
    for idx in range(num_windows):
        ctx = precomputed_kg_contexts[idx]
        violations = ctx.get("violations", [])
        violation_breakdown["total_violations"] += len(violations)
    if n_faulty > 0:
        n_with_violations = 0
        n_with_propagation = 0
        n_with_v_or_p = 0
        n_with_both = 0
        for idx in range(num_windows):
            if not faulty_mask[idx]:
                continue
            ctx = precomputed_kg_contexts[idx]
            has_v = len(ctx.get("violations", [])) > 0
            has_p = len(ctx.get("anomaly_propagation", [])) > 0
            if has_v:
                n_with_violations += 1
            if has_p:
                n_with_propagation += 1
            if has_v or has_p:
                n_with_v_or_p += 1
            if has_v and has_p:
                n_with_both += 1
        print()
        print("Per-faulty-window KG context (LLM prompt substance):")
        print(f"  Faulty windows: {n_faulty}")
        print(f"  With violations: {n_with_violations} ({100 * n_with_violations / n_faulty:.1f}%)")
        print(f"  With anomaly_propagation: {n_with_propagation} ({100 * n_with_propagation / n_faulty:.1f}%)")
        print(f"  With violations or propagation: {n_with_v_or_p} ({100 * n_with_v_or_p / n_faulty:.1f}%)")
        print(f"  With both: {n_with_both} ({100 * n_with_both / n_faulty:.1f}%)")
        if n_with_v_or_p < n_faulty * 0.5:
            print("  ⚠️  Sparse KG context: many faulty windows lack violations/propagation.")
            print("      LLM gets prompt structure but little substance → poor fault typing.")
    print("  Violation breakdown (all windows, GDN correlation violations):")
    print(f"    total: {violation_breakdown['total_violations']}")
    print()

    # Run LLM predictions with KG context
    from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

    model_name = model_repo or "granite-4.0-h-micro"
    print("Running LLM predictions with KG context...")
    window_labels_pred = []
    is_faulty_pred = []  # Direct LLM is_faulty boolean
    sensor_labels_pred = []
    sensor_labels_pred_raw = []
    fault_types_pred = []
    reasoning_list = []
    processing_times = []

    with tqdm(total=num_windows, desc="KG-LLM", unit="window", dynamic_ncols=True) as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()
            kg_context = precomputed_kg_contexts[window_idx]
            window_stats = kg.window_stats.get(window_idx, {})
            sensor_scores = {
                name: float(stats.anomaly_score)
                for name, stats in window_stats.items()
            }
            messages = build_kg_prompt(
                kg_context,
                sensor_scores,
                sensor_names,
                sensor_thresholds_dict,
                max_violations=max_violations_in_prompt,
                window_data_unnorm=unnormalized_windows[window_idx],
                sensor_population_stats=sensor_population_stats,
            )
            if debug_prompt and window_idx == 0:
                print("--- DEBUG: first-window user prompt ---")
                print(messages[1]["content"])
                print("--- END DEBUG ---")

            prediction = call_llm_fault_diagnosis(
                client,
                model_name,
                messages,
                sensor_names,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            sensor_labels_filtered = filter_sensor_labels_to_root_only(prediction, sensor_names)
            sensor_labels_raw = prediction.get("sensor_labels", sensor_labels_filtered.copy())

            window_labels_pred.append(prediction["window_label"])
            is_faulty_pred.append(int(prediction.get("is_faulty", prediction["window_label"] > 0)))
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
    is_faulty_pred = np.array(is_faulty_pred)  # Direct LLM is_faulty boolean (0/1)
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

    # Build aligned reference reasoning list (same length as predictions)
    n_pred = len(reasoning_list)
    if ref_reasoning_full is not None:
        ref_reasoning_list = list(ref_reasoning_full[:n_pred])
        ref_reasoning_list += [""] * max(0, n_pred - len(ref_reasoning_list))
    else:
        ref_reasoning_list = [""] * n_pred

    # Compute metrics
    print("Computing evaluation metrics...")
    fault_types_true = fault_types_for_metrics
    kg_time = kg_build_time + kg_precompute_time

    metrics = compute_all_metrics(
        y_true_window=window_labels_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=sensor_names,
        fault_types=fault_types_true if fault_types_true is not None else None,
        fault_types_pred=fault_types_pred,
        reasoning=reasoning_list,
        reference_reasoning=ref_reasoning_list,
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
        "kg_build_time_seconds": float(kg_build_time),
        "kg_precompute_contexts_time_seconds": float(kg_precompute_time),
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
        "stratified_limit": sample_indices_for_results is not None,
        "stratified_limit_seed": stratified_limit_seed
        if sample_indices_for_results is not None
        else None,
        "sample_indices": sample_indices_for_results.tolist()
        if sample_indices_for_results is not None
        else None,
        "violation_breakdown": violation_breakdown,
        "metrics": metrics,
        "predictions": {
            "is_faulty": is_faulty_pred.tolist(),         # Direct LLM is_faulty boolean
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
    dataset_path: str,
    model_path: str,
    lm_url: str,
    limit: Optional[int] = None,
    stratify_limit: bool = True,
    stratified_limit_seed: int = 42,
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
        stratify_limit=stratify_limit,
        stratified_limit_seed=stratified_limit_seed,
    )

    preds = res["predictions"]
    n_pred = len(preds["window_labels"])
    data = np.load(dataset_path, allow_pickle=True)
    sensor_labels_true = data["sensor_labels"]
    fault_types = data.get("fault_types", None)
    sample_ix = res.get("sample_indices")
    metadata_path = (
        Path(dataset_path).parent / f"{Path(dataset_path).stem}_metadata.json"
    )
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        sensor_names = meta["dataset_info"]["sensor_names"]
    else:
        sensor_names = list(DEFAULT_SENSOR_NAMES)
    ref_reason = data.get("reference_reasoning", None)
    if ref_reason is not None and hasattr(ref_reason, "tolist"):
        ref_reason = ref_reason.tolist()
    else:
        ref_reason = [""] * len(sensor_labels_true)

    if sample_ix is not None:
        ix = np.array(sample_ix, dtype=np.int64)
        sensor_labels_true = sensor_labels_true[ix]
        if fault_types is not None:
            fault_types = fault_types[ix]
        ref_reason = [ref_reason[int(j)] for j in ix]
    else:
        sensor_labels_true = sensor_labels_true[:n_pred]
        if fault_types is not None:
            fault_types = fault_types[:n_pred]
        ref_reason = ref_reason[:n_pred]

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
        "--temperature",
        type=float,
        default=0.1,
        help="LLM sampling temperature (default 0.1; lower = more deterministic)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap windows; default is stratified by fault_types (needs fault_types in .npz unless --no-stratify-limit)",
    )
    parser.add_argument(
        "--no-stratify-limit",
        action="store_false",
        dest="stratify_limit",
        help="With --limit, take the first N windows in file order instead of stratified sampling",
    )
    parser.set_defaults(stratify_limit=True)
    parser.add_argument(
        "--stratified-limit-seed",
        type=int,
        default=42,
        help="Random seed when using stratified --limit (default: 42)",
    )
    parser.add_argument(
        "--sanity-check",
        action="store_true",
        help="Run KG sanity check only (verify violations/propagation non-trivial, then exit)",
    )
    parser.add_argument(
        "--sanity-check-full",
        action="store_true",
        help="Run sanity check on full dataset (default: sample of 3 windows)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "gdn_only"],
        default="full",
        help="Evaluation mode: full=KG-LLM pipeline, gdn_only=GDN predictions only",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP read timeout in seconds for each LLM request (default: 120)",
    )
    parser.add_argument(
        "--max-violations-in-prompt",
        type=int,
        default=None,
        help="Include only the top-K GDN violations by deviation (default: all)",
    )
    parser.add_argument(
        "--debug-prompt",
        action="store_true",
        help="Print the first window's user prompt to stdout (for prompt inspection)",
    )

    args = parser.parse_args()

    if args.sanity_check:
        run_kg_sanity_check(
            dataset_path=Path(args.dataset),
            model_path=Path(args.model_path),
            batch_size=args.batch_size,
            device=args.device,
            sample_windows=None if args.sanity_check_full else [100, 500, 1000],
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
            stratify_limit=args.stratify_limit,
            stratified_limit_seed=args.stratified_limit_seed,
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
        stratify_limit=args.stratify_limit,
        stratified_limit_seed=args.stratified_limit_seed,
        llm_timeout=args.timeout,
        max_violations_in_prompt=args.max_violations_in_prompt,
        debug_prompt=args.debug_prompt,
    )


if __name__ == "__main__":
    main()
