"""
Evaluate Serialised KG->LLM method on shared evaluation dataset.

This script:
1. Loads shared dataset
2. Processes normalized windows through GDN->KG pipeline
3. Extracts KG context for each window
4. Formats KG-enhanced prompts for LLM
5. Runs LLM inference with KG context
6. Compares predictions to ground truth
7. Computes evaluation metrics
"""

import numpy as np
import json
import argparse
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time
import sys
from tqdm import tqdm

HOST = "http://127.0.0.1:1234"
LLM_MODEL = "granite-4.0-h-micro"
# Add paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from kg.create_kg import GDNPredictor
from kg.create_kg import (
    KnowledgeGraph,
    EXPECTED_CORRELATIONS,
    SENSOR_SUBSYSTEMS,
    SENSOR_DESCRIPTIONS,
)
from llm.evaluation.evaluate_llm_baseline import filter_sensor_labels_to_root_only
from llm.evaluation.schemas import FaultDiagnosis
from llm.evaluation.utils import parse_structured_response, parsed_to_prediction

SYSTEM_PROMPT = """You are an expert automotive fault diagnostics system.
You will receive structured sensor graph data from an OBD-II vehicle.
Your task is to identify faulty sensors and classify the fault type.

RULES FOR FAULTY SENSORS:
1. Review the 'ANOMALOUS SENSORS' list. These sensors have been flagged by a neural network.
2. If a sensor is in this list, it is highly likely to be faulty. You should include it in your output unless the provided context strongly proves it is a false alarm.
3. If the list says '(none above threshold)', output an empty list for faulty sensors.
4. Only use sensor names from the provided valid list.

DIAGNOSTIC GUIDE:
- If a sensor's value flatlines or its correlation breaks completely, it may be a DROPOUT or STUCK fault.
- If multiple normally-correlated sensors (like RPM and SPEED) break their relationship, look for an inconsistency or dropout in one of them.
- If a sensor shows gradual divergence over the propagation path, it is likely a gradual_drift.
- Use the PROPAGATION PATH to find the root cause sensor (the first sensor in the chain).

Respond with JSON only."""


def build_kag_prompt(
    window_idx: int,
    sensor_scores: Dict[str, float],
    violations: List[Tuple[str, str, float, float]],
    propagation_chain: List[str],
    sensor_cols: List[str],
    sensor_threshold: float,
    sensor_thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, str]]:
    def _thr(name: str) -> float:
        return sensor_thresholds.get(name, sensor_threshold) if sensor_thresholds else sensor_threshold

    above_threshold = [
        (name, score)
        for name, score in sensor_scores.items()
        if score > _thr(name)
    ]
    above_threshold.sort(key=lambda x: x[1], reverse=True)

    thr_label = "per-sensor thresholds" if sensor_thresholds else f"{sensor_threshold:.2f}"
    lines = []
    lines.append(f"ANOMALOUS SENSORS (score > {thr_label}):")
    if above_threshold:
        for name, score in above_threshold:
            lines.append(f"  {name}: anomaly_score={score:.3f}")
    else:
        lines.append("  (none above threshold)")

    lines.append("")
    lines.append("VIOLATED CORRELATIONS:")
    violation_block_lines = []
    for a, b, exp, act in violations[:5]:
        if exp > 0.6 and act < 0.2:
            violation_block_lines.append(
                f"  {a} and {b} are normally highly correlated, but this relationship is broken (expected {exp:.2f}, actual {act:.2f})."
            )
        elif exp > 0 and act < 0:
            violation_block_lines.append(
                f"  {a} and {b} normally move together, but are now moving in opposite directions."
            )
        else:
            violation_block_lines.append(
                f"  {a} <-> {b} correlation deviated significantly from normal (expected {exp:.2f}, actual {act:.2f})."
            )
    violation_block = (
        "\n".join(violation_block_lines)
        if violation_block_lines
        else "  (none detected)"
    )
    lines.append(violation_block)

    lines.append("")
    lines.append("PROPAGATION PATH (root -> downstream):")
    if propagation_chain:
        lines.append(" -> ".join(propagation_chain))
    else:
        lines.append("(none)")

    lines.append("")
    lines.append("VALID SENSOR NAMES (use exactly these in faulty_sensors field):")
    lines.append(", ".join(sensor_cols))
    lines.append("")
    lines.append(
        "FAULT_TYPE: <COOLANT_DROPOUT | VSS_DROPOUT | MAF_SCALE_LOW | TPS_STUCK | gradual_drift | normal>"
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]


def _call_llm_http(messages: List[Dict[str, str]]) -> str:
    url = f"{HOST.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "FaultDiagnosis",
                "strict": True,
                "schema": FaultDiagnosis.model_json_schema(),
            },
        },
    }
    resp = requests.post(url, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


class _HTTPLLMClient:
    """Minimal HTTP-based LLM client (uses requests, no LMInference)."""

    def chat_completions(self, messages, **kwargs):
        url = f"{HOST.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens"),
            "stream": False,
        }
        if "response_format" in kwargs:
            payload["response_format"] = kwargs["response_format"]
        payload = {k: v for k, v in payload.items() if v is not None}
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


from llm.evaluation.metrics import compute_all_metrics, format_metrics_report
from kg.similarity import compute_window_similarity


def extract_window_kg_context(
    kg: KnowledgeGraph,
    window_idx: int,
    temporal_context_windows: int = 2,
) -> Dict[str, any]:
    """
    Extract KG context for a specific window.

    Args:
        kg: KnowledgeGraph instance with built KG
        window_idx: Index of the window to extract context for
        temporal_context_windows: Number of previous windows to include

    Returns:
        Dictionary with KG context:
        - 'entities': List of entities with types
        - 'relationships': List of relationship triples
        - 'violations': List of relationship violations
        - 'temporal_context': Temporal information from previous windows
        - 'anomaly_propagation': Relevant anomaly propagation chains
    """
    # Use get_window_kg method if available, otherwise build context manually
    try:
        kg_context = kg.get_window_kg(
            window_idx, temporal_context_windows=temporal_context_windows
        )
        return kg_context
    except Exception:
        # Fallback: build context manually
        context = {
            "entities": [],
            "relationships": [],
            "violations": [],
            "temporal_context": [],
            "anomaly_propagation": [],
            "distribution_thresholds": None,
            "stage_features": {},
        }

        # Get current window graph and stats
        if window_idx not in kg.window_graphs:
            return context

        window_graph = kg.window_graphs[window_idx]
        window_stats = kg.window_stats.get(window_idx, {})

        # Get distribution thresholds if available
        thresholds = getattr(kg, "distribution_thresholds", None)
        if thresholds:
            context["distribution_thresholds"] = thresholds

        # Extract entities with types and subsystems
        for sensor_name in kg.sensor_names:
            desc = SENSOR_DESCRIPTIONS.get(sensor_name, {})
            subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, "Unknown")
            # Use distribution-based threshold if available
            if thresholds:
                anomaly_threshold_per_sensor = thresholds.get(
                    "anomaly_threshold_per_sensor", {}
                )
                anomaly_threshold = anomaly_threshold_per_sensor.get(
                    sensor_name, thresholds.get("anomaly_threshold_global", 0.5)
                )
            else:
                anomaly_threshold = 0.5  # Fallback
            stat = window_stats.get(sensor_name)
            is_faulty = stat.anomaly_score > anomaly_threshold if stat else False

            entity_info = {
                "name": sensor_name,
                "type": "Sensor",
                "subsystem": subsystem,
                "description": desc.get("description", ""),
                "is_faulty": is_faulty,  # Based on GDN prediction threshold, not ground truth
            }
            context["entities"].append(entity_info)

        # Extract relationships from current window
        # Include all violations and relationships involving sensors with high GDN prediction scores
        # Also include significant correlations (threshold 0.3)
        correlation_threshold = 0.3
        prediction_threshold = 0.5  # Threshold for GDN predictions (not ground truth)
        anomalous_sensors = {
            sensor_name
            for sensor_name, stat in window_stats.items()
            if stat.anomaly_score > prediction_threshold
        }  # Based on GDN predictions

    for u, v, data in window_graph.edges(data=True):
        edge_type = data.get("edge_type", "correlates_with")

        # Support both old and new attribute formats for backward compatibility
        correlation = data.get("correlation", 0)  # Old format (preserved)
        correlation_strength = data.get(
            "correlation_strength", abs(correlation)
        )  # New format
        correlation_direction = data.get(
            "correlation_direction", "positive" if correlation > 0 else "negative"
        )

        # Domain knowledge expectations (new format)
        violates_domain = data.get("violates_domain_expectation", False)
        domain_expected_type = data.get("domain_expected_type", None)
        domain_expected_strength = data.get("domain_expected_strength", None)

        # GDN expectations (new format)
        expected_correlation_gdn = data.get(
            "expected_correlation_gdn", data.get("expected_correlation", 0)
        )  # Fallback to old format
        deviation_from_gdn = data.get(
            "deviation_from_gdn", data.get("correlation_deviation", 0)
        )  # Fallback to old format
        violates_gdn = data.get("violates_gdn_expectation", False)

        # GDN scores
        gdn_score_source = data.get("gdn_score_source", 0)
        gdn_score_target = data.get("gdn_score_target", 0)
        potential_fault_indicator = data.get("potential_fault_indicator", False)

        # Include if:
        # 1. It's a violation (domain or GDN expectation violated)
        # 2. It involves an anomalous sensor (contextual information)
        # 3. It's a significant correlation (above threshold)
        # 4. It's a potential fault indicator
        is_violation = violates_domain or violates_gdn
        involves_anomaly = u in anomalous_sensors or v in anomalous_sensors
        is_significant = correlation_strength >= correlation_threshold

        if not (
            is_violation
            or involves_anomaly
            or is_significant
            or potential_fault_indicator
        ):
            continue

        relationship = {
            "source": u,
            "target": v,
            "relation": edge_type,
            "correlation": float(correlation),  # Preserved for backward compatibility
            "correlation_strength": float(correlation_strength),
            "correlation_direction": correlation_direction,
            "expected_correlation_gdn": float(expected_correlation_gdn),
            "deviation_from_gdn": float(deviation_from_gdn),
            "violates_domain_expectation": violates_domain,
            "violates_gdn_expectation": violates_gdn,
            "gdn_score_source": float(gdn_score_source),
            "gdn_score_target": float(gdn_score_target),
            "potential_fault_indicator": potential_fault_indicator,
        }

        # Add domain knowledge if available
        if domain_expected_type:
            relationship["domain_expected_type"] = domain_expected_type
        if domain_expected_strength:
            relationship["domain_expected_strength"] = domain_expected_strength
        if "violation_type" in data:
            relationship["violation_type"] = data["violation_type"]

        context["relationships"].append(relationship)

        # Track violations separately (both domain and GDN violations)
        if is_violation:
            context["violations"].append(relationship)

        # Extract temporal context from previous windows
        for prev_idx in range(
            max(0, window_idx - temporal_context_windows), window_idx
        ):
            if prev_idx in kg.window_stats:
                prev_stats = kg.window_stats[prev_idx]
                temporal_info = {
                    "window_idx": prev_idx,
                    "faulty_sensors": [],
                    "anomaly_scores": {},
                }

                # Use prediction threshold (0.5) for GDN predictions, not ground truth
                prediction_threshold = 0.5
                for sensor_name, stat in prev_stats.items():
                    if (
                        stat.anomaly_score > prediction_threshold
                    ):  # Based on GDN prediction threshold
                        temporal_info["faulty_sensors"].append(sensor_name)
                        temporal_info["anomaly_scores"][sensor_name] = float(
                            stat.anomaly_score
                        )

                if temporal_info["faulty_sensors"]:
                    context["temporal_context"].append(temporal_info)

    # Extract relevant anomaly propagation chains
    for chain in kg.anomaly_propagation_chains:
        root_window = chain.get("root_window", -1)
        propagation_timeline = chain.get("propagation_timeline", [])

        if root_window == window_idx:
            context["anomaly_propagation"].append(
                {
                    "type": "root",
                    "root_sensor": chain.get("root_sensor", ""),
                    "root_window": root_window,
                    "affected_sensors": chain.get("affected_sensors", []),
                }
            )
        else:
            for timeline_entry in propagation_timeline:
                if timeline_entry.get("window") == window_idx:
                    context["anomaly_propagation"].append(
                        {
                            "type": "propagation",
                            "root_sensor": chain.get("root_sensor", ""),
                            "root_window": root_window,
                            "affected_sensors": timeline_entry.get(
                                "affected_sensors", []
                            ),
                        }
                    )
                    break

    return context


def format_kg_context_as_adjacency_matrix(
    kg_context: Dict[str, any], window_idx: int, kg: KnowledgeGraph
) -> str:
    """
    Format KG context as compact adjacency matrix for LLM prompt.

    More compact than text format, better for smaller models.

    Args:
        kg_context: Context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg: KnowledgeGraph instance

    Returns:
        Formatted string with adjacency matrix representation
    """
    lines = []
    lines.append("Knowledge Graph (Adjacency Matrix Format):")

    # Get sensor names in order
    sensor_names = kg.sensor_names
    num_sensors = len(sensor_names)

    # Build correlation matrix
    corr_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    deviation_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    violation_matrix = [[False] * num_sensors for _ in range(num_sensors)]
    gdn_expected_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]
    gdn_score_matrix = [[0.0] * num_sensors for _ in range(num_sensors)]

    # Get thresholds
    thresholds = getattr(kg, "distribution_thresholds", None)
    window_stats = kg.window_stats.get(window_idx, {})

    # Fill matrices from relationships
    for rel in kg_context.get("relationships", []):
        source = rel["source"]
        target = rel["target"]

        try:
            src_idx = sensor_names.index(source)
            tgt_idx = sensor_names.index(target)
        except ValueError:
            continue

        corr = rel.get("correlation", 0.0)
        deviation = rel.get("deviation_from_gdn", 0.0)
        violates = rel.get("violates_gdn_expectation", False) or rel.get(
            "violates_domain_expectation", False
        )
        gdn_expected = rel.get("expected_correlation_gdn", 0.0)
        gdn_src = rel.get("gdn_score_source", 0.0)
        gdn_tgt = rel.get("gdn_score_target", 0.0)

        corr_matrix[src_idx][tgt_idx] = corr
        deviation_matrix[src_idx][tgt_idx] = deviation
        violation_matrix[src_idx][tgt_idx] = violates
        gdn_expected_matrix[src_idx][tgt_idx] = gdn_expected
        gdn_score_matrix[src_idx][tgt_idx] = max(gdn_src, gdn_tgt)

    # Sensor status
    lines.append("\nSensor Status:")
    sensor_status = []
    for i, sensor_name in enumerate(sensor_names):
        stat = window_stats.get(sensor_name)
        if thresholds:
            anomaly_threshold_per_sensor = thresholds.get(
                "anomaly_threshold_per_sensor", {}
            )
            threshold = anomaly_threshold_per_sensor.get(
                sensor_name, thresholds.get("anomaly_threshold_global", 0.5)
            )
        else:
            threshold = 0.5
        is_faulty = stat.anomaly_score > threshold if stat else False
        status = "ANOMALOUS" if is_faulty else "Normal"
        score = stat.anomaly_score if stat else 0.0
        sensor_status.append((sensor_name, status, score))
        lines.append(f"  {i}: {sensor_name} [{status}] (score: {score:.3f})")

    # Correlation Matrix
    lines.append("\nCorrelation Matrix (row -> col):")
    lines.append("     " + " ".join([f"{i:>4}" for i in range(num_sensors)]))
    for i, sensor_name in enumerate(sensor_names):
        row_str = f"{i:>3}: "
        for j in range(num_sensors):
            corr = corr_matrix[i][j]
            if abs(corr) < 0.01:
                row_str += "  .  "
            else:
                row_str += f"{corr:>5.2f}"
        # Remove " ()" suffix for cleaner display
        sensor_display = sensor_name.replace(" ()", "")
        lines.append(row_str + f"  [{sensor_display[:20]}]")

    # Deviation Matrix (only show violations)
    lines.append("\nDeviation from GDN (violations only, row -> col):")
    has_violations = False
    for i, sensor_name in enumerate(sensor_names):
        row_violations = []
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                dev = deviation_matrix[i][j]
                row_violations.append(f"{j}:{dev:.3f}")
                has_violations = True
        if row_violations:
            sensor_display = sensor_name.replace(" ()", "")
            lines.append(f"  {i} [{sensor_display[:20]}]: {', '.join(row_violations)}")
    if not has_violations:
        lines.append("  No violations detected")

    # GDN Expected vs Actual (for violations)
    lines.append("\nGDN Expected vs Actual (violations only):")
    has_violations = False
    for i, sensor_name in enumerate(sensor_names):
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                expected = gdn_expected_matrix[i][j]
                actual = corr_matrix[i][j]
                dev = deviation_matrix[i][j]
                tgt_name = sensor_names[j].replace(" ()", "")[:20]
                lines.append(
                    f"  {sensor_name[:15]} -> {tgt_name}: expected {expected:.3f}, actual {actual:.3f}, deviation {dev:.3f}"
                )
                has_violations = True
    if not has_violations:
        lines.append("  No violations detected")

    # Summary: Sensors with violations
    lines.append("\nViolation Summary:")
    violation_sensors = set()
    for i in range(num_sensors):
        for j in range(num_sensors):
            if violation_matrix[i][j]:
                violation_sensors.add(sensor_names[i])
                violation_sensors.add(sensor_names[j])

    if violation_sensors:
        lines.append(
            f"  Sensors involved in violations: {', '.join(sorted(violation_sensors))}"
        )
    else:
        lines.append("  No violations detected - all relationships normal")

    # Threshold context
    if thresholds:
        lines.append("\nThreshold Context:")
        lines.append(
            f"  Deviation threshold (90th percentile): {thresholds.get('deviation_threshold', 0.3):.3f}"
        )
        lines.append(f"  Deviation p50: {thresholds.get('deviation_p50', 0.0):.3f}")
        lines.append(f"  Deviation p95: {thresholds.get('deviation_p95', 0.0):.3f}")
        lines.append(
            f"  Anomaly threshold (global): {thresholds.get('anomaly_threshold_global', 0.5):.3f}"
        )

    # Embedding distances (compact format)
    if window_idx in getattr(kg, "window_embeddings", {}):
        embedding_data = kg.window_embeddings[window_idx]
        dist_normal = embedding_data.get("dist_normal", 0.0)
        dist_anomalous = embedding_data.get("dist_anomalous", 0.0)
        confidence = embedding_data.get("confidence", 0.0)

        lines.append("\nEmbedding Distances:")
        lines.append(f"  Distance to normal center: {dist_normal:.3f}")
        lines.append(f"  Distance to anomalous center: {dist_anomalous:.3f}")
        lines.append(f"  Confidence: {confidence:.3f}")

        # Interpretation
        if dist_normal < 0.12:
            interpretation = "Likely normal"
        elif dist_normal > 0.12 and dist_anomalous < 0.15:
            interpretation = "Likely anomalous"
        else:
            interpretation = "Uncertain/edge case"
        lines.append(f"  Interpretation: {interpretation}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_kg_context_for_llm(
    kg_context: Dict[str, any],
    window_idx: int,
    kg: KnowledgeGraph,
    use_adjacency_matrix: bool = False,
) -> str:
    """
    Format KG context as structured LLM prompt section.

    Shows structured knowledge graph representation: entities, relationships, violations,
    temporal context, and anomaly propagation. NO raw sensor data.

    Args:
        kg_context: Context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg: KnowledgeGraph instance
        use_adjacency_matrix: If True, use compact adjacency matrix format instead of verbose text

    Returns:
        Formatted string for LLM prompt with structured KG representation
    """
    if use_adjacency_matrix:
        return format_kg_context_as_adjacency_matrix(kg_context, window_idx, kg)
    lines = []
    lines.append("Knowledge Graph Representation:")

    thresholds = getattr(kg, "distribution_thresholds", None)

    # All entities with their status and metadata
    lines.append("\nENTITIES:")
    for entity in kg_context["entities"]:
        status = "⚠️ ANOMALOUS" if entity.get("is_faulty") else "✓ Normal"
        lines.append(f"{status}: {entity['name']}")
        lines.append(f"  Subsystem: {entity['subsystem']}")
        if entity.get("description"):
            lines.append(f"  Description: {entity['description']}")

    # All relationships (not filtered - show all significant ones)
    lines.append("\nRELATIONSHIPS:")
    if kg_context["relationships"]:
        for rel in kg_context["relationships"]:
            rel_type = rel["relation"]
            source = rel["source"]
            target = rel["target"]
            corr = rel.get("correlation", 0)

            # Check for violations (domain or GDN expectations)
            violates_domain = rel.get("violates_domain_expectation", False)
            violates_gdn = rel.get("violates_gdn_expectation", False)
            correlation_direction = rel.get("correlation_direction", "positive")
            correlation_strength = rel.get("correlation_strength", abs(corr))
            potential_fault = rel.get("potential_fault_indicator", False)
            violation_confidence = rel.get("violation_confidence", 0.0)

            if violates_domain or violates_gdn:
                # Violation detected
                violation_types = []
                if violates_domain:
                    violation_types.append("domain expectation")
                if violates_gdn:
                    violation_types.append("GDN expectation")

                violation_type_detail = rel.get("violation_type", "unknown")
                lines.append(
                    f"⚠️ VIOLATION ({', '.join(violation_types)}): {source} --[{rel_type}]--> {target}"
                )
                lines.append(
                    f"  Correlation: {corr:.3f} ({correlation_direction}, strength: {correlation_strength:.3f})"
                )

                if violates_domain:
                    domain_type = rel.get("domain_expected_type", "unknown")
                    domain_strength = rel.get("domain_expected_strength", "unknown")
                    lines.append(
                        f"  Domain expected: {domain_type} ({domain_strength})"
                    )
                    if violation_type_detail != "unknown":
                        lines.append(f"  Violation type: {violation_type_detail}")

                if violates_gdn:
                    exp_corr_gdn = rel.get("expected_correlation_gdn", 0)
                    dev_gdn = rel.get("deviation_from_gdn", 0)
                    lines.append(f"  GDN expected correlation: {exp_corr_gdn:.3f}")
                    lines.append(f"  Deviation from GDN: {dev_gdn:.3f}")

                    # Include distribution context in violation message
                    if thresholds:
                        deviation_p50 = thresholds.get("deviation_p50", 0)
                        deviation_p95 = thresholds.get("deviation_p95", 0)
                        deviation_threshold = thresholds.get("deviation_threshold", 0.3)

                        # Show percentile context
                        if dev_gdn > deviation_p95:
                            lines.append(
                                f"  [Severe: deviation > 95th percentile ({deviation_p95:.3f})]"
                            )
                        elif dev_gdn > deviation_p50:
                            lines.append(
                                f"  [Moderate: deviation > 50th percentile ({deviation_p50:.3f})]"
                            )
                        else:
                            lines.append(
                                f"  [Threshold: {deviation_threshold:.3f} (90th percentile)]"
                            )

                    # Include violation confidence
                    if violation_confidence > 0.5:
                        lines.append(
                            f"  [High confidence violation ({violation_confidence:.2f}) based on training features]"
                        )

                if potential_fault:
                    gdn_src = rel.get("gdn_score_source", 0)
                    gdn_tgt = rel.get("gdn_score_target", 0)
                    # Include distribution context for anomaly scores
                    if thresholds:
                        anomaly_p50 = thresholds.get("anomaly_p50", 0)
                        anomaly_p95 = thresholds.get("anomaly_p95", 0)
                        anomaly_threshold = thresholds.get(
                            "anomaly_threshold_global", 0.5
                        )
                        src_percentile = (
                            "high"
                            if gdn_src > anomaly_p95
                            else "moderate"
                            if gdn_src > anomaly_p50
                            else "low"
                        )
                        tgt_percentile = (
                            "high"
                            if gdn_tgt > anomaly_p95
                            else "moderate"
                            if gdn_tgt > anomaly_p50
                            else "low"
                        )
                        lines.append(
                            f"  ⚠️ Potential fault indicator (GDN scores: {gdn_src:.2f} [{src_percentile}], {gdn_tgt:.2f} [{tgt_percentile}], threshold: {anomaly_threshold:.2f})"
                        )
                    else:
                        lines.append(
                            f"  ⚠️ Potential fault indicator (GDN scores: {gdn_src:.2f}, {gdn_tgt:.2f})"
                        )
            else:
                # Normal relationship
                lines.append(f"{source} --[{rel_type}]--> {target}")
                lines.append(
                    f"  Correlation: {corr:.3f} ({correlation_direction}, strength: {correlation_strength:.3f})"
                )

                # Show domain expectations if available
                domain_type = rel.get("domain_expected_type")
                if domain_type:
                    domain_strength = rel.get("domain_expected_strength", "unknown")
                    lines.append(
                        f"  Domain expected: {domain_type} ({domain_strength})"
                    )

                # Show GDN expectations
                exp_corr_gdn = rel.get("expected_correlation_gdn", 0)
                if exp_corr_gdn != 0:
                    dev_gdn = rel.get("deviation_from_gdn", 0)
                    lines.append(f"  GDN expected correlation: {exp_corr_gdn:.3f}")
                    if thresholds and dev_gdn > 0:
                        # Show deviation even for non-violations if significant
                        deviation_p50 = thresholds.get("deviation_p50", 0)
                        if dev_gdn > deviation_p50:
                            lines.append(
                                f"  Deviation: {dev_gdn:.3f} (within normal range)"
                            )

                if potential_fault:
                    gdn_src = rel.get("gdn_score_source", 0)
                    gdn_tgt = rel.get("gdn_score_target", 0)
                    lines.append(f"  GDN scores: {gdn_src:.2f}, {gdn_tgt:.2f}")
    else:
        lines.append("No significant relationships detected.")

    # Temporal context (all relevant previous windows)
    if kg_context["temporal_context"]:
        lines.append("\nTEMPORAL CONTEXT:")
        for temp in kg_context["temporal_context"]:
            lines.append(f"Window {temp['window_idx']}:")
            if temp["faulty_sensors"]:
                lines.append(f"  Faulty sensors: {', '.join(temp['faulty_sensors'])}")
                if temp.get("anomaly_scores"):
                    scores_str = ", ".join(
                        [
                            f"{sensor}={score:.2f}"
                            for sensor, score in list(temp["anomaly_scores"].items())[
                                :3
                            ]
                        ]
                    )
                    if scores_str:
                        lines.append(f"  Anomaly scores: {scores_str}")
            else:
                lines.append("  No faults detected")

    # Anomaly propagation chains
    if kg_context["anomaly_propagation"]:
        lines.append("\nANOMALY PROPAGATION:")
        for prop in kg_context["anomaly_propagation"]:
            prop_type = prop.get("type", "unknown")
            root_sensor = prop.get("root_sensor", "unknown")
            root_window = prop.get("root_window", -1)
            affected = prop.get("affected_sensors", [])

            if prop_type == "root":
                lines.append("Root cause detected:")
                lines.append(f"  Root sensor: {root_sensor} at window {root_window}")
                if affected:
                    lines.append(f"  Affected sensors: {', '.join(affected)}")
            else:
                lines.append(f"Propagation from window {root_window}:")
                lines.append(f"  Root sensor: {root_sensor}")
                if affected:
                    lines.append(
                        f"  Affected sensors in this window: {', '.join(affected)}"
                    )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_embedding_context(window_idx: int, kg: KnowledgeGraph) -> str:
    """
    Format embedding-space analysis for LLM prompt.

    Args:
        window_idx: Index of the window
        kg: KnowledgeGraph instance with window_embeddings

    Returns:
        Formatted markdown string with embedding-space analysis
    """
    lines = []

    # Check if embedding data is available
    if window_idx not in kg.window_embeddings:
        return ""

    embedding_data = kg.window_embeddings[window_idx]
    dist_normal = embedding_data["dist_normal"]
    dist_anomalous = embedding_data["dist_anomalous"]
    confidence = embedding_data["confidence"]

    # Typical ranges (from plan)
    normal_mean = 0.085
    normal_std = 0.03
    anomalous_mean = 0.138
    anomalous_std = 0.04

    # Compute z-scores
    z_score_normal = (dist_normal - normal_mean) / normal_std if normal_std > 0 else 0.0
    z_score_anomalous = (
        (dist_anomalous - anomalous_mean) / anomalous_std if anomalous_std > 0 else 0.0
    )

    lines.append("\nEmbedding Space Analysis:")

    # Distance to normal center
    lines.append(f"\nDistance to Normal Center: {dist_normal:.4f}")
    normal_range_str = f"{normal_mean:.3f} ± {normal_std:.3f}"
    if dist_normal < normal_mean - normal_std:
        interpretation = "significantly closer than typical"
    elif dist_normal < normal_mean + normal_std:
        interpretation = "within typical normal range"
    else:
        interpretation = f"{abs(z_score_normal):.1f} standard deviations above typical"
    lines.append(f"  Typical normal range: {normal_range_str}")
    lines.append(f"  Interpretation: {interpretation}")
    lines.append(f"  Z-score: {z_score_normal:.2f}")

    # Distance to anomalous center
    lines.append(f"\nDistance to Anomalous Center: {dist_anomalous:.4f}")
    anomalous_range_str = f"{anomalous_mean:.3f} ± {anomalous_std:.3f}"
    if dist_anomalous < anomalous_mean - anomalous_std:
        interpretation = "significantly closer than typical"
    elif dist_anomalous < anomalous_mean + anomalous_std:
        interpretation = "within typical anomalous range"
    else:
        interpretation = (
            f"{abs(z_score_anomalous):.1f} standard deviations above typical"
        )
    lines.append(f"  Typical anomalous range: {anomalous_range_str}")
    lines.append(f"  Interpretation: {interpretation}")
    lines.append(f"  Z-score: {z_score_anomalous:.2f}")

    # Confidence score
    lines.append(f"\nConfidence Score: {confidence:.3f}")
    if confidence > 0.7:
        conf_interpretation = "high confidence (likely normal)"
    elif confidence > 0.3:
        conf_interpretation = "moderate confidence (uncertain)"
    else:
        conf_interpretation = "low confidence (likely anomalous)"
    lines.append(f"  Interpretation: {conf_interpretation}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def format_window_with_kg_for_llm(
    window_data: np.ndarray,
    sensor_names: List[str],
    kg_context: Dict[str, any],
    window_idx: int,
    kg: KnowledgeGraph,
    statistical_features: Optional[np.ndarray] = None,
    use_statistical_features: bool = True,
    use_adjacency_matrix: bool = False,
) -> str:
    """
    Format prompt with structured KG representation only.

    NO raw sensor data - only structured knowledge graph representation.
    LLM reasons over structured KG, not raw time series data.

    Args:
        window_data: (window_size, num_sensors) array - unnormalized sensor values (not used, kept for API compatibility)
        sensor_names: List of sensor names
        kg_context: KG context dictionary from extract_window_kg_context()
        window_idx: Current window index
        kg: KnowledgeGraph instance
        statistical_features: Optional statistical features (not used, kept for API compatibility)
        use_statistical_features: Whether to include statistical features (not used, kept for API compatibility)

    Returns:
        Complete formatted prompt string with structured KG representation only
    """
    # Format KG context section (structured representation)
    kg_section = format_kg_context_for_llm(
        kg_context, window_idx, kg, use_adjacency_matrix=use_adjacency_matrix
    )

    # Build prompt similar to baseline format for comparability
    lines = []
    lines.append(
        "You are an automotive diagnostic expert analyzing OBD-II sensor data."
    )
    lines.append("")
    lines.append("Task: Identify which sensors are faulty and describe the fault type.")
    lines.append("")
    lines.append(
        "The following knowledge graph representation was generated from sensor data analysis:"
    )
    lines.append("")
    lines.append(kg_section)
    lines.append("")

    # Add embedding-space explanation if embeddings are available (simplified)
    if window_idx in kg.window_embeddings:
        embedding_data = kg.window_embeddings[window_idx]
        dist_normal = embedding_data["dist_normal"]
        dist_anomalous = embedding_data["dist_anomalous"]
        confidence = embedding_data["confidence"]

        lines.append("Embedding Analysis:")
        lines.append(f"Distance to normal center: {dist_normal:.3f}")
        lines.append(f"Distance to anomalous center: {dist_anomalous:.3f}")
        lines.append(f"Confidence: {confidence:.2f}")
        if dist_normal < 0.12:
            lines.append("Interpretation: Likely normal")
        elif dist_normal > 0.12 and dist_anomalous < 0.15:
            lines.append("Interpretation: Likely anomalous")
        else:
            lines.append("Interpretation: Uncertain/edge case")
        lines.append("")

    lines.append(
        "Please analyze this knowledge graph representation and provide your diagnosis."
    )
    lines.append("")
    lines.append(
        "IMPORTANT: You MUST respond with ONLY a valid JSON object. No markdown, no code blocks (no ```), no explanations before or after."
    )
    lines.append("")
    lines.append("Required JSON format:")
    lines.append(
        '{"root_cause_sensors": ["SENSOR_NAME"] or [], "affected_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [], "faulty_sensors": ["SENSOR_NAME_1", "SENSOR_NAME_2"] or [], "fault_type": "FAULT_TYPE" or null, "reasoning": "explanation", "confidence": 0.85}'
    )
    lines.append("")
    lines.append(
        "REASONING: Be extensive and didactic. Write 3–6 sentences that: (1) state which evidence you used (which relationships/violations, which deviation or correlation values), (2) explain step-by-step how that evidence leads to the root cause or to normal operation, (3) briefly say why other sensors were or were not considered. Write so a reader can follow your logic."
    )
    lines.append("")
    lines.append("CRITICAL ANALYSIS INSTRUCTIONS:")
    lines.append(
        "- Carefully examine ALL relationship violations in the knowledge graph above"
    )
    lines.append(
        "- Look at the deviation values and correlation patterns to identify the root cause"
    )
    lines.append(
        "- Do NOT default to any specific fault type - base your diagnosis on the actual violations present"
    )
    lines.append(
        "- If multiple sensors show violations, identify which one has the strongest evidence as root cause"
    )
    lines.append("- If no violations exceed thresholds, conclude normal operation")
    lines.append("")
    lines.append("IMPORTANT:")
    lines.append(
        "- root_cause_sensors: The PRIMARY sensor(s) causing the fault (usually 1 sensor)"
    )
    lines.append(
        "- affected_sensors: Secondary sensors that are affected by the root cause but are NOT the primary fault source"
    )
    lines.append(
        "- faulty_sensors: For backward compatibility, include ALL faulty sensors (root + affected combined)"
    )
    lines.append("")
    lines.append("Example 1 (no fault - most common case):")
    lines.append(
        '{"root_cause_sensors": [], "affected_sensors": [], "faulty_sensors": [], "fault_type": null, "reasoning": "I examined all relationships in the knowledge graph. No relationship has a GDN expectation violation above the threshold (deviations are all below 0.3). All correlations are within expected ranges. All entities are marked as normal. The evidence does not support any fault. I conclude normal operation.", "confidence": 0.9}'
    )
    lines.append("")
    lines.append("Example 2 (gradual_drift - multiple sensors with small violations):")
    lines.append(
        '{"root_cause_sensors": ["ENGINE_RPM"], "affected_sensors": ["VEHICLE_SPEED"], "faulty_sensors": ["ENGINE_RPM", "VEHICLE_SPEED"], "fault_type": "gradual_drift", "reasoning": "The knowledge graph shows multiple small violations (deviations 0.35-0.45) across ENGINE_RPM and VEHICLE_SPEED relationships. No single violation is severe, but the pattern of multiple correlated small deviations suggests gradual sensor drift rather than a sudden dropout. ENGINE_RPM shows the strongest violation (0.45 deviation) so I treat it as root cause. VEHICLE_SPEED is affected because it correlates with ENGINE_RPM and shows related violations.", "confidence": 0.75}'
    )
    lines.append("")
    lines.append("Example 3 (COOLANT_DROPOUT - single sensor with severe violation):")
    lines.append(
        '{"root_cause_sensors": ["COOLANT_TEMPERATURE"], "affected_sensors": ["ENGINE_LOAD"], "faulty_sensors": ["COOLANT_TEMPERATURE", "ENGINE_LOAD"], "fault_type": "COOLANT_DROPOUT", "reasoning": "COOLANT_TEMPERATURE has a severe GDN expectation violation (deviation 0.72) with ENGINE_LOAD. The correlation is broken - expected positive relationship but actual is negative. This pattern indicates a dropout fault at the coolant sensor. ENGINE_LOAD is listed as affected because it appears in the violated relationship, but the primary fault is at the coolant sensor.", "confidence": 0.85}'
    )
    lines.append("")
    lines.append("Example 4 (VSS_DROPOUT - single sensor with large deviation):")
    lines.append(
        '{"root_cause_sensors": ["VEHICLE_SPEED"], "affected_sensors": [], "faulty_sensors": ["VEHICLE_SPEED"], "fault_type": "VSS_DROPOUT", "reasoning": "The knowledge graph shows a GDN expectation violation on VEHICLE_SPEED with deviation 0.85. The correlation with ENGINE_RPM is broken (expected positive, actual negative). This large deviation and broken correlation pattern indicates a dropout fault at the vehicle speed sensor. I treat VEHICLE_SPEED as root cause because it has the largest deviation; no other sensor has comparable violation strength.", "confidence": 0.9}'
    )
    lines.append("")
    lines.append("Available sensor names:")
    for name in kg.sensor_names:
        lines.append(f"  - {name.replace(' ()', '')}")
    lines.append("")
    lines.append(
        "FAULT_TYPE: <COOLANT_DROPOUT | VSS_DROPOUT | MAF_SCALE_LOW | TPS_STUCK | gradual_drift | normal>"
    )
    lines.append("")
    lines.append(
        "CRITICAL: Output ONLY the JSON object. Start with { and end with }. No other text."
    )

    return "\n".join(lines)


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
        detected_embed_dim = (
            checkpoint["sensor_embeddings"].shape[1]
            if "sensor_embeddings" in checkpoint
            else 32
        )
    except Exception:
        detected_embed_dim = 32

    predictor = GDNPredictor(
        model_path=str(model_path),
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=detected_embed_dim,
        top_k=3,
        hidden_dim=32,
        device=device,
    )

    kg_data = predictor.process_for_kg(
        X_windows=normalized_windows,
        sensor_labels=sensor_labels_true,
        window_labels=window_labels_true.astype(np.int64),
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
    )

    n_nodes = kg.kg.number_of_nodes()
    n_edges = kg.kg.number_of_edges()
    n_window_graphs = len(kg.window_graphs)
    n_prop_chains = len(kg.anomaly_propagation_chains)

    print("KG accumulation check:")
    print(f"  Nodes: {n_nodes}, Edges: {n_edges}")
    print(f"  Per-window graphs: {n_window_graphs} (should equal {num_windows})")
    print(f"  Anomaly propagation chains: {n_prop_chains}")
    expected_nodes = num_sensors + num_sensors * num_windows
    print(
        f"  Expected nodes: ~{expected_nodes} (8 sensors + 8*{num_windows} temporal nodes)"
    )
    if n_nodes < expected_nodes * 0.1 and num_windows > 100:
        print(
            "  WARNING: Node count seems low - KG may not be accumulating across windows"
        )
    print()

    if sample_windows is None:
        sample_windows = [0]
        if num_windows > 10:
            sample_windows.extend([num_windows // 4, num_windows // 2, num_windows - 1])

    zero_violation_count = 0
    for window_idx in sample_windows:
        if window_idx >= num_windows:
            continue
        context = extract_window_kg_context(kg, window_idx, temporal_context_windows=2)
        violations = context.get("violations", [])
        propagation = context.get("anomaly_propagation", [])
        window_stats = kg.window_stats.get(window_idx, {})
        sensor_scores = (
            {n: float(s.anomaly_score) for n, s in window_stats.items()}
            if window_stats
            else {}
        )

        prop_chain = []
        for p in propagation:
            root = p.get("root_sensor", "")
            affected = p.get("affected_sensors", [])
            if root:
                prop_chain = [root] + list(affected)
                break

        n_v = len(violations)
        if n_v == 0:
            zero_violation_count += 1

        print(f"Window {window_idx}:")
        print(f"  Violations: {n_v}")
        print(f"  Propagation chain: {prop_chain or '(none)'}")
        above = [(n, s) for n, s in sensor_scores.items() if s > 0.3]
        above.sort(key=lambda x: x[1], reverse=True)
        print(f"  Anomalous sensors (score>0.3): {above[:5] or 'none'}")
        print()

    if zero_violation_count == len(sample_windows) and len(sample_windows) > 1:
        print("WARNING: Violations are 0 across all sampled windows.")
        print("GDN-KG-LLM will perform identically to baseline - KG context is empty.")
    elif zero_violation_count > 0:
        print(
            f"Note: {zero_violation_count}/{len(sample_windows)} sampled windows had 0 violations."
        )
    else:
        print("OK: KG context appears non-trivial (violations/propagation present).")
    print("=" * 80)


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
    use_statistical_features: bool = True,
    limit: Optional[int] = None,
    use_embeddings: bool = True,
    use_adjacency_matrix: bool = False,  # Use compact adjacency matrix format instead of verbose text
) -> Dict[str, any]:
    """
    Evaluate Serialised KG->LLM method on shared dataset.

    Args:
        dataset_path: Path to shared dataset (.npz file)
        model_path: Path to trained GDN model checkpoint
        output_path: Optional path to save results JSON
        batch_size: Batch size for GDN inference
        device: Device to run on ('cuda' or 'cpu')
        model_repo: LLM model repository identifier
        max_tokens: Maximum tokens for LLM generation (None = no limit)
        temperature: LLM sampling temperature
        use_statistical_features: Whether to include statistical features in prompts
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

    # HTTP-based LLM client (requests to LM Studio, no LMInference)
    model = _HTTPLLMClient()
    tokenizer = model
    try:
        resp = requests.get(f"{HOST.rstrip('/')}/v1/models", timeout=5)
        resp.raise_for_status()
        print(f"  ✓ Connected to LM Studio at {HOST} (model: {LLM_MODEL})")
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to LM Studio at {HOST}: {e}. "
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
    print()

    # Initialize GDN Predictor (reuse from evaluate_gdn_kg.py)
    print("Initializing GDN Predictor...")
    start_time = time.time()

    try:
        import torch

        checkpoint = torch.load(model_path, map_location="cpu")
        calibrated = checkpoint.get("calibrated_thresholds", {}) if isinstance(checkpoint, dict) else {}
        sensor_threshold = float(calibrated.get("sensor", checkpoint.get("sensor_threshold", 0.30) if isinstance(checkpoint, dict) else 0.30))
        if "sensor_embeddings" in checkpoint:
            detected_embed_dim = checkpoint["sensor_embeddings"].shape[1]
            print(f"  Detected embed_dim from checkpoint: {detected_embed_dim}")
        else:
            detected_embed_dim = 32
    except Exception:
        detected_embed_dim = 32
        sensor_threshold = 0.30

    predictor = GDNPredictor(
        model_path=model_path,
        sensor_names=sensor_names,
        window_size=300,
        embed_dim=detected_embed_dim,
        top_k=3,
        hidden_dim=32,
        device=device,
    )

    print(f"  Model loaded in {time.time() - start_time:.2f} seconds")
    print()

    # Process data for KG (reuse from evaluate_gdn_kg.py)
    print("Processing data through GDN...")
    start_time = time.time()

    with tqdm(
        total=1, desc="GDN Data Processing", unit="step", dynamic_ncols=True
    ) as pbar:
        kg_data = predictor.process_for_kg(
            X_windows=normalized_windows,
            sensor_labels=sensor_labels_true,  # Ground truth kept for evaluation only
            window_labels=window_labels_true,  # Ground truth kept for evaluation only
            batch_size=batch_size,
            apply_global_mask=False,
        )
        pbar.update(1)

    gdn_time = time.time() - start_time
    print(f"  GDN processing completed in {gdn_time:.2f} seconds")
    print()

    # Extract and store embeddings if available
    embedding_time = 0.0
    similarity_edges = None
    if use_embeddings and "window_embeddings" in kg_data:
        print("Extracting and storing window embeddings...")
        start_time = time.time()

        window_embeddings = kg_data["window_embeddings"]
        distances_to_normal = kg_data["distances_to_normal"]
        distances_to_anomalous = kg_data["distances_to_anomalous"]
        center_embeddings = kg_data["center_embeddings"]

        # Store embeddings in kg (will be created below)
        # We'll do this after kg is created

        embedding_time = time.time() - start_time
        print(f"  Embeddings extracted in {embedding_time:.2f} seconds")
        print()

    # Build Knowledge Graph using GDN predictions (not ground truth labels)
    print("Building Knowledge Graph...")
    start_time = time.time()

    kg = KnowledgeGraph(
        sensor_names=kg_data["sensor_names"],
        sensor_embeddings=kg_data["sensor_embeddings"],
        adjacency_matrix=kg_data["adjacency_matrix"],
    )
    kg.construct(
        X_windows=kg_data["X_windows"],
        gdn_predictions=kg_data[
            "gdn_predictions"
        ],  # GDN predictions, not ground truth labels
        X_windows_unnormalized=kg_data.get("X_windows_unnormalized"),
        sensor_labels_true=sensor_labels_true,  # For data-driven thresholds only
        window_labels_true=window_labels_true,  # For data-driven thresholds only
    )

    kg_time = time.time() - start_time
    print(f"  Knowledge Graph built in {kg_time:.2f} seconds")
    print(f"  Nodes: {kg.number_of_nodes()}, Edges: {kg.number_of_edges()}")
    print()

    # Store embeddings in kg if available
    if use_embeddings and "window_embeddings" in kg_data:
        print("Storing window embeddings in KG...")
        start_time = time.time()

        for window_idx in range(num_windows):
            if window_idx < len(window_embeddings):
                # Store embeddings directly in kg.window_embeddings
                kg.window_embeddings[window_idx] = {
                    "embedding": window_embeddings[window_idx],
                    "dist_normal": distances_to_normal[window_idx],
                    "dist_anomalous": distances_to_anomalous[window_idx],
                    "confidence": 1.0
                    - (
                        distances_to_normal[window_idx]
                        / (
                            distances_to_normal[window_idx]
                            + distances_to_anomalous[window_idx]
                            + 1e-8
                        )
                    ),
                }

        print(f"  Stored embeddings for {len(kg.window_embeddings)} windows")
        print(
            f"  Embedding storage completed in {time.time() - start_time:.2f} seconds"
        )
        print()

        # Compute window similarities
        print("Computing window similarities...")
        start_time = time.time()
        similarity_edges = compute_window_similarity(kg.window_embeddings, k=5)
        similarity_time = time.time() - start_time
        print(
            f"  Computed {len(similarity_edges)} similarity edges in {similarity_time:.2f} seconds"
        )
        print()

    # Run LLM predictions with KG context (direct LLM, in-memory KG only)
    print("Running LLM predictions with KG context...")
    window_labels_pred = []
    sensor_labels_pred = []  # Filtered (root-only) predictions
    sensor_labels_pred_raw = []  # Raw (all sensors) predictions
    fault_types_pred = []
    reasoning_list = []
    processing_times = []

    with tqdm(
        total=num_windows,
        desc="KG-LLM",
        unit="window",
        dynamic_ncols=True,
        mininterval=1,
    ) as pbar:
        for window_idx in range(num_windows):
            start_time = time.time()

            kg_context = extract_window_kg_context(
                kg, window_idx, temporal_context_windows=2
            )
            window_stats = kg.window_stats.get(window_idx, {})
            sensor_scores = {
                name: float(stats.anomaly_score)
                for name, stats in window_stats.items()
            }
            violations = []
            for rel in kg_context.get("violations", []):
                a = rel.get("source", "")
                b = rel.get("target", "")
                exp = float(
                    rel.get(
                        "expected_correlation_gdn",
                        rel.get("expected_correlation", 0),
                    )
                )
                act = float(rel.get("correlation", 0))
                if a and b:
                    violations.append((a, b, exp, act))
            propagation_chain = []
            for prop in kg_context.get("anomaly_propagation", []):
                root = prop.get("root_sensor", "")
                affected = prop.get("affected_sensors", [])
                if root:
                    propagation_chain = [root] + list(affected)
                    break

            per_sensor = predictor.per_sensor_thresholds
            sensor_thresholds_dict = (
                {sensor_names[i]: float(per_sensor[i]) for i in range(len(sensor_names))}
                if len(per_sensor) == len(sensor_names)
                else None
            )
            messages = build_kag_prompt(
                window_idx,
                sensor_scores,
                violations,
                propagation_chain,
                sensor_names,
                sensor_threshold,
                sensor_thresholds=sensor_thresholds_dict,
            )
            try:
                raw_json = _call_llm_http(messages)
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

            # Apply root-only filtering for precision improvement (if not already filtered)
            if "sensor_labels_raw" not in prediction:
                sensor_labels_filtered = filter_sensor_labels_to_root_only(
                    prediction, sensor_names
                )
                sensor_labels_raw = prediction.get(
                    "sensor_labels", sensor_labels_filtered.copy()
                )
            else:
                sensor_labels_filtered = prediction.get(
                    "sensor_labels", np.zeros(len(sensor_names), dtype=np.float32)
                )
                sensor_labels_raw = prediction.get(
                    "sensor_labels_raw", sensor_labels_filtered.copy()
                )

            window_labels_pred.append(prediction["window_label"])
            sensor_labels_pred.append(
                sensor_labels_filtered
            )  # Use filtered (root-only) for metrics
            sensor_labels_pred_raw.append(sensor_labels_raw)  # Keep raw for analysis
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
    sensor_labels_pred = np.array(sensor_labels_pred)  # Filtered (root-only)
    sensor_labels_pred_raw = np.array(sensor_labels_pred_raw)  # Raw (all sensors)

    avg_processing_time = np.mean(processing_times)
    total_processing_time = np.sum(processing_times)
    llm_time = total_processing_time

    print(f"  Average processing time: {avg_processing_time:.4f} seconds/window")
    print(f"  Total LLM processing time: {llm_time:.2f} seconds")
    print()

    # Convert window_labels_true to sensor-indexed format (0-8)
    # The dataset stores window_labels as window indices, not sensor-indexed labels
    window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
    for i in range(len(window_labels_true)):
        faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
        if len(faulty_indices) > 0:
            window_labels_true_converted[i] = (
                faulty_indices[0] + 1
            )  # 1-indexed (sensor 0 -> label 1)
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
    total_time = gdn_time + kg_time + llm_time
    metrics["efficiency"] = {
        "gdn_processing_time_seconds": float(gdn_time),
        "kg_build_time_seconds": float(kg_time),
        "llm_processing_time_seconds": float(llm_time),
        "total_processing_time_seconds": float(total_time),
        "windows_per_second": float(num_windows / total_time),
        "kg_nodes": int(kg.number_of_nodes()),
        "kg_edges": int(kg.number_of_edges()),
    }

    # Print report
    report = format_metrics_report(metrics)
    print(report)

    # Save results
    results = {
        "method": "gdn_kg_llm",
        "dataset": str(dataset_path),
        "gdn_model": str(model_path),
        "llm_model": model_repo,
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
        "--dataset", type=str, required=True, help="Path to shared dataset .npz file"
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
        "--no-statistical-features",
        action="store_true",
        help="Disable statistical features in prompts",
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

    args = parser.parse_args()

    if args.sanity_check:
        run_kg_sanity_check(
            dataset_path=Path(args.dataset),
            model_path=Path(args.model_path),
            batch_size=args.batch_size,
            device=args.device,
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
        use_statistical_features=not args.no_statistical_features,
        limit=args.limit,
        use_embeddings=args.use_embeddings,
        use_adjacency_matrix=args.use_adjacency_matrix,
    )


if __name__ == "__main__":
    main()
