#!/usr/bin/env python3
"""
Unified Knowledge Graph Creation Script

Creates knowledge graphs from trained GDN models by:
1. Loading trained GDN model
2. Running inference on data windows
3. Building knowledge graph with temporal relationships
4. Saving as NetworkX pickle or JSON file

Usage:
    python kg/create_kg.py --model_path checkpoints/stage2_clean_phase2_50ep_*/stage2_clean_best.pt --data_path data/carOBD/obdiidata --output_path kg_output/graph.pkl

Note: Only Stage 2 clean checkpoints (stage2_clean_best.pt from train_stage2_clean.py) are accepted.
"""

import os
import sys
import argparse
import pickle
import json
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gdn_model import GDN

# ============================================================================
# Sensor Metadata and Subsystem Mapping
# ============================================================================

SENSOR_SUBSYSTEMS = {
    "ENGINE_RPM ()": "Engine System",
    "ENGINE_LOAD ()": "Engine System",
    "COOLANT_TEMPERATURE ()": "Engine System",
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()": "Fuel System",
    "LONG_TERM_FUEL_TRIM_BANK_1 ()": "Fuel System",
    "INTAKE_MANIFOLD_PRESSURE ()": "Intake System",
    "THROTTLE ()": "Intake System",
    "VEHICLE_SPEED ()": "Drivetrain",
}

SENSOR_DESCRIPTIONS = {
    "ENGINE_RPM ()": {
        "description": "Engine speed in revolutions per minute",
        "unit": "rpm",
        "normal_range": (600, 6000),
        "fault_injection_eligible": True,
        "subsystem": "Engine System",
    },
    "VEHICLE_SPEED ()": {
        "description": "Vehicle speed sensor reading",
        "unit": "mi/h",
        "normal_range": (0, 120),
        "fault_injection_eligible": True,
        "subsystem": "Drivetrain",
    },
    "THROTTLE ()": {
        "description": "Throttle position percentage",
        "unit": "%",
        "normal_range": (0, 100),
        "fault_injection_eligible": True,
        "subsystem": "Intake System",
    },
    "ENGINE_LOAD ()": {
        "description": "Calculated engine load value",
        "unit": "%",
        "normal_range": (0, 100),
        "fault_injection_eligible": True,
        "subsystem": "Engine System",
    },
    "COOLANT_TEMPERATURE ()": {
        "description": "Engine coolant temperature",
        "unit": "C",
        "normal_range": (70, 110),
        "fault_injection_eligible": True,
        "subsystem": "Engine System",
    },
    "INTAKE_MANIFOLD_PRESSURE ()": {
        "description": "Intake manifold absolute pressure",
        "unit": "psig",
        "normal_range": (0, 20),
        "fault_injection_eligible": True,
        "subsystem": "Intake System",
    },
    "SHORT_TERM_FUEL_TRIM_BANK_1 ()": {
        "description": "Short-term fuel trim adjustment",
        "unit": "%",
        "normal_range": (-25, 25),
        "fault_injection_eligible": True,
        "subsystem": "Fuel System",
    },
    "LONG_TERM_FUEL_TRIM_BANK_1 ()": {
        "description": "Long-term fuel trim adjustment",
        "unit": "%",
        "normal_range": (-25, 25),
        "fault_injection_eligible": True,
        "subsystem": "Fuel System",
    },
}

EXPECTED_CORRELATIONS = {
    ("ENGINE_RPM ()", "VEHICLE_SPEED ()"): {
        "type": "expected_to_increase_with",
        "strength": "strong",
        "description": "RPM and vehicle speed should correlate positively under normal conditions",
    },
    ("THROTTLE ()", "ENGINE_LOAD ()"): {
        "type": "expected_to_increase_with",
        "strength": "strong",
        "description": "Throttle position and engine load should increase together",
    },
    ("THROTTLE ()", "INTAKE_MANIFOLD_PRESSURE ()"): {
        "type": "expected_to_increase_with",
        "strength": "moderate",
        "description": "Throttle opening increases intake manifold pressure",
    },
    ("ENGINE_RPM ()", "COOLANT_TEMPERATURE ()"): {
        "type": "correlates_with",
        "strength": "weak",
        "description": "Higher RPM may increase coolant temperature over time",
    },
    ("SHORT_TERM_FUEL_TRIM_BANK_1 ()", "LONG_TERM_FUEL_TRIM_BANK_1 ()"): {
        "type": "correlates_with",
        "strength": "moderate",
        "description": "Short-term and long-term fuel trim adjustments are related",
    },
    ("INTAKE_MANIFOLD_PRESSURE ()", "SHORT_TERM_FUEL_TRIM_BANK_1 ()"): {
        "type": "correlates_with",
        "strength": "moderate",
        "description": "Intake pressure affects fuel trim calculations",
    },
}

# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class WindowStats:
    """Statistics for a sensor within a window."""

    mean: float
    std: float
    min: float
    max: float
    variance: float
    num_zeros: int
    trend: float
    median: float
    q25: float
    q75: float
    variation_from_normal: float
    anomaly_score: float  # GDN prediction score


# ============================================================================
# Helper Functions for GDN Operations
# ============================================================================


def load_gdn_model(model_path: str, device: str = "cpu") -> Tuple[GDN, Dict[str, Any]]:
    """
    Load trained GDN model from checkpoint.

    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Tuple of (model, metadata_dict) where metadata contains:
        - sensor_names: List of sensor names
        - window_size: Window size
        - embed_dim: Embedding dimension
        - top_k: Top-K neighbors
        - hidden_dim: Hidden dimension
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "create_kg expects a dict checkpoint. Got raw state dict."
        )
    if checkpoint.get("stage") != 2 or checkpoint.get("stage2_mode") != "clean":
        print(
            "  Note: Checkpoint may not be Stage 2 clean. Attempting to load anyway..."
        )

    # Extract metadata from checkpoint
    if isinstance(checkpoint, dict):
        sensor_names = checkpoint.get("sensor_names", [])
        window_size = checkpoint.get("window_size", 300)
        embed_dim = checkpoint.get("embed_dim", 32)
        top_k = checkpoint.get("top_k", 5)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        # Fallback if checkpoint is just state_dict
        state_dict = checkpoint
        sensor_names = []
        window_size = 300
        embed_dim = 32
        top_k = 5
        hidden_dim = 64

    # Initialize GDN model
    num_sensors = len(sensor_names) if sensor_names else 8  # Default to 8 sensors

    model = GDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=embed_dim,
        top_k=top_k,
        hidden_dim=hidden_dim,
    ).to(device)

    # Remap GAT keys for PyG version compatibility (lin vs lin_src/lin_dst)
    state_dict = dict(state_dict)
    model_expects_lin = "gat.lin.weight" in model.state_dict()
    model_expects_lin_src = "gat.lin_src.weight" in model.state_dict()
    ckpt_has_lin = "gat.lin.weight" in state_dict
    ckpt_has_lin_src = "gat.lin_src.weight" in state_dict

    if model_expects_lin_src and ckpt_has_lin:
        lin_weight = state_dict.pop("gat.lin.weight")
        state_dict["gat.lin_src.weight"] = lin_weight.clone()
        state_dict["gat.lin_dst.weight"] = lin_weight.clone()
    elif model_expects_lin and ckpt_has_lin_src:
        state_dict["gat.lin.weight"] = state_dict.pop("gat.lin_src.weight")
        state_dict.pop("gat.lin_dst.weight", None)

    model.load_state_dict(state_dict, strict=True)
    print("  ✓ Model loaded successfully")

    model.eval()

    metadata = {
        "sensor_names": sensor_names,
        "window_size": window_size,
        "embed_dim": embed_dim,
        "top_k": top_k,
        "hidden_dim": hidden_dim,
    }

    print(f"✓ Loaded model: {num_sensors} sensors, window_size={window_size}")
    return model, metadata


def extract_sensor_embeddings(model: GDN) -> np.ndarray:
    """
    Extract learned sensor embeddings from the model.

    Args:
        model: Trained GDN model

    Returns:
        sensor_embeddings: (num_sensors, embed_dim) numpy array
    """
    with torch.no_grad():
        embeddings = model.sensor_embeddings.cpu().numpy()
    return embeddings


def compute_adjacency_matrix(sensor_embeddings: np.ndarray) -> np.ndarray:
    """
    Compute adjacency matrix from sensor embeddings using cosine similarity.

    Args:
        sensor_embeddings: (num_sensors, embed_dim) numpy array

    Returns:
        adjacency_matrix: (num_sensors, num_sensors) numpy array
            Values are cosine similarities scaled to [0.1, 1.0] range
    """
    # Normalize embeddings (L2 normalization for cosine similarity)
    embeddings_norm = F.normalize(
        torch.from_numpy(sensor_embeddings), p=2, dim=1
    ).numpy()

    # Compute cosine similarity matrix
    similarity_matrix = np.dot(
        embeddings_norm, embeddings_norm.T
    )  # (num_sensors, num_sensors)

    # Scale from [-1, 1] to [0.1, 1.0]
    adjacency_matrix = (similarity_matrix + 1.0) / 2.0
    adjacency_matrix = np.clip(adjacency_matrix, 0.1, 1.0)

    # Zero out diagonal (no self-loops)
    np.fill_diagonal(adjacency_matrix, 0.0)

    return adjacency_matrix


def predict_anomalies(
    model: GDN,
    X_windows: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
    return_global: bool = False,
    apply_global_mask: bool = True,
    global_mask_threshold: float = 0.5,
) -> np.ndarray:
    """
    Run inference on data windows to get sensor anomaly probabilities.

    Args:
        model: Trained GDN model
        X_windows: (num_windows, window_size, num_sensors) input windows
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        sensor_probs: (num_windows, num_sensors) numpy array of anomaly probabilities
            (or raw/masked if global mask is enabled).
            If `return_global=True`, returns tuple of (sensor_probs, global_probs).
    """
    if (
        X_windows.dim() != 3
        if isinstance(X_windows, torch.Tensor)
        else len(X_windows.shape) != 3
    ):
        raise ValueError(
            f"Expected 3D input (num_windows, window_size, num_sensors), got shape {X_windows.shape}"
        )

    # Convert to tensor if needed
    if isinstance(X_windows, np.ndarray):
        X_windows = torch.from_numpy(X_windows).float()

    num_windows = X_windows.shape[0]
    X_windows = X_windows.to(device)

    all_sensor_probs = []
    all_global_probs = []
    model.eval()

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = X_windows[i : i + batch_size]
            if apply_global_mask or return_global:
                sensor_logits, global_logits = model(batch, return_global=True)
                global_probs = torch.sigmoid(global_logits)
            else:
                sensor_logits = model(batch, return_global=False)
                global_probs = torch.zeros((batch.shape[0],), device=batch.device)

            sensor_probs = torch.sigmoid(
                sensor_logits
            )  # Convert logits to probabilities
            if apply_global_mask:
                support = (global_probs >= global_mask_threshold).float().unsqueeze(1)
                sensor_probs = sensor_probs * support
            all_sensor_probs.append(sensor_probs.cpu().numpy())
            if return_global:
                all_global_probs.append(global_probs.cpu().numpy())

    sensor_probs = np.concatenate(all_sensor_probs, axis=0)
    if return_global:
        global_probs = np.concatenate(all_global_probs, axis=0)
        return sensor_probs, global_probs
    return sensor_probs


class GDNPredictor:
    """
    Wrapper for GDN model loading and inference. Provides predict() and process_for_kg()
    for compatibility with evaluation scripts.
    """

    def __init__(
        self,
        model_path: str,
        sensor_names: List[str],
        window_size: int = 300,
        embed_dim: int = 32,
        top_k: int = 3,
        hidden_dim: int = 32,
        device: str = "cpu",
    ):
        self.model, self._metadata = load_gdn_model(model_path, device)
        self.sensor_names = sensor_names
        self.window_size = window_size
        self.device = device

    def predict(
        self,
        X_windows: np.ndarray,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Return (num_windows, num_sensors) anomaly scores."""
        return predict_anomalies(
            self.model,
            X_windows,
            batch_size=batch_size,
            device=self.device,
        )

    def process_for_kg(
        self,
        X_windows: np.ndarray,
        sensor_labels: Optional[np.ndarray] = None,
        window_labels: Optional[np.ndarray] = None,
        batch_size: int = 32,
        apply_global_mask: bool = False,
    ) -> Dict[str, Any]:
        """Return dict with sensor_names, sensor_embeddings, adjacency_matrix, X_windows, gdn_predictions."""
        gdn_predictions = predict_anomalies(
            self.model,
            X_windows,
            batch_size=batch_size,
            device=self.device,
            apply_global_mask=apply_global_mask,
        )
        sensor_embeddings = extract_sensor_embeddings(self.model)
        adjacency_matrix = compute_adjacency_matrix(sensor_embeddings)
        return {
            "sensor_names": self.sensor_names,
            "sensor_embeddings": sensor_embeddings,
            "adjacency_matrix": adjacency_matrix,
            "X_windows": X_windows,
            "gdn_predictions": gdn_predictions,
        }


def extract_window_embeddings(
    model: GDN, X_windows: np.ndarray, batch_size: int = 32, device: str = "cpu"
) -> np.ndarray:
    """
    Extract window embeddings for distance-based scoring.

    Args:
        model: Trained GDN model
        X_windows: (num_windows, window_size, num_sensors) input windows
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        embeddings: (num_windows, hidden_dim) numpy array of embeddings
    """
    if isinstance(X_windows, np.ndarray):
        X_windows = torch.from_numpy(X_windows).float()

    num_windows = X_windows.shape[0]
    X_windows = X_windows.to(device)

    all_embeddings = []
    model.eval()

    with torch.no_grad():
        for i in range(0, num_windows, batch_size):
            batch = X_windows[i : i + batch_size]
            embeddings = model.get_embeddings(batch)  # (B, hidden_dim)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    return embeddings


# ============================================================================
# KnowledgeGraph Class
# ============================================================================


class KnowledgeGraph:
    """
    Unified Knowledge Graph class that handles construction and per-window retrieval.

    This class combines GDN inference results with KG construction, providing:
    - Temporal relationship tracking
    - Anomaly propagation analysis
    - Per-window KG retrieval for evaluation
    - NetworkX serialization
    - Neo4j export
    """

    def __init__(
        self,
        sensor_names: List[str],
        sensor_embeddings: np.ndarray,
        adjacency_matrix: np.ndarray,
    ):
        """
        Initialize the Knowledge Graph.

        Args:
            sensor_names: List of sensor names (must match order in data)
            sensor_embeddings: Learned sensor embeddings from GDN (num_sensors, embed_dim)
            adjacency_matrix: Adjacency matrix from GDN (num_sensors, num_sensors)
        """
        self.sensor_names = sensor_names
        self.sensor_embeddings = sensor_embeddings
        self.adjacency_matrix = adjacency_matrix
        self.num_sensors = len(sensor_names)
        self.sensor_to_idx = {name: idx for idx, name in enumerate(sensor_names)}

        # Initialize graph structures
        self.kg = nx.MultiDiGraph()  # Main knowledge graph
        self.window_graphs = {}  # Per-window graphs
        self.window_stats = {}  # Per-window statistics
        self.window_embeddings = {}  # Per-window embedding data
        self.anomaly_propagation_chains = []  # Fault propagation chains
        self.distribution_thresholds = None  # Distribution-based thresholds
        self.X_windows = None  # Normalized windows
        self.X_windows_unnormalized = None  # Unnormalized windows

        # Initialize sensor nodes
        self._initialize_sensor_nodes()

    def number_of_nodes(self) -> int:
        """Return the number of nodes in the graph (delegates to inner NetworkX graph)."""
        return self.kg.number_of_nodes()

    def number_of_edges(self) -> int:
        """Return the number of edges in the graph (delegates to inner NetworkX graph)."""
        return self.kg.number_of_edges()

    def _initialize_sensor_nodes(self):
        """Initialize sensor nodes in the knowledge graph with metadata"""
        for sensor_name in self.sensor_names:
            subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, "Unknown")
            description = SENSOR_DESCRIPTIONS.get(sensor_name, {})

            self.kg.add_node(
                sensor_name,
                type="sensor",
                subsystem=subsystem,
                description=description.get("description", ""),
                unit=description.get("unit", ""),
                normal_range=description.get("normal_range", (0, 100)),
                fault_injection_eligible=description.get(
                    "fault_injection_eligible", False
                ),
            )

    def construct(
        self,
        X_windows: np.ndarray,
        gdn_predictions: np.ndarray,
        X_windows_unnormalized: Optional[np.ndarray] = None,
        sensor_labels_true: Optional[np.ndarray] = None,
        window_labels_true: Optional[np.ndarray] = None,
    ) -> "KnowledgeGraph":
        """
        Main method: Build KG by traversing windows temporally.

        Builds knowledge graph from GDN model outputs (predictions), NOT ground truth labels.

        Args:
            X_windows: (num_windows, window_size, num_sensors) normalized sensor data windows
            gdn_predictions: (num_windows, num_sensors) GDN anomaly scores (0.0-1.0) per sensor per window
            X_windows_unnormalized: (num_windows, window_size, num_sensors) unnormalized windows (optional)
            sensor_labels_true: Optional (num_windows, num_sensors) ground truth labels (for thresholds only)
            window_labels_true: Optional (num_windows,) ground truth window labels (for thresholds only)

        Returns:
            self (for method chaining)
        """
        num_windows = len(X_windows)

        # Store window data
        self.X_windows = X_windows
        self.X_windows_unnormalized = X_windows_unnormalized

        # First pass: Process all windows
        print(f"Processing {num_windows} windows...")
        for window_idx in tqdm(range(num_windows), desc="Building KG"):
            window_data = X_windows[window_idx]
            window_gdn_scores = gdn_predictions[window_idx]
            self._process_window(window_idx, window_data, window_gdn_scores)

            if window_idx > 0:
                self._build_temporal_edges(
                    window_idx - 1,
                    window_idx,
                    X_windows[window_idx - 1],
                    window_data,
                    gdn_predictions[window_idx - 1],
                    window_gdn_scores,
                )

        # Compute distribution-based thresholds
        self.distribution_thresholds = self._compute_distribution_thresholds(
            gdn_predictions, sensor_labels_true, window_labels_true
        )

        # Second pass: Update edges with distribution-based thresholds
        for window_idx in range(num_windows):
            window_data = X_windows[window_idx]
            window_gdn_scores = gdn_predictions[window_idx]
            self._update_edges_with_thresholds(
                window_idx, window_data, window_gdn_scores
            )

        # Track anomaly propagation
        self._track_anomaly_propagation(gdn_predictions)

        print(
            f"✓ Knowledge Graph built: {self.number_of_nodes()} nodes, {self.number_of_edges()} edges"
        )
        return self

    def _process_window(
        self, window_idx: int, window_data: np.ndarray, gdn_scores: np.ndarray
    ):
        """Process a single window and build its graph."""
        window_stats = {}
        for sensor_idx, sensor_name in enumerate(self.sensor_names):
            sensor_values = window_data[:, sensor_idx]

            stats = WindowStats(
                mean=float(np.mean(sensor_values)),
                std=float(np.std(sensor_values)),
                min=float(np.min(sensor_values)),
                max=float(np.max(sensor_values)),
                variance=float(np.var(sensor_values)),
                num_zeros=int(np.sum(sensor_values == 0)),
                trend=float(
                    np.polyfit(np.arange(len(sensor_values)), sensor_values, 1)[0]
                )
                if len(sensor_values) > 1
                else 0.0,
                median=float(np.median(sensor_values)),
                q25=float(np.percentile(sensor_values, 25)),
                q75=float(np.percentile(sensor_values, 75)),
                variation_from_normal=self._compute_variation_from_normal(
                    sensor_name, sensor_values
                ),
                anomaly_score=float(gdn_scores[sensor_idx]),
            )
            window_stats[sensor_name] = stats

        self.window_stats[window_idx] = window_stats

        # Build correlation matrix
        correlation_matrix = np.corrcoef(window_data.T)
        window_graph = nx.Graph()

        # Get thresholds
        if self.distribution_thresholds:
            thresholds = self.distribution_thresholds
            anomaly_threshold_per_sensor = thresholds.get(
                "anomaly_threshold_per_sensor", {}
            )
            anomaly_threshold_global = thresholds.get("anomaly_threshold_global", 0.5)
            deviation_threshold = thresholds.get("deviation_threshold", 0.3)
        else:
            anomaly_threshold_per_sensor = {}
            anomaly_threshold_global = 0.5
            deviation_threshold = 0.3

        # Add nodes
        for sensor_name, stats in window_stats.items():
            threshold = anomaly_threshold_per_sensor.get(
                sensor_name, anomaly_threshold_global
            )
            window_graph.add_node(
                sensor_name,
                window_idx=window_idx,
                mean=stats.mean,
                std=stats.std,
                min=stats.min,
                max=stats.max,
                variation_from_normal=stats.variation_from_normal,
                is_faulty=bool(stats.anomaly_score > threshold),
            )

        # Add edges
        for i, sensor_i in enumerate(self.sensor_names):
            for j, sensor_j in enumerate(self.sensor_names):
                if i >= j:
                    continue

                window_corr = correlation_matrix[i, j]
                expected_corr = self.adjacency_matrix[i, j]

                edge_type, edge_attrs = self._infer_semantic_edge(
                    sensor_i,
                    sensor_j,
                    window_corr,
                    expected_corr,
                    window_stats[sensor_i],
                    window_stats[sensor_j],
                    anomaly_threshold=anomaly_threshold_global,
                    deviation_threshold=deviation_threshold,
                )

                if edge_type:
                    edge_data = {
                        "window_idx": window_idx,
                        "edge_type": edge_type,
                        "correlation": float(window_corr),
                        **edge_attrs,
                    }
                    window_graph.add_edge(sensor_i, sensor_j, **edge_data)
                    self.kg.add_edge(sensor_i, sensor_j, **edge_data)

        self.window_graphs[window_idx] = window_graph

    def _infer_semantic_edge(
        self,
        sensor_i: str,
        sensor_j: str,
        window_corr: float,
        expected_corr_gdn: float,
        stats_i: WindowStats,
        stats_j: WindowStats,
        anomaly_threshold: float = 0.5,
        deviation_threshold: float = 0.3,
    ) -> Tuple[Optional[str], Dict]:
        """Infer semantic edge between two sensors."""
        if np.isnan(window_corr) or np.isinf(window_corr) or abs(window_corr) < 0.1:
            return None, {}

        edge_type = "correlates_with"
        edge_attrs = {
            "correlation_strength": abs(window_corr),
            "correlation_direction": "positive" if window_corr > 0 else "negative",
            "expected_correlation_gdn": float(expected_corr_gdn),
            "deviation_from_gdn": float(abs(window_corr - expected_corr_gdn)),
            "gdn_score_source": float(stats_i.anomaly_score),
            "gdn_score_target": float(stats_j.anomaly_score),
        }

        # Check domain knowledge
        expected_rel = EXPECTED_CORRELATIONS.get(
            (sensor_i, sensor_j)
        ) or EXPECTED_CORRELATIONS.get((sensor_j, sensor_i))
        if expected_rel:
            edge_attrs["domain_expected_type"] = expected_rel["type"]
            edge_attrs["domain_expected_strength"] = expected_rel["strength"]
            # Simple violation check
            if "increase_with" in expected_rel["type"] and window_corr < 0:
                edge_attrs["violates_domain_expectation"] = True
            else:
                edge_attrs["violates_domain_expectation"] = False
        else:
            edge_attrs["violates_domain_expectation"] = False

        # Check GDN expectation violation
        deviation = abs(window_corr - expected_corr_gdn)
        edge_attrs["violates_gdn_expectation"] = deviation > deviation_threshold

        # Potential fault indicator
        if edge_attrs.get("violates_domain_expectation", False) or edge_attrs.get(
            "violates_gdn_expectation", False
        ):
            if (
                stats_i.anomaly_score > anomaly_threshold
                or stats_j.anomaly_score > anomaly_threshold
            ):
                edge_attrs["potential_fault_indicator"] = True
            else:
                edge_attrs["potential_fault_indicator"] = False
        else:
            edge_attrs["potential_fault_indicator"] = False

        return edge_type, edge_attrs

    def _build_temporal_edges(
        self,
        prev_window_idx: int,
        curr_window_idx: int,
        prev_window_data: np.ndarray,
        curr_window_data: np.ndarray,
        prev_gdn_scores: np.ndarray,
        curr_gdn_scores: np.ndarray,
    ):
        """Build temporal edges connecting consecutive windows."""
        prev_stats = self.window_stats[prev_window_idx]
        curr_stats = self.window_stats[curr_window_idx]
        prediction_threshold = 0.5

        for sensor_name in self.sensor_names:
            prev_stat = prev_stats[sensor_name]
            curr_stat = curr_stats[sensor_name]

            # Temporal continuation
            self.kg.add_edge(
                f"{sensor_name}@window_{prev_window_idx}",
                f"{sensor_name}@window_{curr_window_idx}",
                edge_type="temporal_continuation",
                sensor=sensor_name,
                source_window=prev_window_idx,
                target_window=curr_window_idx,
            )

            # Value change
            value_change = curr_stat.mean - prev_stat.mean
            if abs(value_change) > 0.1:
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type="value_change",
                    sensor=sensor_name,
                    value_change=float(value_change),
                    source_window=prev_window_idx,
                    target_window=curr_window_idx,
                )

            # Anomaly propagation
            prev_faulty = prev_stat.anomaly_score > prediction_threshold
            curr_faulty = curr_stat.anomaly_score > prediction_threshold

            if prev_faulty and curr_faulty:
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type="anomaly_propagation",
                    sensor=sensor_name,
                    propagation_type="persists",
                    source_window=prev_window_idx,
                    target_window=curr_window_idx,
                )
            elif not prev_faulty and curr_faulty:
                self.kg.add_edge(
                    f"{sensor_name}@window_{prev_window_idx}",
                    f"{sensor_name}@window_{curr_window_idx}",
                    edge_type="anomaly_propagation",
                    sensor=sensor_name,
                    propagation_type="appears",
                    source_window=prev_window_idx,
                    target_window=curr_window_idx,
                )

    def _track_anomaly_propagation(
        self, gdn_predictions: np.ndarray, threshold: float = 0.5
    ):
        """Track how faults propagate across windows."""
        num_windows = len(gdn_predictions)
        first_occurrence_all = {}

        for window_idx in range(num_windows):
            for sensor_idx, sensor_name in enumerate(self.sensor_names):
                if gdn_predictions[window_idx, sensor_idx] > threshold:
                    if sensor_name not in first_occurrence_all:
                        first_occurrence_all[sensor_name] = window_idx

        for root_sensor, root_window in first_occurrence_all.items():
            chain = {
                "root_sensor": root_sensor,
                "root_window": root_window,
                "gdn_score": float(
                    gdn_predictions[root_window, self.sensor_to_idx[root_sensor]]
                ),
                "affected_sensors": [],
                "propagation_timeline": [],
            }

            affected_first_occurrence = {}
            for window_idx in range(root_window + 1, num_windows):
                for sensor_idx, sensor_name in enumerate(self.sensor_names):
                    if gdn_predictions[window_idx, sensor_idx] > threshold:
                        if sensor_name in first_occurrence_all:
                            sensor_first_window = first_occurrence_all[sensor_name]
                            if (
                                sensor_first_window > root_window
                                and sensor_name not in affected_first_occurrence
                            ):
                                affected_first_occurrence[sensor_name] = (
                                    sensor_first_window
                                )
                                chain["propagation_timeline"].append(
                                    {
                                        "window": sensor_first_window,
                                        "affected_sensors": [sensor_name],
                                        "gdn_score": float(
                                            gdn_predictions[
                                                sensor_first_window, sensor_idx
                                            ]
                                        ),
                                    }
                                )

            if chain["propagation_timeline"]:
                chain["propagation_timeline"].sort(key=lambda x: x["window"])
                self.anomaly_propagation_chains.append(chain)

    def _compute_variation_from_normal(
        self, sensor_name: str, sensor_values: np.ndarray
    ) -> float:
        """Compute variation from normal operating range"""
        description = SENSOR_DESCRIPTIONS.get(sensor_name, {})
        normal_range = description.get("normal_range", (0, 100))
        mean_value = np.mean(sensor_values)
        normal_mean = (normal_range[0] + normal_range[1]) / 2
        normal_span = normal_range[1] - normal_range[0]
        if normal_span == 0:
            return 0.0
        variation = abs(mean_value - normal_mean) / normal_span
        return float(variation)

    def _compute_distribution_thresholds(
        self,
        gdn_predictions: np.ndarray,
        sensor_labels_true: Optional[np.ndarray] = None,
        window_labels_true: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Compute distribution-based thresholds."""
        all_anomaly_scores = []
        per_sensor_scores = {sensor_name: [] for sensor_name in self.sensor_names}

        for window_idx, window_stats in self.window_stats.items():
            for sensor_name, stats in window_stats.items():
                score = stats.anomaly_score
                all_anomaly_scores.append(score)
                per_sensor_scores[sensor_name].append(score)

        if len(all_anomaly_scores) > 0:
            anomaly_threshold_global = float(np.percentile(all_anomaly_scores, 95))
        else:
            anomaly_threshold_global = 0.5

        anomaly_threshold_per_sensor = {}
        for sensor_name in self.sensor_names:
            if len(per_sensor_scores[sensor_name]) > 0:
                anomaly_threshold_per_sensor[sensor_name] = float(
                    np.percentile(per_sensor_scores[sensor_name], 95)
                )
            else:
                anomaly_threshold_per_sensor[sensor_name] = anomaly_threshold_global

        return {
            "anomaly_threshold_global": anomaly_threshold_global,
            "anomaly_threshold_per_sensor": anomaly_threshold_per_sensor,
            "deviation_threshold": 0.3,
        }

    def _update_edges_with_thresholds(
        self, window_idx: int, window_data: np.ndarray, gdn_scores: np.ndarray
    ):
        """Update edges with distribution-based thresholds."""
        if window_idx not in self.window_graphs:
            return

        graph = self.window_graphs[window_idx]
        window_stats = self.window_stats.get(window_idx, {})
        thresholds = self.distribution_thresholds

        if not thresholds:
            return

        anomaly_threshold = thresholds.get("anomaly_threshold_global", 0.5)
        deviation_threshold = thresholds.get("deviation_threshold", 0.3)
        correlation_matrix = np.corrcoef(window_data.T)

        for i, sensor_i in enumerate(self.sensor_names):
            for j, sensor_j in enumerate(self.sensor_names):
                if i >= j or not graph.has_edge(sensor_i, sensor_j):
                    continue

                window_corr = correlation_matrix[i, j]
                expected_corr_gdn = self.adjacency_matrix[i, j]
                deviation_from_gdn = abs(window_corr - expected_corr_gdn)

                stats_i = window_stats.get(sensor_i)
                stats_j = window_stats.get(sensor_j)

                if stats_i is None or stats_j is None:
                    continue

                edge_attrs = graph[sensor_i][sensor_j]
                if edge_attrs.get("window_idx") == window_idx:
                    edge_attrs["deviation_from_gdn"] = float(deviation_from_gdn)
                    edge_attrs["violates_gdn_expectation"] = (
                        deviation_from_gdn > deviation_threshold
                    )

                    if edge_attrs.get(
                        "violates_gdn_expectation", False
                    ) or edge_attrs.get("violates_domain_expectation", False):
                        if (
                            stats_i.anomaly_score > anomaly_threshold
                            or stats_j.anomaly_score > anomaly_threshold
                        ):
                            edge_attrs["potential_fault_indicator"] = True
                        else:
                            edge_attrs["potential_fault_indicator"] = False

    def get_window_kg(
        self, window_idx: int, temporal_context_windows: int = 2
    ) -> Dict[str, Any]:
        """
        Retrieve KG context for a specific window (for evaluation).

        Args:
            window_idx: Index of the window
            temporal_context_windows: Number of previous windows to include

        Returns:
            Dictionary with KG context for the window
        """
        context = {
            "entities": [],
            "relationships": [],
            "violations": [],
            "temporal_context": [],
            "anomaly_propagation": [],
            "distribution_thresholds": self.distribution_thresholds,
        }

        if window_idx not in self.window_graphs:
            return context

        window_graph = self.window_graphs[window_idx]
        window_stats = self.window_stats.get(window_idx, {})
        thresholds = self.distribution_thresholds

        # Extract entities
        for sensor_name in self.sensor_names:
            desc = SENSOR_DESCRIPTIONS.get(sensor_name, {})
            subsystem = SENSOR_SUBSYSTEMS.get(sensor_name, "Unknown")
            stat = window_stats.get(sensor_name)

            if thresholds:
                anomaly_threshold_per_sensor = thresholds.get(
                    "anomaly_threshold_per_sensor", {}
                )
                anomaly_threshold = anomaly_threshold_per_sensor.get(
                    sensor_name, thresholds.get("anomaly_threshold_global", 0.5)
                )
            else:
                anomaly_threshold = 0.5

            is_faulty = stat.anomaly_score > anomaly_threshold if stat else False

            context["entities"].append(
                {
                    "name": sensor_name,
                    "type": "Sensor",
                    "subsystem": subsystem,
                    "description": desc.get("description", ""),
                    "is_faulty": is_faulty,
                }
            )

        # Extract relationships
        prediction_threshold = 0.5
        anomalous_sensors = {
            sensor_name
            for sensor_name, stat in window_stats.items()
            if stat.anomaly_score > prediction_threshold
        }

        for u, v, data in window_graph.edges(data=True):
            correlation_strength = data.get(
                "correlation_strength", abs(data.get("correlation", 0))
            )
            is_violation = data.get("violates_domain_expectation", False) or data.get(
                "violates_gdn_expectation", False
            )
            involves_anomaly = u in anomalous_sensors or v in anomalous_sensors
            is_significant = correlation_strength >= 0.3
            potential_fault = data.get("potential_fault_indicator", False)

            if not (
                is_violation or involves_anomaly or is_significant or potential_fault
            ):
                continue

            relationship = {
                "source": u,
                "target": v,
                "relation": data.get("edge_type", "correlates_with"),
                "correlation": float(data.get("correlation", 0)),
                "correlation_strength": float(correlation_strength),
                "correlation_direction": data.get("correlation_direction", "positive"),
                "expected_correlation_gdn": float(
                    data.get("expected_correlation_gdn", 0)
                ),
                "deviation_from_gdn": float(data.get("deviation_from_gdn", 0)),
                "violates_domain_expectation": data.get(
                    "violates_domain_expectation", False
                ),
                "violates_gdn_expectation": data.get("violates_gdn_expectation", False),
                "gdn_score_source": float(data.get("gdn_score_source", 0)),
                "gdn_score_target": float(data.get("gdn_score_target", 0)),
                "potential_fault_indicator": potential_fault,
            }

            context["relationships"].append(relationship)
            if is_violation:
                context["violations"].append(relationship)

        # Extract temporal context
        for prev_idx in range(
            max(0, window_idx - temporal_context_windows), window_idx
        ):
            if prev_idx in self.window_stats:
                prev_stats = self.window_stats[prev_idx]
                temporal_info = {
                    "window_idx": prev_idx,
                    "faulty_sensors": [],
                    "anomaly_scores": {},
                }

                for sensor_name, stat in prev_stats.items():
                    if stat.anomaly_score > prediction_threshold:
                        temporal_info["faulty_sensors"].append(sensor_name)
                        temporal_info["anomaly_scores"][sensor_name] = float(
                            stat.anomaly_score
                        )

                if temporal_info["faulty_sensors"]:
                    context["temporal_context"].append(temporal_info)

        # Extract anomaly propagation
        for chain in self.anomaly_propagation_chains:
            root_window = chain.get("root_window", -1)
            if root_window == window_idx:
                context["anomaly_propagation"].append(
                    {
                        "type": "root",
                        "root_sensor": chain.get("root_sensor", ""),
                        "root_window": root_window,
                    }
                )
            else:
                for timeline_entry in chain.get("propagation_timeline", []):
                    if timeline_entry.get("window") == window_idx:
                        context["anomaly_propagation"].append(
                            {
                                "type": "propagation",
                                "root_sensor": chain.get("root_sensor", ""),
                                "root_window": root_window,
                            }
                        )
                        break

        return context

    def save(self, path: str):
        """
        Save KG to file (NetworkX pickle or JSON).

        Args:
            path: Path to save file (.pkl or .json)
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix == ".pkl":
            with open(path, "wb") as f:
                pickle.dump(self, f)
            print(f"✓ Saved KG to {path} (pickle format)")
        elif path.suffix == ".json":
            # Convert to JSON-serializable format
            data = {
                "sensor_names": self.sensor_names,
                "sensor_embeddings": self.sensor_embeddings.tolist(),
                "adjacency_matrix": self.adjacency_matrix.tolist(),
                "window_graphs": {
                    str(k): nx.node_link_data(v) for k, v in self.window_graphs.items()
                },
                "window_stats": {
                    str(k): {name: asdict(stats) for name, stats in v.items()}
                    for k, v in self.window_stats.items()
                },
                "anomaly_propagation_chains": self.anomaly_propagation_chains,
                "distribution_thresholds": self.distribution_thresholds,
            }
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved KG to {path} (JSON format)")
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. Use .pkl or .json"
            )

    @classmethod
    def load(cls, path: str) -> "KnowledgeGraph":
        """
        Load KG from file.

        Args:
            path: Path to saved KG file

        Returns:
            Loaded KnowledgeGraph instance
        """
        path = Path(path)

        if path.suffix == ".pkl":
            # Use custom unpickler to handle module imports
            import sys
            import importlib

            # Ensure the module is loaded
            module = sys.modules.get(cls.__module__)
            if module is None:
                module = importlib.import_module(cls.__module__)

            # Create custom unpickler that uses the module's globals
            class CustomUnpickler(pickle.Unpickler):
                def find_class(self, module, name):
                    if module == "__main__" and name == "WindowStats":
                        # Map __main__ to actual module
                        return getattr(sys.modules[cls.__module__], name)
                    return super().find_class(module, name)

            with open(path, "rb") as f:
                unpickler = CustomUnpickler(f)
                kg = unpickler.load()
            print(f"✓ Loaded KG from {path}")
            return kg
        elif path.suffix == ".json":
            with open(path, "r") as f:
                data = json.load(f)

            # Reconstruct KG
            kg = cls(
                sensor_names=data["sensor_names"],
                sensor_embeddings=np.array(data["sensor_embeddings"]),
                adjacency_matrix=np.array(data["adjacency_matrix"]),
            )

            # Reconstruct window graphs
            for k, v in data["window_graphs"].items():
                kg.window_graphs[int(k)] = nx.node_link_graph(v)

            # Reconstruct window stats
            for k, v in data["window_stats"].items():
                kg.window_stats[int(k)] = {
                    name: WindowStats(**stats) for name, stats in v.items()
                }

            kg.anomaly_propagation_chains = data.get("anomaly_propagation_chains", [])
            kg.distribution_thresholds = data.get("distribution_thresholds")

            print(f"✓ Loaded KG from {path}")
            return kg
        else:
            raise ValueError(
                f"Unsupported file format: {path.suffix}. Use .pkl or .json"
            )


# ============================================================================
# Data Loading Functions
# ============================================================================


def load_and_preprocess_data(
    data_path: str, sensor_cols: List[str], window_size: int = 300
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess data for KG creation.

    Args:
        data_path: Path to data directory or preprocessed windows file
        sensor_cols: List of sensor column names
        window_size: Window size for sliding windows

    Returns:
        Tuple of (X_windows, X_windows_unnormalized) where:
        - X_windows: (num_windows, window_size, num_sensors) normalized windows
        - X_windows_unnormalized: (num_windows, window_size, num_sensors) unnormalized windows
    """
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd

    data_path = Path(data_path)

    # Check if it's a preprocessed file
    if data_path.is_file() and data_path.suffix in [".npz", ".pkl"]:
        print(f"Loading preprocessed data from {data_path}...")
        if data_path.suffix == ".npz":
            data = np.load(data_path)
            X_windows = data["X_windows"]
            X_windows_unnormalized = data.get("X_windows_unnormalized", X_windows)
        else:
            with open(data_path, "rb") as f:
                data = pickle.load(f)
            X_windows = data["X_windows"]
            X_windows_unnormalized = data.get("X_windows_unnormalized", X_windows)
        print(f"✓ Loaded {len(X_windows)} windows")
        return X_windows, X_windows_unnormalized

    # Otherwise, load from CSV files
    print(f"Loading data from {data_path}...")
    df_list = []
    for file in data_path.glob("*.csv"):
        df = pd.read_csv(file, index_col=False)
        df["drive_id"] = file.name
        df_list.append(df)

    if not df_list:
        raise ValueError(f"No CSV files found in {data_path}")

    print(f"Loaded {len(df_list)} files")
    data = pd.concat(df_list, ignore_index=True)
    print(f"Total samples: {len(data):,}")

    # Preprocessing
    print("Preprocessing data...")
    data = data.drop(
        columns=[
            "WARM_UPS_SINCE_CODES_CLEARED ()",
            "TIME_SINCE_TROUBLE_CODES_CLEARED ()",
        ],
        errors="ignore",
    )

    # Remove duplicates and fill missing timestamps
    time_col = "ENGINE_RUN_TINE ()"
    id_col = "drive_id"
    if time_col in data.columns:
        data = data.sort_values([id_col, time_col]).reset_index(drop=True)
        data = data.groupby([id_col, time_col], as_index=False).first()

    # Remove zero variance columns
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != id_col and data[col].std() == 0:
            data = data.drop(columns=[col])

    # Filter to only sensor columns that exist
    sensor_cols = [col for col in sensor_cols if col in data.columns]
    print(f"Using {len(sensor_cols)} sensor columns")

    # Build windows
    print(f"Building windows (window_size={window_size})...")
    X_list = []
    X_unnormalized_list = []

    scaler = MinMaxScaler()

    for drive_id, group in data.groupby(id_col):
        if len(group) < window_size:
            continue

        sensor_data = group[sensor_cols].values
        sensor_data_unnormalized = sensor_data.copy()

        # Normalize
        sensor_data = scaler.fit_transform(sensor_data)

        # Create sliding windows
        for t in range(len(sensor_data) - window_size + 1):
            X_list.append(sensor_data[t : t + window_size])
            X_unnormalized_list.append(sensor_data_unnormalized[t : t + window_size])

    X_windows = np.stack(X_list)
    X_windows_unnormalized = np.stack(X_unnormalized_list)

    print(f"✓ Created {len(X_windows)} windows")
    return X_windows, X_windows_unnormalized


# ============================================================================
# Main Function
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Create Knowledge Graph from trained GDN model"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to Stage 2 clean checkpoint (stage2_clean_best.pt from train_stage2_clean.py)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to data directory or preprocessed windows file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save KG (NetworkX pickle or JSON). If not provided, KG is not saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cpu/cuda). Auto-detects if not specified.",
    )
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--window_size",
        type=int,
        default=300,
        help="Window size for data loading (default: 300)",
    )
    parser.add_argument(
        "--sensor_cols",
        type=str,
        nargs="+",
        default=None,
        help="Sensor column names. If not provided, uses default 8 sensors.",
    )
    parser.add_argument(
        "--disable_global_mask",
        action="store_true",
        help="Disable global-context masking of sensor scores",
    )
    parser.add_argument(
        "--global_mask_threshold",
        type=float,
        default=0.5,
        help="Global anomaly threshold used for gating sensor scores",
    )

    args = parser.parse_args()

    # Device detection
    if args.cpu_only:
        device = "cpu"
        print("Using device: cpu (forced via --cpu_only flag)")
    elif args.device is not None:
        device = args.device
        print(f"Using device: {device} (specified via --device)")
    else:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using device: cuda (auto-detected)")
        else:
            device = "cpu"
            print("Using device: cpu (auto-detected)")

    # Default sensor columns
    if args.sensor_cols is None:
        sensor_cols = [
            "ENGINE_RPM ()",
            "VEHICLE_SPEED ()",
            "THROTTLE ()",
            "ENGINE_LOAD ()",
            "COOLANT_TEMPERATURE ()",
            "INTAKE_MANIFOLD_PRESSURE ()",
            "SHORT_TERM_FUEL_TRIM_BANK_1 ()",
            "LONG_TERM_FUEL_TRIM_BANK_1 ()",
        ]
    else:
        sensor_cols = args.sensor_cols

    # Load model
    print("\n" + "=" * 80)
    print("Step 1: Loading GDN Model")
    print("=" * 80)
    model, metadata = load_gdn_model(args.model_path, device)
    sensor_names = metadata.get("sensor_names", sensor_cols)
    window_size = metadata.get("window_size", args.window_size)

    # Extract GDN outputs
    print("\n" + "=" * 80)
    print("Step 2: Extracting GDN Outputs")
    print("=" * 80)
    sensor_embeddings = extract_sensor_embeddings(model)
    adjacency_matrix = compute_adjacency_matrix(sensor_embeddings)
    print(f"✓ Extracted sensor embeddings: shape {sensor_embeddings.shape}")
    print(f"✓ Computed adjacency matrix: shape {adjacency_matrix.shape}")

    # Load data
    print("\n" + "=" * 80)
    print("Step 3: Loading Data")
    print("=" * 80)
    X_windows, X_windows_unnormalized = load_and_preprocess_data(
        args.data_path, sensor_cols, window_size
    )

    # Run inference
    print("\n" + "=" * 80)
    print("Step 4: Running GDN Inference")
    print("=" * 80)
    gdn_predictions = predict_anomalies(
        model,
        X_windows,
        batch_size=args.batch_size,
        device=device,
        apply_global_mask=not args.disable_global_mask,
        global_mask_threshold=args.global_mask_threshold,
    )
    print(f"✓ Generated predictions for {len(gdn_predictions)} windows")

    # Build Knowledge Graph
    print("\n" + "=" * 80)
    print("Step 5: Building Knowledge Graph")
    print("=" * 80)
    kg = KnowledgeGraph(sensor_names, sensor_embeddings, adjacency_matrix)
    kg.construct(
        X_windows=X_windows,
        gdn_predictions=gdn_predictions,
        X_windows_unnormalized=X_windows_unnormalized,
    )

    # Save to file if requested
    if args.output_path:
        print("\n" + "=" * 80)
        print("Step 6: Saving Knowledge Graph")
        print("=" * 80)
        kg.save(args.output_path)

    print("\n" + "=" * 80)
    print("✓ Knowledge Graph Creation Complete!")
    print("=" * 80)
    print(f"  Nodes: {kg.number_of_nodes()}")
    print(f"  Edges: {kg.number_of_edges()}")
    print(f"  Windows: {len(kg.window_graphs)}")
    if args.output_path:
        print(f"  Saved to: {args.output_path}")
    print()


if __name__ == "__main__":
    main()
