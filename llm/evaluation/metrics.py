"""
Evaluation metrics for comparing diagnostic methods.

Provides metrics for:
- Window-level accuracy
- Sensor-level precision/recall/F1
- Per-fault-type metrics
- Confusion matrices
- BERTScore for reasoning quality (unified pipeline)
"""

import numpy as np

# --- Unified pipeline metrics (for run_eval / compare_methods) ---


def compute_window_metrics(y_true, y_pred):
    """Binary window-level precision, recall, F1, accuracy."""
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "accuracy": float(acc)}


def compute_sensor_metrics(y_true, y_pred):
    """Sensor-level precision, recall, F1 (flattened binary)."""
    from sklearn.metrics import precision_recall_fscore_support

    p, r, f1, _ = precision_recall_fscore_support(
        y_true.flatten(), y_pred.flatten(),
        average="binary", zero_division=0
    )
    return {"precision": float(p), "recall": float(r), "f1": float(f1)}


def compute_fault_type_accuracy(y_true_types, y_pred_types):
    """Accuracy on fault type prediction, only for windows where ground truth is faulty."""
    pairs = [(t, p) for t, p in zip(y_true_types, y_pred_types) if t not in (None, "", "normal")]
    if not pairs:
        return {"accuracy": 0.0, "n": 0}
    correct = sum(1 for t, p in pairs if t == p)
    return {"accuracy": correct / len(pairs), "n": len(pairs)}


def compute_bertscore(references, hypotheses, lang="en"):
    """BERTScore for reasoning quality. Only on faulty windows."""
    if not references or not hypotheses:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    try:
        from bert_score import score as bert_score

        P, R, F = bert_score(hypotheses, references, lang=lang, verbose=False)
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F.mean().item()),
        }
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def compute_all_metrics_unified(results, sensor_cols):
    """
    Compute all metrics from per-window results (unified format for compare_methods).

    results: list of dicts, one per window:
      window_label_true, window_label_pred (0=normal, 1=faulty)
      sensor_labels_true, sensor_labels_pred (list[int] length D)
      fault_type_true, fault_type_pred (str)
      reasoning (str), reference_reasoning (str)
    """
    wt = np.array([r["window_label_true"] for r in results])
    wp = np.array([r["window_label_pred"] for r in results])
    st = np.array([r["sensor_labels_true"] for r in results])
    sp = np.array([r["sensor_labels_pred"] for r in results])
    ft_true = [r.get("fault_type_true") or "normal" for r in results]
    ft_pred = [r.get("fault_type_pred") or "normal" for r in results]

    faulty_idx = [i for i, r in enumerate(results) if r["window_label_true"] == 1]
    refs = [results[i].get("reference_reasoning") or "" for i in faulty_idx]
    hyps = [results[i].get("reasoning") or "" for i in faulty_idx]

    return {
        "window": compute_window_metrics(wt, wp),
        "sensor": compute_sensor_metrics(st, sp),
        "fault_type": compute_fault_type_accuracy(ft_true, ft_pred),
        "bertscore": compute_bertscore(refs, hyps),
    }


# --- Legacy metrics (for existing evaluators) ---

from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def compute_window_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute window-level metrics (multi-class classification: 0-8).
    
    Window labels are sensor-indexed:
    - 0 = no fault
    - 1-8 = anomalous sensor index (1-indexed: sensor 0 -> label 1, sensor 7 -> label 8)
    
    Args:
        y_true: (N,) array - true window labels (0-8)
        y_pred: (N,) array - predicted window labels (0-8)
        sensor_names: Optional list of sensor names for per-class metrics
        
    Returns:
        Dictionary with accuracy, precision, recall, F1 (weighted and macro),
        and optionally per-class metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted metrics (accounts for class imbalance)
    precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Macro metrics (unweighted average across classes)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics = {
        'window_accuracy': float(accuracy),
        'window_precision_weighted': float(precision_weighted),
        'window_recall_weighted': float(recall_weighted),
        'window_f1_weighted': float(f1_weighted),
        'window_precision_macro': float(precision_macro),
        'window_recall_macro': float(recall_macro),
        'window_f1_macro': float(f1_macro),
        # Keep backward compatibility aliases (using weighted as default)
        'window_precision': float(precision_weighted),
        'window_recall': float(recall_weighted),
        'window_f1': float(f1_weighted)
    }
    
    # Per-class metrics (for each sensor class 0-8)
    if sensor_names is not None:
        num_classes = len(sensor_names) + 1  # 0 (no fault) + sensors
        per_class_metrics = {}
        
        # Class 0: No fault
        class_0_mask_true = (y_true == 0)
        class_0_mask_pred = (y_pred == 0)
        per_class_metrics['no_fault'] = {
            'precision': float(precision_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'recall': float(recall_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'f1': float(f1_score(class_0_mask_true, class_0_mask_pred, zero_division=0)),
            'support': int(class_0_mask_true.sum())
        }
        
        # Classes 1-8: Each sensor
        for sensor_idx in range(len(sensor_names)):
            class_label = sensor_idx + 1  # 1-indexed
            class_mask_true = (y_true == class_label)
            class_mask_pred = (y_pred == class_label)
            
            per_class_metrics[sensor_names[sensor_idx]] = {
                'precision': float(precision_score(class_mask_true, class_mask_pred, zero_division=0)),
                'recall': float(recall_score(class_mask_true, class_mask_pred, zero_division=0)),
                'f1': float(f1_score(class_mask_true, class_mask_pred, zero_division=0)),
                'support': int(class_mask_true.sum())
            }
        
        metrics['per_class'] = per_class_metrics
    
    return metrics


def compute_sensor_level_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, any]:
    """
    Compute sensor-level metrics (multi-label classification).
    
    Args:
        y_true: (N, num_sensors) binary array - true sensor labels
        y_pred: (N, num_sensors) binary array - predicted sensor labels
        sensor_names: Optional list of sensor names for per-sensor metrics
        
    Returns:
        Dictionary with overall and per-sensor metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Flatten for overall metrics
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, zero_division=0)
    
    metrics = {
        'sensor_accuracy': float(accuracy),
        'sensor_precision': float(precision),
        'sensor_recall': float(recall),
        'sensor_f1': float(f1)
    }
    
    # Per-sensor metrics
    if sensor_names:
        per_sensor = {}
        num_sensors = len(sensor_names)
        
        for i, sensor_name in enumerate(sensor_names):
            if i < y_true.shape[1]:
                sensor_true = y_true[:, i]
                sensor_pred = y_pred[:, i]
                
                per_sensor[sensor_name] = {
                    'accuracy': float(accuracy_score(sensor_true, sensor_pred)),
                    'precision': float(precision_score(sensor_true, sensor_pred, zero_division=0)),
                    'recall': float(recall_score(sensor_true, sensor_pred, zero_division=0)),
                    'f1': float(f1_score(sensor_true, sensor_pred, zero_division=0)),
                    'true_positives': int(np.sum((sensor_true == 1) & (sensor_pred == 1))),
                    'false_positives': int(np.sum((sensor_true == 0) & (sensor_pred == 1))),
                    'false_negatives': int(np.sum((sensor_true == 1) & (sensor_pred == 0))),
                    'true_negatives': int(np.sum((sensor_true == 0) & (sensor_pred == 0)))
                }
        
        metrics['per_sensor'] = per_sensor
    
    return metrics


def compute_per_fault_type_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_types: np.ndarray,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, Dict]:
    """
    Compute metrics per fault type.
    
    Args:
        y_true: (N, num_sensors) binary array - true sensor labels
        y_pred: (N, num_sensors) binary array - predicted sensor labels
        fault_types: (N,) array of fault type strings
        sensor_names: Optional list of sensor names
        
    Returns:
        Dictionary with metrics per fault type
    """
    # Filter out None and empty strings before getting unique values
    # (numpy.unique doesn't work well with None in object arrays)
    valid_fault_types = [ft for ft in fault_types if ft is not None and ft != '']
    unique_fault_types = list(set(valid_fault_types)) if valid_fault_types else []
    
    per_fault_metrics = {}
    
    for fault_type in unique_fault_types:
        # Find windows with this fault type
        mask = fault_types == fault_type
        if mask.sum() == 0:
            continue
        
        fault_y_true = y_true[mask]
        fault_y_pred = y_pred[mask]
        
        # Window-level metrics for this fault type
        window_true = (fault_y_true.sum(axis=1) > 0).astype(int)
        window_pred = (fault_y_pred.sum(axis=1) > 0).astype(int)
        
        per_fault_metrics[fault_type] = {
            'num_windows': int(mask.sum()),
            'window_accuracy': float(accuracy_score(window_true, window_pred)),
            'window_precision': float(precision_score(window_true, window_pred, zero_division=0)),
            'window_recall': float(recall_score(window_true, window_pred, zero_division=0)),
            'window_f1': float(f1_score(window_true, window_pred, zero_division=0)),
            'sensor_accuracy': float(accuracy_score(fault_y_true.flatten(), fault_y_pred.flatten())),
            'sensor_precision': float(precision_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_recall': float(recall_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_f1': float(f1_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0))
        }
    
    return per_fault_metrics


def compute_confusion_matrices(
    y_true_sensor: np.ndarray,
    y_pred_sensor: np.ndarray,
    y_true_window: Optional[np.ndarray] = None,
    y_pred_window: Optional[np.ndarray] = None,
    sensor_names: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Compute confusion matrices for window-level and sensor-level predictions.
    
    Args:
        y_true_sensor: (N, num_sensors) binary array - true sensor labels
        y_pred_sensor: (N, num_sensors) binary array - predicted sensor labels
        y_true_window: (N,) array - true window labels (0-8, sensor-indexed)
        y_pred_window: (N,) array - predicted window labels (0-8, sensor-indexed)
        sensor_names: Optional list of sensor names
        
    Returns:
        Dictionary with confusion matrices
    """
    y_true_sensor = y_true_sensor.astype(int)
    y_pred_sensor = y_pred_sensor.astype(int)
    
    # Window-level confusion matrix (multi-class: 0-8)
    if y_true_window is not None and y_pred_window is not None:
        y_true_window = y_true_window.astype(int)
        y_pred_window = y_pred_window.astype(int)
        # Get unique labels that actually exist in TRUE labels (sklearn requires at least one label in y_true)
        # Convert to Python int to avoid type mismatch issues
        unique_true_labels = sorted([int(l) for l in set(y_true_window)])
        # Filter to only include valid labels (0-8) that exist in true labels
        all_possible_labels = list(range(9))
        labels_to_use = [l for l in all_possible_labels if l in unique_true_labels]
        
        if len(labels_to_use) > 0:
            # Multi-class confusion matrix (only for labels that exist in y_true)
            # Ensure labels are Python ints, not numpy types
            labels_to_use = [int(l) for l in labels_to_use]
            window_cm = confusion_matrix(y_true_window, y_pred_window, labels=labels_to_use)
        else:
            # Fallback: use all labels that exist in y_true (no filtering to 0-8)
            # This handles edge cases where labels might be outside 0-8 range
            if len(unique_true_labels) > 0:
                # Convert to Python ints
                unique_true_labels = [int(l) for l in unique_true_labels]
                window_cm = confusion_matrix(y_true_window, y_pred_window, labels=unique_true_labels)
            else:
                # Last resort: no labels specified (sklearn will infer)
                window_cm = confusion_matrix(y_true_window, y_pred_window)
    else:
        # Fallback: binary window-level (for backward compatibility)
        window_true = (y_true_sensor.sum(axis=1) > 0).astype(int)
        window_pred = (y_pred_sensor.sum(axis=1) > 0).astype(int)
        window_cm = confusion_matrix(window_true, window_pred)
    
    # Sensor-level confusion matrix (flattened)
    sensor_true_flat = y_true_sensor.flatten()
    sensor_pred_flat = y_pred_sensor.flatten()
    sensor_cm = confusion_matrix(sensor_true_flat, sensor_pred_flat)
    
    matrices = {
        'window_confusion_matrix': window_cm.tolist(),
        'sensor_confusion_matrix': sensor_cm.tolist()
    }
    
    # Per-sensor confusion matrices
    if sensor_names:
        per_sensor_cm = {}
        for i, sensor_name in enumerate(sensor_names):
            if i < y_true_sensor.shape[1]:
                sensor_true = y_true_sensor[:, i]
                sensor_pred = y_pred_sensor[:, i]
                per_sensor_cm[sensor_name] = confusion_matrix(sensor_true, sensor_pred).tolist()
        
        matrices['per_sensor_confusion_matrices'] = per_sensor_cm
    
    return matrices


def compute_all_metrics(
    y_true_window: np.ndarray,
    y_pred_window: np.ndarray,
    y_true_sensor: np.ndarray,
    y_pred_sensor: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    fault_types: Optional[np.ndarray] = None
) -> Dict[str, any]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true_window: (N,) binary array - true window labels
        y_pred_window: (N,) binary array - predicted window labels
        y_true_sensor: (N, num_sensors) binary array - true sensor labels
        y_pred_sensor: (N, num_sensors) binary array - predicted sensor labels
        sensor_names: Optional list of sensor names
        fault_types: Optional (N,) array of fault type strings
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Window-level metrics (multi-class: 0-8)
    metrics['window_level'] = compute_window_level_metrics(
        y_true_window, y_pred_window, sensor_names
    )
    
    # Sensor-level metrics
    metrics['sensor_level'] = compute_sensor_level_metrics(
        y_true_sensor, y_pred_sensor, sensor_names
    )
    
    # Confusion matrices
    metrics['confusion_matrices'] = compute_confusion_matrices(
        y_true_sensor, y_pred_sensor, y_true_window, y_pred_window, sensor_names
    )
    
    # Per-fault-type metrics
    if fault_types is not None and len(fault_types) > 0:
        try:
            metrics['per_fault_type'] = compute_per_fault_type_metrics(
                y_true_sensor, y_pred_sensor, fault_types, sensor_names
            )
        except Exception as e:
            # If fault type metrics fail, continue without them
            pass
    
    return metrics


def format_metrics_report(metrics: Dict[str, any]) -> str:
    """
    Format metrics as a human-readable report string.
    
    Args:
        metrics: Dictionary from compute_all_metrics
        
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("="*80)
    lines.append("EVALUATION METRICS REPORT")
    lines.append("="*80)
    lines.append("")
    
    # Window-level metrics
    if 'window_level' in metrics:
        lines.append("Window-Level Metrics (Sensor-Indexed: 0-8):")
        lines.append("-" * 40)
        wl = metrics['window_level']
        lines.append(f"  Accuracy:  {wl['window_accuracy']:.4f}")
        lines.append(f"  Precision (weighted): {wl['window_precision_weighted']:.4f}")
        lines.append(f"  Recall (weighted):    {wl['window_recall_weighted']:.4f}")
        lines.append(f"  F1 Score (weighted):  {wl['window_f1_weighted']:.4f}")
        lines.append(f"  Precision (macro): {wl['window_precision_macro']:.4f}")
        lines.append(f"  Recall (macro):    {wl['window_recall_macro']:.4f}")
        lines.append(f"  F1 Score (macro):  {wl['window_f1_macro']:.4f}")
        
        # Per-class metrics if available
        if 'per_class' in wl:
            lines.append("\n  Per-Class Metrics:")
            for class_name, class_metrics in wl['per_class'].items():
                if class_metrics['support'] > 0:  # Only show classes with support
                    lines.append(f"    {class_name}:")
                    lines.append(f"      Precision: {class_metrics['precision']:.4f}")
                    lines.append(f"      Recall:    {class_metrics['recall']:.4f}")
                    lines.append(f"      F1:        {class_metrics['f1']:.4f}")
                    lines.append(f"      Support:   {class_metrics['support']}")
        lines.append("")
    
    # Sensor-level metrics
    if 'sensor_level' in metrics:
        lines.append("Sensor-Level Metrics:")
        lines.append("-" * 40)
        sl = metrics['sensor_level']
        lines.append(f"  Accuracy:  {sl['sensor_accuracy']:.4f}")
        lines.append(f"  Precision: {sl['sensor_precision']:.4f}")
        lines.append(f"  Recall:    {sl['sensor_recall']:.4f}")
        lines.append(f"  F1 Score:  {sl['sensor_f1']:.4f}")
        lines.append("")
        
        # Per-sensor metrics
        if 'per_sensor' in sl:
            lines.append("Per-Sensor Metrics:")
            lines.append("-" * 40)
            for sensor_name, sensor_metrics in sl['per_sensor'].items():
                lines.append(f"  {sensor_name}:")
                lines.append(f"    Accuracy:  {sensor_metrics['accuracy']:.4f}")
                lines.append(f"    Precision: {sensor_metrics['precision']:.4f}")
                lines.append(f"    Recall:    {sensor_metrics['recall']:.4f}")
                lines.append(f"    F1 Score:  {sensor_metrics['f1']:.4f}")
                lines.append(f"    TP: {sensor_metrics['true_positives']}, "
                           f"FP: {sensor_metrics['false_positives']}, "
                           f"FN: {sensor_metrics['false_negatives']}, "
                           f"TN: {sensor_metrics['true_negatives']}")
            lines.append("")
    
    # Per-fault-type metrics
    if 'per_fault_type' in metrics:
        lines.append("Per-Fault-Type Metrics:")
        lines.append("-" * 40)
        for fault_type, ft_metrics in metrics['per_fault_type'].items():
            lines.append(f"  {fault_type} (N={ft_metrics['num_windows']}):")
            lines.append(f"    Window F1: {ft_metrics['window_f1']:.4f}")
            lines.append(f"    Sensor F1: {ft_metrics['sensor_f1']:.4f}")
        lines.append("")
    
    lines.append("="*80)
    
    return "\n".join(lines)


def compute_embedding_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    distances_to_normal: np.ndarray,
    distances_to_anomalous: np.ndarray
) -> Dict[str, float]:
    """
    Compute embedding-based metrics.
    
    Args:
        y_true: (N,) binary array - true labels (0=normal, 1=anomalous)
        y_pred: (N,) binary array - predicted labels (0=normal, 1=anomalous)
        distances_to_normal: (N,) array - distances to normal center
        distances_to_anomalous: (N,) array - distances to anomalous center
    
    Returns:
        Dictionary with embedding metrics
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Distance separation: mean(dist_anomalous) - mean(dist_normal)
    normal_mask = y_true == 0
    anomalous_mask = y_true == 1
    
    if normal_mask.any() and anomalous_mask.any():
        mean_dist_normal = float(np.mean(distances_to_normal[normal_mask]))
        mean_dist_anomalous = float(np.mean(distances_to_anomalous[anomalous_mask]))
        distance_separation = mean_dist_anomalous - mean_dist_normal
    else:
        distance_separation = 0.0
        mean_dist_normal = 0.0
        mean_dist_anomalous = 0.0
    
    # Distance AUC: ROC AUC using dist_to_normal as score
    # Higher distance = more anomalous, so use negative distance for AUC
    try:
        if len(np.unique(y_true)) > 1:  # Need both classes
            distance_auc = float(roc_auc_score(y_true, -distances_to_normal))
        else:
            distance_auc = 0.0
    except Exception:
        distance_auc = 0.0
    
    # Confidence calibration: correlation between confidence and correctness
    # Confidence = 1 / (1 + exp(dist_normal - dist_anomalous))
    # Higher confidence when dist_normal < dist_anomalous (closer to normal)
    confidence_scores = 1.0 / (1.0 + np.exp(distances_to_normal - distances_to_anomalous))
    correctness = (y_true == y_pred).astype(float)
    
    try:
        confidence_calibration = float(np.corrcoef(confidence_scores, correctness)[0, 1])
        if np.isnan(confidence_calibration):
            confidence_calibration = 0.0
    except Exception:
        confidence_calibration = 0.0
    
    return {
        'distance_separation': float(distance_separation),
        'mean_dist_normal': float(mean_dist_normal),
        'mean_dist_anomalous': float(mean_dist_anomalous),
        'distance_auc': float(distance_auc),
        'confidence_calibration': float(confidence_calibration)
    }


def analyze_embedding_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    embeddings: np.ndarray,
    centers: np.ndarray
) -> Dict[str, Any]:
    """
    Analyze embedding-space characteristics of prediction errors.
    
    Args:
        y_true: (N,) binary array - true labels
        y_pred: (N,) binary array - predicted labels
        embeddings: (N, hidden_dim) array - window embeddings
        centers: (2, hidden_dim) array - class centers [normal, anomalous]
    
    Returns:
        Dictionary with error analysis
    """
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    
    # Find errors
    false_positives = (y_true == 0) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    normal_center = centers[0]
    anomalous_center = centers[1]
    
    # Compute distances for errors
    fp_distances_normal = []
    fp_distances_anomalous = []
    fn_distances_normal = []
    fn_distances_anomalous = []
    
    if false_positives.any():
        fp_embeddings = embeddings[false_positives]
        fp_distances_normal = [
            float(np.linalg.norm(emb - normal_center)) for emb in fp_embeddings
        ]
        fp_distances_anomalous = [
            float(np.linalg.norm(emb - anomalous_center)) for emb in fp_embeddings
        ]
    
    if false_negatives.any():
        fn_embeddings = embeddings[false_negatives]
        fn_distances_normal = [
            float(np.linalg.norm(emb - normal_center)) for emb in fn_embeddings
        ]
        fn_distances_anomalous = [
            float(np.linalg.norm(emb - anomalous_center)) for emb in fn_embeddings
        ]
    
    # Determine which center errors are closer to
    fp_closer_to_normal = sum(1 for d_n, d_a in zip(fp_distances_normal, fp_distances_anomalous) if d_n < d_a)
    fp_closer_to_anomalous = len(fp_distances_normal) - fp_closer_to_normal
    
    fn_closer_to_normal = sum(1 for d_n, d_a in zip(fn_distances_normal, fn_distances_anomalous) if d_n < d_a)
    fn_closer_to_anomalous = len(fn_distances_normal) - fn_closer_to_normal
    
    return {
        'num_false_positives': int(false_positives.sum()),
        'num_false_negatives': int(false_negatives.sum()),
        'fp_mean_dist_normal': float(np.mean(fp_distances_normal)) if fp_distances_normal else 0.0,
        'fp_mean_dist_anomalous': float(np.mean(fp_distances_anomalous)) if fp_distances_anomalous else 0.0,
        'fn_mean_dist_normal': float(np.mean(fn_distances_normal)) if fn_distances_normal else 0.0,
        'fn_mean_dist_anomalous': float(np.mean(fn_distances_anomalous)) if fn_distances_anomalous else 0.0,
        'fp_closer_to_normal': int(fp_closer_to_normal),
        'fp_closer_to_anomalous': int(fp_closer_to_anomalous),
        'fn_closer_to_normal': int(fn_closer_to_normal),
        'fn_closer_to_anomalous': int(fn_closer_to_anomalous),
        'error_analysis': {
            'false_positives': {
                'closer_to_normal': fp_closer_to_normal > fp_closer_to_anomalous,
                'mean_dist_normal': float(np.mean(fp_distances_normal)) if fp_distances_normal else 0.0,
                'mean_dist_anomalous': float(np.mean(fp_distances_anomalous)) if fp_distances_anomalous else 0.0
            },
            'false_negatives': {
                'closer_to_normal': fn_closer_to_normal > fn_closer_to_anomalous,
                'mean_dist_normal': float(np.mean(fn_distances_normal)) if fn_distances_normal else 0.0,
                'mean_dist_anomalous': float(np.mean(fn_distances_anomalous)) if fn_distances_anomalous else 0.0
            }
        }
    }


def plot_distance_distributions(
    distances_normal_class: np.ndarray,
    distances_anomalous_class: np.ndarray,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot histogram of distance distributions for each true class.
    
    Args:
        distances_normal_class: (N_normal,) array - distances to normal center for normal windows
        distances_anomalous_class: (N_anomalous,) array - distances to anomalous center for anomalous windows
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histograms
    ax.hist(
        distances_normal_class,
        bins=30,
        alpha=0.6,
        color='blue',
        label=f'Normal Windows (N={len(distances_normal_class)})',
        edgecolor='darkblue'
    )
    
    ax.hist(
        distances_anomalous_class,
        bins=30,
        alpha=0.6,
        color='red',
        label=f'Anomalous Windows (N={len(distances_anomalous_class)})',
        edgecolor='darkred'
    )
    
    # Add vertical lines for mean distances
    mean_normal = np.mean(distances_normal_class) if len(distances_normal_class) > 0 else 0.0
    mean_anomalous = np.mean(distances_anomalous_class) if len(distances_anomalous_class) > 0 else 0.0
    
    ax.axvline(mean_normal, color='blue', linestyle='--', linewidth=2, label=f'Mean Normal: {mean_normal:.3f}')
    ax.axvline(mean_anomalous, color='red', linestyle='--', linewidth=2, label=f'Mean Anomalous: {mean_anomalous:.3f}')
    
    ax.set_xlabel('Distance to Class Center', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Distances to Class Centers', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved distance distribution plot to {save_path}")
    
    return fig
