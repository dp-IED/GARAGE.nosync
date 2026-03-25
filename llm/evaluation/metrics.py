"""
Evaluation metrics for comparing diagnostic methods.

Provides metrics for:
- Window-level binary fault detection (compute_window_metrics / unified "window" key):
    binary {0=normal, 1=faulty} — agrees with window_label_pred derived from sensor-indexed
    window_label (item 1 in this taxonomy).
- Sensor-indexed multiclass localization (compute_window_level_metrics / "window_sensor_class" key):
    0=no fault, 1..D=first faulty sensor 1-indexed — same scheme as legacy results/*.json.
- Sensor-level multilabel precision/recall/F1 (compute_sensor_level_metrics):
    flattened micro metrics over all (window × sensor) binary cells.
- Per-fault-type stratified metrics
- Confusion matrices
- BERTScore for reasoning quality (unified pipeline):
    precision/recall/f1 are semantic similarity scores, not classifier metrics.
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
    """
    Sensor-level precision, recall, F1 (micro over all window × sensor binary cells).
    Flattens the (N, D) arrays before computing binary P/R/F1.
    """
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


def compute_fault_type_classification_metrics(y_true_types, y_pred_types):
    """
    Multi-class precision, recall, F1 for fault type classification.
    Evaluated only on windows where ground truth is faulty (excludes 'normal' ground truth).
    Returns overall weighted/macro metrics plus per-class breakdown.
    """
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    pairs = [
        (t, p) for t, p in zip(y_true_types, y_pred_types)
        if t not in (None, "", "normal")
    ]
    if not pairs:
        return {"accuracy": 0.0, "weighted_precision": 0.0, "weighted_recall": 0.0,
                "weighted_f1": 0.0, "macro_precision": 0.0, "macro_recall": 0.0,
                "macro_f1": 0.0, "per_class": {}, "n": 0}

    y_true = [t for t, _ in pairs]
    y_pred = [p for _, p in pairs]
    classes = sorted(set(y_true))

    accuracy = accuracy_score(y_true, y_pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0, labels=classes
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=classes
    )
    p_per, r_per, f1_per, sup = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=classes
    )

    per_class = {
        cls: {
            "precision": float(p_per[i]),
            "recall": float(r_per[i]),
            "f1": float(f1_per[i]),
            "support": int(sup[i]),
        }
        for i, cls in enumerate(classes)
    }

    return {
        "accuracy": float(accuracy),
        "weighted_precision": float(p_w),
        "weighted_recall": float(r_w),
        "weighted_f1": float(f1_w),
        "macro_precision": float(p_m),
        "macro_recall": float(r_m),
        "macro_f1": float(f1_m),
        "per_class": per_class,
        "n": len(pairs),
    }


def compute_bertscore(references, hypotheses, lang="en", device=None):
    """
    BERTScore for reasoning quality (faulty windows only).

    Returns precision/recall/f1 as semantic similarity scores in [0, 1],
    not as classifier precision/recall. Higher = generated reasoning is more
    semantically similar to the reference explanation.

    Uses CUDA when available; otherwise CPU (avoids passing ``device=None``, which
    can pick backends that break on some macOS setups).

    Raises if ``bert-score`` / torch are missing or if scoring fails — LLM evals
    must not silently substitute zeros. Empty ``references`` returns zeros (no
    faulty-window pairs to score).
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"BERTScore: length mismatch (refs={len(references)}, hyps={len(hypotheses)})."
        )

    if not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        import torch
        from bert_score import score as bert_score_fn
    except ImportError as e:
        raise ImportError(
            "BERTScore requires `bert-score` (and torch). "
            "Install with: pip install bert-score  (see requirements.txt)"
        ) from e

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    refs = [str(r) if r is not None else "" for r in references]
    hyps = [str(h) if h is not None else "" for h in hypotheses]

    try:
        P, R, F = bert_score_fn(
            hyps, refs, lang=lang, verbose=False, device=device, batch_size=32
        )
        return {
            "precision": float(P.mean().item()),
            "recall": float(R.mean().item()),
            "f1": float(F.mean().item()),
        }
    except Exception as e:
        raise RuntimeError(
            f"BERTScore computation failed ({type(e).__name__}: {e}). "
            "Resolve the error before treating eval results as valid."
        ) from e


def compute_all_metrics_unified(results, sensor_cols):
    """
    Thin wrapper around compute_all_metrics for the compare_methods format.

    results: list of dicts, one per window:
      window_label_true, window_label_pred (0=normal, 1=faulty)
      sensor_labels_true, sensor_labels_pred (list[float] length D)
      fault_type_true, fault_type_pred (str)
      reasoning (str), reference_reasoning (str)

    Returned keys mirror compute_all_metrics: window_level, sensor_level,
    confusion_matrices, per_fault_type, fault_type_classification, bertscore.
    Additionally adds "window" (binary) and "sensor" (micro) keys for
    backwards-compatibility with compare_methods callers.
    """
    wt_bin = np.array([r["window_label_true"] for r in results])
    wp_bin = np.array([r["window_label_pred"] for r in results])
    st = np.array([r["sensor_labels_true"] for r in results])
    sp = np.array([r["sensor_labels_pred"] for r in results])
    ft_true = np.array([r.get("fault_type_true") or "normal" for r in results])
    ft_pred = [r.get("fault_type_pred") or "normal" for r in results]
    reasoning = [r.get("reasoning") or "" for r in results]
    ref_reasoning = [r.get("reference_reasoning") or "" for r in results]

    # Derive sensor-indexed class labels (0=no fault, 1..D=first faulty sensor)
    y_true_window_class = window_labels_sensor_indexed_from_sensor_binary(st)
    y_pred_window_class = window_labels_sensor_indexed_from_sensor_binary(
        (sp > 0.5).astype(np.float32)
    )

    metrics = compute_all_metrics(
        y_true_window=y_true_window_class,
        y_pred_window=y_pred_window_class,
        y_true_sensor=st,
        y_pred_sensor=(sp > 0.5).astype(np.float32),
        sensor_names=sensor_cols,
        fault_types=ft_true,
        fault_types_pred=ft_pred,
        reasoning=reasoning,
        reference_reasoning=ref_reasoning,
        window_is_faulty_true=wt_bin,
    )
    # Add binary window and micro sensor shims for backwards-compatibility
    metrics["window"] = compute_window_metrics(wt_bin, wp_bin)
    metrics["sensor"] = compute_sensor_metrics(st, sp)
    return metrics


# --- Legacy metrics (for existing evaluators) ---

from typing import Dict, List, Optional, Tuple, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns


def window_labels_sensor_indexed_from_sensor_binary(sensor_binary: np.ndarray) -> np.ndarray:
    """
    Per-window labels for compute_window_level_metrics: 0 = no fault, else 1 + index of first
    sensor marked faulty in each row. Matches KG/LLM eval conversion from sensor_labels.
    """
    x = np.asarray(sensor_binary)
    row_any = (x > 0).any(axis=1)
    idx = np.argmax((x > 0).astype(np.int8), axis=1)
    out = np.zeros(x.shape[0], dtype=np.int64)
    out[row_any] = idx[row_any].astype(np.int64) + 1
    return out


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

    Overall metrics (sensor_precision/recall/f1) are **micro** averages: computed by
    flattening the (N, D) arrays into a single binary vector and treating every
    (window, sensor) cell as an independent prediction. Per-sensor metrics are per-column.

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


def _normalize_fault_type_label(value: Any) -> str:
    """Strip and stringify fault-type labels for comparison (empty/None -> 'normal')."""
    if value is None:
        return "normal"
    s = str(value).strip()
    return s if s else "normal"


def compute_per_fault_type_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fault_types: np.ndarray,
    sensor_names: Optional[List[str]] = None,
    fault_types_pred: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Metrics stratified by ground-truth fault type (faulty windows only).

    For each true fault type T, reports:
    - fault_type_match_accuracy: among windows whose ground truth is T, fraction where
      the predicted fault_type string equals T (same notion as per-class recall for T
      when evaluating only faulty windows).
    - Sensor-level metrics on the same subset (multi-label over sensors).

    Args:
        y_true: (N, num_sensors) binary array - true sensor labels
        y_pred: (N, num_sensors) binary array - predicted sensor labels
        fault_types: (N,) array of fault type strings
        sensor_names: Optional list of sensor names
        fault_types_pred: Optional length-N predicted fault type strings; required for
            fault_type_match_accuracy (omit keys if None).
    """
    valid_fault_types = [ft for ft in fault_types if ft is not None and ft != '' and ft != 'normal']
    unique_fault_types = list(set(valid_fault_types)) if valid_fault_types else []

    per_fault_metrics: Dict[str, Dict[str, Any]] = {}

    for fault_type in unique_fault_types:
        mask = fault_types == fault_type
        n_win = int(mask.sum())
        if n_win == 0:
            continue

        fault_y_true = y_true[mask]
        fault_y_pred = y_pred[mask]

        true_label_norm = _normalize_fault_type_label(fault_type)
        entry: Dict[str, Any] = {
            'num_windows': n_win,
            'sensor_accuracy': float(accuracy_score(fault_y_true.flatten(), fault_y_pred.flatten())),
            'sensor_precision': float(precision_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_recall': float(recall_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
            'sensor_f1': float(f1_score(fault_y_true.flatten(), fault_y_pred.flatten(), zero_division=0)),
        }

        if fault_types_pred is not None and len(fault_types_pred) == len(fault_types):
            idx = np.where(mask)[0]
            correct = 0
            for i in idx:
                if _normalize_fault_type_label(fault_types_pred[int(i)]) == true_label_norm:
                    correct += 1
            entry['fault_type_match_accuracy'] = float(correct / n_win) if n_win else 0.0
            entry['fault_type_correct'] = int(correct)

        per_fault_metrics[fault_type] = entry

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
    fault_types: Optional[np.ndarray] = None,
    fault_types_pred: Optional[List[str]] = None,
    reasoning: Optional[List[str]] = None,
    reference_reasoning: Optional[List[str]] = None,
    window_is_faulty_true: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics.

    Args:
        y_true_window: (N,) sensor-indexed true labels (0 = no fault, 1..D = first faulty sensor)
        y_pred_window: (N,) same scheme for predictions
        y_true_sensor: (N, num_sensors) binary array - true sensor labels
        y_pred_sensor: (N, num_sensors) binary array - predicted sensor labels
        sensor_names: Optional list of sensor names
        fault_types: Optional (N,) array of ground-truth fault type strings
        fault_types_pred: Optional list of predicted fault type strings (same length as fault_types)
        reasoning: Optional (N,) list of model-generated reasoning strings
        reference_reasoning: Optional (N,) list of ground-truth reasoning strings
        window_is_faulty_true: Optional (N,) binary array marking truly faulty windows.
            If None, derived from y_true_window > 0. Used to restrict BERTScore to
            faulty windows only.

    Returns:
        Dictionary with all metrics. Includes 'bertscore' when reasoning and
        reference_reasoning are both provided.
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

    # Per-fault-type: fault-type classification rate per stratum + sensor localization on that stratum
    if fault_types is not None and len(fault_types) > 0:
        try:
            metrics['per_fault_type'] = compute_per_fault_type_metrics(
                y_true_sensor,
                y_pred_sensor,
                fault_types,
                sensor_names,
                fault_types_pred=list(fault_types_pred)
                if fault_types_pred is not None
                else None,
            )
        except Exception as e:
            import warnings
            warnings.warn(f"per_fault_type metrics computation failed and will be absent from results: {e}")

    # Fault type classification metrics (measures fault type string prediction accuracy)
    if fault_types is not None and fault_types_pred is not None:
        try:
            ft_true_list = [str(ft) if ft is not None else "normal" for ft in fault_types]
            metrics['fault_type_classification'] = compute_fault_type_classification_metrics(
                ft_true_list, list(fault_types_pred)
            )
        except Exception as e:
            import warnings
            warnings.warn(f"fault_type_classification metrics computation failed and will be absent from results: {e}")

    # BERTScore: semantic similarity between generated and reference reasoning (faulty windows only)
    if reasoning is not None and reference_reasoning is not None:
        n = int(np.asarray(y_true_window).shape[0])
        if len(reasoning) != n or len(reference_reasoning) != n:
            raise ValueError(
                f"BERTScore alignment error: len(y_true_window)={n}, "
                f"len(reasoning)={len(reasoning)}, "
                f"len(reference_reasoning)={len(reference_reasoning)}"
            )

        faulty_mask = (
            np.asarray(window_is_faulty_true) > 0
            if window_is_faulty_true is not None
            else np.asarray(y_true_window) > 0
        )
        faulty_mask = faulty_mask.astype(bool)
        n_faulty = int(np.sum(faulty_mask))
        refs = [
            reference_reasoning[i] or ""
            for i in range(n)
            if faulty_mask[i]
        ]
        hyps = [reasoning[i] or "" for i in range(n) if faulty_mask[i]]
        if n_faulty > 0 and len(refs) == 0:
            raise ValueError(
                "BERTScore: faulty_mask marks faulty windows but extracted no ref/hyp pairs"
            )
        if n_faulty == 0:
            metrics["bertscore"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        else:
            metrics["bertscore"] = compute_bertscore(refs, hyps)

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
    
    # Fault-type classification (faulty windows only; see compute_fault_type_classification_metrics)
    if 'fault_type_classification' in metrics:
        ftc = metrics['fault_type_classification']
        lines.append("Fault-Type Classification (ground-truth faulty windows only):")
        lines.append("-" * 40)
        lines.append(f"  Accuracy:        {ftc.get('accuracy', 0):.4f}")
        lines.append(f"  Weighted F1:     {ftc.get('weighted_f1', 0):.4f}")
        lines.append(f"  Macro F1:        {ftc.get('macro_f1', 0):.4f}")
        lines.append(f"  n (faulty only): {ftc.get('n', 0)}")
        lines.append("")

    # Per-fault-type metrics (stratified by true fault type)
    if 'per_fault_type' in metrics:
        lines.append("Per-Fault-Type Metrics (stratified by ground-truth fault type):")
        lines.append("-" * 40)
        for fault_type, ft_metrics in metrics['per_fault_type'].items():
            lines.append(f"  {fault_type} (N={ft_metrics['num_windows']}):")
            if 'fault_type_match_accuracy' in ft_metrics:
                lines.append(
                    f"    Fault-type match: {ft_metrics['fault_type_match_accuracy']:.4f} "
                    f"({ft_metrics.get('fault_type_correct', '?')}/{ft_metrics['num_windows']} windows)"
                )
            lines.append(f"    Sensor F1:        {ft_metrics['sensor_f1']:.4f}")
        lines.append("")

    if "bertscore" in metrics:
        bs = metrics["bertscore"]
        lines.append("BERTScore (faulty windows only; semantic similarity):")
        lines.append("-" * 40)
        lines.append(
            f"  Precision: {bs['precision']:.4f}  Recall: {bs['recall']:.4f}  F1: {bs['f1']:.4f}"
        )
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


def _bertscore_smoke_test() -> None:
    """
    Quick sanity check (no LLM): tiny BERTScore call + optional replay from
    results/gdn_kg_llm.json + test.npz when those files exist.
    Run from repo root:  python llm/evaluation/metrics.py
    """
    import json
    from pathlib import Path

    tiny = compute_bertscore(
        ["The coolant temperature sensor shows intermittent dropout."],
        ["Coolant temp dropped unexpectedly, affecting related sensors."],
    )
    assert tiny["f1"] > 0.2, f"expected non-degenerate BERTScore, got {tiny}"

    root = Path(__file__).resolve().parents[2]
    res_path = root / "results" / "gdn_kg_llm.json"
    npz_path = root / "data" / "shared_dataset" / "test.npz"
    if not (res_path.is_file() and npz_path.is_file()):
        print("BERTScore smoke: tiny test OK", tiny)
        return

    with open(res_path) as f:
        d = json.load(f)
    si = d.get("sample_indices")
    if not si:
        print("BERTScore smoke: tiny test OK", tiny)
        return

    import numpy as np

    si = np.array(si, dtype=np.int64)
    data = np.load(npz_path, allow_pickle=True)
    sl = data["sensor_labels"][si]
    rr = data["reference_reasoning"][si]
    reasoning = d["predictions"]["reasoning"]
    y_true = np.zeros(len(si), dtype=np.int64)
    for i in range(len(si)):
        fi = np.where(sl[i] > 0)[0]
        y_true[i] = int(fi[0] + 1) if len(fi) else 0
    faulty = y_true > 0
    refs = [str(rr[i]) for i in range(len(si)) if faulty[i]]
    hyps = [str(reasoning[i]) for i in range(len(si)) if faulty[i]]
    replay = compute_bertscore(refs, hyps)
    assert replay["f1"] > 0.2, f"replay BERTScore degenerate: {replay}"
    print("BERTScore smoke: tiny OK", tiny)
    print(f"BERTScore smoke: replay {len(refs)} faulty windows F1={replay['f1']:.4f}")


if __name__ == "__main__":
    _bertscore_smoke_test()
