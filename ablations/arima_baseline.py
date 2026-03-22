"""
ARIMA baseline for anomaly detection. Unsupervised: per-sensor forecast error as anomaly score.
Fits one ARIMA model per sensor on training data, applies to test windows, uses high residual as anomaly.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def _concatenate_sensor_series(windows: np.ndarray, sensor_idx: int) -> np.ndarray:
    """Extract (N, 300) -> (N*300,) for one sensor."""
    return windows[:, :, sensor_idx].flatten()


def fit_arima_models(
    train_windows: np.ndarray,
    order: Tuple[int, int, int] = (2, 0, 2),
    max_windows: Optional[int] = None,
) -> List[object]:
    """
    Fit one ARIMA model per sensor on concatenated training series.
    Returns list of fitted ARIMAResults (one per sensor).
    max_windows: if set, use only first N windows to limit fitting time.
    """
    if max_windows is not None:
        train_windows = train_windows[:max_windows]
    num_sensors = train_windows.shape[2]
    models = []
    for d in range(num_sensors):
        series = _concatenate_sensor_series(train_windows, d)
        if np.var(series) < 1e-10:
            res = ARIMA(series, order=(0, 0, 0)).fit()
        else:
            try:
                res = ARIMA(series, order=order).fit()
            except Exception:
                res = ARIMA(series, order=(1, 0, 0)).fit()
        models.append(res)
    return models


def apply_arima_to_window(
    model: object,
    window_sensor: np.ndarray,
) -> np.ndarray:
    """
    Apply fitted ARIMA to a window's sensor series, return absolute residuals.
    window_sensor: (300,) or (window_size,)
    """
    try:
        new_res = model.apply(window_sensor)
        residuals = np.abs(new_res.resid)
        return residuals
    except Exception:
        return np.zeros_like(window_sensor)


def compute_anomaly_scores(
    models: List[object],
    windows: np.ndarray,
) -> np.ndarray:
    """
    For each window, compute per-sensor anomaly score (max residual in window).
    Returns (N, num_sensors) array of scores.
    """
    num_windows = windows.shape[0]
    num_sensors = windows.shape[2]
    scores = np.zeros((num_windows, num_sensors), dtype=np.float32)

    for i in range(num_windows):
        for d in range(min(num_sensors, len(models))):
            model = models[d]
            window_sensor = windows[i, :, d]
            residuals = apply_arima_to_window(model, window_sensor)
            scores[i, d] = np.max(residuals) if len(residuals) > 0 else 0.0

    return scores


def _tune_threshold(
    scores: np.ndarray,
    y_true: np.ndarray,
    percentiles: np.ndarray,
) -> float:
    """Find percentile threshold that maximizes sensor-level F1."""
    from sklearn.metrics import f1_score
    best_f1 = 0.0
    best_thr = np.percentile(scores, 95)
    for p in percentiles:
        thr = np.percentile(scores, p)
        pred = (scores > thr).astype(np.float32)
        f1 = f1_score(y_true.flatten(), pred.flatten(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


def fit_arima(
    train_data: Dict,
    val_data: Dict,
    checkpoint_path: Path,
    order: Tuple[int, int, int] = (2, 0, 2),
    max_train_windows: Optional[int] = 300,
) -> Tuple[List[object], float]:
    """
    Fit ARIMA per sensor, tune threshold on val set.
    Saves checkpoint and returns (models, threshold).
    """
    train_windows = train_data["normalized_windows"]
    models = fit_arima_models(train_windows, order=order, max_windows=max_train_windows)

    val_scores = compute_anomaly_scores(models, val_data["normalized_windows"])
    percentiles = np.linspace(80, 99, 10)
    threshold = _tune_threshold(
        val_scores,
        val_data["sensor_labels"],
        percentiles,
    )

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(
            {"models": models, "order": order, "threshold": threshold},
            f,
        )

    return models, threshold


def load_arima_checkpoint(checkpoint_path: Path) -> Tuple[List[object], float]:
    """Load fitted ARIMA models and threshold."""
    with open(checkpoint_path, "rb") as f:
        data = pickle.load(f)
    return data["models"], data["threshold"]


def predict_arima(
    models: List[object],
    windows: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Return (N, num_sensors) binary predictions from anomaly scores."""
    scores = compute_anomaly_scores(models, windows)
    return (scores > threshold).astype(np.float32)


def run_arima(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    train: bool = True,
    dataset_path: str = "data/shared_dataset/test.npz",
) -> Dict:
    """
    Fit (if needed), run inference on test, compute metrics, save results.
    Returns result dict matching gdn_only schema.
    """
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

    if train or not checkpoint_path.exists():
        print("Fitting ARIMA baseline...")
        models, threshold = fit_arima(
            train_data=train_data,
            val_data=val_data,
            checkpoint_path=checkpoint_path,
            max_train_windows=300,
        )
    else:
        models, threshold = load_arima_checkpoint(checkpoint_path)

    sensor_labels_pred = predict_arima(
        models,
        test_data["normalized_windows"],
        threshold,
    )
    window_labels_pred = (sensor_labels_pred.sum(axis=1) > 0).astype(np.int64)

    sensor_labels_true = test_data["sensor_labels"]
    window_is_faulty_true = test_data["window_is_faulty"]

    metrics = compute_all_metrics(
        y_true_window=window_is_faulty_true,
        y_pred_window=window_labels_pred,
        y_true_sensor=sensor_labels_true,
        y_pred_sensor=sensor_labels_pred,
        sensor_names=test_data["sensor_names"],
        fault_types=test_data.get("fault_types"),
    )

    print(format_metrics_report(metrics))

    results = {
        "method": "arima_baseline",
        "dataset": dataset_path,
        "num_windows": int(len(sensor_labels_pred)),
        "metrics": metrics,
        "predictions": {
            "window_labels": window_labels_pred.tolist(),
            "sensor_labels": sensor_labels_pred.tolist(),
        },
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        import json
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {output_path}")

    return results
