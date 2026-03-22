"""
LSTM baseline for anomaly detection. Supervised classification matching GDN stage 2:
predicts per-sensor binary labels from multivariate time series windows.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LSTMBaseline(nn.Module):
    """
    LSTM-based per-sensor anomaly classifier.
    Input: (B, window_size, num_sensors)
    Output: (B, num_sensors) logits
    """

    def __init__(
        self,
        num_sensors: int = 8,
        window_size: int = 300,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.classifier(h)


def train_lstm(
    train_data: Dict,
    val_data: Dict,
    checkpoint_path: Path,
    epochs: int = 75,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cpu",
    seed: int = 42,
) -> LSTMBaseline:
    """Train LSTM baseline and save best checkpoint based on val F1."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train = torch.tensor(train_data["normalized_windows"], dtype=torch.float32)
    y_train = torch.tensor(train_data["sensor_labels"], dtype=torch.float32)
    X_val = torch.tensor(val_data["normalized_windows"], dtype=torch.float32)
    y_val = torch.tensor(val_data["sensor_labels"], dtype=torch.float32)

    num_sensors = X_train.shape[2]
    window_size = X_train.shape[1]

    model = LSTMBaseline(
        num_sensors=num_sensors,
        window_size=window_size,
        hidden_dim=32,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_f1 = 0.0
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            logits_val = model(X_val.to(device))
            probs_val = torch.sigmoid(logits_val).cpu().numpy()
        val_f1 = _sensor_f1(y_val.numpy(), (probs_val > 0.5).astype(np.float32))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_sensors": num_sensors,
                    "window_size": window_size,
                    "hidden_dim": 32,
                    "sensor_threshold": 0.5,
                },
                checkpoint_path,
            )

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1}/{epochs}  train_loss={train_loss / len(train_loader):.4f}  val_f1={val_f1:.4f}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def _sensor_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Binary F1 on flattened sensor labels."""
    from sklearn.metrics import f1_score
    return float(f1_score(y_true.flatten(), y_pred.flatten(), zero_division=0))


def _tune_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    thresholds: np.ndarray,
) -> float:
    """Find threshold that maximizes sensor-level F1."""
    from sklearn.metrics import f1_score
    best_f1 = 0.0
    best_thr = 0.5
    for thr in thresholds:
        pred = (probs > thr).astype(np.float32)
        f1 = f1_score(y_true.flatten(), pred.flatten(), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


def predict_lstm(
    model: LSTMBaseline,
    X: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """Return (N, num_sensors) binary predictions."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    return (probs > threshold).astype(np.float32)


def load_lstm_checkpoint(checkpoint_path: Path, device: str = "cpu") -> Tuple[LSTMBaseline, float]:
    """Load trained LSTM and return (model, sensor_threshold)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LSTMBaseline(
        num_sensors=ckpt["num_sensors"],
        window_size=ckpt["window_size"],
        hidden_dim=ckpt.get("hidden_dim", 32),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    threshold = float(ckpt.get("sensor_threshold", 0.5))
    return model, threshold


def run_lstm(
    train_data: Dict,
    val_data: Dict,
    test_data: Dict,
    checkpoint_path: Path,
    output_path: Optional[Path] = None,
    train: bool = True,
    device: str = "cpu",
    dataset_path: str = "data/shared_dataset/test.npz",
    epochs: int = 75,
) -> Dict:
    """
    Train (if needed), run inference on test, compute metrics, save results.
    Returns result dict matching gdn_only schema.
    """
    import sys
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from llm.evaluation.metrics import compute_all_metrics, format_metrics_report

    if train or not checkpoint_path.exists():
        print("Training LSTM baseline...")
        model = train_lstm(
            train_data=train_data,
            val_data=val_data,
            checkpoint_path=checkpoint_path,
            device=device,
            epochs=epochs,
        )
        probs_val = _get_probs_for_threshold_tuning(model, val_data["normalized_windows"], device)
        thresholds = np.linspace(0.2, 0.8, 13)
        threshold = _tune_threshold(
            probs_val,
            val_data["sensor_labels"],
            thresholds,
        )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ckpt["sensor_threshold"] = threshold
        torch.save(ckpt, checkpoint_path)
    else:
        model, threshold = load_lstm_checkpoint(checkpoint_path, device)

    sensor_labels_pred = predict_lstm(
        model,
        test_data["normalized_windows"],
        threshold=threshold,
        device=device,
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
        "method": "lstm_baseline",
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


def _get_probs_for_threshold_tuning(model: LSTMBaseline, X: np.ndarray, device: str) -> np.ndarray:
    """Get raw probabilities (before threshold) for tuning."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(X_t), 32):
            batch = X_t[i : i + 32].to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)
