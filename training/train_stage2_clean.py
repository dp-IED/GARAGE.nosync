#!/usr/bin/env python3
"""
Stage 2 (Clean): Sensor anomaly fine-tuning on frozen Stage 1 representation.

This script intentionally removes the unstable components from the previous Stage 2:
- no adaptive thresholding inside the training objective,
- no mandatory hard-mining in default mode,
- optional periodic calibration pass for threshold persistence.
"""

import argparse
import os
import random
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    WeightedRandomSampler,
    SubsetRandomSampler,
)
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

# Shared project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.gdn_model import GDN

# Reuse Stage 1 helpers to keep preprocessing and windowing identical
from training.train_stage1 import (
    DATA_PATH,
    SENSOR_COLS,
    WINDOW_SIZE,
)


def build_clean_windows(
    df,
    sensor_cols,
    id_col,
    time_col,
    window_size,
    horizon=1,
    scaler=None,
):
    """Build windows from clean data only. Returns normalized windows."""
    df = df.copy().sort_values([id_col, time_col])
    df_sensors = df[[id_col, time_col] + list(sensor_cols)].copy()

    if scaler is None:
        scaler = MinMaxScaler()
        df_sensors[sensor_cols] = scaler.fit_transform(df_sensors[sensor_cols])
    else:
        df_sensors[sensor_cols] = scaler.transform(df_sensors[sensor_cols])

    X_list, y_list = [], []
    for _, group in df_sensors.groupby(id_col):
        values = group[sensor_cols].values
        T_, _ = values.shape
        if T_ <= window_size + horizon:
            continue

        for t in range(T_ - window_size - horizon + 1):
            X_list.append(values[t : t + window_size])
            y_list.append(values[t + window_size + horizon - 1])

    X = torch.tensor(np.stack(X_list), dtype=torch.float32)
    y = torch.tensor(np.stack(y_list), dtype=torch.float32)
    return X, y, scaler


torch.set_default_dtype(torch.float32)


# ============================================================================
# Constants
# ============================================================================
NUM_EPOCHS = 40
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EMBED_DIM = 16
TOP_K = 7
HIDDEN_DIM = 32

# default objective knobs
LAMBDA_SENSOR = 1.0
LAMBDA_WINDOW = 0.1
SENSOR_POS_WEIGHT_SCALE = 0.6
SENSOR_POS_WEIGHT_SCALE_MIN = 0.3
SENSOR_POS_WEIGHT_SCALE_DECAY_EPOCHS = 20
SENSOR_POS_WEIGHT_CAP = 12.0
WINDOW_SCORE_WEIGHT = 0.25  # Lower so global branch can win and receive gradients (was 0.5, win_branch_global_pct was 0%)


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


def _sensor_weight_scale(
    epoch: int,
    num_epochs: int,
    scale_init: float,
    scale_min: float,
    decay_epochs: int,
    use_schedule: bool = True,
) -> float:
    if not use_schedule:
        return scale_init
    if decay_epochs <= 0 or num_epochs <= 1:
        return max(scale_init, scale_min)
    if epoch >= num_epochs - 1:
        return scale_min
    progress = min(float(epoch), float(decay_epochs)) / float(decay_epochs)
    return scale_init - (scale_init - scale_min) * min(progress, 1.0)


def _focal_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    p = torch.sigmoid(logits)
    p_t = p * targets + (1 - p) * (1 - targets)
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    focal_weight = (1 - p_t).clamp(min=1e-4) ** gamma
    loss = focal_weight * bce
    return loss.mean() if reduction == "mean" else loss


def _best_f1_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    if scores.numel() == 0:
        return 0.5

    scores_np = scores.detach().cpu().flatten().numpy()
    labels_np = labels.detach().cpu().flatten().numpy().astype(np.int64)
    if scores_np.size == 0 or np.unique(labels_np).size <= 1:
        return 0.5

    precision, recall, thresholds = precision_recall_curve(labels_np, scores_np)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = int(np.argmax(f1_scores[:-1])) if len(f1_scores) > 1 else 0
    return float(np.clip(thresholds[min(best_idx, len(thresholds) - 1)], 0.0, 1.0))


def compute_per_sensor_thresholds(
    sensor_probs: torch.Tensor, sensor_labels: torch.Tensor
) -> np.ndarray:
    """
    Compute one threshold per sensor via PR-curve F1 optimization.
    sensor_probs: (N, num_sensors), sensor_labels: (N, num_sensors)
    Returns: (num_sensors,) array
    """
    thresholds = []
    for i in range(sensor_probs.shape[1]):
        col_labels = sensor_labels[:, i].detach().cpu().numpy().astype(np.int64)
        col_scores = sensor_probs[:, i].detach().cpu().numpy()
        if (
            col_labels.sum() < 5
        ):  # too sparse for reliable calibration (e.g. THROTTLE support=1)
            thresholds.append(0.5)
            continue
        precision, recall, thresh = precision_recall_curve(col_labels, col_scores)
        f1 = np.nan_to_num(
            2 * precision * recall / (precision + recall + 1e-8), nan=0.0
        )
        best_idx = np.argmax(f1[:-1]) if len(f1) > 1 else 0
        thresholds.append(
            float(np.clip(thresh[min(best_idx, len(thresh) - 1)], 0.0, 1.0))
        )
    return np.array(thresholds)


def _binary_metrics_from_predictions(
    sensor_labels: torch.Tensor,
    sensor_pred: torch.Tensor,
    window_scores: Optional[torch.Tensor],
    window_labels: Optional[torch.Tensor],
    window_threshold: float,
    sensor_threshold: Union[float, np.ndarray],
) -> Dict[str, Any]:
    if window_scores is None:
        window_scores = sensor_pred.topk(k=2, dim=-1).values.mean(dim=-1)

    if window_labels is None:
        if sensor_labels.ndim == 1:
            window_true = sensor_labels.long()
        else:
            window_true = (sensor_labels.sum(dim=1) > 0).long()
    else:
        window_true = window_labels.reshape(-1).long()

    if isinstance(sensor_threshold, np.ndarray):
        thr = torch.from_numpy(sensor_threshold).to(sensor_pred.device)
        sensor_pred_binary = (sensor_pred > thr).float()
    else:
        sensor_pred_binary = (sensor_pred > sensor_threshold).float()
    window_pred = (window_scores > window_threshold).long()
    if window_pred.numel() != window_true.numel():
        raise ValueError("window_labels and window_scores must align in length")

    window_tp = ((window_pred == 1) & (window_true == 1)).sum().item()
    window_fp = ((window_pred == 1) & (window_true == 0)).sum().item()
    window_fn = ((window_pred == 0) & (window_true == 1)).sum().item()

    window_prec = window_tp / (window_tp + window_fp + 1e-6)
    window_rec = window_tp / (window_tp + window_fn + 1e-6)
    window_f1 = 2 * window_prec * window_rec / (window_prec + window_rec + 1e-6)

    sensor_tp = (sensor_pred_binary * sensor_labels).sum(dim=0)
    sensor_fp = (sensor_pred_binary * (1.0 - sensor_labels)).sum(dim=0)
    sensor_fn = ((1.0 - sensor_pred_binary) * sensor_labels).sum(dim=0)

    sensor_prec = (sensor_tp / (sensor_tp + sensor_fp + 1e-6)).cpu().numpy()
    sensor_rec = (sensor_tp / (sensor_tp + sensor_fn + 1e-6)).cpu().numpy()
    sensor_f1 = (
        (2 * (sensor_prec * sensor_rec) / (sensor_prec + sensor_rec + 1e-8)).mean()
        if len(sensor_prec)
        else 0.0
    )

    return {
        "window_precision": float(window_prec),
        "window_recall": float(window_rec),
        "window_f1": float(window_f1),
        "sensor_precision": sensor_prec,
        "sensor_recall": sensor_rec,
        "sensor_f1": float(sensor_f1),
        "window_pred_rate": float(window_pred.float().mean().item()),
    }


def _compute_epoch_hardness(
    dataset,
    model,
    device,
    hard_loss_alpha: float = 0.6,
    hard_loss_beta: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Optional scanner to estimate per-sample hardness for hard mining.
    Uses current model predictions and BCE loss for all samples.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
    )
    scores = []
    indices = []
    start_idx = 0
    with torch.no_grad():
        for X_batch, _, sensor_labels_batch, window_labels_batch in loader:
            batch_size = X_batch.size(0)
            batch_idx = torch.arange(start_idx, start_idx + batch_size)
            start_idx += batch_size
            indices.append(batch_idx)

            X_batch = X_batch.to(device)
            sensor_labels_batch = sensor_labels_batch.to(device)
            window_labels_batch = window_labels_batch.to(device).float()

            sensor_logits, global_logits, _ = model(
                X_batch, return_global=True, return_sensor_embeddings=True
            )
            sensor_error = F.binary_cross_entropy_with_logits(
                sensor_logits, sensor_labels_batch, reduction="none"
            ).mean(dim=1)
            global_error = F.binary_cross_entropy_with_logits(
                global_logits, window_labels_batch
            )
            sample_hardness = (
                hard_loss_alpha * sensor_error + hard_loss_beta * global_error
            )
            scores.append(sample_hardness.cpu())

    if len(indices) == 0:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    return torch.cat(scores), torch.cat(indices)


def _build_hard_mined_loader(
    dataset,
    hardness,
    sample_indices,
    hard_ratio,
    batch_size,
    num_workers,
    hard_mining_seed,
    pin_memory=False,
):
    hardness = hardness.clone().detach().float()
    if hardness.numel() == 0:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    hard_ratio = float(np.clip(hard_ratio, 0.0, 1.0))
    num_samples = len(sample_indices)
    if num_samples == 0:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    hard_count = int(num_samples * hard_ratio)
    hard_count = max(0, min(num_samples, hard_count))
    easy_count = max(1, num_samples - hard_count)

    gen = torch.Generator(device="cpu").manual_seed(hard_mining_seed)
    hard_sampler = (
        WeightedRandomSampler(
            weights=hardness.clamp(min=0.0) + 1e-6,
            num_samples=max(hard_count, 1),
            replacement=True,
            generator=gen,
        )
        if hard_count > 0
        else None
    )
    hard_idx_list = (
        torch.tensor(list(iter(hard_sampler)), dtype=torch.long)
        if hard_sampler is not None
        else torch.empty(0, dtype=torch.long)
    )

    easy_perm = torch.randperm(num_samples, generator=gen)[:easy_count]
    easy_idx_list = sample_indices[easy_perm]

    combined_indices = torch.cat(
        [hard_idx_list, easy_idx_list[: num_samples - hard_idx_list.numel()]]
    )
    combined_indices = combined_indices[
        torch.randperm(combined_indices.numel(), generator=gen)
    ]
    combined_indices = combined_indices.clamp(min=0, max=num_samples - 1)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=SubsetRandomSampler(combined_indices.tolist()),
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def train_stage2_clean(
    train_loader,
    val_loader,
    calib_loader,
    num_sensors: int,
    window_size: int,
    stage1_checkpoint_path: str,
    num_epochs: int = NUM_EPOCHS,
    device: str = "cpu",
    lambda_sensor: float = LAMBDA_SENSOR,
    lambda_window: float = LAMBDA_WINDOW,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    checkpoint_dir: str = "checkpoints",
    checkpoint_name: Optional[str] = None,
    sensor_pos_weight_scale: float = SENSOR_POS_WEIGHT_SCALE,
    sensor_pos_weight_min: float = SENSOR_POS_WEIGHT_SCALE_MIN,
    sensor_pos_weight_decay_epochs: int = SENSOR_POS_WEIGHT_SCALE_DECAY_EPOCHS,
    use_scheduled_sensor_pos_weight: bool = True,
    sensor_pos_weight_cap: float = SENSOR_POS_WEIGHT_CAP,
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",
    gradient_accumulation_steps: int = 1,
    use_amp: bool = False,
    max_batches_per_epoch: Optional[int] = None,
    hard_mining: bool = False,
    hard_ratio: float = 0.2,
    hard_ratio_start: float = 0.3,
    hard_ratio_end: float = 0.3,
    hard_ratio_switch_epoch: int = 0,
    hard_mining_seed: int = 42,
    hard_mining_start_epoch: int = 0,
    window_score_weight: float = WINDOW_SCORE_WEIGHT,
    lambda_aux_global: float = 0.2,
    window_pos_weight: float = 1.0,
    use_fixed_train_threshold: bool = True,
    train_window_thr: float = 0.5,
    train_sensor_thr: float = 0.5,
    calibrate_every: int = 1,
    freeze_backbone_epochs: int = 3,
    use_calibrated_threshold_for_checkpoint: bool = True,
    num_workers: int = 4,
    use_focal_loss: bool = False,
    focal_gamma: float = 2.0,
    scheduler_patience: int = 15,
    scheduler_factor: float = 0.7,
    lr_min: float = 1e-6,
):
    print(f"\nLoading Stage 1 checkpoint from {stage1_checkpoint_path}...")
    base_state_dict, base_metadata = _load_checkpoint_state(
        stage1_checkpoint_path, device=device
    )

    model = GDN(
        num_nodes=num_sensors,
        window_size=window_size,
        embed_dim=EMBED_DIM,
        top_k=TOP_K,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    if use_compile and hasattr(torch, "compile"):
        print(f"\nCompiling model with mode='{compile_mode}'...")
        print("  Note: First epoch will be slower due to compilation")
        model = torch.compile(model, mode=compile_mode)
    elif use_compile:
        print("\nWarning: torch.compile() not available (requires PyTorch 2.0+)")
        print("  Continuing without compilation")

    # Validate GAT format matches before loading
    model_uses_lin_src = any("gat.lin_src" in k for k in model.state_dict().keys())
    ckpt_has_lin_src = any("gat.lin_src" in k for k in base_state_dict.keys())
    ckpt_has_lin = any("gat.lin.weight" in k for k in base_state_dict.keys())

    if model_uses_lin_src and ckpt_has_lin:
        raise ValueError(
            "GAT format mismatch: model expects gat.lin_src/lin_dst (PyG >= 2.3) "
            "but checkpoint uses gat.lin (older PyG). "
            "Re-train Stage 1 with the current PyG version."
        )
    if not model_uses_lin_src and ckpt_has_lin_src:
        raise ValueError(
            "GAT format mismatch: model expects gat.lin (older PyG) "
            "but checkpoint uses gat.lin_src/lin_dst (PyG >= 2.3). "
            "Re-train Stage 1 with the current PyG version."
        )

    print("  ✓ GAT format validated — no remapping needed")

    missing, unexpected = model.load_state_dict(base_state_dict, strict=False)
    unexpected_filtered = [k for k in unexpected if "global_classifier" not in k]
    missing_filtered = [k for k in missing if "global_classifier" not in k]
    if missing_filtered:
        raise ValueError(f"Unexpected missing keys in checkpoint: {missing_filtered}")
    if unexpected_filtered:
        raise ValueError(f"Unexpected extra keys in checkpoint: {unexpected_filtered}")
    print("  ✓ Stage 1 checkpoint loaded successfully")

    # Default train-all on first run.
    for p in model.parameters():
        p.requires_grad = True

    # Warm-up defaults: only fine-tune top heads + embeddings while keeping
    # temporal/GAT blocks frozen to preserve Stage-1 representation.
    if freeze_backbone_epochs > 0:
        for name, param in model.named_parameters():
            if name.startswith(("temporal_encoder", "gat", "gat_norm")):
                if (
                    "sensor_embeddings" in name
                    or "sensor_classifier" in name
                    or "global_classifier" in name
                    or "layer_norm" in name
                ):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    # Separate parameter groups (embeddings at reduced LR by default).
    # Keep all model parameters in the optimizer so a later unfreeze step can
    # immediately start updating them without rebuilding the optimizer.
    embedding_params = [model.sensor_embeddings]
    other_params = [
        p for n, p in model.named_parameters() if p is not model.sensor_embeddings
    ]
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": learning_rate, "weight_decay": weight_decay},
            {
                "params": embedding_params,
                "lr": learning_rate * 0.1,
                "weight_decay": weight_decay,
            },
        ],
        # If other_params are empty initially, optimizer still tracks embedding params.
    )

    WARMUP_EPOCHS = 8
    base_lr = optimizer.param_groups[0]["lr"]

    def warmup_fn(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS  # scales lr from ~0 to base_lr
        return 1.0  # hand off to ReduceLROnPlateau after warmup

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    # Optional mixed precision
    scaler = None
    if use_amp and device.startswith("cuda"):
        scaler = GradScaler()
    elif use_amp:
        print("  ⚠ AMP requested but not on CUDA device, disabling AMP")
        use_amp = False

    # Losses
    window_criterion = nn.BCEWithLogitsLoss()

    # Per-sensor pos_weight: ratio of negatives to positives per sensor column
    all_sensor_labels = []
    for _, _, sensor_labels_batch, _ in train_loader:
        all_sensor_labels.append(sensor_labels_batch.to("cpu"))
    train_sensor_labels = (
        torch.cat(all_sensor_labels, dim=0)
        if all_sensor_labels
        else torch.zeros(0, num_sensors)
    )

    per_sensor_pos_weight = []
    for s in range(train_sensor_labels.shape[1]):
        n_pos = train_sensor_labels[:, s].sum().item()
        n_neg = train_sensor_labels.shape[0] - n_pos
        if n_pos > 0:
            w = min(n_neg / n_pos, sensor_pos_weight_cap)
            w = max(w, 1.0)
        else:
            w = 1.0
        per_sensor_pos_weight.append(w)

    per_sensor_pos_weight = torch.tensor(
        per_sensor_pos_weight, dtype=torch.float32, device=device
    )
    print(f"Per-sensor pos-weights: {per_sensor_pos_weight.cpu().tolist()}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=scheduler_patience,
        factor=scheduler_factor,
        min_lr=lr_min,
    )

    best_val_loss = float("inf")
    best_balanced_score = -1.0
    best_epoch = 0
    patience_counter = 0
    max_patience = 25
    pin_memory = device.startswith("cuda")
    train_dataset = train_loader.dataset
    metric_history = []
    best_thresholds = {"window": 0.5, "sensor": 0.5}

    print(f"\n{'=' * 80}")
    print("Stage 2: Clean Sensor-Only Anomaly Fine-tuning")
    print(f"{'=' * 80}")
    print(f"Embedding dim: {EMBED_DIM}, Hidden dim: {HIDDEN_DIM}")
    print(f"Lambda_sensor: {lambda_sensor}, Lambda_window: {lambda_window}")
    print(f"Window score weight (inference only): {window_score_weight}")
    print(f"Freeze backbone epochs: {freeze_backbone_epochs}")
    print(f"Device: {device}")
    print(
        f"Hard mining enabled: {hard_mining} (start epoch: {hard_mining_start_epoch})"
    )
    print(f"Sensor embeddings LR: {learning_rate * 0.1:.6f}")
    print(
        f"Trainable param mode: {'embeddings + heads (temporal/GAT frozen)' if freeze_backbone_epochs > 0 else 'full model'}\n"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_basename = f"stage2_clean_{checkpoint_name}.pt" if checkpoint_name else "stage2_clean_best.pt"
    best_checkpoint_path = os.path.join(checkpoint_dir, ckpt_basename)

    for epoch in range(num_epochs):
        if epoch == freeze_backbone_epochs:
            for name, param in model.named_parameters():
                if "temporal_encoder" in name or "gru" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print(
                f"Epoch {epoch}: Unfreezing GAT + heads. GRU remains frozen until epoch {freeze_backbone_epochs + 10}."
            )
        elif epoch == freeze_backbone_epochs + 10:
            for param in model.parameters():
                param.requires_grad = True
            print(f"Epoch {epoch}: Fully unfreezing GRU. All parameters now trainable.")

        current_scale = _sensor_weight_scale(
            epoch=epoch,
            num_epochs=num_epochs,
            scale_init=sensor_pos_weight_scale,
            scale_min=sensor_pos_weight_min,
            decay_epochs=sensor_pos_weight_decay_epochs,
            use_schedule=use_scheduled_sensor_pos_weight,
        )
        current_sensor_pos_weight = torch.clamp(
            per_sensor_pos_weight * current_scale, min=1.0, max=sensor_pos_weight_cap
        )
        current_sensor_pos_weight = torch.clamp(
            current_sensor_pos_weight, min=0.1, max=sensor_pos_weight_cap
        )

        if hard_ratio_switch_epoch <= 0:
            current_hard_ratio = hard_ratio_end
        elif epoch < hard_ratio_switch_epoch:
            if hard_ratio_switch_epoch <= 1:
                current_hard_ratio = hard_ratio_end
            else:
                alpha = float(epoch) / float(hard_ratio_switch_epoch)
                current_hard_ratio = (
                    hard_ratio_start + (hard_ratio_end - hard_ratio_start) * alpha
                )
        else:
            current_hard_ratio = hard_ratio_end
        current_hard_ratio = float(np.clip(current_hard_ratio, 0.0, 1.0))

        if hard_mining and epoch >= hard_mining_start_epoch:
            hardness, mapped_indices = _compute_epoch_hardness(
                train_dataset,
                model,
                device=device,
            )
            if len(mapped_indices) != len(train_dataset):
                train_iter = _build_hard_mined_loader(
                    train_dataset,
                    torch.zeros(len(train_dataset)),
                    torch.arange(len(train_dataset)),
                    0.0,
                    train_loader.batch_size,
                    num_workers,
                    hard_mining_seed=hard_mining_seed + epoch,
                    pin_memory=pin_memory,
                )
            else:
                train_iter = _build_hard_mined_loader(
                    train_dataset,
                    hardness,
                    mapped_indices,
                    current_hard_ratio,
                    train_loader.batch_size,
                    num_workers,
                    hard_mining_seed=hard_mining_seed + epoch,
                    pin_memory=pin_memory,
                )
        else:
            train_iter = train_loader
            if hard_mining:
                if epoch + 1 == hard_mining_start_epoch:
                    print(
                        f"Hard mining scheduled but not active until epoch {hard_mining_start_epoch + 1}."
                    )

        if isinstance(train_iter, DataLoader):
            train_iter = train_iter
        else:
            # Fallback: make it a materialized iterable if needed
            train_iter = list(train_iter)

        if max_batches_per_epoch is not None and isinstance(train_iter, list):
            train_iter = train_iter[:max_batches_per_epoch]
        elif max_batches_per_epoch is not None and hasattr(train_iter, "__iter__"):
            train_iter = list(train_iter)[:max_batches_per_epoch]

        model.train()

        train_loss_sensor = 0.0
        train_loss_window = 0.0

        with tqdm(
            train_iter, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False
        ) as pbar:
            optimizer.zero_grad()

            for batch_idx, batch in enumerate(pbar):
                X_batch, _, sensor_labels_batch, window_labels_batch = batch
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                if use_amp and scaler is not None:
                    with autocast():
                        sensor_logits, global_logits, sensor_embeddings = model(
                            X_batch, return_global=True, return_sensor_embeddings=True
                        )
                        if use_focal_loss:
                            loss_sensor = _focal_bce_with_logits(
                                sensor_logits,
                                sensor_labels_batch,
                                current_sensor_pos_weight,
                                gamma=focal_gamma,
                            )
                        else:
                            loss_sensor = F.binary_cross_entropy_with_logits(
                                sensor_logits,
                                sensor_labels_batch,
                                pos_weight=current_sensor_pos_weight,
                                reduction="none",
                            ).mean()
                        # Training: use global logits directly — cleaner gradient signal
                        loss_window = F.binary_cross_entropy_with_logits(
                            global_logits.squeeze(-1),
                            window_labels_batch.float(),
                            pos_weight=torch.tensor(window_pos_weight, device=device),
                        )
                        loss = (
                            lambda_sensor * loss_sensor
                            + lambda_window * loss_window
                        ) / gradient_accumulation_steps
                else:
                    sensor_logits, global_logits, sensor_embeddings = model(
                        X_batch, return_global=True, return_sensor_embeddings=True
                    )
                    if use_focal_loss:
                        loss_sensor = _focal_bce_with_logits(
                            sensor_logits,
                            sensor_labels_batch,
                            current_sensor_pos_weight,
                            gamma=focal_gamma,
                        )
                    else:
                        loss_sensor = F.binary_cross_entropy_with_logits(
                            sensor_logits,
                            sensor_labels_batch,
                            pos_weight=current_sensor_pos_weight,
                            reduction="none",
                        ).mean()
                    # Training: use global logits directly — cleaner gradient signal
                    loss_window = F.binary_cross_entropy_with_logits(
                        global_logits.squeeze(-1),
                        window_labels_batch.float(),
                        pos_weight=torch.tensor(window_pos_weight, device=device),
                    )
                    loss = (
                        lambda_sensor * loss_sensor + lambda_window * loss_window
                    ) / gradient_accumulation_steps

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"NaN/Inf in loss at epoch {epoch + 1}, skipping batch")
                    continue

                if use_amp and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    if use_amp and scaler is not None:
                        scaler.unscale_(optimizer)

                    trainable_params = [
                        p for p in model.parameters() if p.requires_grad
                    ]
                    if len(trainable_params) > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)

                    if use_amp and scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    optimizer.zero_grad()

                train_loss_sensor += (
                    loss_sensor.item() * X_batch.size(0) * gradient_accumulation_steps
                )
                train_loss_window += (
                    loss_window.item() * X_batch.size(0) * gradient_accumulation_steps
                )

        # Handle leftover accumulation
        if (batch_idx + 1) % gradient_accumulation_steps != 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            if len(trainable_params) > 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.5)

            if use_amp and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        train_loss_sensor /= len(train_loader.dataset)
        train_loss_window /= len(train_loader.dataset)

        # Validation + metrics
        model.eval()

        val_loss_sensor = 0.0
        val_loss_window = 0.0
        all_sensor_labels = []
        all_sensor_probs = []
        all_window_scores = []
        all_window_labels = []

        with torch.no_grad():
            for vb_idx, (
                X_batch,
                _,
                sensor_labels_batch,
                window_labels_batch,
            ) in enumerate(val_loader):
                X_batch = X_batch.to(device)
                sensor_labels_batch = sensor_labels_batch.to(device)
                window_labels_batch = window_labels_batch.long().to(device)

                sensor_logits, global_logits, sensor_embeddings = model(
                    X_batch, return_global=True, return_sensor_embeddings=True
                )
                if use_focal_loss:
                    loss_sensor = _focal_bce_with_logits(
                        sensor_logits,
                        sensor_labels_batch,
                        current_sensor_pos_weight,
                        gamma=focal_gamma,
                    )
                else:
                    loss_sensor = F.binary_cross_entropy_with_logits(
                        sensor_logits,
                        sensor_labels_batch,
                        pos_weight=current_sensor_pos_weight,
                        reduction="none",
                    ).mean()
                # Val loss: BCE on global_logits (consistent with training)
                loss_window = F.binary_cross_entropy_with_logits(
                    global_logits.squeeze(-1),
                    window_labels_batch.float(),
                    pos_weight=torch.tensor(window_pos_weight, device=device),
                )

                val_loss_sensor += loss_sensor.item() * X_batch.size(0)
                val_loss_window += loss_window.item() * X_batch.size(0)

                all_sensor_labels.append(sensor_labels_batch)
                all_window_labels.append(window_labels_batch)
                sensor_probs = torch.sigmoid(sensor_logits)
                all_sensor_probs.append(sensor_probs)
                # EXPERIMENT: global-only window scores (KIT diagnostic)
                window_score_inference = torch.sigmoid(global_logits.squeeze(-1))
                all_window_scores.append(window_score_inference)

        val_loss_sensor /= len(val_loader.dataset)
        val_loss_window /= len(val_loader.dataset)
        val_total_loss = (
            lambda_sensor * val_loss_sensor
            + lambda_window * val_loss_window
        )

        all_sensor_labels = torch.cat(all_sensor_labels, dim=0)
        all_window_labels = torch.cat(all_window_labels, dim=0)
        all_sensor_probs = torch.cat(all_sensor_probs, dim=0)
        all_window_scores = torch.cat(all_window_scores, dim=0)

        fixed_metrics = _binary_metrics_from_predictions(
            all_sensor_labels,
            all_sensor_probs,
            window_scores=all_window_scores,
            window_labels=all_window_labels,
            window_threshold=train_window_thr if use_fixed_train_threshold else 0.5,
            sensor_threshold=train_sensor_thr if use_fixed_train_threshold else 0.5,
        )

        # Fixed-threshold reporting always available.
        fixed_threshold = train_window_thr if use_fixed_train_threshold else 0.5

        if calibrate_every > 0 and (
            (epoch + 1) % calibrate_every == 0 or epoch == num_epochs - 1
        ):
            # Calibrate thresholds on held-out calibration set (not val)
            calib_sensor_labels_list = []
            calib_sensor_probs_list = []
            calib_window_scores_list = []
            calib_window_labels_list = []
            with torch.no_grad():
                for (
                    X_batch,
                    _,
                    sensor_labels_batch,
                    window_labels_batch,
                ) in calib_loader:
                    X_batch = X_batch.to(device)
                    sensor_labels_batch = sensor_labels_batch.to(device)
                    window_labels_batch = window_labels_batch.long().to(device)
                    sensor_logits, global_logits, _ = model(
                        X_batch, return_global=True, return_sensor_embeddings=True
                    )
                    calib_sensor_labels_list.append(sensor_labels_batch)
                    calib_sensor_probs_list.append(torch.sigmoid(sensor_logits))
                    calib_window_labels_list.append(window_labels_batch)
                    # EXPERIMENT: global-only window scores (KIT diagnostic)
                    window_score_inference = torch.sigmoid(global_logits.squeeze(-1))
                    calib_window_scores_list.append(window_score_inference)
            calib_sensor_labels = torch.cat(calib_sensor_labels_list, dim=0)
            calib_sensor_probs = torch.cat(calib_sensor_probs_list, dim=0)
            calib_window_scores = torch.cat(calib_window_scores_list, dim=0)
            calib_window_labels = torch.cat(calib_window_labels_list, dim=0)

            calibrated_window_thr = _best_f1_threshold(
                calib_window_scores,
                calib_window_labels.reshape(-1).float(),
            )
            calibrated_sensor_thr = _best_f1_threshold(
                calib_sensor_probs, calib_sensor_labels
            )
            # Calibration set is held out during training — do not use train split here
            per_sensor_thr = compute_per_sensor_thresholds(
                calib_sensor_probs, calib_sensor_labels
            )
            THRESHOLD_FLOOR = 0.0
            calibrated_window_thr = max(calibrated_window_thr, THRESHOLD_FLOOR)
            calibrated_sensor_thr = max(calibrated_sensor_thr, THRESHOLD_FLOOR)
            print(
                f"Calibrated thresholds (after floor {THRESHOLD_FLOOR}): "
                f"window={calibrated_window_thr:.4f}, sensor={calibrated_sensor_thr:.4f}, "
                f"per_sensor={per_sensor_thr.tolist()}"
            )
            # Compute cal_metrics on val set using calibrated thresholds (for checkpoint selection)
            cal_metrics = _binary_metrics_from_predictions(
                all_sensor_labels,
                all_sensor_probs,
                window_scores=all_window_scores,
                window_labels=all_window_labels,
                window_threshold=calibrated_window_thr,
                sensor_threshold=per_sensor_thr,
            )
        else:
            calibrated_window_thr = best_thresholds["window"]
            calibrated_sensor_thr = best_thresholds["sensor"]
            cal_metrics = None

        if cal_metrics is not None:
            best_thresholds["window"] = calibrated_window_thr
            best_thresholds["sensor"] = calibrated_sensor_thr
            best_thresholds["per_sensor"] = per_sensor_thr.tolist()

        scheduler.step(val_total_loss)
        if epoch < WARMUP_EPOCHS:
            warmup_scheduler.step()

        if cal_metrics is not None:
            w, s = cal_metrics["window_f1"], cal_metrics["sensor_f1"]
            current_balanced = 2 * w * s / (w + s) if (w + s) > 0 else 0.0
            calibrated_balanced = current_balanced
        else:
            current_balanced = min(
                fixed_metrics["window_f1"],
                fixed_metrics["sensor_f1"],
            )
            calibrated_balanced = float("nan")

        # Primary checkpoint metric: calibrated window F1 alone
        current_score = (
            cal_metrics["window_f1"]
            if cal_metrics is not None
            else fixed_metrics["window_f1"]
        )
        if current_score > best_balanced_score or (
            current_score == best_balanced_score and val_total_loss < best_val_loss
        ):
            best_balanced_score = current_score
            best_val_loss = val_total_loss
            best_epoch = epoch + 1
            patience_counter = 0
            threshold_snapshot = {
                "window": best_thresholds["window"]
                if use_calibrated_threshold_for_checkpoint
                else fixed_threshold,
                "sensor": best_thresholds["sensor"]
                if use_calibrated_threshold_for_checkpoint
                else train_sensor_thr,
                "per_sensor": best_thresholds.get("per_sensor", []),
            }
            ckpt_payload = {
                "stage": 2,
                "stage2_mode": "clean",
                "model_state_dict": model.state_dict(),
                "checkpoint_stage1": stage1_checkpoint_path,
                "sensor_names": SENSOR_COLS,
                "window_size": window_size,
                "embed_dim": EMBED_DIM,
                "top_k": TOP_K,
                "hidden_dim": HIDDEN_DIM,
                "sensor_pos_weight_scale": current_scale,
                "sensor_pos_weight_cap": sensor_pos_weight_cap,
                "lambda_sensor": lambda_sensor,
                "lambda_window": lambda_window,
                "window_score_weight": window_score_weight,
                "hard_mining": hard_mining,
                "hard_ratio_end": hard_ratio_end,
                "hard_ratio_start": hard_ratio_start,
                "hard_ratio_switch_epoch": hard_ratio_switch_epoch,
                "freeze_backbone_epochs": freeze_backbone_epochs,
                "metric_history": metric_history,
                "best_balanced_score": best_balanced_score,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "calibrated_thresholds": threshold_snapshot,
                "train_thresholds": {
                    "window": train_window_thr,
                    "sensor": train_sensor_thr,
                },
            }
            torch.save(ckpt_payload, best_checkpoint_path)

            metric_label = "Harmonic" if cal_metrics is not None else "Balanced"
            print(
                f"  ✓ Best model saved (Window F1: {best_balanced_score:.4f}, {metric_label}: {current_balanced:.4f}, Val Loss: {val_total_loss:.4f})"
            )
        else:
            patience_counter += 1

        metric_entry = {
            "epoch": epoch + 1,
            "train_sensor_loss": float(train_loss_sensor),
            "train_window_loss": float(train_loss_window),
            "val_total_loss": float(val_total_loss),
            "val_sensor_loss": float(val_loss_sensor),
            "val_window_loss": float(val_loss_window),
            "fixed_window_f1": float(fixed_metrics["window_f1"]),
            "fixed_sensor_f1": float(fixed_metrics["sensor_f1"]),
            "fixed_window_recall": float(fixed_metrics["window_recall"]),
            "fixed_sensor_recall": float(np.mean(fixed_metrics["sensor_recall"])),
            "balanced_fixed": float(current_balanced),
            "window_thr_fixed": fixed_threshold,
            "sensor_thr_fixed": train_sensor_thr,
            "window_thr_calibrated": float(best_thresholds["window"]),
            "sensor_thr_calibrated": float(best_thresholds["sensor"]),
            "calibrated_window_f1": float(cal_metrics["window_f1"])
            if cal_metrics is not None
            else float("nan"),
            "calibrated_sensor_f1": float(cal_metrics["sensor_f1"])
            if cal_metrics is not None
            else float("nan"),
            "calibrated_balanced": float(calibrated_balanced)
            if not isinstance(calibrated_balanced, float)
            or not np.isnan(calibrated_balanced)
            else float("nan"),
        }
        metric_history.append(metric_entry)

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            calib_text = (
                f"ValCalib(W,S): ({cal_metrics['window_f1']:.4f} , {cal_metrics['sensor_f1']:.4f}) | "
                if cal_metrics is not None
                else "ValCalib(W,S): not computed | "
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"SensorLoss: {train_loss_sensor:.4f} | "
                f"WindowLoss: {train_loss_window:.4f} | "
                f"ValTotal: {val_total_loss:.4f} | "
                f"ValFixed(W,S): ({fixed_metrics['window_f1']:.4f}, {fixed_metrics['sensor_f1']:.4f}) | "
                f"{calib_text}"
                f"ThrWin: {best_thresholds['window']:.3f} | "
                f"ThrSens: {best_thresholds['sensor']:.3f} | "
                f"scale: {current_scale:.3f} | "
                f"hard_ratio: {current_hard_ratio:.2f}"
            )
        else:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"SensorLoss: {train_loss_sensor:.4f} | "
                f"WindowLoss: {train_loss_window:.4f} | "
                f"ValTotal: {val_total_loss:.4f} | "
                f"ValFixed W/S: ({fixed_metrics['window_f1']:.4f}, {fixed_metrics['sensor_f1']:.4f})"
            )

        if patience_counter >= max_patience:
            print(
                f"\nEarly stopping at epoch {epoch + 1} (no improvement for {max_patience} epochs)"
            )
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model_state = checkpoint["model_state_dict"]

    # Validate GAT format matches before loading
    model_uses_lin_src = any("gat.lin_src" in k for k in model.state_dict().keys())
    ckpt_has_lin_src = any("gat.lin_src" in k for k in model_state.keys())
    ckpt_has_lin = any("gat.lin.weight" in k for k in model_state.keys())

    if model_uses_lin_src and ckpt_has_lin:
        raise ValueError(
            "GAT format mismatch: model expects gat.lin_src/lin_dst (PyG >= 2.3) "
            "but checkpoint uses gat.lin (older PyG). "
            "Re-train Stage 1 with the current PyG version."
        )
    if not model_uses_lin_src and ckpt_has_lin_src:
        raise ValueError(
            "GAT format mismatch: model expects gat.lin (older PyG) "
            "but checkpoint uses gat.lin_src/lin_dst (PyG >= 2.3). "
            "Re-train Stage 1 with the current PyG version."
        )

    print("  ✓ GAT format validated — no remapping needed")

    missing, unexpected = model.load_state_dict(model_state, strict=False)
    unexpected_filtered = [k for k in unexpected if "global_classifier" not in k]
    missing_filtered = [k for k in missing if "global_classifier" not in k]
    if missing_filtered:
        raise ValueError(f"Unexpected missing keys in checkpoint: {missing_filtered}")
    if unexpected_filtered:
        raise ValueError(f"Unexpected extra keys in checkpoint: {unexpected_filtered}")
    print("  ✓ Best checkpoint loaded successfully")

    # Post-training train+val threshold calibration (always runs)
    model.eval()
    all_scores, all_labels = [], []
    with torch.no_grad():
        for loader in (train_loader, val_loader):
            for X_batch, _, sensor_labels_batch, window_labels_batch in loader:
                X_batch = X_batch.to(device)
                sensor_logits, _, _ = model(
                    X_batch, return_global=True, return_sensor_embeddings=True
                )
                sensor_probs = torch.sigmoid(sensor_logits)
                window_scores = (
                    sensor_probs.topk(k=2, dim=-1).values.mean(dim=-1).cpu().numpy()
                )
                window_labels = (window_labels_batch > 0).float().cpu().numpy()
                all_scores.append(window_scores)
                all_labels.append(window_labels)
    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)
    best_f1, best_thresh = 0.0, 0.5
    for t in np.arange(0.30, 0.901, 0.005):
        f1 = f1_score(all_labels, (all_scores >= t).astype(np.float32), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(
        f"[Calibration] Best train+val threshold: {best_thresh:.4f} (F1={best_f1:.4f})"
    )
    checkpoint["calibrated_threshold_trainval"] = {
        "window": float(best_thresh),
        "f1": float(best_f1),
    }
    torch.save(checkpoint, best_checkpoint_path)

    print(f"\n{'=' * 80}")
    print("Stage 2 clean training complete!")
    print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")
    print(f"Best balanced score: {checkpoint['best_balanced_score']:.4f}")
    print(f"Best epoch: {checkpoint['best_epoch']}")
    print(f"Saved as: {best_checkpoint_path}")
    print(f"{'=' * 80}\n")
    return model


def main():
    parser = argparse.ArgumentParser(description="Stage 2 clean training script")
    parser.add_argument(
        "--data_path", type=str, default=DATA_PATH, help="Path to data directory"
    )
    parser.add_argument(
        "--stage1_checkpoint", type=str, required=True, help="Stage 1 checkpoint path"
    )
    parser.add_argument(
        "--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs"
    )
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument(
        "--lambda_sensor",
        type=float,
        default=LAMBDA_SENSOR,
        help="Sensor BCE coefficient",
    )
    parser.add_argument(
        "--lambda_window",
        type=float,
        default=LAMBDA_WINDOW,
        help="Window BCE coefficient",
    )
    parser.add_argument(
        "--lambda_aux_global",
        type=float,
        default=0.2,
        help="Auxiliary BCE on global_logits so global branch gets gradient",
    )
    parser.add_argument(
        "--window_pos_weight",
        type=float,
        default=1.0,
        help="Positive class weight for window BCE loss. >1 penalises false negatives, "
        "<1 penalises false positives. Use >1 to reduce recall-bias.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory for stage2_clean checkpoints",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default=None,
        help="Custom checkpoint basename (e.g. 'exp1'). Saves as stage2_clean_<name>.pt instead of stage2_clean_best.pt",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    parser.add_argument("--cpu_only", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--sensor_pos_weight_scale",
        type=float,
        default=SENSOR_POS_WEIGHT_SCALE,
        help="Pos-weight init scale",
    )
    parser.add_argument(
        "--sensor_pos_weight_min",
        type=float,
        default=SENSOR_POS_WEIGHT_SCALE_MIN,
        help="Pos-weight min scale",
    )
    parser.add_argument(
        "--sensor_pos_weight_decay_epochs",
        type=int,
        default=SENSOR_POS_WEIGHT_SCALE_DECAY_EPOCHS,
        help="Pos-weight decay epochs",
    )
    parser.add_argument(
        "--disable_sensor_pos_weight_schedule",
        action="store_true",
        help="Keep fixed positive-class weight scale",
    )
    parser.add_argument(
        "--sensor_pos_weight_cap",
        type=float,
        default=SENSOR_POS_WEIGHT_CAP,
        help="Positive-class weight cap",
    )
    parser.add_argument(
        "--use_compile",
        action="store_true",
        help="Use torch.compile() for model forward",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile() mode",
    )
    parser.add_argument("--use_amp", action="store_true", help="Enable AMP on CUDA")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--max_batches_per_epoch",
        type=int,
        default=None,
        help="Limit batches per epoch",
    )
    parser.add_argument(
        "--fault_rate",
        type=float,
        default=0.25,
        help="Fault injection rate for synthetic anomalies (0=0%%, 0.25=25%%, default: 0.25)",
    )

    parser.add_argument("--hard_mining", action="store_true", help="Enable hard mining")
    parser.add_argument(
        "--hard_ratio", type=float, default=0.2, help="Target hard sample ratio"
    )
    parser.add_argument(
        "--hard_ratio_start",
        type=float,
        default=0.3,
        help="Hard ratio at epoch 0 when switching on",
    )
    parser.add_argument(
        "--hard_ratio_end",
        type=float,
        default=0.3,
        help="Hard ratio after switch epoch",
    )
    parser.add_argument(
        "--hard_ratio_switch_epoch",
        type=int,
        default=0,
        help="Epoch index when hard ratio reaches target end value",
    )
    parser.add_argument(
        "--hard_mining_start_epoch",
        type=int,
        default=0,
        help="Epoch to start hard mining",
    )
    parser.add_argument(
        "--hard_mining_seed", type=int, default=42, help="Seed for hard mining sampling"
    )
    parser.add_argument(
        "--window_score_weight",
        type=float,
        default=WINDOW_SCORE_WEIGHT,
        help="Weight for aggregated sensor score in window score",
    )
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=3,
        help="Keep only sensor embeddings trainable for first N epochs",
    )
    parser.add_argument(
        "--train_sensor_threshold",
        type=float,
        default=0.5,
        help="Sensor threshold used for fixed training metrics",
    )
    parser.add_argument(
        "--train_window_threshold",
        type=float,
        default=0.5,
        help="Window threshold used for fixed training metrics",
    )
    parser.add_argument(
        "--calibrate_every",
        type=int,
        default=1,
        help="Re-fit thresholds every N epochs (0/negative disables periodic tuning)",
    )
    parser.add_argument(
        "--no_calibrated_threshold_for_checkpoint",
        action="store_true",
        help="Use train thresholds (0.5) in checkpoint instead of calibrated (default: save calibrated)",
    )
    parser.add_argument(
        "--use_focal_loss",
        action="store_true",
        help="Use focal loss for sensor BCE instead of standard BCE",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (default: 2.0)",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=15,
        help="ReduceLROnPlateau patience (default: 15)",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.7,
        help="Factor by which LR is reduced on plateau (default: 0.7).",
    )
    parser.add_argument(
        "--lr_min",
        type=float,
        default=1e-6,
        help="Minimum learning rate for ReduceLROnPlateau (default: 1e-6)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cpu_only:
        device = "cpu"
        print("Using device: cpu (forced via --cpu_only)")
    elif args.device is not None:
        device = args.device
        if device == "mps":
            print("Warning: MPS not supported. Falling back to CPU.")
            device = "cpu"
        print(f"Using device: {device} (specified)")
    else:
        if torch.cuda.is_available():
            device = "cuda"
            print("Using device: cuda (auto-detected)")
        else:
            device = "cpu"
            print("Using device: cpu (auto-detected)")

    print(f"\nLoading from .npz dataset: {args.data_path}")
    train_npz = os.path.join(args.data_path, "train.npz")
    val_npz = os.path.join(args.data_path, "val.npz")

    if not os.path.exists(train_npz) or not os.path.exists(val_npz):
        raise FileNotFoundError(
            f"train.npz / val.npz not found in {args.data_path}. "
            f"Run: python data/create_shared_dataset.py --output-dir {args.data_path}"
        )

    train_data = np.load(train_npz, allow_pickle=True)
    val_data = np.load(val_npz, allow_pickle=True)

    X_train_sensor = torch.tensor(train_data["normalized_windows"], dtype=torch.float32)
    train_sensor_labels = torch.tensor(train_data["sensor_labels"], dtype=torch.float32)
    train_window_labels = torch.tensor(train_data["window_is_faulty"], dtype=torch.long)

    X_val_sensor = torch.tensor(val_data["normalized_windows"], dtype=torch.float32)
    val_sensor_labels = torch.tensor(val_data["sensor_labels"], dtype=torch.float32)
    val_window_labels = torch.tensor(val_data["window_is_faulty"], dtype=torch.long)

    # Stage 2 loss does not use y_* directly; last timestep keeps TensorDataset shape consistent
    y_train = X_train_sensor[:, -1, :]
    y_val = X_val_sensor[:, -1, :]

    # Val doubles as calibration set
    X_calib_sensor = X_val_sensor
    calib_sensor_labels = val_sensor_labels
    calib_window_labels = val_window_labels
    y_calib = y_val

    train_faulty = train_window_labels.sum().item()
    val_faulty = val_window_labels.sum().item()
    print(f"  Train: {train_faulty}/{len(X_train_sensor)} faulty windows")
    print(f"  Val:   {val_faulty}/{len(X_val_sensor)} faulty windows")

    train_ds = TensorDataset(
        X_train_sensor, y_train, train_sensor_labels, train_window_labels
    )
    val_ds = TensorDataset(X_val_sensor, y_val, val_sensor_labels, val_window_labels)
    calib_ds = TensorDataset(
        X_calib_sensor, y_calib, calib_sensor_labels, calib_window_labels
    )
    print(
        f"Calib set: {len(calib_ds)} windows, {calib_sensor_labels.sum().item():.0f} faulty sensors"
    )

    pin_memory = device.startswith("cuda")
    g = torch.Generator().manual_seed(args.seed)
    if args.num_workers > 0:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
        calib_loader = DataLoader(
            calib_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=True,
            prefetch_factor=2,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            generator=g,
            num_workers=0,
            pin_memory=False,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        calib_loader = DataLoader(
            calib_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    num_sensors = len(SENSOR_COLS)
    print(f"\nTrain windows: {len(train_ds)}, Sensors: {num_sensors}")

    calibrate_every = args.calibrate_every
    if calibrate_every is None or calibrate_every <= 0:
        calibrate_every = 0

    train_stage2_clean(
        train_loader=train_loader,
        val_loader=val_loader,
        calib_loader=calib_loader,
        num_sensors=num_sensors,
        window_size=WINDOW_SIZE,
        stage1_checkpoint_path=args.stage1_checkpoint,
        num_epochs=args.epochs,
        device=device,
        lambda_sensor=args.lambda_sensor,
        lambda_window=args.lambda_window,
        lambda_aux_global=args.lambda_aux_global,
        learning_rate=args.lr,
        weight_decay=WEIGHT_DECAY,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_name=args.checkpoint_name,
        sensor_pos_weight_scale=args.sensor_pos_weight_scale,
        sensor_pos_weight_min=args.sensor_pos_weight_min,
        sensor_pos_weight_decay_epochs=args.sensor_pos_weight_decay_epochs,
        use_scheduled_sensor_pos_weight=not args.disable_sensor_pos_weight_schedule,
        sensor_pos_weight_cap=args.sensor_pos_weight_cap,
        use_compile=args.use_compile,
        compile_mode=args.compile_mode,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        use_amp=args.use_amp,
        max_batches_per_epoch=args.max_batches_per_epoch,
        hard_mining=args.hard_mining,
        hard_ratio=args.hard_ratio,
        hard_ratio_start=args.hard_ratio_start,
        hard_ratio_end=args.hard_ratio_end,
        hard_ratio_switch_epoch=args.hard_ratio_switch_epoch,
        hard_mining_seed=args.hard_mining_seed,
        hard_mining_start_epoch=args.hard_mining_start_epoch,
        window_score_weight=args.window_score_weight,
        window_pos_weight=args.window_pos_weight,
        use_fixed_train_threshold=True,
        train_window_thr=args.train_window_threshold,
        train_sensor_thr=args.train_sensor_threshold,
        calibrate_every=calibrate_every,
        freeze_backbone_epochs=args.freeze_backbone_epochs,
        use_calibrated_threshold_for_checkpoint=not args.no_calibrated_threshold_for_checkpoint,
        num_workers=args.num_workers,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        lr_min=args.lr_min,
    )

    print("✓ Stage 2 clean training complete!")


if __name__ == "__main__":
    main()
