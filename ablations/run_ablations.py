#!/usr/bin/env python3
"""
Run LSTM and ARIMA ablation baselines. Uses the same splits and shared dataset as GDN.
"""

import argparse
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ablations.data_loader import load_all_splits
from ablations.lstm_baseline import run_lstm
from ablations.arima_baseline import run_arima


def main():
    parser = argparse.ArgumentParser(
        description="Run LSTM and ARIMA ablation baselines (same splits as GDN)"
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/shared_dataset"),
        help="Path to shared dataset directory (train.npz, val.npz, test.npz)",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval", "train_and_eval"],
        default="train_and_eval",
        help="train: fit only; eval: load checkpoint and evaluate; train_and_eval: both",
    )
    parser.add_argument(
        "--baseline",
        choices=["lstm", "arima", "all"],
        default="all",
        help="Which baseline(s) to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Subsample test set for quick runs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for result JSON files",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for LSTM (cpu or cuda)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=75,
        help="LSTM training epochs (default 75)",
    )
    args = parser.parse_args()

    data_path = args.data_path
    if not data_path.is_absolute():
        data_path = project_root / data_path

    print("Loading shared dataset...")
    train_data, val_data, test_data = load_all_splits(data_path, limit_test=args.limit)
    if args.limit:
        print(f"  Test set limited to {args.limit} windows")
    print(f"  Train: {len(train_data['normalized_windows'])} windows")
    print(f"  Val:   {len(val_data['normalized_windows'])} windows")
    print(f"  Test:  {len(test_data['normalized_windows'])} windows")
    print()

    checkpoint_dir = args.checkpoint_dir
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = project_root / checkpoint_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    test_npz_path = str(data_path / "test.npz")
    do_train = args.mode in ("train", "train_and_eval")

    if args.baseline in ("lstm", "all"):
        print("=" * 60)
        print("LSTM Baseline")
        print("=" * 60)
        run_lstm(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            checkpoint_path=checkpoint_dir / "lstm_baseline.pt",
            output_path=output_dir / "lstm_baseline.json",
            train=do_train,
            device=args.device,
            dataset_path=test_npz_path,
            epochs=args.epochs,
        )
        print()

    if args.baseline in ("arima", "all"):
        print("=" * 60)
        print("ARIMA Baseline")
        print("=" * 60)
        run_arima(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            checkpoint_path=checkpoint_dir / "arima_baseline.pkl",
            output_path=output_dir / "arima_baseline.json",
            train=do_train,
            dataset_path=test_npz_path,
        )
        print()

    print("Done.")


if __name__ == "__main__":
    main()
