#!/usr/bin/env python3
"""
Compare GDN, LSTM, and ARIMA results on the full test set, and plot LLM-based
methods (GDN+KG+LLM vs LLM-only) when those result JSONs are present.

Also: ``python ablations/compare_results.py diagnose`` — binary FPR on normal
windows and sensor-level false positives on the stratified slice (see plan).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

RESULTS_DIR = project_root / "results"
FIGURES_DIR = project_root / "figures"


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def load_all_rows():
    files = {
        "GDN": RESULTS_DIR / "gdn_only.json",
        "LSTM": RESULTS_DIR / "lstm_baseline.json",
        "ARIMA": RESULTS_DIR / "arima_baseline.json",
    }
    rows = []
    for name, path in files.items():
        if not path.exists():
            continue
        d = load_metrics(path)
        m = d["metrics"]
        wl = m["window_level"]
        sl = m["sensor_level"]
        rows.append({
            "method": name,
            "num_windows": d["num_windows"],
            "window_acc": wl["window_accuracy"],
            "window_prec": wl["window_precision"],
            "window_rec": wl["window_recall"],
            "window_f1": wl["window_f1"],
            "sensor_acc": sl["sensor_accuracy"],
            "sensor_prec": sl["sensor_precision"],
            "sensor_rec": sl["sensor_recall"],
            "sensor_f1": sl["sensor_f1"],
        })
    return rows


def load_llm_comparison_rows():
    files = [
        ("GDN+KG+LLM", RESULTS_DIR / "gdn_kg_llm.json"),
        ("LLM only", RESULTS_DIR / "llm_baseline.json"),
    ]
    rows = []
    for name, path in files:
        if not path.exists():
            continue
        d = load_metrics(path)
        m = d["metrics"]
        wl = m["window_level"]
        sl = m["sensor_level"]
        rows.append({
            "method": name,
            "num_windows": d["num_windows"],
            "window_acc": wl["window_accuracy"],
            "window_prec": wl["window_precision"],
            "window_rec": wl["window_recall"],
            "window_f1": wl["window_f1"],
            "sensor_acc": sl["sensor_accuracy"],
            "sensor_prec": sl["sensor_precision"],
            "sensor_rec": sl["sensor_recall"],
            "sensor_f1": sl["sensor_f1"],
        })
    return rows


def save_llm_bar_chart():
    rows = load_llm_comparison_rows()
    if len(rows) < 2:
        return

    methods = [r["method"] for r in rows]
    metrics = ["Window F1", "Window Acc", "Sensor F1", "Sensor Acc"]
    keys = ["window_f1", "window_acc", "sensor_f1", "sensor_acc"]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, r in enumerate(rows):
        vals = [r[k] * 100 for k in keys]
        offset = width * (i - 0.5)
        ax.bar(x + offset, vals, width, label=r["method"])

    nwin = rows[0]["num_windows"]
    ax.set_ylabel("Score (%)")
    ax.set_title(f"GDN+KG+LLM vs LLM only ({nwin} windows)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = FIGURES_DIR / "llm_method_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_bar_chart():
    rows = load_all_rows()
    if not rows:
        return

    methods = [r["method"] for r in rows]
    metrics = ["Window F1", "Window Acc", "Sensor F1", "Sensor Acc"]
    keys = ["window_f1", "window_acc", "sensor_f1", "sensor_acc"]

    x = np.arange(len(metrics))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, r in enumerate(rows):
        vals = [r[k] * 100 for k in keys]
        offset = width * (i - len(rows) / 2 + 0.5)
        ax.bar(x + offset, vals, width, label=r["method"])

    ax.set_ylabel("Score (%)")
    ax.set_title("Model Comparison (full test set, 1549 windows)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'model_comparison.png'}")


def _truth_sensor_labels(npz_path: Path, sample_indices: Optional[List[int]]) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    sl = np.asarray(data["sensor_labels"], dtype=np.float32)
    if sample_indices is not None:
        ix = np.array(sample_indices, dtype=np.int64)
        sl = sl[ix]
    return sl


def _pred_faulty_from_window_labels(window_labels: np.ndarray) -> np.ndarray:
    return np.array(window_labels, dtype=np.int64) > 0


def _normal_window_sensor_fp_rate(sensor_pred: np.ndarray, normal_mask: np.ndarray) -> tuple[float, float, float]:
    """Returns (fraction of normal windows with any predicted faulty sensor, mean pred sensors per normal row, total slots FP rate)."""
    if not np.any(normal_mask):
        return float("nan"), float("nan"), float("nan")
    sub = np.asarray(sensor_pred, dtype=np.float32)[normal_mask]
    any_fp = np.any(sub > 0, axis=1)
    frac_windows = float(np.mean(any_fp))
    mean_sensors = float(np.mean(sub.sum(axis=1)))
    flat_fp = float(sub.sum() / sub.size)
    return frac_windows, mean_sensors, flat_fp


def main_diagnose(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose GDN vs LLM on the same stratified slice (binary FPR + sensor noise on normals).",
        prog="compare_results.py diagnose",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "data/shared_dataset/test.npz",
        help="Path to test .npz (must contain sensor_labels)",
    )
    parser.add_argument(
        "--kg-json",
        type=Path,
        default=RESULTS_DIR / "gdn_kg_llm.json",
        help="GDN+KG+LLM results JSON (source of sample_indices if present)",
    )
    parser.add_argument(
        "--gdn-json",
        type=Path,
        default=None,
        help="gdn_only results JSON (optional; default tries results/gdn_only_200_strat.json then gdn_only.json)",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="llm_baseline JSON (optional; default tries llm_baseline_200_strat.json then llm_baseline.json)",
    )
    parser.add_argument(
        "--audit",
        type=int,
        default=10,
        help="Max normal windows where KG predicts fault to print reasoning snippets for",
    )
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        print(f"ERROR: dataset not found: {args.dataset}")
        print("Run on machine with data, or pass --dataset.")
        return 1

    if not args.kg_json.exists():
        print(f"ERROR: KG JSON not found: {args.kg_json}")
        return 1

    kg = load_metrics(args.kg_json)
    sample_indices = kg.get("sample_indices")

    sl_true = _truth_sensor_labels(args.dataset, sample_indices)
    true_faulty = sl_true.sum(axis=1) > 0
    normal_mask = ~true_faulty
    n_normal = int(normal_mask.sum())
    n_faulty = int(true_faulty.sum())

    print("=" * 80)
    print("Diagnose: stratified slice vs ground truth (sensor_labels)")
    print("=" * 80)
    print(f"Dataset: {args.dataset}")
    print(f"KG JSON: {args.kg_json}")
    if sample_indices is not None:
        print(f"sample_indices: {len(sample_indices)} windows (stratified/reordered)")
    else:
        print("sample_indices: absent — assuming predictions align with dataset row order")
    print(f"Windows in slice: {len(sl_true)} (normal={n_normal}, faulty={n_faulty})")
    print()

    gdn_path = args.gdn_json
    if gdn_path is None:
        for cand in (
            RESULTS_DIR / "gdn_only_200_strat.json",
            RESULTS_DIR / "gdn_only.json",
        ):
            if cand.exists():
                gdn_path = cand
                break
    bl_path = args.baseline_json
    if bl_path is None:
        for cand in (
            RESULTS_DIR / "llm_baseline_200_strat.json",
            RESULTS_DIR / "llm_baseline.json",
        ):
            if cand.exists():
                bl_path = cand
                break

    gdn_wl: Optional[np.ndarray] = None
    gdn_sl: Optional[np.ndarray] = None
    if gdn_path is not None and gdn_path.exists():
        gdn_doc = load_metrics(gdn_path)
        si_g = gdn_doc.get("sample_indices")
        if sample_indices is not None and si_g is not None:
            if sample_indices != si_g:
                print("WARNING: sample_indices differ between KG JSON and GDN JSON — comparisons are invalid.")
            else:
                print(f"OK: sample_indices match KG and GDN ({gdn_path.name}).")
        elif sample_indices is not None and si_g is None:
            print(
                f"WARNING: GDN JSON has no sample_indices ({gdn_path.name}); ensure same slice before trusting comparison."
            )
        gdn_wl = np.array(gdn_doc["predictions"]["window_labels"], dtype=np.int64)
        gdn_sl = np.array(gdn_doc["predictions"]["sensor_labels"], dtype=np.float32)
        print()

    def row(name: str, pred_faulty: np.ndarray) -> None:
        if n_normal == 0:
            fpr = float("nan")
        else:
            fpr = float(np.mean(pred_faulty[normal_mask]))
        if n_faulty == 0:
            tpr_faulty = float("nan")
        else:
            tpr_faulty = float(np.mean(pred_faulty[true_faulty]))
        print(
            f"{name:<22}  FPR_norm={fpr:.4f}  TPR_faulty={tpr_faulty:.4f}  "
            f"(pred_fault|normal)={int(np.sum(pred_faulty[normal_mask]))}/{n_normal}  "
            f"(pred_fault|faulty)={int(np.sum(pred_faulty[true_faulty]))}/{n_faulty}"
        )

    print("--- Binary fault (pred = window_label > 0) ---")
    print("  FPR_norm = false-positive rate on truly normal windows; TPR_faulty = recall on faulty windows.")
    kg_wl = np.array(kg["predictions"]["window_labels"], dtype=np.int64)
    kg_sl = np.array(kg["predictions"]["sensor_labels"], dtype=np.float32)
    row("GDN+KG+LLM", _pred_faulty_from_window_labels(kg_wl))

    if gdn_wl is not None:
        row("GDN only", _pred_faulty_from_window_labels(gdn_wl))
    else:
        print(
            "GDN only              (skip — no JSON; run: python llm/evaluation/evaluate_gdn_kg_llm.py "
            "--mode gdn_only --limit 200 --stratified-limit-seed 42 --model-path ... "
            f"--output {RESULTS_DIR / 'gdn_only_200_strat.json'})"
        )

    bl_sl: Optional[np.ndarray] = None
    if bl_path is not None and bl_path.exists():
        bl = load_metrics(bl_path)
        bl_wl = np.array(bl["predictions"]["window_labels"], dtype=np.int64)
        bl_sl = np.array(bl["predictions"]["sensor_labels"], dtype=np.float32)
        bl_pf = _pred_faulty_from_window_labels(bl_wl)
        row("LLM baseline", bl_pf)
        if n_faulty > 0 and float(np.mean(bl_pf[true_faulty])) < 0.05:
            print(
                "  → Baseline TPR_faulty≈0: model almost never predicts a fault; low FPR_norm can be trivial (always-normal policy)."
            )
    else:
        print(
            "LLM baseline          (skip — no JSON; run: python llm/evaluation/evaluate_llm_baseline.py "
            f"--limit 200 --seed 42 --output {RESULTS_DIR / 'llm_baseline_200_strat.json'})"
        )

    print()
    print("--- Normal windows: sensor-level false positives ---")
    w1, m1, flat1 = _normal_window_sensor_fp_rate(kg_sl, normal_mask)
    print(
        f"GDN+KG+LLM  frac_normal_with_any_pred_sensor={w1:.4f}  "
        f"mean_pred_sensors_per_normal_row={m1:.4f}  flat_FP_rate={flat1:.4f}"
    )
    if gdn_sl is not None:
        w2, m2, flat2 = _normal_window_sensor_fp_rate(gdn_sl, normal_mask)
        print(
            f"GDN only    frac_normal_with_any_pred_sensor={w2:.4f}  "
            f"mean_pred_sensors_per_normal_row={m2:.4f}  flat_FP_rate={flat2:.4f}"
        )
    if bl_sl is not None:
        wb, mb, flatb = _normal_window_sensor_fp_rate(bl_sl, normal_mask)
        print(
            f"LLM baseline frac_normal_with_any_pred_sensor={wb:.4f}  "
            f"mean_pred_sensors_per_normal_row={mb:.4f}  flat_FP_rate={flatb:.4f}"
        )

    print()
    print("--- Qualitative audit (normal truth, KG predicts fault) ---")
    kg_pf = _pred_faulty_from_window_labels(kg_wl)
    disagree = np.where(normal_mask & kg_pf)[0]
    reasoning = kg["predictions"].get("reasoning") or []
    for k, i in enumerate(disagree[: max(0, args.audit)]):
        r = reasoning[i] if i < len(reasoning) else ""
        snippet = (r[:280] + "…") if len(r) > 280 else r
        ds_ix = sample_indices[i] if sample_indices is not None else i
        print(f"  slice_row={i}  dataset_index={ds_ix}")
        print(f"    reasoning: {snippet!r}")
        if gdn_wl is not None and gdn_sl is not None:
            gdn_pf_row = int(gdn_wl[i]) > 0
            gdn_any_s = bool(np.any(gdn_sl[i] > 0))
            print(f"    GDN_only pred_fault_window={gdn_pf_row}  any_GDN_sensor={gdn_any_s}")
    if len(disagree) == 0:
        print("  (none)")

    print()
    print("Interpretation (short):")
    print("  High GDN FPR_norm + high KG FPR_norm → GDN/thresholds/violations likely drive errors.")
    print("  Low GDN FPR_norm + high KG FPR_norm → prompt/LLM policy, parsing, or KG context likely primary.")
    print("  Baseline FPR_norm≈0 with TPR_faulty≈0 → not comparable to KG+LLM; model is effectively always-normal.")
    print("=" * 80)
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "diagnose":
        raise SystemExit(main_diagnose(sys.argv[2:]))

    rows = load_all_rows()
    if not rows:
        print("No result files found.")
        return

    print("=" * 80)
    print("Model Comparison (full test set, 1549 windows)")
    print("=" * 80)
    print()
    print(f"{'Method':<10} {'Win Acc':>8} {'Win Prec':>8} {'Win Rec':>8} {'Win F1':>8}  {'Sen Acc':>8} {'Sen Prec':>8} {'Sen Rec':>8} {'Sen F1':>8}")
    print("-" * 80)
    for r in rows:
        print(f"{r['method']:<10} {r['window_acc']:>8.2%} {r['window_prec']:>8.2%} {r['window_rec']:>8.2%} {r['window_f1']:>8.2%}  {r['sensor_acc']:>8.2%} {r['sensor_prec']:>8.2%} {r['sensor_rec']:>8.2%} {r['sensor_f1']:>8.2%}")
    print()
    print("Win = Window-level, Sen = Sensor-level")
    print()

    save_bar_chart()
    save_llm_bar_chart()


if __name__ == "__main__":
    main()
