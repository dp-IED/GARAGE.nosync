#!/usr/bin/env python3
"""
Compare result JSONs under results/ and write figures under figures/.

Model comparison chart: GDN (from evaluate_gdn_kg_llm.py --mode gdn_only), LSTM, ARIMA when present.
Neural/LLM pipeline chart: GDN-only + GDN+KG+LLM + LLM-only **only if** JSONs share the same
sample_indices (same --limit and stratified seed as in evaluate_llm_baseline.py / evaluate_gdn_kg_llm.py).
A stratified --limit of 200 and 200 are different window sets than --limit 400, so LLM scores are
not comparable across those runs — do not plot them on one bar chart.

Also: ``python ablations/compare_results.py diagnose`` — binary FPR on normal windows and
sensor-level false positives on the stratified slice (see plan).
"""

import argparse
import json
from pathlib import Path
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

RESULTS_DIR = project_root / "results"
FIGURES_DIR = project_root / "figures"


def load_metrics(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def same_eval_slice(doc_a: dict, doc_b: dict) -> bool:
    """True if both JSONs refer to the same ordered set of dataset rows."""
    si_a = doc_a.get("sample_indices")
    si_b = doc_b.get("sample_indices")
    if si_a is not None and si_b is not None:
        return list(si_a) == list(si_b)
    if si_a is None and si_b is None:
        return int(doc_a["num_windows"]) == int(doc_b["num_windows"])
    return False


def find_matching_gdn_only_json(anchor: dict) -> Optional[Path]:
    """GDN-only result whose sample_indices match anchor (e.g. gdn_kg_llm.json)."""
    n = int(anchor["num_windows"])
    candidates = [
        RESULTS_DIR / f"gdn_only_{n}_strat.json",
        RESULTS_DIR / "gdn_only_400_strat.json",
        RESULTS_DIR / "gdn_only_200_strat.json",
        RESULTS_DIR / "gdn_only.json",
    ]
    tried: set[Path] = set()
    for path in candidates:
        if path in tried or not path.exists():
            continue
        tried.add(path)
        d = load_metrics(path)
        if d.get("method") != "gdn_only":
            continue
        if same_eval_slice(anchor, d):
            return path
    return None


def find_matching_llm_baseline_json(anchor: dict) -> Optional[Path]:
    for path in sorted(RESULTS_DIR.glob("llm_baseline*.json")):
        d = load_metrics(path)
        if d.get("method") != "llm_baseline":
            continue
        if same_eval_slice(anchor, d):
            return path
    return None


def load_all_rows():
    """GDN + LSTM + ARIMA for broad comparison (window counts may differ per file)."""
    rows = []
    gdn_candidates = []
    if (RESULTS_DIR / "gdn_kg_llm.json").exists():
        n = int(load_metrics(RESULTS_DIR / "gdn_kg_llm.json")["num_windows"])
        gdn_candidates.append(RESULTS_DIR / f"gdn_only_{n}_strat.json")
    gdn_candidates.extend(
        [
            RESULTS_DIR / "gdn_only_400_strat.json",
            RESULTS_DIR / "gdn_only_200_strat.json",
            RESULTS_DIR / "gdn_only.json",
        ]
    )
    seen_paths: set[Path] = set()
    for path in gdn_candidates:
        if path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        d = load_metrics(path)
        if d.get("method") != "gdn_only":
            continue
        rows.append(_metrics_row("GDN", d))
        break

    for name, path in (
        ("LSTM", RESULTS_DIR / "lstm_baseline.json"),
        ("ARIMA", RESULTS_DIR / "arima_baseline.json"),
    ):
        if not path.exists():
            continue
        rows.append(_metrics_row(name, load_metrics(path)))
    return rows


def _metrics_row(name: str, d: dict) -> dict:
    m = d["metrics"]
    wl = m["window_level"]
    sl = m["sensor_level"]
    return {
        "method": name,
        "num_windows": d["num_windows"],
        "stratified": bool(d.get("stratified_limit")),
        "window_acc": wl["window_accuracy"],
        "window_prec": wl["window_precision"],
        "window_rec": wl["window_recall"],
        "window_f1": wl["window_f1"],
        "sensor_acc": sl["sensor_accuracy"],
        "sensor_prec": sl["sensor_precision"],
        "sensor_rec": sl["sensor_recall"],
        "sensor_f1": sl["sensor_f1"],
    }


def load_pipeline_figure_rows() -> Tuple[list, list]:
    """
    Rows for llm_method_comparison.png: GDN-only, GDN+KG+LLM, LLM-only on the **same** eval slice
    (matching sample_indices). Second return value is human-readable notes / missing-file hints.
    """
    notes: list = []
    kg_path = RESULTS_DIR / "gdn_kg_llm.json"
    if not kg_path.exists():
        return [], ["No results/gdn_kg_llm.json — run evaluate_gdn_kg_llm.py (full mode)."]

    kg = load_metrics(kg_path)
    rows: list = []

    gdn_path = find_matching_gdn_only_json(kg)
    if gdn_path is not None:
        rows.append(_metrics_row("GDN only", load_metrics(gdn_path)))
    else:
        n = int(kg["num_windows"])
        seed = kg.get("stratified_limit_seed", 42)
        notes.append(
            "No matching GDN-only JSON for this slice. Run: "
            f"python llm/evaluation/evaluate_gdn_kg_llm.py --mode gdn_only --limit {n} "
            f"--stratified-limit-seed {seed} --model-path ... "
            f"--output results/gdn_only_{n}_strat.json"
        )

    rows.append(_metrics_row("GDN+KG+LLM", kg))

    llm_path = find_matching_llm_baseline_json(kg)
    if llm_path is not None:
        rows.append(_metrics_row("LLM only", load_metrics(llm_path)))
    else:
        notes.append(
            "No llm_baseline*.json matches gdn_kg_llm sample_indices. Run evaluate_llm_baseline.py "
            f"with the same --limit and --seed as KG eval (seed {kg.get('stratified_limit_seed', 42)})."
        )

    order = {"GDN only": 0, "GDN+KG+LLM": 1, "LLM only": 2}
    rows.sort(key=lambda r: order.get(r["method"], 9))
    return rows, notes


def _plot_metric_bars(rows: list, title: str, out_name: str, figsize: tuple = (8, 5)) -> None:
    if not rows:
        return
    metrics = ["Window F1", "Window Acc", "Sensor F1", "Sensor Acc"]
    keys = ["window_f1", "window_acc", "sensor_f1", "sensor_acc"]
    x = np.arange(len(metrics))
    n = len(rows)
    width = min(0.35, 0.8 / max(n, 1))
    fig, ax = plt.subplots(figsize=figsize)
    for i, r in enumerate(rows):
        vals = [r[k] * 100 for k in keys]
        offset = width * (i - (n - 1) / 2)
        ax.bar(x + offset, vals, width, label=r["method"])
    ax.set_ylabel("Score (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    out = FIGURES_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_llm_bar_chart():
    rows, notes = load_pipeline_figure_rows()
    for msg in notes:
        print(f"Pipeline chart: {msg}")
    if len(rows) < 2:
        print("Skip llm_method_comparison.png: need gdn_kg_llm.json plus at least one matching series.")
        return
    kg = load_metrics(RESULTS_DIR / "gdn_kg_llm.json")
    nwin = int(kg["num_windows"])
    seed = kg.get("stratified_limit_seed", 42)
    strat = bool(kg.get("stratified_limit"))
    suffix = f"{nwin} windows"
    if strat:
        suffix += f", stratified (seed {seed})"
    _plot_metric_bars(
        rows,
        f"Same eval slice — GDN / KG+LLM / LLM ({suffix}; sample_indices must match)",
        "llm_method_comparison.png",
        figsize=(9, 5),
    )


def save_overview_two_panel():
    """Classical baselines vs same-slice neural/LLM pipeline."""
    classical = load_all_rows()
    llm_rows, _ = load_pipeline_figure_rows()
    if not classical and len(llm_rows) < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["Win F1", "Win Acc", "Sen F1", "Sen Acc"]
    keys = ["window_f1", "window_acc", "sensor_f1", "sensor_acc"]

    if classical:
        ax = axes[0]
        x = np.arange(len(metrics))
        w = min(0.25, 0.7 / len(classical))
        for i, r in enumerate(classical):
            vals = [r[k] * 100 for k in keys]
            off = w * (i - (len(classical) - 1) / 2)
            ax.bar(x + off, vals, w, label=r["method"])
        ax.set_ylabel("Score (%)")
        detail = ", ".join(f"{r['method']} n={r['num_windows']}" for r in classical)
        ax.set_title(f"Baselines ({detail})")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    else:
        axes[0].set_visible(False)

    if len(llm_rows) >= 2:
        ax = axes[1]
        x = np.arange(len(metrics))
        n_b = len(llm_rows)
        w = min(0.28, 0.75 / max(n_b, 1))
        for i, r in enumerate(llm_rows):
            vals = [r[k] * 100 for k in keys]
            off = w * (i - (n_b - 1) / 2)
            ax.bar(x + off, vals, w, label=r["method"])
        ax.set_ylabel("Score (%)")
        nwin = llm_rows[0]["num_windows"]
        ax.set_title(f"Same slice: GDN / KG+LLM / LLM (n={nwin})")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    else:
        axes[1].set_visible(False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.suptitle("GARAGE — latest eval summary", fontsize=12, y=1.02)
    fig.tight_layout()
    out = FIGURES_DIR / "results_overview_panels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def save_bar_chart():
    rows = load_all_rows()
    if not rows:
        return

    metrics = ["Window F1", "Window Acc", "Sensor F1", "Sensor Acc"]
    keys = ["window_f1", "window_acc", "sensor_f1", "sensor_acc"]

    x = np.arange(len(metrics))
    n_r = len(rows)
    width = min(0.22, 0.72 / max(n_r, 1))
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, r in enumerate(rows):
        vals = [r[k] * 100 for k in keys]
        offset = width * (i - (n_r - 1) / 2)
        ax.bar(x + offset, vals, width, label=f"{r['method']} (n={r['num_windows']})")

    ax.set_ylabel("Score (%)")
    ax.set_title("Model comparison — GDN / LSTM / ARIMA (n = eval windows per series)")
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
        help="gdn_only results JSON (optional; default: file matching gdn_kg_llm sample_indices, else first gdn_only*.json)",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="llm_baseline JSON (optional; default: llm_baseline*.json matching KG slice, else first match)",
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
    n_slice = int(kg["num_windows"])
    seed_eval = kg.get("stratified_limit_seed", 42)
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
        gdn_path = find_matching_gdn_only_json(kg)
        if gdn_path is None:
            for cand in (
                RESULTS_DIR / f"gdn_only_{kg['num_windows']}_strat.json",
                RESULTS_DIR / "gdn_only_400_strat.json",
                RESULTS_DIR / "gdn_only_200_strat.json",
                RESULTS_DIR / "gdn_only.json",
            ):
                if cand.exists():
                    gdn_path = cand
                    break
    bl_path = args.baseline_json
    if bl_path is None:
        bl_path = find_matching_llm_baseline_json(kg)
        if bl_path is None:
            for cand in sorted(RESULTS_DIR.glob("llm_baseline*.json")):
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
            f"--mode gdn_only --limit {n_slice} --stratified-limit-seed {seed_eval} --model-path ... "
            f"--output {RESULTS_DIR / f'gdn_only_{n_slice}_strat.json'})"
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
            f"--limit {n_slice} --seed {seed_eval} --output {RESULTS_DIR / 'llm_baseline.json'})"
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
    print("Model comparison table: GDN (if present) + LSTM + ARIMA — per-file window counts")
    print("=" * 80)
    print()
    print(f"{'Method':<16} {'Win Acc':>8} {'Win Prec':>8} {'Win Rec':>8} {'Win F1':>8}  {'Sen Acc':>8} {'Sen Prec':>8} {'Sen Rec':>8} {'Sen F1':>8}")
    print("-" * 96)
    for r in rows:
        print(
            f"{r['method']:<16} {r['window_acc']:>8.2%} {r['window_prec']:>8.2%} {r['window_rec']:>8.2%} {r['window_f1']:>8.2%}  "
            f"{r['sensor_acc']:>8.2%} {r['sensor_prec']:>8.2%} {r['sensor_rec']:>8.2%} {r['sensor_f1']:>8.2%}"
        )
    print()
    print("Win = Window-level, Sen = Sensor-level")
    print()

    save_bar_chart()
    save_llm_bar_chart()
    save_overview_two_panel()

    print()
    print("=" * 80)
    print("Latest results snapshot (JSON in results/)")
    print("=" * 80)
    if (RESULTS_DIR / "gdn_kg_llm.json").exists():
        kg_doc = load_metrics(RESULTS_DIR / "gdn_kg_llm.json")
        gdn_m = find_matching_gdn_only_json(kg_doc)
        llm_m = find_matching_llm_baseline_json(kg_doc)
        print("\nSame-slice trio (matching sample_indices to gdn_kg_llm.json):")
        for label, p in (
            ("GDN+KG+LLM", RESULTS_DIR / "gdn_kg_llm.json"),
            ("GDN only", gdn_m),
            ("LLM only", llm_m),
        ):
            if p is None or not Path(p).exists():
                print(f"  {label:<14}  (no matching JSON)")
                continue
            d = load_metrics(p)
            wl = d["metrics"]["window_level"]
            sl = d["metrics"]["sensor_level"]
            print(
                f"  {label:<14}  n={d['num_windows']:<4}  "
                f"win_acc={wl['window_accuracy']:.2%}  win_f1={wl['window_f1']:.2%}  "
                f"sen_acc={sl['sensor_accuracy']:.2%}  sen_f1={sl['sensor_f1']:.2%}  ({p.name})"
            )
    for solo, label in (
        (RESULTS_DIR / "lstm_baseline.json", "LSTM full test"),
        (RESULTS_DIR / "arima_baseline.json", "ARIMA (subset)"),
    ):
        if solo.exists():
            d = load_metrics(solo)
            wl = d["metrics"]["window_level"]
            sl = d["metrics"]["sensor_level"]
            print(
                f"\n{label}:  n={d['num_windows']}  "
                f"win_acc={wl['window_accuracy']:.2%}  win_f1={wl['window_f1']:.2%}  "
                f"sen_acc={sl['sensor_accuracy']:.2%}  sen_f1={sl['sensor_f1']:.2%}  ({solo.name})"
            )
    print()


if __name__ == "__main__":
    main()
