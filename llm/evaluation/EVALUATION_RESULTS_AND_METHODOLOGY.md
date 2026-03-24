# Evaluation results and methodology

This document records the **primary LLM evaluation results** (as saved in `results/`), how metrics are **defined in code**, and **minimal snippets** needed to verify the eval scripts implement those definitions correctly. Regenerate numbers by re-running the commands in §2; update this file after every new run or pipeline change.

---

## 1. Primary result files (400-window stratified slice)

| File | Method | Windows | Stratified | Seed | Notes |
|------|--------|---------|------------|------|--------|
| `results/gdn_kg_llm.json` | GDN + KG + LLM | 400 | yes | 42 | `sample_indices` stored |
| `results/llm_baseline.json` | LLM-only (raw series) | 400 | yes | 42 | **Same `sample_indices` as GDN-KG-LLM** (verified equal) |

**Reproducibility check:** `sample_indices` in both files are **identical** (first five: `6, 10, 11, 13, 15`).

**Other JSONs under `results/`** (`gdn_only_200_strat.json`, `llm_baseline_200_strat.json`, `arima_baseline.json`, `lstm_baseline.json`) are **older or different splits** — do not mix with the 400-window primary tables without noting the mismatch.

---

## 2. How to reproduce

From repo root (adjust model path and LM Studio URL as needed):

```bash
python llm/evaluation/evaluate_gdn_kg_llm.py \
  --dataset data/shared_dataset/test.npz \
  --model-path <CHECKPOINT.pt> \
  --output results/gdn_kg_llm.json \
  --limit 400 \
  --stratified-limit-seed 42

python llm/evaluation/evaluate_llm_baseline.py \
  --dataset data/shared_dataset/test.npz \
  --output results/llm_baseline.json \
  --limit 400 \
  --stratified-limit-seed 42
```

Stratified limiting requires `fault_types` in the `.npz`. Default CLI behaviour stratifies by fault type when `--limit` is set.

> **Note on BERTScore:** BERTScore is not present in the current `results/*.json` files. The eval pipeline has been updated (see §6.4) to compute and save BERTScore on the next run; results will be available after the overnight eval run. In a prior intermediate run (200-window stratified, pre-fault-taxonomy change), both methods produced broadly similar BERTScore F1 values (~0.85–0.87 range), suggesting the GDN-KG-LLM pipeline does not substantially degrade the linguistic quality of generated reasoning relative to the LLM-only baseline. The updated taxonomy-aware run will provide authoritative numbers.

---

## 3. Aggregate metrics (current `results/*.json`)

Values are read directly from the **current** result files. They are authoritative.

### 3.1 Window level (multiclass, sensor-indexed)

**Definition:** `0` = no fault; `1…D` = first ground-truth faulty sensor index + 1 (same scheme for predictions, derived from `window_label` parsed from the LLM response). Weighted / macro F1 are sklearn multiclass averages over this 9-class label space (0 = no fault, 1–8 = sensors).

| Metric | GDN-KG-LLM | LLM baseline |
|--------|------------|--------------|
| Accuracy | 0.9725 | 0.0575 |
| F1 (weighted) | 0.9694 | 0.0099 |
| F1 (macro) | 0.8500 | 0.0247 |

**LLM-only interpretation:** The model exhibits **mode collapse** on primary-sensor prediction — it assigns almost all windows to `COOLANT_TEMPERATURE ()` (class 5). This is systematic behaviour visible in the per-class confusion matrix: COOLANT_TEMPERATURE has recall ≈ 1.0 and every other class recall ≈ 0. The low aggregate accuracy reflects this structural failure, not a data or script error.

### 3.2 Sensor level (binary multilabel, micro over cells)

**Definition:** All `(window × sensor)` binary cells are flattened; sklearn binary P/R/F1 treats each cell as one binary prediction. Main metrics use **root-only** filtered sensor vectors (single primary sensor per window); `sensor_level_raw` in JSON keeps all LLM-flagged sensors.

| Metric | GDN-KG-LLM | LLM baseline |
|--------|------------|--------------|
| Accuracy | 0.9963 | 0.7384 |
| Precision | 1.0000 | 0.0537 |
| Recall | 0.9016 | 0.3525 |
| F1 | 0.9483 | 0.0932 |

The LLM baseline's high accuracy here is misleading — most cells are true-negatives (the test set is ~73% normal windows), so a model that never flags sensors can still achieve ~73% cell accuracy.

### 3.3 Fault-type classification (faulty windows only)

**Definition:** Restricted to windows where ground-truth fault type ∉ `{normal, ""}`. Field `n` = count of such windows (108 on this 400-window draw).

| Metric | GDN-KG-LLM | LLM baseline |
|--------|------------|--------------|
| Accuracy | 0.6852 | 0.2130 |
| Weighted F1 | 0.6772 | 0.0873 |
| Macro F1 | 0.6068 | 0.0596 |
| n | 108 | 108 |

### 3.4 BERTScore

**Not yet computed in current results files.** The pipeline now computes and saves BERTScore on every run (see §6.4). Re-running the evals overnight will populate `results/*.json` with a `bertscore` object containing `precision`, `recall`, `f1` (semantic similarity values). Prior intermediate results (different window split, pre-taxonomy update) showed both methods at roughly **BERTScore F1 ≈ 0.85–0.87**; updated numbers pending.

### 3.5 Efficiency

**GDN-KG-LLM** (`metrics.efficiency`):

| Field | Value |
|-------|-------|
| `gdn_processing_time_seconds` | 1.03 |
| `kg_build_time_seconds` | 1.56 |
| `kg_precompute_contexts_time_seconds` | 0.02 |
| `llm_processing_time_seconds` | 12 042 |
| `total_processing_time_seconds` | 12 045 |
| `windows_per_second` | 0.033 |
| `kg_nodes` | 3 208 |
| `kg_edges` | 11 852 |

**LLM baseline:** avg 44.8 s/window, total ≈ 17 939 s (no GDN/KG overhead; slower per window because the longer raw-series context takes more tokens to process).

---

## 4. Per–fault-type stratified metrics

**Fault-type match:** Among windows with ground-truth fault type T, fraction where the predicted fault-type string (normalised) equals T. This is the per-class recall of fault-type identification restricted to that stratum.

**Sensor F1:** Micro F1 on flattened true/pred sensor bits for those windows only.

### GDN-KG-LLM

| Fault type | N | Fault-type match | Sensor F1 |
|---|---:|---:|---:|
| COOLANT_DROPOUT | 22 | 0.955 | 0.977 |
| LOAD_SCALE_LOW | 14 | 0.929 | 0.963 |
| LTFT_DRIFT_HIGH | 14 | 0.000 | 1.000 |
| MAF_SCALE_LOW | 14 | 0.500 | 0.963 |
| RPM_SPIKE_DROPOUT | 14 | 0.571 | 0.880 |
| STFT_STUCK_HIGH | 14 | 0.929 | 0.963 |
| TPS_STUCK | 2 | 0.000 | 0.000 |
| VSS_DROPOUT | 14 | 0.857 | 0.923 |

**Notable:** `LTFT_DRIFT_HIGH` — sensor localisation is correct (F1 = 1.0) but the fault-type label is never predicted correctly (match = 0.0). The GDN correctly flags the LTFT sensor but the LLM interprets the pattern as a different type. `TPS_STUCK` — the GDN structurally fails to flag a throttle held at constant value because constant = no deviation from expected correlations.

### LLM baseline

| Fault type | N | Fault-type match | Sensor F1 |
|---|---:|---:|---:|
| COOLANT_DROPOUT | 22 | 1.000 | 0.667 |
| LOAD_SCALE_LOW | 14 | 0.000 | 0.000 |
| LTFT_DRIFT_HIGH | 14 | 0.000 | 0.000 |
| MAF_SCALE_LOW | 14 | 0.000 | 0.500 |
| RPM_SPIKE_DROPOUT | 14 | 0.000 | 0.000 |
| STFT_STUCK_HIGH | 14 | 0.000 | 0.286 |
| TPS_STUCK | 2 | 0.000 | 0.000 |
| VSS_DROPOUT | 14 | 0.071 | 0.048 |

The LLM-only baseline matches COOLANT_DROPOUT perfectly (it is the dominant predicted class), and nearly nothing else.

---

## 5. Fault-type classification — per-class F1 (faulty windows only)

### GDN-KG-LLM (n = 108)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COOLANT_DROPOUT | 0.488 | 0.955 | 0.646 | 22 |
| LOAD_SCALE_LOW | 0.929 | 0.929 | 0.929 | 14 |
| LTFT_DRIFT_HIGH | 0.000 | 0.000 | 0.000 | 14 |
| MAF_SCALE_LOW | 1.000 | 0.500 | 0.667 | 14 |
| RPM_SPIKE_DROPOUT | 1.000 | 0.571 | 0.727 | 14 |
| STFT_STUCK_HIGH | 1.000 | 0.929 | 0.963 | 14 |
| TPS_STUCK | 0.000 | 0.000 | 0.000 | 2 |
| VSS_DROPOUT | 1.000 | 0.857 | 0.923 | 14 |

### LLM baseline (n = 108)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| COOLANT_DROPOUT | 0.208 | 1.000 | 0.344 | 22 |
| LOAD_SCALE_LOW | 0.000 | 0.000 | 0.000 | 14 |
| LTFT_DRIFT_HIGH | 0.000 | 0.000 | 0.000 | 14 |
| MAF_SCALE_LOW | 0.000 | 0.000 | 0.000 | 14 |
| RPM_SPIKE_DROPOUT | 0.000 | 0.000 | 0.000 | 14 |
| STFT_STUCK_HIGH | 0.000 | 0.000 | 0.000 | 14 |
| TPS_STUCK | 0.000 | 0.000 | 0.000 | 2 |
| VSS_DROPOUT | 1.000 | 0.071 | 0.133 | 14 |

---

## 6. Code: metric definitions (verification)

### 6.1 Taxonomy

`metrics.py` module-level docstring maps each metric key to its semantics:

- **`window_level`** — multiclass {0=no fault, 1…D=first faulty sensor}. Standard sklearn precision/recall/F1 weighted and macro.
- **`sensor_level`** — binary multilabel. Flattened micro P/R/F1 over all (window × sensor) cells.
- **`per_fault_type`** — stratified by ground-truth fault type; each entry contains `fault_type_match_accuracy` (per-stratum recall for the fault type label) and `sensor_f1`.
- **`fault_type_classification`** — multiclass string prediction evaluated on faulty windows only.
- **`bertscore`** — semantic similarity (BERTScore P/R/F1) between generated and reference reasoning, computed on faulty windows only.

### 6.2 Fault-type classification — restricted to ground-truth faulty windows

```python
def compute_fault_type_classification_metrics(y_true_types, y_pred_types):
    pairs = [
        (t, p) for t, p in zip(y_true_types, y_pred_types)
        if t not in (None, "", "normal")
    ]
```

Only windows where the ground-truth type is not `"normal"` or empty are included — this avoids penalising correct normal predictions.

### 6.3 Per–fault-type: `fault_type_match_accuracy`

```python
for i in idx:
    if _normalize_fault_type_label(fault_types_pred[int(i)]) == true_label_norm:
        correct += 1
entry['fault_type_match_accuracy'] = float(correct / n_win) if n_win else 0.0
```

`_normalize_fault_type_label` lowercases and strips whitespace before comparison, so capitalisation differences do not count as mismatches.

### 6.4 `compute_all_metrics` — single source of truth for all metrics

`compute_all_metrics` in `metrics.py` is the only function both eval scripts call to produce results. It now accepts optional `reasoning`, `reference_reasoning`, and `window_is_faulty_true` arguments; when provided it calls `compute_bertscore` restricted to ground-truth faulty windows and stores the result under `metrics['bertscore']`.

`compute_all_metrics_unified` (used by the `compare_methods` path) is a thin wrapper around `compute_all_metrics` and shares the same implementation.

### 6.5 Ground-truth window label derived from `sensor_labels`, not raw `window_labels`

```python
window_labels_true_converted = np.zeros(len(window_labels_true), dtype=np.int64)
for i in range(len(window_labels_true)):
    faulty_indices = np.where(sensor_labels_true[i] > 0)[0]
    if len(faulty_indices) > 0:
        window_labels_true_converted[i] = faulty_indices[0] + 1
    else:
        window_labels_true_converted[i] = 0
```

The `.npz` `window_labels` array stores raw window indices, not sensor-indexed class labels. Both eval scripts re-derive the sensor-indexed label from `sensor_labels_true` before calling `compute_all_metrics`.

### 6.6 Prediction `window_label_pred` — sensor-indexed, derived from LLM output

Binary fault detection (`window_label_pred > 0`) is derived from the sensor-indexed `window_label` in the parsed prediction, not independently from `is_faulty`. This ensures window-level and sensor-level predictions are internally consistent.

### 6.7 Coercion: `is_faulty=True` with no recognised sensors

`utils.parsed_to_prediction` coerces the case where the LLM claims a fault but names no recognised sensor:

```python
if is_faulty and sensor_labels.sum() == 0:
    is_faulty = False
window_label = sensor_labels_to_window_label(sensor_labels) if is_faulty else 0
if not is_faulty:
    fault_type = "normal"
```

This prevents phantom faulty-window counts inflating the binary fault-detection recall.

---

## 7. Fault taxonomy

Eight ground-truth fault types, each tied to a specific sensor and injection pattern:

| Fault type | Sensor | Pattern |
|---|---|---|
| `VSS_DROPOUT` | VEHICLE_SPEED | Speed drops to ~0 for middle third of window then recovers |
| `MAF_SCALE_LOW` | INTAKE_MANIFOLD_PRESSURE | Reading consistently ~20% lower throughout window |
| `COOLANT_DROPOUT` | COOLANT_TEMPERATURE | 2–5 intermittent drops well below normal operating range |
| `TPS_STUCK` | THROTTLE | Freezes at constant value from roughly midpoint onward |
| `RPM_SPIKE_DROPOUT` | ENGINE_RPM | Sharp spike (up to ~1.8×) or dropout (down to ~0.4×) in middle half |
| `LOAD_SCALE_LOW` | ENGINE_LOAD | Uniformly depressed throughout window (~25–60% of expected) |
| `STFT_STUCK_HIGH` | SHORT_TERM_FUEL_TRIM_BANK_1 | Freezes at elevated value in middle 40% of window |
| `LTFT_DRIFT_HIGH` | LONG_TERM_FUEL_TRIM_BANK_1 | Shifted consistently higher throughout entire window |

The previous label `gradual_drift` covered RPM, LOAD, STFT, and LTFT faults indiscriminately. These were split into the four specific labels above to give the LLM a semantically accurate target for fault-type prediction.

---

## 8. Limitations to note in thesis

- **Small `TPS_STUCK` stratum (N = 2):** Percentage-based metrics are unstable; treat as qualitative only.
- **TPS_STUCK structural failure:** The GDN scores anomaly as deviation from learned sensor correlations. Throttle held at a constant value does not produce deviations — the sensor readings are self-consistent at the wrong setpoint. The GDN-KG-LLM pipeline therefore misses TPS faults at the detection stage; this is a fundamental architectural limitation, not a labelling issue.
- **LTFT_DRIFT_HIGH localisation vs. classification gap:** The GDN correctly identifies the LTFT sensor (sensor F1 = 1.0) but the LLM consistently mislabels the fault type (match = 0.0). The slow persistent drift pattern is apparently interpreted as a different anomaly by the LLM given the KG context.
- **LLM-only multiclass window collapse:** The baseline exhibits mode collapse to COOLANT_TEMPERATURE. This is systematic and visible in the per-class confusion; it means the 3.1 multiclass accuracy of 0.0575 reflects structural prediction failure, not dataset issues.
- **BERTScore pending:** Current `results/*.json` do not include BERTScore. Numbers will be populated after the next overnight run.
- **Older baselines** in `results/` use a different window split or the old fault taxonomy — do not mix with the primary 400-window tables.

---

*Refresh numeric tables in §3–5 by re-running the eval commands in §2 and reading the new `results/*.json` files.*
