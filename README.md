# GARAGE: GrAph-based Reasoning for Automotive diagnostics GEneration

Automotive **OBD-II** time-series anomaly detection and diagnostic reasoning. The main approach trains a **Graph Deviation Network (GDN)** on sliding windows of eight normalised sensors, then feeds GDN outputs into a **knowledge graph** built from window embedding similarity; a **local LLM** (via LM Studio) produces structured fault hypotheses and natural-language reasoning. This repository also includes **LLM-only** and classical **ARIMA / LSTM** baselines.

## Method

1. **Data**  
   Raw drives under `data/carOBD/obdiidata/` are turned into a single **`data/shared_dataset/`** split (`train.npz`, `val.npz`, `test.npz`) with injected synthetic faults, per-sensor labels, fault-type strings (e.g. `VSS_DROPOUT`, `COOLANT_DROPOUT`), and optional reference reasoning for evaluation.

2. **GDN (two stages)**  
   - **Stage 1:** Self-supervised forecasting on clean windows (GRU temporal encoder + GAT over sensors).  
   - **Stage 2:** Fault-aware training using Stage 1 weights; per-sensor anomaly scores drive detection.

3. **GDN + KG + LLM**  
   For each test window, GDN scores identify candidate anomalies. Similar windows are linked in a small **KG** (cosine / Euclidean similarity on embeddings). The LLM receives a compact summary (violations + graph context) and returns a structured prediction (faulty or not, sensors, fault type, reasoning).

4. **Baselines**  
   - **LLM-only:** Same schema, but the prompt uses raw (unnormalised) series context without GDN/KG.  
   - **GDN-only:** Metrics from GDN outputs without LLM (`evaluate_gdn_kg_llm.py --mode gdn_only`).  
   - **ARIMA / LSTM:** Under `ablations/`.

Metric definitions (window-level multiclass, sensor-level multilabel, fault-type on faulty windows, optional BERTScore) are implemented in `llm/evaluation/metrics.py`. A detailed results write-up lives in `llm/evaluation/EVALUATION_RESULTS_AND_METHODOLOGY.md`.

## Repository structure

| Path | Role |
|------|------|
| `data/carOBD/obdiidata/` | Raw per-drive CSVs |
| `data/create_shared_dataset.py` | Builds `data/shared_dataset/*.npz` |
| `data/shared_dataset/` | Train / val / test tensors and metadata (created by script) |
| `models/gdn_model.py` | GDN architecture and losses |
| `training/train_stage1.py`, `train_stage2.py` | Two-stage training |
| `training/fault_injection.py` | Synthetic fault generation for Stage 2 / test |
| `kg/similarity.py` | Window–window similarity edges for the KG |
| `llm/inference.py` | LM Studio OpenAI-compatible HTTP client |
| `llm/evaluation/` | Eval scripts, metrics, schemas, stratified sampling |
| `ablations/` | ARIMA, LSTM, shared loaders, `run_ablations.py` |
| `results/` | Primary JSON outputs from eval runs |
| `results-final/` | Curated comparison artifacts (e.g. `comparison_table.csv`) |
| `figures/` | Plots for papers / slides |
| `checkpoints/` | Saved `.pt` checkpoints (after training) |
| `train_all.sh` | End-to-end: shared dataset → Stage 1 → Stage 2 |

## Key results

**Primary LLM comparison (400 windows, stratified by fault type, seed 42)** — see `results/gdn_kg_llm.json` and `results/llm_baseline.json` (same `sample_indices` in both files).

| Setting | GDN-KG-LLM | LLM-only baseline |
|--------|------------|-------------------|
| Window-level accuracy (multiclass sensor index) | **0.97** | 0.06 (mode collapse to one sensor class) |
| Window-level F1 (weighted) | **0.97** | 0.01 |
| Sensor-level F1 (micro, root sensors) | **0.95** | 0.09 |
| Fault-type accuracy (faulty windows only, *n* = 108) | **0.69** | 0.21 |

The LLM-only baseline achieves misleadingly high **sensor-level accuracy** on mostly-normal windows; precision and F1 stay low. The full analysis, per–fault-type tables, efficiency timings, and limitations (e.g. `TPS_STUCK`, `LTFT_DRIFT_HIGH` behaviour) are in `llm/evaluation/EVALUATION_RESULTS_AND_METHODOLOGY.md`.

**Alternative summary table** (different metric slice / run): `results-final/comparison_table.csv` compares GDN-only, LLM baseline, and GDN-KG-LLM on window/sensor/fault-type and BERTScore-style figures — **do not mix** those numbers with the 400-window JSONs without checking the methodology doc.

## Quickstart

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the shared dataset and train GDN

From the repo root:

```bash
bash train_all.sh
```

This runs `data/create_shared_dataset.py`, then Stage 1 and Stage 2 training, and writes checkpoints under `checkpoints/`. With the default names in `train_all.sh`, expect `stage1_best_forecast_stage1_best.pt` (Stage 1) and `stage2_clean_stage2_clean_best.pt` (Stage 2).

To build data only:

```bash
python data/create_shared_dataset.py \
  --raw-data-path data/carOBD/obdiidata \
  --output-dir data/shared_dataset
```

### 3. LLM inference (LM Studio)

Start [LM Studio](https://lmstudio.ai/) (or any OpenAI-compatible server on port **1234**), load your model, and enable the **local server**. Eval scripts default to `http://localhost:1234/v1`. The LLM baseline defaults to model id `granite-4.0-h-micro-GGUF` (`--model-repo`); align the loaded model name with that flag.

### 4. Evaluation

**GDN + KG + LLM** (requires a Stage 2 checkpoint):

```bash
python llm/evaluation/evaluate_gdn_kg_llm.py \
  --dataset data/shared_dataset/test.npz \
  --model-path checkpoints/stage2_clean_stage2_clean_best.pt \
  --output results/gdn_kg_llm.json \
  --limit 400 \
  --stratified-limit-seed 42
```

Use a different `--model-path` if you changed `--checkpoint_name` during training.

**LLM-only baseline** (same stratified 400-window slice):

```bash
python llm/evaluation/evaluate_llm_baseline.py \
  --dataset data/shared_dataset/test.npz \
  --output results/llm_baseline.json \
  --limit 400 \
  --seed 42
```

**GDN-only** (no LLM):

```bash
python llm/evaluation/evaluate_gdn_kg_llm.py \
  --mode gdn_only \
  --dataset data/shared_dataset/test.npz \
  --model-path checkpoints/stage2_clean_stage2_clean_best.pt \
  --output results/gdn_only.json
```

Optional: `ablations/run_ablations.py` and related scripts for ARIMA/LSTM. Increase `--timeout` if the first LLM requests time out.