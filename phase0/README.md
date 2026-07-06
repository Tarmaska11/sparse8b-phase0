# Sparse8B Phase-0 harness

CPU-only evaluation of TEAL / gate-select activation sparsity on MLC-quantized Llama
models (`Llama-3.1-8B q3f16_0`, `Llama-3.2-3B q4f16_0`). No GPU, no `transformers`.
A bit-exact CPU sim (`model_sim.py`) streams the model one layer at a time from the MLC
shards and applies sparsity exactly where the planned device kernels will.

## Scripts

| script | does |
|---|---|
| `common.py` / `q3loader.py` / `model_sim.py` | config, bit-exact dequant, streamed forward + sparsity hooks (pre-existing) |
| `data.py` | wikitext-2 + streamed ultrachat -> fixed token windows |
| `calibrate.py` | one forward over a wiki/chat mix -> per-(layer,class) magnitude thresholds for a sparsity grid |
| `ppl.py` | perplexity (wikitext test + ultrachat test) under a sparsity config; stores raw nll/count for shard merge |
| `bench.py` | zero-shot MMLU / ARC-Challenge letter accuracy under a sparsity config |
| `traces.py` | FFN gate-select firing traces (per-neuron counts + sampled packed masks) |
| `missim.py` | offline cache-miss simulation over the traces at several hot-set budgets |
| `report.py` | merges ppl/bench shards + missim.csv into `REPORT.md` with the GATE-S1 verdict |
| `golden_test.py` | greedy decode sanity check (no kv-cache) to validate the sim vs device |

## Run order

```
calibrate  ->  ppl / bench / traces  ->  missim  ->  report
```

`calibrate.py` must run first (per model); `ppl`, `bench`, `traces` consume its
`thresholds-<model>.json`. `missim` needs the trace `.npz` files; `report` needs the
ppl/bench JSONs plus `missim.csv`.

## Where models land

`snapshot_download` into `models/<repo-basename>`, e.g.
`models/Llama-3.1-8B-Instruct-q3f16_0-MLC` and
`models/Llama-3.2-3B-Instruct-q4f16_0-MLC`. `common.resolve_model_dir` finds them
(the 8B also has a repo-root `q3f16_1` local fallback). Pass `--models-root` an
absolute path in CI.

## Local run (Linux)

```bash
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download as d; \
  d('mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC', \
    local_dir='models/Llama-3.2-3B-Instruct-q4f16_0-MLC', \
    allow_patterns=['*.json','*.bin'])"
python calibrate.py --model 3b-q4 --models-root models --tokens 32768 --threads 4
python ppl.py  --model 3b-q4 --models-root models --sparsity 0 --threads 4
python bench.py --model 3b-q4 --models-root models --sparsity 0 --suite both --threads 4
```

## Local run (Windows PowerShell)

```powershell
pip install -r requirements.txt
python -c "from huggingface_hub import snapshot_download as d; d('mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC', local_dir='models/Llama-3.2-3B-Instruct-q4f16_0-MLC', allow_patterns=['*.json','*.bin'])"
python calibrate.py --model 3b-q4 --models-root models --tokens 32768 --threads 4
python ppl.py --model 3b-q4 --models-root models --sparsity 0 --threads 4
```

Sparse example (after calibrate):

```
python ppl.py  --model 8b-q3 --thresholds thresholds-8b-q3.json --sparsity 50 --select teal
python bench.py --model 8b-q3 --thresholds thresholds-8b-q3.json --sparsity 50 --select gate
python traces.py --model 8b-q3 --thresholds thresholds-8b-q3.json --sparsity 50 --select gate --shard 0/4
python missim.py --traces "traces_*.npz" --model 8b-q3
python report.py --dir . --out REPORT.md
```

## CI usage

`.github/workflows/sparse8b-phase0.yml`, triggered by **workflow_dispatch** (inputs:
`task` = all|calib|ppl|bench|traces|report, `grid` = core|full) or a push to
`ci/sparse8b-phase0`. Jobs: `calib-8b`, `calib-3b` (upload thresholds) -> `ppl`,
`bench`, `traces` (matrix, need calib) -> `report` (downloads all artifacts, runs
`missim.py` + `report.py`, uploads `phase0-report` and echoes `REPORT.md` to the job
summary). Weights are cached under `models/` keyed `mlc-models-v1-<model>`.

## RAM / time expectations (ubuntu-latest, 4 vCPU, 16 GB)

- calibrate: ~2-3 h/model; hidden states for the calib windows + one layer of weights in RAM.
- ppl / bench / traces: ~1-3 h each depending on `--tokens`; shard to stay under the 6 h cap.
- traces masks are packed + sampled; `traces.py` auto-halves sampling if the packed
  masks would exceed ~2.5 GB.
- All jobs `timeout-minutes: 350`.

## GATE-S1 criteria

A sparse config passes GATE-S1 when, versus the **3B q4 dense** baseline:

- `achieved_model_wide >= 0.40`, and
- `ppl_overall < ppl_overall(3b dense)`, and
- `mmlu_acc >= mmlu(3b)` and `arc_acc >= arc(3b)`, and
- cache-miss `p95_mb <= 25` at the hot-fraction closest to 0.68.

`report.py` prints PASS/FAIL, the winning config (lowest ppl among passers), and the
full ppl / bench / missim tables.
