# Adaptive Distributed SGD

This project implements a reproducible multi-process benchmark for comparing distributed SGD synchronization strategies:

- synchronous Parameter Server (`sync_ps`)
- asynchronous/stale synchronous Parameter Server (`async_ps_ssp`)
- Ring AllReduce (`ring_allreduce`)
- adaptive runtime switching (`adaptive`)

The C-6 contribution is the adaptive synchronization controller. It monitors worker-time heterogeneity and switches between Ring AllReduce and Async/SSP Parameter Server with hysteresis, cooldown, and a small absolute-spread gate to ignore measurement noise.

## Setup

```powershell
python -m pip install -e .
```

No PyTorch, CUDA, Linux traffic control, or GPU cluster is required. The simulator uses real Python worker processes and deterministic communication-delay modeling.

## Quick Demo

```powershell
python -m distributed_sgd.run --config configs/quick_demo.json
python -m distributed_sgd.analyze --results results/raw --out results/figures
```

Outputs:

- `results/raw/steps.csv`: per-step measurements
- `results/raw/summary.csv`: per-run aggregate metrics
- `results/figures/*.png`: scalability, baseline, adaptive timeline, sensitivity, and failure plots
- `results/figures/improvement_table.csv`: adaptive-vs-best-static improvement calculation

## Full Evaluation

```powershell
python -m distributed_sgd.run_matrix --config configs/full_eval.json
python -m distributed_sgd.analyze --results results/raw --out results/figures
```

The full config evaluates 1, 2, 4, 8, and 12 workers across homogeneous, bandwidth-limited, heterogeneous, dynamic mixed, and failure workloads with five repeats.

## Workload Model

The ML workload is softmax regression trained with mini-batch SGD on a synthetic classification dataset generated with a fixed seed. Each worker process receives a shard of the training set and computes gradients independently.

Communication is modeled with:

- payload size derived from model parameter count
- configurable bandwidth in Mbps
- fixed latency
- jitter
- strategy-specific data movement formulas

Limited-bandwidth workloads cap bandwidth at 100 Mbps and raise latency. Heterogeneous workloads slow 20% of workers by 3x. Dynamic workloads alternate between homogeneous and heterogeneous phases.

## Reproducibility

All experiments are controlled by JSON configs under `configs/`. Seeds, worker counts, strategy list, workload list, model size, network settings, and adaptive thresholds are explicit. Raw CSV output is suitable for rerunning analysis or independent verification.

## Demo Flow

1. Show `configs/quick_demo.json`.
2. Run the quick demo command.
3. Open `results/raw/summary.csv` and point out mean iteration time, throughput, straggler impact, and mode switches.
4. Run analysis.
5. Present `baseline_comparison.png`, `adaptive_mode_timeline.png`, `failure_recovery_timeline.png`, and `improvement_table.csv`.

## Interactive Next.js Dashboard

The frontend dashboard lives in `web/` and visualizes the generated simulator CSV files.

```powershell
cd web
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

The dashboard includes workload and worker-count filters, strategy comparison charts, adaptive switching timeline, failure scenario table, C-6 improvement metric, and demo talking points.

Build check:

```powershell
npm run build
```

Deploy from the `web/` directory with Vercel:

```powershell
npx vercel login
npx vercel --prod
```

If you already have a token:

```powershell
npx vercel --prod --token YOUR_VERCEL_TOKEN
```
