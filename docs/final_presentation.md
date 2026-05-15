# Final Presentation Outline

## Slide 1: Title

Adaptive Distributed SGD: Parameter Server vs Ring AllReduce

## Slide 2: Problem

Distributed training shifts the bottleneck from computation to communication. Static synchronization choices fail when worker speed or network conditions change.

## Slide 3: Literature Gap

Parameter Server handles stragglers but centralizes bandwidth. Ring AllReduce balances communication but is barrier-synchronous. Existing systems generally choose one mode before training starts.

## Slide 4: System Design

Show four components:

- worker processes
- Parameter Server/coordinator
- Ring AllReduce communication model
- adaptive mode-switch controller

## Slide 5: C-6 Novelty

The adaptive controller monitors worker-time coefficient of variation and switches:

- Ring -> PS when CV > 0.30
- PS -> Ring when CV < 0.15
- cooldown prevents oscillation
- absolute spread gate filters tiny timing noise
- worker crash forces adaptive fallback to PS

## Slide 6: Experimental Setup

Windows-friendly multi-process simulator, NumPy/sklearn softmax regression, synthetic fixed-seed dataset, modeled bandwidth/latency/jitter, CSV logging, reproducible JSON configs.

## Slide 7: Workloads

Homogeneous, limited bandwidth, heterogeneous, heterogeneous plus limited bandwidth, dynamic mixed workload, and worker crash.

## Slide 8: Baselines

Compare `sync_ps`, `async_ps_ssp`, `ring_allreduce`, and `adaptive`.

## Slide 9: Scalability Results

Use `results/figures/scalability_*.png`.

## Slide 10: Comparative Baseline Analysis

Use `results/figures/baseline_comparison.png` and explain which static baseline wins under each workload.

## Slide 11: Adaptive Timeline

Use `results/figures/adaptive_mode_timeline.png` to show runtime switching.

## Slide 12: Failure Scenario

Use `results/figures/failure_recovery_timeline.png`. Ring fails after worker crash; PS/adaptive continue degraded.

## Slide 13: Sensitivity Analysis

Use `results/figures/sensitivity_heatmap.png`. Discuss threshold and scale trade-offs.

## Slide 14: Measurable Improvement

Use `results/figures/improvement_table.csv`. Report:

```text
(best_static_time - adaptive_time) / best_static_time * 100
```

## Slide 15: Trade-offs

Adaptive mode helps under changing/heterogeneous conditions. It adds monitoring complexity and can be unnecessary when the cluster is stable and homogeneous.

## Slide 16: Demo

```powershell
python -m distributed_sgd.run --config configs/quick_demo.json
python -m distributed_sgd.analyze --results results/raw --out results/figures
cd web
npm run dev
```

Open `http://localhost:3000` and use the dashboard to show dynamic workload improvement, adaptive mode switches, and worker-crash behavior.
