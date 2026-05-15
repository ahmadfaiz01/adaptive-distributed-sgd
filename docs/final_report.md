# Adaptive Distributed SGD: Final Report Draft

## Literature Gap

Distributed deep learning commonly uses either Parameter Server synchronization or collective communication such as Ring AllReduce. Parameter Server designs tolerate asynchronous progress and stragglers more naturally, but centralize communication and can bottleneck at the server. Ring AllReduce spreads communication evenly and performs well on homogeneous clusters, but it is barrier-synchronous and therefore sensitive to slow or failed workers.

The gap addressed here is runtime adaptation. Existing systems usually choose the synchronization architecture before the training job starts. This project evaluates whether a lightweight controller can switch between modes during training when worker heterogeneity changes.

## Experimental Setup

The implementation is a Windows-friendly multi-process simulator using Python, NumPy, pandas, matplotlib, and sklearn. Each worker is a real process with a private shard of the training data. The workload is softmax regression trained by mini-batch SGD on a synthetic classification dataset with fixed random seeds.

Communication time is modeled explicitly from payload size, bandwidth, latency, and jitter. This replaces Linux `tc netem` while keeping experiments reproducible on the available machine.

## Workload Modeling

The evaluation includes:

- homogeneous workers with high bandwidth
- homogeneous workers with limited bandwidth
- heterogeneous workers with high bandwidth
- heterogeneous workers with limited bandwidth
- dynamic mixed workload with phase changes
- worker crash failure scenario

Heterogeneity is injected by slowing 20% of workers by 3x. Limited bandwidth caps the modeled network at 100 Mbps. The dynamic workload alternates between homogeneous and heterogeneous phases.

## Implemented Strategies

- `sync_ps`: all workers send gradients to a central server; the server averages gradients and broadcasts updated parameters.
- `async_ps_ssp`: workers update through a Parameter Server using a bounded-staleness approximation with default staleness bound 3.
- `ring_allreduce`: workers synchronize through an emulated ring collective and are blocked by the slowest worker.
- `adaptive`: starts in Ring mode, monitors worker-time coefficient of variation, switches to PS mode when CV exceeds 0.30 with a meaningful absolute spread, and switches back when CV falls below 0.15 after cooldown.

## Metrics

The experiment records mean and p95 iteration time, throughput, communication fraction, straggler impact rate, validation loss, time to target loss, mode-switch count, failure status, and recovery time.

The measurable C-6 improvement is computed as:

```text
improvement = (best_static_time - adaptive_time) / best_static_time * 100
```

## Failure Evaluation

The worker-crash workload terminates one worker mid-run. Ring AllReduce is expected to fail because the ring collective cannot complete after membership loss. Parameter Server and adaptive modes continue in degraded mode with fewer active workers. Adaptive mode also switches to PS mode after failure if it was previously in Ring mode.

## Sensitivity Analysis

The default sensitivity surface compares adaptive performance across workload and worker-count settings. For deeper analysis, rerun experiments with modified `high_cv_threshold`, `low_cv_threshold`, and `monitor_window` in the JSON config, then regenerate figures.

## C-6 Novelty Justification

The novel contribution is a system-level adaptive synchronization controller, not a new learning algorithm. It observes runtime behavior, changes communication strategy, compares against static baselines, and exposes measurable trade-offs:

- Benefit: lower iteration time under dynamic or heterogeneous workloads.
- Cost: monitoring overhead, possible switch instability, and less benefit in stable homogeneous conditions.
- Trade-off: Ring mode is bandwidth-efficient when workers are balanced; PS mode is more resilient when workers diverge or fail.

## Reproducibility

All runs are defined by JSON configs in `configs/`. Raw per-step and summary data are written to CSV. Figures and improvement tables are generated from the raw data, so the final claims can be traced back to measurements.
