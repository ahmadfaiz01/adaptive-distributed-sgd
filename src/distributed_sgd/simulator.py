from __future__ import annotations

from dataclasses import asdict
import hashlib
from multiprocessing import Pipe, Process
from pathlib import Path
import csv
import math
import time
from typing import Any

import numpy as np

from .config import AdaptiveConfig, ExperimentConfig, NetworkConfig, WorkloadConfig
from .model import evaluate_loss, init_params, make_dataset, shard_dataset
from .worker import worker_loop


SUCCESS = "success"
FAILED = "failed"
DEGRADED = "degraded"


def _network_for_workload(base: NetworkConfig, workload: WorkloadConfig) -> NetworkConfig:
    if not workload.limited_bandwidth:
        return base
    limited = asdict(base)
    limited["bandwidth_mbps"] = min(base.bandwidth_mbps, 100.0)
    limited["latency_ms"] = max(base.latency_ms, 4.0)
    limited["jitter_ms"] = max(base.jitter_ms, 1.0)
    return NetworkConfig(**limited)


def _comm_delay_seconds(strategy: str, workers: int, payload_bytes: int, network: NetworkConfig, rng: np.random.Generator):
    if workers <= 0:
        return 0.0
    bandwidth_bytes = max(network.bandwidth_mbps, 1e-9) * 1_000_000 / 8
    latency = network.latency_ms / 1000.0
    jitter = max(0.0, rng.normal(0.0, network.jitter_ms / 1000.0))
    if strategy in {"sync_ps", "async_ps_ssp", "ps"}:
        bytes_moved = 2 * workers * payload_bytes
        bottleneck = bytes_moved / bandwidth_bytes
        return 2 * latency + bottleneck + jitter
    if strategy in {"ring_allreduce", "ring"}:
        factor = 2 * (workers - 1) / workers if workers > 1 else 0.0
        return 2 * (workers - 1) * latency + factor * payload_bytes / bandwidth_bytes + jitter
    raise ValueError(f"unknown strategy for communication delay: {strategy}")


def _worker_slowdowns(step: int, worker_count: int, workload: WorkloadConfig):
    slowdowns = [1.0] * worker_count
    hetero = workload.heterogeneous
    if workload.dynamic:
        hetero = 15 <= step < 30
    if hetero:
        n_slow = max(1, math.ceil(worker_count * workload.straggler_fraction)) if worker_count > 1 else 0
        for wid in range(worker_count - n_slow, worker_count):
            slowdowns[wid] = workload.straggler_slowdown
    return slowdowns


class WorkerPool:
    def __init__(self, shards: list[tuple[np.ndarray, np.ndarray]], batch_size: int, seed: int):
        self.parents = []
        self.processes = []
        for wid, (x, y) in enumerate(shards):
            parent, child = Pipe()
            proc = Process(target=worker_loop, args=(wid, child, x, y, batch_size, seed), daemon=True)
            proc.start()
            child.close()
            self.parents.append(parent)
            self.processes.append(proc)

    def request_grad(self, worker_id: int, params: np.ndarray, slowdown: float):
        self.parents[worker_id].send({"cmd": "grad", "params": params, "slowdown": slowdown})

    def recv_grad(self, worker_id: int):
        return self.parents[worker_id].recv()

    def crash(self, worker_id: int):
        if self.processes[worker_id].is_alive():
            self.parents[worker_id].send({"cmd": "crash"})

    def close(self):
        for parent, proc in zip(self.parents, self.processes):
            try:
                if proc.is_alive():
                    parent.send({"cmd": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        for proc in self.processes:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()
        for parent in self.parents:
            try:
                parent.close()
            except OSError:
                pass


def _maybe_switch_mode(
    current_mode: str,
    step: int,
    worker_compute_times: list[float],
    last_switch_step: int,
    adaptive: AdaptiveConfig,
):
    if step == 0 or (step + 1) % adaptive.monitor_window != 0:
        return current_mode, last_switch_step, None
    if step - last_switch_step < adaptive.cooldown_steps:
        return current_mode, last_switch_step, None
    arr = np.array(worker_compute_times, dtype=float)
    cv = float(arr.std() / (arr.mean() + 1e-12))
    spread = float(arr.max() - arr.min())
    if current_mode == "ring" and cv > adaptive.high_cv_threshold and spread > 0.003:
        return "ps", step, f"cv={cv:.3f} > {adaptive.high_cv_threshold:.3f}"
    if current_mode == "ps" and cv < adaptive.low_cv_threshold:
        return "ring", step, f"cv={cv:.3f} < {adaptive.low_cv_threshold:.3f}"
    return current_mode, last_switch_step, None


def run_experiment(
    config: ExperimentConfig,
    workload: WorkloadConfig,
    strategy: str,
    worker_count: int,
    repeat: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stable = int(hashlib.sha256(f"{workload.name}:{strategy}".encode("utf-8")).hexdigest()[:8], 16)
    seed = config.seed + repeat * 10_000 + worker_count * 101 + stable % 997
    rng = np.random.default_rng(seed)
    x_train, x_val, y_train, y_val = make_dataset(
        config.n_samples,
        config.n_features,
        config.n_classes,
        seed,
    )
    shards = shard_dataset(x_train, y_train, worker_count)
    params = init_params(x_train.shape[1], config.n_classes, seed)
    network = _network_for_workload(config.network, workload)
    payload_bytes = int(params.size * 8 * config.network.payload_scale)
    pool = WorkerPool(shards, config.batch_size, seed)

    alive = set(range(worker_count))
    mode = "ring" if strategy in {"ring_allreduce", "adaptive"} else "ps"
    last_switch_step = -config.adaptive.cooldown_steps
    status = SUCCESS
    failure_step = None
    recovery_step = None
    rows: list[dict[str, Any]] = []
    started_run = time.perf_counter()

    try:
        for step in range(config.steps):
            if workload.failure and step == workload.crash_step and workload.crash_worker in alive:
                pool.crash(workload.crash_worker)
                alive.remove(workload.crash_worker)
                failure_step = step
                status = DEGRADED
                if strategy == "ring_allreduce":
                    status = FAILED
                    rows.append(_failure_row(config, workload, strategy, worker_count, repeat, step, mode, status))
                    break
                if strategy == "adaptive":
                    mode = "ps"
                    last_switch_step = step
                    recovery_step = step + config.failure_timeout_steps

            if not alive:
                status = FAILED
                rows.append(_failure_row(config, workload, strategy, worker_count, repeat, step, mode, status))
                break

            slowdowns = _worker_slowdowns(step, worker_count, workload)
            active_workers = sorted(alive)
            step_started = time.perf_counter()
            for wid in active_workers:
                pool.request_grad(wid, params, slowdowns[wid])
            replies = [pool.recv_grad(wid) for wid in active_workers]
            measured_compute = time.perf_counter() - step_started

            grads = [reply["grad"] for reply in replies]
            worker_times = [float(reply["compute_time"]) for reply in replies]
            local_loss = float(np.mean([reply["loss"] for reply in replies]))
            straggler_rate = float(any(t > 2.0 * (np.mean(worker_times) + 1e-12) for t in worker_times))

            effective_strategy = "ring_allreduce" if mode == "ring" else "async_ps_ssp" if strategy == "adaptive" else strategy
            comm_strategy = "ring" if mode == "ring" else "ps"
            comm_time = _comm_delay_seconds(comm_strategy, len(active_workers), payload_bytes, network, rng)

            if mode == "ring":
                worker_cv = float(np.std(worker_times) / (np.mean(worker_times) + 1e-12))
                barrier_penalty = 1.0 + 20.0 * worker_cv
                iteration_time = max(worker_times) * barrier_penalty + comm_time
                grad = np.mean(grads, axis=0)
                params = params - config.learning_rate * grad
            elif strategy == "async_ps_ssp" or strategy == "adaptive":
                staleness_factor = 1.0 / max(1, min(config.staleness_bound, len(active_workers)))
                iteration_time = np.mean(worker_times) + comm_time * 0.65
                grad = np.mean(grads, axis=0)
                params = params - config.learning_rate * staleness_factor * grad
            else:
                iteration_time = max(worker_times) + comm_time
                grad = np.mean(grads, axis=0)
                params = params - config.learning_rate * grad

            wall_sleep = max(0.0, iteration_time - measured_compute)
            if wall_sleep > 0:
                time.sleep(min(wall_sleep * config.wall_time_scale, 0.01))

            val_loss = evaluate_loss(params, x_val, y_val)
            next_mode = mode
            switch_reason = ""
            if strategy == "adaptive" and len(active_workers) > 1:
                next_mode, last_switch_step, reason = _maybe_switch_mode(mode, step, worker_times, last_switch_step, config.adaptive)
                if reason:
                    switch_reason = reason
                    mode = next_mode

            rows.append({
                "workload": workload.name,
                "strategy": strategy,
                "effective_strategy": effective_strategy,
                "workers": worker_count,
                "repeat": repeat,
                "step": step,
                "status": status,
                "mode": mode,
                "active_workers": len(active_workers),
                "iteration_time_s": iteration_time,
                "compute_time_s": float(np.mean(worker_times)),
                "max_compute_time_s": float(np.max(worker_times)),
                "comm_time_s": comm_time,
                "comm_fraction": float(comm_time / max(iteration_time, 1e-12)),
                "local_loss": local_loss,
                "val_loss": val_loss,
                "throughput_samples_s": float(len(active_workers) * config.batch_size / max(iteration_time, 1e-12)),
                "worker_cv": float(np.std(worker_times) / (np.mean(worker_times) + 1e-12)),
                "straggler_impact": straggler_rate,
                "mode_switched": 1 if switch_reason else 0,
                "switch_reason": switch_reason,
                "failure_step": failure_step if failure_step is not None else "",
                "recovery_step": recovery_step if recovery_step is not None else "",
            })
    finally:
        pool.close()

    run_time = time.perf_counter() - started_run
    summary = summarize_rows(rows, config, workload, strategy, worker_count, repeat, run_time, failure_step, recovery_step)
    return rows, summary


def _failure_row(config, workload, strategy, worker_count, repeat, step, mode, status):
    return {
        "workload": workload.name,
        "strategy": strategy,
        "effective_strategy": strategy,
        "workers": worker_count,
        "repeat": repeat,
        "step": step,
        "status": status,
        "mode": mode,
        "active_workers": worker_count - 1,
        "iteration_time_s": "",
        "compute_time_s": "",
        "max_compute_time_s": "",
        "comm_time_s": "",
        "comm_fraction": "",
        "local_loss": "",
        "val_loss": "",
        "throughput_samples_s": "",
        "worker_cv": "",
        "straggler_impact": "",
        "mode_switched": 0,
        "switch_reason": "ring failed after worker crash",
        "failure_step": step,
        "recovery_step": "",
    }


def summarize_rows(rows, config, workload, strategy, worker_count, repeat, run_time, failure_step, recovery_step):
    valid = [r for r in rows if isinstance(r.get("iteration_time_s"), float)]
    if not valid:
        mean_iter = p95_iter = throughput = comm_fraction = straggler = final_loss = math.nan
        time_to_target = math.nan
        switches = 0
        status = FAILED
    else:
        iters = np.array([r["iteration_time_s"] for r in valid], dtype=float)
        mean_iter = float(iters.mean())
        p95_iter = float(np.percentile(iters, 95))
        throughput = float(np.mean([r["throughput_samples_s"] for r in valid]))
        comm_fraction = float(np.mean([r["comm_fraction"] for r in valid]))
        straggler = float(np.mean([r["straggler_impact"] for r in valid]))
        final_loss = float(valid[-1]["val_loss"])
        switches = int(sum(r["mode_switched"] for r in valid))
        status = rows[-1]["status"]
        time_to_target = math.nan
        elapsed = 0.0
        for r in valid:
            elapsed += r["iteration_time_s"]
            if r["val_loss"] <= config.target_loss:
                time_to_target = elapsed
                break

    recovery_time = ""
    if failure_step is not None and recovery_step is not None:
        recovery_time = max(0, recovery_step - failure_step)

    return {
        "workload": workload.name,
        "strategy": strategy,
        "workers": worker_count,
        "repeat": repeat,
        "status": status,
        "steps_completed": len(valid),
        "mean_iteration_time_s": mean_iter,
        "p95_iteration_time_s": p95_iter,
        "throughput_samples_s": throughput,
        "comm_fraction": comm_fraction,
        "straggler_impact_rate": straggler,
        "final_val_loss": final_loss,
        "time_to_target_loss_s": time_to_target,
        "mode_switches": switches,
        "failure_step": failure_step if failure_step is not None else "",
        "recovery_time_steps": recovery_time,
        "wall_clock_run_s": run_time,
    }


def write_csv(path: str | Path, rows: list[dict[str, Any]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
