from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass
class NetworkConfig:
    bandwidth_mbps: float = 1000.0
    latency_ms: float = 1.0
    jitter_ms: float = 0.2
    payload_scale: float = 4000.0


@dataclass
class WorkloadConfig:
    name: str = "homogeneous_high_bw"
    heterogeneous: bool = False
    limited_bandwidth: bool = False
    dynamic: bool = False
    failure: bool = False
    straggler_fraction: float = 0.2
    straggler_slowdown: float = 3.0
    crash_worker: int = 0
    crash_step: int = 25


@dataclass
class AdaptiveConfig:
    monitor_window: int = 10
    high_cv_threshold: float = 0.30
    low_cv_threshold: float = 0.15
    cooldown_steps: int = 10


@dataclass
class ExperimentConfig:
    strategies: list[str] = field(default_factory=lambda: ["sync_ps", "async_ps_ssp", "ring_allreduce", "adaptive"])
    worker_counts: list[int] = field(default_factory=lambda: [1, 2, 4])
    repeats: int = 1
    steps: int = 40
    batch_size: int = 32
    learning_rate: float = 0.2
    n_samples: int = 6000
    n_features: int = 32
    n_classes: int = 3
    seed: int = 123
    staleness_bound: int = 3
    failure_timeout_steps: int = 1
    target_loss: float = 0.70
    wall_time_scale: float = 0.05
    network: NetworkConfig = field(default_factory=NetworkConfig)
    workloads: list[WorkloadConfig] = field(default_factory=list)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    output_dir: str = "results/raw"


def _merge_dataclass(cls: type, data: dict[str, Any]):
    known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**known)


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    network = _merge_dataclass(NetworkConfig, raw.get("network", {}))
    adaptive = _merge_dataclass(AdaptiveConfig, raw.get("adaptive", {}))
    workloads = [_merge_dataclass(WorkloadConfig, item) for item in raw.get("workloads", [])]
    base = {k: v for k, v in raw.items() if k not in {"network", "adaptive", "workloads"}}
    return ExperimentConfig(network=network, adaptive=adaptive, workloads=workloads, **base)


def ensure_default_workloads(config: ExperimentConfig) -> ExperimentConfig:
    if config.workloads:
        return config
    config.workloads = [
        WorkloadConfig(name="homogeneous_high_bw"),
        WorkloadConfig(name="homogeneous_limited_bw", limited_bandwidth=True),
        WorkloadConfig(name="heterogeneous_high_bw", heterogeneous=True),
        WorkloadConfig(name="heterogeneous_limited_bw", heterogeneous=True, limited_bandwidth=True),
        WorkloadConfig(name="dynamic_mixed", dynamic=True, limited_bandwidth=True),
        WorkloadConfig(name="worker_crash", failure=True, heterogeneous=True),
    ]
    return config
