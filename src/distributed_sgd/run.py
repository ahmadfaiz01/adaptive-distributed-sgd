from __future__ import annotations

import argparse
from pathlib import Path

from .config import ensure_default_workloads, load_config
from .simulator import run_experiment, write_csv


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Run one adaptive distributed SGD experiment config.")
    parser.add_argument("--config", required=True, help="Path to a JSON experiment config.")
    args = parser.parse_args(argv)

    config = ensure_default_workloads(load_config(args.config))
    raw_dir = Path(config.output_dir)
    all_steps = []
    summaries = []

    for workload in config.workloads:
        for strategy in config.strategies:
            for workers in config.worker_counts:
                for repeat in range(config.repeats):
                    print(f"run workload={workload.name} strategy={strategy} workers={workers} repeat={repeat}")
                    rows, summary = run_experiment(config, workload, strategy, workers, repeat)
                    all_steps.extend(rows)
                    summaries.append(summary)

    write_csv(raw_dir / "steps.csv", all_steps)
    write_csv(raw_dir / "summary.csv", summaries)
    print(f"wrote {raw_dir / 'steps.csv'}")
    print(f"wrote {raw_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
