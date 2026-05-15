from __future__ import annotations

from multiprocessing.connection import Connection
import time

import numpy as np

from .model import softmax_loss_and_grad


def worker_loop(worker_id: int, conn: Connection, x: np.ndarray, y: np.ndarray, batch_size: int, seed: int):
    rng = np.random.default_rng(seed + worker_id * 997)
    cursor = 0
    order = rng.permutation(len(y))

    while True:
        msg = conn.recv()
        cmd = msg.get("cmd")
        if cmd == "stop":
            conn.close()
            return
        if cmd == "crash":
            conn.close()
            return
        if cmd != "grad":
            conn.send({"worker_id": worker_id, "error": f"unknown command {cmd}"})
            continue

        params = msg["params"]
        slowdown = float(msg.get("slowdown", 1.0))
        if cursor + batch_size > len(y):
            order = rng.permutation(len(y))
            cursor = 0
        idx = order[cursor: cursor + batch_size]
        cursor += batch_size

        started = time.perf_counter()
        loss, grad = softmax_loss_and_grad(params, x[idx], y[idx])
        if slowdown > 1.0:
            time.sleep(0.0025 * (slowdown - 1.0))
        elapsed = time.perf_counter() - started
        conn.send({
            "worker_id": worker_id,
            "grad": grad,
            "loss": loss,
            "compute_time": elapsed,
            "slowdown": slowdown,
        })
