"use client";

import { AlertTriangle, CheckCircle2, Pause, Play, RotateCcw, Server, Share2, Zap } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type Row = Record<string, string>;

const strategyColors: Record<string, string> = {
  sync_ps: "#6b7280",
  async_ps_ssp: "#374151",
  ring_allreduce: "#ef4444",
  adaptive: "#b91c1c"
};

const strategyNames: Record<string, string> = {
  sync_ps: "Sync PS",
  async_ps_ssp: "Async PS",
  ring_allreduce: "Ring AllReduce",
  adaptive: "Adaptive"
};

const workloadNames: Record<string, string> = {
  homogeneous_high_bw: "Homogeneous",
  heterogeneous_high_bw: "Heterogeneous",
  dynamic_mixed: "Dynamic Mixed",
  worker_crash: "Worker Crash"
};

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let quoted = false;
  for (const char of line) {
    if (char === '"') quoted = !quoted;
    else if (char === "," && !quoted) {
      out.push(current);
      current = "";
    } else current += char;
  }
  out.push(current);
  return out;
}

function parseCsv(text: string): Row[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function n(row: Row | undefined, key: string): number {
  const value = Number(row?.[key]);
  return Number.isFinite(value) ? value : 0;
}

function fmt(value: number, digits = 2) {
  if (!Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

function modeLabel(mode: string) {
  return mode === "ps" ? "Parameter Server" : "Ring";
}

function StatusBadge({ status }: { status: string }) {
  return <span className={`status ${status}`}>{status || "n/a"}</span>;
}

function WorkerNode({ id, active, slow }: { id: number; active: boolean; slow: boolean }) {
  return (
    <div className={`worker ${active ? "active" : ""} ${slow ? "slow" : ""}`}>
      <span>W{id + 1}</span>
      <small>{slow ? "slow" : "ready"}</small>
    </div>
  );
}

function StrategyBars({ rows }: { rows: Row[] }) {
  const valid = rows.filter((row) => row.status !== "failed");
  const max = Math.max(...valid.map((row) => n(row, "mean_iteration_time_s")), 0.001);
  return (
    <div className="bars">
      {rows.map((row) => {
        const value = n(row, "mean_iteration_time_s");
        const width = row.status === "failed" ? 14 : Math.max(10, (value / max) * 100);
        return (
          <div className="bar-row" key={row.strategy}>
            <div className="bar-label">
              <span className="dot" style={{ background: strategyColors[row.strategy] }} />
              {strategyNames[row.strategy]}
            </div>
            <div className="bar-track">
              <div className="bar-fill" style={{ width: `${width}%`, background: strategyColors[row.strategy] }} />
            </div>
            <strong>{row.status === "failed" ? "failed" : `${fmt(value, 3)}s`}</strong>
          </div>
        );
      })}
    </div>
  );
}

function TimelineChart({ rows, cursor }: { rows: Row[]; cursor: number }) {
  const valid = rows.filter((row) => row.strategy === "adaptive" && row.iteration_time_s);
  const maxStep = Math.max(...valid.map((row) => n(row, "step")), 1);
  const maxIter = Math.max(...valid.map((row) => n(row, "iteration_time_s")), 0.001);
  const visible = valid.filter((row) => n(row, "step") <= cursor);
  const points = visible
    .map((row) => {
      const x = 44 + (n(row, "step") / maxStep) * 612;
      const y = 225 - (n(row, "iteration_time_s") / maxIter) * 170;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg className="line-chart" viewBox="0 0 700 260" role="img" aria-label="Animated adaptive iteration timeline">
      {[0, 1, 2].map((tick) => (
        <line key={tick} x1="44" x2="660" y1={225 - tick * 70} y2={225 - tick * 70} stroke="#e4e4e7" />
      ))}
      <polyline points={points} fill="none" stroke="#dc2626" strokeWidth="4" strokeLinecap="round" strokeLinejoin="round" />
      {visible
        .filter((row) => row.mode_switched === "1")
        .map((row) => {
          const x = 44 + (n(row, "step") / maxStep) * 612;
          return (
            <g key={`${row.step}-${row.mode}`}>
              <line x1={x} x2={x} y1="34" y2="225" stroke="#27272a" strokeDasharray="5 6" />
              <text x={x + 8} y="48" fontSize="12" fill="#27272a">
                switch to {row.mode}
              </text>
            </g>
          );
        })}
      <text x="350" y="248" textAnchor="middle" fontSize="12" fill="#71717a">
        training step
      </text>
    </svg>
  );
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Row[]>([]);
  const [steps, setSteps] = useState<Row[]>([]);
  const [improvements, setImprovements] = useState<Row[]>([]);
  const [workload, setWorkload] = useState("dynamic_mixed");
  const [workers, setWorkers] = useState("4");
  const [running, setRunning] = useState(false);
  const [cursor, setCursor] = useState(0);

  useEffect(() => {
    Promise.all([
      fetch("/data/summary.csv").then((res) => res.text()),
      fetch("/data/steps.csv").then((res) => res.text()),
      fetch("/data/improvement_table.csv").then((res) => res.text())
    ]).then(([summaryCsv, stepsCsv, improvementCsv]) => {
      setSummary(parseCsv(summaryCsv));
      setSteps(parseCsv(stepsCsv));
      setImprovements(parseCsv(improvementCsv));
    });
  }, []);

  const workerOptions = useMemo(
    () => Array.from(new Set(summary.map((row) => row.workers))).sort((a, b) => Number(a) - Number(b)),
    [summary]
  );
  const relevantWorkloads = ["dynamic_mixed", "heterogeneous_high_bw", "homogeneous_high_bw", "worker_crash"];
  const filteredSummary = summary.filter((row) => row.workload === workload && row.workers === workers);
  const filteredSteps = steps.filter((row) => row.workload === workload && row.workers === workers && row.repeat === "0");
  const adaptiveRows = filteredSteps.filter((row) => row.strategy === "adaptive" && row.iteration_time_s);
  const maxStep = Math.max(...adaptiveRows.map((row) => n(row, "step")), 35);
  const currentStep = adaptiveRows.find((row) => n(row, "step") === cursor) ?? adaptiveRows[adaptiveRows.length - 1];
  const currentMode = currentStep?.mode ?? "ring";
  const currentCv = n(currentStep, "worker_cv");
  const improvement = improvements.find((row) => row.workload === workload && row.workers === workers);
  const adaptive = filteredSummary.find((row) => row.strategy === "adaptive");
  const bestStatic = filteredSummary
    .filter((row) => row.strategy !== "adaptive" && row.status !== "failed")
    .sort((a, b) => n(a, "mean_iteration_time_s") - n(b, "mean_iteration_time_s"))[0];
  const failureRows = summary.filter((row) => row.workload === "worker_crash" && row.workers === workers);
  const isCrash = workload === "worker_crash" && cursor >= 18;
  const slowStart = workload.includes("heterogeneous") || (workload === "dynamic_mixed" && cursor >= 15 && cursor < 30) || workload === "worker_crash";

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => {
      setCursor((value) => {
        if (value >= maxStep) {
          setRunning(false);
          return maxStep;
        }
        return value + 1;
      });
    }, 260);
    return () => window.clearInterval(timer);
  }, [running, maxStep]);

  function reset() {
    setRunning(false);
    setCursor(0);
  }

  return (
    <main className="page">
      <section className="hero">
        <div>
          <p className="eyebrow">Distributed SGD simulation</p>
          <h1>Adaptive synchronization, visualized.</h1>
          <p className="hero-copy">
            Watch workers train, detect stragglers, switch between Ring AllReduce and Parameter Server, and compare the measured outcome.
          </p>
        </div>
        <div className="hero-actions">
          <button className="primary" onClick={() => setRunning((value) => !value)}>
            {running ? <Pause size={18} /> : <Play size={18} />}
            {running ? "Pause" : "Start Simulation"}
          </button>
          <button className="secondary" onClick={reset}>
            <RotateCcw size={18} />
            Reset
          </button>
        </div>
      </section>

      <section className="shell">
        <div className="toolbar">
          <label>
            Workload
            <select value={workload} onChange={(event) => { setWorkload(event.target.value); reset(); }}>
              {relevantWorkloads.map((item) => (
                <option key={item} value={item}>
                  {workloadNames[item]}
                </option>
              ))}
            </select>
          </label>
          <label>
            Scale
            <select value={workers} onChange={(event) => { setWorkers(event.target.value); reset(); }}>
              {workerOptions.map((item) => (
                <option key={item} value={item}>
                  {item} workers
                </option>
              ))}
            </select>
          </label>
          <div className="step-pill">Step {cursor} / {maxStep}</div>
        </div>

        <div className="layout">
          <section className="sim-card">
            <div className="card-head">
              <div>
                <h2>Simulation</h2>
                <p>{workloadNames[workload]} workload with {workers} worker processes.</p>
              </div>
              <div className={`mode ${currentMode}`}>
                {currentMode === "ps" ? <Server size={18} /> : <Share2 size={18} />}
                {modeLabel(currentMode)}
              </div>
            </div>

            <div className={`network-scene ${currentMode}`}>
              <div className="server-node">
                <Server size={28} />
                <span>PS</span>
              </div>
              <div className="worker-grid">
                {Array.from({ length: Number(workers) || 4 }).map((_, index) => (
                  <WorkerNode key={index} id={index} active={!isCrash || index !== 0} slow={slowStart && index >= Math.max(1, Number(workers) - 1)} />
                ))}
              </div>
              <div className="ring-line" />
            </div>

            <div className="stats-strip">
              <div>
                <span>Variation</span>
                <strong>{fmt(currentCv, 3)}</strong>
              </div>
              <div>
                <span>Adaptive</span>
                <strong>{fmt(n(adaptive, "mean_iteration_time_s"), 3)}s</strong>
              </div>
              <div>
                <span>Best static</span>
                <strong>{bestStatic ? strategyNames[bestStatic.strategy] : "n/a"}</strong>
              </div>
              <div>
                <span>Improvement</span>
                <strong>{improvement ? `${fmt(n(improvement, "adaptive_improvement_percent"), 1)}%` : "n/a"}</strong>
              </div>
            </div>
          </section>

          <aside className="insight-card">
            <h2>Demo Notes</h2>
            <div className="talking-point">
              <Zap size={18} />
              <p>Ring is strong when workers finish together.</p>
            </div>
            <div className="talking-point">
              <Server size={18} />
              <p>Parameter Server handles slow workers better.</p>
            </div>
            <div className="talking-point">
              <CheckCircle2 size={18} />
              <p>Adaptive switches modes during runtime.</p>
            </div>
          </aside>
        </div>

        <div className="charts">
          <section className="panel">
            <div className="card-head compact">
              <h2>Runtime Comparison</h2>
              <p>Lower is better.</p>
            </div>
            <StrategyBars rows={filteredSummary} />
          </section>

          <section className="panel">
            <div className="card-head compact">
              <h2>Adaptive Timeline</h2>
              <p>Switches appear during playback.</p>
            </div>
            <TimelineChart rows={filteredSteps} cursor={cursor} />
          </section>
        </div>

        <section className="panel final-panel">
          <div>
            <h2>Failure Check</h2>
            <p>Ring fails after a crash. PS and adaptive continue in degraded mode.</p>
          </div>
          <div className="failure-grid">
            {failureRows.map((row) => (
              <div className="failure-item" key={row.strategy}>
                <span>{strategyNames[row.strategy]}</span>
                <StatusBadge status={row.status} />
              </div>
            ))}
          </div>
          <div className="warning">
            <AlertTriangle size={18} />
            C-6 claim: adaptive improves dynamic workloads by switching instead of staying static.
          </div>
        </section>
      </section>
    </main>
  );
}
