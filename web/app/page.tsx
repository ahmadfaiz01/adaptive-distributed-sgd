"use client";

import { Activity, BarChart3, GitBranch, Play, RefreshCcw, ShieldAlert, SlidersHorizontal } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type Row = Record<string, string>;

const strategyColors: Record<string, string> = {
  sync_ps: "#2563eb",
  async_ps_ssp: "#0f9f6e",
  ring_allreduce: "#dc2626",
  adaptive: "#7c3aed"
};

const strategyNames: Record<string, string> = {
  sync_ps: "Sync Parameter Server",
  async_ps_ssp: "Async/SSP Parameter Server",
  ring_allreduce: "Ring AllReduce",
  adaptive: "Adaptive"
};

const workloadNames: Record<string, string> = {
  homogeneous_high_bw: "Homogeneous, high bandwidth",
  homogeneous_limited_bw: "Homogeneous, limited bandwidth",
  heterogeneous_high_bw: "Heterogeneous, high bandwidth",
  heterogeneous_limited_bw: "Heterogeneous, limited bandwidth",
  dynamic_mixed: "Dynamic mixed",
  worker_crash: "Worker crash"
};

const metricLabels: Record<string, string> = {
  mean_iteration_time_s: "Mean iteration time",
  p95_iteration_time_s: "P95 iteration time",
  throughput_samples_s: "Throughput",
  comm_fraction: "Communication fraction",
  straggler_impact_rate: "Straggler impact"
};

function parseCsv(text: string): Row[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const values = splitCsvLine(line);
    return Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""]));
  });
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      out.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  out.push(current);
  return out;
}

function n(row: Row, key: string): number {
  const value = Number(row[key]);
  return Number.isFinite(value) ? value : 0;
}

function fmt(value: number, unit = "") {
  if (!Number.isFinite(value)) return "n/a";
  if (Math.abs(value) >= 1000) return `${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}${unit}`;
  if (Math.abs(value) >= 10) return `${value.toFixed(2)}${unit}`;
  return `${value.toFixed(3)}${unit}`;
}

function unique(rows: Row[], key: string) {
  return Array.from(new Set(rows.map((row) => row[key]).filter(Boolean)));
}

function BarComparison({ rows, metric }: { rows: Row[]; metric: string }) {
  const max = Math.max(...rows.map((row) => n(row, metric)), 0.001);
  return (
    <div className="chart" role="img" aria-label="Strategy comparison bar chart">
      <svg viewBox="0 0 760 310" width="100%" height="100%">
        {[0, 1, 2, 3].map((tick) => (
          <line key={tick} x1="70" x2="730" y1={260 - tick * 60} y2={260 - tick * 60} stroke="#e5e7eb" />
        ))}
        {rows.map((row, index) => {
          const value = n(row, metric);
          const height = Math.max(4, (value / max) * 205);
          const x = 95 + index * 155;
          const y = 260 - height;
          const color = strategyColors[row.strategy] ?? "#64748b";
          return (
            <g key={`${row.strategy}-${row.workers}`}>
              <rect x={x} y={y} width="84" height={height} fill={color} rx="5" />
              <text x={x + 42} y={y - 8} textAnchor="middle" fontSize="13" fill="#18202a">
                {fmt(value)}
              </text>
              <text x={x + 42} y="287" textAnchor="middle" fontSize="12" fill="#647180">
                {row.strategy.replace("_", " ")}
              </text>
            </g>
          );
        })}
        <text x="18" y="45" transform="rotate(-90 18 45)" fontSize="12" fill="#647180">
          {metricLabels[metric]}
        </text>
      </svg>
    </div>
  );
}

function AdaptiveTimeline({ rows }: { rows: Row[] }) {
  const valid = rows.filter((row) => row.strategy === "adaptive" && row.iteration_time_s);
  const maxStep = Math.max(...valid.map((row) => n(row, "step")), 1);
  const maxIter = Math.max(...valid.map((row) => n(row, "iteration_time_s")), 0.001);
  const points = valid
    .map((row) => {
      const x = 55 + (n(row, "step") / maxStep) * 660;
      const y = 250 - (n(row, "iteration_time_s") / maxIter) * 205;
      return `${x},${y}`;
    })
    .join(" ");
  return (
    <div className="chart" role="img" aria-label="Adaptive mode timeline">
      <svg viewBox="0 0 760 310" width="100%" height="100%">
        {[0, 1, 2, 3].map((tick) => (
          <line key={tick} x1="55" x2="720" y1={250 - tick * 60} y2={250 - tick * 60} stroke="#e5e7eb" />
        ))}
        <polyline points={points} fill="none" stroke="#7c3aed" strokeWidth="3" />
        {valid
          .filter((row) => row.mode_switched === "1")
          .map((row) => {
            const x = 55 + (n(row, "step") / maxStep) * 660;
            return (
              <g key={`${row.workers}-${row.step}-${row.mode}`}>
                <line x1={x} x2={x} y1="35" y2="252" stroke="#111827" strokeDasharray="5 5" />
                <text x={x + 6} y="48" fontSize="12" fill="#111827">
                  switch to {row.mode}
                </text>
              </g>
            );
          })}
        <text x="380" y="292" textAnchor="middle" fontSize="12" fill="#647180">
          training step
        </text>
      </svg>
    </div>
  );
}

function StatusPill({ status }: { status: string }) {
  return <span className={`status ${status}`}>{status || "n/a"}</span>;
}

export default function Dashboard() {
  const [summary, setSummary] = useState<Row[]>([]);
  const [steps, setSteps] = useState<Row[]>([]);
  const [improvements, setImprovements] = useState<Row[]>([]);
  const [workload, setWorkload] = useState("dynamic_mixed");
  const [workers, setWorkers] = useState("4");
  const [metric, setMetric] = useState("mean_iteration_time_s");

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

  const workloads = useMemo(() => unique(summary, "workload"), [summary]);
  const workerOptions = useMemo(() => unique(summary, "workers").sort((a, b) => Number(a) - Number(b)), [summary]);
  const filteredSummary = useMemo(
    () => summary.filter((row) => row.workload === workload && row.workers === workers),
    [summary, workload, workers]
  );
  const filteredSteps = useMemo(
    () => steps.filter((row) => row.workload === workload && row.workers === workers && row.repeat === "0"),
    [steps, workload, workers]
  );
  const adaptive = filteredSummary.find((row) => row.strategy === "adaptive");
  const bestStatic = filteredSummary
    .filter((row) => row.strategy !== "adaptive" && row.status !== "failed")
    .sort((a, b) => n(a, "mean_iteration_time_s") - n(b, "mean_iteration_time_s"))[0];
  const improvement = improvements.find((row) => row.workload === workload && row.workers === workers);
  const failureRows = summary.filter((row) => row.workload === "worker_crash");

  return (
    <main className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <div className="brand-mark">SGD</div>
            <div>
              <p className="eyebrow">Parallel and Distributed Computing</p>
              <h1>Adaptive Distributed SGD Dashboard</h1>
            </div>
          </div>
          <div className="top-actions">
            <span className="pill">Parameter Server</span>
            <span className="pill">Ring AllReduce</span>
            <span className="pill">Adaptive C-6</span>
          </div>
        </div>
      </header>

      <section className="shell">
        <div className="controls">
          <div className="control">
            <label>Workload</label>
            <select value={workload} onChange={(event) => setWorkload(event.target.value)}>
              {workloads.map((item) => (
                <option key={item} value={item}>
                  {workloadNames[item] ?? item}
                </option>
              ))}
            </select>
          </div>
          <div className="control">
            <label>Workers</label>
            <select value={workers} onChange={(event) => setWorkers(event.target.value)}>
              {workerOptions.map((item) => (
                <option key={item} value={item}>
                  {item} worker processes
                </option>
              ))}
            </select>
          </div>
          <div className="control">
            <label>Metric</label>
            <select value={metric} onChange={(event) => setMetric(event.target.value)}>
              {Object.entries(metricLabels).map(([key, label]) => (
                <option key={key} value={key}>
                  {label}
                </option>
              ))}
            </select>
          </div>
          <div className="control">
            <label>Current comparison</label>
            <strong>{workloadNames[workload] ?? workload}</strong>
          </div>
        </div>

        <div className="grid three">
          <div className="panel metric">
            <h3>
              <Activity size={16} /> Adaptive iteration time
            </h3>
            <div className="metric-value">{fmt(n(adaptive ?? {}, "mean_iteration_time_s"), "s")}</div>
            <p className="muted">Lower is better. This is the main speed metric.</p>
          </div>
          <div className="panel metric">
            <h3>
              <BarChart3 size={16} /> Best static baseline
            </h3>
            <div className="metric-value">{bestStatic ? strategyNames[bestStatic.strategy] : "n/a"}</div>
            <p className="muted">{bestStatic ? `${fmt(n(bestStatic, "mean_iteration_time_s"), "s")} mean iteration time` : "No static baseline available"}</p>
          </div>
          <div className="panel metric">
            <h3>
              <RefreshCcw size={16} /> C-6 improvement
            </h3>
            <div className="metric-value">{improvement ? fmt(n(improvement, "adaptive_improvement_percent"), "%") : "n/a"}</div>
            <p className="muted">Positive means adaptive beat the best static strategy.</p>
          </div>
        </div>

        <div className="grid two" style={{ marginTop: 16 }}>
          <div className="panel">
            <h2>
              <SlidersHorizontal size={18} /> Strategy Comparison
            </h2>
            <BarComparison rows={filteredSummary} metric={metric} />
            <div className="legend">
              {Object.entries(strategyColors).map(([strategy, color]) => (
                <span key={strategy}>
                  <span className="dot" style={{ background: color }} />
                  {strategyNames[strategy]}
                </span>
              ))}
            </div>
          </div>

          <div className="panel">
            <h2>
              <GitBranch size={18} /> Adaptive Timeline
            </h2>
            <AdaptiveTimeline rows={filteredSteps} />
            <p className="explain">
              Vertical dashed lines are mode switches. The controller watches worker-time variation and moves from Ring to PS when stragglers appear.
            </p>
          </div>
        </div>

        <div className="grid two" style={{ marginTop: 16 }}>
          <div className="panel">
            <h2>
              <ShieldAlert size={18} /> Failure Scenario
            </h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Strategy</th>
                    <th>Workers</th>
                    <th>Status</th>
                    <th>Steps</th>
                    <th>Crash step</th>
                    <th>Recovery</th>
                  </tr>
                </thead>
                <tbody>
                  {failureRows.map((row) => (
                    <tr key={`${row.strategy}-${row.workers}`}>
                      <td>{strategyNames[row.strategy] ?? row.strategy}</td>
                      <td>{row.workers}</td>
                      <td>
                        <StatusPill status={row.status} />
                      </td>
                      <td>{row.steps_completed}</td>
                      <td>{row.failure_step || "none"}</td>
                      <td>{row.recovery_time_steps || "n/a"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="panel">
            <h2>
              <Play size={18} /> Demo Script
            </h2>
            <div className="timeline">
              <div className="event">
                <strong>1. Pick dynamic mixed</strong>
                Show that adaptive beats the best static baseline by switching modes.
              </div>
              <div className="event">
                <strong>2. Pick worker crash</strong>
                Show Ring fails, while PS and adaptive continue degraded.
              </div>
              <div className="event">
                <strong>3. Change the metric</strong>
                Compare iteration time, throughput, communication fraction, and straggler impact.
              </div>
            </div>
          </div>
        </div>

        <div className="panel" style={{ marginTop: 16 }}>
          <h2>What This Proves</h2>
          <div className="compare">
            <p className="explain">
              <strong>Homogeneous</strong> means all workers have similar speed. Ring AllReduce is usually strong there because every worker reaches the barrier together.
            </p>
            <p className="explain">
              <strong>Heterogeneous</strong> means workers differ in speed. A slow worker becomes a straggler, so synchronous Ring waits while PS-based methods tolerate the slowdown better.
            </p>
          </div>
          <p className="explain">
            The C-6 contribution is the adaptive controller: it changes synchronization strategy at runtime instead of choosing one static architecture before training starts.
          </p>
        </div>
      </section>
    </main>
  );
}
