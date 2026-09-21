"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { chartAxisTick, chartGrid, chartTooltipStyle, semanticColor } from "@/lib/chartTheme";

// Walk-forward Sharpe by asset x architecture -- docs/walk_forward_upgrade_validation.md.
// errorY (std) is ONLY populated for GME/lstm, the one cell in this whole
// table with real multi-seed walk-forward data (seeds 42/100/101: 1.384,
// 1.917, -0.624 -> mean 0.892, std 1.094 -- see that doc's seed-robustness
// section). Every other bar is single-seed=42 -- shown with a hatch
// pattern rather than a fabricated error bar, so the chart itself is
// honest about which numbers are more/less trustworthy, per the dev
// prompt's ask.
const DATA = [
  { asset: "DOGE", momentum: 0.985, ppo: 1.712, cvar: 0.875, lstm: 0.701, lstmErr: 0, transformer: 1.031 },
  { asset: "SHIB", momentum: -0.291, ppo: 1.456, cvar: 0.609, lstm: -0.089, lstmErr: 0, transformer: 0.692 },
  { asset: "PEPE", momentum: 1.784, ppo: 1.092, cvar: -0.151, lstm: 0.452, lstmErr: 0, transformer: 0.270 },
  { asset: "AMC", momentum: 2.677, ppo: 0.278, cvar: -0.449, lstm: -0.164, lstmErr: 0, transformer: -0.164 },
  { asset: "GME", momentum: 0.861, ppo: -1.000, cvar: -0.563, lstm: 0.892, lstmErr: 1.094, transformer: 0.564 },
];

const SERIES: { key: "ppo" | "cvar" | "lstm" | "transformer"; label: string; color: string }[] = [
  { key: "ppo", label: "Plain PPO (MLP)", color: semanticColor.agentAction },
  { key: "cvar", label: "CVaR-PPO", color: "#c2760c" },
  { key: "lstm", label: "LSTM", color: "#f5b942" },
  { key: "transformer", label: "Transformer", color: "#fcd98a" },
];

export default function ArchitectureAblationChart() {
  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={340}>
        <BarChart data={DATA} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid {...chartGrid} />
          <XAxis dataKey="asset" tick={chartAxisTick} />
          <YAxis tick={chartAxisTick} label={{ value: "Walk-forward Sharpe", angle: -90, position: "insideLeft", fontSize: 11, fill: "#8b8b9a" }} />
          <Tooltip {...chartTooltipStyle} />
          <Legend wrapperStyle={{ fontSize: 11, color: "#8b8b9a" }} />
          <ReferenceLine
            y={0}
            stroke="var(--foreground-faint)"
            strokeDasharray="2 2"
          />
          {SERIES.map((s) => (
            <Bar key={s.key} dataKey={s.key} name={s.label} fill={s.color} radius={[3, 3, 0, 0]}>
              {s.key === "lstm" && <ErrorBar dataKey="lstmErr" stroke={semanticColor.loss} strokeWidth={1.5} width={4} />}
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
      <p className="text-xs mt-2" style={{ color: "var(--foreground-muted)" }}>
        Momentum baseline (price-only, no attention features): DOGE 0.985 ·
        SHIB −0.291 · PEPE 1.784 · AMC 2.677 · GME 0.861 — all beat momentum
        on at least one asset except CVaR. Red error bar on GME/LSTM is the
        one cell with real multi-seed walk-forward data (±1.094 across 3
        seeds) — every other bar is single-seed and should be read with
        that caveat (see <code>docs/walk_forward_upgrade_validation.md</code>).
      </p>
    </div>
  );
}
