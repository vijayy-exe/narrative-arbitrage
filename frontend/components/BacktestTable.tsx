import type { BacktestResponse } from "@/lib/api";

// Mirrors src/eval/walk_forward.py's aggregate_results() metric columns
// exactly, so this table shows precisely what docs/backtest_report.md
// reports -- no metric is invented or recomputed here.
const METRICS = [
  "Cumulative Return",
  "Sharpe Ratio",
  "Sortino Ratio",
  "Max Drawdown",
  "win_rate",
  "avg_trade_duration",
];

export default function BacktestTable({ data }: { data: BacktestResponse }) {
  const methods = Object.keys(data.methods);

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm border-collapse">
        <thead>
          <tr className="text-left border-b border-zinc-200 dark:border-zinc-800">
            <th className="py-2 pr-4 font-medium text-zinc-500">Metric</th>
            {methods.map((m) => (
              <th key={m} className="py-2 pr-4 font-medium text-zinc-500 capitalize">
                {m === "ppo" ? "PPO" : m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {METRICS.map((metric) => (
            <tr key={metric} className="border-b border-zinc-100 dark:border-zinc-900">
              <td className="py-1.5 pr-4 text-zinc-600 dark:text-zinc-400">{metric}</td>
              {methods.map((m) => {
                const v = data.methods[m]?.[metric];
                return (
                  <td key={m} className="py-1.5 pr-4 font-mono">
                    {typeof v === "number" ? v.toFixed(4) : "—"}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-zinc-500">
        Phase 5 walk-forward backtest (docs/backtest_report.md) -- mean
        out-of-sample metrics across all rolls, retrained from scratch each
        roll. Not affected by the live/replay toggle above.
      </p>
    </div>
  );
}
