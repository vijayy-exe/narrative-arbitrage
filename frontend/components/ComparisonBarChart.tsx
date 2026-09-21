"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export type ComparisonBar = {
  label: string;
  value: number;
  highlight?: "baseline" | "best" | "worst" | "default";
};

const COLOR = {
  baseline: "#71717a", // zinc-500 -- the number every bar is measured against
  best: "#16a34a", // green-600
  worst: "#dc2626", // red-600
  default: "#2563eb", // blue-600
};

/** Horizontal bar chart for comparing one metric (e.g. walk-forward Sharpe)
 * across several named methods/variants -- built for
 * docs/walk_forward_upgrade_validation.md's headline comparison table,
 * reused wherever this project needs a "several methods, one metric" bar
 * chart (kept generic rather than DOGE/GME-specific). All data is passed
 * in as props, sourced from the real docs tables -- nothing here computes
 * or invents a number. */
export default function ComparisonBarChart({
  data,
  metricLabel,
  height = 280,
}: {
  data: ComparisonBar[];
  metricLabel: string;
  height?: number;
}) {
  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={height}>
        <BarChart data={data} layout="vertical" margin={{ top: 4, right: 24, left: 8, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.25} horizontal={false} />
          <XAxis type="number" tick={{ fontSize: 11 }} label={{ value: metricLabel, position: "insideBottom", offset: -2, fontSize: 11 }} />
          <YAxis type="category" dataKey="label" tick={{ fontSize: 12 }} width={150} />
          <ReferenceLine x={0} stroke="currentColor" opacity={0.3} />
          <Tooltip
            contentStyle={{ fontSize: 12 }}
            formatter={(value: number) => value.toFixed(3)}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} isAnimationActive={false}>
            {data.map((d, i) => (
              <Cell key={i} fill={COLOR[d.highlight ?? "default"]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
