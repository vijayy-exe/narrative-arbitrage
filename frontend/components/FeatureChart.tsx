"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FeatureHistoryPoint } from "@/lib/api";
import { chartAxisTick, chartGrid, chartLegendStyle, chartTooltipStyle, semanticColor } from "@/lib/chartTheme";

// Attention-derivative data uses the shared --color-attention-signal family
// (see app/globals.css) -- every raw signal source in this app reuses these
// same three tints so a viewer learns "this hue = something the agent SAW,"
// distinct from agent-action or gain/loss colors used elsewhere.
const COLORS = [semanticColor.attentionSignal, semanticColor.attentionSignal2, semanticColor.attentionSignal3, semanticColor.agentAction];

export default function FeatureChart({
  title,
  history,
  seriesKeys,
}: {
  title: string;
  history: FeatureHistoryPoint[];
  seriesKeys: string[];
}) {
  const data = history.map((h) => ({ date: h.date, ...h.features }));

  return (
    <div className="w-full">
      <h3 className="text-sm font-medium mb-2" style={{ color: "var(--foreground-muted)" }}>
        {title}
      </h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data} margin={{ top: 4, right: 8, left: -16, bottom: 0 }}>
          <CartesianGrid {...chartGrid} />
          <XAxis dataKey="date" tick={chartAxisTick} minTickGap={40} />
          <YAxis tick={chartAxisTick} width={48} />
          <Tooltip {...chartTooltipStyle} />
          <Legend wrapperStyle={chartLegendStyle} />
          {seriesKeys.map((key, i) => (
            <Line
              key={key}
              type="monotone"
              dataKey={key}
              stroke={COLORS[i % COLORS.length]}
              dot={false}
              connectNulls
              strokeWidth={1.5}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
