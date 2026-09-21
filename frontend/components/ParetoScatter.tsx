"use client";

import {
  CartesianGrid,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";

import { chartAxisTick, chartGrid, chartTooltipStyle, semanticColor } from "@/lib/chartTheme";

export interface ParetoPoint {
  asset: string;
  cvarAlpha: number;
  sharpe: number;
  maxDrawdown: number;
  isReference?: boolean;
  referenceLabel?: string;
}

const ASSET_COLOR: Record<string, string> = {
  GME: semanticColor.attentionSignal,
  AMC: semanticColor.agentAction,
};

export default function ParetoScatter({ points }: { points: ParetoPoint[] }) {
  const assets = Array.from(new Set(points.map((p) => p.asset)));

  return (
    <div className="w-full">
      <ResponsiveContainer width="100%" height={360}>
        <ScatterChart margin={{ top: 10, right: 24, left: 8, bottom: 24 }}>
          <CartesianGrid {...chartGrid} />
          <XAxis
            type="number"
            dataKey="maxDrawdown"
            name="Max Drawdown"
            tick={chartAxisTick}
            label={{ value: "Walk-forward Max Drawdown", position: "insideBottom", offset: -12, fontSize: 11, fill: "#8b8b9a" }}
          />
          <YAxis
            type="number"
            dataKey="sharpe"
            name="Sharpe"
            tick={chartAxisTick}
            label={{ value: "Walk-forward Sharpe", angle: -90, position: "insideLeft", fontSize: 11, fill: "#8b8b9a" }}
          />
          <ZAxis range={[80, 260]} />
          <Tooltip
            {...chartTooltipStyle}
            cursor={{ strokeDasharray: "3 3" }}
            content={({ payload }) => {
              if (!payload || !payload.length) return null;
              const p = payload[0].payload as ParetoPoint;
              return (
                <div style={chartTooltipStyle.contentStyle} className="px-3 py-2">
                  <div className="font-medium mb-1">
                    {p.isReference ? p.referenceLabel : `${p.asset}, α=${p.cvarAlpha}`}
                  </div>
                  <div>Sharpe: {p.sharpe.toFixed(4)}</div>
                  <div>Max Drawdown: {p.maxDrawdown.toFixed(4)}</div>
                </div>
              );
            }}
          />
          {assets.map((asset) => (
            <Scatter
              key={asset}
              name={asset}
              data={points.filter((p) => p.asset === asset && !p.isReference)}
              fill={ASSET_COLOR[asset] ?? semanticColor.neutral}
              shape="circle"
            />
          ))}
          <Scatter
            name="Reference points"
            data={points.filter((p) => p.isReference)}
            fill={semanticColor.priceReward}
            shape="diamond"
          />
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex gap-4 text-xs mt-1" style={{ color: "var(--foreground-muted)" }}>
        {assets.map((asset) => (
          <span key={asset} className="flex items-center gap-1.5">
            <span
              className="inline-block h-2 w-2 rounded-full"
              style={{ background: ASSET_COLOR[asset] ?? semanticColor.neutral }}
            />
            {asset} (α sweep, circles)
          </span>
        ))}
        <span className="flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2"
            style={{ background: semanticColor.priceReward, transform: "rotate(45deg)" }}
          />
          Reference points (◆ original PPO / CVaR α=0.2)
        </span>
      </div>
    </div>
  );
}
