"use client";

import { useRouter } from "next/navigation";

import ArchitectureAblationChart from "@/components/ArchitectureAblationChart";
import CorrelationNetwork, { type NetworkEdge, type NetworkNode } from "@/components/CorrelationNetwork";
import ParetoScatter, { type ParetoPoint } from "@/components/ParetoScatter";

// All data below is copied directly from this repo's own generated docs
// (docs/joint_portfolio_validation.md, docs/pareto_frontier.md,
// docs/walk_forward_upgrade_validation.md) -- none is invented or
// recomputed client-side. This page is static/historical research data,
// like /research, not live trading state.

const NETWORK_NODES: NetworkNode[] = [
  { id: "DOGE", category: "crypto", size: 1.1 },
  { id: "GME", category: "stock", size: 1.0 },
  { id: "AMC", category: "stock", size: 0.8 },
  { id: "PEPE", category: "crypto", size: 0.9 },
  { id: "SHIB", category: "crypto", size: 1.0 },
];

const NETWORK_EDGES: NetworkEdge[] = [
  { source: "DOGE", target: "GME", correlation: 0.258 },
  { source: "DOGE", target: "AMC", correlation: 0.264 },
  { source: "DOGE", target: "PEPE", correlation: 0.7 },
  { source: "DOGE", target: "SHIB", correlation: 0.665 },
  { source: "GME", target: "AMC", correlation: 0.397 },
  { source: "GME", target: "PEPE", correlation: -0.212 },
  { source: "GME", target: "SHIB", correlation: -0.011 },
  { source: "AMC", target: "PEPE", correlation: 0.106 },
  { source: "AMC", target: "SHIB", correlation: 0.166 },
  { source: "PEPE", target: "SHIB", correlation: 0.643 },
];

const PARETO_POINTS: ParetoPoint[] = [
  { asset: "GME", cvarAlpha: 0.9, sharpe: -0.9944, maxDrawdown: -0.0959 },
  { asset: "GME", cvarAlpha: 0.7, sharpe: -0.9839, maxDrawdown: -0.0955 },
  { asset: "GME", cvarAlpha: 0.5, sharpe: -0.8869, maxDrawdown: -0.0908 },
  { asset: "GME", cvarAlpha: 0.3, sharpe: -0.7037, maxDrawdown: -0.0824 },
  { asset: "GME", cvarAlpha: 0.1, sharpe: -0.6216, maxDrawdown: -0.0638 },
  { asset: "AMC", cvarAlpha: 0.9, sharpe: -0.0084, maxDrawdown: -0.1573 },
  { asset: "AMC", cvarAlpha: 0.7, sharpe: 0.0351, maxDrawdown: -0.1548 },
  { asset: "AMC", cvarAlpha: 0.5, sharpe: 0.1828, maxDrawdown: -0.1566 },
  { asset: "AMC", cvarAlpha: 0.3, sharpe: 0.1182, maxDrawdown: -0.1553 },
  { asset: "AMC", cvarAlpha: 0.1, sharpe: -1.1006, maxDrawdown: -0.1508 },
  { asset: "GME", cvarAlpha: NaN, sharpe: -0.9995, maxDrawdown: -0.1296, isReference: true, referenceLabel: "GME: original PPO" },
  { asset: "GME", cvarAlpha: 0.2, sharpe: -0.5626, maxDrawdown: -0.073, isReference: true, referenceLabel: "GME: CVaR α=0.2 (original)" },
  { asset: "AMC", cvarAlpha: NaN, sharpe: 0.278, maxDrawdown: -0.2314, isReference: true, referenceLabel: "AMC: original PPO" },
  { asset: "AMC", cvarAlpha: 0.2, sharpe: -0.449, maxDrawdown: -0.1546, isReference: true, referenceLabel: "AMC: CVaR α=0.2 (original)" },
];

export default function LabPage() {
  const router = useRouter();

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <h1 className="text-2xl font-semibold mb-1">The Lab</h1>
      <p className="text-sm mb-10" style={{ color: "var(--foreground-muted)" }}>
        Interactive views of this project&apos;s Tier 2 research — cross-asset
        structure, the CVaR risk/return tradeoff, and every architecture
        compared side by side. Click a node in the network below to jump to
        that asset.
      </p>

      <Section
        title="Cross-asset correlation — the joint policy's emergent structure"
        subtitle="No label ever told the policy which assets are crypto. Its own per-asset action correlations recover exactly that grouping."
      >
        <CorrelationNetwork
          nodes={NETWORK_NODES}
          edges={NETWORK_EDGES}
          onNodeClick={(id) => router.push(`/asset/${id}`)}
        />
      </Section>

      <Section
        title="Pareto frontier — CVaR's risk/return tradeoff, made visible"
        subtitle="GME: lower α helps Sharpe AND drawdown together. AMC: a real tradeoff peak — but no α beats the original PPO baseline (◆)."
      >
        <ParetoScatter points={PARETO_POINTS} />
      </Section>

      <Section
        title="Architecture ablation — every variant, every asset, walk-forward"
        subtitle="None beats original PPO at the basket level. LSTM wins on GME specifically — and only that cell has real multi-seed error bars."
      >
        <ArchitectureAblationChart />
      </Section>
    </main>
  );
}

function Section({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section className="mb-14 rounded-xl border p-6" style={{ borderColor: "var(--border)", background: "var(--background-elevated)" }}>
      <h2 className="text-base font-medium mb-1">{title}</h2>
      <p className="text-xs mb-5" style={{ color: "var(--foreground-muted)" }}>
        {subtitle}
      </p>
      {children}
    </section>
  );
}
