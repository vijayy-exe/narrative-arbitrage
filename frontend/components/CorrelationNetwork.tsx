"use client";

import { useMemo, useState } from "react";

// Self-contained force-directed layout -- no external graph library. With
// only 5 nodes (the tradeable basket), a full force-simulation package
// (react-force-graph, d3-force) is more dependency weight than the problem
// needs; a small hand-rolled spring simulation gets the same "clusters
// emerge visually" result docs/joint_portfolio_validation.md's finding
// calls for, at zero added bundle cost.

export interface NetworkNode {
  id: string;
  category: string;
  size?: number;
}

export interface NetworkEdge {
  source: string;
  target: string;
  correlation: number; // -1..1, from the joint portfolio policy's action correlations
}

const CATEGORY_COLOR: Record<string, string> = {
  crypto: "var(--color-attention-signal)",
  stock: "var(--color-agent-action)",
};

const WIDTH = 520;
const HEIGHT = 380;
const EDGE_THRESHOLD = 0.15; // only draw edges above this |correlation| to avoid clutter, per the dev prompt

function simulateLayout(nodes: NetworkNode[], edges: NetworkEdge[], iterations = 400) {
  const positions: Record<string, { x: number; y: number; vx: number; vy: number }> = {};
  const cx = WIDTH / 2;
  const cy = HEIGHT / 2;
  nodes.forEach((n, i) => {
    const angle = (2 * Math.PI * i) / nodes.length;
    positions[n.id] = {
      x: cx + 100 * Math.cos(angle),
      y: cy + 100 * Math.sin(angle),
      vx: 0,
      vy: 0,
    };
  });

  const REPEL = 3200;
  const SPRING_K = 0.02;
  const DAMPING = 0.85;

  for (let iter = 0; iter < iterations; iter++) {
    for (const a of nodes) {
      let fx = 0;
      let fy = 0;
      for (const b of nodes) {
        if (a.id === b.id) continue;
        const dx = positions[a.id].x - positions[b.id].x;
        const dy = positions[a.id].y - positions[b.id].y;
        const distSq = Math.max(dx * dx + dy * dy, 25);
        const force = REPEL / distSq;
        const dist = Math.sqrt(distSq);
        fx += (dx / dist) * force;
        fy += (dy / dist) * force;
      }
      // Spring toward center (keeps the whole graph from drifting off-canvas)
      fx += (cx - positions[a.id].x) * 0.01;
      fy += (cy - positions[a.id].y) * 0.01;

      positions[a.id].vx = (positions[a.id].vx + fx) * DAMPING;
      positions[a.id].vy = (positions[a.id].vy + fy) * DAMPING;
    }
    for (const { source, target, correlation } of edges) {
      if (Math.abs(correlation) < EDGE_THRESHOLD) continue;
      const a = positions[source];
      const b = positions[target];
      const dx = b.x - a.x;
      const dy = b.y - a.y;
      const dist = Math.max(Math.sqrt(dx * dx + dy * dy), 1);
      // Higher |correlation| -> shorter target distance -> stronger pull
      const targetDist = 220 - Math.abs(correlation) * 160;
      const diff = dist - targetDist;
      const force = SPRING_K * diff * Math.sign(correlation === 0 ? 1 : correlation) * Math.abs(correlation);
      const fx = (dx / dist) * force;
      const fy = (dy / dist) * force;
      a.vx += fx;
      a.vy += fy;
      b.vx -= fx;
      b.vy -= fy;
    }
    for (const n of nodes) {
      positions[n.id].x += positions[n.id].vx;
      positions[n.id].y += positions[n.id].vy;
      positions[n.id].x = Math.max(30, Math.min(WIDTH - 30, positions[n.id].x));
      positions[n.id].y = Math.max(30, Math.min(HEIGHT - 30, positions[n.id].y));
    }
  }
  return positions;
}

export default function CorrelationNetwork({
  nodes,
  edges,
  onNodeClick,
}: {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
  onNodeClick?: (id: string) => void;
}) {
  const positions = useMemo(() => simulateLayout(nodes, edges), [nodes, edges]);
  const [hoveredEdge, setHoveredEdge] = useState<NetworkEdge | null>(null);
  const [hoveredNode, setHoveredNode] = useState<string | null>(null);

  return (
    <div className="relative">
      <svg width={WIDTH} height={HEIGHT} className="w-full h-auto" viewBox={`0 0 ${WIDTH} ${HEIGHT}`}>
        {edges
          .filter((e) => Math.abs(e.correlation) >= EDGE_THRESHOLD)
          .map((e) => {
            const a = positions[e.source];
            const b = positions[e.target];
            const isPositive = e.correlation > 0;
            const strokeWidth = 0.5 + Math.abs(e.correlation) * 5;
            const isHovered = hoveredEdge === e;
            return (
              <line
                key={`${e.source}-${e.target}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                stroke={isPositive ? "var(--color-attention-signal)" : "var(--color-loss)"}
                strokeWidth={isHovered ? strokeWidth + 1.5 : strokeWidth}
                opacity={isHovered ? 0.95 : 0.35 + Math.abs(e.correlation) * 0.35}
                onMouseEnter={() => setHoveredEdge(e)}
                onMouseLeave={() => setHoveredEdge(null)}
                style={{ cursor: "pointer" }}
              />
            );
          })}
        {nodes.map((n) => {
          const pos = positions[n.id];
          const r = 14 + (n.size ?? 1) * 6;
          return (
            <g
              key={n.id}
              transform={`translate(${pos.x},${pos.y})`}
              style={{ cursor: onNodeClick ? "pointer" : "default" }}
              onClick={() => onNodeClick?.(n.id)}
              onMouseEnter={() => setHoveredNode(n.id)}
              onMouseLeave={() => setHoveredNode(null)}
            >
              <circle
                r={r}
                fill="var(--background-elevated-2)"
                stroke={CATEGORY_COLOR[n.category] ?? "var(--color-price-reward)"}
                strokeWidth={hoveredNode === n.id ? 3 : 2}
              />
              <text
                textAnchor="middle"
                dy="0.35em"
                fontSize={12}
                fontFamily="var(--font-jetbrains-mono), monospace"
                fill="var(--foreground)"
              >
                {n.id}
              </text>
            </g>
          );
        })}
      </svg>
      <div className="mt-2 text-xs h-5" style={{ color: "var(--foreground-muted)" }}>
        {hoveredEdge
          ? `${hoveredEdge.source} ↔ ${hoveredEdge.target}: action correlation ${hoveredEdge.correlation.toFixed(3)}`
          : "Hover an edge for its exact correlation · edge color: blue = positive, red = negative · thickness = strength"}
      </div>
    </div>
  );
}
