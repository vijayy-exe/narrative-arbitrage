"use client";

import { useEffect, useRef } from "react";

import type { Decision } from "@/lib/api";
import { gainLossColor } from "@/lib/chartTheme";

// Horizontal timeline of the agent's own past decisions (decision.history
// from getDecision) -- what the agent DID at each step, color-coded by
// position sign via the shared gain/loss convention (app/globals.css /
// chartTheme.ts), not a new palette of its own.
//
// The newest entry gets `animate-pulse-glow` (defined in globals.css,
// unused until this component) via a `key` derived from its own date --
// React remounts that one DOM node whenever the date changes, so the CSS
// animation (which only plays once per mount) replays exactly when a
// genuinely new decision has landed, not on every poll tick that returns
// the same latest entry. See backend/replay.py's replay_decision, which
// already dedupes history entries by date for exactly this reason.
export default function AgentDecisionStrip({ history }: { history: Decision[] }) {
  const scrollerRef = useRef<HTMLDivElement>(null);
  const latestLabel = history[history.length - 1]?.date ?? history[history.length - 1]?.timestamp;

  // Keep the newest decision in view as the strip grows -- otherwise a
  // ticker that only ever appends to the right silently scrolls the thing
  // you actually care about (what the agent just did) off-screen.
  useEffect(() => {
    const el = scrollerRef.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [latestLabel]);

  if (history.length === 0) {
    return (
      <p className="text-sm" style={{ color: "var(--foreground-muted)" }}>
        No decision history yet.
      </p>
    );
  }

  return (
    <div ref={scrollerRef} className="flex gap-2 overflow-x-auto pb-2 scroll-smooth">
      {history.map((d, i) => {
        const isNewest = i === history.length - 1;
        const label = d.date ?? d.timestamp ?? "—";
        const color = gainLossColor(d.target_position);
        return (
          <div
            key={isNewest ? `newest-${label}` : `${label}-${i}`}
            className={
              "flex-shrink-0 w-16 rounded-md border px-2 py-1.5 text-center cursor-default " +
              (isNewest ? "animate-pulse-glow" : "")
            }
            style={{ borderColor: "var(--border)", background: "var(--background-elevated)" }}
            title={`${label} — ${d.reasoning}`}
          >
            <div className="font-numeric text-xs" style={{ color }}>
              {d.target_position >= 0 ? "+" : ""}
              {d.target_position.toFixed(2)}
            </div>
            <div
              className="text-[10px] mt-0.5 truncate"
              style={{ color: "var(--foreground-faint)" }}
            >
              {label.length >= 10 ? label.slice(5, 10) : label}
            </div>
          </div>
        );
      })}
    </div>
  );
}
