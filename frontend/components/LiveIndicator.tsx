"use client";

// A small "as of <date>" marker whose dot glows (animate-pulse-glow, see
// globals.css) exactly when `date` changes -- same remount-by-key
// mechanism as AgentDecisionStrip's newest entry. Honest about cadence:
// in live mode the underlying attention signals only refresh every 6h
// (src/realtime/scheduler.py's --interval-hours default), so this will
// glow rarely; in replay mode the clock advances every
// REPLAY_SECONDS_PER_DAY=3s (backend/replay.py), so it glows almost every
// poll. Either way the glow reflects a REAL change in `date`, not a timer.
export default function LiveIndicator({ date, mode }: { date: string; mode: "live" | "replay" }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs" style={{ color: "var(--foreground-muted)" }}>
      <span
        key={date}
        className="inline-block w-1.5 h-1.5 rounded-full animate-pulse-glow"
        style={{ background: "var(--color-attention-signal)", color: "var(--color-attention-signal)" }}
      />
      {mode} &middot; as of {date}
    </span>
  );
}
