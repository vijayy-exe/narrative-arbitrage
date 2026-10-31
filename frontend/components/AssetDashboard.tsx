"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

import {
  type BacktestResponse,
  type DecisionResponse,
  type FeaturesResponse,
  type Mode,
  getBacktest,
  getDecision,
  getFeatures,
} from "@/lib/api";
import AgentDecisionStrip from "@/components/AgentDecisionStrip";
import BacktestTable from "@/components/BacktestTable";
import FeatureChart from "@/components/FeatureChart";
import LiveIndicator from "@/components/LiveIndicator";
import MethodBadge from "@/components/MethodBadge";
import ModeToggle from "@/components/ModeToggle";

// Attention-derivative series worth charting, grouped by source -- kept to
// a couple of charts per the dev plan's "don't over-build the UI" note,
// not every column in docs/feature_dictionary.md.
const CHART_GROUPS: { title: string; keys: string[] }[] = [
  { title: "Wikipedia pageviews (level + velocity)", keys: ["pageviews", "pageviews_velocity"] },
  { title: "Google Trends search interest", keys: ["search_interest", "search_interest_velocity"] },
  { title: "Hacker News attention (stories, points)", keys: ["hn_story_count", "hn_points"] },
];

// Poll cadence per mode -- NOT the same number for both, because the
// underlying data genuinely moves at different real speeds: replay
// advances one historical day every REPLAY_SECONDS_PER_DAY=3s
// (backend/replay.py), while live attention signals only refresh every
// --interval-hours=6 (src/realtime/scheduler.py default). Polling live
// every 3s would just hit the same unchanged row for hours -- 60s is
// frequent enough to pick up a real refresh promptly without pretending
// data arrives faster than it does.
const POLL_INTERVAL_MS: Record<Mode, number> = { replay: 3000, live: 60_000 };

export default function AssetDashboard({ symbol }: { symbol: string }) {
  const [mode, setMode] = useState<Mode>("replay");
  const [features, setFeatures] = useState<FeaturesResponse | null>(null);
  const [decision, setDecision] = useState<DecisionResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Animated polish: flash the price stat gain/loss-colored when it
  // actually moves between polls, using the ref's PREVIOUS value (updated
  // in the effect below, after this render reads it) so the comparison is
  // against what was on screen a moment ago, not the value just fetched.
  const prevPriceRef = useRef<number | null>(null);
  const priceFlash: "gain" | "loss" | null =
    decision && prevPriceRef.current !== null && decision.price !== prevPriceRef.current
      ? decision.price > prevPriceRef.current
        ? "gain"
        : "loss"
      : null;
  useEffect(() => {
    if (decision) prevPriceRef.current = decision.price;
  }, [decision?.price]);

  useEffect(() => {
    let cancelled = false;
    setError(null);

    const poll = () => {
      Promise.all([getFeatures(symbol, mode), getDecision(symbol, mode)])
        .then(([f, d]) => {
          if (cancelled) return;
          setError(null);
          setFeatures(f);
          setDecision(d);
        })
        .catch((e) => !cancelled && setError(String(e)));
    };

    poll();
    const id = setInterval(poll, POLL_INTERVAL_MS[mode]);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [symbol, mode]);

  useEffect(() => {
    getBacktest(symbol)
      .then(setBacktest)
      .catch(() => setBacktest(null)); // e.g. NFT collections were never walk-forward backtested
  }, [symbol]);

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link href="/" className="text-sm text-zinc-500 hover:underline">
        &larr; All assets
      </Link>

      <div className="flex items-center justify-between mt-2 mb-6">
        <h1 className="text-2xl font-semibold">{symbol}</h1>
        <ModeToggle mode={mode} onChange={setMode} />
      </div>

      {error && (
        <p className="text-sm text-red-600 mb-6">
          {error}
          {mode === "live" && (
            <>
              {" "}
              -- no live data yet? Run{" "}
              <code className="text-xs">python -m src.realtime.scheduler --once</code> then{" "}
              <code className="text-xs">python -m src.realtime.infer</code>, or switch to Replay.
            </>
          )}
        </p>
      )}

      {decision && (
        <section className="mb-8 rounded-xl border border-zinc-200 dark:border-zinc-800 p-4">
          <div className="flex items-center justify-between gap-3 mb-3">
            <div className="flex items-center gap-3">
              <span className="text-xs uppercase tracking-wide text-zinc-500">
                Current decision ({decision.mode})
              </span>
              <MethodBadge method={decision.method} />
            </div>
            <LiveIndicator date={decision.date ?? decision.timestamp ?? "—"} mode={mode} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-3 text-sm">
            <Stat label="Target position" value={decision.target_position.toFixed(3)} />
            <Stat
              label="Price"
              value={decision.price.toPrecision(6)}
              flashKey={decision.price}
              flashColor={priceFlash}
            />
            <Stat label="As of" value={decision.date ?? decision.timestamp ?? "—"} />
          </div>
          <p className="text-xs text-zinc-500 mb-4">{decision.reasoning}</p>
          <div>
            <h3 className="text-xs uppercase tracking-wide text-zinc-500 mb-2">
              Decision history ({decision.history.length})
            </h3>
            <AgentDecisionStrip history={decision.history} />
          </div>
        </section>
      )}

      {features && (
        <section className="mb-10 space-y-8">
          <LiveIndicator date={features.date} mode={mode} />
          {CHART_GROUPS.map((g) => (
            <FeatureChart key={g.title} title={g.title} history={features.history} seriesKeys={g.keys} />
          ))}
        </section>
      )}

      <section>
        <h2 className="text-lg font-medium mb-3">
          {decision?.mode ?? mode} decision vs. Phase 5 walk-forward backtest
        </h2>
        {backtest ? (
          <BacktestTable data={backtest} />
        ) : (
          <p className="text-sm text-zinc-500">
            No walk-forward backtest results for {symbol} (see docs/backtest_report.md).
          </p>
        )}
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  flashKey,
  flashColor,
}: {
  label: string;
  value: string;
  flashKey?: string | number;
  flashColor?: "gain" | "loss" | null;
}) {
  const color =
    flashColor === "gain" ? "var(--color-gain)" : flashColor === "loss" ? "var(--color-loss)" : undefined;
  return (
    <div>
      <div className="text-xs text-zinc-500">{label}</div>
      <div
        key={flashKey !== undefined ? String(flashKey) : undefined}
        className={"font-mono " + (flashColor ? "animate-pulse-glow" : "")}
        style={color ? { color } : undefined}
      >
        {value}
      </div>
    </div>
  );
}
