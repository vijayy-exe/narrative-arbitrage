"use client";

import { useEffect, useState } from "react";
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
import BacktestTable from "@/components/BacktestTable";
import FeatureChart from "@/components/FeatureChart";
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

export default function AssetDashboard({ symbol }: { symbol: string }) {
  const [mode, setMode] = useState<Mode>("replay");
  const [features, setFeatures] = useState<FeaturesResponse | null>(null);
  const [decision, setDecision] = useState<DecisionResponse | null>(null);
  const [backtest, setBacktest] = useState<BacktestResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    Promise.all([getFeatures(symbol, mode), getDecision(symbol, mode)])
      .then(([f, d]) => {
        if (cancelled) return;
        setFeatures(f);
        setDecision(d);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
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
          <div className="flex items-center gap-3 mb-3">
            <span className="text-xs uppercase tracking-wide text-zinc-500">
              Current decision ({decision.mode})
            </span>
            <MethodBadge method={decision.method} />
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-3 text-sm">
            <Stat label="Target position" value={decision.target_position.toFixed(3)} />
            <Stat label="Price" value={decision.price.toPrecision(6)} />
            <Stat label="As of" value={decision.date ?? decision.timestamp ?? "—"} />
          </div>
          <p className="text-xs text-zinc-500">{decision.reasoning}</p>
        </section>
      )}

      {features && (
        <section className="mb-10 space-y-8">
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

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-zinc-500">{label}</div>
      <div className="font-mono">{value}</div>
    </div>
  );
}
