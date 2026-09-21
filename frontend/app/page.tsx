"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import { type AssetInfo, listAssets } from "@/lib/api";
import MethodBadge from "@/components/MethodBadge";

// The headline walk-forward numbers are static (sourced directly from
// docs/walk_forward_upgrade_validation.md, not invented) -- they describe
// a completed research result, not live state, so they don't belong behind
// a live fetch the way asset list/portfolio data does.
const HEADLINE_SHARPE = 0.712;
const HEADLINE_LABEL = "walk-forward Sharpe, original PPO";
const BASKET_SIZE = 5;

export default function Home() {
  const [assets, setAssets] = useState<AssetInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listAssets()
      .then((r) => setAssets(r.assets))
      .catch((e) => setError(String(e)));
  }, []);

  return (
    <main className="mx-auto max-w-5xl px-6 py-10">
      <section className="mb-12 border-b pb-10" style={{ borderColor: "var(--border)" }}>
        <p className="text-xs uppercase tracking-widest mb-3 font-numeric" style={{ color: "var(--color-attention-signal)" }}>
          Attention-derivative RL trading research
        </p>
        <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight mb-3 max-w-2xl">
          Does attention lead price?
        </h1>
        <p className="text-sm max-w-2xl mb-6" style={{ color: "var(--foreground-muted)" }}>
          An RL agent that trades on Wikipedia edit velocity and Google Trends
          slope, never a price chart — plus every upgrade built on top of it,
          honestly re-validated, including the parts that didn&apos;t survive.
        </p>
        <div className="flex flex-wrap gap-6">
          <Stat label="Assets traded" value={String(BASKET_SIZE)} color="var(--foreground)" />
          <Stat
            label={HEADLINE_LABEL}
            value={HEADLINE_SHARPE.toFixed(3)}
            color="var(--color-gain)"
          />
          <Stat label="RL variants tested" value="9" color="var(--color-agent-action)" />
          <Stat label="Tests passing" value="154/154" color="var(--color-attention-signal)" />
        </div>
        <div className="flex gap-3 mt-7">
          <Link
            href="/research"
            className="text-sm px-4 py-2 rounded-lg font-medium transition-colors"
            style={{ background: "var(--color-attention-signal)", color: "#03131c" }}
          >
            Read the research journey →
          </Link>
          <Link
            href="/lab"
            className="text-sm px-4 py-2 rounded-lg font-medium transition-colors border"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            Explore the data →
          </Link>
        </div>
      </section>

      <p className="text-sm mb-6" style={{ color: "var(--foreground-muted)" }}>
        Live attention-derivative signals, agent decisions, and walk-forward
        backtest comparison, per asset.{" "}
        {assets && <AssetMethodSummary assets={assets} />}
      </p>

      {error && (
        <p className="text-sm mb-6" style={{ color: "var(--color-loss)" }}>
          Could not reach the backend at the configured API base URL ({error}).
          Is <code className="text-xs font-numeric">uvicorn backend.main:app --port 8000</code> running?
        </p>
      )}

      {!assets && !error && <p className="text-sm" style={{ color: "var(--foreground-muted)" }}>Loading assets…</p>}

      <ul className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {assets?.map((a) => (
          <li key={a.symbol}>
            <Link
              href={`/asset/${a.symbol}`}
              className="flex items-center justify-between rounded-xl border px-4 py-3 transition-colors group"
              style={{ borderColor: "var(--border)", background: "var(--background-elevated)" }}
            >
              <div>
                <div className="font-medium font-numeric group-hover:opacity-80">{a.symbol}</div>
                <div className="text-xs uppercase" style={{ color: "var(--foreground-faint)" }}>{a.category}</div>
              </div>
              <MethodBadge method={a.method} />
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}

function Stat({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div>
      <div className="text-2xl font-semibold font-numeric" style={{ color }}>
        {value}
      </div>
      <div className="text-xs mt-0.5" style={{ color: "var(--foreground-faint)" }}>
        {label}
      </div>
    </div>
  );
}

/**
 * Which assets actually have a trained PPO checkpoint vs. the momentum
 * baseline -- read from the live `assets` response (a.method, set by the
 * backend from has_trained_policy()) rather than hardcoded, so this stays
 * true automatically as more checkpoints get trained.
 */
function AssetMethodSummary({ assets }: { assets: AssetInfo[] }) {
  const trained = assets.filter((a) => a.method === "ppo").map((a) => a.symbol);
  const baseline = assets.filter((a) => a.method !== "ppo").map((a) => a.symbol);

  if (trained.length === 0) {
    return <>No asset has a trained PPO checkpoint yet -- every asset below uses the momentum baseline.</>;
  }
  if (baseline.length === 0) {
    return <>Every asset below has a trained PPO checkpoint.</>;
  }
  return (
    <>
      {trained.join(", ")} {trained.length === 1 ? "has" : "have"} a trained PPO
      checkpoint; {baseline.join(", ")} {baseline.length === 1 ? "uses" : "use"} the
      momentum baseline, clearly labeled below.
    </>
  );
}
