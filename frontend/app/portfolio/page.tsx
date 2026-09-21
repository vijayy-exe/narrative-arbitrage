"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

import {
  type Decision,
  type Mode,
  type PortfolioResponse,
  getDecision,
  getPortfolio,
} from "@/lib/api";
import MethodBadge from "@/components/MethodBadge";
import ModeToggle from "@/components/ModeToggle";

export default function PortfolioPage() {
  const [mode, setMode] = useState<Mode>("replay");
  const [portfolio, setPortfolio] = useState<PortfolioResponse | null>(null);
  const [decisions, setDecisions] = useState<Record<string, Decision[]>>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setError(null);
    getPortfolio(mode)
      .then(async (pf) => {
        if (cancelled) return;
        setPortfolio(pf);
        const entries = await Promise.all(
          Object.keys(pf.assets).map(async (symbol) => {
            try {
              const d = await getDecision(symbol, mode);
              // `d.history` already ends with the current decision (`d`
              // itself) -- most-recent-first for display, not duplicated.
              const log = d.history.length > 0 ? d.history : [d];
              return [symbol, [...log].reverse().slice(0, 10)] as const;
            } catch {
              return [symbol, []] as const;
            }
          })
        );
        if (!cancelled) setDecisions(Object.fromEntries(entries));
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [mode]);

  return (
    <main className="mx-auto max-w-4xl px-6 py-10">
      <Link href="/" className="text-sm text-zinc-500 hover:underline">
        &larr; All assets
      </Link>
      <div className="flex items-center justify-between mt-2 mb-6">
        <h1 className="text-2xl font-semibold">Simulated portfolio</h1>
        <ModeToggle mode={mode} onChange={setMode} />
      </div>

      {error && <p className="text-sm text-red-600 mb-6">{error}</p>}

      {portfolio && (
        <>
          <p className="text-sm text-zinc-500 mb-4">
            Total simulated portfolio value across all assets:{" "}
            <span className="font-mono text-zinc-900 dark:text-zinc-100">
              ${portfolio.total_portfolio_value.toFixed(2)}
            </span>{" "}
            ({portfolio.mode} mode -- a local simulation only, no real orders
            are ever placed).
          </p>

          <div className="overflow-x-auto mb-10">
            <table className="min-w-full text-sm border-collapse">
              <thead>
                <tr className="text-left border-b border-zinc-200 dark:border-zinc-800">
                  <th className="py-2 pr-4 font-medium text-zinc-500">Asset</th>
                  <th className="py-2 pr-4 font-medium text-zinc-500">Method</th>
                  <th className="py-2 pr-4 font-medium text-zinc-500">Position</th>
                  <th className="py-2 pr-4 font-medium text-zinc-500">Last price</th>
                  <th className="py-2 pr-4 font-medium text-zinc-500">Unrealized P&amp;L</th>
                  <th className="py-2 pr-4 font-medium text-zinc-500">Portfolio value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(portfolio.assets).map(([symbol, pf]) => (
                  <tr key={symbol} className="border-b border-zinc-100 dark:border-zinc-900">
                    <td className="py-1.5 pr-4">
                      <Link href={`/asset/${symbol}`} className="hover:underline">
                        {symbol}
                      </Link>
                    </td>
                    <td className="py-1.5 pr-4">
                      <MethodBadge method={pf.method} />
                    </td>
                    <td className="py-1.5 pr-4 font-mono">{pf.position.toFixed(3)}</td>
                    <td className="py-1.5 pr-4 font-mono">
                      {pf.last_price != null ? pf.last_price.toPrecision(6) : "—"}
                    </td>
                    <td
                      className={
                        "py-1.5 pr-4 font-mono " +
                        (pf.unrealized_pnl >= 0 ? "text-emerald-600" : "text-red-600")
                      }
                    >
                      {pf.unrealized_pnl.toFixed(2)}
                    </td>
                    <td className="py-1.5 pr-4 font-mono">{pf.cash.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="text-lg font-medium mb-3">Decision log</h2>
          <div className="space-y-6">
            {Object.entries(decisions).map(([symbol, log]) => (
              <div key={symbol}>
                <h3 className="text-sm font-medium mb-2">{symbol}</h3>
                {log.length === 0 ? (
                  <p className="text-xs text-zinc-500">No decisions logged yet.</p>
                ) : (
                  <ul className="text-xs space-y-1">
                    {log.map((d, i) => (
                      <li key={i} className="flex gap-3 text-zinc-600 dark:text-zinc-400">
                        <span className="font-mono w-24 shrink-0">
                          {d.date ?? d.timestamp ?? "—"}
                        </span>
                        <span className="w-20 shrink-0 uppercase">
                          {d.method === "ppo" ? "PPO" : "momentum"}
                        </span>
                        <span className="font-mono w-14 shrink-0">
                          {d.target_position.toFixed(3)}
                        </span>
                        <span className="truncate">{d.reasoning}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </main>
  );
}
