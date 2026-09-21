"use client";

import ComparisonBarChart, { type ComparisonBar } from "@/components/ComparisonBarChart";

// Every number on this page is copied directly from this repo's own
// generated docs (docs/backtest_report.md, docs/risk_sensitivity_ablation.md,
// docs/architecture_ablation.md, docs/walk_forward_upgrade_validation.md,
// docs/multiseed_report.md, docs/joint_portfolio_validation.md) -- none is
// invented, estimated, or recomputed here. This page is static (not
// fetched from the backend) because it's a historical research narrative,
// not live/replay trading state -- unlike every other page in this app.

const SINGLE_WINDOW_SHARPE: ComparisonBar[] = [
  { label: "MLP (baseline)", value: -0.127, highlight: "baseline" },
  { label: "CVaR-PPO", value: 0.456, highlight: "best" },
  { label: "LSTM", value: 1.161, highlight: "best" },
  { label: "Transformer", value: 0.873, highlight: "best" },
];

const WALK_FORWARD_SHARPE: ComparisonBar[] = [
  { label: "Momentum (baseline)", value: 1.119, highlight: "baseline" },
  { label: "PPO / MLP (original)", value: 0.712, highlight: "best" },
  { label: "Transformer", value: 0.512, highlight: "worst" },
  { label: "LSTM", value: 0.451, highlight: "worst" },
  { label: "CVaR-PPO", value: 0.11, highlight: "worst" },
  { label: "Supervised / XGBoost", value: 0.065, highlight: "worst" },
];

const GME_SEED_CHECK: ComparisonBar[] = [
  { label: "Original PPO (baseline)", value: -1.0, highlight: "baseline" },
  { label: "LSTM, seed 42", value: 1.384, highlight: "default" },
  { label: "LSTM, seed 100", value: 1.917, highlight: "default" },
  { label: "LSTM, seed 101", value: -0.624, highlight: "worst" },
  { label: "LSTM, 3-member ensemble", value: 2.139, highlight: "best" },
];

const JOINT_PORTFOLIO: ComparisonBar[] = [
  { label: "Sum of 5 independent policies", value: 0.288, highlight: "baseline" },
  { label: "Joint portfolio policy", value: 1.38, highlight: "best" },
];

export default function ResearchPage() {
  return (
    <main className="mx-auto max-w-3xl px-6 py-10">
      <h1 className="text-2xl font-semibold mb-1">The Research Journey</h1>
      <p className="text-sm text-zinc-500 mb-10">
        How this project&apos;s RL upgrade path actually went — including
        the part where most of it didn&apos;t survive re-validation. Every
        number below is sourced from this repo&apos;s own docs/ reports.
      </p>

      <Section title="1. The hypothesis">
        <p>
          Attention derivatives (Wikipedia edit velocity, Google Trends
          search-interest acceleration) should lead price for
          narrative-driven assets. A PPO agent trained on nothing but these
          signals — never price technicals — should be able to beat a
          price-only momentum baseline, evaluated with leakage-audited
          walk-forward backtesting, not a single lucky window.
        </p>
      </Section>

      <Section title="2. It held for 2 of 5 assets, and failed hard for GME">
        <p>
          Walk-forward Sharpe: attention-only PPO beat momentum on DOGE
          (1.71 vs 0.98) and SHIB (1.46 vs −0.29), but lost on AMC, PEPE,
          and catastrophically on GME (−1.00 vs momentum&apos;s 0.86).
        </p>
      </Section>

      <Section title="3. Four upgrades looked like clear wins — single-window">
        <p>
          A risk-sensitive (CVaR) objective and two new architectures
          (Transformer, LSTM) were each trained on all 5 assets and
          evaluated on a single validation window — the same protocol the
          project&apos;s original checkpoints used. All three looked like
          unambiguous improvements over the plain MLP baseline:
        </p>
        <ComparisonBarChart data={SINGLE_WINDOW_SHARPE} metricLabel="Basket-avg Sharpe (single window)" height={200} />
      </Section>

      <Section title="4. Walk-forward re-validation reversed almost all of it">
        <p>
          Re-running all three under the SAME 48-roll walk-forward protocol
          that produced the original numbers told a completely different
          story. <strong>None of the three upgrades beat the original
          plain-MLP PPO.</strong> CVaR — the single-window winner on 4 of 5
          assets — is the worst RL variant tried under walk-forward.
        </p>
        <ComparisonBarChart data={WALK_FORWARD_SHARPE} metricLabel="Basket-avg Sharpe (walk-forward)" height={280} />
        <p className="mt-3 text-xs text-zinc-500">
          This is the single most important finding of the whole upgrade
          path: single-window ablation, in this project, consistently
          overstated real generalization — across every architecture and
          objective tried, without exception.
        </p>
      </Section>

      <Section title="5. The one real win isn't seed-robust — until it's an ensemble">
        <p>
          LSTM on GME specifically flipped from −1.00 to +1.38 under
          walk-forward — the largest improvement found anywhere in this
          project. Re-run at two more seeds, it doesn&apos;t fully hold:
          one comes back negative. But a 3-member seed ensemble
          (independently-seeded models, actions averaged at every step)
          doesn&apos;t just split the difference — it beats every
          individual seed tried, including the best one:
        </p>
        <ComparisonBarChart data={GME_SEED_CHECK} metricLabel="GME walk-forward Sharpe" height={260} />
        <p className="mt-3 text-xs text-zinc-500">
          This is the one piece of the whole upgrade path that clearly
          earns its extra complexity: ensembling a walk-forward-validated
          win, not adding another individually-untested architecture.
        </p>
      </Section>

      <Section title="6. And the joint multi-asset policy learned real structure">
        <p>
          A separate follow-up: a joint policy trading all 5 assets at
          once (instead of 5 independent single-asset policies) discovered
          real cross-asset structure with no explicit label: its per-asset
          actions on DOGE, PEPE, and SHIB — the three small-cap crypto
          assets — correlate 0.64–0.70 with each other, distinctly higher
          than any crypto/meme-stock pair.
        </p>
        <ComparisonBarChart data={JOINT_PORTFOLIO} metricLabel="Sharpe (single window)" height={160} />
        <p className="mt-3 text-xs text-zinc-500">
          Same &quot;smoother ratio, worse absolute outcome&quot; pattern
          seen with CVaR — better Sharpe/Sortino, worse cumulative return
          and drawdown. See docs/joint_portfolio_validation.md and
          docs/walk_forward_upgrade_validation.md for the full picture,
          including the seed-ensemble result for GME/LSTM.
        </p>
      </Section>

      <Section title="7. The honest bottom line">
        <p>
          This isn&apos;t a story of steady improvement. It&apos;s a story
          about how easy it is to fool yourself with a single validation
          window, and what it actually takes to find out — re-running
          everything under the protocol that matters, checking seeds, and
          reporting the result that contradicts your own earlier headline
          finding instead of quietly leaving it standing.
        </p>
      </Section>
    </main>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mb-10">
      <h2 className="text-base font-medium mb-2">{title}</h2>
      <div className="text-sm text-zinc-600 dark:text-zinc-400 space-y-2">{children}</div>
    </section>
  );
}
