# Narrative Arbitrage — Context Document: From Working Prototype to Research-Grade RL Project

## 1. Honest Diagnosis of Where It Stands

Based on the current status report, the project has a real, working pipeline (6 of 7 phases done, 84/90 tests passing, live loop code-complete). But the RL itself is currently **vanilla and shallow**, which is why the results are weak:

- **Plain PPO, single seed, single train/validate split** — no RecurrentPPO (skipped entirely), no hyperparameter search, no ensemble
- **Flat MLP policy network** on time-series features — the state is literally called "attention *velocity*" but the network has no architecture designed to model sequences or attention at all
- **Optimizes expected return only** — no risk-sensitivity in the objective itself, which is *exactly* why AMC (−2.03 Sharpe) and GME (−3.82 Sharpe) blew up: the agent was never told to care about tail risk, only average reward
- **Single-asset silos** — five independent policies, no joint/portfolio-level learning, so the agent can't learn correlation or diversification behavior across the basket
- **No offline pretraining** — PPO learns from scratch via pure exploration on a small, expensive-to-collect dataset, which is data-inefficient and likely why single-window and walk-forward Sharpe disagree so much per-asset
- **Reward hacking risk flagged but unresolved** for AMC/GME — the two worst performers are exactly the ones that need the most scrutiny before trusting any attribution on them

None of this is a criticism of the engineering — the pipeline, tests, and honesty about staleness (the DOGE-only issue) are genuinely solid. The gap is specifically in **RL sophistication**, which is the part that needs to be "best in field" for this to stand out.

## 2. What "Best in RL Field" Actually Means Here

Being state-of-the-art in applied RL for trading isn't about using the newest algorithm name — it's about addressing the specific failure modes visible in your own results. Each upgrade below is a direct response to something in your status report, not a generic add-on.

### 2.1 Risk-sensitive objective (directly fixes AMC/GME)
Standard PPO maximizes *expected* reward. A **distributional critic** (quantile regression, à la QR-DQN/IQN) or a **CVaR-constrained PPO** objective optimizes for the *worst-case tail* of the return distribution, not just the mean. This is the single highest-leverage change — it's precisely why an agent can look fine on average and still catastrophically blow up on specific assets, which is your current symptom exactly.

### 2.2 Attention-based state encoder (thematically and technically apt)
Right now the state (attention-velocity features) likely feeds a flat MLP. Since the entire premise of the project is "attention dynamics," using a **lightweight Transformer or temporal-attention encoder** to process the time-series of attention-derivative features — rather than treating each timestep as an unordered vector — is both technically stronger (captures how *acceleration patterns unfold*, not just point-in-time values) and a strong narrative hook for a write-up: "an attention mechanism trading on attention."

### 2.3 Recurrent memory, done properly (currently a skipped phase)
RecurrentPPO was cut, so half your original hypothesis ("does sequential memory help?") was never tested. Implementing it now — and directly comparing it against the Transformer encoder above — turns a scope cut into a genuine ablation study: MLP vs. LSTM-PPO vs. Transformer-PPO, on the same data, same protocol. That comparison alone is publishable-adjacent.

### 2.4 Joint / portfolio-level training (currently five silos)
Training one policy per asset independently means the agent can never learn "when DOGE and SHIB attention spike together, reduce size in both because they're correlated." Moving to a **single multi-asset policy** with a shared encoder and per-asset action heads (or a portfolio-allocation action space) is a meaningfully harder and more realistic RL problem, and matches how real quant desks actually operate.

### 2.5 Offline pretraining before online fine-tuning
Given how little real narrative-driven-asset history exists, pure from-scratch PPO is data-starved. Pretraining via **behavior cloning on the momentum/supervised baselines** (already built in Phase 2) or a lightweight **Decision Transformer** warm-start, then fine-tuning with PPO, is a well-established way to make limited data go further — and it directly explains why your single-window numbers are so unstable (3.95 → 1.71 walk-forward average for DOGE is a huge variance signal of undertrained/undersampled policies).

### 2.6 Reward hacking as a formal check, not a TODO
The status report already flags this as unresolved for AMC/GME. A **reward decomposition + hacking audit** (already scaffolded per `reward_hacking_check.py`) should be run and reported on before any of the above upgrades, since building risk-sensitivity on top of a hacked reward just hides the problem more convincingly.

### 2.7 Proper hyperparameter search
Single seed 42, single train/validate split is not defensible as "the result." A small **Optuna sweep** (learning rate, entropy coefficient, GAE lambda, network width) across a handful of seeds, reporting mean ± std Sharpe rather than one number, is table-stakes for any RL result to be taken seriously.

## 3. What This Buys You Academically

For an SOP / portfolio narrative, this reframes the project from "I trained PPO on some data" to:

> *"I identified that my initial agent was optimizing the wrong objective — expected return instead of risk-adjusted, tail-aware return — which explained why two of five assets produced catastrophic single-asset losses. I then redesigned the policy around a distributional, risk-sensitive objective, added an attention-based sequence encoder matching the problem's own premise, and ran a controlled ablation across memory architectures (MLP vs. recurrent vs. Transformer) to isolate what actually helped."*

That is a genuinely strong research narrative — the failure you already found (AMC/GME) is not a weakness to hide, it's the origin story for every upgrade in the implementation document that follows.

## 4. Honest Scope Note

Not all of this needs to happen for the project to be "best in field" — the highest-leverage subset is: **(1) risk-sensitive objective, (2) attention-based encoder, (3) RecurrentPPO comparison, (4) reward hacking audit.** Joint portfolio training and Decision Transformer pretraining are the reach items — strong to include if time allows, fine to describe as "future work" if not. The implementation document sequences them by leverage, not by difficulty, so you can stop at any point and still have a coherent story.
