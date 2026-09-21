# Live Loop — First Real Run (2026-09-20)

`docs/CURRENT_STATUS.md` originally flagged this as "the only phase with
zero runtime verification" — `src/realtime/scheduler.py`/`infer.py` had
unit tests against synthetic fixtures, but had never been run against real
live APIs. This document records the first real run and what it found.

## What was run

```
python -m src.realtime.scheduler --once   # fetch today's live data, all 5 assets
python -m src.realtime.infer              # make a PPO/momentum decision per asset
```

**Result: both worked end-to-end for the first time.** `data/live/live.db`
now has real rows in every table (`live_features`: 5, `live_prices`: 5,
`live_biz_signals`: 5 — the new 4chan signal from Upgrade 0 worked live on
its first real attempt, `decisions`: 7, `portfolio`: 5). The FastAPI
backend was verified to correctly serve this data (`/api/features/DOGE?
mode=live`, `/api/decision/DOGE?mode=live`, `/api/portfolio?mode=live` all
return 200 with real, live-fetched values) — closing the other half of the
original gap ("verify frontend's live views render something").

## A real, previously-undiscovered bug was found and fixed on the first attempt

`python -m src.realtime.infer` crashed on every asset with:
```
TypeError: unsupported operand type(s) for -: 'NoneType' and 'int'
```

**Root cause:** Wikipedia's pageviews API always lags 1-2 days behind real
time, so a live fetch for "today" is a `None` (JSON `null`) for every
asset, every single time this runs — not a rare failure mode, the NORMAL
case. `src/realtime/infer.py`'s `_live_rows_as_frame()` built a DataFrame
directly from the JSON-decoded payload without converting `None` to
`np.nan`, so `recompute_derivatives_on_index()`'s `Series.diff()` call hit
a `TypeError` trying to subtract across a `None`/`int` pair inside an
`object`-dtype column. **This means the live inference path, as originally
written, could never have completed a single real decision cycle** — it
was untested against real fetched data and the untested path was broken.
Fixed in `_live_rows_as_frame()` (coerce `None -> np.nan` explicitly);
regression test added
(`tests/realtime/test_infer.py::test_build_ppo_observation_handles_none_valued_live_features`,
the first test in this project to insert a live row with a `None` value
before calling `build_ppo_observation`). Full suite: 122/122 passing.

This is exactly the value of actually running code end-to-end rather than
trusting synthetic-fixture unit tests alone — every existing test passed
throughout this project's history while this bug sat live in the one code
path that mattered most for Phase 7's "real live demo" goal.

## What "a few days" of continuous operation still requires

Running the scheduler continuously for days (the dev plan's own Phase 7
ask, and `docs/CURRENT_STATUS.md`'s suggested next step) is not something
achievable inside a single interactive session regardless of effort — it
requires real wall-clock days. What's proven now: one full cycle works,
including the previously-fatal bug. To actually run it continuously,
either:

**Option A — foreground, via the scheduler's own built-in loop:**
```
cd ~/narrative-arbitrage && source .venv/bin/activate
python -m src.realtime.scheduler --interval-hours 6
```
Runs forever (or until killed), refreshing every 6 hours. Needs a terminal
that stays open (or `nohup ... &`/`tmux`/`screen` to survive logout).

**Option B — cron, refreshing every 6 hours, no terminal required:**
```
0 */6 * * * cd /Users/vijaysreeram/narrative-arbitrage && .venv/bin/python -m src.realtime.scheduler --once >> data/live/scheduler.log 2>&1
15 */6 * * * cd /Users/vijaysreeram/narrative-arbitrage && .venv/bin/python -m src.realtime.infer >> data/live/infer.log 2>&1
```
Add via `crontab -e`. The 15-minute offset between the two lines gives the
scheduler line time to finish before `infer` reads its output — this
project's actual live scheduler run above took several minutes end-to-end
for all 5 assets (Wikipedia/Trends/HN/4chan/price fetches, sequentially,
with politeness delays between assets), so 15 minutes is a safe buffer, not
an arbitrary number.

Neither option was left running by this session — that's a decision for
whoever owns this machine, not something to leave running unattended
without being asked.
