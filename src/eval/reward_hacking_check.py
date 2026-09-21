"""Phase 4a -- reward-hacking / policy-collapse check.

python -m src.eval.reward_hacking_check [--asset DOGE] [--model-path models/ppo_DOGE/model.zip]

Inspects a trained PPO policy's action distribution over its validation
window and flags whether it has collapsed to a degenerate strategy
(always-hold-cash, always-max-long, or always-max-short) rather than
learning something that actually responds to the input features -- any of
these collapsed modes can still show a positive Sharpe by luck on one
window, so this check is a required complement to the Sharpe/Sortino
numbers in validation_report.md, not a replacement for them.
"""
import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.agents.common import get_paths, get_train_val_split, select_default_asset
from src.agents.train_ppo import make_env, rollout_deterministic

# Thresholds are deliberately conservative (easy to trigger) -- this check
# is meant to flag "worth a human look", not to be a precise statistical test.
LOW_VARIANCE_STD_THRESHOLD = 0.05
COLLAPSE_FRACTION_THRESHOLD = 0.90
NEAR_ZERO = 0.1
NEAR_EXTREME = 0.9


def classify_action_distribution(actions):
    """Returns (is_collapsed: bool, label: str, stats: dict)."""
    actions = np.asarray(actions)
    mean, std = float(actions.mean()), float(actions.std())

    frac_near_zero = float(np.mean(np.abs(actions) < NEAR_ZERO))
    frac_near_max_long = float(np.mean(actions > NEAR_EXTREME))
    frac_near_max_short = float(np.mean(actions < -NEAR_EXTREME))

    stats = {
        "mean_action": mean,
        "std_action": std,
        "min_action": float(actions.min()),
        "max_action": float(actions.max()),
        "frac_near_zero": frac_near_zero,
        "frac_near_max_long": frac_near_max_long,
        "frac_near_max_short": frac_near_max_short,
    }

    if std >= LOW_VARIANCE_STD_THRESHOLD:
        return False, "no collapse detected (action varies with input)", stats

    if frac_near_zero >= COLLAPSE_FRACTION_THRESHOLD:
        return True, "COLLAPSED: always-hold-cash (flat)", stats
    if frac_near_max_long >= COLLAPSE_FRACTION_THRESHOLD:
        return True, "COLLAPSED: always-max-long", stats
    if frac_near_max_short >= COLLAPSE_FRACTION_THRESHOLD:
        return True, "COLLAPSED: always-max-short", stats

    # Low variance but not clearly pinned at one of the three named modes --
    # still worth flagging as "nearly constant", just not one of the
    # standard degenerate labels.
    return True, "COLLAPSED: near-constant action (not clearly one named mode)", stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    asset = args.asset or select_default_asset()
    model_path = Path(args.model_path or f"models/ppo_{asset}/model.zip")
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found -- train a policy first with "
            f"`python -m src.agents.train_ppo --asset {asset}`."
        )

    features_path, price_path = get_paths(asset)
    _, val_start, val_end = get_train_val_split(asset, args.train_frac)
    val_env = make_env(asset, features_path, price_path, start_date=val_start, end_date=val_end)

    model = PPO.load(model_path)
    _, actions = rollout_deterministic(model, val_env)

    is_collapsed, label, stats = classify_action_distribution(actions)

    print(f"Asset: {asset}")
    print(f"Model: {model_path}")
    print(f"Validation window: {val_start.date()} -> {val_end.date()} ({len(actions)} actions)")
    print()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    print()
    print(("FLAG: " if is_collapsed else "OK: ") + label)

    return is_collapsed


if __name__ == "__main__":
    main()
