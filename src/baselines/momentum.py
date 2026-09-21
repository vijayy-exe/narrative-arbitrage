import pandas as pd
import numpy as np
import yaml
from pathlib import Path

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates

def calculate_metrics(returns):
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    # Sortino: like Sharpe but only penalizes downside deviation (returns
    # below 0), not upside volatility -- required by the dev plan alongside
    # Sharpe/max drawdown, previously missing from this function entirely.
    downside = returns[returns < 0]
    downside_std = downside.std()
    # downside_std is NaN when there are 0-1 downside observations (no
    # negative returns at all, or exactly one), not just when it's exactly
    # 0 -- NaN is truthy in Python, so a bare `if downside_std` guard alone
    # does not catch it.
    sortino = (
        (returns.mean() / downside_std) * np.sqrt(252)
        if pd.notna(downside_std) and downside_std != 0
        else 0
    )

    cum_return = (1 + returns).prod() - 1

    # Max Drawdown
    cum_rets = (1 + returns).cumprod()
    running_max = cum_rets.cummax()
    drawdown = (cum_rets - running_max) / running_max
    max_dd = drawdown.min()

    return {
        "Cumulative Return": cum_return,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Max Drawdown": max_dd
    }

def run_momentum_strategy(symbol, window=20):
    print(f"Running momentum strategy for {symbol}...")
    price_path = Path(f"data/raw/price/{symbol}.csv")
    if not price_path.exists():
        print(f"Price data for {symbol} not found. Skipping.")
        return None

    df = pd.read_csv(price_path, index_col='date', parse_dates=True).sort_index()
    df['returns'] = df['Close'].pct_change()

    # Momentum: Long if trailing window return is positive
    df['momentum'] = df['Close'].pct_change(window)
    df['position'] = np.where(df['momentum'] > 0, 1, -1)

    # Strategy returns: position at t-1 * return at t
    df['strat_returns'] = df['position'].shift(1) * df['returns']
    strat_returns = df['strat_returns'].dropna()

    metrics = calculate_metrics(strat_returns)
    metrics["Start Date"] = strat_returns.index.min().date().isoformat()
    metrics["End Date"] = strat_returns.index.max().date().isoformat()
    metrics["N Days"] = len(strat_returns)
    return metrics

def main():
    assets, dates = load_config()

    all_assets = []
    for category in assets.values():
        for asset in category:
            all_assets.append(asset.get('ticker') or asset.get('symbol') or asset.get('id'))

    results = {}
    for symbol in all_assets:
        res = run_momentum_strategy(symbol)
        if res:
            results[symbol] = res

    results_df = pd.DataFrame(results).T
    print("\n--- Momentum Strategy Results ---")
    print(results_df)

    # Persist a report -- the dev plan explicitly asks for "a small report
    # (markdown or notebook)" with the metrics and the exact date range used
    # (for later look-ahead-bias auditing); previously this only printed to
    # the console and nothing was saved.
    report_dir = Path("docs/baseline_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "momentum_report.md"
    with open(report_path, "w") as f:
        f.write("# Momentum Baseline Report\n\n")
        f.write(
            "Naive price-momentum strategy: long if trailing-window return "
            "is positive, short otherwise. Uses price only, no attention "
            "features -- this is the sanity-check floor, not the project's "
            "actual method.\n\n"
        )
        f.write(f"Window: {20} trading days\n\n")
        f.write("## Per-asset date range and metrics\n\n")
        f.write(results_df.to_markdown())
        f.write("\n")
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
