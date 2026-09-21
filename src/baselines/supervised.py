import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data_pipeline.features import recompute_derivatives_on_index

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates

def calculate_metrics(returns):
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    # Sortino: penalizes only downside deviation -- required by the dev
    # plan alongside Sharpe/max drawdown, previously missing.
    downside = returns[returns < 0]
    downside_std = downside.std()
    # See momentum.py's calculate_metrics for why pd.notna() is needed here
    # (downside_std is NaN, not 0, with 0-1 downside observations, and NaN
    # is truthy).
    sortino = (
        (returns.mean() / downside_std) * np.sqrt(252)
        if pd.notna(downside_std) and downside_std != 0
        else 0
    )

    cum_return = (1 + returns).prod() - 1

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

def run_supervised_strategy(symbol):
    print(f"Running supervised strategy for {symbol}...")
    feature_path = Path(f"data/processed/{symbol}_features.csv")
    price_path = Path(f"data/raw/price/{symbol}.csv")
    
    if not feature_path.exists() or not price_path.exists():
        print(f"Data for {symbol} not found. Skipping.")
        return None
    
    df_feat = pd.read_csv(feature_path, index_col='date', parse_dates=True).sort_index()
    df_price = pd.read_csv(price_path, index_col='date', parse_dates=True).sort_index()
    
    # Target: Next day return direction
    df_price['next_return'] = df_price['Close'].pct_change().shift(-1)
    df_price['target'] = (df_price['next_return'] > 0).astype(int)
    
    # Merge features and target
    df = df_feat.join(df_price[['target', 'next_return']], how='inner')

    # BUG FIX (found 2026-09-07): recompute velocity/acceleration on THIS
    # join's own (price-aligned) index rather than trusting df_feat's
    # calendar-grid-computed columns -- for stock assets (weekday-only
    # price index) this was silently zeroing out search_interest_velocity/
    # _acceleration entirely (Google Trends' weekly report day, Sunday, is
    # never a stock trading day). See src/env/trading_env.py's identical
    # fix and src/data_pipeline/features.py's recompute_derivatives_on_index
    # docstring for the full explanation.
    df = recompute_derivatives_on_index(df)
    df = df.dropna()
    
    if len(df) < 30:
        print(f"Not enough data for {symbol}.")
        return None
    
    # Time-series split (80% train, 20% test)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split]
    test_df = df.iloc[split:]
    
    X_cols = [c for c in df_feat.columns if c != 'date']
    X_train = train_df[X_cols]
    y_train = train_df['target']
    X_test = test_df[X_cols]
    y_test = test_df['target']
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    preds = model.predict(X_test_scaled)
    probs = model.predict_proba(X_test_scaled)[:, 1]
    
    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # Strategy backtest on test set
    # position = 1 if predicted Up, -1 if predicted Down
    test_returns = test_df['next_return']
    strat_positions = np.where(preds == 1, 1, -1)
    strat_returns = strat_positions * test_returns
    
    metrics = calculate_metrics(strat_returns)
    metrics['Accuracy'] = acc
    metrics['AUC'] = auc

    # Explicit train/test date boundaries -- the dev plan specifically asks
    # for this "to make look-ahead-bias auditing easy later"; previously
    # neither printed nor returned anywhere.
    metrics['Train Start'] = train_df.index.min().date().isoformat()
    metrics['Train End'] = train_df.index.max().date().isoformat()
    metrics['Test Start'] = test_df.index.min().date().isoformat()
    metrics['Test End'] = test_df.index.max().date().isoformat()
    print(
        f"{symbol}: train {metrics['Train Start']} -> {metrics['Train End']} "
        f"({len(train_df)} rows), test {metrics['Test Start']} -> {metrics['Test End']} "
        f"({len(test_df)} rows)"
    )

    return metrics

def main():
    assets, dates = load_config()

    all_assets = []
    for category in assets.values():
        for asset in category:
            all_assets.append(asset.get('ticker') or asset.get('symbol') or asset.get('id'))

    results = {}
    for symbol in all_assets:
        res = run_supervised_strategy(symbol)
        if res:
            results[symbol] = res

    results_df = pd.DataFrame(results).T
    print("\n--- Supervised Strategy Results ---")
    print(results_df)

    # Persist a report -- see momentum.py for the same requirement/rationale.
    report_dir = Path("docs/baseline_reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "supervised_report.md"
    with open(report_path, "w") as f:
        f.write("# Supervised Baseline Report\n\n")
        f.write(
            "XGBoost classifier predicting next-day return direction from "
            "attention-derivative features only (never price). Time-series "
            "80/20 train/test split per asset, no shuffling. This is a "
            "baseline to beat, not the project's actual method (RL).\n\n"
        )
        f.write("## Per-asset date boundaries and metrics\n\n")
        f.write(results_df.to_markdown())
        f.write("\n")
    print(f"\nSaved report to {report_path}")

if __name__ == "__main__":
    main()
