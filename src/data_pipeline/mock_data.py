"""Synthetic data generator for LOCAL DEV/DEMO USE ONLY.

WARNING: this module previously wrote its fabricated (Geometric-Brownian-
Motion price / Gaussian-bump Wikipedia / "Lorem ipsum" Reddit text) output
directly to data/raw/price/, data/raw/wikipedia/, and data/raw/reddit/ --
the EXACT same paths the real fetchers (price.py, wikipedia.py, reddit.py)
read from and skip-if-exists write to. That meant running this script once
silently and permanently poisoned the "real" pipeline with fake data,
because every real fetcher's idempotency check (`if not file_path.exists()`)
would then see the fake file and never overwrite it with real data.

That is exactly what had happened in this repo: data/raw/price,
data/raw/wikipedia, and data/raw/reddit were all found to contain output
from this generator, not live API data, despite looking superficially like
a real cached pipeline run. Those fake files have been deleted and
regenerated for real (see docs/feature_dictionary.md and the Phase 1
completion report for details).

To prevent this from happening again, this module now writes to
data/raw_mock/ (a clearly-separate tree) instead. Use it only for local
smoke-testing of downstream code (features.py, the env, etc.) when you
deliberately want synthetic data -- never point it at data/raw/ again.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime, timedelta

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates

def generate_time_series(start_date, end_date, length=None):
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    if length is None:
        length = (end - start).days + 1
    return pd.date_range(start=start, periods=length, freq='D')

def generate_asset_data(symbol, name, start, end):
    print(f"Generating mock data for {symbol}...")
    dates = generate_time_series(start, end)
    n = len(dates)
    
    # 1. Price Data (Geometric Brownian Motion)
    # Simulate a "meme" event: high volatility, sudden spike
    price = 100.0
    prices = []
    for i in range(n):
        # Add a spike around the middle of the range
        vol = 0.02 if (n//3 < i < 2*n//3) else 0.01
        drift = 0.001 if (n//3 < i < 2*n//3) else 0.0
        price *= (1 + np.random.normal(drift, vol))
        prices.append(price)
    
    df_price = pd.DataFrame({
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(100000, 1000000, n)
    }, index=dates)
    df_price.index.name = 'date'
    
    # 2. Wikipedia Data
    # Views and edits usually lead price spikes
    # Create a peak slightly before the price spike
    peak_center = n // 3 - 10
    views = [1000 + 5000 * np.exp(-((i - peak_center)**2) / (2 * 10**2)) + np.random.randint(0, 500) for i in range(n)]
    edits = [10 + 50 * np.exp(-((i - peak_center)**2) / (2 * 5**2)) + np.random.randint(0, 10) for i in range(n)]
    unique_editors = [5 + 20 * np.exp(-((i - peak_center)**2) / (2 * 5**2)) + np.random.randint(0, 5) for i in range(n)]
    reverts = [1 + 10 * np.exp(-((i - peak_center)**2) / (2 * 3**2)) + np.random.randint(0, 2) for i in range(n)]
    
    df_wiki = pd.DataFrame({
        'pageviews': views,
        'edit_count': edits,
        'unique_editors': unique_editors,
        'revert_count': reverts
    }, index=dates)
    df_wiki.index.name = 'date'
    
    # 3. Reddit Data
    # Sentiment usually diverges or peaks during spikes
    sentiment = [0.1 + 0.4 * np.sin(i/10) + np.random.normal(0, 0.1) for i in range(n)]
    posts = [5 + 20 * np.exp(-((i - peak_center)**2) / (2 * 15**2)) + np.random.randint(0, 10) for i in range(n)]
    comments = [50 + 200 * np.exp(-((i - peak_center)**2) / (2 * 15**2)) + np.random.randint(0, 100) for i in range(n)]
    
    # Generate synthetic text for Novelty Score
    # Use a few base keywords and rotate them
    keywords = ["moon", "rocket", "squeeze", "diamond hands", "bullish", "bearish", "dump", "pump"]
    texts = []
    for i in range(n):
        # If peak, use more "hype" words
        num_words = 5 if (abs(i - peak_center) < 20) else 2
        words = np.random.choice(keywords, num_words)
        texts.append(" ".join(words) + " " + "Lorem ipsum dolor sit amet " * 5)
        
    df_reddit_raw = pd.DataFrame({
        'date': dates,
        'post_count': [1] * n,
        'comment_count': comments,
        'sentiment': sentiment,
        'text': texts
    })
    
    # Aggregated reddit data
    df_reddit_agg = df_reddit_raw.groupby('date').agg({
        'post_count': 'sum',
        'comment_count': 'sum',
        'sentiment': 'mean'
    })
    
    return df_price, df_wiki, df_reddit_raw, df_reddit_agg

def main():
    assets, dates = load_config()
    start = dates['start_date']
    end = dates['end_date']
    
    # Raw directories -- deliberately NOT data/raw/, see module docstring.
    raw_price_dir = Path("data/raw_mock/price")
    raw_wiki_dir = Path("data/raw_mock/wikipedia")
    raw_reddit_dir = Path("data/raw_mock/reddit")
    
    for d in [raw_price_dir, raw_wiki_dir, raw_reddit_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    all_assets = []
    for category in assets.values():
        for asset in category:
            all_assets.append(asset)
            
    for asset in all_assets:
        symbol = asset.get('ticker') or asset.get('symbol') or asset.get('id')
        name = asset.get('name')
        
        df_price, df_wiki, df_reddit_raw, df_reddit_agg = generate_asset_data(symbol, name, start, end)
        
        # Save to CSVs exactly as the pipeline expects
        df_price.to_csv(raw_price_dir / f"{symbol}.csv")
        df_wiki.to_csv(raw_wiki_dir / f"{symbol}.csv")
        df_reddit_raw.to_csv(raw_reddit_dir / f"{symbol}_raw.csv", index=False)
        df_reddit_agg.to_csv(raw_reddit_dir / f"{symbol}.csv")
        
    print("\nMock data generation complete. data/raw_mock/ now populated (NOT data/raw/).")

if __name__ == "__main__":
    main()
