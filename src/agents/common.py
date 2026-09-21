"""Shared helpers for Phase 4 (agent training) scripts: picking which asset
to train on, and splitting its date range into a train/validation window.
Factored out of train_ppo.py so reward_hacking_check.py (and later,
Phase 5's walk-forward harness) can reuse the exact same split logic rather
than risk it drifting between scripts.
"""
from pathlib import Path

import pandas as pd


def available_assets():
    """Symbols with both a processed feature file and raw price file --
    the two NarrativeTradingEnv requires. NFT collections currently have no
    price data (Phase 1a never implemented OpenSea/NFT price ingestion --
    see dev plan Phase 7), so they're excluded here, not silently broken.
    """
    processed_dir = Path("data/processed")
    price_dir = Path("data/raw/price")
    symbols = []
    for feat_path in sorted(processed_dir.glob("*_features.csv")):
        symbol = feat_path.stem.removesuffix("_features")
        if (price_dir / f"{symbol}.csv").exists():
            symbols.append(symbol)
    return symbols


def select_default_asset():
    """Picks the asset with the longest aligned (feature ∩ price) history,
    per the dev plan's Phase 4a instruction: "pick the one with the
    longest, cleanest history from Phase 1". Ties broken alphabetically for
    determinism.
    """
    best_symbol, best_len = None, -1
    for symbol in available_assets():
        feat = pd.read_csv(f"data/processed/{symbol}_features.csv", index_col="date", parse_dates=True)
        price = pd.read_csv(f"data/raw/price/{symbol}.csv", index_col="date", parse_dates=True)
        n = len(feat.join(price[["Close"]], how="inner"))
        if n > best_len:
            best_symbol, best_len = symbol, n
    if best_symbol is None:
        raise FileNotFoundError(
            "No asset has both data/processed/<symbol>_features.csv and "
            "data/raw/price/<symbol>.csv -- run the Phase 1 pipeline first."
        )
    return best_symbol


def get_paths(symbol):
    return (
        Path(f"data/processed/{symbol}_features.csv"),
        Path(f"data/raw/price/{symbol}.csv"),
    )


def get_train_val_split(symbol, train_frac=0.8):
    """Chronological (no shuffling) split of the asset's aligned date range,
    matching src/baselines/supervised.py's split convention. Returns
    (train_end_date, val_start_date, val_end_date) -- train uses
    [start, train_end_date], validation uses [val_start_date, val_end_date],
    with no overlap (val_start_date is the row immediately after
    train_end_date).
    """
    features_path, price_path = get_paths(symbol)
    feat = pd.read_csv(features_path, index_col="date", parse_dates=True)
    price = pd.read_csv(price_path, index_col="date", parse_dates=True)
    joined = feat.join(price[["Close"]], how="inner").sort_index()

    split = int(len(joined) * train_frac)
    if split < 2 or split >= len(joined) - 1:
        raise ValueError(
            f"{symbol}: train_frac={train_frac} gives a degenerate split "
            f"({split} train rows, {len(joined) - split} val rows) for "
            f"{len(joined)} total aligned rows."
        )
    train_end_date = joined.index[split - 1]
    val_start_date = joined.index[split]
    val_end_date = joined.index[-1]
    return train_end_date, val_start_date, val_end_date
