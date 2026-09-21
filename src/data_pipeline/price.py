import os
import yaml
import pandas as pd
import yfinance as yf
import ccxt
from pathlib import Path

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates

def fetch_stock_price(ticker, start, end):
    print(f"Fetching stock price for {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        return pd.DataFrame()
    # Recent yfinance versions return MultiIndex columns (Price, Ticker) even
    # for a single ticker -- flatten to plain column names, otherwise the
    # cached CSV ends up with a broken multi-row header ("Price,Close,...",
    # "Ticker,GME,GME,...", "Date,,,,,") instead of one clean header row.
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index.name = 'date'
    return data[['Close', 'Open', 'High', 'Low', 'Volume']]

def fetch_crypto_price(symbol, start, end):
    print(f"Fetching crypto price for {symbol}...")
    exchange = ccxt.binance()
    # Convert dates to timestamps
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ts = exchange.parse8601(f"{end}T23:59:59Z")

    all_ohlcv = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol + '/USDT', timeframe='1d', since=since)
        if not ohlcv:
            break
        all_ohlcv.extend(ohlcv)
        since = ohlcv[-1][0] + 86400000 # Add 1 day in ms
        # Basic check to avoid infinite loop or going too far
        if ohlcv[-1][0] > end_ts:
            break

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date
    df = df.set_index('date')
    # BUG FIX: the loop above only stops fetching once a batch's LAST candle
    # crosses end_date -- it still appends that whole (partially
    # out-of-range) batch first. Without this filter, the cached CSV silently
    # contained hundreds of extra rows past configs/date_range.yaml's
    # end_date (observed: DOGE/SHIB caches ran ~16 months past the
    # configured end date). Trim explicitly to the configured window.
    start_dt = pd.to_datetime(start).date()
    end_dt = pd.to_datetime(end).date()
    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
    return df[['Open', 'High', 'Low', 'Close', 'Volume']]

def main():
    assets, dates = load_config()
    start = dates['start_date']
    end = dates['end_date']

    raw_dir = Path("data/raw/price")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Stocks
    for stock in assets['meme_stocks']:
        ticker = stock['ticker']
        file_path = raw_dir / f"{ticker}.csv"
        if not file_path.exists():
            df = fetch_stock_price(ticker, start, end)
            df.to_csv(file_path)
        else:
            print(f"Price for {ticker} already exists. Skipping.")

    # Crypto
    for crypto in assets['small_cap_crypto']:
        symbol = crypto['symbol']
        file_path = raw_dir / f"{symbol}.csv"
        if not file_path.exists():
            df = fetch_crypto_price(symbol, start, end)
            df.to_csv(file_path)
        else:
            print(f"Price for {symbol} already exists. Skipping.")

if __name__ == "__main__":
    main()
