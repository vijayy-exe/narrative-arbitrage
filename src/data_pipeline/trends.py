import os
import yaml
import pandas as pd
from pytrends.request import TrendReq
from pathlib import Path

def load_config():
    with open("configs/assets.yaml", "r") as f:
        assets = yaml.safe_load(f)
    with open("configs/date_range.yaml", "r") as f:
        dates = yaml.safe_load(f)
    return assets, dates

def fetch_trends(keyword, start_date, end_date):
    print(f"Fetching Google Trends for {keyword}...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([keyword], cat=0, timeframe=f"{start_date} {end_date}")
        df = pytrends.interest_over_time()

        if df.empty:
            return pd.DataFrame()

        # Google Trends returns a column with the keyword name
        # We rename it to 'search_interest'
        df = df[[keyword]].rename(columns={keyword: 'search_interest'})
        df.index.name = 'date'
        return df
    except Exception as e:
        print(f"Error fetching trends for {keyword}: {e}")
        return pd.DataFrame()

def main():
    assets, dates = load_config()
    start = dates['start_date']
    end = dates['end_date']

    raw_dir = Path("data/raw/trends")
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_assets = []
    for category in assets.values():
        for asset in category:
            all_assets.append(asset)

    for asset in all_assets:
        name = asset.get('name')
        symbol = asset.get('ticker') or asset.get('symbol') or asset.get('id')

        file_path = raw_dir / f"{symbol}.csv"
        if not file_path.exists():
            df = fetch_trends(name, start, end)
            if df.empty:
                print(f"No trends data found for {name}")
                continue
            df.to_csv(file_path)
        else:
            print(f"Trends data for {symbol} already exists. Skipping.")

if __name__ == "__main__":
    main()
