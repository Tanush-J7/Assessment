"""
Market Data Ingestion Module
Fetches historical OHLCV data for target assets from HKEX and KRX,
converts timestamps to UTC, and computes Support/Resistance levels.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from typing import Dict, Any, List
import json

from src.config import TARGET_ASSETS, RAW_DATA_DIR, PROCESSED_DATA_DIR

MARKET_DATA_DIR = RAW_DATA_DIR / "market_data"
MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)


def calculate_support_resistance(df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes standard pivot points and support/resistance levels.
    P = (High + Low + Close) / 3
    R1 = 2P - Low, R2 = P + (High - Low)
    S1 = 2P - High, S2 = P - (High - Low)
    """
    if df.empty:
        return {}
    
    last_row = df.iloc[-1]
    high = float(last_row["High"])
    low = float(last_row["Low"])
    close = float(last_row["Close"])
    
    pivot = (high + low + close) / 3.0
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    
    rolling_min = float(df["Low"].min())
    rolling_max = float(df["High"].max())
    
    return {
        "pivot_point": round(pivot, 2),
        "resistance_1": round(r1, 2),
        "resistance_2": round(r2, 2),
        "support_1": round(s1, 2),
        "support_2": round(s2, 2),
        "rolling_min_10d": round(rolling_min, 2),
        "rolling_max_10d": round(rolling_max, 2),
    }


def fetch_ticker_data(ticker: str, period: str = "10d", interval: str = "1h") -> pd.DataFrame:
    """
    Fetches market data from yfinance and aligns index to UTC.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(period=period, interval=interval)
        
        if df.empty:
            # Fallback to daily if hourly not available
            df = yf_ticker.history(period=period, interval="1d")
            
        if df.empty:
            raise ValueError(f"No market data returned for {ticker}")
            
        # Ensure UTC timezone alignment
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
            
        df.reset_index(inplace=True)
        df.rename(columns={"index": "timestamp_utc", "Datetime": "timestamp_utc", "Date": "timestamp_utc"}, inplace=True)
        df["ticker"] = ticker
        
        # Save raw data
        filename = MARKET_DATA_DIR / f"{ticker.replace('.', '_')}_{interval}.csv"
        df.to_csv(filename, index=False)
        return df
    except Exception as e:
        print(f"[Warning] Failed to fetch live data for {ticker}: {e}")
        return pd.DataFrame()


def ingest_all_market_data(period: str = "10d", interval: str = "1h") -> Dict[str, Any]:
    """
    Fetches market data for all 10 assets across HKEX and KRX.
    Extracts current price and Support/Resistance levels for each.
    """
    summary = {}
    
    for exchange, assets in TARGET_ASSETS.items():
        summary[exchange] = []
        for asset in assets:
            ticker = asset["ticker"]
            name = asset["name"]
            print(f"Fetching market data for {name} ({ticker}) on {exchange}...")
            
            df = fetch_ticker_data(ticker, period=period, interval=interval)
            
            if not df.empty:
                current_price = float(df.iloc[-1]["Close"])
                prev_price = float(df.iloc[-2]["Close"]) if len(df) > 1 else current_price
                pct_change = round(((current_price - prev_price) / prev_price) * 100, 2)
                sr_levels = calculate_support_resistance(df)
            else:
                current_price = 0.0
                pct_change = 0.0
                sr_levels = {}
                
            ticker_summary = {
                "ticker": ticker,
                "asset_name": name,
                "exchange": exchange,
                "current_price": round(current_price, 2),
                "price_change_pct": pct_change,
                "sr_levels": sr_levels,
                "data_points": len(df),
                "last_updated_utc": datetime.now(timezone.utc).isoformat()
            }
            summary[exchange].append(ticker_summary)
            
    # Save processed summary
    out_file = PROCESSED_DATA_DIR / "market_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        
    print(f"Market data ingestion complete. Summary saved to {out_file}")
    return summary


if __name__ == "__main__":
    ingest_all_market_data()
