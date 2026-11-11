"""
Feature Alignment Module
Aligns social posts, news sentiment, and price bars into a unified,
UTC-normalized time-series feature matrix for predictive modeling.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.sentiment import analyze_text_sentiment, calculate_engagement_weight

ALIGNMENT_DIR = PROCESSED_DATA_DIR / "aligned_features"
ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes standard Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def build_aligned_feature_matrix(ticker: str) -> pd.DataFrame:
    """
    Loads price data, social posts, and news for a ticker,
    aligns timestamps to hourly intervals (UTC), and calculates features.
    """
    price_file = RAW_DATA_DIR / "market_data" / f"{ticker.replace('.', '_')}_1h.csv"
    social_file = RAW_DATA_DIR / "social" / f"{ticker.replace('.', '_')}_social.json"
    news_file = RAW_DATA_DIR / "news" / f"{ticker.replace('.', '_')}_news.json"

    if not price_file.exists():
        print(f"[Warning] Market data file not found for {ticker}")
        return pd.DataFrame()

    df_price = pd.read_csv(price_file)
    df_price["timestamp_utc"] = pd.to_datetime(df_price["timestamp_utc"], utc=True)
    df_price.sort_values("timestamp_utc", inplace=True)
    df_price.set_index("timestamp_utc", inplace=True)

    # Process Social Data
    social_records = []
    if social_file.exists():
        with open(social_file, "r", encoding="utf-8") as f:
            raw_posts = json.load(f)
            for p in raw_posts:
                ts = pd.to_datetime(p["timestamp_utc"], utc=True)
                pol = (
                    0.75 if p.get("sentiment_label") == "bullish"
                    else -0.75 if p.get("sentiment_label") == "bearish"
                    else analyze_text_sentiment(p.get("text", ""))
                )
                weight = calculate_engagement_weight(p)
                social_records.append({
                    "timestamp_utc": ts,
                    "sentiment_score": pol,
                    "engagement_weight": weight,
                    "weighted_sentiment": pol * weight,
                    "mentions": 1
                })

    df_social = pd.DataFrame(social_records)
    if not df_social.empty:
        df_social.set_index("timestamp_utc", inplace=True)
        # Resample to 1-hour bins
        resampled_social = df_social.resample("1h").agg({
            "mentions": "sum",
            "weighted_sentiment": "sum",
            "engagement_weight": "sum"
        }).fillna(0)
        resampled_social["hourly_social_sentiment"] = np.where(
            resampled_social["engagement_weight"] > 0,
            resampled_social["weighted_sentiment"] / resampled_social["engagement_weight"],
            0.0
        )
    else:
        resampled_social = pd.DataFrame(index=df_price.index)
        resampled_social["mentions"] = 0
        resampled_social["hourly_social_sentiment"] = 0.0

    # Align price and social features using merge_asof / reindex
    df_aligned = pd.merge_asof(
        df_price.reset_index(),
        resampled_social.reset_index(),
        on="timestamp_utc",
        direction="backward"
    ).fillna(0)

    # Technical Features
    df_aligned["return_1h"] = df_aligned["Close"].pct_change().fillna(0)
    df_aligned["volatility_5h"] = df_aligned["return_1h"].rolling(5, min_periods=1).std().fillna(0)
    df_aligned["sma_5"] = df_aligned["Close"].rolling(5, min_periods=1).mean()
    df_aligned["sma_10"] = df_aligned["Close"].rolling(10, min_periods=1).mean()
    df_aligned["rsi_14"] = calculate_rsi(df_aligned["Close"], 14)

    # Pivot Point & S/R Features
    pivot = (df_aligned["High"] + df_aligned["Low"] + df_aligned["Close"]) / 3.0
    df_aligned["dist_from_pivot"] = (df_aligned["Close"] - pivot) / (pivot + 1e-9)

    # Rolling Sentiment Features
    df_aligned["sentiment_momentum"] = df_aligned["hourly_social_sentiment"].diff().fillna(0)
    df_aligned["sentiment_ma_3h"] = df_aligned["hourly_social_sentiment"].rolling(3, min_periods=1).mean()

    # Target Variables (n+1)
    df_aligned["target_price_n1"] = df_aligned["Close"].shift(-1)
    df_aligned["target_return_n1"] = df_aligned["return_1h"].shift(-1)
    df_aligned["target_direction_n1"] = (df_aligned["target_return_n1"] > 0).astype(int)

    # Drop last row where target_price_n1 is NaN
    df_aligned_clean = df_aligned.dropna(subset=["target_price_n1"]).copy()

    # Save to disk
    out_file = ALIGNMENT_DIR / f"{ticker.replace('.', '_')}_aligned.csv"
    df_aligned_clean.to_csv(out_file, index=False)

    return df_aligned_clean


def build_all_aligned_matrices() -> Dict[str, pd.DataFrame]:
    """Builds aligned feature matrices for all target tickers."""
    from src.config import TARGET_ASSETS
    matrices = {}
    for exchange, assets in TARGET_ASSETS.items():
        for asset in assets:
            ticker = asset["ticker"]
            df = build_aligned_feature_matrix(ticker)
            matrices[ticker] = df
            print(f"Aligned feature matrix created for {ticker} ({len(df)} samples).")
    return matrices


if __name__ == "__main__":
    build_all_aligned_matrices()
