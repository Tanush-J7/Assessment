"""
Predictive Modeling and Out-of-Sample Validation Module
Trains machine learning models to predict asset price at n+1 using sentiment and technical features.
Performs strictly chronological train/test out-of-sample evaluation and computes trading positions.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

from src.config import TARGET_ASSETS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features.sentiment import compute_ticker_sentiment

MODEL_RESULTS_DIR = PROCESSED_DATA_DIR / "model_results"
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLS = [
    "Close",
    "Volume",
    "return_1h",
    "volatility_5h",
    "sma_5",
    "sma_10",
    "rsi_14",
    "dist_from_pivot",
    "hourly_social_sentiment",
    "sentiment_momentum",
    "sentiment_ma_3h",
    "mentions"
]


def determine_position(current_price: float, pred_price: float, sentiment_direction: str, sr_levels: Dict[str, float]) -> str:
    """
    Synthesizes a rule-based trading position recommendation based on:
    - Predicted return at n+1
    - Social/news sentiment direction
    - Proximity to key support/resistance levels
    """
    if current_price <= 0:
        return "Neutral / Hold"

    expected_return = (pred_price - current_price) / current_price
    s1 = sr_levels.get("support_1", 0.0)
    r1 = sr_levels.get("resistance_1", float("inf"))

    # Near support with positive sentiment -> Strong Buy
    if expected_return > 0.002:
        if sentiment_direction == "Bullish" or current_price <= s1 * 1.01:
            return "Long (Buy)"
        return "Moderate Long"
    elif expected_return < -0.002:
        if sentiment_direction == "Bearish" or current_price >= r1 * 0.99:
            return "Short (Sell)"
        return "Moderate Short"
    else:
        return "Hold / Neutral"


def train_and_evaluate_ticker(ticker: str, asset_name: str, exchange: str) -> Dict[str, Any]:
    """
    Trains predictive model for an asset and validates on out-of-sample split.
    """
    aligned_file = PROCESSED_DATA_DIR / "aligned_features" / f"{ticker.replace('.', '_')}_aligned.csv"
    if not aligned_file.exists():
        print(f"[Warning] Aligned feature file missing for {ticker}")
        return {}

    df = pd.read_csv(aligned_file)
    if len(df) < 15:
        print(f"[Warning] Insufficient sample size for {ticker}")
        return {}

    X = df[FEATURE_COLS].copy()
    y = df["target_price_n1"].copy()

    # Out-of-sample split (chronological 80/20 train/test)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train model (Gradient Boosting Regressor)
    model = GradientBoostingRegressor(n_estimators=60, max_depth=3, random_state=42, learning_rate=0.08)
    model.fit(X_train, y_train)

    # Out-of-sample predictions
    y_pred_test = model.predict(X_test)

    # Metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
    mae = float(mean_absolute_error(y_test, y_pred_test))
    mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)

    # Directional Accuracy (Hit Ratio)
    test_current_price = X_test["Close"].values
    actual_dir = np.sign(y_test.values - test_current_price)
    pred_dir = np.sign(y_pred_test - test_current_price)
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    # Predict next future price (n+1) using latest available bar
    latest_features = X.iloc[[-1]]
    latest_price = float(df.iloc[-1]["Close"])
    predicted_next_price = float(model.predict(latest_features)[0])
    predicted_change_pct = round(((predicted_next_price - latest_price) / latest_price) * 100, 2)

    # Load market summary for S/R levels
    sr_levels = {}
    market_summary_file = PROCESSED_DATA_DIR / "market_summary.json"
    if market_summary_file.exists():
        with open(market_summary_file, "r", encoding="utf-8") as f:
            ms = json.load(f)
            for item in ms.get(exchange, []):
                if item["ticker"] == ticker:
                    sr_levels = item.get("sr_levels", {})
                    break

    # Load social and news for composite sentiment
    social_file = RAW_DATA_DIR / "social" / f"{ticker.replace('.', '_')}_social.json"
    news_file = RAW_DATA_DIR / "news" / f"{ticker.replace('.', '_')}_news.json"
    posts = []
    news_items = []
    if social_file.exists():
        with open(social_file, "r", encoding="utf-8") as f:
            posts = json.load(f)
    if news_file.exists():
        with open(news_file, "r", encoding="utf-8") as f:
            news_items = json.load(f)

    sentiment_info = compute_ticker_sentiment(posts, news_items)
    position = determine_position(latest_price, predicted_next_price, sentiment_info["direction"], sr_levels)

    # Feature Importance
    importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLS, model.feature_importances_)}
    sorted_importance = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    result = {
        "ticker": ticker,
        "asset_name": asset_name,
        "exchange": exchange,
        "current_price": round(latest_price, 2),
        "predicted_price_n1": round(predicted_next_price, 2),
        "predicted_change_pct": predicted_change_pct,
        "sentiment_score": sentiment_info["sentiment_score"],
        "sentiment_direction": sentiment_info["direction"],
        "social_mentions": sentiment_info["social_mentions"],
        "recommended_position": position,
        "sr_levels": sr_levels,
        "validation_metrics": {
            "test_samples": len(y_test),
            "rmse": round(rmse, 3),
            "mae": round(mae, 3),
            "mape_pct": round(mape, 2),
            "directional_accuracy_pct": round(directional_accuracy, 1)
        },
        "feature_importances": sorted_importance,
        "test_actual_vs_pred": [
            {"actual": round(float(a), 2), "predicted": round(float(p), 2)}
            for a, p in zip(y_test.values[-8:], y_pred_test[-8:])
        ]
    }

    return result


def train_and_evaluate_all() -> Dict[str, List[Dict[str, Any]]]:
    """Runs modeling and out-of-sample validation for all assets."""
    summary = {}
    for exchange, assets in TARGET_ASSETS.items():
        summary[exchange] = []
        for asset in assets:
            ticker = asset["ticker"]
            name = asset["name"]
            print(f"Training predictive model for {name} ({ticker}) on {exchange}...")
            res = train_and_evaluate_ticker(ticker, name, exchange)
            if res:
                summary[exchange].append(res)

    # Save results
    out_file = PROCESSED_DATA_DIR / "predictions_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll models trained and evaluated. Summary saved to {out_file}")
    return summary


if __name__ == "__main__":
    train_and_evaluate_all()
