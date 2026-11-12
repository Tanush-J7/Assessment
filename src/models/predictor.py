"""
Predictive Modeling and Out-of-Sample Validation Module
Trains machine learning models to predict percentage return at n+1 using news, social sentiment,
and stationary technical features.
Performs strictly chronological train/test holdout and TimeSeriesSplit walk-forward validation.
Computes Naive Persistence Baseline (P_{t+1} = P_t) and compares model vs baseline.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import TARGET_ASSETS, PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.features.sentiment import compute_ticker_sentiment

MODEL_RESULTS_DIR = PROCESSED_DATA_DIR / "model_results"
MODEL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# FIX 1 & 2: Feature set with stationary technicals, social sentiment, and Exa news sentiment
FEATURE_COLS = [
    # Stationary Technical Indicators
    "return_1h",
    "volatility_5h",
    "dist_from_sma_5",
    "dist_from_sma_10",
    "rsi_14",
    "dist_from_pivot",
    "Volume",
    
    # Social Community Sentiment
    "hourly_social_sentiment",
    "sentiment_momentum",
    "sentiment_ma_3h",
    "mentions",
    
    # Exa News Sentiment (FIX 1)
    "news_sentiment",
    "news_positive_count",
    "news_negative_count",
    "news_article_count",
    "news_sentiment_ma_6h"
]

SOCIAL_FEATURES = ["hourly_social_sentiment", "sentiment_momentum", "sentiment_ma_3h", "mentions"]
NEWS_FEATURES = ["news_sentiment", "news_positive_count", "news_negative_count", "news_article_count", "news_sentiment_ma_6h"]
TECHNICAL_FEATURES = ["return_1h", "volatility_5h", "dist_from_sma_5", "dist_from_sma_10", "rsi_14", "dist_from_pivot", "Volume"]


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
    FIX 1, 2, 5, 6:
    - Trains model on target_return_n1 (percentage return)
    - Reconstructs predicted price: P_pred = P_curr * (1 + R_pred)
    - Compares against Naive Persistence Baseline (P_{t+1} = P_t)
    - Performs TimeSeriesSplit walk-forward validation
    """
    aligned_file = PROCESSED_DATA_DIR / "aligned_features" / f"{ticker.replace('.', '_')}_aligned.csv"
    if not aligned_file.exists():
        print(f"[Warning] Aligned feature file missing for {ticker}")
        return {}

    df = pd.read_csv(aligned_file)
    if len(df) < 20:
        print(f"[Warning] Insufficient sample size for {ticker} ({len(df)} samples)")
        return {}

    X = df[FEATURE_COLS].copy().fillna(0)
    y_return = df["target_return_n1"].copy()
    y_price = df["target_price_n1"].copy()
    current_prices = df["Close"].copy()

    # 1. Chronological Holdout Split (80% Train, 20% Test)
    split_idx = int(len(df) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_return.iloc[:split_idx], y_return.iloc[split_idx:]
    test_current_price = current_prices.iloc[split_idx:].values
    test_actual_price = y_price.iloc[split_idx:].values

    # 2. Train Model on Return
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=2,
        random_state=42,
        learning_rate=0.04,
        subsample=0.85
    )
    model.fit(X_train, y_train)

    # 3. Model Predictions on Holdout Test Set
    pred_return_test = model.predict(X_test)
    pred_price_test = test_current_price * (1.0 + pred_return_test)

    # 4. Model Evaluation Metrics
    model_rmse = float(np.sqrt(mean_squared_error(test_actual_price, pred_price_test)))
    model_mae = float(mean_absolute_error(test_actual_price, pred_price_test))
    model_mape = float(np.mean(np.abs((test_actual_price - pred_price_test) / test_actual_price)) * 100)

    actual_dir = np.sign(y_test.values)
    pred_dir = np.sign(pred_return_test)
    directional_accuracy = float(np.mean(actual_dir == pred_dir) * 100)

    # 5. FIX 5: Naive Persistence Baseline (Next Price = Current Price, i.e. return = 0)
    baseline_pred_price = test_current_price
    baseline_rmse = float(np.sqrt(mean_squared_error(test_actual_price, baseline_pred_price)))
    baseline_mae = float(mean_absolute_error(test_actual_price, baseline_pred_price))
    baseline_mape = float(np.mean(np.abs((test_actual_price - baseline_pred_price) / test_actual_price)) * 100)
    # Naive baseline predicts zero return, yielding 0% directional call accuracy
    baseline_dir_accuracy = 0.0

    # 6. FIX 6: TimeSeriesSplit Walk-Forward Cross-Validation
    tscv = TimeSeriesSplit(n_splits=3)
    cv_folds = []
    for fold_idx, (cv_tr, cv_te) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[cv_tr], X.iloc[cv_te]
        y_tr, y_te = y_return.iloc[cv_tr], y_return.iloc[cv_te]
        cv_curr = current_prices.iloc[cv_te].values
        cv_act = y_price.iloc[cv_te].values

        m_cv = GradientBoostingRegressor(n_estimators=50, max_depth=2, random_state=42, learning_rate=0.04, subsample=0.85)
        m_cv.fit(X_tr, y_tr)
        f_pred_ret = m_cv.predict(X_te)
        f_pred_price = cv_curr * (1.0 + f_pred_ret)

        f_rmse = float(np.sqrt(mean_squared_error(cv_act, f_pred_price)))
        f_base_rmse = float(np.sqrt(mean_squared_error(cv_act, cv_curr)))
        f_dir = float(np.mean(np.sign(y_te.values) == np.sign(f_pred_ret)) * 100)

        cv_folds.append({
            "fold": fold_idx + 1,
            "train_samples": len(cv_tr),
            "test_samples": len(cv_te),
            "model_rmse": round(f_rmse, 3),
            "baseline_rmse": round(f_base_rmse, 3),
            "directional_accuracy_pct": round(f_dir, 1)
        })

    cv_mean_model_rmse = round(float(np.mean([f["model_rmse"] for f in cv_folds])), 3)
    cv_mean_base_rmse = round(float(np.mean([f["baseline_rmse"] for f in cv_folds])), 3)
    cv_mean_hit_ratio = round(float(np.mean([f["directional_accuracy_pct"] for f in cv_folds])), 1)

    # 7. Predict Next Future Interval (n+1)
    latest_features = X.iloc[[-1]]
    latest_price = float(df.iloc[-1]["Close"])
    predicted_return_n1 = float(model.predict(latest_features)[0])
    predicted_next_price = latest_price * (1.0 + predicted_return_n1)
    predicted_change_pct = round(predicted_return_n1 * 100, 2)

    # 8. Load S/R levels & Sentiment
    sr_levels = {}
    market_summary_file = PROCESSED_DATA_DIR / "market_summary.json"
    if market_summary_file.exists():
        with open(market_summary_file, "r", encoding="utf-8") as f:
            ms = json.load(f)
            for item in ms.get(exchange, []):
                if item["ticker"] == ticker:
                    sr_levels = item.get("sr_levels", {})
                    break

    social_file = RAW_DATA_DIR / "social" / f"{ticker.replace('.', '_')}_social.json"
    news_file = RAW_DATA_DIR / "news" / f"{ticker.replace('.', '_')}_news.json"
    posts, news_items = [], []
    if social_file.exists():
        with open(social_file, "r", encoding="utf-8") as f:
            posts = json.load(f)
    if news_file.exists():
        with open(news_file, "r", encoding="utf-8") as f:
            news_items = json.load(f)

    sentiment_info = compute_ticker_sentiment(posts, news_items)
    position = determine_position(latest_price, predicted_next_price, sentiment_info["direction"], sr_levels)

    # 9. Feature Importance Breakdown (Technical vs Social vs News)
    importances = {feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLS, model.feature_importances_)}
    sorted_importance = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    social_imp = round(sum(importances.get(f, 0) for f in SOCIAL_FEATURES) * 100, 1)
    news_imp = round(sum(importances.get(f, 0) for f in NEWS_FEATURES) * 100, 1)
    tech_imp = round(sum(importances.get(f, 0) for f in TECHNICAL_FEATURES) * 100, 1)

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
            "model_rmse": round(model_rmse, 3),
            "baseline_rmse": round(baseline_rmse, 3),
            "model_mae": round(model_mae, 3),
            "baseline_mae": round(baseline_mae, 3),
            "model_mape_pct": round(model_mape, 2),
            "baseline_mape_pct": round(baseline_mape, 2),
            "directional_accuracy_pct": round(directional_accuracy, 1),
            "rmse": round(model_rmse, 3), # Backward compatibility
            "mae": round(model_mae, 3),
            "mape_pct": round(model_mape, 2)
        },
        "walk_forward_cv": {
            "folds": cv_folds,
            "mean_model_rmse": cv_mean_model_rmse,
            "mean_baseline_rmse": cv_mean_base_rmse,
            "mean_hit_ratio_pct": cv_mean_hit_ratio
        },
        "feature_contributions": {
            "technical_pct": tech_imp,
            "social_sentiment_pct": social_imp,
            "news_sentiment_pct": news_imp
        },
        "feature_importances": sorted_importance,
        "test_actual_vs_pred": [
            {
                "actual": round(float(a), 2),
                "predicted": round(float(p), 2),
                "baseline": round(float(b), 2)
            }
            for a, p, b in zip(test_actual_price[-8:], pred_price_test[-8:], baseline_pred_price[-8:])
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
            print(f"Training predictive return model for {name} ({ticker}) on {exchange}...")
            res = train_and_evaluate_ticker(ticker, name, exchange)
            if res:
                summary[exchange].append(res)

    out_file = PROCESSED_DATA_DIR / "predictions_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAll models trained and evaluated. Summary saved to {out_file}")
    return summary


if __name__ == "__main__":
    train_and_evaluate_all()
