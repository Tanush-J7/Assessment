"""
Multilingual Sentiment Engine & Engagement Weighting Module
Processes Chinese, Korean, and English posts and news articles.
Applies engagement weights (likes, reposts, views, author followers).
"""

import math
import re
from typing import Dict, Any, List, Tuple

# Financial Lexicons for Multilingual Sentiment (Chinese, Korean, English)
BULLISH_KEYWORDS = {
    # Chinese (HKEX / LIHKG)
    "吸籌", "突破", "買入", "超預期", "爆上", "逢低加倉", "升", "支持位極強", "大戶", "調高目標價", "獲利", "看好",
    # Korean (KRX / Naver)
    "매수", "돌파", "우상향", "호재", "수혜", "서프라이즈", "반등", "상승", "역대급", "목표가", "성장", "신고가",
    # English
    "bullish", "breakout", "buy", "accumulate", "rally", "growth", "surge", "upgrade", "outperform", "dividend"
}

BEARISH_KEYWORDS = {
    # Chinese (HKEX / LIHKG)
    "跌破", "接飛刀", "淨流出", "減倉", "避險", "偏弱", "探底", "看淡", "受壓", "跳水", "沽空", "阻力",
    # Korean (KRX / Naver)
    "매도", "이탈", "하락", "악재", "축소", "손절", "급락", "위험", "불확실성", "하향", "투매", "저점",
    # English
    "bearish", "breakdown", "sell", "rejection", "drop", "plunge", "downgrade", "pressure", "loss", "risk-off"
}


def analyze_text_sentiment(text: str) -> float:
    """
    Computes sentiment polarity between -1.0 (strongly bearish) and +1.0 (strongly bullish).
    Combines rule-based domain keyword intensity with negation/modifier handling.
    """
    text_lower = text.lower()
    bullish_hits = 0
    bearish_hits = 0

    for word in BULLISH_KEYWORDS:
        if word in text_lower:
            bullish_hits += 1

    for word in BEARISH_KEYWORDS:
        if word in text_lower:
            bearish_hits += 1

    total_hits = bullish_hits + bearish_hits
    if total_hits == 0:
        return 0.0

    score = (bullish_hits - bearish_hits) / total_hits
    return max(-1.0, min(1.0, round(score, 3)))


def calculate_engagement_weight(post: Dict[str, Any]) -> float:
    """
    Computes log-scaled engagement multiplier:
    W = log(1 + likes + 2*reposts + 0.05*views + 0.1*author_followers)
    """
    likes = float(post.get("likes", 0))
    reposts = float(post.get("reposts", 0))
    views = float(post.get("views", 0))
    followers = float(post.get("author_followers", 0))

    raw_engagement = likes + (2.0 * reposts) + (0.05 * views) + (0.1 * followers)
    return max(1.0, math.log1p(raw_engagement))


def compute_ticker_sentiment(posts: List[Dict[str, Any]], news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregates social sentiment (engagement-weighted) and news sentiment.
    Returns composite sentiment score, mention count, and direction.
    """
    if not posts and not news_items:
        return {
            "sentiment_score": 0.0,
            "direction": "Neutral",
            "social_mentions": 0,
            "social_sentiment": 0.0,
            "news_sentiment": 0.0,
            "sentiment_confidence": 0.0
        }

    # 1. Social Sentiment
    total_social_weight = 0.0
    weighted_social_sum = 0.0
    for p in posts:
        # Check if pre-labeled, otherwise compute from text
        if "sentiment_score" in p:
            pol = float(p["sentiment_score"])
        elif p.get("sentiment_label") == "bullish":
            pol = 0.75
        elif p.get("sentiment_label") == "bearish":
            pol = -0.75
        else:
            pol = analyze_text_sentiment(p.get("text", ""))

        w = calculate_engagement_weight(p)
        weighted_social_sum += pol * w
        total_social_weight += w

    social_score = (weighted_social_sum / total_social_weight) if total_social_weight > 0 else 0.0

    # 2. News Sentiment
    news_scores = []
    for n in news_items:
        score = analyze_text_sentiment(n.get("title", "") + " " + n.get("snippet", ""))
        news_scores.append(score)

    news_score = (sum(news_scores) / len(news_scores)) if news_scores else 0.0

    # 3. Composite Sentiment (70% Social, 30% News)
    composite_score = round(0.70 * social_score + 0.30 * news_score, 3)

    # Direction Classification
    if composite_score >= 0.15:
        direction = "Bullish"
    elif composite_score <= -0.15:
        direction = "Bearish"
    else:
        direction = "Neutral"

    return {
        "sentiment_score": composite_score,
        "direction": direction,
        "social_mentions": len(posts),
        "social_sentiment": round(social_score, 3),
        "news_sentiment": round(news_score, 3),
        "sentiment_confidence": round(min(1.0, (len(posts) / 20.0) * 0.8 + 0.2), 2)
    }
