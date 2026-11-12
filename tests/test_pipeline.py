"""
Automated Test Suite for HengSeng & KRX Sentiment Price Prediction Agent
Tests Fix 1 (News sentiment in features), Fix 2 (Percentage return target),
Fix 3 & 4 (Apify payload builder & deduplication), Fix 5 (Naive baseline),
and Fix 6 (TimeSeriesSplit walk-forward validation).
"""

import unittest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from src.config import TARGET_ASSETS, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.sentiment import analyze_text_sentiment, calculate_engagement_weight
from src.features.alignment import build_aligned_feature_matrix
from src.ingestion.apify_scraper import (
    build_actor_payload,
    validate_actor_payload,
    deduplicate_posts,
    get_asset_metadata
)
from src.models.predictor import (
    FEATURE_COLS,
    NEWS_FEATURES,
    SOCIAL_FEATURES,
    TECHNICAL_FEATURES,
    train_and_evaluate_ticker
)


class TestPipelineFixes(unittest.TestCase):

    def test_fix1_news_features_in_aligned_matrix(self):
        """Verify FIX 1: Exa News features are integrated into the feature matrix."""
        df = build_aligned_feature_matrix("0700.HK")
        self.assertFalse(df.empty, "Aligned feature matrix should not be empty")

        for news_col in NEWS_FEATURES:
            self.assertIn(news_col, df.columns, f"News feature '{news_col}' must be in aligned DataFrame")

        # Verify no NaN values in news columns
        for news_col in NEWS_FEATURES:
            self.assertEqual(df[news_col].isna().sum(), 0, f"News column '{news_col}' contains NaNs")

    def test_fix2_return_target_and_price_reconstruction(self):
        """Verify FIX 2: Model trains on target_return_n1 and price reconstruction is exact."""
        df = build_aligned_feature_matrix("0700.HK")
        self.assertIn("target_return_n1", df.columns, "target_return_n1 must exist")
        self.assertIn("target_price_n1", df.columns, "target_price_n1 must exist")

        # Test mathematical equivalence: target_price_n1 == Close * (1 + target_return_n1)
        expected_price = df["Close"] * (1.0 + df["target_return_n1"])
        np.testing.assert_allclose(df["target_price_n1"].values, expected_price.values, rtol=1e-5,
                                   err_msg="Price reconstruction formula must be mathematically exact")

    def test_fix3_and_fix4_apify_payload_and_cashtags(self):
        """Verify FIX 3 & 4: Actor payload builder creates source-specific schemas and includes cashtags/keywords."""
        # Test Telegram schema
        tg_payload = build_actor_payload("apify/telegram-scraper", "telegram", "0700.HK", "Tencent Holdings", "HKEX", 10)
        self.assertIn("searchQueries", tg_payload)
        self.assertIn("messagesLimit", tg_payload)
        # Check cashtags and keywords are included
        queries_str = " ".join(tg_payload["searchQueries"])
        self.assertTrue("$0700" in queries_str or "0700" in queries_str)

        # Validate payload
        is_valid, msg = validate_actor_payload("apify/telegram-scraper", tg_payload)
        self.assertTrue(is_valid, f"Telegram payload should be valid: {msg}")

        # Test LIHKG Cheerio schema
        lihkg_payload = build_actor_payload("apify/cheerio-scraper", "lihkg", "0700.HK", "Tencent Holdings", "HKEX", 10)
        self.assertIn("startUrls", lihkg_payload)
        self.assertIn("pageFunction", lihkg_payload)
        is_valid, msg = validate_actor_payload("apify/cheerio-scraper", lihkg_payload)
        self.assertTrue(is_valid, f"LIHKG payload should be valid: {msg}")

        # Test Naver Finance schema
        naver_payload = build_actor_payload("apify/cheerio-scraper", "naver_finance", "005930.KS", "Samsung Electronics", "KRX", 10)
        self.assertIn("startUrls", naver_payload)
        self.assertTrue("005930" in naver_payload["startUrls"][0]["url"])

    def test_fix4_deduplication(self):
        """Verify FIX 4: Duplicate posts are correctly filtered out."""
        sample_posts = [
            {"post_id": "101", "timestamp_utc": "2026-11-10T12:00:00Z", "text": "Bullish breakout on Tencent!"},
            {"post_id": "101", "timestamp_utc": "2026-11-10T12:00:00Z", "text": "Bullish breakout on Tencent!"},  # Exact duplicate
            {"post_id": "102", "timestamp_utc": "2026-11-10T13:00:00Z", "text": "Alibaba earnings beat."}
        ]
        deduped = deduplicate_posts(sample_posts)
        self.assertEqual(len(deduped), 2, "Duplicate post should have been filtered out")

    def test_fix5_and_fix6_baseline_and_walk_forward_cv(self):
        """Verify FIX 5 & 6: Naive baseline and TimeSeriesSplit cross-validation are calculated."""
        result = train_and_evaluate_ticker("0700.HK", "Tencent Holdings", "HKEX")
        self.assertIsNotNone(result)

        vm = result["validation_metrics"]
        self.assertIn("model_rmse", vm)
        self.assertIn("baseline_rmse", vm)
        self.assertGreater(vm["model_rmse"], 0.0)
        self.assertGreater(vm["baseline_rmse"], 0.0)

        # Walk-forward CV
        wf = result["walk_forward_cv"]
        self.assertIn("folds", wf)
        self.assertEqual(len(wf["folds"]), 3, "TimeSeriesSplit should evaluate exactly 3 folds")
        for fold in wf["folds"]:
            self.assertIn("train_samples", fold)
            self.assertIn("test_samples", fold)
            self.assertIn("model_rmse", fold)
            self.assertIn("baseline_rmse", fold)
            self.assertGreater(fold["train_samples"], 0)
            self.assertGreater(fold["test_samples"], 0)

        # Feature contributions
        fc = result["feature_contributions"]
        self.assertIn("technical_pct", fc)
        self.assertIn("social_sentiment_pct", fc)
        self.assertIn("news_sentiment_pct", fc)


if __name__ == "__main__":
    unittest.main()
