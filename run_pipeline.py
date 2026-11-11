"""
Master Pipeline Orchestrator
Executes the full end-to-end sentiment price prediction pipeline:
1. Ingestion: Market Data (yfinance UTC bars)
2. Ingestion: Exa.com News
3. Ingestion: Apify Social Scraper (LIHKG, Telegram, Naver)
4. Feature Engineering: Multilingual NLP & UTC Alignment
5. Modeling: Out-of-Sample Predictive Model (n+1 price/return)
6. Reporting: Generates visual HTML report
"""

import sys
from datetime import datetime, timezone

from src.ingestion.market_data import ingest_all_market_data
from src.ingestion.exa_news import ingest_all_news
from src.ingestion.apify_scraper import ingest_all_social
from src.features.alignment import build_all_aligned_matrices
from src.models.predictor import train_and_evaluate_all
from src.dashboard.generate_html_report import generate_html_report


def run_pipeline():
    start_time = datetime.now(timezone.utc)
    print("=" * 70)
    print("🚀 Starting HengSeng Sentiment Price Prediction Agent Pipeline")
    print(f"   Execution Timestamp: {start_time.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 70)

    print("\n[Step 1/6] Ingesting Market Data & S/R Levels (yfinance)...")
    ingest_all_market_data(period="10d", interval="1h")

    print("\n[Step 2/6] Ingesting Recent Financial News (Exa.com)...")
    ingest_all_news(days=10)

    print("\n[Step 3/6] Scraping Social Media Community Sentiment (Apify)...")
    ingest_all_social(days=10)

    print("\n[Step 4/6] Aligning Multilingual Sentiment & Technical Features...")
    build_all_aligned_matrices()

    print("\n[Step 5/6] Training Out-of-Sample Predictive Model (Price at n+1)...")
    train_and_evaluate_all()

    print("\n[Step 6/6] Generating Visual Summary Report...")
    report_path = generate_html_report()

    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 70)
    print("✅ Pipeline executed successfully in {:.1f}s!".format(duration))
    print(f"📊 Visual Report: file://{report_path.resolve() if report_path else 'N/A'}")
    print("💻 Interactive Dashboard: run 'streamlit run src/dashboard/app.py'")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
