# HengSeng & KRX Sentiment Price Prediction Agent

> **Intern Position Assessment Submission — AI Agents Developer**  
> **Applicant Submission for:** Gilad (CEO, CrowdWisdomTrading)  
> **Contact / Submission Email:** `gilad@crowdwisdomtrading.com`

---

## 📌 Executive Summary

This project implements an end-to-end Python-based multi-source sentiment analysis and price prediction pipeline for equities listed on the **Hong Kong Stock Exchange (HKEX / Hang Seng)** and the **Korea Exchange (KRX)**.

The system ingests real-time and historical market price bars (10-day lookback, UTC-aligned), alternative social media community sentiment (via **Apify**), and semantic financial news feeds (via **Exa.com**). It trains a predictive model utilizing **strict out-of-sample walk-forward validation** (zero lookahead leakage) to forecast each asset's price at $n+1$ and synthesize actionable trading positions.

---

## 🏛️ Target Assets & Social Sources

### 1. Selected Assets (10 total: 5 HKEX, 5 KRX)
| Exchange | Ticker | Company Name | Sector |
|---|---|---|---|
| **HKEX** | `0700.HK` | Tencent Holdings | Internet & Tech |
| **HKEX** | `9988.HK` | Alibaba Group | E-Commerce & Cloud |
| **HKEX** | `3690.HK` | Meituan | Local Services & Delivery |
| **HKEX** | `1211.HK` | BYD Company | Electric Vehicles & Batteries |
| **HKEX** | `0005.HK` | HSBC Holdings | Banking & Finance |
| **KRX** | `005930.KS` | Samsung Electronics | Semiconductors & Consumer Tech |
| **KRX** | `000660.KS` | SK Hynix | High-Bandwidth Memory (HBM) |
| **KRX** | `035420.KS` | NAVER | Internet, AI & Search |
| **KRX** | `005380.KS` | Hyundai Motor | Automotive & Mobility |
| **KRX** | `051910.KS` | LG Chem | Chemicals & EV Materials |

### 2. Multi-Source Community Coverage ($\ge 2$ per exchange)
*   **Hong Kong (HKEX):**
    1.  **LIHKG (`lihkg.com/category/stock`)** — High-velocity Hong Kong retail trading forum.
    2.  **Telegram Channels & Public Trading Groups** — Real-time ticker discussions & alerts.
*   **South Korea (KRX):**
    1.  **Naver Finance Stock Discussion Boards (`종목토론실`)** — Epicenter of Korean retail investor sentiment.
    2.  **Telegram Channels & Blind** — Verified tech community and retail trading sentiment.

---

## 🏗️ Architecture & Pipeline Flow

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│  Market Data   │     │  Apify Social  │     │    Exa.com     │
│   (yfinance)   │     │ Scraper Engine │     │   News Engine  │
│  10-Day UTC    │     │ (LIHKG/Naver/  │     │ (Semantic NLP) │
│   OHLCV Bars   │     │   Telegram)    │     │                │
└───────┬────────┘     └───────┬────────┘     └───────┬────────┘
        │                      │                      │
        ▼                      ▼                      ▼
┌──────────────────────────────────────────────────────────────┐
│                  Feature Engineering Engine                  │
│  • Multilingual Lexicon Sentiment (Traditional Chinese /     │
│    Korean / English)                                         │
│  • Engagement Multiplier: log(1 + Likes + 2*Reposts + ...)   │
│  • Support & Resistance: Pivot, R1, R2, S1, S2, Rolling Ext  │
│  • UTC Timestamp Alignment & Resampling                      │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│           Predictive Modeling & Out-of-Sample Valid          │
│  • Chronological 80/20 Train/Test Walk-Forward Split         │
│  • Gradient Boosting Regressor (Predict Target: Price n+1)   │
│  • Evaluation: RMSE, MAE, MAPE, Directional Hit Ratio (%)    │
│  • Rule-Based Position Synthesis (Long / Short / Neutral)    │
└──────────────────────────────┬───────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────┐
│                     Visual Presentation                      │
│  • Streamlit Interactive Web App (`src/dashboard/app.py`)    │
│  • Standalone Static HTML Report (`reports/visual_report.html`)│
└──────────────────────────────────────────────────────────────┘
```

---

## 📐 Quantitative Methodology

### 1. Engagement-Weighted Sentiment
Social media influence is non-linear. Raw sentiment is weighted by engagement velocity:
$$W_i = \max\left(1.0, \ln\left(1 + \text{likes}_i + 2 \times \text{reposts}_i + 0.05 \times \text{views}_i + 0.1 \times \text{followers}_i\right)\right)$$
$$\text{Social Sentiment} = \frac{\sum (S_i \cdot W_i)}{\sum W_i}$$
Composite sentiment blends $70\%$ social community signal with $30\%$ news sentiment from Exa.com.

### 2. Support & Resistance Calculations
Daily Pivot Points are computed per asset:
*   $\text{Pivot } (P) = \frac{\text{High} + \text{Low} + \text{Close}}{3}$
*   $\text{Resistance } 1 (R_1) = 2P - \text{Low}, \quad R_2 = P + (\text{High} - \text{Low})$
*   $\text{Support } 1 (S_1) = 2P - \text{High}, \quad S_2 = P - (\text{High} - \text{Low})$

### 3. Out-of-Sample Predictive Model ($n+1$) & Walk-Forward Validation
*   **Target Formulation:** Trained strictly on stationary percentage return:
    $$R_{n+1} = \frac{\text{Close}_{n+1} - \text{Close}_n}{\text{Close}_n}$$
    Future price is reconstructed without leakage: $\hat{P}_{n+1} = \text{Close}_n \times (1 + \hat{R}_{n+1})$.
*   **Feature Matrix (16 Features):**
    - *Technicals:* `return_1h`, `volatility_5h`, `dist_from_sma_5`, `dist_from_sma_10`, `rsi_14`, `dist_from_pivot`, `Volume`.
    - *Social Sentiment:* `hourly_social_sentiment`, `sentiment_momentum`, `sentiment_ma_3h`, `mentions`.
    - *Exa News Sentiment:* `news_sentiment`, `news_positive_count`, `news_negative_count`, `news_article_count`, `news_sentiment_ma_6h`.
*   **Naive Persistence Baseline Benchmark:** Evaluates model against the baseline ($\hat{P}_{n+1} = P_n$, i.e. zero return) across the exact same test bars.
*   **Validation Strategy:** 3-fold chronological `TimeSeriesSplit` walk-forward cross-validation plus final 20% holdout test evaluation. Zero random shuffling, zero future lookahead.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Tanush-J7/Assessment.git
cd Assessment

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file from `.env.example`:
```ini
APIFY_API_TOKEN=your_apify_token_here
EXA_API_KEY=your_exa_api_key_here

APIFY_TELEGRAM_ACTOR_ID=lofomachines/telegram-keyword-search-scraper
APIFY_WEB_SCRAPER_ACTOR_ID=apify/web-scraper
APIFY_LIHKG_ACTOR_ID=apify/cheerio-scraper
APIFY_NAVER_ACTOR_ID=apify/cheerio-scraper
```
> *Note: If API tokens are not provided, the pipeline automatically falls back to the built-in realistic historical simulation dataset for instantaneous offline reproduction.*

### 3. Run the Complete Pipeline & Automated Tests
```bash
# Run automated test suite
python -m unittest tests/test_pipeline.py

# Run complete pipeline end-to-end
python run_pipeline.py
```
This executes all 6 steps in sequence and outputs:
- `data/processed/market_summary.json`
- `data/processed/predictions_summary.json`
- `reports/visual_report.html`

### 4. Launch the Interactive Dashboard
```bash
streamlit run src/dashboard/app.py
```
Open your browser at `http://localhost:8501` to view:
- Sortable Leaderboard by Social Mentions, Sentiment Score, and Return
- Candlestick Charts with Support & Resistance overlays
- Out-of-Sample Test predictions vs actuals
- Feature importance rankings

---

## 📊 Summary of Results

| Exchange | Ticker | Current Price | Pred Price ($n+1$) | Direction | Position | Model RMSE | Base RMSE | Hit Ratio |
|---|---|---|---|---|---|---|---|---|
| **HKEX** | `0700.HK` (Tencent) | 400.00 | 400.90 | Neutral | **Long (Buy)** | 3.155 | 2.625 | 28.6% |
| **HKEX** | `9988.HK` (Alibaba) | 94.00 | 93.57 | Neutral | **Short (Sell)** | **1.038** | **1.118** | **78.6%** |
| **HKEX** | `3690.HK` (Meituan) | 72.00 | 72.04 | Neutral | **Hold / Neutral** | 0.708 | 0.667 | 42.9% |
| **HKEX** | `1211.HK` (BYD) | 67.60 | 67.22 | Bullish | **Short (Sell)** | 0.734 | 0.663 | 50.0% |
| **HKEX** | `0005.HK` (HSBC) | 141.10 | 141.11 | Neutral | **Hold / Neutral** | 1.184 | 0.634 | 50.0% |
| **KRX** | `005930.KS` (Samsung) | 208,500.00 | 208,944.41 | Bullish | **Long (Buy)** | **2,033.779** | **2,142.331** | 33.3% |
| **KRX** | `000660.KS` (SK Hynix) | 1,267,000.00 | 1,248,330.24 | Bullish | **Short (Sell)** | 20,177.107 | 12,372.685 | 41.7% |
| **KRX** | `035420.KS` (NAVER) | 181,900.00 | 183,669.87 | Neutral | **Long (Buy)** | 4,578.498 | 3,379.811 | 25.0% |
| **KRX** | `005380.KS` (Hyundai) | 265,500.00 | 266,066.63 | Neutral | **Long (Buy)** | 1,511.304 | 1,410.526 | 66.7% |
| **KRX** | `051910.KS` (LG Chem) | 235,500.00 | 234,283.76 | Neutral | **Short (Sell)** | 2,429.283 | 2,250.000 | 50.0% |

### 📝 Live Ingestion & LG Chem Methodology Disclosure
*   **Exa.com News:** 100% live semantic news articles ingested across all 10 target assets (8 articles per ticker, 80 total) within the 10-day lookback window.
*   **Apify Social Media Scraping:** Dual-source live scraping executed via `lofomachines/telegram-keyword-search-scraper` (Telegram) and `apify/cheerio-scraper` (LIHKG for HKEX, Naver Finance 종목토론실 for KRX). 9 out of 10 assets contain 100% live scraped data across both platforms.
*   **LG Chem (`051910.KS`) Disclosure:** Live Naver Finance 종목토론실 scraping succeeded via Cheerio (`apify/cheerio-scraper`). Telegram keyword search for Korean hashtag `#LG화학` encountered an Apify container timeout; the pipeline automatically supplemented representative Telegram data to ensure pipeline execution without failure. Raw JSON records on disk explicitly distinguish data provenance via `"data_type": "live"` (`"is_live": true`) vs `"data_type": "representative"` (`"is_live": false`).

---

## 📬 Submission Checklist
- [x] Python-based sentiment pipeline ($n+1$ price prediction)
- [x] Min. 2 sources per exchange (LIHKG + Telegram for HKEX; Naver + Telegram for KRX)
- [x] Apify & Exa.com integration with documented Actor IDs
- [x] 10 assets analyzed (5 HKEX, 5 KRX)
- [x] Post timestamps (UTC alignment), engagement metrics & author follower weighting
- [x] Per-ticker extraction: Ticker, Name, Price, Sentiment, Direction, Position, Support/Resistance
- [x] Visual Report: Sorted by mentions/sentiment (Streamlit + standalone HTML)
- [x] Rigorous out-of-sample validation metrics (RMSE, MAE, MAPE, Hit Ratio)
