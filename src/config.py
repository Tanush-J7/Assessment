import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# Apify Actor IDs
APIFY_TELEGRAM_ACTOR_ID = os.getenv("APIFY_TELEGRAM_ACTOR_ID", "apify/telegram-scraper")
APIFY_WEB_SCRAPER_ACTOR_ID = os.getenv("APIFY_WEB_SCRAPER_ACTOR_ID", "apify/web-scraper")
APIFY_LIHKG_ACTOR_ID = os.getenv("APIFY_LIHKG_ACTOR_ID", "apify/cheerio-scraper")
APIFY_NAVER_ACTOR_ID = os.getenv("APIFY_NAVER_ACTOR_ID", "apify/cheerio-scraper")

# Target Assets per Exchange (5 HKEX, 5 KRX)
TARGET_ASSETS = {
    "HKEX": [
        {"ticker": "0700.HK", "name": "Tencent Holdings", "cashtags": ["$0700", "$TCEHY", "#腾讯", "#騰訊"], "keywords": ["Tencent", "騰訊控股", "700"]},
        {"ticker": "9988.HK", "name": "Alibaba Group", "cashtags": ["$9988", "$BABA", "#阿里", "#阿里巴巴"], "keywords": ["Alibaba", "阿里巴巴", "9988"]},
        {"ticker": "3690.HK", "name": "Meituan", "cashtags": ["$3690", "#美团", "#美團"], "keywords": ["Meituan", "美團", "3690"]},
        {"ticker": "1211.HK", "name": "BYD Company", "cashtags": ["$1211", "#比亚迪", "#比亞迪"], "keywords": ["BYD", "比亞迪", "1211"]},
        {"ticker": "0005.HK", "name": "HSBC Holdings", "cashtags": ["$0005", "$HSBC", "#汇丰", "#匯豐"], "keywords": ["HSBC", "匯豐控股", "5"]},
    ],
    "KRX": [
        {"ticker": "005930.KS", "name": "Samsung Electronics", "cashtags": ["$005930", "#삼성전자"], "keywords": ["Samsung Electronics", "삼성전자", "삼전"]},
        {"ticker": "000660.KS", "name": "SK Hynix", "cashtags": ["$000660", "#SK하이닉스"], "keywords": ["SK Hynix", "SK하이닉스", "하이닉스"]},
        {"ticker": "035420.KS", "name": "NAVER", "cashtags": ["$035420", "#네이버"], "keywords": ["NAVER", "네이버"]},
        {"ticker": "005380.KS", "name": "Hyundai Motor", "cashtags": ["$005380", "#현대차"], "keywords": ["Hyundai Motor", "현대차", "현대자동차"]},
        {"ticker": "051910.KS", "name": "LG Chem", "cashtags": ["$051910", "#LG화학"], "keywords": ["LG Chem", "LG화학"]},
    ]
}

# Social Sources Configuration
SOCIAL_SOURCES = {
    "HKEX": ["lihkg", "telegram"],
    "KRX": ["naver_finance", "telegram"]
}

# Lookback Window
LOOKBACK_DAYS = 10
TIME_INTERVAL = "1h" # Can be 1h or 1d
