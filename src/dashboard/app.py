"""
Interactive Streamlit Dashboard & Visual Report
Visualizes HKEX & KRX sentiment price prediction results,
leaderboards sorted by mentions/sentiment, candlestick charts with S/R levels,
and out-of-sample validation metrics.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
from pathlib import Path
import os

st.set_page_config(
    page_title="HangSeng & KRX Sentiment Price Prediction Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PREDICTIONS_FILE = PROCESSED_DATA_DIR / "predictions_summary.json"


@st.cache_data
def load_predictions():
    if not PREDICTIONS_FILE.exists():
        return {}
    with open(PREDICTIONS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    st.title("📈 HangSeng & KRX Sentiment Price Prediction Agent")
    st.markdown(
        "**AI Agents Developer Assessment** | Multi-source sentiment ingestion (Apify + Exa.com) "
        "and out-of-sample validated predictive model for $n+1$ price forecasting."
    )

    data = load_predictions()
    if not data:
        st.error("No prediction results found. Please run the training pipeline first (`python -m src.models.predictor`).")
        return

    # Top KPI Metrics
    all_tickers = []
    for ex_assets in data.values():
        all_tickers.extend(ex_assets)

    total_assets = len(all_tickers)
    total_mentions = sum(item.get("social_mentions", 0) for item in all_tickers)
    bullish_count = sum(1 for item in all_tickers if item.get("sentiment_direction") == "Bullish")
    bearish_count = sum(1 for item in all_tickers if item.get("sentiment_direction") == "Bearish")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Assets Analyzed", f"{total_assets} (HKEX & KRX)")
    col2.metric("Total Social Mentions", f"{total_mentions} posts")
    col3.metric("Bullish Assets", f"{bullish_count} / {total_assets}")
    col4.metric("Bearish Assets", f"{bearish_count} / {total_assets}")

    st.divider()

    # Exchange Selection
    selected_exchange = st.sidebar.selectbox("Select Exchange", ["HKEX (Hong Kong)", "KRX (South Korea)"])
    exchange_key = "HKEX" if "HKEX" in selected_exchange else "KRX"

    exchange_assets = data.get(exchange_key, [])
    if not exchange_assets:
        st.warning(f"No data available for {exchange_key}")
        return

    # Sidebar Sorting
    st.sidebar.subheader("Sort Leaderboard By")
    sort_by = st.sidebar.radio(
        "Sorting Metric",
        ["Social Mentions (High to Low)", "Sentiment Score (High to Low)", "Predicted Return (High to Low)"]
    )

    # Convert to DataFrame for sorting & display
    records = []
    for item in exchange_assets:
        sr = item.get("sr_levels", {})
        vm = item.get("validation_metrics", {})
        fc = item.get("feature_contributions", {})
        records.append({
            "Ticker": item["ticker"],
            "Asset Name": item["asset_name"],
            "Current Price": item["current_price"],
            "Predicted Price (n+1)": item["predicted_price_n1"],
            "Exp. Return (%)": item["predicted_change_pct"],
            "Sentiment Score": item["sentiment_score"],
            "Direction": item["sentiment_direction"],
            "Social Mentions": item["social_mentions"],
            "Position": item["recommended_position"],
            "Support (S1)": sr.get("support_1", 0.0),
            "Resistance (R1)": sr.get("resistance_1", 0.0),
            "Model RMSE": vm.get("model_rmse", vm.get("rmse", 0.0)),
            "Baseline RMSE": vm.get("baseline_rmse", 0.0),
            "Hit Ratio (%)": vm.get("directional_accuracy_pct", 0.0),
            "Social Imp (%)": fc.get("social_sentiment_pct", 0.0),
            "News Imp (%)": fc.get("news_sentiment_pct", 0.0)
        })

    df_display = pd.DataFrame(records)

    if sort_by == "Social Mentions (High to Low)":
        df_display.sort_values(by="Social Mentions", ascending=False, inplace=True)
    elif sort_by == "Sentiment Score (High to Low)":
        df_display.sort_values(by="Sentiment Score", ascending=False, inplace=True)
    else:
        df_display.sort_values(by="Exp. Return (%)", ascending=False, inplace=True)

    # Display Leaderboard
    st.subheader(f"📊 {selected_exchange} — Assets Leaderboard (Sorted by {sort_by})")
    
    # Styled Table
    def highlight_direction(val):
        if val == "Bullish":
            return "background-color: #d4edda; color: #155724; font-weight: bold;"
        elif val == "Bearish":
            return "background-color: #f8d7da; color: #721c24; font-weight: bold;"
        return "background-color: #e2e3e5; color: #383d41;"

    styled_df = df_display.style.map(highlight_direction, subset=["Direction"])
    st.dataframe(styled_df, width="stretch")

    st.divider()

    # Asset Deep Dive Section
    st.subheader("🔍 Asset Deep-Dive & Forecast Verification")
    selected_ticker = st.selectbox(
        "Choose an Asset for Detailed Analysis",
        options=df_display["Ticker"].tolist(),
        format_func=lambda x: f"{x} - {df_display.loc[df_display['Ticker'] == x, 'Asset Name'].values[0]}"
    )

    asset_detail = next(item for item in exchange_assets if item["ticker"] == selected_ticker)

    tab1, tab2, tab3 = st.tabs(["Price & S/R Chart", "Out-of-Sample & Walk-Forward Validation", "Feature Contribution & News"])

    with tab1:
        price_file = RAW_DATA_DIR / "market_data" / f"{selected_ticker.replace('.', '_')}_1h.csv"
        if price_file.exists():
            df_price = pd.read_csv(price_file)
            fig = go.Figure()
            # Candlestick or Line
            if all(col in df_price.columns for col in ["Open", "High", "Low", "Close"]):
                fig.add_trace(go.Candlestick(
                    x=df_price["timestamp_utc"],
                    open=df_price["Open"],
                    high=df_price["High"],
                    low=df_price["Low"],
                    close=df_price["Close"],
                    name="OHLC Price"
                ))
            else:
                fig.add_trace(go.Scatter(x=df_price["timestamp_utc"], y=df_price["Close"], mode="lines", name="Close Price"))

            # S/R Horizontal lines
            sr = asset_detail.get("sr_levels", {})
            if "support_1" in sr:
                fig.add_hline(y=sr["support_1"], line_dash="dash", line_color="green", annotation_text=f"S1: {sr['support_1']}")
            if "resistance_1" in sr:
                fig.add_hline(y=sr["resistance_1"], line_dash="dash", line_color="red", annotation_text=f"R1: {sr['resistance_1']}")
            if "pivot_point" in sr:
                fig.add_hline(y=sr["pivot_point"], line_dash="dot", line_color="blue", annotation_text=f"Pivot: {sr['pivot_point']}")

            fig.update_layout(
                title=f"{asset_detail['asset_name']} ({selected_ticker}) — 10-Day Price Action & S/R Levels (UTC)",
                xaxis_title="UTC Timestamp",
                yaxis_title="Price",
                height=480,
                xaxis_rangeslider_visible=False
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Price history file not found.")

    with tab2:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        vm = asset_detail.get("validation_metrics", {})
        wf = asset_detail.get("walk_forward_cv", {})

        col_m1.metric("Test Samples (Holdout)", vm.get("test_samples", 0))
        col_m2.metric(
            "Model RMSE vs Baseline",
            f"{vm.get('model_rmse', vm.get('rmse', 0.0))}",
            delta=f"Baseline: {vm.get('baseline_rmse', 0.0)}",
            delta_color="inverse"
        )
        col_m3.metric(
            "Model MAE vs Baseline",
            f"{vm.get('model_mae', vm.get('mae', 0.0))}",
            delta=f"Baseline: {vm.get('baseline_mae', 0.0)}",
            delta_color="inverse"
        )
        col_m4.metric("Directional Hit Ratio", f"{vm.get('directional_accuracy_pct', 0.0)}%")

        st.markdown("#### 🔄 Walk-Forward Cross-Validation (TimeSeriesSplit)")
        if wf and "folds" in wf:
            df_folds = pd.DataFrame(wf["folds"])
            col_wf1, col_wf2 = st.columns([2, 1])
            with col_wf1:
                st.dataframe(df_folds, width="stretch")
            with col_wf2:
                st.info(
                    f"**CV Mean Model RMSE:** {wf.get('mean_model_rmse', 0.0)}\n\n"
                    f"**CV Mean Baseline RMSE:** {wf.get('mean_baseline_rmse', 0.0)}\n\n"
                    f"**CV Mean Directional Hit Ratio:** {wf.get('mean_hit_ratio_pct', 0.0)}%"
                )

        test_data = asset_detail.get("test_actual_vs_pred", [])
        if test_data:
            df_test = pd.DataFrame(test_data)
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(y=df_test["actual"], mode="lines+markers", name="Actual Price", line=dict(color="black", width=2)))
            fig_val.add_trace(go.Scatter(y=df_test["predicted"], mode="lines+markers", name="Model Predicted (n+1)", line=dict(color="royalblue", dash="dash")))
            if "baseline" in df_test.columns:
                fig_val.add_trace(go.Scatter(y=df_test["baseline"], mode="lines", name="Naive Baseline (P_t)", line=dict(color="orange", dash="dot")))
            fig_val.update_layout(
                title="Holdout Test: Actual vs Model Predicted Price vs Naive Baseline",
                xaxis_title="Out-of-Sample Test Bar Index",
                yaxis_title="Price",
                height=380
            )
            st.plotly_chart(fig_val, width="stretch")

    with tab3:
        fc = asset_detail.get("feature_contributions", {})
        st.write("#### 🧩 Feature Category Contributions")
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric("Technical Indicators", f"{fc.get('technical_pct', 0.0)}%")
        col_c2.metric("Social Community Sentiment", f"{fc.get('social_sentiment_pct', 0.0)}%")
        col_c3.metric("Exa News Sentiment", f"{fc.get('news_sentiment_pct', 0.0)}%")

        st.write("#### Detailed Feature Importance Breakdown")
        importances = asset_detail.get("feature_importances", {})
        if importances:
            df_imp = pd.DataFrame(list(importances.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
            fig_imp = px.bar(df_imp, x="Importance", y="Feature", orientation="h", title="Gradient Boosting Split Importance")
            st.plotly_chart(fig_imp, width="stretch")

        st.write("#### Recent Exa.com News Headlines")
        news_file = RAW_DATA_DIR / "news" / f"{selected_ticker.replace('.', '_')}_news.json"
        if news_file.exists():
            with open(news_file, "r", encoding="utf-8") as f:
                news_items = json.load(f)
                for n in news_items[:3]:
                    st.markdown(f"- **[{n['title']}]({n['url']})** ({n['published_date_utc'][:10]})")


if __name__ == "__main__":
    main()
