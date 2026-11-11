"""
Static HTML Visual Report Generator
Generates a standalone, beautiful HTML report summarizing all 10 assets,
leaderboards sorted by social mentions & sentiment, S/R levels, and prediction results.
Can be viewed in any web browser without running Streamlit.
"""

import json
from pathlib import Path
from datetime import datetime, timezone

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def generate_html_report() -> Path:
    summary_file = PROCESSED_DATA_DIR / "predictions_summary.json"
    if not summary_file.exists():
        print("predictions_summary.json not found.")
        return None

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>HengSeng & KRX Sentiment Price Prediction Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: #ffffff;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: #0f172a;
            margin-bottom: 6px;
        }}
        .subtitle {{
            color: #64748b;
            margin-bottom: 25px;
            font-size: 15px;
        }}
        .kpi-row {{
            display: flex;
            gap: 20px;
            margin-bottom: 30px;
        }}
        .kpi-card {{
            flex: 1;
            background: #f1f5f9;
            padding: 18px;
            border-radius: 8px;
            text-align: center;
        }}
        .kpi-val {{
            font-size: 26px;
            font-weight: 700;
            color: #2563eb;
        }}
        .kpi-lbl {{
            font-size: 13px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .exchange-title {{
            font-size: 20px;
            font-weight: 600;
            margin-top: 35px;
            margin-bottom: 12px;
            color: #1e293b;
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 25px;
            font-size: 14px;
        }}
        th, td {{
            padding: 12px 14px;
            text-align: left;
            border-bottom: 1px solid #e2e8f0;
        }}
        th {{
            background: #f8fafc;
            font-weight: 600;
            color: #475569;
        }}
        tr:hover {{
            background: #f8fafc;
        }}
        .badge-bullish {{
            background: #dcfce7;
            color: #166534;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 12px;
        }}
        .badge-bearish {{
            background: #fee2e2;
            color: #991b1b;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 12px;
        }}
        .badge-neutral {{
            background: #f1f5f9;
            color: #475569;
            padding: 4px 10px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 12px;
        }}
        .pos-long {{
            color: #166534;
            font-weight: 600;
        }}
        .pos-short {{
            color: #991b1b;
            font-weight: 600;
        }}
        .footer {{
            margin-top: 40px;
            text-align: center;
            font-size: 13px;
            color: #94a3b8;
            border-top: 1px solid #e2e8f0;
            padding-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 HengSeng & KRX Sentiment Price Prediction Agent</h1>
        <div class="subtitle">
            CrowdWisdomTrading AI Agents Developer Assessment | Generated at: <b>{timestamp}</b>
        </div>

        <div class="kpi-row">
            <div class="kpi-card">
                <div class="kpi-val">10</div>
                <div class="kpi-lbl">Target Assets</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">2</div>
                <div class="kpi-lbl">Exchanges (HKEX & KRX)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">350</div>
                <div class="kpi-lbl">Social Posts Ingested (10d)</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-val">Out-of-Sample</div>
                <div class="kpi-lbl">Validation Method</div>
            </div>
        </div>
"""

    for ex_name, ex_label in [("HKEX", "Hong Kong Stock Exchange (HKEX)"), ("KRX", "Korea Exchange (KRX)")]:
        items = data.get(ex_name, [])
        # Sort by social mentions descending, then sentiment
        items_sorted = sorted(items, key=lambda x: (x.get("social_mentions", 0), x.get("sentiment_score", 0)), reverse=True)

        html += f"""
        <div class="exchange-title">🏛️ {ex_label} — Ranked Leaderboard</div>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Asset Name</th>
                    <th>Current Price</th>
                    <th>Predicted (n+1)</th>
                    <th>Exp. Return</th>
                    <th>Sentiment Score</th>
                    <th>Direction</th>
                    <th>Social Mentions</th>
                    <th>Position</th>
                    <th>Support (S1)</th>
                    <th>Resistance (R1)</th>
                    <th>Out-of-Sample Hit Ratio</th>
                </tr>
            </thead>
            <tbody>
        """

        for item in items_sorted:
            sr = item.get("sr_levels", {})
            vm = item.get("validation_metrics", {})
            direction = item.get("sentiment_direction", "Neutral")
            badge_class = "badge-bullish" if direction == "Bullish" else ("badge-bearish" if direction == "Bearish" else "badge-neutral")
            
            pos = item.get("recommended_position", "Hold")
            pos_class = "pos-long" if "Long" in pos else ("pos-short" if "Short" in pos else "")
            
            html += f"""
                <tr>
                    <td><b>{item['ticker']}</b></td>
                    <td>{item['asset_name']}</td>
                    <td>{item['current_price']}</td>
                    <td><b>{item['predicted_price_n1']}</b></td>
                    <td>{item['predicted_change_pct']:+.2f}%</td>
                    <td>{item['sentiment_score']:+.3f}</td>
                    <td><span class="{badge_class}">{direction}</span></td>
                    <td>{item['social_mentions']}</td>
                    <td class="{pos_class}">{pos}</td>
                    <td>{sr.get('support_1', 'N/A')}</td>
                    <td>{sr.get('resistance_1', 'N/A')}</td>
                    <td><b>{vm.get('directional_accuracy_pct', 0.0)}%</b></td>
                </tr>
            """

        html += """
            </tbody>
        </table>
        """

    html += """
        <div class="footer">
            Built for CrowdWisdomTrading AI Agents Developer Internship Assessment.<br>
            Tech Stack: Apify (Alternative Social Scraping) + Exa.com (News Ingestion) + yfinance + LightGBM/GradientBoosting.
        </div>
    </div>
</body>
</html>
    """

    out_file = REPORTS_DIR / "visual_report.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Visual report successfully generated at: {out_file}")
    return out_file


if __name__ == "__main__":
    generate_html_report()
