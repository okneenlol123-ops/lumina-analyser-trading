# daytrading_offline.py
import streamlit as st
import pandas as pd
import random
import hashlib
from datetime import datetime, timedelta
import plotly.graph_objects as go

# -------------------
# Streamlit Config
# -------------------
st.set_page_config(page_title="Offline Daytrading Analyzer", layout="wide", page_icon="📈")

# -------------------
# Assets
# -------------------
ETFS = [{"id": f"ETF_{i}", "name": f"ETF {i}"} for i in range(1, 11)]
STOCKS = [{"id": f"STOCK_{i}", "name": f"Aktie {i}"} for i in range(1, 21)]
CRYPTOS = [{"id": f"CRYPTO_{i}", "name": f"Krypto {i}"} for i in range(1, 11)]

# -------------------
# Deterministische Preisserie
# -------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def generate_candles(asset_id, interval_minutes=5, periods=100):
    rnd = random.Random(deterministic_seed(asset_id + str(interval_minutes)))
    price = 100.0
    candles = []
    ts = datetime.utcnow()
    delta = timedelta(minutes=interval_minutes)
    for _ in range(periods):
        open_p = price
        close_p = max(0.01, open_p * (1 + (rnd.random() - 0.5) * 0.02))
        high_p = max(open_p, close_p) * (1 + rnd.random() * 0.01)
        low_p = min(open_p, close_p) * (1 - rnd.random() * 0.01)
        candles.append({
            "datetime": ts,
            "open": round(open_p, 2),
            "high": round(high_p, 2),
            "low": round(low_p, 2),
            "close": round(close_p, 2)
        })
        price = close_p
        ts -= delta
    return pd.DataFrame(candles[::-1])

# -------------------
# Candlestick Pattern Recognition
# -------------------
def recognize_candlestick_pattern(df: pd.DataFrame):
    if len(df) < 3:
        return "Halten"
    last = df.iloc[-1]
    prev = df.iloc[-2]

    body = abs(last['close'] - last['open'])
    candle_range = last['high'] - last['low']
    upper_shadow = last['high'] - max(last['open'], last['close'])
    lower_shadow = min(last['open'], last['close']) - last['low']

    # Doji
    if body / candle_range < 0.2:
        return "Halten (Doji)"
    # Bullish Engulfing
    if last['close'] > last['open'] and prev['close'] < prev['open'] and last['close'] > prev['open']:
        return "Kaufen (Bullish Engulfing)"
    # Bearish Engulfing
    if last['close'] < last['open'] and prev['close'] > prev['open'] and last['close'] < prev['open']:
        return "Verkaufen (Bearish Engulfing)"
    # Hammer
    if lower_shadow > 2 * body and last['close'] > last['open']:
        return "Kaufen (Hammer)"
    # Shooting Star
    if upper_shadow > 2 * body and last['close'] < last['open']:
        return "Verkaufen (Shooting Star)"
    return "Halten"

# -------------------
# UI
# -------------------
st.title("Offline Daytrading Analyzer")

# Interval selection
interval_map = {
    "1 Minute": 1, "5 Minuten": 5, "10 Minuten": 10, "30 Minuten": 30,
    "1 Stunde": 60, "3 Stunden": 180, "12 Stunden": 720, "1 Tag": 1440
}
interval_str = st.selectbox("Intervall wählen", list(interval_map.keys()))
interval = interval_map[interval_str]

# Category selection
category = st.radio("Kategorie", ["ETF", "Aktien", "Krypto"])
if category == "ETF":
    assets = ETFS
elif category == "Aktien":
    assets = STOCKS
else:
    assets = CRYPTOS

# Asset search
search = st.text_input("Asset suchen")
if search:
    assets = [a for a in assets if search.lower() in a['name'].lower()]

# Display assets
for asset in assets:
    st.subheader(asset['name'])
    df = generate_candles(asset['id'], interval_minutes=interval, periods=100)

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['datetime'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    rec = recognize_candlestick_pattern(df)
    st.info(f"Empfehlung: {rec}")

st.markdown("---")
st.write("Alle Kurse offline simuliert, deterministisch und wiederholbar.")
