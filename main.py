import streamlit as st
import random
from datetime import datetime, timedelta
import math

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
st.set_page_config(page_title="Daytrading Simulator", layout="wide", page_icon="📈")

# DARK THEME (offline)
st.markdown("""
    <style>
    html, body, [class*="css"] {
        background-color: #000 !important;
        color: #e6eef6 !important;
    }
    .stButton>button {
        background-color: #111; color: #e6eef6; border: 1px solid #2f2f2f;
        border-radius: 6px; padding: 8px 20px;
    }
    .buy {color:#00ff99; font-weight:bold;}
    .sell {color:#ff3366; font-weight:bold;}
    .neutral {color:#cccccc;}
    .candles {font-family: monospace; white-space: pre;}
    </style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# ASSETS
# ----------------------------------------------------------
ETFS = [f"ETF_{n}" for n in [
    "Deutschland", "USA", "Europa", "Asien", "EmergingMarkets",
    "Tech", "S&P500", "DAX", "Nachhaltig", "Welt"
]]

STOCKS = [f"Aktie_{n}" for n in [
    "Apple", "Tesla", "Microsoft", "Amazon", "Nvidia", "Meta",
    "Alphabet", "Netflix", "Intel", "AMD", "IBM", "CocaCola",
    "Bayer", "SAP", "Allianz", "Siemens", "VW", "Mercedes",
    "Shell", "DeutscheBank"
]]

CRYPTOS = [f"Krypto_{n}" for n in [
    "Bitcoin", "Ethereum", "Solana", "Cardano", "Ripple",
    "Dogecoin", "Polkadot", "Chainlink", "Tron", "Litecoin"
]]

# ----------------------------------------------------------
# CANDLE GENERATOR
# ----------------------------------------------------------
def generate_candles(asset, intervals=50, start_price=100.0):
    """Simuliere Candle-Daten (Open, High, Low, Close)"""
    candles = []
    price = start_price
    rnd = random.Random(hash(asset) % 999999)
    for i in range(intervals):
        change = rnd.uniform(-2, 2)
        open_p = price
        close_p = max(0.1, price + change)
        high = max(open_p, close_p) + rnd.uniform(0, 1)
        low = min(open_p, close_p) - rnd.uniform(0, 1)
        candles.append((open_p, high, low, close_p))
        price = close_p
    return candles

# ----------------------------------------------------------
# ASCII CHART RENDERER (Offline)
# ----------------------------------------------------------
def render_candle_chart(candles, height=20, width=60):
    """Zeichnet Kerzen als ASCII-Balken"""
    max_p = max(c[1] for c in candles)
    min_p = min(c[2] for c in candles)
    scale = height / (max_p - min_p + 1e-6)
    chart = [[" " for _ in range(width)] for _ in range(height)]
    step = max(1, len(candles)//width)
    for i, c in enumerate(candles[::step]):
        o,h,l,cl = c
        top = int((h - min_p) * scale)
        bottom = int((l - min_p) * scale)
        open_y = int((o - min_p) * scale)
        close_y = int((cl - min_p) * scale)
        col = i
        for y in range(bottom, top):
            if y < height and col < width:
                chart[height-1-y][col] = "|"
        body_top = max(open_y, close_y)
        body_bottom = min(open_y, close_y)
        for y in range(body_bottom, body_top+1):
            if y < height and col < width:
                chart[height-1-y][col] = "█" if cl >= o else "░"
    return "\n".join("".join(row) for row in chart)

# ----------------------------------------------------------
# ANALYSE
# ----------------------------------------------------------
def analyze_pattern(candles):
    """Einfache Muster-Analyse für Kauf/Nichtkauf"""
    last = candles[-5:]
    closes = [c[3] for c in last]
    avg_diff = sum(closes[i]-closes[i-1] for i in range(1,len(closes))) / (len(closes)-1)
    trend = "steigend" if avg_diff > 0 else "fallend"
    volatility = max(c[1]-c[2] for c in last)
    if avg_diff > 0.5 and volatility < 2:
        recommendation = "Kaufen"
        risk = "Niedrig"
        potential = "+3–7%"
    elif avg_diff < -0.5 and volatility > 2:
        recommendation = "Nicht kaufen"
        risk = "Hoch"
        potential = "-5–10%"
    else:
        recommendation = "Beobachten"
        risk = "Mittel"
        potential = "±2–3%"
    return trend, risk, recommendation, potential

# ----------------------------------------------------------
# UI
# ----------------------------------------------------------
st.title("📊 Offline Daytrading Simulator")
st.write("Candlestick-Simulation mit Analyse (Offline, keine echten Daten).")

category = st.selectbox("Kategorie wählen", ["ETFs", "Aktien", "Krypto"])
if category == "ETFs":
    asset_list = ETFS
elif category == "Aktien":
    asset_list = STOCKS
else:
    asset_list = CRYPTOS

asset = st.selectbox("Asset wählen", asset_list)
interval = st.selectbox("Zeitraum / Intervall", [
    "1 Minute", "5 Minuten", "10 Minuten", "30 Minuten", "1 Stunde", "3 Stunden", "12 Stunden", "1 Tag"
])

candles = generate_candles(asset, 60, start_price=100 + hash(asset)%50)
chart_text = render_candle_chart(candles)
trend, risk, recommendation, potential = analyze_pattern(candles)

st.subheader(f"{asset} — Candlestick Chart ({interval})")
st.markdown(f"<div class='candles'>{chart_text}</div>", unsafe_allow_html=True)
st.write(f"**Trend:** {trend.capitalize()}")
st.write(f"**Risiko:** {risk}")
st.write(f"**Erwartetes Ergebnis:** {potential}")
if recommendation == "Kaufen":
    st.markdown("**Empfehlung:** <span class='buy'>🟢 Kaufen</span>", unsafe_allow_html=True)
elif recommendation == "Nicht kaufen":
    st.markdown("**Empfehlung:** <span class='sell'>🔴 Nicht kaufen</span>", unsafe_allow_html=True)
else:
    st.markdown("**Empfehlung:** <span class='neutral'>⚪ Beobachten</span>", unsafe_allow_html=True)

# ----------------------------------------------------------
# FAKE LIVE UPDATE (optional)
# ----------------------------------------------------------
if st.button("🔁 Neue Simulation"):
    st.experimental_rerun()
