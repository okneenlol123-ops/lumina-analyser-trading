# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import json, os, random
from datetime import datetime, timedelta

# -------------------
# Config & Files
# -------------------
st.set_page_config(page_title="Offline Daytrading Analyzer", page_icon="📈", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"

# -------------------
# Utility functions
# -------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -------------------
# Initialize
# -------------------
if not os.path.exists(PORTFOLIO_FILE):
    save_json(PORTFOLIO_FILE, [])
if not os.path.exists(SETTINGS_FILE):
    save_json(SETTINGS_FILE, {"theme":"dark"})

portfolio = load_json(PORTFOLIO_FILE, [])
settings = load_json(SETTINGS_FILE, {"theme":"dark"})

# -------------------
# Assets
# -------------------
ETFS = [{"id":f"ETF_{i}","name":f"ETF {i}"} for i in range(1,11)]
STOCKS = [{"id":f"STOCK_{i}","name":f"Aktie {i}"} for i in range(1,21)]
CRYPTOS = [{"id":f"CRYPTO_{i}","name":f"Krypto {i}"} for i in range(1,11)]

ALL_ASSETS = ETFS + STOCKS + CRYPTOS

# -------------------
# Deterministic price series generator
# -------------------
def deterministic_seed(s:str)->int:
    return int(abs(hash(s)) % (2**32))

def generate_candle_series(asset_id, periods=100, interval_minutes=5):
    rnd = random.Random(deterministic_seed(asset_id))
    df = pd.DataFrame(columns=["datetime","open","high","low","close","volume"])
    price = rnd.uniform(20,200)
    for i in range(periods):
        dt = datetime.utcnow() - timedelta(minutes=interval_minutes*(periods-i))
        o = price
        c = max(0.1, o * (1 + rnd.uniform(-0.02,0.02)))
        h = max(o,c) * (1 + rnd.uniform(0,0.01))
        l = min(o,c) * (1 - rnd.uniform(0,0.01))
        vol = rnd.randint(100,1000)
        df.loc[i] = [dt,o,h,l,c,vol]
        price = c
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

# -------------------
# Candlestick analysis (simplified patterns)
# -------------------
def analyze_candles(df):
    """
    Returns recommendation based on simple candle patterns:
    - Bullish engulfing, hammer -> Buy
    - Bearish engulfing, shooting star -> Sell
    """
    if len(df)<2: return "Halten", 0.0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    # Simple rules
    body = abs(last["close"] - last["open"])
    prev_body = abs(prev["close"] - prev["open"])
    candle_ratio = body/(prev_body+0.0001)
    # Bullish engulfing
    if last["close"] > last["open"] and candle_ratio>1.5 and last["close"]>prev["close"]:
        return "Kaufen", min(body*5,10)
    # Bearish engulfing
    if last["close"] < last["open"] and candle_ratio>1.5 and last["close"]<prev["close"]:
        return "Verkaufen", -min(body*5,10)
    # Hammer / Shooting star
    lower_wick = last["open"] - last["low"]
    upper_wick = last["high"] - last["close"]
    if lower_wick>2*body:
        return "Kaufen", min(lower_wick*3,8)
    if upper_wick>2*body:
        return "Verkaufen", -min(upper_wick*3,8)
    return "Halten", 0.0

# -------------------
# UI
# -------------------
st.title("Offline Daytrading Analyzer 📈")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Marktplatz","Portfolio","Einstellungen"])

# -------------------
# Marktplatz / Daytrading
# -------------------
if page=="Marktplatz":
    st.header("Daytrading Marktplatz")
    asset_type = st.selectbox("Asset Kategorie", ["ETF","Aktie","Krypto"])
    if asset_type=="ETF": assets=ETFS
    elif asset_type=="Aktie": assets=STOCKS
    else: assets=CRYPTOS
    # Search
    search = st.text_input("Suche Asset")
    if search:
        assets = [a for a in assets if search.lower() in a["name"].lower()]
    interval_label = st.selectbox("Intervall", ["1min","5min","10min","30min","1h","3h","12h","1d"])
    interval_map = {"1min":1,"5min":5,"10min":10,"30min":30,"1h":60,"3h":180,"12h":720,"1d":1440}
    interval = interval_map[interval_label]

    for a in assets:
        st.subheader(a["name"])
        df = generate_candle_series(a["id"], periods=100, interval_minutes=interval)
        recommendation, expected = analyze_candles(df)
        fig = go.Figure(data=[go.Candlestick(
            x=df['datetime'],
            open=df['open'], high=df['high'],
            low=df['low'], close=df['close'],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Empfehlung: **{recommendation}** • Geschätzter Gewinn: {expected:.2f}%")
        # Portfolio add form
        with st.form(key=f"add_{a['id']}"):
            qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
            buy_price = st.number_input("Kaufpreis", min_value=0.0001, value=float(df["close"].iloc[-1]), step=0.01, key=f"bp_{a['id']}")
            if st.form_submit_button("Hinzufügen"):
                portfolio.append({"id":a["id"],"name":a["name"],"qty":qty,"buy_price":buy_price,"added_at":datetime.utcnow().isoformat()})
                save_json(PORTFOLIO_FILE, portfolio)
                st.success(f"{a['name']} hinzugefügt.")

# -------------------
# Portfolio
# -------------------
elif page=="Portfolio":
    st.header("Portfolio")
    if not portfolio:
        st.info("Portfolio ist leer.")
    else:
        for p in portfolio:
            st.write(f"{p['name']} | Menge: {p['qty']} | Kaufpreis: {p['buy_price']:.2f}")
            df = generate_candle_series(p["id"], periods=100, interval_minutes=5)
            fig = go.Figure(data=[go.Candlestick(
                x=df['datetime'],
                open=df['open'], high=df['high'],
                low=df['low'], close=df['close'],
                increasing_line_color='green', decreasing_line_color='red'
            )])
            st.plotly_chart(fig, use_container_width=True)
            recommendation, expected = analyze_candles(df)
            st.write(f"Empfehlung aktuell: **{recommendation}** • Geschätzter Gewinn: {expected:.2f}%")
        if st.button("Portfolio löschen"):
            portfolio.clear()
            save_json(PORTFOLIO_FILE, portfolio)
            st.experimental_rerun()

# -------------------
# Einstellungen
# -------------------
elif page=="Einstellungen":
    st.header("Einstellungen")
    theme = st.selectbox("Theme", ["dark","light"])
    settings["theme"] = theme
    if st.button("Speichern"):
        save_json(SETTINGS_FILE, settings)
        st.success("Einstellungen gespeichert.")
