import streamlit as st
import pandas as pd
import numpy as np
import json, os, random
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ------------------- Setup -------------------
st.set_page_config(page_title="Daytrading Analyzer", layout="wide", page_icon="📊")
PORTFOLIO_FILE = "portfolio.json"

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if not os.path.exists(PORTFOLIO_FILE):
    save_json(PORTFOLIO_FILE, [])

portfolio = load_json(PORTFOLIO_FILE, [])

# ------------------- Assets -------------------
ETFS = [f"ETF {i}" for i in range(1, 11)]
STOCKS = [f"Aktie {i}" for i in range(1, 21)]
CRYPTOS = [f"Krypto {i}" for i in range(1, 11)]

ALL_ASSETS = ETFS + STOCKS + CRYPTOS

# ------------------- Candle Generator -------------------
def deterministic_seed(name):
    return int(abs(hash(name)) % (2**32))

def generate_candles(asset, periods=100, interval_min=5):
    rnd = random.Random(deterministic_seed(asset))
    price = rnd.uniform(50, 200)
    candles = []
    for i in range(periods):
        dt = datetime.utcnow() - timedelta(minutes=interval_min*(periods-i))
        open_p = price
        close_p = open_p * (1 + rnd.uniform(-0.02, 0.02))
        high_p = max(open_p, close_p) * (1 + rnd.uniform(0, 0.01))
        low_p = min(open_p, close_p) * (1 - rnd.uniform(0, 0.01))
        candles.append([dt, open_p, high_p, low_p, close_p])
        price = close_p
    df = pd.DataFrame(candles, columns=["Datetime", "Open", "High", "Low", "Close"])
    return df

# ------------------- Candle Pattern Analyzer -------------------
def analyze_pattern(df):
    if len(df) < 2:
        return "Halten", 0.0

    last = df.iloc[-1]
    prev = df.iloc[-2]
    body = abs(last["Close"] - last["Open"])
    prev_body = abs(prev["Close"] - prev["Open"])

    lower_wick = last["Open"] - last["Low"]
    upper_wick = last["High"] - last["Close"]

    # Mustererkennung
    if last["Close"] > last["Open"] and body > prev_body * 1.5:
        return "Kaufen", random.uniform(2, 10)
    if last["Close"] < last["Open"] and body > prev_body * 1.5:
        return "Verkaufen", -random.uniform(2, 10)
    if lower_wick > 2 * body:
        return "Kaufen", random.uniform(1, 6)
    if upper_wick > 2 * body:
        return "Verkaufen", -random.uniform(1, 6)
    if body < 0.002 * last["Open"]:
        return "Halten", 0.0
    return "Halten", 0.0

# ------------------- Chart Renderer -------------------
def draw_candlestick(df, asset_name):
    fig = go.Figure(
        data=[go.Candlestick(
            x=df["Datetime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="green",
            decreasing_line_color="red"
        )]
    )
    fig.update_layout(
        title=f"Candlestick Chart - {asset_name}",
        xaxis_title="Zeit",
        yaxis_title="Preis",
        template="plotly_dark",
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# ------------------- App UI -------------------
st.title("📊 Offline Daytrading Analyzer")

page = st.sidebar.radio("Navigation", ["Marktplatz", "Portfolio", "Einstellungen"])

interval_map = {
    "1min": 1, "5min": 5, "10min": 10, "30min": 30,
    "1h": 60, "3h": 180, "12h": 720, "1d": 1440
}

# ------------------- Marktplatz -------------------
if page == "Marktplatz":
    st.header("Daytrading Marktplatz")

    asset_type = st.selectbox("Kategorie", ["ETF", "Aktie", "Krypto"])
    if asset_type == "ETF":
        assets = ETFS
    elif asset_type == "Aktie":
        assets = STOCKS
    else:
        assets = CRYPTOS

    search = st.text_input("Suche nach Namen")
    if search:
        assets = [a for a in assets if search.lower() in a.lower()]

    interval_choice = st.selectbox("Zeitintervall", list(interval_map.keys()))
    interval = interval_map[interval_choice]

    for asset in assets:
        st.subheader(asset)
        df = generate_candles(asset, periods=100, interval_min=interval)
        rec, exp = analyze_pattern(df)
        fig = draw_candlestick(df, asset)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"**Empfehlung:** {rec}  |  **Erwarteter Gewinn:** {exp:.2f}%")

        with st.form(key=f"add_{asset}"):
            qty = st.number_input("Menge", min_value=1, value=1)
            price = st.number_input("Kaufpreis", min_value=1.0, value=float(df['Close'].iloc[-1]))
            if st.form_submit_button("Hinzufügen ins Portfolio"):
                portfolio.append({
                    "name": asset,
                    "qty": qty,
                    "buy_price": price,
                    "added_at": datetime.utcnow().isoformat()
                })
                save_json(PORTFOLIO_FILE, portfolio)
                st.success(f"{asset} hinzugefügt!")

# ------------------- Portfolio -------------------
elif page == "Portfolio":
    st.header("Dein Portfolio")
    if not portfolio:
        st.info("Noch keine Einträge.")
    else:
        total_value = 0
        for item in portfolio:
            df = generate_candles(item["name"], periods=60)
            current_price = float(df["Close"].iloc[-1])
            gain = (current_price - item["buy_price"]) / item["buy_price"] * 100
            total_value += item["qty"] * current_price
            rec, exp = analyze_pattern(df)
            st.subheader(f"{item['name']} — {rec}")
            st.write(f"Aktueller Preis: {current_price:.2f} | Kaufpreis: {item['buy_price']:.2f} | Gewinn: {gain:.2f}%")
            fig = draw_candlestick(df, item["name"])
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"Gesamtwert Portfolio: {total_value:.2f}")

# ------------------- Einstellungen -------------------
elif page == "Einstellungen":
    st.header("Theme Auswahl")
    theme = st.selectbox("Theme", ["Dunkel (Standard)", "Hell"])
    if st.button("Speichern"):
        st.success("Theme gespeichert (optisch in dieser Demo nicht aktiv).")
