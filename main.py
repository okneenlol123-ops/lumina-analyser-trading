import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Finanzanalyse – Echte Daten ohne yfinance", page_icon="📈", layout="wide")

# --- Dark Mode CSS ---
st.markdown("""
    <style>
    body {background-color: #0E1117; color: #E1E8ED;}
    .stButton>button {background-color: #1F2937; color: #E1E8ED; border-radius: 6px;}
    .css-18e3th9, .css-1d391kg {background-color: #0E1117;}
    </style>
""", unsafe_allow_html=True)

st.title("📊 Finanzanalyse – Echte Daten (ohne yfinance)")
st.write("Live-Daten von CoinGecko (Krypto) und Stooq (Aktien & ETFs). Keine API-Keys nötig.")

# --- Funktionen ---
def fetch_stooq_data(symbol, days=365):
    """Lädt historische Daten (Aktien/ETFs) von stooq.com."""
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    try:
        df = pd.read_csv(url)
        df = df.rename(columns={"Date": "date", "Close": "close"})
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        cutoff = datetime.utcnow() - timedelta(days=days)
        df = df[df["date"] >= cutoff]
        return df
    except Exception:
        return pd.DataFrame(columns=["date", "close"])

def fetch_crypto_data(coin_id, days=365):
    """Holt historische Kryptodaten von CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    r = requests.get(url, params={"vs_currency":"usd","days":days})
    data = r.json().get("prices", [])
    df = pd.DataFrame(data, columns=["timestamp","close"])
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.groupby(df["date"].dt.date).last().reset_index()
    return df[["date","close"]]

def add_sma(df, window):
    df[f"SMA{window}"] = df["close"].rolling(window=window).mean()
    return df

def plot_chart(df, title, currency="€"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], name="Kurs", line=dict(color="#5bc0de", width=2)))
    if "SMA20" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA20"], name="SMA20", line=dict(color="#f0ad4e", dash="dash")))
    if "SMA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA50"], name="SMA50", line=dict(color="#d9534f", dash="dot")))
    fig.update_layout(
        title=title,
        paper_bgcolor="#0E1117",
        plot_bgcolor="#0E1117",
        font_color="#E1E8ED",
        xaxis_title="Datum",
        yaxis_title=f"Preis ({currency})",
    )
    return fig

def recommendation(df):
    if len(df) < 50:
        return False, "Nicht genug Daten"
    if df["close"].iloc[-1] > df["SMA50"].iloc[-1]:
        return True, "Preis > SMA50"
    else:
        return False, "Preis < SMA50"

# --- Zeitraum Optionen ---
intervals = {"1 Monat": 30, "3 Monate": 90, "6 Monate": 180, "1 Jahr": 365, "5 Jahre": 1825, "10 Jahre": 3650}

# --- Tabs ---
tabs = st.tabs(["ETFs", "Kryptowährungen", "Aktien"])

# ETFs
with tabs[0]:
    st.header("🌍 ETFs")
    etfs = {"Deutschland": "XDAX.DE", "USA": "SPY.US", "Europa": "VGK.US", "Asien": "AAXJ.US", "Welt": "VT.US"}
    for name, symbol in etfs.items():
        col1, col2 = st.columns([3, 1])
        with col2:
            interval = st.selectbox(f"Zeitraum für {name}", intervals.keys(), key=f"etf_{name}")
        df = fetch_stooq_data(symbol, intervals[interval])
        if df.empty:
            st.error(f"Keine Daten für {name}")
            continue
        df = add_sma(df, 20)
        df = add_sma(df, 50)
        buy, reason = recommendation(df)
        with col1:
            st.subheader(f"{name} ({symbol})")
            st.plotly_chart(plot_chart(df, f"{name} ETF"), use_container_width=True)
        with col2:
            st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} €**")
            st.success("Kaufen") if buy else st.error("Nicht kaufen")
            st.caption(reason)
        st.markdown("---")

# Kryptos
with tabs[1]:
    st.header("💰 Kryptowährungen")
    cryptos = {"Bitcoin": "bitcoin", "Ethereum": "ethereum", "Solana": "solana", "BNB": "binancecoin", "Cardano": "cardano"}
    for name, coin_id in cryptos.items():
        col1, col2 = st.columns([3, 1])
        with col2:
            interval = st.selectbox(f"Zeitraum für {name}", intervals.keys(), key=f"crypto_{name}")
        df = fetch_crypto_data(coin_id, intervals[interval])
        df = add_sma(df, 20)
        df = add_sma(df, 50)
        buy, reason = recommendation(df)
        with col1:
            st.subheader(name)
            st.plotly_chart(plot_chart(df, f"{name} (USD)", "$"), use_container_width=True)
        with col2:
            st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} $**")
            st.success("Kaufen") if buy else st.error("Nicht kaufen")
            st.caption(reason)
        st.markdown("---")

# Aktien
with tabs[2]:
    st.header("🏦 Aktien")
    stocks = {"Apple": "AAPL.US", "Tesla": "TSLA.US", "Microsoft": "MSFT.US", "Siemens": "SIE.DE", "Amazon": "AMZN.US"}
    for name, symbol in stocks.items():
        col1, col2 = st.columns([3, 1])
        with col2:
            interval = st.selectbox(f"Zeitraum für {name}", intervals.keys(), key=f"stock_{name}")
        df = fetch_stooq_data(symbol, intervals[interval])
        if df.empty:
            st.error(f"Keine Daten für {name}")
            continue
        df = add_sma(df, 20)
        df = add_sma(df, 50)
        buy, reason = recommendation(df)
        with col1:
            st.subheader(f"{name} ({symbol})")
            st.plotly_chart(plot_chart(df, name), use_container_width=True)
        with col2:
            st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} €**")
            st.success("Kaufen") if buy else st.error("Nicht kaufen")
            st.caption(reason)
        st.markdown("---")

st.info("✅ Live-Daten von CoinGecko & Stooq – keine API-Keys, kein yfinance erforderlich.")
