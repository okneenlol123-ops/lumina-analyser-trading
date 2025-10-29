# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- App Config ---
st.set_page_config(page_title="ETF & Krypto Analyse (Demo)", page_icon="📊", layout="wide")

# --- Helper: Fake Data ---
def generate_fake_data(days=180, start_price=100):
    """Simuliert Kursdaten (Zufallsbewegung)"""
    dates = [(datetime.utcnow() - timedelta(days=i)).date() for i in range(days)][::-1]
    prices = [start_price]
    for _ in range(days - 1):
        change = np.random.normal(0, 0.01)  # ~1% tägliche Schwankung
        prices.append(prices[-1] * (1 + change))
    df = pd.DataFrame({"date": dates, "close": prices})
    return df

def calc_sma(df, window):
    return df["close"].rolling(window=window, min_periods=window).mean()

def get_recommendation(df):
    """SMA20 > SMA50 => Kaufen"""
    df["SMA20"] = calc_sma(df, 20)
    df["SMA50"] = calc_sma(df, 50)
    if len(df.dropna()) == 0:
        return False, "Nicht genug Daten"
    sma20, sma50 = df["SMA20"].iloc[-1], df["SMA50"].iloc[-1]
    if sma20 > sma50:
        return True, f"SMA20 ({sma20:.2f}) > SMA50 ({sma50:.2f})"
    else:
        return False, f"SMA20 ({sma20:.2f}) ≤ SMA50 ({sma50:.2f})"

def make_chart(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["date"], y=df["close"], mode="lines", name="Preis", line=dict(width=2)))
    if "SMA20" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA20"], mode="lines", name="SMA20", line=dict(dash="dash")))
    if "SMA50" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["SMA50"], mode="lines", name="SMA50", line=dict(dash="dot")))
    fig.update_layout(title=title, height=350, margin=dict(l=20, r=20, t=40, b=20))
    return fig

# --- UI ---
st.title("📊 ETF & Krypto Analyse (Offline Demo)")
st.write("Diese Version nutzt keine externen Finanzdaten. Kurse werden simuliert. Keine Anlageberatung!")

days = st.slider("Anzahl Tage (Demo-Daten)", 60, 365, 180)
etfs = ["Deutschland", "USA", "Europa", "Asien", "Welt"]
cryptos = ["Bitcoin", "Ethereum", "Solana", "BNB", "Cardano"]

tabs = st.tabs(["ETFs", "Kryptowährungen"])

# --- ETFs ---
with tabs[0]:
    st.header("ETFs")
    for name in etfs:
        df = generate_fake_data(days=days, start_price=100 + np.random.randint(20, 200))
        buy, reason = get_recommendation(df)
        fig = make_chart(df, f"{name} (Demo)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Letzter Preis", f"{df['close'].iloc[-1]:.2f} €")
            if buy:
                st.success("Empfehlung: Kaufen")
            else:
                st.error("Empfehlung: Nicht kaufen")
            st.caption(reason)
            st.button(f"Kaufen (Demo) – {name}", key=f"buy_etf_{name}")
            st.button(f"Nicht kaufen (Demo) – {name}", key=f"sell_etf_{name}")
        st.markdown("---")

# --- Kryptowährungen ---
with tabs[1]:
    st.header("Kryptowährungen")
    for name in cryptos:
        df = generate_fake_data(days=days, start_price=1000 + np.random.randint(1000, 50000))
        buy, reason = get_recommendation(df)
        fig = make_chart(df, f"{name} (Demo)")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.metric("Letzter Preis (USD)", f"{df['close'].iloc[-1]:.2f}")
            if buy:
                st.success("Empfehlung: Kaufen")
            else:
                st.error("Empfehlung: Nicht kaufen")
            st.caption(reason)
            st.button(f"Kaufen (Demo) – {name}", key=f"buy_crypto_{name}")
            st.button(f"Nicht kaufen (Demo) – {name}", key=f"sell_crypto_{name}")
        st.markdown("---")

st.markdown("### ℹ️ Hinweis")
st.info("Dies ist eine Offline-Demo ohne Live-Datenquellen. Daten sind zufällig generiert.")
