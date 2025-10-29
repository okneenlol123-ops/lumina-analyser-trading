import streamlit as st
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="ETF & Krypto Analyse – Graphen", page_icon="📈", layout="wide")

st.title("📈 ETF & Krypto Analyse – Offline Graphen")
st.write("Alle Daten simuliert. Keine API, keine externen Module. Keine Anlageberatung.")

# --- Helper Functions ---
def generate_fake_data(days=180, start_price=100):
    """Simuliert Preisbewegungen mit SMA20 und SMA50."""
    today = datetime.utcnow().date()
    prices = []
    for i in range(days):
        date = today - timedelta(days=days - i - 1)
        if i == 0:
            price = start_price
        else:
            price = round(prices[-1][1] * (1 + random.uniform(-0.02, 0.02)), 2)
        prices.append([date, price])
    # SMA20 und SMA50 simuliert
    sma20 = []
    sma50 = []
    for i in range(days):
        sma20.append(round(sum([p[1] for p in prices[max(0, i-19):i+1]])/min(i+1,20),2))
        sma50.append(round(sum([p[1] for p in prices[max(0, i-49):i+1]])/min(i+1,50),2))
    for i in range(days):
        prices[i].extend([sma20[i], sma50[i]])
    return prices  # [date, price, sma20, sma50]

def simple_recommendation(prices):
    """Letzter Preis über SMA50 -> Kaufen"""
    if not prices:
        return False, "Keine Daten"
    last_price = prices[-1][1]
    last_sma50 = prices[-1][3]
    if last_price > last_sma50:
        return True, f"Letzter Preis ({last_price:.2f}) über SMA50 ({last_sma50:.2f})"
    else:
        return False, f"Letzter Preis ({last_price:.2f}) unter SMA50 ({last_sma50:.2f})"

def prepare_line_chart_data(prices):
    """Erzeugt dicts für Streamlit line_chart"""
    data_dict = {
        "Preis": {str(p[0]): p[1] for p in prices},
        "SMA20": {str(p[0]): p[2] for p in prices},
        "SMA50": {str(p[0]): p[3] for p in prices}
    }
    return data_dict

def filter_prices_by_interval(prices, interval):
    """Filtert Daten nach gewähltem Zeitraum"""
    today = datetime.utcnow().date()
    if interval == "1 Monat":
        cutoff = today - timedelta(days=30)
    elif interval == "3 Monate":
        cutoff = today - timedelta(days=90)
    elif interval == "6 Monate":
        cutoff = today - timedelta(days=180)
    elif interval == "1 Jahr":
        cutoff = today - timedelta(days=365)
    elif interval == "5 Jahre":
        cutoff = today - timedelta(days=365*5)
    elif interval == "10 Jahre":
        cutoff = today - timedelta(days=365*10)
    else:
        cutoff = today - timedelta(days=180)
    return [p for p in prices if p[0] >= cutoff]

# --- UI ---
interval_options = ["1 Monat","3 Monate","6 Monate","1 Jahr","5 Jahre","10 Jahre"]

tabs = st.tabs(["ETFs","Kryptowährungen"])

# --- ETFs ---
with tabs[0]:
    st.header("ETFs")
    etfs = ["Deutschland","USA","Europa","Asien","Welt"]
    for name in etfs:
        col1, col2 = st.columns([3,1])
        with col2:
            interval = st.selectbox(f"Zeitraum für {name}", interval_options, key=f"etf_interval_{name}")
        data = generate_fake_data(365*10, start_price=random.randint(80,200))  # max 10 Jahre
        filtered = filter_prices_by_interval(data, interval)
        chart_data = prepare_line_chart_data(filtered)
        buy, reason = simple_recommendation(filtered)
        with col1:
            st.subheader(f"{name} (Demo)")
            st.line_chart(chart_data)
        with col2:
            st.write(f"Letzter Preis: **{filtered[-1][1]:.2f} €**")
            if buy:
                st.success("Empfehlung: Kaufen")
            else:
                st.error("Empfehlung: Nicht kaufen")
            st.caption(reason)
            st.button(f"Kaufen (Demo) – {name}", key=f"etf_buy_{name}")
            st.button(f"Nicht kaufen (Demo) – {name}", key=f"etf_sell_{name}")
        st.markdown("---")

# --- Kryptowährungen ---
with tabs[1]:
    st.header("Kryptowährungen")
    cryptos = ["Bitcoin","Ethereum","Solana","BNB","Cardano"]
    for name in cryptos:
        col1, col2 = st.columns([3,1])
        with col2:
            interval = st.selectbox(f"Zeitraum für {name}", interval_options, key=f"crypto_interval_{name}")
        data = generate_fake_data(365*10, start_price=random.randint(1000,60000))
        filtered = filter_prices_by_interval(data, interval)
        chart_data = prepare_line_chart_data(filtered)
        buy, reason = simple_recommendation(filtered)
        with col1:
            st.subheader(f"{name} (Demo)")
            st.line_chart(chart_data)
        with col2:
            st.write(f"Letzter Preis: **{filtered[-1][1]:.2f} USD**")
            if buy:
                st.success("Empfehlung: Kaufen")
            else:
                st.error("Empfehlung: Nicht kaufen")
            st.caption(reason)
            st.button(f"Kaufen (Demo) – {name}", key=f"crypto_buy_{name}")
            st.button(f"Nicht kaufen (Demo) – {name}", key=f"crypto_sell_{name}")
        st.markdown("---")

st.markdown("### ℹ️ Hinweis")
st.info("Offline-Demo. Graphen zeigen simulierte Kurse und SMA-Linien. Zeitraum für jeden Graphen wählbar.")
