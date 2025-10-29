import streamlit as st
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="ETF & Krypto Graphen", page_icon="📈", layout="wide")

st.title("📈 ETF & Krypto Analyse – Offline Graphen")
st.write("Alle Daten sind simuliert. Keine externe Module, keine API. Keine Anlageberatung.")

# --- Helper functions ---
def generate_fake_data(days=180, start_price=100):
    """Simuliert Preisbewegungen"""
    today = datetime.utcnow().date()
    prices = []
    price = start_price
    for i in range(days):
        price = round(price * (1 + random.uniform(-0.02, 0.02)), 2)
        date = today - timedelta(days=days - i)
        prices.append((date, price))
    return prices

def simple_recommendation(prices):
    """Letzter Preis über Durchschnitt -> Kaufen"""
    if not prices:
        return False, "Keine Daten"
    last_price = prices[-1][1]
    avg_price = sum(p[1] for p in prices) / len(prices)
    if last_price > avg_price:
        return True, f"Letzter Preis ({last_price:.2f}) über Durchschnitt ({avg_price:.2f})"
    else:
        return False, f"Letzter Preis ({last_price:.2f}) unter Durchschnitt ({avg_price:.2f})"

def prepare_line_data(prices):
    """Wandelt Liste von (Datum, Preis) in dict für st.line_chart"""
    # letzte 30 Tage für Übersicht
    last_prices = prices[-30:]
    return {str(date): price for date, price in last_prices}

# --- UI ---
days = st.slider("Zeitraum (Tage)", 30, 365, 180)

tabs = st.tabs(["ETFs", "Kryptowährungen"])

# --- ETFs ---
with tabs[0]:
    st.header("ETFs")
    etfs = ["Deutschland", "USA", "Europa", "Asien", "Welt"]
    for name in etfs:
        data = generate_fake_data(days, start_price=random.randint(80, 200))
        buy, reason = simple_recommendation(data)
        chart_data = prepare_line_data(data)
        st.subheader(f"{name} (Demo)")
        st.line_chart(chart_data)
        st.write(f"Letzter Preis: **{data[-1][1]:.2f} €**")
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
    cryptos = ["Bitcoin", "Ethereum", "Solana", "BNB", "Cardano"]
    for name in cryptos:
        data = generate_fake_data(days, start_price=random.randint(1000, 60000))
        buy, reason = simple_recommendation(data)
        chart_data = prepare_line_data(data)
        st.subheader(f"{name} (Demo)")
        st.line_chart(chart_data)
        st.write(f"Letzter Preis: **{data[-1][1]:.2f} USD**")
        if buy:
            st.success("Empfehlung: Kaufen")
        else:
            st.error("Empfehlung: Nicht kaufen")
        st.caption(reason)
        st.button(f"Kaufen (Demo) – {name}", key=f"crypto_buy_{name}")
        st.button(f"Nicht kaufen (Demo) – {name}", key=f"crypto_sell_{name}")
        st.markdown("---")

st.markdown("### ℹ️ Hinweis")
st.info("Offline-Demo. Graphen zeigen die simulierten Kurse der letzten 30 Tage.")
