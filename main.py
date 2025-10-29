# main.py
import streamlit as st
import random
from datetime import datetime, timedelta

st.set_page_config(page_title="ETF & Krypto Analyse (Minimal)", page_icon="📊", layout="wide")

st.title("📊 ETF & Krypto Analyse – Offline Version")
st.write("Diese Version funktioniert ohne externe Module. Alle Daten sind simuliert. Keine Anlageberatung.")

# --- Helper functions ---
def generate_fake_data(days=180, start_price=100):
    """Erstellt einfache simulierte Preisdaten als Liste von (Datum, Preis)."""
    today = datetime.utcnow().date()
    prices = []
    price = start_price
    for i in range(days):
        price = round(price * (1 + random.uniform(-0.02, 0.02)), 2)
        date = today - timedelta(days=days - i)
        prices.append((date, price))
    return prices

def simple_recommendation(prices):
    """Einfache Regel: Wenn letzter Preis > Durchschnitt → Kaufen"""
    if not prices:
        return False, "Keine Daten"
    last_price = prices[-1][1]
    avg_price = sum(p[1] for p in prices) / len(prices)
    if last_price > avg_price:
        return True, f"Letzter Preis ({last_price:.2f}) über Durchschnitt ({avg_price:.2f})"
    else:
        return False, f"Letzter Preis ({last_price:.2f}) unter Durchschnitt ({avg_price:.2f})"

def show_text_chart(prices, title):
    """Zeigt einen einfachen ASCII-Chart (Textgrafik)."""
    st.markdown(f"#### {title}")
    if not prices:
        st.write("Keine Daten verfügbar.")
        return
    # Normalisiere Werte für Text-Diagramm
    values = [p[1] for p in prices]
    min_val, max_val = min(values), max(values)
    scale = 20 / (max_val - min_val) if max_val != min_val else 1
    chart_lines = ""
    for date, val in prices[-30:]:  # letzte 30 Tage
        height = int((val - min_val) * scale)
        chart_lines += f"{date} | " + "█" * height + f" ({val})\n"
    st.text(chart_lines)

# --- Benutzeroptionen ---
days = st.slider("Zeitraum (Tage)", 30, 365, 180)
st.write("Es werden zufällige Kursbewegungen angezeigt (Demo).")

tabs = st.tabs(["ETFs", "Kryptowährungen"])

# --- ETFs ---
with tabs[0]:
    etfs = ["Deutschland", "USA", "Europa", "Asien", "Welt"]
    st.header("ETFs")
    for name in etfs:
        data = generate_fake_data(days, start_price=random.randint(80, 200))
        buy, reason = simple_recommendation(data)
        show_text_chart(data, f"{name} (Demo)")
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
    cryptos = ["Bitcoin", "Ethereum", "Solana", "BNB", "Cardano"]
    st.header("Kryptowährungen")
    for name in cryptos:
        data = generate_fake_data(days, start_price=random.randint(1000, 60000))
        buy, reason = simple_recommendation(data)
        show_text_chart(data, f"{name} (Demo)")
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
st.info("Diese Version verwendet keinerlei externe Pakete oder APIs. Alles läuft offline mit simulierten Daten.")
