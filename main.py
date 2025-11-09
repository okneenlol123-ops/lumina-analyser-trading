import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# -------------------------------
# 🔧 App Setup
# -------------------------------
st.set_page_config(page_title="Live Daytrading Analyzer", layout="wide")
st.title("📊 Live Daytrading Analyzer – Aktien, ETFs & Krypto")

# 🔑 API-Key sicher laden
try:
    API_KEY = st.secrets["api_keys"]["ALPHAVANTAGE"]
except Exception:
    st.error("❌ Kein API Key gefunden! Bitte füge ihn unter Settings → Secrets hinzu.")
    st.stop()

# -------------------------------
# ⚙️ Auswahloptionen
# -------------------------------
st.sidebar.header("⚙️ Einstellungen")
symbol = st.sidebar.selectbox(
    "📈 Wähle Asset",
    ["AAPL", "TSLA", "NVDA", "MSFT", "QQQ", "SPY", "BTCUSD", "ETHUSD"]
)
interval = st.sidebar.selectbox(
    "⏱️ Intervall",
    ["1min", "5min", "15min", "30min", "60min"]
)

# -------------------------------
# 📡 Datenabruf
# -------------------------------
st.info(f"Lade Live-Daten für **{symbol}** ({interval}) ...")

url = (
    f"https://www.alphavantage.co/query?"
    f"function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&apikey={API_KEY}"
)

response = requests.get(url)
data = response.json()

if not any("Time Series" in k for k in data.keys()):
    st.error("⚠️ Keine Daten erhalten – überprüfe API Key oder Limit (5 Calls/min bei Free Account).")
    st.stop()

key = [k for k in data.keys() if "Time Series" in k][0]
df = pd.DataFrame(data[key]).T
df.columns = ["Open", "High", "Low", "Close", "Volume"]
df = df.astype(float)
df = df.iloc[::-1]  # älteste zuerst

# -------------------------------
# 🕯️ Candlestick Chart
# -------------------------------
fig = go.Figure(
    data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color="lime",
            decreasing_line_color="red",
        )
    ]
)

fig.update_layout(
    template="plotly_dark",
    title=f"📊 {symbol} – {interval} Chart (Alpha Vantage)",
    xaxis_rangeslider_visible=False,
    height=600,
)

st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# 📈 Analyse
# -------------------------------
st.subheader("📊 Analyse & Empfehlung")

latest = df.iloc[-1]
previous = df.iloc[-2]
change = ((latest["Close"] - previous["Close"]) / previous["Close"]) * 100

col1, col2, col3 = st.columns(3)
col1.metric("Letzter Kurs", f"${latest['Close']:.2f}")
col2.metric("Änderung", f"{change:.2f}%")
col3.metric("Volumen", f"{latest['Volume']:.0f}")

if change > 0.3:
    st.success("📈 Der Kurs steigt – *Empfehlung: Kaufen* ✅")
elif change < -0.3:
    st.error("📉 Der Kurs fällt – *Empfehlung: Nicht kaufen* ❌")
else:
    st.warning("⚖️ Seitwärtsbewegung – *Abwarten empfohlen* ⚠️")

# -------------------------------
# ⚠️ Risikoabschätzung
# -------------------------------
st.subheader("⚠️ Risikoanalyse")
risk_level = abs(change)
if risk_level > 2:
    st.error("🚨 Hohes Risiko – starke Volatilität erkannt!")
elif risk_level > 1:
    st.warning("⚠️ Mittleres Risiko – moderate Schwankungen.")
else:
    st.info("✅ Niedriges Risiko – stabile Bewegung.")

st.caption("Datenquelle: Alpha Vantage (live) – Intervall max. 5 Calls/Minute bei Free API-Key.")
