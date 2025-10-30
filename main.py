import streamlit as st
import yfinance as yf
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- Page config & Dark Mode CSS ---
st.set_page_config(page_title="Finanzanalyse – Dark Mode", page_icon="📊", layout="wide")

# Dark mode styling
st.markdown(
    """
    <style>
    body {
        background-color: #0E1117;
        color: #E1E8ED;
    }
    .css-18e3th9 {background-color: #0E1117;}
    .css-1d391kg {background-color: #0E1117;}
    .stButton>button {background-color: #1F2937; color: #E1E8ED;}
    </style>
    """, unsafe_allow_html=True
)

st.title("📈 Finanzanalyse – Dark Mode")
st.write("Echte Daten für ETFs, Kryptowährungen & Aktien. Auswahl Zeitraum + technische Indikatoren (SMA20/50).")

# --- Helper functions ---
def fetch_stock_data(ticker, period="1y"):
    """Holt historische Daten für Aktien/ETFs via yfinance."""
    df = yf.Ticker(ticker).history(period=period)
    df = df.reset_index().rename(columns={"Date":"date", "Close":"close"})
    df['date'] = df['date'].dt.date
    return df[['date','close']]

def fetch_crypto_data(coingecko_id, days=365):
    """Holt historische Krypto-Preise von CoinGecko."""
    url = f"https://api.coingecko.com/api/v3/coins/{coingecko_id}/market_chart"
    r = requests.get(url, params={"vs_currency":"usd","days":days})
    data = r.json().get("prices", [])
    df = pd.DataFrame(data, columns=["ts","close"])
    df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
    df = df.groupby('date', as_index=False).last()
    return df[['date','close']]

def add_sma(df, window):
    df[f"SMA{window}"] = df['close'].rolling(window=window).mean()
    return df

def recommendation(df):
    """Empfehlung: Preis > SMA50 => Kaufen."""
    if df.empty or 'SMA50' not in df.columns:
        return False, "Keine Daten"
    last = df.iloc[-1]
    price = last['close']
    sma50 = last['SMA50']
    if price > sma50:
        return True, f"Preis ({price:.2f}) > SMA50 ({sma50:.2f})"
    else:
        return False, f"Preis ({price:.2f}) ≤ SMA50 ({sma50:.2f})"

def plot_data(df, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['close'], mode='lines', name='Close', line=dict(color='#5bc0de', width=2)))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA20'], mode='lines', name='SMA20', line=dict(color='#f0ad4e', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['SMA50'], mode='lines', name='SMA50', line=dict(color='#d9534f', width=1, dash='dot')))
    fig.update_layout(
        plot_bgcolor='#0E1117',
        paper_bgcolor='#0E1117',
        font_color='#E1E8ED',
        title=title,
        xaxis_title='Datum',
        yaxis_title='Preis',
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# --- UI & Auswahl ---
interval_options = {
    "1 Monat": "30d",
    "3 Monate": "90d",
    "6 Monate": "180d",
    "1 Jahr": "1y",
    "5 Jahre": "5y",
    "10 Jahre": "10y"
}

tabs = st.tabs(["ETFs","Kryptowährungen","Aktien"])

with tabs[0]:
    st.header("ETFs")
    etfs = {"Deutschland":"EWG", "USA":"SPY", "Europa":"VGK", "Asien":"AAXJ", "Welt":"VT"}
    for name, ticker in etfs.items():
        interval_label = st.selectbox(f"Zeitraum für {name}", list(interval_options.keys()), key=f"etf_{name}")
        period = interval_options[interval_label]
        df = fetch_stock_data(ticker, period=period)
        df = add_sma(df, 20)
        df = add_sma(df, 50)
        buy, reason = recommendation(df)
        st.subheader(f"{name} – {ticker}")
        st.plotly_chart(plot_data(df, f"{name} ({ticker})"), use_container_width=True)
        st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} $**")
        if buy:
            st.success("Empfehlung: Kaufen")
        else:
            st.error("Empfehlung: Nicht kaufen")
        st.caption(reason)
        st.button(f"Kaufen (Demo) – {ticker}", key=f"buy_etf_{ticker}")
        st.button(f"Nicht kaufen (Demo) – {ticker}", key=f"sell_etf_{ticker}")
        st.markdown("---")

with tabs[1]:
    st.header("Kryptowährungen")
    cryptos = {"Bitcoin":"bitcoin", "Ethereum":"ethereum", "Solana":"solana", "BNB":"binancecoin", "Cardano":"cardano"}
    for name, cid in cryptos.items():
        interval_label = st.selectbox(f"Zeitraum für {name}", list(interval_options.keys()), key=f"crypto_{name}")
        days = int(interval_options[interval_label].rstrip('d').rstrip('y')) if interval_options[interval_label].endswith('d') else 365 * int(interval_options[interval_label].rstrip('y'))
        df = fetch_crypto_data(cid, days=days)
        df = add_sma(df,20)
        df = add_sma(df,50)
        buy, reason = recommendation(df)
        st.subheader(f"{name}")
        st.plotly_chart(plot_data(df, f"{name} (USD)"), use_container_width=True)
        st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} USD**")
        if buy:
            st.success("Empfehlung: Kaufen")
        else:
            st.error("Empfehlung: Nicht kaufen")
        st.caption(reason)
        st.button(f"Kaufen (Demo) – {name}", key=f"buy_crypto_{name}")
        st.button(f"Nicht kaufen (Demo) – {name}", key=f"sell_crypto_{name}")
        st.markdown("---")

with tabs[2]:
    st.header("Aktien")
    stocks = {"Apple":"AAPL", "Tesla":"TSLA", "Microsoft":"MSFT", "Siemens":"SIE.DE", "Amazon":"AMZN"}
    for name, ticker in stocks.items():
        interval_label = st.selectbox(f"Zeitraum für {name}", list(interval_options.keys()), key=f"stock_{name}")
        period = interval_options[interval_label]
        df = fetch_stock_data(ticker, period=period)
        df = add_sma(df,20)
        df = add_sma(df,50)
        buy, reason = recommendation(df)
        st.subheader(f"{name} – {ticker}")
        st.plotly_chart(plot_data(df, f"{name} ({ticker})"), use_container_width=True)
        st.write(f"Letzter Preis: **{df['close'].iloc[-1]:.2f} $**")
        if buy:
            st.success("Empfehlung: Kaufen")
        else:
            st.error("Empfehlung: Nicht kaufen")
        st.caption(reason)
        st.button(f"Kaufen (Demo) – {ticker}", key=f"buy_stock_{ticker}")
        st.button(f"Nicht kaufen (Demo) – {ticker}", key=f"sell_stock_{ticker}")
        st.markdown("---")

st.markdown("### ℹ️ Hinweis")
st.info("Dies ist eine Demo-App mit echten Daten – technische Indikator-Berechnung (SMA) ist vereinfacht. Keine persönliche Anlageberatung.")
