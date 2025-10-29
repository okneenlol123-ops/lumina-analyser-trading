# main.py
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="ETF & Krypto Analyse", layout="wide", page_icon="📈")

# ---------- Helpers ----------
def sma_series(series: pd.Series, period: int) -> pd.Series:
    """Compute simple moving average with same length (NaN for first periods)."""
    return series.rolling(window=period, min_periods=period).mean()

def fetch_etf_history(ticker: str, days: int = 180) -> pd.DataFrame:
    """Fetch daily close prices for ETF using yfinance. Returns DataFrame with index=date and 'close' column."""
    end = datetime.utcnow().date()
    start = end - timedelta(days=days*2)  # extra buffer to account for non-trading days
    try:
        df = yf.download(ticker, start=start.isoformat(), end=(end + timedelta(days=1)).isoformat(), progress=False, threads=False)
        if df.empty:
            raise ValueError("No data from yfinance")
        df = df.rename(columns={"Adj Close": "adj_close", "Close": "close"})
        # prefer adjusted if available
        close_col = "adj_close" if "adj_close" in df.columns else "close"
        res = pd.DataFrame({"close": df[close_col]}).dropna().reset_index()
        res['date'] = res['Date'].dt.date.astype(str)
        res = res[['date', 'close']]
        return res.tail(days)
    except Exception as e:
        # fallback: demo series
        return generate_demo_series(days=days, start=100 + np.random.rand() * 50)

def fetch_crypto_history_coingecko(symbol: str, days: int = 180) -> pd.DataFrame:
    """
    symbol: coingecko id like 'bitcoin', 'ethereum' (not BTC)
    days: number of days to fetch (max usually large)
    Returns DataFrame with date (YYYY-MM-DD) and close
    """
    # CoinGecko API: /coins/{id}/market_chart?vs_currency=usd&days={days}
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{symbol}/market_chart"
        resp = requests.get(url, params={"vs_currency": "usd", "days": days})
        resp.raise_for_status()
        data = resp.json()
        prices = data.get("prices", [])  # list of [timestamp, price]
        if not prices:
            raise ValueError("No price data")
        df = pd.DataFrame(prices, columns=["ts", "close"])
        df['date'] = pd.to_datetime(df['ts'], unit='ms').dt.date.astype(str)
        # keep one value per date: take last of each day
        df = df.groupby('date', as_index=False).last()[['date', 'close']]
        return df.tail(days)
    except Exception as e:
        # fallback demo
        return generate_demo_series(days=days, start=20000 if symbol=='bitcoin' else 1000)

def generate_demo_series(days=180, start=100):
    arr = []
    price = float(start)
    for i in range(days):
        change = (np.random.rand() - 0.48) * max(0.6, price * 0.01)
        price = max(0.1, price + change)
        d = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        arr.append({"date": d, "close": round(float(price), 4)})
    return pd.DataFrame(arr)

def compute_recommendation(df: pd.DataFrame):
    """Compute SMA20 and SMA50, return dict with fields: buy(bool), reason(str), last_price"""
    if df is None or df.empty or len(df) < 50:
        return {"buy": False, "reason": "Nicht genug Daten für SMA (benötigt 50 Tage)", "last_price": None, "sma20": None, "sma50": None}
    series = pd.Series(df['close'].values)
    sma20 = sma_series(series, 20)
    sma50 = sma_series(series, 50)
    last_sma20 = float(sma20.dropna().iloc[-1]) if sma20.dropna().size > 0 else None
    last_sma50 = float(sma50.dropna().iloc[-1]) if sma50.dropna().size > 0 else None
    last_price = float(series.iloc[-1])
    if last_sma20 is None or last_sma50 is None:
        return {"buy": False, "reason": "Nicht genug Daten für SMA20/50", "last_price": last_price, "sma20": last_sma20, "sma50": last_sma50}
    if last_sma20 > last_sma50:
        return {"buy": True, "reason": f"SMA20 ({last_sma20:.2f}) > SMA50 ({last_sma50:.2f})", "last_price": last_price, "sma20": last_sma20, "sma50": last_sma50}
    else:
        return {"buy": False, "reason": f"SMA20 ({last_sma20:.2f}) ≤ SMA50 ({last_sma50:.2f})", "last_price": last_price, "sma20": last_sma20, "sma50": last_sma50}

def plot_price_with_sma(df: pd.DataFrame, title: str):
    """Return a Plotly Figure with close, SMA20, SMA50"""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title)
        return fig

    ddf = df.copy()
    ddf['close'] = ddf['close'].astype(float)
    ddf['sma20'] = ddf['close'].rolling(20, min_periods=20).mean()
    ddf['sma50'] = ddf['close'].rolling(50, min_periods=50).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ddf['date'], y=ddf['close'], mode='lines', name='Close', line=dict(width=2)))
    fig.add_trace(go.Scatter(x=ddf['date'], y=ddf['sma20'], mode='lines', name='SMA20', line=dict(width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=ddf['date'], y=ddf['sma50'], mode='lines', name='SMA50', line=dict(width=1, dash='dot')))
    fig.update_layout(title=title, margin=dict(l=20,r=20,t=30,b=20), height=360, xaxis=dict(showgrid=False), yaxis=dict(showgrid=True))
    return fig

# ---------- App UI ----------
st.title("📈 ETF & Krypto Analyse (dealerless)")
st.markdown("Zwei Kategorien: **ETFs** und **Kryptowährungen**. Empfehlung = SMA20 vs SMA50 (Demo, keine Anlageberatung).")

col1, col2 = st.columns([3,1])

with col2:
    st.header("Einstellungen")
    days = st.number_input("Tage (Chartbereich)", min_value=30, max_value=720, value=180, step=10)
    refresh = st.button("Aktualisieren")
    st.markdown("**Hinweis:** ETFs per `yfinance`, Kryptowährungen per CoinGecko API. Keine Orders/Trades ausgeführt.")

# Default tickers (du kannst einfach editieren)
etf_defaults = {
    "Deutschland": "EWG",   # iShares MSCI Germany ETF (US ticker als Beispiel)
    "USA": "SPY",
    "Europa": "VGK",
    "Asien": "AAXJ",
    "Welt": "VT"
}
crypto_defaults = {
    "Bitcoin": "bitcoin",   # CoinGecko ids
    "Ethereum": "ethereum",
    "BNB": "binancecoin",
    "Cardano": "cardano",
    "Solana": "solana"
}

with col1:
    st.subheader("Kategorien wählen")
    tabs = st.tabs(["ETFs", "Kryptowährungen"])

    # -------- ETFs tab --------
    with tabs[0]:
        st.markdown("**ETFs** (wähle oder editiere Ticker). Tip: für XETRA-Ticker YFinance verwendet '.DE' (z.B. `EXS1.DE`) — nicht alle Börsen sind verfügbar.")
        etf_cols = st.columns([1,1,1,1,1])
        etf_inputs = []
        keys = list(etf_defaults.keys())
        for i, name in enumerate(keys):
            with etf_cols[i]:
                t = st.text_input(f"{name}", value=etf_defaults[name], key=f"etf_{i}")
                etf_inputs.append((name, t.strip()))

        # Load & render ETFs
        for idx, (region, ticker) in enumerate(etf_inputs):
            st.markdown("---")
            st.markdown(f"**{region} — {ticker}**")
            placeholder = st.empty()
            try:
                df = fetch_etf_history(ticker, days=days)
            except Exception as e:
                df = generate_demo_series(days=days, start=100 + idx*10)
            rec = compute_recommendation(df)
            fig = plot_price_with_sma(df, f"{region} · {ticker}")
            left, right = st.columns([3,1])
            with left:
                st.plotly_chart(fig, use_container_width=True)
            with right:
                if rec["last_price"] is not None:
                    st.metric(label="Letzter Preis (USD)", value=f"{rec['last_price']:.2f}")
                else:
                    st.write("Letzter Preis: —")
                if rec["buy"]:
                    st.success("Empfehlung: Kaufen")
                else:
                    st.error("Empfehlung: Nicht kaufen")
                st.caption(rec["reason"])
                st.button(f"Kaufen (Demo) {ticker}", key=f"buy_etf_{idx}")
                st.button(f"Nicht kaufen (Demo) {ticker}", key=f"sell_etf_{idx}")

    # -------- Crypto tab --------
    with tabs[1]:
        st.markdown("**Kryptowährungen** (CoinGecko IDs). Beispiele: `bitcoin`, `ethereum`, `binancecoin`, `cardano`, `solana`")
        crypto_cols = st.columns([1,1,1,1,1])
        crypto_inputs = []
        keys_c = list(crypto_defaults.keys())
        for i, name in enumerate(keys_c):
            with crypto_cols[i]:
                t = st.text_input(f"{name}", value=crypto_defaults[name], key=f"crypto_{i}")
                crypto_inputs.append((name, t.strip()))

        # Load & render cryptos
        for idx, (label, cg_id) in enumerate(crypto_inputs):
            st.markdown("---")
            st.markdown(f"**{label} — {cg_id} (USD)**")
            try:
                dfc = fetch_crypto_history_coingecko(cg_id, days=days)
            except Exception as e:
                dfc = generate_demo_series(days=days, start=30000 if cg_id=='bitcoin' else 1000)
            rec_c = compute_recommendation(dfc)
            figc = plot_price_with_sma(dfc, f"{label} · USD")
            left, right = st.columns([3,1])
            with left:
                st.plotly_chart(figc, use_container_width=True)
            with right:
                if rec_c["last_price"] is not None:
                    st.metric(label="Letzter Preis (USD)", value=f"{rec_c['last_price']:.2f}")
                else:
                    st.write("Letzter Preis: —")
                if rec_c["buy"]:
                    st.success("Empfehlung: Kaufen")
                else:
                    st.error("Empfehlung: Nicht kaufen")
                st.caption(rec_c["reason"])
                st.button(f"Kaufen (Demo) {cg_id}", key=f"buy_crypto_{idx}")
                st.button(f"Nicht kaufen (Demo) {cg_id}", key=f"sell_crypto_{idx}")

# Footer / Disclaimer
st.markdown("---")
st.markdown("""
**Wichtig:** Diese App zeigt nur historische Preise und eine einfache technische Regel (SMA20 vs SMA50) als Demo.  
**Keine Anlageberatung**. Es werden keine Orders an Broker oder Exchanges gesendet — alles ist rein informativ/demohaft.
""")
