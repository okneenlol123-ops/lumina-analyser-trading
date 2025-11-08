import streamlit as st
import matplotlib.pyplot as plt

st.title("✅ Matplotlib funktioniert!")

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [10, 15, 7], color='lime')
ax.set_facecolor("#000000")
st.pyplot(fig)# main.py
# Offline Daytrading Simulator with matplotlib candlesticks + live update
# No external market APIs. Deterministic simulation per asset id.
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from datetime import datetime, timedelta
import time
import random
import math

# --------------------
# Config
# --------------------
st.set_page_config(page_title="Offline Daytrading (Matplotlib Candles)", layout="wide", page_icon="📈")
st.title("Offline Daytrading — Candlestick Simulator (matplotlib)")

# --------------------
# Utilities / Persistence
# --------------------
PORTFOLIO_FILE = "portfolio.json"

def save_portfolio(port):
    try:
        import json
        with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
            json.dump(port, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def load_portfolio():
    try:
        import json, os
        if os.path.exists(PORTFOLIO_FILE):
            with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

# --------------------
# Assets (10 ETFs, 20 Stocks, 10 Crypto)
# --------------------
ETFS = [
 "iShares DAX", "SP500 ETF", "MSCI World", "EuroStoxx", "Asia-Pacific",
 "Emerging Mkts", "Tech Leaders ETF", "Value ETF", "Dividend ETF", "Global SmallCap"
]
STOCKS = [
 "Apple","Microsoft","Amazon","Tesla","NVIDIA","Alphabet","Meta","Netflix",
 "Intel","AMD","SAP","Siemens","Allianz","Bayer","Volkswagen","Mercedes",
 "Shell","BP","DeutscheBank","SiemensEnergy"
]
CRYPTOS = [
 "Bitcoin","Ethereum","Solana","Cardano","Polkadot","Chainlink","Ripple","Litecoin","Dogecoin","Avalanche"
]

ALL_ASSETS = [{"id": f"ETF_{i}", "name": ETFS[i]} for i in range(len(ETFS))] + \
             [{"id": f"ST_{i}", "name": STOCKS[i]} for i in range(len(STOCKS))] + \
             [{"id": f"CR_{i}", "name": CRYPTOS[i]} for i in range(len(CRYPTOS))]

ASSET_MAP = {a["id"]: a for a in ALL_ASSETS}
NAME_TO_ID = {a["name"].lower(): a["id"] for a in ALL_ASSETS}

# --------------------
# Deterministic minute generator -> candles aggregation
# --------------------
def deterministic_seed(s: str) -> int:
    # stable seed across runs for same asset id
    return int(abs(hash(s)) % (2**31))

def generate_minute_series(seed_name: str, minutes: int, start_price: float):
    rnd = random.Random(deterministic_seed(seed_name))
    price = float(start_price)
    series = []
    now = datetime.utcnow()
    for i in range(minutes):
        # small random walk with mild drift and heteroskedasticity
        drift = (rnd.random() - 0.48) * 0.0006
        vol_factor = 1 + (rnd.random() - 0.5) * 0.01
        price = max(0.0001, price * (1 + drift) * vol_factor)
        ts = now - timedelta(minutes=minutes - i - 1)
        series.append({"ts": ts, "price": round(price, 6)})
    return series

def minutes_to_ohlc(min_series, candle_mins=5):
    if not min_series:
        return []
    candles = []
    chunk = []
    start_ts = None
    for i, p in enumerate(min_series, start=1):
        if start_ts is None:
            start_ts = p["ts"]
        chunk.append(p["price"])
        if i % candle_mins == 0:
            o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
            candles.append({"t": start_ts, "open": o, "high": h, "low": l, "close": c})
            chunk = []
            start_ts = None
    if chunk:
        # last partial candle
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
        candles.append({"t": start_ts or min_series[-1]["ts"], "open": o, "high": h, "low": l, "close": c})
    return candles

def build_candles(asset_id: str, timeframe_label: str, periods: int, start_price=100.0):
    tf_map = {"1m":1,"5m":5,"10m":10,"30m":30,"1h":60,"3h":180,"12h":720,"1d":1440}
    candle_mins = tf_map.get(timeframe_label, 5)
    minutes_needed = periods * candle_mins
    min_series = generate_minute_series(asset_id + "_" + timeframe_label, minutes_needed, start_price)
    candles = minutes_to_ohlc(min_series, candle_mins)
    # ensure at least 'periods' (pad if needed)
    if len(candles) < periods:
        pad = periods - len(candles)
        pad_c = [candles[0]] * pad if candles else [{"t": datetime.utcnow(), "open":start_price,"high":start_price,"low":start_price,"close":start_price}] * pad
        candles = pad_c + candles
    return candles[-periods:]

# --------------------
# Pattern recognition (more advanced)
# --------------------
def is_doji(c):
    body = abs(c["close"] - c["open"])
    total = c["high"] - c["low"]
    return total > 0 and (body / total) < 0.15

def is_hammer(c):
    body = abs(c["close"] - c["open"])
    lower = min(c["open"], c["close"]) - c["low"]
    return body > 0 and lower > 2 * body

def is_shooting_star(c):
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["open"], c["close"])
    return body > 0 and upper > 2 * body

def is_bullish_engulfing(prev, cur):
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])

def is_bearish_engulfing(prev, cur):
    return (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def is_three_white_soldiers(candles):
    if len(candles) < 3: return False
    a, b, c = candles[-3], candles[-2], candles[-1]
    return (a["close"] > a["open"]) and (b["close"] > b["open"]) and (c["close"] > c["open"]) and (b["close"] > a["close"]) and (c["close"] > b["close"])

def analyze_patterns(candles):
    # checks multiple patterns and builds a score
    score = 0
    reasons = []
    cur = candles[-1]
    prev = candles[-2] if len(candles) > 1 else None

    if is_doji(cur):
        reasons.append("Doji (Unentschlossen)")
    if prev and is_bullish_engulfing(prev, cur):
        score += 2
        reasons.append("Bullish Engulfing")
    if prev and is_bearish_engulfing(prev, cur):
        score -= 2
        reasons.append("Bearish Engulfing")
    if is_hammer(cur):
        score += 1
        reasons.append("Hammer (Bullish Wende)")
    if is_shooting_star(cur):
        score -= 1
        reasons.append("Shooting Star (Bearish Wende)")
    if is_three_white_soldiers(candles):
        score += 2
        reasons.append("Three White Soldiers (Stark Bullish)")

    # volatility and trend
    closes = [c["close"] for c in candles[-20:]]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))] if len(closes) > 1 else [0]
    vol = (np.std(returns) if len(returns) > 1 else 0)
    trend = sum(returns)
    if trend > 0.005:
        score += 1
        reasons.append("Positiver Momentum")
    elif trend < -0.005:
        score -= 1
        reasons.append("Negatives Momentum")
    # final recommendation
    if score >= 2:
        rec = "Kaufen"
    elif score <= -2:
        rec = "Verkaufen"
    else:
        rec = "Halten / Beobachten"
    # risk label from vol
    if vol < 0.001:
        risk = "Niedrig"
    elif vol < 0.01:
        risk = "Mittel"
    else:
        risk = "Hoch"
    return {"score": score, "reasons": reasons, "recommendation": rec, "risk": risk, "volatility": float(vol)}

# --------------------
# Plotting: Matplotlib candlestick (manual)
# --------------------
def plot_candles_mat(candles, title="Candlestick", sma_windows=(20,50), volume=False):
    # candles: list of dicts with t, open, high, low, close
    df = pd.DataFrame(candles)
    df['t_pos'] = range(len(df))
    fig, ax = plt.subplots(figsize=(10,4), facecolor="#0a0a0a")
    ax.set_facecolor("#0a0a0a")
    # plot wicks and bodies
    for idx, row in df.iterrows():
        o = row['open']; h = row['high']; l = row['low']; c = row['close']; x = row['t_pos']
        color = "#00cc66" if c >= o else "#ff4d66"
        # wick
        ax.add_line(Line2D([x, x], [l, h], linewidth=1, color="#aaaaaa", alpha=0.9))
        # body rectangle
        width = 0.6
        bottom = min(o,c); height = max(0.001, abs(c-o))
        rect = Rectangle((x - width/2, bottom), width, height, facecolor=color, edgecolor=color, linewidth=0.5, alpha=0.95)
        ax.add_patch(rect)
    # SMA overlays
    closes = df['close'].values
    xs = df['t_pos'].values
    for w, col in zip(sma_windows, ["#66ccff", "#ffcc66"]):
        if len(closes) >= w:
            sma = pd.Series(closes).rolling(w).mean().values
            ax.plot(xs, sma, color=col, linewidth=1.3, alpha=0.9)
    # formatting
    ax.set_title(title, color="#e6eef6", pad=8)
    ax.tick_params(colors="#9aa6b2")
    ax.set_xticks(xs[::max(1, len(xs)//8)])
    labels = [df['t'].iloc[i].strftime("%H:%M\n%d.%m") for i in ax.get_xticks().astype(int)]
    ax.set_xticklabels(labels, color="#9aa6b2")
    # hide spines
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(color="#111111")
    plt.tight_layout()
    return fig

# --------------------
# App state / Controls
# --------------------
if "live" not in st.session_state:
    st.session_state.live = False
if "series_cache" not in st.session_state:
    st.session_state.series_cache = {}
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_portfolio()

# --------------------
# UI Layout
# --------------------
left, right = st.columns([3,1])
with right:
    st.header("Controls")
    tf = st.selectbox("Timeframe", ["1m","5m","10m","30m","1h","3h","12h","1d"], index=1)
    periods = st.slider("Anzahl Kerzen", min_value=20, max_value=800, value=120, step=10)
    start_price = st.number_input("Startpreis (Simulation)", value=100.0, step=0.1)
    refresh_sec = st.number_input("Live Refresh (s)", min_value=1, max_value=10, value=2, step=1)
    search = st.text_input("Asset Suche (Name)")    
    if st.button("Start Live") and not st.session_state.live:
        st.session_state.live = True
    if st.button("Stop Live"):
        st.session_state.live = False
    if st.button("Nächste Kerze (manuell)"):
        st.session_state._step = st.session_state.get("_step", 0) + 1

    st.markdown("---")
    st.subheader("Portfolio")
    if st.session_state.portfolio:
        for p in st.session_state.portfolio:
            st.write(f"- {p['name']} • qty {p['qty']} • buy {p['buy_price']:.2f}")
    else:
        st.write("Portfolio leer")
    if st.button("Portfolio exportieren"):
        import json
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")
    st.markdown("---")
    st.write("Tipps:")
    st.write("- Starte Live um neue simulierte Kerzen zu sehen.")
    st.write("- Suche Asset mit Name (z.B. 'Apple').")

with left:
    st.header("Chart & Analyse")

    # choose asset
    if search:
        matches = [a for a in ALL_ASSETS if search.lower() in a["name"].lower()]
    else:
        matches = ALL_ASSETS
    # select box
    sel_name = st.selectbox("Wähle Asset", [a["name"] for a in matches], index=0)
    # find id
    sel_id = NAME_TO_ID.get(sel_name.lower(), None)
    if not sel_id:
        # fallback build id from name
        sel_id = f"AS_{hash(sel_name)%10000}"

    # caching series by asset + tf + periods + start_price
    cache_key = f"{sel_id}|{tf}|{periods}|{int(start_price)}"
    if cache_key not in st.session_state.series_cache:
        candles = build_candles(sel_id, tf, periods, start_price=start_price)
        st.session_state.series_cache[cache_key] = candles
    else:
        candles = st.session_state.series_cache[cache_key]

    # if live -> append new candle(s) every refresh
    def append_new_candle(candles_list):
        # generate one-minute series for base behavior and create new aggregated candle
        # We simulate new minute data and create one candle at timeframe resolution
        # Simpler: take last candle close and apply small random move
        rnd = random.Random(deterministic_seed(sel_id + str(time.time())))
        last = candles_list[-1]
        last_close = last["close"]
        # small move depending on timeframe (bigger timeframe -> bigger moves)
        timeframe_scale = {"1m":0.002,"5m":0.004,"10m":0.006,"30m":0.01,"1h":0.015,"3h":0.03,"12h":0.05,"1d":0.08}
        scale = timeframe_scale.get(tf, 0.005)
        change_pct = (rnd.random() - 0.5) * 2 * scale
        new_close = max(0.0001, last_close * (1 + change_pct))
        high = max(last_close, new_close) * (1 + rnd.random()*0.003)
        low = min(last_close, new_close) * (1 - rnd.random()*0.003)
        open_p = last_close
        new_candle = {"t": datetime.utcnow(), "open": open_p, "high": high, "low": low, "close": new_close}
        candles_list.append(new_candle)
        # keep length
        if len(candles_list) > periods:
            candles_list.pop(0)
        return candles_list

    # Live loop logic: if live True, generate new candle and rerun after sleep
    if st.session_state.live:
        # append new candle and trigger rerun after small wait
        candles = append_new_candle(candles)
        st.session_state.series_cache[cache_key] = candles
        # draw chart now and sleep then rerun
        fig = plot_candles_mat(candles, title=f"{sel_name} — {tf} ({periods} Kerzen)")
        st.pyplot(fig)
        analysis = analyze_patterns(candles)
        st.markdown(f"**Empfehlung:** **{analysis['recommendation']}**  •  Risiko: {analysis['risk']}  •  Gründe: {', '.join(analysis['reasons']) if analysis['reasons'] else '—'}")
        # flush UI then sleep then rerun
        time.sleep(max(0.5, float(refresh_sec:=refresh_sec)))
        st.experimental_rerun()
    else:
        # not live: show static chart and analysis
        fig = plot_candles_mat(candles, title=f"{sel_name} — {tf} ({periods} Kerzen)")
        st.pyplot(fig)
        analysis = analyze_patterns(candles)
        st.markdown("### Analyse")
        st.write(f"- **Empfehlung:** **{analysis['recommendation']}**")
        st.write(f"- **Risiko:** {analysis['risk']}")
        st.write(f"- **Volatilität (std returns):** {analysis['volatility']:.6f}")
        st.write(f"- **Erkannte Muster / Gründe:** {', '.join(analysis['reasons']) if analysis['reasons'] else 'Keine starken Muster'}")

    # quick add to portfolio
    st.markdown("---")
    st.subheader("Schnellaktion")
    col1, col2 = st.columns([2,1])
    with col1:
        qty = st.number_input("Menge hinzufügen", min_value=0.0, value=1.0, step=0.1)
        buy_price = float(candles[-1]["close"])
    with col2:
        if st.button("Zum Portfolio hinzufügen"):
            st.session_state.portfolio.append({"id": sel_id, "name": sel_name, "qty": qty, "buy_price": buy_price, "added_at": datetime.utcnow().isoformat()})
            save_portfolio(st.session_state.portfolio)
            st.success(f"{sel_name} hinzugefügt (qty {qty})")

# --------------------
# End
# --------------------
st.markdown("---")
st.markdown("<div style='color:#9aa6b2; font-size:12px;'>Offline-Simulation — deterministische Kerzen. Live-Modus generiert simulierte Kerzen regelmäßig.</div>", unsafe_allow_html=True)
