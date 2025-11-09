# main.py
# Offline Daytrading Simulator — Streamlit-only, SVG candlesticks, pattern analyzer
import streamlit as st
import random
import json
import os
from datetime import datetime, timedelta
from math import floor

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Offline Daytrading (SVG Candles)", layout="wide")
st.title("Offline Daytrading Simulator — SVG Candles (kein Plotly, kein Matplotlib)")

PORTFOLIO_FILE = "portfolio.json"
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ----------------------------
# Assets: 10 ETFs, 20 Stocks, 10 Crypto
# ----------------------------
ETFS = [
    "iShares DAX", "SP500 ETF", "MSCI World", "EuroStoxx", "Asia Pacific ETF",
    "Emerging Mkts ETF", "Tech Leaders ETF", "Value ETF", "Dividend ETF", "Global SmallCap"
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

NAME_TO_ID = {a["name"].lower(): a["id"] for a in ALL_ASSETS}
ID_TO_NAME = {a["id"]: a["name"] for a in ALL_ASSETS}

# ----------------------------
# Deterministic generator helpers
# ----------------------------
def deterministic_seed(s: str) -> int:
    # stable seed for same asset string
    return abs(hash(s)) % (2**31)

def generate_price_walk(seed_name: str, steps: int, start_price: float):
    rnd = random.Random(deterministic_seed(seed_name))
    price = float(start_price)
    series = []
    for _ in range(steps):
        # controlled random walk
        drift = (rnd.random() - 0.49) * 0.004
        shock = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + shock))
        series.append(round(price, 6))
    return series

def make_ohlc_from_prices(prices, candle_size=1):
    # If prices are per-minute, aggregate by candle_size to OHLC
    ohlc = []
    for i in range(0, len(prices), candle_size):
        chunk = prices[i:i+candle_size]
        if not chunk:
            continue
        o = chunk[0]
        c = chunk[-1]
        h = max(chunk)
        l = min(chunk)
        ohlc.append({"open": o, "high": h, "low": l, "close": c})
    return ohlc

# ----------------------------
# SVG Candlestick renderer
# ----------------------------
def render_candles_svg(ohlc_list, width_px=1000, height_px=360, margin=40, show_sma=(20,50)):
    # ohlc_list is list of dicts {open, high, low, close} (old->new)
    n = len(ohlc_list)
    if n == 0:
        return "<svg></svg>"
    # compute price range
    highs = [c["high"] for c in ohlc_list]
    lows = [c["low"] for c in ohlc_list]
    max_p = max(highs)
    min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    chart_w = width_px - margin * 2
    chart_h = height_px - margin * 2
    candle_w = max(3, chart_w / n * 0.7)
    spacing = chart_w / n

    def y_pos(price):
        return margin + chart_h - (price - min_p) / (max_p - min_p) * chart_h

    # SMA calculation
    closes = [c["close"] for c in ohlc_list]
    def sma(period):
        res = []
        for i in range(len(closes)):
            if i+1 < period:
                res.append(None)
            else:
                res.append(sum(closes[i+1-period:i+1]) / period)
        return res

    sma1 = sma(show_sma[0]) if show_sma and show_sma[0] else []
    sma2 = sma(show_sma[1]) if show_sma and show_sma[1] else []

    # Start SVG
    svg_parts = []
    svg_parts.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    # background
    svg_parts.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#0b0b0b" />')
    # grid lines and price labels
    for i in range(6):
        y = margin + i * (chart_h / 5)
        price_label = round(max_p - i * (max_p - min_p) / 5, 4)
        svg_parts.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#1a1a1a" stroke-width="1"/>')
        svg_parts.append(f'<text x="{5}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

    # draw candles
    for idx, c in enumerate(ohlc_list):
        x_center = margin + idx * spacing + spacing/2
        x_left = x_center - candle_w/2
        x_right = x_center + candle_w/2
        # positions
        y_open = y_pos(c["open"]); y_close = y_pos(c["close"])
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        # wick
        svg_parts.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        # body
        up = c["close"] >= c["open"]
        color = "#00cc66" if up else "#ff4d66"
        body_y = min(y_open, y_close)
        body_h = max(1, abs(y_open - y_close))
        svg_parts.append(f'<rect x="{x_left}" y="{body_y}" width="{candle_w}" height="{body_h}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')

    # draw SMAs as polylines
    def polyline_from(values, stroke, width=1.5):
        pts = []
        for i, v in enumerate(values):
            if v is None:
                pts.append(None)
                continue
            x = margin + i * spacing + spacing/2
            y = y_pos(v)
            pts.append(f"{x},{y}")
        # join segments skipping None
        segs = []
        cur = []
        for p in pts:
            if p is None:
                if cur:
                    segs.append(" ".join(cur))
                    cur = []
            else:
                cur.append(p)
        if cur:
            segs.append(" ".join(cur))
        lines = []
        for s in segs:
            lines.append(f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />')
        return "\n".join(lines)

    if sma1:
        svg_parts.append(polyline_from(sma1, "#66ccff", width=1.6))
    if sma2:
        svg_parts.append(polyline_from(sma2, "#ffcc66", width=1.6))

    # footer/time labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i * spacing + spacing/2
        t = (datetime.utcnow() - timedelta(minutes=(n - 1 - i))).strftime("%H:%M")
        svg_parts.append(f'<text x="{x-20}" y="{height_px-8}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg_parts.append('</svg>')
    return "\n".join(svg_parts)

# ----------------------------
# Pattern recognition functions
# ----------------------------
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

def analyze_candles_full(candles):
    # returns analysis dict
    reasons = []
    score = 0
    cur = candles[-1]
    prev = candles[-2] if len(candles) >= 2 else None

    if is_doji(cur):
        reasons.append("Doji (Unentschlossen)")
    if prev and is_bullish_engulfing(prev, cur):
        reasons.append("Bullish Engulfing")
        score += 2
    if prev and is_bearish_engulfing(prev, cur):
        reasons.append("Bearish Engulfing")
        score -= 2
    if is_hammer(cur):
        reasons.append("Hammer (Bullish)")
        score += 1
    if is_shooting_star(cur):
        reasons.append("Shooting Star (Bearish)")
        score -= 1
    if is_three_white_soldiers(candles):
        reasons.append("Three White Soldiers (Stark bullish)")
        score += 2

    # momentum & volatility
    closes = [c["close"] for c in candles[-30:]]
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))] if len(closes)>1 else [0.0]
    import statistics
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    trend = sum(returns)

    if trend > 0.01:
        reasons.append("Starker Aufwärtstrend")
        score += 1
    if trend < -0.01:
        reasons.append("Starker Abwärtstrend")
        score -= 1

    # recommendation mapping
    if score >= 2:
        rec = "Kaufen"
    elif score <= -2:
        rec = "Verkaufen"
    else:
        rec = "Halten / Beobachten"

    # risk label
    if vol < 0.001:
        risk = "Niedrig"
    elif vol < 0.01:
        risk = "Mittel"
    else:
        risk = "Hoch"

    return {"score": score, "reasons": reasons, "recommendation": rec, "volatility": vol, "risk": risk}

# ----------------------------
# UI state & controls
# ----------------------------
if "series_cache" not in st.session_state:
    st.session_state.series_cache = {}
if "portfolio" not in st.session_state:
    try:
        with open(PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            st.session_state.portfolio = json.load(f)
    except Exception:
        st.session_state.portfolio = []

# layout
col_main, col_side = st.columns([3, 1])

with col_side:
    st.header("Steuerung")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "10m", "30m", "1h", "3h", "12h", "1d"], index=1)
    periods = st.slider("Anzahl Kerzen", min_value=20, max_value=500, value=120, step=10)
    start_price = st.number_input("Startpreis (Sim)", min_value=0.01, value=100.0, step=0.1)
    show_sma1 = st.checkbox("SMA 20 anzeigen", value=True)
    show_sma2 = st.checkbox("SMA 50 anzeigen", value=False)
    search = st.text_input("Asset Suche (Name)", value="")
    refresh = st.button("Neue Simulation / Refresh")
    add_qty = st.number_input("Schnell: Menge zum Portfolio", min_value=0.0, value=1.0, step=0.1)
    if st.button("Portfolio exportieren"):
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")

with col_main:
    st.header("Chart & Analyse")
    # asset selection / filter
    filter_list = [a["name"] for a in ALL_ASSETS if search.strip().lower() in a["name"].lower()] if search else [a["name"] for a in ALL_ASSETS]
    sel = st.selectbox("Wähle Asset", filter_list, index=0)
    asset_id = NAME_TO_ID.get(sel.lower(), f"AS_{abs(hash(sel))%100000}")

    # cache key
    cache_key = f"{asset_id}|{timeframe}|{periods}|{int(start_price)}"
    if refresh or cache_key not in st.session_state.series_cache:
        # generate minute-level series then aggregate to timeframe
        tf_map = {"1m":1, "5m":5, "10m":10, "30m":30, "1h":60, "3h":180, "12h":720, "1d":1440}
        mins = periods * tf_map.get(timeframe, 5)
        prices = generate_price_walk(asset_id + "|" + timeframe, mins, start_price)
        ohlc = make_ohlc_from_prices(prices, candle_size=tf_map.get(timeframe,5))
        # ensure length and pad if needed
        if len(ohlc) < periods:
            pad = periods - len(ohlc)
            pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price}
            ohlc = [pad_item]*pad + ohlc
        st.session_state.series_cache[cache_key] = ohlc[-periods:]
    candles = st.session_state.series_cache[cache_key]

    # render svg
    show_sma = (20 if show_sma1 else 0, 50 if show_sma2 else 0)
    svg = render_candles_svg(candles, width_px=1000, height_px=420, margin=48, show_sma=show_sma)
    st.components.v1.html(svg, height=460)

    # analysis
    analysis = analyze_candles_full(candles)
    st.subheader("Analyse")
    st.write(f"**Empfehlung:** {analysis['recommendation']}")
    st.write(f"**Risiko:** {analysis['risk']}  •  **Volatilität:** {analysis['volatility']:.6f}")
    if analysis['reasons']:
        st.write("**Erkannte Muster / Gründe:**")
        for r in analysis['reasons']:
            st.write(f"- {r}")
    else:
        st.write("Keine starken Muster erkannt.")

    st.markdown("---")
    st.subheader("Schnellaktionen")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Zum Portfolio hinzufügen"):
            cur_price = candles[-1]["close"]
            st.session_state.portfolio.append({"id": asset_id, "name": sel, "qty": add_qty, "buy_price": cur_price, "added_at": datetime.utcnow().isoformat()})
            with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.portfolio, f, indent=2, ensure_ascii=False)
            st.success(f"{sel} ({add_qty}) zum Portfolio hinzugefügt.")
    with c2:
        if st.button("Details anzeigen"):
            last = candles[-1]
            prev = candles[-2] if len(candles) > 1 else None
            st.write("Letzte Kerze:", last)
            st.write("Vorherige Kerze:", prev)

    # portfolio summary
    st.markdown("---")
    st.subheader("Portfolio")
    if st.session_state.portfolio:
        total_value = 0.0
        for item in st.session_state.portfolio:
            # simulate current price by regenerating 1 period
            pid = item.get("id")
            key2 = f"{pid}|{timeframe}|{20}|{int(item.get('buy_price',100))}"
            tmp_prices = generate_price_walk(pid + "|p", 20 * 1, item.get("buy_price", 100))
            cur_price = tmp_prices[-1]
            total_value += cur_price * float(item.get("qty", 0.0))
            st.write(f"- {item['name']} • qty {item['qty']} • buy {item['buy_price']:.2f} • cur est {cur_price:.2f}")
        st.write(f"**Geschätzter Portfolio-Wert:** {total_value:.2f}")
    else:
        st.write("Portfolio leer. Füge Positionen hinzu.")

st.markdown("---")
st.caption("Offline-Simulator: Candles sind simuliert, deterministisch (gleicher Asset-Name → gleiche Serie). SVG-Render ohne externe Grafikbibliotheken.")
