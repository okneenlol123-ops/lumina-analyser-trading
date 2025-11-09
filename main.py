# main.py
# Offline Daytrading Simulator + SMC Overlays (SVG) — Streamlit only
import streamlit as st
import random
import json
import os
from datetime import datetime, timedelta
from math import floor

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="SMC Daytrading Simulator (SVG)", layout="wide")
st.title("Offline Daytrading Simulator — SMC Overlays (SVG)")

PORTFOLIO_FILE = "portfolio.json"
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ----------------------------
# Assets
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
    return abs(hash(s)) % (2**31)

def generate_price_walk(seed_name: str, steps: int, start_price: float):
    rnd = random.Random(deterministic_seed(seed_name))
    price = float(start_price)
    series = []
    for _ in range(steps):
        drift = (rnd.random() - 0.49) * 0.004
        shock = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + shock))
        series.append(round(price, 6))
    return series

def make_ohlc_from_prices(prices, candle_size=1):
    ohlc = []
    for i in range(0, len(prices), candle_size):
        chunk = prices[i:i+candle_size]
        if not chunk: continue
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
        ohlc.append({"open": o, "high": h, "low": l, "close": c})
    return ohlc

# ----------------------------
# SMC detection helpers (heuristic)
# ----------------------------
def find_swings(candles, lookback=3):
    """Find local swing highs and lows. Returns lists of (idx, price)."""
    highs = []
    lows = []
    n = len(candles)
    for i in range(lookback, n-lookback):
        window_high = max(c["high"] for c in candles[i-lookback:i+lookback+1])
        window_low  = min(c["low"] for c in candles[i-lookback:i+lookback+1])
        if candles[i]["high"] == window_high:
            highs.append((i, candles[i]["high"]))
        if candles[i]["low"] == window_low:
            lows.append((i, candles[i]["low"]))
    return highs, lows

def detect_bos_ch(chandles):
    """Detect simple Break of Structure (BOS) or Change of Character (CHOCH).
       Heuristic: last swing high broken by close above -> BOS up; similarly for down.
       Returns dict with 'type' (up/down/none) and 'level' and 'index'."""
    candles = chandles
    highs, lows = find_swings(candles, lookback=3)
    res = {"type": None, "level": None, "index": None, "broken_from": None}
    if not highs or not lows:
        return res
    # find last two swing highs and lows
    last_high_idx, last_high = highs[-1] if highs else (None, None)
    last_low_idx, last_low   = lows[-1] if lows else (None, None)
    # check if price recently broke above previous swing high -> BOS up
    last_close = candles[-1]["close"]
    if last_high is not None and last_close > last_high:
        res.update({"type": "BOS_up", "level": last_high, "index": last_high_idx, "broken_from": "high"})
        return res
    # check if price broke below previous swing low -> BOS down
    if last_low is not None and last_close < last_low:
        res.update({"type": "BOS_down", "level": last_low, "index": last_low_idx, "broken_from": "low"})
        return res
    return res

def find_liquidity_zone(candles, lookback=20):
    """Find recent liquidity zone: cluster of lows (for buy-side) or highs (for sell-side).
       Returns a POI zone as (low, high) and index range."""
    closes = [c["close"] for c in candles]
    lows = [c["low"] for c in candles[-lookback:]]
    highs = [c["high"] for c in candles[-lookback:]]
    # heuristic: find min low area as buy-side liquidity, and max high area as sell-side liquidity
    min_low = min(lows) if lows else None
    max_high = max(highs) if highs else None
    # define small zone +- a bit
    if min_low is not None:
        low_zone = (min_low * 0.997, min_low * 1.003)
    else:
        low_zone = None
    if max_high is not None:
        high_zone = (max_high * 0.997, max_high * 1.003)
    else:
        high_zone = None
    return {"buy_zone": low_zone, "sell_zone": high_zone, "min_low": min_low, "max_high": max_high}

# ----------------------------
# SVG renderer with SMC overlays
# ----------------------------
def render_svg_with_smc(candles, width_px=1000, height_px=480, margin=48, show_sma=(20,50),
                        show_bos=True, show_poi=True, show_entry_stop=True, show_bsl_ssl=True):
    n = len(candles)
    if n == 0:
        return "<svg></svg>"
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    max_p = max(highs); min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    chart_h = height_px - margin * 3  # leave space below
    chart_w = width_px - margin * 2
    candle_w = max(3, chart_w / n * 0.7)
    spacing = chart_w / n

    def y_pos(price):
        return margin + chart_h - (price - min_p) / (max_p - min_p) * chart_h

    # SMA
    closes = [c["close"] for c in candles]
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

    # SMC detection
    bos = detect_bos_ch(candles) if show_bos else {"type": None}
    poi = find_liquidity_zone(candles, lookback=30) if show_poi else {}
    markers = []
    if show_bsl_ssl:
        # BSL = recent lows cluster (buy-side liquidity zones)
        if poi.get("min_low"):
            # find indices within small tolerance
            tol = (poi["min_low"] * 0.003)
            for i, c in enumerate(candles[-60:], start=max(0, len(candles)-60)):
                if abs(c["low"] - poi["min_low"]) <= tol:
                    markers.append({"idx": i, "type": "BSL", "price": c["low"]})
        if poi.get("max_high"):
            tol = (poi["max_high"] * 0.003)
            for i, c in enumerate(candles[-60:], start=max(0, len(candles)-60)):
                if abs(c["high"] - poi["max_high"]) <= tol:
                    markers.append({"idx": i, "type": "SSL", "price": c["high"]})

    # Entry/Stop heuristics: create a POI entry box around min_low and stop below it
    entry_box = None
    stop_box = None
    if show_entry_stop and poi.get("min_low"):
        # entry area: slightly above liquidity zone
        z_low, z_high = poi["buy_zone"]
        entry_box = {"y1": z_low * 0.999, "y2": z_high * 1.006}
        stop_box = {"y1": z_low * 0.98, "y2": z_low * 0.995}

    # start svg
    svg = []
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#0b0b0b" />')

    # grid & price labels
    for i in range(6):
        y = margin + i * (chart_h / 5)
        price_label = round(max_p - i * (max_p - min_p) / 5, 4)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#161616" stroke-width="1"/>')
        svg.append(f'<text x="6" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

    # draw POI buy zone rectangle (green) and sell zone (if any)
    if show_poi and poi.get("buy_zone"):
        y_top = y_pos(entry_box["y2"]) if entry_box else y_pos(poi["buy_zone"][1])
        y_bot = y_pos(entry_box["y1"]) if entry_box else y_pos(poi["buy_zone"][0])
        svg.append(f'<rect x="{margin+2}" y="{y_top}" width="{chart_w-4}" height="{max(2, y_bot - y_top)}" fill="#1b4630" opacity="0.25" />')
        svg.append(f'<text x="{margin+6}" y="{y_top+14}" font-size="12" fill="#9ee6c9">POI / LQ Sweep</text>')
    if show_poi and poi.get("sell_zone"):
        y_top_s = y_pos(poi["sell_zone"][1])
        y_bot_s = y_pos(poi["sell_zone"][0])
        svg.append(f'<rect x="{margin+2}" y="{y_top_s}" width="{chart_w-4}" height="{max(2, y_bot_s - y_top_s)}" fill="#4a1630" opacity="0.12" />')
        svg.append(f'<text x="{margin+6}" y="{y_top_s+14}" font-size="12" fill="#f3b1c1">Sell Liquidity</text>')

    # draw candles
    for idx, c in enumerate(candles):
        x_center = margin + idx * spacing + spacing/2
        x_left = x_center - candle_w/2
        body_y_open = y_pos(c["open"]); body_y_close = y_pos(c["close"])
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        svg.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        up = c["close"] >= c["open"]
        color = "#00cc66" if up else "#ff4d66"
        by = min(body_y_open, body_y_close)
        bh = max(1, abs(body_y_open - body_y_close))
        svg.append(f'<rect x="{x_left}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1" />')

    # draw BOS line if detected
    if show_bos and bos.get("type") and bos.get("level") is not None:
        level = bos["level"]
        y = y_pos(level)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#FFD700" stroke-width="2" stroke-dasharray="8,6"/>')
        label = "BOS↑" if bos["type"] == "BOS_up" else "BOS↓"
        svg.append(f'<text x="{width_px-margin-90}" y="{y-6}" font-size="13" fill="#FFD700">{label}</text>')

    # draw markers (BSL / SSL)
    for m in markers:
        i = m["idx"]
        x_center = margin + i * spacing + spacing/2
        if m["type"] == "BSL":
            y = y_pos(m["price"])
            svg.append(f'<circle cx="{x_center}" cy="{y}" r="5" fill="#00ff99" opacity="0.9"/>')
            svg.append(f'<text x="{x_center+8}" y="{y+4}" font-size="11" fill="#9ee6c9">BSL</text>')
        else:
            y = y_pos(m["price"])
            svg.append(f'<rect x="{x_center-5}" y="{y-5}" width="10" height="10" fill="#ff9999" opacity="0.9"/>')
            svg.append(f'<text x="{x_center+8}" y="{y+4}" font-size="11" fill="#f7b1b9">SSL</text>')

    # draw entry & stop boxes
    if show_entry_stop and entry_box and stop_box:
        y_entry_top = y_pos(entry_box["y2"]); y_entry_bot = y_pos(entry_box["y1"])
        y_stop_top = y_pos(stop_box["y2"]); y_stop_bot = y_pos(stop_box["y1"])
        svg.append(f'<rect x="{margin+chart_w*0.6}" y="{y_entry_top}" width="{chart_w*0.35}" height="{max(2, y_entry_bot - y_entry_top)}" fill="#234b7d" opacity="0.18" />')
        svg.append(f'<text x="{margin+chart_w*0.62}" y="{y_entry_top+16}" font-size="12" fill="#99ccff">Entry Zone</text>')
        svg.append(f'<rect x="{margin+chart_w*0.6}" y="{y_stop_top}" width="{chart_w*0.35}" height="{max(2, y_stop_bot - y_stop_top)}" fill="#7a2f2f" opacity="0.18" />')
        svg.append(f'<text x="{margin+chart_w*0.62}" y="{y_stop_top+16}" font-size="12" fill="#f6a6a6">Stop Zone</text>')

    # SMAs as polylines
    def polyline(values, stroke, width=1.5):
        pts = []
        for i, v in enumerate(values):
            if v is None:
                pts.append(None)
            else:
                x = margin + i * spacing + spacing/2
                y = y_pos(v)
                pts.append(f"{x},{y}")
        segs = []
        cur = []
        for p in pts:
            if p is None:
                if cur:
                    segs.append(" ".join(cur)); cur=[]
            else:
                cur.append(p)
        if cur: segs.append(" ".join(cur))
        out = []
        for s in segs:
            out.append(f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />')
        return "\n".join(out)

    if sma1:
        svg.append(polyline(sma1, "#66ccff", width=1.6))
    if sma2:
        svg.append(polyline(sma2, "#ffcc66", width=1.6))

    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i * spacing + spacing/2
        t = (datetime.utcnow() - timedelta(minutes=(n - 1 - i))).strftime("%H:%M")
        svg.append(f'<text x="{x-20}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

# ----------------------------
# Full analysis used for text + reasons
# ----------------------------
def analyze_candles_full(candles):
    reasons = []; score = 0
    cur = candles[-1]; prev = candles[-2] if len(candles)>=2 else None
    # basic patterns reuse
    def is_doji(c):
        body = abs(c["close"] - c["open"]); total = c["high"] - c["low"]
        return total>0 and (body/total) < 0.15
    def is_hammer(c):
        body = abs(c["close"] - c["open"]); lower = min(c["open"], c["close"])-c["low"]; return body>0 and lower>2*body
    def is_shooting_star(c):
        body = abs(c["close"] - c["open"]); upper = c["high"]-max(c["open"], c["close"]); return body>0 and upper>2*body
    def is_bullish_engulfing(prev, cur):
        return (cur["close"]>cur["open"]) and (prev["close"]<prev["open"]) and (cur["open"]<prev["close"]) and (cur["close"]>prev["open"])
    def is_bearish_engulfing(prev, cur):
        return (cur["close"]<cur["open"]) and (prev["close"]>prev["open"]) and (cur["open"]>prev["close"]) and (cur["close"]<prev["open"])

    if is_doji(cur): reasons.append("Doji (Unentschlossen)")
    if prev and is_bullish_engulfing(prev, cur): reasons.append("Bullish Engulfing"); score+=2
    if prev and is_bearish_engulfing(prev, cur): reasons.append("Bearish Engulfing"); score-=2
    if is_hammer(cur): reasons.append("Hammer"); score+=1
    if is_shooting_star(cur): reasons.append("Shooting Star"); score-=1

    closes = [c["close"] for c in candles[-30:]]
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1, len(closes))] if len(closes)>1 else [0.0]
    import statistics
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    trend = sum(returns)
    if trend > 0.01: reasons.append("Positives Momentum"); score+=1
    if trend < -0.01: reasons.append("Negatives Momentum"); score-=1

    rec = "Kaufen" if score>=2 else ("Verkaufen" if score<=-2 else "Halten / Beobachten")
    if vol < 0.001: risk="Niedrig"
    elif vol < 0.01: risk="Mittel"
    else: risk="Hoch"
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

col_main, col_side = st.columns([3,1])
with col_side:
    st.header("Steuerung")
    timeframe = st.selectbox("Timeframe", ["1m","5m","10m","30m","1h","3h","12h","1d"], index=1)
    periods = st.slider("Anzahl Kerzen", min_value=30, max_value=400, value=120, step=10)
    start_price = st.number_input("Startpreis (Sim)", min_value=0.01, value=100.0, step=0.1)
    show_sma1 = st.checkbox("SMA 20 anzeigen", value=True)
    show_sma2 = st.checkbox("SMA 50 anzeigen", value=False)
    search = st.text_input("Asset Suche (Name)", value="")
    refresh = st.button("Neue Simulation / Refresh")
    show_bos = st.checkbox("BOS Linien anzeigen", value=True)
    show_poi = st.checkbox("POI / LQ Zone anzeigen", value=True)
    show_entry_stop = st.checkbox("Entry/Stop Zonen anzeigen", value=True)
    show_bsl_ssl = st.checkbox("BSL/SSL Markierungen anzeigen", value=True)
    add_qty = st.number_input("Schnell: Menge zum Portfolio", min_value=0.0, value=1.0, step=0.1)
    if st.button("Portfolio exportieren"):
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")

with col_main:
    st.header("Chart & SMC Analyse")
    filter_list = [a["name"] for a in ALL_ASSETS if search.strip().lower() in a["name"].lower()] if search else [a["name"] for a in ALL_ASSETS]
    sel = st.selectbox("Wähle Asset", filter_list, index=0)
    asset_id = NAME_TO_ID.get(sel.lower(), f"AS_{abs(hash(sel))%100000}")
    cache_key = f"{asset_id}|{timeframe}|{periods}|{int(start_price)}"
    if refresh or cache_key not in st.session_state.series_cache:
        tf_map = {"1m":1,"5m":5,"10m":10,"30m":30,"1h":60,"3h":180,"12h":720,"1d":1440}
        minutes = periods * tf_map.get(timeframe, 5)
        prices = generate_price_walk(asset_id + "|" + timeframe, minutes, start_price)
        ohlc = make_ohlc_from_prices(prices, candle_size=tf_map.get(timeframe,5))
        if len(ohlc) < periods:
            pad = periods - len(ohlc)
            pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price}
            ohlc = [pad_item]*pad + ohlc
        st.session_state.series_cache[cache_key] = ohlc[-periods:]
    candles = st.session_state.series_cache[cache_key]

    markers = []  # SBL/SSL markers will be included in render if enabled
    svg = render_svg_with_smc(candles, width_px=1000, height_px=480, margin=48,
                              show_sma=(20 if show_sma1 else 0, 50 if show_sma2 else 0),
                              show_bos=show_bos, show_poi=show_poi, show_entry_stop=show_entry_stop, show_bsl_ssl=show_bsl_ssl)
    st.components.v1.html(svg, height=520)

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
            last = candles[-1]; prev = candles[-2] if len(candles)>1 else None
            st.write("Letzte Kerze:", last); st.write("Vorherige Kerze:", prev)

    st.markdown("---")
    st.subheader("Portfolio")
    if st.session_state.portfolio:
        total_value = 0.0
        for item in st.session_state.portfolio:
            pid = item.get("id")
            tmp_prices = generate_price_walk(pid + "|p", 20, item.get("buy_price", 100))
            cur_price = tmp_prices[-1]
            total_value += cur_price * float(item.get("qty", 0.0))
            st.write(f"- {item['name']} • qty {item['qty']} • buy {item['buy_price']:.2f} • cur est {cur_price:.2f}")
        st.write(f"**Geschätzter Portfolio-Wert:** {total_value:.2f}")
    else:
        st.write("Portfolio leer. Füge Positionen hinzu.")

st.markdown("---")
st.caption("Offline-Simulator: deterministisch (gleicher Asset-Name → gleiche Serie). SVG-Render mit SMC-Overlays (BOS, POI/LQ, BSL/SSL, Entry/Stop).")
