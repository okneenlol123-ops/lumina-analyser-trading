# main.py
# Offline Daytrading Simulator — SVG candles + markers + RSI + dynamic Stop-Loss + Long/Short
import streamlit as st
import random
import json
import os
from datetime import datetime, timedelta
from math import floor

# ----------------------------
# Config
# ----------------------------
st.set_page_config(page_title="Offline Daytrading (Stop-Loss + Long/Short)", layout="wide")
st.title("Offline Daytrading — Stop-Loss, Long/Short & SVG Candles (kein Plotly)")

PORTFOLIO_FILE = "portfolio.json"
if not os.path.exists(PORTFOLIO_FILE):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

# ----------------------------
# Assets (10 ETFs, 20 Stocks, 10 Crypto)
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
        if not chunk:
            continue
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
        ohlc.append({"open": o, "high": h, "low": l, "close": c})
    return ohlc

# ----------------------------
# RSI helper
# ----------------------------
def compute_rsi(closes, period=14):
    if len(closes) < period + 1:
        return [None]*len(closes)
    deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [None] * (period)
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        rsi = 100 - (100 / (1 + rs))
        rsis.append(round(rsi, 2))
    return rsis

# ----------------------------
# Pattern detection (markers)
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

def detect_markers(candles):
    markers = []
    for i in range(1, len(candles)):
        cur = candles[i]; prev = candles[i-1]
        if is_bullish_engulfing(prev, cur) or is_hammer(cur):
            markers.append({"idx": i, "type": "buy", "reason": "Bullish/Hammer"})
        if is_bearish_engulfing(prev, cur) or is_shooting_star(cur):
            markers.append({"idx": i, "type": "sell", "reason": "Bearish/Shooting Star"})
    # three white soldiers detection
    for i in range(2, len(candles)):
        if (candles[i-2]["close"] > candles[i-2]["open"] and
            candles[i-1]["close"] > candles[i-1]["open"] and
            candles[i]["close"] > candles[i]["open"] and
            candles[i-1]["close"] > candles[i-2]["close"] and
            candles[i]["close"] > candles[i-1]["close"]):
            markers.append({"idx": i, "type": "buy", "reason": "Three White Soldiers"})
    # deduplicate markers by (idx,type)
    seen = set()
    uniq = []
    for m in markers:
        key = (m["idx"], m["type"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq

# ----------------------------
# Stop-Loss calculation (dynamic)
# ----------------------------
def calculate_dynamic_stop(entry_price, candles, position_type="long"):
    """
    Dynamic stop based on recent volatility:
    - compute std of returns over last N closes -> vol
    - recommended stop pct = clamp(vol*3, 0.01..0.10) (i.e. 1%..10%)
    - for long: stop = entry_price * (1 - stop_pct)
    - for short: stop = entry_price * (1 + stop_pct)
    Returns (stop_price, stop_pct)
    """
    closes = [c["close"] for c in candles[-30:]] if len(candles) >= 2 else [entry_price]
    returns = [(closes[i]-closes[i-1]) / closes[i-1] for i in range(1, len(closes))] if len(closes) > 1 else [0.0]
    import statistics
    vol = statistics.pstdev(returns) if len(returns) > 0 else 0.0
    # map vol to pct
    recommended_pct = max(0.01, min(0.10, vol * 3))
    if recommended_pct < 0.01:
        recommended_pct = 0.01
    if position_type == "long":
        stop_price = entry_price * (1 - recommended_pct)
    else:
        stop_price = entry_price * (1 + recommended_pct)
    return round(stop_price, 6), round(recommended_pct, 4), vol

# ----------------------------
# SVG renderer with markers + RSI + stop line
# ----------------------------
def render_candles_svg_with_markers(candles, markers, stop_line=None, width_px=1000, height_px=460, margin=48, show_sma=(20,50), rsi_vals=None):
    n = len(candles)
    if n == 0:
        return "<svg></svg>"
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    max_p = max(highs); min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    chart_h = height_px - margin * 3  # leave space for RSI
    chart_w = width_px - margin * 2
    candle_w = max(3, chart_w / n * 0.7)
    spacing = chart_w / n

    def y_pos(price):
        return margin + chart_h - (price - min_p) / (max_p - min_p) * chart_h

    closes = [c["close"] for c in candles]
    def sma(period):
        res = []
        for i in range(len(closes)):
            if i+1 < period: res.append(None)
            else: res.append(sum(closes[i+1-period:i+1]) / period)
        return res
    sma1 = sma(show_sma[0]) if show_sma and show_sma[0] else []
    sma2 = sma(show_sma[1]) if show_sma and show_sma[1] else []

    svg = []
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#0b0b0b" />')

    # grid & price labels
    for i in range(6):
        y = margin + i * (chart_h / 5)
        price_label = round(max_p - i * (max_p - min_p) / 5, 4)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#161616" stroke-width="1"/>')
        svg.append(f'<text x="{6}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

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

    # markers: draw arrows, shift them a bit to avoid overlap when many candles
    # shift scale increases with number of candles
    shift_px = max(8, min(28, int(200 / n)))  # fewer candles -> small shift; many candles -> slightly larger
    for m in markers:
        i = m["idx"]
        if i < 0 or i >= n: continue
        x_center = margin + i * spacing + spacing/2
        c = candles[i]
        y_high = y_pos(c["high"])
        y_low = y_pos(c["low"])
        # push markers higher or lower by shift_px
        if m["type"] == "buy":
            points = f"{x_center-6},{y_high - (12 + shift_px)} {x_center+6},{y_high - (12 + shift_px)} {x_center},{y_high - (2 + shift_px)}"
            svg.append(f'<polygon points="{points}" fill="#00ff88" opacity="0.98"><title>{m.get("reason","buy")}</title></polygon>')
        else:
            points = f"{x_center-6},{y_low + (12 + shift_px)} {x_center+6},{y_low + (12 + shift_px)} {x_center},{y_low + (2 + shift_px)}"
            svg.append(f'<polygon points="{points}" fill="#ff7788" opacity="0.98"><title>{m.get("reason","sell")}</title></polygon>')

    # SMAs
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
                    segs.append(" ".join(cur))
                    cur = []
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

    # stop_line: dict {'price':..., 'type': 'long'/'short', 'pct':...}
    if stop_line:
        sp = stop_line.get("price")
        if sp is not None:
            y_sp = y_pos(sp)
            color = "#ffcc00" if stop_line.get("type") == "long" else "#ff4444"
            svg.append(f'<line x1="{margin}" y1="{y_sp}" x2="{width_px-margin}" y2="{y_sp}" stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>')
            label = f"Stop ({'Long' if stop_line.get('type')=='long' else 'Short'}): {sp:.4f} ({stop_line.get('pct')*100:.2f}%)"
            svg.append(f'<rect x="{width_px-margin-220}" y="{y_sp-14}" width="220" height="18" fill="#111" opacity="0.8"/>')
            svg.append(f'<text x="{width_px-margin-216}" y="{y_sp}" font-size="12" fill="{color}">{label}</text>')

    # RSI area
    rsi_area_top = margin + chart_h + 10
    rsi_area_h = height_px - rsi_area_top - 10
    svg.append(f'<rect x="{margin}" y="{rsi_area_top}" width="{chart_w}" height="{rsi_area_h}" fill="#090909" stroke="#111" />')
    svg.append(f'<text x="{margin}" y="{rsi_area_top-2}" font-size="12" fill="#9aa6b2">RSI(14)</text>')
    if rsi_vals:
        pts = []
        for i, v in enumerate(rsi_vals):
            if v is None:
                pts.append(None); continue
            x = margin + i * spacing + spacing/2
            y = rsi_area_top + rsi_area_h - (v / 100.0) * rsi_area_h
            pts.append(f"{x},{y}")
        seg = []
        cur = []
        for p in pts:
            if p is None:
                if cur: seg.append(" ".join(cur)); cur=[]
            else:
                cur.append(p)
        if cur: seg.append(" ".join(cur))
        for s in seg:
            svg.append(f'<polyline points="{s}" fill="none" stroke="#ff66cc" stroke-width="1.2" />')
        y30 = rsi_area_top + rsi_area_h - 0.3 * rsi_area_h
        y70 = rsi_area_top + rsi_area_h - 0.7 * rsi_area_h
        svg.append(f'<line x1="{margin}" y1="{y30}" x2="{width_px-margin}" y2="{y30}" stroke="#333" stroke-dasharray="3,3" />')
        svg.append(f'<line x1="{margin}" y1="{y70}" x2="{width_px-margin}" y2="{y70}" stroke="#333" stroke-dasharray="3,3" />')

    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i * spacing + spacing/2
        t = (datetime.utcnow() - timedelta(minutes=(n - 1 - i))).strftime("%H:%M")
        svg.append(f'<text x="{x-20}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

# ----------------------------
# Full analysis (with stop suggestion explanation)
# ----------------------------
def analyze_candles_with_stop(candles, position_type="long", entry_price=None):
    analysis = analyze_candles_full(candles) if 'analyze_candles_full' in globals() else {"reasons":[], "recommendation":"Halten", "volatility":0.0, "risk":"Unbekannt"}
    if entry_price is None:
        entry_price = candles[-1]["close"]
    stop_price, stop_pct, vol = calculate_dynamic_stop(entry_price, candles, position_type=position_type)
    # explanation text
    explanation = []
    explanation.append(f"Stop-Loss (dynamisch): {stop_pct*100:.2f}% basierend auf Volatilität (σ ~ {vol:.6f}).")
    explanation.append("Empfehlung: Platzieren Sie Stop-Loss, um Verluste automatisch zu begrenzen.")
    explanation.append("Take-Profit: typischerweise 1.5–3× des Stop-Abstands (Risk:Reward).")
    if position_type == "short":
        explanation.append("Short-Position: Sie profitieren, wenn der Kurs fällt. Stop liegt oberhalb des Einstiegs.")
    else:
        explanation.append("Long-Position: Sie profitieren, wenn der Kurs steigt. Stop liegt unterhalb des Einstiegs.")
    return {"analysis": analysis, "stop_price": stop_price, "stop_pct": stop_pct, "explanation": explanation}

# ----------------------------
# Minimal helper analyze (re-using earlier logic)
# ----------------------------
def analyze_candles_full(candles):
    reasons = []; score = 0
    cur = candles[-1]; prev = candles[-2] if len(candles)>=2 else None
    if is_doji(cur): reasons.append("Doji")
    if prev and is_bullish_engulfing(prev, cur): reasons.append("Bullish Engulfing"); score += 2
    if prev and is_bearish_engulfing(prev, cur): reasons.append("Bearish Engulfing"); score -= 2
    if is_hammer(cur): reasons.append("Hammer"); score += 1
    if is_shooting_star(cur): reasons.append("Shooting Star"); score -= 1
    if is_three_white_soldiers(candles): reasons.append("Three White Soldiers"); score += 2
    closes = [c["close"] for c in candles[-30:]]
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))] if len(closes)>1 else [0.0]
    import statistics
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    trend = sum(returns)
    if trend > 0.01: reasons.append("Positives Momentum"); score += 1
    if trend < -0.01: reasons.append("Negatives Momentum"); score -= 1
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
    periods = st.slider("Anzahl Kerzen", min_value=20, max_value=500, value=120, step=10)
    start_price = st.number_input("Startpreis (Sim)", min_value=0.01, value=100.0, step=0.1)
    show_sma1 = st.checkbox("SMA 20 anzeigen", value=True)
    show_sma2 = st.checkbox("SMA 50 anzeigen", value=False)
    search = st.text_input("Asset Suche (Name)", value="")
    refresh = st.button("Neue Simulation / Refresh")
    add_qty = st.number_input("Schnell: Menge zum Portfolio", min_value=0.0, value=1.0, step=0.1)
    pos_type = st.radio("Positionstyp", ["long","short"], index=0)
    custom_stop_pct = st.slider("Manueller Stop-Loss (%) (optional, 0 = auto)", min_value=0.0, max_value=20.0, value=0.0, step=0.1)
    if st.button("Portfolio exportieren"):
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")

with col_main:
    st.header("Chart & Analyse")
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

    # detect markers & compute rsi
    markers = detect_markers(candles)
    closes = [c["close"] for c in candles]
    rsi = compute_rsi(closes, period=14)

    # compute suggested stop
    entry_price = candles[-1]["close"]
    stop_price_auto, stop_pct_auto, vol = calculate_dynamic_stop(entry_price, candles, position_type=pos_type)
    # if user provided manual stop pct > 0, override
    if custom_stop_pct > 0.0:
        stop_pct = custom_stop_pct / 100.0
        if pos_type == "long":
            stop_price = round(entry_price * (1 - stop_pct), 6)
        else:
            stop_price = round(entry_price * (1 + stop_pct), 6)
    else:
        stop_pct = stop_pct_auto
        stop_price = stop_price_auto

    stop_line = {"price": stop_price, "type": pos_type, "pct": stop_pct}

    svg = render_candles_svg_with_markers(candles, markers, stop_line=stop_line, width_px=1000, height_px=460, margin=48, show_sma=(20 if show_sma1 else 0, 50 if show_sma2 else 0), rsi_vals=rsi)
    st.components.v1.html(svg, height=520)

    # analysis block
    full_analysis = analyze_candles_full(candles)
    st.subheader("Analyse & Stop-Empfehlung")
    st.write(f"**Einstieg (letzter Schlusskurs):** {entry_price:.6f}")
    st.write(f"**Positionstyp:** {pos_type.upper()}  •  **Vorschlag Stop-Loss:** {stop_price:.6f} ({stop_pct*100:.2f}%)")
    st.write(f"**Volatilität (σ der Returns):** {vol:.6f}")
    st.write(f"**Empfehlung (Muster-Score):** {full_analysis['recommendation']}  •  Risiko: {full_analysis['risk']}")
    if full_analysis['reasons']:
        st.write("**Erkannte Muster / Gründe:**")
        for r in full_analysis['reasons']:
            st.write(f"- {r}")

    st.markdown("#### Was ist Stop-Loss / Take-Profit / Short-Position?")
    st.write("""
    **Stop-Loss:** Automatische Order, die eine Position bei Erreichen eines Preises schließt, um Verluste zu begrenzen.
    In dieser App wird der Stop dynamisch aus der Volatilität berechnet:  
    - Höhere Volatilität → weiter entfernter Stop (mehr Raum),  
    - Niedrige Volatilität → enger Stop.  
    **Take-Profit:** Zielpreis, bei dem du Gewinne sicherst (häufig 1.5–3× des Stop-Abstands).
    **Short-Position:** Du verkaufst zuerst und kaufst später zurück — Profit, wenn der Kurs fällt. Stop-Loss bei Short liegt **über** dem Einstiegsniveau.
    """)

    st.markdown("---")
    st.subheader("Schnellaktionen")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Zum Portfolio hinzufügen (Einstieg)"):
            st.session_state.portfolio.append({"id": asset_id, "name": sel, "qty": add_qty, "buy_price": entry_price, "position": pos_type, "stop_price": stop_price, "stop_pct": stop_pct, "added_at": datetime.utcnow().isoformat()})
            with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                json.dump(st.session_state.portfolio, f, indent=2, ensure_ascii=False)
            st.success(f"{sel} ({add_qty}) als {pos_type} hinzugefügt. Stop: {stop_price:.6f}")
    with c2:
        if st.button("Details anzeigen"):
            st.write("Letzte Kerze:", candles[-1])
            st.write("Stop-Loss Vorschlag:", stop_price, f"({stop_pct*100:.2f}%)")

    st.markdown("---")
    st.subheader("Portfolio")
    if st.session_state.portfolio:
        total_value = 0.0
        for item in st.session_state.portfolio:
            pid = item.get("id")
            tmp_prices = generate_price_walk(pid + "|p", 20, item.get("buy_price", 100))
            cur_price = tmp_prices[-1]
            if item.get("position","long") == "long":
                total_value += cur_price * float(item.get("qty", 0.0))
            else:
                # short P&L estimate: (entry - cur) * qty
                total_value += (item.get("buy_price", cur_price) - cur_price) * float(item.get("qty", 0.0))
            st.write(f"- {item['name']} • pos {item.get('position','long')} • qty {item['qty']} • entry {item['buy_price']:.6f} • cur est {cur_price:.6f} • stop {item.get('stop_price'):.6f}")
        st.write(f"**Geschätzter Portfolio-Wert (Longs = Marktwert, Shorts = unreal. P&L):** {total_value:.2f}")
    else:
        st.write("Portfolio leer. Füge Positionen hinzu.")

st.markdown("---")
st.caption("Offline-Simulator — deterministisch (gleicher Asset-Name → gleiche Serie). SVG-Render ohne externe Grafikbibliotheken.")
