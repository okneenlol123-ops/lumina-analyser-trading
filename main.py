# main.py
# Analyzer Deluxe — Alpha Vantage integration + Offline fallback
# Streamlit-only UI, SVG candlesticks, pattern recognition, dynamic stop-loss, long/short, trade simulator
# WARNING: API key is embedded per user request. For security consider moving to st.secrets.

import streamlit as st
import json, os, math, random, time, urllib.request, urllib.parse
from datetime import datetime, timedelta
from math import floor

# -----------------------
# Configuration / API Key
# -----------------------
st.set_page_config(page_title="Analyzer Deluxe — Live & Offline", layout="wide")
APP_TITLE = "Analyzer Deluxe — Live (Alpha Vantage) + Offline Fallback"
st.title(APP_TITLE)

# === INSERTED API KEY (per user request) ===
ALPHAVANTAGE_KEY = "22XGVO0TQ1UV167C"
# If you prefer secrets, comment above and use:
# try:
#     ALPHAVANTAGE_KEY = st.secrets["api_keys"]["ALPHAVANTAGE"]
# except Exception:
#     ALPHAVANTAGE_KEY = None

# -----------------------
# Files / Persistence
# -----------------------
PORTFOLIO_FILE = "portfolio.json"
HISTORY_FILE = "history.json"

def ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)

ensure_file(PORTFOLIO_FILE, [])
ensure_file(HISTORY_FILE, [])

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -----------------------
# Asset lists
# -----------------------
ETFS = [
 "iShares DAX", "SP500 ETF", "MSCI World", "EuroStoxx", "Asia Pacific ETF",
 "Emerging Mkts ETF", "Tech Leaders ETF", "Value ETF", "Dividend ETF", "Global SmallCap"
]
STOCKS = [
 "AAPL","MSFT","AMZN","TSLA","NVDA","GOOGL","META","NFLX",
 "INTC","AMD","SAP","SIE.DE","ALV.DE","BAYN.DE","VOW3.DE","DAI.DE",
 "RDSA","BP","DBK.DE","SIE.DE"  # some German tickers as examples
]
CRYPTOS = [
 "BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOT-USD","LINK-USD","XRP-USD","LTC-USD","DOGE-USD","AVAX-USD"
]

ALL_ASSETS = [{"id": f"ETF_{i}", "name": ETFS[i], "symbol": None} for i in range(len(ETFS))] + \
             [{"id": f"ST_{s}", "name": s, "symbol": s} for s in STOCKS] + \
             [{"id": f"CR_{c}", "name": c, "symbol": c} for c in CRYPTOS]

NAME_TO_SYMBOL = {a["name"].lower(): a["symbol"] for a in ALL_ASSETS if a["symbol"]}

# -----------------------
# Utility helpers
# -----------------------
def deterministic_seed(s: str) -> int:
    return abs(hash(s)) % (2**31)

def human_ts():
    return datetime.utcnow().isoformat() + "Z"

# -----------------------
# Alpha Vantage fetch (no requests dependency)
# -----------------------
def fetch_alpha_intraday(symbol: str, interval: str = "5min", outputsize: str = "compact"):
    """
    Returns list of OHLC dicts sorted oldest->newest:
    [{"t":datetime, "open":, "high":, "low":, "close":, "volume":}, ...]
    On error returns None.
    """
    if not ALPHAVANTAGE_KEY:
        return None
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": ALPHAVANTAGE_KEY,
        "outputsize": outputsize,
        "datatype": "json"
    }
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    except Exception as e:
        return None
    # find key containing "Time Series"
    ts_key = None
    for k in data.keys():
        if "Time Series" in k:
            ts_key = k
            break
    if not ts_key:
        return None
    try:
        entries = data[ts_key]
        items = []
        # entries keys are timestamps
        for t_str in sorted(entries.keys()):
            row = entries[t_str]
            o = float(row["1. open"])
            h = float(row["2. high"])
            l = float(row["3. low"])
            c = float(row["4. close"])
            vol = float(row.get("5. volume", 0))
            # convert t_str to datetime
            try:
                t_dt = datetime.fromisoformat(t_str)
            except Exception:
                # fallback parse
                t_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
            items.append({"t": t_dt, "open": o, "high": h, "low": l, "close": c, "volume": vol})
        # ensure sorted oldest->newest
        items_sorted = sorted(items, key=lambda x: x["t"])
        return items_sorted
    except Exception:
        return None

# -----------------------
# Offline fallback generator (per-asset deterministic)
# -----------------------
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

def prices_to_ohlc(prices, candle_size=1):
    ohlc = []
    for i in range(0, len(prices), candle_size):
        chunk = prices[i:i+candle_size]
        if not chunk: continue
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
        ohlc.append({"t": None, "open": o, "high": h, "low": l, "close": c, "volume": 0})
    # set timestamps relative to now
    now = datetime.utcnow()
    minutes = candle_size
    for i in range(len(ohlc)):
        ohlc[i]["t"] = now - timedelta(minutes=(len(ohlc)-1-i) * minutes)
    return ohlc

# -----------------------
# Candlestick SVG renderer (robust)
# -----------------------
def render_candles_svg(candles, markers=None, stop_line=None, show_sma=(20,50), show_volume=True, width_px=1000, height_px=480):
    """
    candles: list of dicts with keys t, open, high, low, close, volume
    markers: [{'idx':int,'type':'buy'|'sell','reason':str}, ...]
    stop_line: {'price':float,'type':'long'|'short','pct':float}
    """
    if markers is None: markers = []
    n = len(candles)
    if n == 0:
        return "<svg></svg>"
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    max_p = max(highs); min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    margin = 48
    chart_w = width_px - 2*margin
    chart_h = int(height_px * 0.68)
    vol_h = int(height_px * 0.12) if show_volume else 0
    rsi_h = int(height_px * 0.16)
    candle_w = max(3, chart_w / n * 0.7)
    spacing = chart_w / n

    def y_pos(price):
        return margin + chart_h - (price - min_p) / (max_p - min_p) * chart_h

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

    svg = []
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#07070a"/>')

    # price grid
    for i in range(6):
        y = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 4)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{6}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

    # candles
    for idx, c in enumerate(candles):
        x_center = margin + idx*spacing + spacing/2
        x_left = x_center - candle_w/2
        body_top = y_pos(max(c["open"], c["close"]))
        body_bottom = y_pos(min(c["open"], c["close"]))
        y_high = y_pos(c["high"])
        y_low = y_pos(c["low"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"

        svg.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        h = max(1, abs(body_bottom - body_top))
        svg.append(f'<rect x="{x_left}" y="{body_top}" width="{candle_w}" height="{h}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')

    # SMAs
    def polyline(vals, stroke, width=1.6):
        pts = []
        for i, v in enumerate(vals):
            if v is None:
                pts.append(None)
            else:
                x = margin + i*spacing + spacing/2
                y = y_pos(v)
                pts.append(f"{x},{y}")
        segs = []; cur = []
        for p in pts:
            if p is None:
                if cur: segs.append(" ".join(cur)); cur=[]
            else:
                cur.append(p)
        if cur: segs.append(" ".join(cur))
        out = []
        for s in segs:
            out.append(f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>')
        return "\n".join(out)

    if sma1:
        svg.append(polyline(sma1, "#66ccff"))
    if sma2:
        svg.append(polyline(sma2, "#ffcc66"))

    # markers (place above/below with shifting to avoid overlap)
    shift_px = max(8, min(28, int(200 / max(1, n))))
    used_positions = {}
    for m in markers:
        i = m["idx"]
        if i < 0 or i >= n: continue
        x_center = margin + i*spacing + spacing/2
        c = candles[i]
        y_top = y_pos(c["high"])
        y_low = y_pos(c["low"])
        key = (i, m["type"])
        # ensure multiple markers at different vertical offsets
        offset_count = used_positions.get(i, 0)
        used_positions[i] = offset_count + 1
        if m["type"] == "buy":
            y = y_top - (12 + offset_count* (shift_px//2))
            color = "#00ff88"
            points = f"{x_center-7},{y} {x_center+7},{y} {x_center},{y+10}"
            svg.append(f'<polygon points="{points}" fill="{color}" opacity="0.98"><title>{m.get("reason","buy")}</title></polygon>')
        else:
            y = y_low + (12 + offset_count*(shift_px//2))
            color = "#ff7788"
            points = f"{x_center-7},{y} {x_center+7},{y} {x_center},{y-10}"
            svg.append(f'<polygon points="{points}" fill="{color}" opacity="0.98"><title>{m.get("reason","sell")}</title></polygon>')

    # stop_line if provided
    if stop_line:
        sp = stop_line.get("price")
        if sp is not None:
            y_sp = y_pos(sp)
            color = "#ffcc00" if stop_line.get("type") == "long" else "#ff4444"
            svg.append(f'<line x1="{margin}" y1="{y_sp}" x2="{width_px-margin}" y2="{y_sp}" stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>')
            label = f"Stop ({stop_line.get('type')}) {sp:.6f} ({stop_line.get('pct')*100:.2f}%)"
            svg.append(f'<rect x="{width_px-margin-260}" y="{y_sp-14}" width="260" height="18" fill="#101010" opacity="0.9"/>')
            svg.append(f'<text x="{width_px-margin-256}" y="{y_sp}" font-size="12" fill="{color}">{label}</text>')

    # RSI area
    rsi_area_top = margin + chart_h + 10
    rsi_area_h = rsi_h
    svg.append(f'<rect x="{margin}" y="{rsi_area_top}" width="{chart_w}" height="{rsi_area_h}" fill="#0b0b0b" stroke="#111"/>')
    svg.append(f'<text x="{margin}" y="{rsi_area_top-4}" font-size="12" fill="#9aa6b2">RSI(14)</text>')

    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        t = candles[i]["t"].strftime("%H:%M") if candles[i]["t"] else ""
        svg.append(f'<text x="{x-20}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

# -----------------------
# Pattern detection
# -----------------------
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
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"]>a["open"]) and (b["close"]>b["open"]) and (c["close"]>c["open"]) and (b["close"]>a["close"]) and (c["close"]>b["close"])

def detect_markers(candles):
    markers=[]
    for i in range(1,len(candles)):
        cur=candles[i]; prev=candles[i-1]
        if is_bullish_engulfing(prev,cur) or is_hammer(cur):
            markers.append({"idx":i,"type":"buy","reason":"Bullish/Hammer"})
        if is_bearish_engulfing(prev,cur) or is_shooting_star(cur):
            markers.append({"idx":i,"type":"sell","reason":"Bearish/Shooting Star"})
    # three white soldiers
    for i in range(2,len(candles)):
        if is_three_white_soldiers(candles[:i+1]):
            markers.append({"idx":i,"type":"buy","reason":"Three White Soldiers"})
    # deduplicate
    seen=set(); uniq=[]
    for m in markers:
        k=(m["idx"],m["type"])
        if k not in seen:
            seen.add(k); uniq.append(m)
    return uniq

# -----------------------
# Stop-loss calc (dynamic)
# -----------------------
def calculate_dynamic_stop(entry_price, candles, position_type="long"):
    closes=[c["close"] for c in candles[-30:]] if len(candles)>=2 else [entry_price]
    returns=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))] if len(closes)>1 else [0.0]
    import statistics
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    recommended_pct = max(0.01, min(0.10, vol*3))
    if position_type=="long":
        stop_price = entry_price*(1-recommended_pct)
    else:
        stop_price = entry_price*(1+recommended_pct)
    return round(stop_price,6), round(recommended_pct,4), vol

# -----------------------
# RSI (14)
# -----------------------
def compute_rsi(closes, period=14):
    if len(closes)<period+1:
        return [None]*len(closes)
    deltas=[closes[i]-closes[i-1] for i in range(1,len(closes))]
    gains=[d if d>0 else 0 for d in deltas]
    losses=[-d if d<0 else 0 for d in deltas]
    avg_gain=sum(gains[:period])/period
    avg_loss=sum(losses[:period])/period
    rsis=[None]*period
    for i in range(period, len(deltas)):
        avg_gain=(avg_gain*(period-1)+gains[i])/period
        avg_loss=(avg_loss*(period-1)+losses[i])/period
        rs=avg_gain/avg_loss if avg_loss!=0 else float('inf')
        rsi=100-(100/(1+rs))
        rsis.append(round(rsi,2))
    return rsis

# -----------------------
# High-level Analyze function
# -----------------------
def analyze_candles(candles, position_type="long", entry_price=None):
    markers = detect_markers(candles)
    closes=[c["close"] for c in candles]
    rsi = compute_rsi(closes, period=14)
    stop_price, stop_pct, vol = calculate_dynamic_stop(entry_price or closes[-1], candles, position_type=position_type)
    # basic pattern score
    reasons=[]; score=0
    last = candles[-1]; prev = candles[-2] if len(candles)>1 else None
    if is_doji(last): reasons.append("Doji")
    if prev and is_bullish_engulfing(prev,last): reasons.append("Bullish Engulfing"); score+=2
    if prev and is_bearish_engulfing(prev,last): reasons.append("Bearish Engulfing"); score-=2
    if is_hammer(last): reasons.append("Hammer"); score+=1
    if is_shooting_star(last): reasons.append("Shooting Star"); score-=1
    if is_three_white_soldiers(candles): reasons.append("Three White Soldiers"); score+=2
    # momentum
    returns=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))] if len(closes)>1 else [0.0]
    trend=sum(returns[-20:]) if len(returns)>0 else 0.0
    if trend>0.01: reasons.append("Positive Momentum"); score+=1
    if trend<-0.01: reasons.append("Negative Momentum"); score-=1
    rec = "Kaufen" if score>=2 else ("Verkaufen" if score<=-2 else "Halten / Beobachten")
    if vol<0.001: risk="Niedrig"
    elif vol<0.01: risk="Mittel"
    else: risk="Hoch"
    return {
        "markers": markers,
        "rsi": rsi,
        "stop_price": stop_price,
        "stop_pct": stop_pct,
        "volatility": vol,
        "reasons": reasons,
        "recommendation": rec,
        "risk": risk
    }

# -----------------------
# UI: Layout & Controls
# -----------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_json(PORTFOLIO_FILE)
if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE)
if "cache" not in st.session_state:
    st.session_state.cache = {}

left, right = st.columns([3,1])
with right:
    st.header("Controls")
    symbol_input = st.text_input("Symbol (Alpha Vantage) or name", value="AAPL")
    interval = st.selectbox("Interval", ["1min","5min","15min","30min","60min"])
    periods = st.slider("Candles", 20, 500, 120, step=10)
    start_price = st.number_input("Start price (fallback)", value=100.0, step=0.1)
    position_type = st.radio("Position", ["long","short"], index=0)
    manual_stop_pct = st.slider("Manual Stop (%) 0=auto", 0.0, 20.0, 0.0, step=0.1)
    fetch_live = st.button("Fetch Live (Alpha Vantage) / Refresh")
    refresh_sim = st.button("Refresh Offline Simulation")
    st.markdown("---")
    st.subheader("Portfolio")
    if st.button("Export Portfolio"):
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")
    if st.button("Clear Portfolio"):
        st.session_state.portfolio = []; save_json(PORTFOLIO_FILE, st.session_state.portfolio); st.success("Portfolio cleared")
    st.markdown("---")
    st.write("Key status:")
    st.write("AlphaVantageKey present" if ALPHAVANTAGE_KEY else "No API key set (use fallback)")

with left:
    st.header("Chart & Analyzer")

    # attempt live fetch
    symbol = symbol_input.strip()
    symbol_for_api = symbol
    # If user entered a known asset name that maps to a symbol, use it
    if symbol.lower() in NAME_TO_SYMBOL:
        symbol_for_api = NAME_TO_SYMBOL[symbol.lower()]

    cache_key = f"{symbol_for_api}|{interval}|{periods}|{start_price}"
    candles = None
    used_live = False

    if fetch_live:
        # try fetch live
        fetched = None
        if symbol_for_api and ALPHAVANTAGE_KEY:
            fetched = fetch_alpha_intraday(symbol_for_api, interval=interval, outputsize="compact")
        if fetched and len(fetched) >= periods//1:
            candles = fetched[-periods:]
            used_live = True
            st.success("Live data loaded from Alpha Vantage.")
            st.session_state.cache[cache_key] = {"candles":candles, "ts": time.time(), "live":True}
        else:
            st.warning("Live fetch failed or insufficient data; falling back to simulation.")
            # continue to simulation below

    # if not fetched, check cache
    if cache_key in st.session_state.cache and not (fetch_live):
        candles = st.session_state.cache[cache_key]["candles"]

    # if still nothing, simulate
    if candles is None or refresh_sim:
        tf_map = {"1min":1,"5min":5,"15min":15,"30min":30,"60min":60}
        mins = periods * tf_map.get(interval, 5)
        prices = generate_price_walk(symbol + "|" + interval, mins, start_price)
        ohlc = prices_to_ohlc(prices, candle_size=tf_map.get(interval,5))
        # ensure correct length
        if len(ohlc) < periods:
            pad = periods - len(ohlc)
            pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price,"volume":0}
            ohlc = [pad_item]*pad + ohlc
        candles = ohlc[-periods:]
        st.session_state.cache[cache_key] = {"candles":candles,"ts":time.time(),"live":False}
        if refresh_sim:
            st.success("Simulation refreshed.")

    # analysis
    entry_price = candles[-1]["close"]
    analysis = analyze_candles(candles, position_type=position_type, entry_price=entry_price)
    markers = analysis["markers"]
    rsi = analysis["rsi"]
    stop_line = {"price": analysis["stop_price"], "type": position_type, "pct": analysis["stop_pct"]}

    # override stop if manual
    if manual_stop_pct > 0.0:
        spct = manual_stop_pct/100.0
        if position_type == "long":
            stopp = round(entry_price*(1-spct),6)
        else:
            stopp = round(entry_price*(1+spct),6)
        stop_line = {"price": stopp, "type": position_type, "pct": spct}

    # render SVG
    svg = render_candles_svg(candles, markers=markers, stop_line=stop_line, show_sma=(20,50), show_volume=True, width_px=1000, height_px=520)
    st.components.v1.html(svg, height=540)

    # display analysis details
    st.subheader("Analysis & Recommendation")
    st.write(f"Recommendation: **{analysis['recommendation']}**")
    st.write(f"Risk: **{analysis['risk']}** • Volatility (σ): {analysis['volatility']:.6f}")
    st.write(f"Entry (last close): {entry_price:.6f} • Suggested stop: {analysis['stop_price']:.6f} ({analysis['stop_pct']*100:.2f}%)")
    if analysis["reasons"]:
        st.write("Detected patterns / reasons:")
        for r in analysis["reasons"]:
            st.write("- " + r)
    else:
        st.write("No strong patterns detected.")

    st.markdown("---")
    st.subheader("Trade Simulator (Quick)")
    qty = st.number_input("Quantity", min_value=0.0, value=1.0, step=0.1)
    tp_multiplier = st.slider("TP / Stop multiplier", 1.0, 5.0, 2.0, step=0.1)
    if st.button("Open Simulated Trade (Market)"):
        entry = entry_price
        stop = stop_line["price"]
        direction = position_type
        if direction == "long":
            tp = round(entry + (entry - stop) * tp_multiplier, 6)
        else:
            tp = round(entry - (stop - entry) * tp_multiplier, 6)
        trade = {
            "id": f"trade_{int(time.time())}",
            "symbol": symbol,
            "direction": direction,
            "qty": qty,
            "entry": entry,
            "stop": stop,
            "tp": tp,
            "opened_at": human_ts()
        }
        st.session_state.portfolio.append(trade)
        save_json(PORTFOLIO_FILE, st.session_state.portfolio)
        st.success(f"Trade opened: {direction} {symbol} qty {qty} entry {entry:.6f} stop {stop:.6f} tp {tp:.6f}")
        # history
        hist = load_json(HISTORY_FILE)
        hist.append({"ts": human_ts(), "action": "open_trade", "trade": trade})
        save_json(HISTORY_FILE, hist)

    st.markdown("---")
    st.subheader("Portfolio (Simulated Trades)")
    if st.session_state.portfolio:
        total_est = 0.0
        for t in st.session_state.portfolio:
            # estimate current price with small deterministic walk
            pid = str(t.get("symbol", t.get("id", "X")))
            tmp_prices = generate_price_walk(pid + "|p", 20, t.get("entry", entry_price))
            cur_price = tmp_prices[-1]
            if t["direction"] == "long":
                val = cur_price * t["qty"]
                pnl = (cur_price - t["entry"]) * t["qty"]
            else:
                val = (t["entry"] - cur_price) * t["qty"]
                pnl = (t["entry"] - cur_price) * t["qty"]
            total_est += val
            st.write(f"- {t['symbol']} | {t['direction']} | qty {t['qty']} | entry {t['entry']:.6f} | cur {cur_price:.6f} | P&L {pnl:.4f}")
        st.write(f"Estimated portfolio value (sim): {total_est:.2f}")
        if st.button("Clear simulated trades"):
            st.session_state.portfolio = []
            save_json(PORTFOLIO_FILE, st.session_state.portfolio)
            st.success("Cleared.")
    else:
        st.write("No positions.")

# Footer
st.markdown("---")
st.caption("Analyzer Deluxe: Live via Alpha Vantage when available; otherwise deterministic offline simulation. Move API key to st.secrets for security.")
