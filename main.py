# main.py
# Analyzer v2.0 — Backtesting, RSI, MACD, Bollinger, adaptive stop, perceptron ML
# Streamlit-only. Uses Alpha Vantage if key provided, else deterministic offline simulation.
# No external Python packages required other than streamlit.

import streamlit as st
import json
import os
import random
import math
import statistics
import time
import urllib.request, urllib.parse
from datetime import datetime, timedelta

# ---------------------
# CONFIG
# ---------------------
st.set_page_config(page_title="Analyzer v2.0 — Backtest & Indicators", layout="wide")
st.title("Analyzer v2.0 — Backtesting, Indicators, Adaptive Stop, Perceptron")

# ---------- FILES ----------
PORTFOLIO_FILE = "portfolio.json"
HISTORY_FILE = "history.json"
MODEL_FILE = "perceptron_model.json"

def ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)
ensure_file(PORTFOLIO_FILE, [])
ensure_file(HISTORY_FILE, [])
ensure_file(MODEL_FILE, {})

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ---------------------
# API KEY (Alpha Vantage)
# ---------------------
# If you want to hardcode your key (not recommended), set here.
# ALPHAVANTAGE_KEY = "22XGVO0TQ1UV167C"   # <-- optional (not recommended to commit)
ALPHAVANTAGE_KEY = None
# Prefer using Streamlit Secrets: st.secrets["api_keys"]["ALPHAVANTAGE"] if set
try:
    if not ALPHAVANTAGE_KEY:
        ALPHAVANTAGE_KEY = st.secrets["api_keys"]["ALPHAVANTAGE"]
except Exception:
    pass

# ---------------------
# Asset lists (examples)
# ---------------------
ETFS = ["iShares DAX","SP500 ETF","MSCI World","EuroStoxx","Asia Pacific ETF","Emerging Mkts ETF","Tech Leaders ETF","Value ETF","Dividend ETF","Global SmallCap"]
STOCKS = ["AAPL","MSFT","AMZN","TSLA","NVDA","GOOGL","META","NFLX","INTC","AMD","SAP","SIE.DE","ALV.DE","BAYN.DE","VOW3.DE","DAI.DE","RDSA","BP","DBK.DE","SIE.DE"]
CRYPTOS = ["BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOT-USD","LINK-USD","XRP-USD","LTC-USD","DOGE-USD","AVAX-USD"]

ALL_ASSETS = ETFS + STOCKS + CRYPTOS

# ---------------------
# Utilities
# ---------------------
def deterministic_seed(s: str) -> int:
    return abs(hash(s)) % (2**31)

def human_ts():
    return datetime.utcnow().isoformat() + "Z"

def now():
    return datetime.utcnow()

# ---------------------
# Fetch from Alpha Vantage (built-in, no requests)
# ---------------------
def fetch_alpha_intraday(symbol: str, interval: str="5min", outputsize: str="compact"):
    """Return list of OHLC dicts oldest->newest or None on error"""
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
    except Exception:
        return None
    # find time series key
    ts_key = None
    for k in data.keys():
        if "Time Series" in k:
            ts_key = k
            break
    if not ts_key:
        return None
    items = []
    try:
        for t_str in sorted(data[ts_key].keys()):
            row = data[ts_key][t_str]
            o = float(row["1. open"]); h = float(row["2. high"]); l = float(row["3. low"]); c = float(row["4. close"])
            vol = float(row.get("5. volume", 0))
            # parse timestamp
            try:
                t_dt = datetime.fromisoformat(t_str)
            except Exception:
                t_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
            items.append({"t": t_dt, "open": o, "high": h, "low": l, "close": c, "volume": vol})
        items_sorted = sorted(items, key=lambda x: x["t"])
        return items_sorted
    except Exception:
        return None

# ---------------------
# Offline generator fallback
# ---------------------
def generate_price_walk(seed_name: str, steps: int, start_price: float):
    rnd = random.Random(deterministic_seed(seed_name))
    price = float(start_price)
    series = []
    for _ in range(steps):
        drift = (rnd.random() - 0.49) * 0.003
        shock = (rnd.random() - 0.5) * 0.018
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
    now_dt = datetime.utcnow()
    minutes = candle_size
    for i in range(len(ohlc)):
        ohlc[i]["t"] = now_dt - timedelta(minutes=(len(ohlc)-1-i)*minutes)
    return ohlc

# ---------------------
# INDICATORS (no numpy)
# ---------------------
def sma(values, period):
    res = []
    for i in range(len(values)):
        if i+1 < period:
            res.append(None)
        else:
            res.append(sum(values[i+1-period:i+1]) / period)
    return res

def ema(values, period):
    res = []
    k = 2 / (period + 1)
    ema_prev = None
    for i, v in enumerate(values):
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v * k + ema_prev * (1-k)
        res.append(ema_prev)
    return res

def macd(values, fast=12, slow=26, signal=9):
    ema_fast = ema(values, fast)
    ema_slow = ema(values, slow)
    macd_line = []
    for f, s in zip(ema_fast, ema_slow):
        macd_line.append((f - s) if (f is not None and s is not None) else None)
    # compute signal as EMA on macd_line (skipping None)
    macd_vals = [v for v in macd_line if v is not None]
    if not macd_vals:
        signal_line = [None]*len(macd_line)
        hist = [None]*len(macd_line)
        return macd_line, signal_line, hist
    # build EMA manually for macd
    signal_vals = ema(macd_vals, signal)
    # map back
    sig_iter = iter(signal_vals)
    signal_line = []
    skip = len(macd_line) - len(macd_vals)
    for i, v in enumerate(macd_line):
        if v is None:
            signal_line.append(None)
        else:
            signal_line.append(next(sig_iter))
    hist = [(m - s) if (m is not None and s is not None) else None for m, s in zip(macd_line, signal_line)]
    return macd_line, signal_line, hist

def rsi(values, period=14):
    if len(values) < period + 1:
        return [None]*len(values)
    deltas = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    rsis = [None]*period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        val = 100 - (100 / (1 + rs))
        rsis.append(round(val,2))
    return rsis

def bollinger(values, period=20, mult=2.0):
    res = []
    for i in range(len(values)):
        if i+1 < period:
            res.append((None, None, None))
        else:
            window = values[i+1-period:i+1]
            m = sum(window)/period
            std = statistics.pstdev(window)
            upper = m + mult * std
            lower = m - mult * std
            res.append((round(m,6), round(upper,6), round(lower,6)))
    return res

# ---------------------
# Pattern detection (same rules as before)
# ---------------------
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

# ---------------------
# STOP LOSS (adaptive)
# ---------------------
def calculate_dynamic_stop(entry_price, candles, position_type="long"):
    closes = [c["close"] for c in candles[-30:]] if len(candles)>=2 else [entry_price]
    returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))] if len(closes)>1 else [0.0]
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    recommended_pct = max(0.005, min(0.15, vol * 3.5))  # a bit wider mapping
    if position_type == "long":
        stop_price = entry_price * (1 - recommended_pct)
    else:
        stop_price = entry_price * (1 + recommended_pct)
    return round(stop_price,6), round(recommended_pct,4), vol

# ---------------------
# SVG Candles renderer (improved)
# ---------------------
def render_candles_svg(candles, markers=None, stop_line=None, sma_periods=(20,50), rsi_vals=None, macd_vals=None, boll=None, width_px=1000, height_px=560):
    if markers is None: markers=[]
    n = len(candles)
    if n == 0:
        return "<svg></svg>"
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    max_p = max(highs); min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    margin = 52
    chart_h = int(height_px * 0.62)
    rsi_h = int(height_px * 0.16)
    candle_w = max(3, (width_px - 2*margin) / n * 0.7)
    spacing = (width_px - 2*margin) / n

    def y_pos(price):
        return margin + chart_h - (price - min_p) / (max_p - min_p) * chart_h

    closes = [c["close"] for c in candles]
    sma1 = sma(closes, sma_periods[0]) if sma_periods[0] else []
    sma2 = sma(closes, sma_periods[1]) if sma_periods[1] else []

    svg = []
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#07070a"/>')

    # price grid and labels
    for i in range(6):
        y = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="6" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

    # candles
    for idx,c in enumerate(candles):
        x_center = margin + idx*spacing + spacing/2
        x_left = x_center - candle_w/2
        y_open = y_pos(c["open"]); y_close = y_pos(c["close"])
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        by = min(y_open, y_close); bh = max(1, abs(y_close - y_open))
        svg.append(f'<rect x="{x_left}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')

    # SMAs
    def polyline(vals, stroke, width=1.6):
        pts=[]; segs=[]; cur=[]
        for i,v in enumerate(vals):
            if v is None:
                if cur: segs.append(" ".join(cur)); cur=[]
            else:
                x = margin + i*spacing + spacing/2; y=y_pos(v)
                cur.append(f"{x},{y}")
        if cur: segs.append(" ".join(cur))
        out=[]
        for s in segs:
            out.append(f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>')
        return "\n".join(out)
    if sma1: svg.append(polyline(sma1, "#66ccff"))
    if sma2: svg.append(polyline(sma2, "#ffcc66"))

    # Bollinger bands shading
    if boll:
        upper = [b[1] for b in boll]; lower=[b[2] for b in boll]
        pts_up=[]; pts_down=[]
        for i,u in enumerate(upper):
            if u is None:
                pts_up.append(None)
            else:
                x=margin+i*spacing+spacing/2; y=y_pos(u); pts_up.append(f"{x},{y}")
        for i,lw in enumerate(lower):
            if lw is None:
                pts_down.append(None)
            else:
                x=margin+i*spacing+spacing/2; y=y_pos(lw); pts_down.append(f"{x},{y}")
        # join contiguous segments and render polygons (upper -> reversed lower)
        def segments_from_pts(pts):
            segs=[]; cur=[]
            for p in pts:
                if p is None:
                    if cur: segs.append(cur); cur=[]
                else:
                    cur.append(p)
            if cur: segs.append(cur)
            return segs
        ups = segments_from_pts(pts_up); downs = segments_from_pts(pts_down)
        for u_seg,d_seg in zip(ups,downs):
            # form polygon points: u_seg + reversed d_seg
            poly = " ".join(u_seg + d_seg[::-1])
            svg.append(f'<polygon points="{poly}" fill="#222a2a" opacity="0.25" stroke="none" />')

    # markers
    shift_px = max(8, min(30, int(200 / max(1, n))))
    used_positions = {}
    for m in detect_markers(candles):
        i = m["idx"]
        if i<0 or i>=n: continue
        x_center = margin + i*spacing + spacing/2
        c = candles[i]
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        offset_count = used_positions.get(i, 0)
        used_positions[i] = offset_count + 1
        if m["type"]=="buy":
            y = y_high - (12 + offset_count*(shift_px//2))
            svg.append(f'<polygon points="{x_center-7},{y} {x_center+7},{y} {x_center},{y+10}" fill="#00ff88" opacity="0.98"><title>{m["reason"]}</title></polygon>')
        else:
            y = y_low + (12 + offset_count*(shift_px//2))
            svg.append(f'<polygon points="{x_center-7},{y} {x_center+7},{y} {x_center},{y-10}" fill="#ff7788" opacity="0.98"><title>{m["reason"]}</title></polygon>')

    # stop_line
    if stop_line and stop_line.get("price") is not None:
        y_sp = y_pos(stop_line["price"])
        color = "#ffcc00" if stop_line.get("type")=="long" else "#ff4444"
        svg.append(f'<line x1="{margin}" y1="{y_sp}" x2="{width_px-margin}" y2="{y_sp}" stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>')
        label = f"Stop({stop_line.get('type')}) {stop_line['price']:.6f} ({stop_line['pct']*100:.2f}%)"
        svg.append(f'<rect x="{width_px-margin-300}" y="{y_sp-16}" width="300" height="20" fill="#101010" opacity="0.9"/>')
        svg.append(f'<text x="{width_px-margin-296}" y="{y_sp-2}" font-size="12" fill="{color}">{label}</text>')

    # RSI area
    rsi_top = margin + chart_h + 12
    rsi_h = int(height_px * 0.16)
    svg.append(f'<rect x="{margin}" y="{rsi_top}" width="{width_px-2*margin}" height="{rsi_h}" fill="#0b0b0b" stroke="#111"/>')
    svg.append(f'<text x="{margin}" y="{rsi_top-4}" font-size="12" fill="#9aa6b2">RSI(14)</text>')
    if rsi_vals:
        pts=[]; segs=[]
        for i,v in enumerate(rsi_vals):
            if v is None: pts.append(None); continue
            x=margin+i*spacing+spacing/2
            y=rsi_top + rsi_h - (v/100.0)*rsi_h
            pts.append(f"{x},{y}")
        cur=[]; segs=[]
        for p in pts:
            if p is None:
                if cur: segs.append(" ".join(cur)); cur=[]
            else:
                cur.append(p)
        if cur: segs.append(" ".join(cur))
        for s in segs:
            svg.append(f'<polyline points="{s}" fill="none" stroke="#ff66cc" stroke-width="1.2" />')
        # horizontal 30/70
        y30 = rsi_top + rsi_h - 0.3*rsi_h
        y70 = rsi_top + rsi_h - 0.7*rsi_h
        svg.append(f'<line x1="{margin}" y1="{y30}" x2="{width_px-margin}" y2="{y30}" stroke="#333" stroke-dasharray="3,3" />')
        svg.append(f'<line x1="{margin}" y1="{y70}" x2="{width_px-margin}" y2="{y70}" stroke="#333" stroke-dasharray="3,3" />')

    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        t = candles[i]["t"].strftime("%H:%M") if candles[i].get("t") else ""
        svg.append(f'<text x="{x-22}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

# ---------------------
# SIMPLE PERCEPTRON (no sklearn)
# ---------------------
class SimplePerceptron:
    def __init__(self, n_features):
        self.n = n_features
        self.weights = [random.uniform(-0.01,0.01) for _ in range(n_features)]
        self.bias = random.uniform(-0.01,0.01)
        self.lr = 0.01

    def predict_raw(self, x):
        s = sum(w*xi for w,xi in zip(self.weights,x)) + self.bias
        return s

    def predict(self, x):
        return 1 if self.predict_raw(x) >= 0 else -1

    def train_epoch(self, X, y):
        errors=0
        for xi, yi in zip(X,y):
            pred = self.predict(xi)
            if pred != yi:
                errors += 1
                # perceptron update
                for j in range(self.n):
                    self.weights[j] += self.lr * yi * xi[j]
                self.bias += self.lr * yi
        return errors

    def train(self, X, y, epochs=50):
        for e in range(epochs):
            err = self.train_epoch(X,y)
            if err == 0:
                break

    def export(self):
        return {"n":self.n, "weights": self.weights, "bias": self.bias}

    def load(self, obj):
        self.n = obj["n"]; self.weights = obj["weights"]; self.bias = obj["bias"]

# create / load model
def load_model():
    m = load_json(MODEL_FILE)
    if m:
        try:
            p = SimplePerceptron(m["n"])
            p.load(m)
            return p
        except Exception:
            pass
    return SimplePerceptron(6)  # default features count

def save_model(p):
    save_json(MODEL_FILE, p.export())

# ---------------------
# FEATURES FOR ML
# ---------------------
def features_from_candles(candles):
    # produce a fixed-length feature vector from last N candles:
    # features: last return, sma_diff (20-50)/price, rsi_last, macd_last, vol25, atr-like
    closes = [c["close"] for c in candles]
    if len(closes) < 26:
        # pad with copies
        while len(closes) < 26:
            closes.insert(0, closes[0] if closes else 100.0)
    last = closes[-1]
    ret1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0.0
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50 if len(closes)>=50 else sma20
    sma_diff = (sma20 - sma50) / last if last != 0 else 0.0
    rsi_vals = rsi(closes, period=14)
    rsi_last = rsi_vals[-1] if rsi_vals and rsi_vals[-1] is not None else 50.0
    macd_line, signal_line, hist = macd(closes, 12, 26, 9)
    macd_last = macd_line[-1] if macd_line and macd_line[-1] is not None else 0.0
    vol = statistics.pstdev([(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]) if len(closes)>1 else 0.0
    # ATR-like: average high-low over last 14
    highs = [c["high"] for c in candles[-14:]] if len(candles)>=14 else [c["high"] for c in candles]
    lows = [c["low"] for c in candles[-14:]] if len(candles)>=14 else [c["low"] for c in candles]
    atr = sum([h-l for h,l in zip(highs,lows)]) / len(highs) if highs else 0.0
    return [ret1, sma_diff, rsi_last/100.0, macd_last, vol, atr/last if last!=0 else 0.0]

# ---------------------
# BACKTEST (simple strategy)
# ---------------------
def backtest_strategy(candles, strategy_fn, initial_cash=10000.0):
    """
    strategy_fn(candles, i) -> signal: 1 buy, -1 sell/short, 0 hold
    Executes market entries and uses stop-loss or TP from strategy return.
    Returns performance dict with trades list and metrics.
    """
    cash = initial_cash
    position = None  # dict {entry_price, qty, direction, stop, tp, entry_index}
    trades = []
    equity_curve = []
    for i in range(len(candles)):
        price = candles[i]["close"]
        signal = strategy_fn(candles, i)
        # check existing position stop/TP
        if position:
            # check stop
            if position["direction"] == "long":
                if price <= position["stop"]:
                    # close at stop
                    pnl = (position["stop"] - position["entry_price"]) * position["qty"]
                    cash += position["qty"] * position["stop"]
                    trades.append({**position, "exit_price": position["stop"], "exit_index": i, "pnl": pnl})
                    position = None
                elif price >= position["tp"]:
                    pnl = (position["tp"] - position["entry_price"]) * position["qty"]
                    cash += position["qty"] * position["tp"]
                    trades.append({**position, "exit_price": position["tp"], "exit_index": i, "pnl": pnl})
                    position = None
            else:  # short
                if price >= position["stop"]:
                    pnl = (position["entry_price"] - position["stop"]) * position["qty"]
                    cash += position["qty"] * (position["entry_price"] - position["stop"])  # simplified
                    trades.append({**position, "exit_price": position["stop"], "exit_index": i, "pnl": pnl})
                    position = None
                elif price <= position["tp"]:
                    pnl = (position["entry_price"] - position["tp"]) * position["qty"]
                    cash += position["qty"] * (position["entry_price"] - position["tp"])
                    trades.append({**position, "exit_price": position["tp"], "exit_index": i, "pnl": pnl})
                    position = None
        # if signal and no position: open new
        if signal and not position:
            direction = "long" if signal == 1 else "short"
            # simple sizing: use 10% of equity
            risk_capital = 0.1 * (cash if cash>0 else initial_cash)
            qty = risk_capital / price if price>0 else 0.0
            # dynamic stop
            stop_price, stop_pct, vol = calculate_dynamic_stop(price, candles[:i+1], position_type=direction)
            # TP
            tp = price + (price - stop_price) * 2 if direction=="long" else price - (stop_price - price) * 2
            position = {"entry_price": price, "qty": qty, "direction": direction, "stop": stop_price, "tp": tp, "entry_index": i}
            # subtract notional for long for simplicity
            if direction=="long":
                cash -= qty * price
            trades.append({**position, "opened": True, "opened_index": i})
        # append equity
        if position:
            if position["direction"]=="long":
                cur_val = cash + position["qty"] * price
            else:
                # unrealized pnl for short: (entry - price) * qty + cash
                cur_val = cash + (position["entry_price"] - price) * position["qty"]
        else:
            cur_val = cash
        equity_curve.append(cur_val)
    # finalize: close remaining at last price
    if position:
        price = candles[-1]["close"]
        if position["direction"]=="long":
            pnl = (price - position["entry_price"]) * position["qty"]
            cash += position["qty"] * price
        else:
            pnl = (position["entry_price"] - price) * position["qty"]
            cash += position["qty"] * (position["entry_price"] - price)
        trades.append({**position, "exit_price": price, "exit_index": len(candles)-1, "pnl": pnl})
        position = None
        equity_curve.append(cash)
    # metrics
    total_return = (cash - initial_cash) / initial_cash if initial_cash else 0.0
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] != 0:
            returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
    avg_ret = statistics.mean(returns) if returns else 0.0
    std_ret = statistics.pstdev(returns) if len(returns)>1 else 0.0
    sharpe_like = (avg_ret / std_ret) if std_ret != 0 else 0.0
    # max drawdown
    peak = -1e9; mdd = 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak if peak>0 else 0.0
        if dd > mdd: mdd = dd
    winners = [t for t in trades if t.get("pnl",0) > 0]
    winrate = len(winners)/len(trades) if trades else 0.0
    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "total_return": total_return,
        "sharpe_like": sharpe_like,
        "max_drawdown": mdd,
        "winrate": winrate,
        "final_cash": cash
    }

# ---------------------
# Strategy examples
# ---------------------
def strategy_sma_cross(candles, i):
    # buy when sma20 crosses above sma50 at i
    closes = [c["close"] for c in candles[:i+1]]
    if len(closes) < 51: return 0
    s20 = sum(closes[-20:]) / 20
    s50 = sum(closes[-50:]) / 50
    prev_s20 = sum(closes[-21:-1]) / 20
    prev_s50 = sum(closes[-51:-1]) / 50
    if prev_s20 <= prev_s50 and s20 > s50:
        return 1
    if prev_s20 >= prev_s50 and s20 < s50:
        return -1
    return 0

def strategy_macd_momentum(candles, i):
    closes = [c["close"] for c in candles[:i+1]]
    if len(closes) < 35: return 0
    m, s, h = macd(closes,12,26,9)
    if m[-1] is None or s[-1] is None: return 0
    if m[-1] > s[-1] and m[-2] <= s[-2]:
        return 1
    if m[-1] < s[-1] and m[-2] >= s[-2]:
        return -1
    return 0

# ---------------------
# UI: controls & main flow
# ---------------------
col1, col2 = st.columns([3,1])
with col2:
    st.header("Settings")
    symbol_input = st.text_input("Symbol (AAPL, TSLA, BTC-USD, or name)", value="AAPL")
    interval = st.selectbox("Interval (for AlphaV)", ["1min","5min","15min","30min","60min"])
    periods = st.slider("Candles / Backtest length", 30, 1000, 200, step=10)
    start_price = st.number_input("Start price (fallback)", value=100.0, step=0.1)
    use_live = st.checkbox("Use Alpha Vantage live data if available", value=True)
    run_fetch = st.button("Fetch / Refresh Data")
    st.markdown("---")
    st.subheader("ML Perceptron")
    model = load_model()
    train_model = st.button("Train perceptron on simulated patterns (offline)")
    st.markdown("---")
    st.subheader("Backtest")
    strategy_choice = st.selectbox("Strategy", ["SMA Crossover", "MACD Momentum"])
    initial_cash = st.number_input("Initial cash", value=10000.0, step=100.0)
    run_backtest = st.button("Run Backtest")
    st.markdown("---")
    st.write("AlphaV key present" if ALPHAVANTAGE_KEY else "No AlphaV key (offline fallback)")

with col1:
    st.header("Chart / Analyzer / Backtest")
    # determine symbol for API
    sym = symbol_input.strip()
    symbol_for_api = sym
    # try to fetch
    candles = None
    used_live = False
    if run_fetch and use_live and ALPHAVANTAGE_KEY:
        fetched = fetch_alpha_intraday(symbol_for_api, interval=interval, outputsize="compact")
        if fetched and len(fetched) >= periods:
            candles = fetched[-periods:]
            used_live = True
            st.success("Fetched live data from Alpha Vantage.")
        else:
            st.warning("Live fetch failed or insufficient data; falling back to simulation.")
    # if not live or not fetched: check cache
    cache_key = f"{sym}|{interval}|{periods}|{start_price}"
    if "cache" not in st.session_state:
        st.session_state.cache = {}
    if not candles and cache_key in st.session_state.cache:
        candles = st.session_state.cache[cache_key]
    if not candles:
        # simulate deterministic candles
        tf_map = {"1min":1,"5min":5,"15min":15,"30min":30,"60min":60}
        mins = periods * tf_map.get(interval,5)
        prices = generate_price_walk(sym + "|" + interval, mins, start_price)
        ohlc = prices_to_ohlc(prices, candle_size=tf_map.get(interval,5))
        if len(ohlc) < periods:
            pad = periods - len(ohlc)
            pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price,"volume":0}
            ohlc = [pad_item]*pad + ohlc
        candles = ohlc[-periods:]
        st.session_state.cache[cache_key] = candles
        if run_fetch:
            st.info("Using deterministic offline simulation.")

    # compute indicators
    closes = [c["close"] for c in candles]
    sma20 = sma(closes,20)
    sma50 = sma(closes,50)
    macd_line, macd_signal, macd_hist = macd(closes,12,26,9)
    rsi_vals = rsi(closes,14)
    boll = bollinger(closes,20,2.0)
    markers = detect_markers(candles)
    # analyze
    entry_price = closes[-1]
    # choose strategy for backtest or live recommendation
    analysis = analyze_candles(candles, position_type="long", entry_price=entry_price) if 'analyze_candles' in globals() else {}
    stop_price, stop_pct, vol = calculate_dynamic_stop(entry_price, candles, position_type="long")
    stop_line = {"price": stop_price, "type":"long", "pct": stop_pct}
    svg = render_candles_svg(candles, markers=markers, stop_line=stop_line, sma_periods=(20,50), rsi_vals=rsi_vals, macd_vals=None, boll=boll, width_px=1100, height_px=580)
    st.components.v1.html(svg, height=620)

    # textual analysis
    st.subheader("Analysis Summary")
    try:
        rec = analysis.get("recommendation","Halten / Beobachten")
        st.write(f"**Recommendation:** {rec}")
        st.write(f"**Risk:** {analysis.get('risk','Unbekannt')}  • Volatility: {analysis.get('volatility',0.0):.6f}")
        if analysis.get("reasons"):
            st.write("Detected patterns:")
            for r in analysis["reasons"]:
                st.write("-", r)
    except Exception:
        st.write("No analysis available.")

    # ML block: train perceptron on simulated labeled patterns
    if train_model:
        st.info("Training perceptron on synthetic labeled patterns...")
        # build synthetic dataset
        X=[]; y=[]
        # generate patterns: bullish samples (hammer / engulfing / three white), bearish samples (shooting star / bearish engulfing)
        for _ in range(600):
            # simulate series
            base = generate_price_walk("train"+str(_), 200, 100.0)
            ohlc = prices_to_ohlc(base, candle_size=5)
            # randomly insert a bullish pattern near the end for positive class
            label = random.choice([1,-1])
            if label == 1:
                # make last candles bullish by injecting up moves
                for i in range(3):
                    ohlc[-1-i]["open"] = ohlc[-1-i]["close"] * (1 - random.uniform(0.01,0.03))
                    ohlc[-1-i]["close"] = ohlc[-1-i]["open"] * (1 + random.uniform(0.01,0.05))
            else:
                for i in range(3):
                    ohlc[-1-i]["open"] = ohlc[-1-i]["close"] * (1 + random.uniform(0.01,0.04))
                    ohlc[-1-i]["close"] = ohlc[-1-i]["open"] * (1 - random.uniform(0.01,0.05))
            feat = features_from_candles(ohlc)
            X.append(feat); y.append(label)
        p = SimplePerceptron(len(X[0]))
        p.train(X,y,epochs=60)
        save_model(p)
        st.success("Perceptron trained and saved.")
        model = p

    # use perceptron to predict current candles
    model = load_model()
    feat_now = features_from_candles(candles)
    ml_pred = model.predict(feat_now)
    st.write("ML Perceptron signal:", "BUY" if ml_pred==1 else "SELL")

    # Backtest
    if run_backtest:
        st.info("Running backtest...")
        strategy_fn = strategy_sma_cross if strategy_choice=="SMA Crossover" else strategy_macd_momentum
        bt = backtest_strategy(candles, strategy_fn, initial_cash=initial_cash)
        st.subheader("Backtest Results")
        st.write(f"Total Return: {bt['total_return']*100:.2f}%")
        st.write(f"Sharpe-like (avg/std per step): {bt['sharpe_like']:.4f}")
        st.write(f"Max Drawdown: {bt['max_drawdown']*100:.2f}%")
        st.write(f"Winrate: {bt['winrate']*100:.2f}%")
        st.write(f"Trades: {len(bt['trades'])}")
        if bt['trades']:
            st.dataframe([{"entry_index":t.get("entry_index"), "exit_index":t.get("exit_index"), "direction":t.get("direction"), "pnl": round(t.get("pnl",0),4)} for t in bt['trades']])
        st.line_chart(bt["equity_curve"])

st.markdown("---")
st.caption("Analyzer v2.0 — Backtesting + Indicators + Perceptron. Uses Alpha Vantage if available; otherwise deterministic offline simulation. Move API key to st.secrets for security.")
