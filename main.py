# main.py
# Analyzer Deluxe Final — Live (Alpha Vantage) + Offline fallback
# Streamlit-only (kein plotly/matplotlib). SVG candlesticks, pattern detection,
# adaptive stop-loss, long/short recommendation, backtesting, perceptron-ML,
# estimated success probability + risk %
#
# Hinweis: API key ist voreingestellt (wenn du ihn in den Code willst).
# Es ist sicherer, den Key in Streamlit Secrets zu speichern:
#   .streamlit/secrets.toml
#   [api_keys]
#   ALPHA_KEY = "22XGVO0TQ1UV167C"
#
# Wenn du den Key im Code belassen möchtest, setze ALPHA_KEY unten.

import streamlit as st
import json, os, math, random, time, urllib.request, urllib.parse
from datetime import datetime, timedelta
import statistics

# -----------------------
# APP CONFIG
# -----------------------
st.set_page_config(page_title="Analyzer Deluxe Final", layout="wide")
st.title("Analyzer Deluxe — Live (Alpha Vantage) + Offline Fallback")

# -----------------------
# API KEY (inserted or from secrets)
# -----------------------
# You asked to include: 22XGVO0TQ1UV167C
# If you prefer, move to st.secrets (recommended).
ALPHA_KEY = "22XGVO0TQ1UV167C"
try:
    if not ALPHA_KEY:
        ALPHA_KEY = st.secrets["api_keys"]["ALPHA_KEY"]
except Exception:
    pass

# -----------------------
# Files / persistence
# -----------------------
PORTFOLIO_FILE = "portfolio.json"
HISTORY_FILE = "history.json"
CACHE_DIR = ".cache_av"
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

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

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
 "RDSA","BP","DBK.DE","SIE.DE"
]
CRYPTOS = [
 "BTC-USD","ETH-USD","SOL-USD","ADA-USD","DOT-USD","LINK-USD","XRP-USD","LTC-USD","DOGE-USD","AVAX-USD"
]

ALL_ASSETS = ETFS + STOCKS + CRYPTOS

# map friendly names to common symbols (best-effort)
KNOWN_SYMBOLS = {
    "aapl": "AAPL", "msft":"MSFT", "amzn":"AMZN", "tsla":"TSLA", "nvda":"NVDA",
    "googl":"GOOGL", "meta":"META", "btc":"BTC-USD", "bitcoin":"BTC-USD", "eth":"ETH-USD"
}

# -----------------------
# util helpers
# -----------------------
def human_ts():
    return datetime.utcnow().isoformat() + "Z"

def deterministic_seed(s: str) -> int:
    return abs(hash(s)) % (2**31)

# -----------------------
# Alpha Vantage fetch (no requests req)
# -----------------------
def fetch_alpha_intraday(symbol: str, interval: str = "5min", outputsize: str = "compact"):
    """
    Returns list of OHLC dicts sorted oldest->newest:
    [{"t":datetime, "open":, "high":, "low":, "close":, "volume":}, ...]
    On error returns None.
    """
    if not ALPHA_KEY:
        return None
    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "apikey": ALPHA_KEY,
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
    ts_key = None
    for k in data.keys():
        if "Time Series" in k:
            ts_key = k
            break
    if not ts_key:
        return None
    try:
        items = []
        for t_str in sorted(data[ts_key].keys()):
            row = data[ts_key][t_str]
            o = float(row["1. open"]); h = float(row["2. high"]); l = float(row["3. low"]); c = float(row["4. close"])
            vol = float(row.get("5. volume", 0))
            try:
                t_dt = datetime.fromisoformat(t_str)
            except Exception:
                try:
                    t_dt = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except Exception:
                    t_dt = datetime.utcnow()
            items.append({"t": t_dt, "open": o, "high": h, "low": l, "close": c, "volume": vol})
        return sorted(items, key=lambda x: x["t"])
    except Exception:
        return None

def cache_path_for(symbol, interval):
    key = f"{symbol}_{interval}.json"
    return os.path.join(CACHE_DIR, key)

def cache_save(symbol, interval, candles):
    path = cache_path_for(symbol, interval)
    try:
        # convert datetimes to iso
        out = []
        for c in candles:
            o = dict(c)
            if isinstance(o.get("t"), datetime):
                o["t"] = o["t"].isoformat()
            out.append(o)
        save_json(path, {"ts": time.time(), "candles": out})
    except Exception:
        pass

def cache_load(symbol, interval, max_age_seconds=3600*24):
    path = cache_path_for(symbol, interval)
    if not os.path.exists(path):
        return None
    try:
        obj = load_json(path)
        if not obj:
            return None
        if time.time() - obj.get("ts", 0) > max_age_seconds:
            return None
        out = []
        for c in obj.get("candles", []):
            o = dict(c)
            if isinstance(o.get("t"), str):
                try:
                    o["t"] = datetime.fromisoformat(o["t"])
                except Exception:
                    o["t"] = datetime.utcnow()
            out.append(o)
        return out
    except Exception:
        return None

# -----------------------
# Offline deterministic generator
# -----------------------
def generate_price_walk(seed: str, steps: int, start_price: float = 100.0):
    rnd = random.Random(deterministic_seed(seed))
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
        if not chunk:
            continue
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk)
        ohlc.append({"t": None, "open": o, "high": h, "low": l, "close": c, "volume": 0})
    now_dt = datetime.utcnow()
    minutes = candle_size
    for i in range(len(ohlc)):
        ohlc[i]["t"] = now_dt - timedelta(minutes=(len(ohlc)-1-i)*minutes)
    return ohlc

# -----------------------
# Indicators (SMA, EMA, MACD, RSI, Bollinger)
# -----------------------
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
    k = 2.0 / (period + 1.0)
    ema_prev = None
    for v in values:
        if ema_prev is None:
            ema_prev = v
        else:
            ema_prev = v * k + ema_prev * (1.0 - k)
        res.append(ema_prev)
    return res

def macd(values, fast=12, slow=26, signal=9):
    if not values:
        return [], [], []
    ef = ema(values, fast)
    es = ema(values, slow)
    mac = []
    for a, b in zip(ef, es):
        mac.append((a - b) if (a is not None and b is not None) else None)
    # compute signal over mac values (filter Nones)
    mac_vals = [v for v in mac if v is not None]
    if not mac_vals:
        return mac, [None]*len(mac), [None]*len(mac)
    sig_vals = ema(mac_vals, signal)
    sig_iter = iter(sig_vals)
    sig_mapped = []
    for v in mac:
        if v is None:
            sig_mapped.append(None)
        else:
            sig_mapped.append(next(sig_iter))
    hist = [(m - s) if (m is not None and s is not None) else None for m, s in zip(mac, sig_mapped)]
    return mac, sig_mapped, hist

def rsi(values, period=14):
    if len(values) < period + 1:
        return [None]*len(values)
    deltas = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    res = [None] * period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        val = 100 - (100 / (1 + rs))
        res.append(round(val, 2))
    return res

def bollinger(values, period=20, mult=2.0):
    res = []
    for i in range(len(values)):
        if i+1 < period:
            res.append((None, None, None))
        else:
            window = values[i+1-period:i+1]
            m = sum(window) / period
            std = statistics.pstdev(window)
            up = m + mult * std
            low = m - mult * std
            res.append((round(m, 6), round(up, 6), round(low, 6)))
    return res

# -----------------------
# Pattern detection (candles)
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
    if len(candles) < 3:
        return False
    a, b, c = candles[-3], candles[-2], candles[-1]
    return (a["close"] > a["open"]) and (b["close"] > b["open"]) and (c["close"] > c["open"]) and (b["close"] > a["close"]) and (c["close"] > b["close"])

def detect_markers(candles):
    markers = []
    for i in range(1, len(candles)):
        cur = candles[i]
        prev = candles[i-1]
        if is_bullish_engulfing(prev, cur) or is_hammer(cur):
            markers.append({"idx": i, "type": "buy", "reason": "Bullish/Hammer"})
        if is_bearish_engulfing(prev, cur) or is_shooting_star(cur):
            markers.append({"idx": i, "type": "sell", "reason": "Bearish/Shooting Star"})
    for i in range(2, len(candles)):
        if is_three_white_soldiers(candles[:i+1]):
            markers.append({"idx": i, "type": "buy", "reason": "Three White Soldiers"})
    # deduplicate
    seen = set(); uniq = []
    for m in markers:
        k = (m["idx"], m["type"])
        if k not in seen:
            seen.add(k); uniq.append(m)
    return uniq

# -----------------------
# Stop-loss (dynamic) - volatility based
# -----------------------
def calculate_dynamic_stop(entry_price, candles, position_type="long"):
    closes = [c["close"] for c in candles[-30:]] if len(candles) >= 2 else [entry_price]
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))] if len(closes) > 1 else [0.0]
    vol = statistics.pstdev(returns) if len(returns) > 0 else 0.0
    # map vol to percentage (wider for higher vol), clamp 0.5%..15%
    recommended_pct = max(0.005, min(0.15, vol * 3.5))
    if position_type == "long":
        stop_price = entry_price * (1 - recommended_pct)
    else:
        stop_price = entry_price * (1 + recommended_pct)
    return round(stop_price, 6), round(recommended_pct, 4), vol

# -----------------------
# SVG renderer (candles + overlays + markers + stop)
# -----------------------
def render_candles_svg(candles, markers=None, stop_line=None, sma_periods=(20,50), rsi_vals=None, boll=None, width_px=1100, height_px=620):
    if markers is None:
        markers = []
    n = len(candles)
    if n == 0:
        return "<svg></svg>"
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    max_p = max(highs); min_p = min(lows)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad

    margin = 56
    chart_h = int(height_px * 0.62)
    rsi_h = int(height_px * 0.16)
    candle_w = max(3, (width_px - 2*margin) / n * 0.7)
    spacing = (width_px - 2*margin) / n

    def y_pos(p):
        return margin + chart_h - (p - min_p) / (max_p - min_p) * chart_h

    closes = [c["close"] for c in candles]
    sma1 = sma(closes, sma_periods[0]) if sma_periods and sma_periods[0] else []
    sma2 = sma(closes, sma_periods[1]) if sma_periods and sma_periods[1] else []

    svg = []
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#07070a"/>')

    # price grid
    for i in range(6):
        y = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')

    # candles
    for idx, c in enumerate(candles):
        x_center = margin + idx*spacing + spacing/2
        x_left = x_center - candle_w/2
        y_open = y_pos(c["open"]); y_close = y_pos(c["close"])
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        by = min(y_open, y_close)
        bh = max(1, abs(y_close - y_open))
        svg.append(f'<rect x="{x_left}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')

    # SMAs
    def polyline(vals, stroke, width=1.6):
        pts = []
        segs = []
        cur = []
        for i, v in enumerate(vals):
            if v is None:
                if cur:
                    segs.append(" ".join(cur))
                    cur = []
            else:
                x = margin + i*spacing + spacing/2
                y = y_pos(v)
                cur.append(f"{x},{y}")
        if cur:
            segs.append(" ".join(cur))
        out = []
        for s in segs:
            out.append(f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round"/>')
        return "\n".join(out)

    if sma1:
        svg.append(polyline(sma1, "#66ccff"))
    if sma2:
        svg.append(polyline(sma2, "#ffcc66"))

    # Bollinger shading
    if boll:
        up_pts = []
        low_pts = []
        for i, b in enumerate(boll):
            if b[1] is None:
                up_pts.append(None)
            else:
                x = margin + i*spacing + spacing/2
                up_pts.append(f"{x},{y_pos(b[1])}")
            if b[2] is None:
                low_pts.append(None)
            else:
                x = margin + i*spacing + spacing/2
                low_pts.append(f"{x},{y_pos(b[2])}")
        def segs_from(pts):
            segs = []; cur = []
            for p in pts:
                if p is None:
                    if cur: segs.append(cur); cur=[]
                else:
                    cur.append(p)
            if cur: segs.append(cur)
            return segs
        ups = segs_from(up_pts); lows = segs_from(low_pts)
        for u_seg, l_seg in zip(ups, lows):
            poly = " ".join(u_seg + l_seg[::-1])
            svg.append(f'<polygon points="{poly}" fill="#223333" opacity="0.22" stroke="none" />')

    # markers (shift offsets to reduce overlap)
    shift_px = max(6, min(30, int(200 / max(1, n))))
    used_counts = {}
    for m in markers:
        i = m["idx"]
        if i < 0 or i >= n:
            continue
        count = used_counts.get(i, 0)
        used_counts[i] = count + 1
        x_center = margin + i*spacing + spacing/2
        c = candles[i]
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        if m["type"] == "buy":
            y = y_high - (10 + count * (shift_px//2))
            color = "#00ff88"
            points = f"{x_center-7},{y} {x_center+7},{y} {x_center},{y+11}"
            svg.append(f'<polygon points="{points}" fill="{color}" opacity="0.98"><title>{m.get("reason","buy")}</title></polygon>')
        else:
            y = y_low + (10 + count * (shift_px//2))
            color = "#ff7788"
            points = f"{x_center-7},{y} {x_center+7},{y} {x_center},{y-11}"
            svg.append(f'<polygon points="{points}" fill="{color}" opacity="0.98"><title>{m.get("reason","sell")}</title></polygon>')

    # stop line
    if stop_line and stop_line.get("price") is not None:
        y_sp = y_pos(stop_line["price"])
        color = "#ffcc00" if stop_line.get("type") == "long" else "#ff4444"
        svg.append(f'<line x1="{margin}" y1="{y_sp}" x2="{width_px-margin}" y2="{y_sp}" stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>')
        label = f"Stop({stop_line.get('type')}) {stop_line['price']:.6f} ({stop_line['pct']*100:.2f}%)"
        svg.append(f'<rect x="{width_px-margin-320}" y="{y_sp-16}" width="320" height="20" fill="#101010" opacity="0.9"/>')
        svg.append(f'<text x="{width_px-margin-316}" y="{y_sp-2}" font-size="12" fill="{color}">{label}</text>')

    # RSI
    rsi_top = margin + chart_h + 12
    rsi_h = rsi_vals and int(height_px * 0.16) or 0
    if rsi_vals:
        svg.append(f'<rect x="{margin}" y="{rsi_top}" width="{width_px-2*margin}" height="{rsi_h}" fill="#0b0b0b" stroke="#111"/>')
        svg.append(f'<text x="{margin}" y="{rsi_top-4}" font-size="12" fill="#9aa6b2">RSI(14)</text>')
        pts = []
        segs = []
        cur = []
        for i, v in enumerate(rsi_vals):
            if v is None:
                if cur: segs.append(" ".join(cur)); cur=[]
            else:
                x = margin + i*spacing + spacing/2
                y = rsi_top + rsi_h - (v/100.0) * rsi_h
                cur.append(f"{x},{y}")
        if cur: segs.append(" ".join(cur))
        for s in segs:
            svg.append(f'<polyline points="{s}" fill="none" stroke="#ff66cc" stroke-width="1.2"/>')
        y30 = rsi_top + rsi_h - 0.3 * rsi_h
        y70 = rsi_top + rsi_h - 0.7 * rsi_h
        svg.append(f'<line x1="{margin}" y1="{y30}" x2="{width_px-margin}" y2="{y30}" stroke="#333" stroke-dasharray="3,3" />')
        svg.append(f'<line x1="{margin}" y1="{y70}" x2="{width_px-margin}" y2="{y70}" stroke="#333" stroke-dasharray="3,3" />')

    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        t = ""
        if candles[i].get("t"):
            try:
                t = candles[i]["t"].strftime("%Y-%m-%d %H:%M")
            except Exception:
                t = str(candles[i]["t"])
        svg.append(f'<text x="{x-40}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')

    svg.append('</svg>')
    return "\n".join(svg)

# -----------------------
# Perceptron (simple)
# -----------------------
class SimplePerceptron:
    def __init__(self, n):
        self.n = n
        self.weights = [random.uniform(-0.01, 0.01) for _ in range(n)]
        self.bias = random.uniform(-0.01, 0.01)
        self.lr = 0.01

    def raw(self, x):
        return sum(w*xi for w, xi in zip(self.weights, x)) + self.bias

    def predict(self, x):
        return 1 if self.raw(x) >= 0 else -1

    def train_epoch(self, X, y):
        errors = 0
        for xi, yi in zip(X, y):
            pred = self.predict(xi)
            if pred != yi:
                errors += 1
                for j in range(self.n):
                    self.weights[j] += self.lr * yi * xi[j]
                self.bias += self.lr * yi
        return errors

    def train(self, X, y, epochs=50):
        for e in range(epochs):
            err = self.train_epoch(X, y)
            if err == 0:
                break

    def export(self):
        return {"n": self.n, "weights": self.weights, "bias": self.bias}

    def load(self, obj):
        self.n = obj["n"]; self.weights = obj["weights"]; self.bias = obj["bias"]

def load_model():
    m = load_json(MODEL_FILE)
    if m:
        try:
            p = SimplePerceptron(m.get("n", 6))
            p.load(m)
            return p
        except Exception:
            pass
    return SimplePerceptron(6)

def save_model(p):
    save_json(MODEL_FILE, p.export())

# -----------------------
# Features for ML
# -----------------------
def features_from_candles(candles):
    closes = [c["close"] for c in candles]
    if len(closes) < 60:
        # pad
        while len(closes) < 60:
            closes.insert(0, closes[0] if closes else 100.0)
    last = closes[-1]
    ret1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0.0
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma20
    sma_diff = (sma20 - sma50) / last if last != 0 else 0.0
    rsi_vals = rsi(closes, 14)
    rsi_last = rsi_vals[-1] if rsi_vals and rsi_vals[-1] is not None else 50.0
    macd_line, sig, hist = macd(closes, 12, 26, 9)
    macd_last = macd_line[-1] if macd_line and macd_line[-1] is not None else 0.0
    vol = statistics.pstdev([(closes[i] - closes[i-1])/closes[i-1] for i in range(1, len(closes))]) if len(closes) > 1 else 0.0
    highs = [c["high"] for c in candles[-14:]] if len(candles) >= 14 else [c["high"] for c in candles]
    lows = [c["low"] for c in candles[-14:]] if len(candles) >= 14 else [c["low"] for c in candles]
    atr = sum([h - l for h, l in zip(highs, lows)]) / len(highs) if highs else 0.0
    return [ret1, sma_diff, rsi_last/100.0, macd_last, vol, atr/last if last != 0 else 0.0]

# -----------------------
# Backtest engine
# -----------------------
def calculate_performance_from_trades(trades, equity_curve, initial_cash):
    total_return = (equity_curve[-1] - initial_cash) / initial_cash if initial_cash else 0.0
    returns = []
    for i in range(1, len(equity_curve)):
        if equity_curve[i-1] != 0:
            returns.append((equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1])
    avg_ret = statistics.mean(returns) if returns else 0.0
    std_ret = statistics.pstdev(returns) if len(returns) > 1 else 0.0
    sharpe_like = (avg_ret / std_ret) if std_ret != 0 else 0.0
    peak = equity_curve[0] if equity_curve else initial_cash
    max_dd = 0.0
    for v in equity_curve:
        if v > peak: peak = v
        dd = (peak - v) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    winners = [t for t in trades if t.get("pnl", 0) > 0]
    winrate = len(winners)/len(trades) if trades else 0.0
    return {"total_return": total_return, "sharpe_like": sharpe_like, "max_dd": max_dd, "winrate": winrate}

def backtest_strategy(candles, strategy_fn, initial_cash=10000.0):
    cash = initial_cash
    position = None
    trades = []
    equity_curve = []
    for i in range(len(candles)):
        price = candles[i]["close"]
        signal = strategy_fn(candles, i)
        # check position exit
        if position:
            if position["direction"] == "long":
                if price <= position["stop"]:
                    pnl = (position["stop"] - position["entry_price"]) * position["qty"]
                    cash += position["qty"] * position["stop"]
                    trades.append({**position, "exit_price": position["stop"], "exit_index": i, "pnl": pnl})
                    position = None
                elif price >= position["tp"]:
                    pnl = (position["tp"] - position["entry_price"]) * position["qty"]
                    cash += position["qty"] * position["tp"]
                    trades.append({**position, "exit_price": position["tp"], "exit_index": i, "pnl": pnl})
                    position = None
            else:
                if price >= position["stop"]:
                    pnl = (position["entry_price"] - position["stop"]) * position["qty"]
                    cash += position["qty"] * (position["entry_price"] - position["stop"])
                    trades.append({**position, "exit_price": position["stop"], "exit_index": i, "pnl": pnl})
                    position = None
                elif price <= position["tp"]:
                    pnl = (position["entry_price"] - position["tp"]) * position["qty"]
                    cash += position["qty"] * (position["entry_price"] - position["tp"])
                    trades.append({**position, "exit_price": position["tp"], "exit_index": i, "pnl": pnl})
                    position = None
        if signal and not position:
            direction = "long" if signal == 1 else "short"
            risk_capital = 0.1 * (cash if cash > 0 else initial_cash)
            qty = risk_capital / price if price > 0 else 0.0
            stop_price, stop_pct, vol = calculate_dynamic_stop(price, candles[:i+1], position_type=direction)
            tp = price + (price - stop_price) * 2 if direction == "long" else price - (stop_price - price) * 2
            position = {"entry_price": price, "qty": qty, "direction": direction, "stop": stop_price, "tp": tp, "entry_index": i}
            if direction == "long":
                cash -= qty * price
            trades.append({**position, "opened_index": i})
        if position:
            if position["direction"] == "long":
                cur_val = cash + position["qty"] * price
            else:
                cur_val = cash + (position["entry_price"] - price) * position["qty"]
        else:
            cur_val = cash
        equity_curve.append(cur_val)
    if position:
        price = candles[-1]["close"]
        if position["direction"] == "long":
            pnl = (price - position["entry_price"]) * position["qty"]
            cash += position["qty"] * price
        else:
            pnl = (position["entry_price"] - price) * position["qty"]
            cash += position["qty"] * (position["entry_price"] - price)
        trades.append({**position, "exit_price": price, "exit_index": len(candles)-1, "pnl": pnl})
        equity_curve.append(cash)
    metrics = calculate_performance_from_trades(trades, equity_curve, initial_cash)
    metrics["trades"] = trades
    metrics["equity_curve"] = equity_curve
    metrics["final_cash"] = cash
    return metrics

# -----------------------
# Strategy examples
# -----------------------
def strategy_sma_cross(candles, i):
    closes = [c["close"] for c in candles[:i+1]]
    if len(closes) < 51:
        return 0
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
    if len(closes) < 35:
        return 0
    m, s, h = macd(closes, 12, 26, 9)
    if len(m) < 2 or m[-1] is None or s[-1] is None or m[-2] is None or s[-2] is None:
        return 0
    if m[-1] > s[-1] and m[-2] <= s[-2]:
        return 1
    if m[-1] < s[-1] and m[-2] >= s[-2]:
        return -1
    return 0

# -----------------------
# Analysis & success estimate
# -----------------------
def analyze_and_estimate(candles, perceptron_model):
    # produce signals from rules
    markers = detect_markers(candles)
    closes = [c["close"] for c in candles]
    s20 = sma(closes, 20)
    s50 = sma(closes, 50)
    macd_line, macd_sig, macd_hist = macd(closes, 12, 26, 9)
    rsi_vals = rsi(closes, 14)
    boll = bollinger(closes, 20, 2.0)
    # determine simple rule-based recommendation
    score = 0
    last = candles[-1]
    prev = candles[-2] if len(candles) > 1 else None
    if is_hammer(last): score += 1
    if prev and is_bullish_engulfing(prev, last): score += 2
    if is_three_white_soldiers(candles): score += 2
    if is_shooting_star(last): score -= 1
    if prev and is_bearish_engulfing(prev, last): score -= 2
    # sma trend
    if s20[-1] and s50[-1]:
        if s20[-1] > s50[-1]:
            score += 1
        else:
            score -= 1
    # macd momentum
    if macd_line and macd_sig and macd_line[-1] is not None and macd_sig[-1] is not None:
        if macd_line[-1] > macd_sig[-1]:
            score += 1
        else:
            score -= 1
    # perceptron model predicts +1 or -1
    feat = features_from_candles(candles)
    ml_signal = perceptron_model.predict(feat) if perceptron_model else 0
    # blend signals
    base_prob = 0.5 + (score / 8.0)  # score roughly -6..+6 -> scale
    ml_bias = 0.15 * (1 if ml_signal == 1 else -1)
    prob = min(max(base_prob + ml_bias, 0.01), 0.99)
    # if backtest info available from previous runs, we could blend with winrate, but we'll produce estimate
    # risk pct from volatility-based stop of last close
    stop_price, stop_pct, vol = calculate_dynamic_stop(closes[-1], candles, position_type="long")
    risk_pct = stop_pct * 100.0
    # Recommendation: if prob > 0.65 -> buy/long, if prob < 0.35 -> short, else hold
    if prob >= 0.65:
        rec = "Kaufen (Long erwartet zu steigen)"
    elif prob <= 0.35:
        rec = "Short / Verkaufen (Erwarteter Fall — verkaufen oder short gehen)"
    else:
        rec = "Halten / Beobachten"
    # map to "success rate approx 80%" style: we present prob as percentage
    success_est = round(prob * 100.0, 1)
    return {
        "markers": markers,
        "sma20": s20,
        "sma50": s50,
        "macd": (macd_line, macd_sig, macd_hist),
        "rsi": rsi_vals,
        "boll": boll,
        "recommendation": rec,
        "success_pct": success_est,
        "risk_pct": round(risk_pct, 2),
        "prob": prob,
        "ml_signal": ml_signal
    }

# -----------------------
# UI + Main
# -----------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_json(PORTFOLIO_FILE) or []

if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE) or []

if "cache" not in st.session_state:
    st.session_state.cache = {}

if "model" not in st.session_state:
    st.session_state.model = load_model()

left_col, right_col = st.columns([3, 1])
with right_col:
    st.header("Kontrollen")
    symbol_in = st.text_input("Symbol oder Asset-Name", value="AAPL")
    interval = st.selectbox("Intervall", ["1min","5min","15min","30min","60min"], index=1)
    periods = st.slider("Anzahl Kerzen", 30, 1000, 240, step=10)
    start_price = st.number_input("Startpreis (Fallback)", value=100.0, step=0.1)
    use_live = st.checkbox("Live via Alpha Vantage (wenn Key & Limit ok)", value=True)
    fetch_live = st.button("Fetch Live / Refresh")
    refresh_sim = st.button("Refresh Simulation")
    custom_manual_stop = st.slider("Manueller Stop-Loss (%) (0 = auto)", 0.0, 20.0, 0.0, step=0.1)
    st.markdown("---")
    st.subheader("Model / ML")
    if st.button("Train Perceptron (synthetic)"):
        st.info("Training Perceptron on simulated labeled patterns...")
        X=[]; y=[]
        for i in range(800):
            base = generate_price_walk("train"+str(i), 300, 100.0)
            ohlc = prices_to_ohlc(base, candle_size=5)
            label = random.choice([1, -1])
            if label == 1:
                # enforce bullish end
                for k in range(3):
                    ohlc[-1-k]["open"] = ohlc[-1-k]["close"] * (1 - random.uniform(0.01,0.03))
                    ohlc[-1-k]["close"] = ohlc[-1-k]["open"] * (1 + random.uniform(0.01,0.05))
            else:
                for k in range(3):
                    ohlc[-1-k]["open"] = ohlc[-1-k]["close"] * (1 + random.uniform(0.01,0.04))
                    ohlc[-1-k]["close"] = ohlc[-1-k]["open"] * (1 - random.uniform(0.01,0.05))
            feat = features_from_candles(ohlc)
            X.append(feat); y.append(label)
        p = SimplePerceptron(len(X[0]))
        p.train(X, y, epochs=80)
        save_model(p)
        st.session_state.model = p
        st.success("Perceptron trained and saved.")
    st.markdown("---")
    st.subheader("Portfolio")
    add_qty = st.number_input("Menge für Schnellkauf", value=1.0, min_value=0.0, step=0.1)
    if st.button("Export Portfolio"):
        st.download_button("Download JSON", data=json.dumps(st.session_state.portfolio, ensure_ascii=False, indent=2), file_name="portfolio.json", mime="application/json")
    if st.button("Clear Portfolio"):
        st.session_state.portfolio = []
        save_json(PORTFOLIO_FILE, st.session_state.portfolio)
        st.success("Portfolio geleert")
    st.markdown("---")
    st.write("AlphaV key:", "vorhanden" if ALPHA_KEY else "nicht gesetzt")
    st.write("Hinweis: Alpha Vantage free tier: 5 calls/min, 500 calls/day.")

with left_col:
    st.header("Chart & Analyzer")
    sym = symbol_in.strip()
    # map known friendly names
    if sym.lower() in KNOWN_SYMBOLS:
        sym_for_api = KNOWN_SYMBOLS[sym.lower()]
    else:
        sym_for_api = sym
    cache_key = f"{sym_for_api}|{interval}|{periods}|{start_price}"
    candles = None
    used_live = False

    if fetch_live and use_live and ALPHA_KEY:
        fetched = fetch_alpha_intraday(sym_for_api, interval=interval, outputsize="compact")
        if fetched and len(fetched) >= periods:
            candles = fetched[-periods:]
            cache_save(sym_for_api, interval, candles)
            used_live = True
            st.success("Live-Daten geladen (Alpha Vantage).")
        else:
            st.warning("Live-Fetch fehlgeschlagen oder zu wenig Daten — fallback auf Cache/Simulation.")
            cached = cache_load(sym_for_api, interval)
            if cached and len(cached) >= periods:
                candles = cached[-periods:]
                st.info("Verwende gecachte Live-Daten.")
    if not candles:
        cached = cache_load(sym_for_api, interval)
        if cached and len(cached) >= periods and not refresh_sim:
            candles = cached[-periods:]
            st.info("Verwende zwischengespeicherte Daten.")
    if not candles or refresh_sim:
        # deterministic simulation
        tf_map = {"1min":1, "5min":5, "15min":15, "30min":30, "60min":60}
        mins = periods * tf_map.get(interval, 5)
        prices = generate_price_walk(sym_for_api + "|" + interval, mins, start_price)
        ohlc = prices_to_ohlc(prices, candle_size=tf_map.get(interval, 5))
        if len(ohlc) < periods:
            pad = periods - len(ohlc)
            pad_item = ohlc[0] if ohlc else {"open": start_price, "high": start_price, "low": start_price, "close": start_price, "volume":0}
            ohlc = [pad_item]*pad + ohlc
        candles = ohlc[-periods:]
        st.session_state.cache[cache_key] = candles
        if refresh_sim:
            st.success("Simulation neu erzeugt.")

    # compute indicators
    closes = [c["close"] for c in candles]
    sma20 = sma(closes, 20)
    sma50 = sma(closes, 50)
    macd_line, macd_sig, macd_hist = macd(closes, 12, 26, 9)
    rsi_vals = rsi(closes, 14)
    boll = bollinger(closes, 20, 2.0)
    markers = detect_markers(candles)

    # analysis & estimate
    model = st.session_state.model
    analysis = analyze_and_estimate(candles, model)
    stop_auto, stop_pct_auto, vol = calculate_dynamic_stop(closes[-1], candles, position_type="long")
    if custom_manual_stop > 0.0:
        manual_pct = custom_manual_stop/100.0
        stop_line = {"price": round(closes[-1] * (1 - manual_pct), 6), "type": "long", "pct": manual_pct}
    else:
        stop_line = {"price": stop_auto, "type": "long", "pct": stop_pct_auto}

    # render svg
    svg = render_candles_svg(candles, markers=analysis["markers"], stop_line=stop_line, sma_periods=(20,50), rsi_vals=analysis["rsi"], boll=analysis["boll"], width_px=1100, height_px=620)
    st.components.v1.html(svg, height=640)

    # results
    st.subheader("Empfehlung & Schätzung")
    st.markdown(f"**Empfehlung:** {analysis['recommendation']}")
    st.markdown(f"**Geschätzte Erfolgswahrscheinlichkeit:** **{analysis['success_pct']}%**  (Schätzung, keine Garantie)")
    st.markdown(f"**Risiko (empf. Stop-Loss-Abstand):** **{analysis['risk_pct']}%**")
    st.markdown(f"**ML-Signal (Perceptron):** {'BUY' if analysis['ml_signal']==1 else 'SELL' if analysis['ml_signal']==-1 else 'NEUTRAL'}")
    if analysis.get("reasons"):
        st.write("Gründe / erkannte Muster:")
        for r in analysis.get("reasons", []):
            st.write("- " + r)
    # quick info about stop-loss / explanation
    st.markdown("**Was ist Stop-Loss?**")
    st.write("Stop-Loss ist eine automatische Order, die deine Position schließt, um Verluste zu begrenzen. Hier wird der Stop basierend auf der kurzfristigen Volatilität empfohlen.")
    st.markdown("---")

    # quick trade simulator / add to portfolio
    st.subheader("Schnellaktionen")
    c1, c2 = st.columns(2)
    with c1:
        qty = st.number_input("Menge (für Trade)", min_value=0.0, value=add_qty if 'add_qty' in locals() else 1.0, step=0.1)
        if st.button("Sim. Trade eröffnen (Market)"):
            entry = closes[-1]
            direction = 1 if analysis["prob"] >= 0.5 else -1
            dir_str = "long" if direction==1 else "short"
            stop = stop_line["price"]
            tp = round(entry + (entry - stop) * 2, 6) if direction==1 else round(entry - (stop - entry) * 2, 6)
            trade = {"id": f"sim_{int(time.time())}", "symbol": sym_for_api, "direction": dir_str, "qty": qty, "entry": entry, "stop": stop, "tp": tp, "opened": human_ts()}
            st.session_state.portfolio.append(trade)
            save_json(PORTFOLIO_FILE, st.session_state.portfolio)
            st.success(f"Simulierter Trade: {dir_str} {sym_for_api} qty {qty} entry {entry:.6f} stop {stop:.6f} tp {tp:.6f}")
    with c2:
        if st.button("Zum Portfolio hinzufügen (nur Speicherung)"):
            item = {"symbol": sym_for_api, "qty": qty, "entry": closes[-1], "position": "auto", "stop": stop_line["price"], "added": human_ts()}
            st.session_state.portfolio.append(item)
            save_json(PORTFOLIO_FILE, st.session_state.portfolio)
            st.success("Position gespeichert.")

    st.markdown("---")
    st.subheader("Portfolio (Sim / Saved)")
    if st.session_state.portfolio:
        total_est = 0.0
        for t in st.session_state.portfolio:
            pid = str(t.get("symbol", t.get("id", "X")))
            tmp_prices = generate_price_walk(pid + "|p", 24, t.get("entry", closes[-1]))
            cur = tmp_prices[-1]
            if t.get("direction","long") == "long" or t.get("position","long") == "long":
                val = cur * t.get("qty", 0.0)
                pnl = (cur - t.get("entry", cur)) * t.get("qty", 0.0)
            else:
                val = (t.get("entry", cur) - cur) * t.get("qty", 0.0)
                pnl = (t.get("entry", cur) - cur) * t.get("qty", 0.0)
            total_est += val
            st.write(f"- {t.get('symbol')} | qty {t.get('qty')} | entry {t.get('entry'):.6f} | cur est {cur:.6f} | P&L {pnl:.4f}")
        st.write(f"Geschätzter Portfoliowert (sim): {total_est:.2f}")
    else:
        st.info("Portfolio leer — füge simulierte Trades hinzu.")

# Footer
st.markdown("---")
st.caption("Analyzer Deluxe Final — Live via Alpha Vantage when available; otherwise deterministic offline simulation. Erfolgswahrscheinlichkeit ist eine Schätzung basierend auf Muster-Score und ML-Signal — keine Anlageberatung.")
