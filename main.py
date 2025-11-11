# main.py
# Lumina Pro — Deep Analyzer (Live Alpha Vantage + Roboflow Image Inference)
# Keys directly embedded per user request:
#   ALPHA_KEY  = "22XGVO0TQ1UV167C"
#   ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
#
# Save as main.py and run: streamlit run main.py

import streamlit as st
import json, os, time, random, math, urllib.request, urllib.parse, mimetypes, io
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Any

# optional PIL
try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Lumina Pro — Deep Analyzer", layout="wide")
st.markdown("""<style>
body { background:#000; color:#e6eef6; }
.stButton>button { background:#111; color:#e6eef6; border:1px solid #222; }
</style>""", unsafe_allow_html=True)
st.title("Lumina Pro — Deep Analyzer")

# -----------------------
# API KEYS (embedded)
# -----------------------
ALPHA_KEY = "22XGVO0TQ1UV167C"
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"

# Roboflow: you may need to set your model path like "your-model/1"
# If you have a specific Roboflow model endpoint, replace MODEL_PATH below:
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"  # <-- adjust if needed

# -----------------------
# FILES & CACHE
# -----------------------
PORTFOLIO_FILE = "portfolio.json"
CACHE_DIR = ".cache_av"
MODEL_FILE = "perceptron_model.json"

def ensure_file(path, default):
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(default, f, indent=2, ensure_ascii=False)

ensure_file(PORTFOLIO_FILE, [])
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
# UTILITIES
# -----------------------
def human_ts(): return datetime.utcnow().isoformat() + "Z"
def deterministic_seed(s: str) -> int: return abs(hash(s)) % (2**31)

def internet_ok():
    try:
        urllib.request.urlopen("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False

ONLINE = internet_ok() and bool(ALPHA_KEY)

# -----------------------
# ALPHA VANTAGE FETCH
# -----------------------
def fetch_alpha_intraday(symbol: str, interval: str = "5min", outputsize: str = "compact"):
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
        with urllib.request.urlopen(url, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
    except Exception:
        return None
    ts_key = None
    for k in data.keys():
        if "Time Series" in k:
            ts_key = k; break
    if not ts_key:
        return None
    try:
        items=[]
        for t_str in sorted(data[ts_key].keys()):
            row = data[ts_key][t_str]
            o=float(row["1. open"]); h=float(row["2. high"]); l=float(row["3. low"]); c=float(row["4. close"])
            vol=float(row.get("5. volume",0))
            try: t_dt=datetime.fromisoformat(t_str)
            except:
                try: t_dt=datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S")
                except: t_dt=datetime.utcnow()
            items.append({"t":t_dt,"open":o,"high":h,"low":l,"close":c,"volume":vol})
        return sorted(items, key=lambda x:x["t"])
    except Exception:
        return None

# caching helpers
def cache_path_for(symbol, interval): return os.path.join(CACHE_DIR, f"{symbol}_{interval}.json")
def cache_save(symbol, interval, candles):
    path = cache_path_for(symbol, interval)
    try:
        out=[]
        for c in candles:
            o = dict(c)
            if isinstance(o.get("t"), datetime): o["t"] = o["t"].isoformat()
            out.append(o)
        save_json(path, {"ts":time.time(), "candles": out})
    except Exception:
        pass

def cache_load(symbol, interval, max_age_seconds=3600*24):
    path = cache_path_for(symbol, interval)
    if not os.path.exists(path):
        return None
    try:
        obj = load_json(path)
        if not obj: return None
        if time.time() - obj.get("ts",0) > max_age_seconds:
            return None
        out=[]
        for c in obj.get("candles", []):
            o = dict(c)
            if isinstance(o.get("t"), str):
                try: o["t"] = datetime.fromisoformat(o["t"])
                except: o["t"] = datetime.utcnow()
            out.append(o)
        return out
    except Exception:
        return None

# -----------------------
# OFFLINE SIMULATOR
# -----------------------
def generate_price_walk(seed: str, steps: int, start_price: float = 100.0):
    rnd = random.Random(deterministic_seed(seed))
    price = float(start_price)
    series = []
    for _ in range(steps):
        drift = (rnd.random() - 0.49) * 0.003
        shock = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + shock))
        series.append(round(price,6))
    return series

def prices_to_ohlc(prices, candle_size=1):
    ohlc=[]
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

# -----------------------
# INDICATORS
# -----------------------
def sma(vals, period):
    res=[]
    for i in range(len(vals)):
        if i+1 < period: res.append(None)
        else: res.append(sum(vals[i+1-period:i+1]) / period)
    return res

def ema(vals, period):
    res=[]; k = 2.0/(period+1.0); prev=None
    for v in vals:
        if prev is None: prev = v
        else: prev = v * k + prev * (1-k)
        res.append(prev)
    return res

def macd(vals, fast=12, slow=26, signal=9):
    if not vals: return [],[],[]
    ef = ema(vals, fast); es = ema(vals, slow)
    mac = []
    for a,b in zip(ef, es):
        mac.append((a-b) if (a is not None and b is not None) else None)
    mac_vals = [v for v in mac if v is not None]
    if not mac_vals: return mac, [None]*len(mac), [None]*len(mac)
    sig_vals = ema(mac_vals, signal)
    sig_iter = iter(sig_vals)
    sig_mapped = []
    for v in mac:
        sig_mapped.append(None if v is None else next(sig_iter))
    hist = [(m-s) if (m is not None and s is not None) else None for m,s in zip(mac, sig_mapped)]
    return mac, sig_mapped, hist

def rsi(values, period=14):
    if len(values) < period+1: return [None]*len(values)
    deltas = [values[i] - values[i-1] for i in range(1, len(values))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    res = [None]*period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        val = 100 - (100 / (1 + rs))
        res.append(round(val,2))
    return res

def bollinger(values, period=20, mult=2.0):
    res=[]
    for i in range(len(values)):
        if i+1 < period: res.append((None,None,None))
        else:
            window = values[i+1-period:i+1]
            m = sum(window)/period; std = statistics.pstdev(window)
            res.append((round(m,6), round(m+mult*std,6), round(m-mult*std,6)))
    return res

def atr(candles, period=14):
    trs=[]
    for i in range(1, len(candles)):
        tr = max(candles[i]["high"] - candles[i]["low"], abs(candles[i]["high"] - candles[i-1]["close"]), abs(candles[i]["low"] - candles[i-1]["close"]))
        trs.append(tr)
    if not trs: return [None]*len(candles)
    res=[None]*(1+len(trs))
    for i in range(period, len(trs)+1):
        window = trs[i-period:i]
        res.append(sum(window)/period)
    return res[:len(candles)]

# -----------------------
# PATTERNS (candles)
# -----------------------
def is_doji(c):
    body = abs(c["close"] - c["open"])
    total = c["high"] - c["low"]
    return total > 0 and (body / total) < 0.15

def is_hammer(c):
    body = abs(c["close"] - c["open"])
    lower = min(c["open"], c["close"]) - c["low"]
    return body > 0 and lower > 2 * body

def is_inverted_hammer(c):
    body = abs(c["close"] - c["open"])
    upper = c["high"] - max(c["open"], c["close"])
    return body > 0 and upper > 2 * body

def is_shooting_star(c): return is_inverted_hammer(c)

def is_bullish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])

def is_bearish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def is_three_white_soldiers(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"]>a["open"]) and (b["close"]>b["open"]) and (c["close"]>c["open"]) and (b["close"]>a["close"]) and (c["close"]>b["close"])

def is_morning_star(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"] < a["open"]) and (is_doji(b) or (b["close"] < b["open"])) and (c["close"] > c["open"]) and c["close"] > (a["close"] + a["open"]) / 2

def is_evening_star(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"] > a["open"]) and (is_doji(b) or (b["close"] > b["open"])) and (c["close"] < c["open"]) and c["close"] < (a["close"] + a["open"]) / 2

def detect_markers(candles):
    markers=[]
    for i in range(1, len(candles)):
        cur = candles[i]; prev = candles[i-1]
        if is_bullish_engulfing(prev, cur): markers.append({"idx":i,"type":"buy","reason":"Bullish Engulfing"})
        if is_bearish_engulfing(prev, cur): markers.append({"idx":i,"type":"sell","reason":"Bearish Engulfing"})
        if is_hammer(cur): markers.append({"idx":i,"type":"buy","reason":"Hammer"})
        if is_shooting_star(cur): markers.append({"idx":i,"type":"sell","reason":"Shooting Star"})
    if is_three_white_soldiers(candles): markers.append({"idx":len(candles)-1,"type":"buy","reason":"3 White Soldiers"})
    if is_morning_star(candles): markers.append({"idx":len(candles)-1,"type":"buy","reason":"Morning Star"})
    if is_evening_star(candles): markers.append({"idx":len(candles)-1,"type":"sell","reason":"Evening Star"})
    # dedupe
    seen=set(); uniq=[]
    for m in markers:
        k=(m["idx"],m["type"],m["reason"])
        if k not in seen: seen.add(k); uniq.append(m)
    return uniq

# -----------------------
# SUPPORT / RESISTANCE (naive)
# -----------------------
def detect_support_resistance(candles, window=20):
    closes=[c["close"] for c in candles]
    supports=[]; resistances=[]
    for i in range(2, len(closes)-2):
        window_vals = closes[max(0,i-window):i+window]
        v = closes[i]
        if v == min(window_vals): supports.append((i,v))
        if v == max(window_vals): resistances.append((i,v))
    supports = sorted(supports, key=lambda x: x[1])[:3]
    resistances = sorted(resistances, key=lambda x: -x[1])[:3]
    return supports, resistances

# -----------------------
# STOP LOSS CALC
# -----------------------
def calculate_dynamic_stop(entry_price, candles, position_type="long"):
    closes=[c["close"] for c in candles[-30:]] if len(candles)>=2 else [entry_price]
    returns=[(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))] if len(closes)>1 else [0.0]
    vol = statistics.pstdev(returns) if len(returns)>0 else 0.0
    recommended_pct = max(0.005, min(0.15, vol * 3.5))
    if position_type=="long":
        stop_price = entry_price * (1 - recommended_pct)
    else:
        stop_price = entry_price * (1 + recommended_pct)
    return round(stop_price,6), round(recommended_pct,4), vol

# -----------------------
# SVG RENDERER FOR CANDLES
# -----------------------
def render_candles_svg(candles, markers=None, stop_line=None, sma_periods=(20,50), rsi_vals=None, boll=None, supports=None, resistances=None, width_px=1100, height_px=620):
    if markers is None: markers=[]
    n=len(candles)
    if n==0: return "<svg></svg>"
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]
    max_p=max(highs); min_p=min(lows)
    pad=(max_p-min_p)*0.08 if (max_p-min_p)>0 else 1.0
    max_p+=pad; min_p-=pad
    margin=56; chart_h=int(height_px*0.62); rsi_h=int(height_px*0.16)
    candle_w = max(3, (width_px-2*margin)/n * 0.7)
    spacing = (width_px-2*margin)/n
    def y_pos(p): return margin + chart_h - (p-min_p)/(max_p-min_p)*chart_h
    closes=[c["close"] for c in candles]
    sma1 = sma(closes, sma_periods[0]) if sma_periods and sma_periods[0] else []
    sma2 = sma(closes, sma_periods[1]) if sma_periods and sma_periods[1] else []
    svg=[]
    svg.append(f'<svg width="{width_px}" height="{height_px}" xmlns="http://www.w3.org/2000/svg">')
    svg.append(f'<rect x="0" y="0" width="{width_px}" height="{height_px}" fill="#07070a"/>')
    # grid
    for i in range(6):
        y = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')
    # candles
    for idx,c in enumerate(candles):
        x_center = margin + idx*spacing + spacing/2
        x_left = x_center - candle_w/2
        y_open = y_pos(c["open"]); y_close = y_pos(c["close"])
        y_high = y_pos(c["high"]); y_low = y_pos(c["low"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{x_center}" y1="{y_high}" x2="{x_center}" y2="{y_low}" stroke="#888" stroke-width="1"/>')
        by=min(y_open,y_close); bh=max(1, abs(y_close-y_open))
        svg.append(f'<rect x="{x_left}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
    # SMAs
    def polyline(vals, stroke, width=1.6):
        segs=[]; cur=[]
        for i,v in enumerate(vals):
            if v is None:
                if cur: segs.append(" ".join(cur)); cur=[]
            else:
                x=margin+i*spacing+spacing/2; y=y_pos(v); cur.append(f"{x},{y}")
        if cur: segs.append(" ".join(cur))
        return "\n".join([f'<polyline points="{s}" fill="none" stroke="{stroke}" stroke-width="{width}" />' for s in segs])
    if sma1: svg.append(polyline(sma1, "#66ccff",1.8))
    if sma2: svg.append(polyline(sma2, "#ffcc66",1.8))
    # boll shading
    if boll:
        ups=[]; lows=[]
        for i,b in enumerate(boll):
            if b[1] is None: ups.append(None)
            else: ups.append(f'{margin+i*spacing+spacing/2},{y_pos(b[1])}')
            if b[2] is None: lows.append(None)
            else: lows.append(f'{margin+i*spacing+spacing/2},{y_pos(b[2])}')
        def segs(pts):
            segs=[]; cur=[]
            for p in pts:
                if p is None:
                    if cur: segs.append(cur); cur=[]
                else: cur.append(p)
            if cur: segs.append(cur)
            return segs
        ups_s = segs(ups); lows_s = segs(lows)
        for u,l in zip(ups_s,lows_s):
            poly = " ".join(u + l[::-1])
            svg.append(f'<polygon points="{poly}" fill="#223333" opacity="0.2"/>')
    # markers
    used_counts={}
    for m in markers:
        i=m["idx"]
        if i<0 or i>=n: continue
        count = used_counts.get(i,0); used_counts[i]=count+1
        x_center = margin + i*spacing + spacing/2
        c = candles[i]; y_high=y_pos(c["high"]); y_low=y_pos(c["low"])
        if m["type"]=="buy":
            y = y_high - (10 + count*6)
            svg.append(f'<polygon points="{x_center-7},{y} {x_center+7},{y} {x_center},{y+11}" fill="#00ff88" opacity="0.98"><title>{m["reason"]}</title></polygon>')
        else:
            y = y_low + (10 + count*6)
            svg.append(f'<polygon points="{x_center-7},{y} {x_center+7},{y} {x_center},{y-11}" fill="#ff7788" opacity="0.98"><title>{m["reason"]}</title></polygon>')
    # stop line
    if stop_line and stop_line.get("price") is not None:
        y_sp = y_pos(stop_line["price"])
        color = "#ffcc00" if stop_line.get("type")=="long" else "#ff4444"
        svg.append(f'<line x1="{margin}" y1="{y_sp}" x2="{width_px-margin}" y2="{y_sp}" stroke="{color}" stroke-width="2" stroke-dasharray="6,4"/>')
        label = f"Stop({stop_line.get('type')}) {stop_line['price']:.6f} ({stop_line['pct']*100:.2f}%)"
        svg.append(f'<rect x="{width_px-margin-320}" y="{y_sp-16}" width="320" height="20" fill="#101010" opacity="0.9"/>')
        svg.append(f'<text x="{width_px-margin-316}" y="{y_sp-2}" font-size="12" fill="{color}">{label}</text>')
    # supports / resistances
    if supports:
        for idx,val in supports:
            x = margin + idx*spacing + spacing/2
            y = y_pos(val)
            svg.append(f'<line x1="{x-6}" y1="{y}" x2="{x+6}" y2="{y}" stroke="#44ffbb" stroke-width="2"/>')
    if resistances:
        for idx,val in resistances:
            x = margin + idx*spacing + spacing/2
            y = y_pos(val)
            svg.append(f'<line x1="{x-6}" y1="{y}" x2="{x+6}" y2="{y}" stroke="#ff6666" stroke-width="2"/>')
    # RSI area
    rsi_top = margin + chart_h + 12
    rsi_h = int(height_px*0.16)
    if rsi_vals:
        svg.append(f'<rect x="{margin}" y="{rsi_top}" width="{width_px-2*margin}" height="{rsi_h}" fill="#0b0b0b" stroke="#111"/>')
        pts=[]; cur=[]
        for i,v in enumerate(rsi_vals):
            if v is None:
                if cur: pts.append(" ".join(cur)); cur=[]
            else:
                x = margin + i*spacing + spacing/2; y = rsi_top + rsi_h - (v/100.0)*rsi_h
                cur.append(f"{x},{y}")
        if cur: pts.append(" ".join(cur))
        for s in pts: svg.append(f'<polyline points="{s}" fill="none" stroke="#ff66cc" stroke-width="1.2" />')
        y30 = rsi_top + rsi_h - 0.3*rsi_h; y70 = rsi_top + rsi_h - 0.7*rsi_h
        svg.append(f'<line x1="{margin}" y1="{y30}" x2="{width_px-margin}" y2="{y30}" stroke="#333" stroke-dasharray="3,3" />')
        svg.append(f'<line x1="{margin}" y1="{y70}" x2="{width_px-margin}" y2="{y70}" stroke="#333" stroke-dasharray="3,3" />')
    # x labels
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        t = ""
        if candles[i].get("t"):
            try: t = candles[i]["t"].strftime("%Y-%m-%d %H:%M")
            except: t = str(candles[i]["t"])
        svg.append(f'<text x="{x-40}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')
    svg.append('</svg>')
    return "\n".join(svg)

# -----------------------
# SIMPLE PERCEPTRON (ML)
# -----------------------
class SimplePerceptron:
    def __init__(self,n):
        self.n=n; self.weights=[random.uniform(-0.01,0.01) for _ in range(n)]; self.bias=random.uniform(-0.01,0.01); self.lr=0.01
    def raw(self,x): return sum(w*xi for w,xi in zip(self.weights,x))+self.bias
    def predict(self,x): return 1 if self.raw(x)>=0 else -1
    def train_epoch(self,X,y):
        errors=0
        for xi,yi in zip(X,y):
            pred=self.predict(xi)
            if pred!=yi:
                errors+=1
                for j in range(self.n): self.weights[j]+=self.lr*yi*xi[j]
                self.bias+=self.lr*yi
        return errors
    def train(self,X,y,epochs=50):
        for e in range(epochs):
            if self.train_epoch(X,y)==0: break
    def export(self): return {"n":self.n,"weights":self.weights,"bias":self.bias}
    def load(self,obj): self.n=obj["n"]; self.weights=obj["weights"]; self.bias=obj["bias"]

def load_model():
    m = load_json(MODEL_FILE)
    if m:
        try:
            p = SimplePerceptron(m.get("n",6)); p.load(m); return p
        except Exception:
            pass
    return SimplePerceptron(6)

def save_model(p):
    save_json(MODEL_FILE, p.export())

# -----------------------
# FEATURES
# -----------------------
def features_from_candles(candles):
    closes = [c["close"] for c in candles]
    if len(closes) < 60:
        while len(closes) < 60: closes.insert(0, closes[0] if closes else 100.0)
    last = closes[-1]
    ret1 = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0.0
    sma20 = sum(closes[-20:]) / 20
    sma50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else sma20
    sma_diff = (sma20 - sma50) / last if last != 0 else 0.0
    rsi_vals = rsi(closes, 14)
    rsi_last = rsi_vals[-1] if rsi_vals and rsi_vals[-1] is not None else 50.0
    macd_line, sig, hist = macd(closes, 12, 26, 9)
    macd_last = macd_line[-1] if macd_line and macd_line[-1] is not None else 0.0
    vol = statistics.pstdev([(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]) if len(closes) > 1 else 0.0
    highs = [c["high"] for c in candles[-14:]] if len(candles) >= 14 else [c["high"] for c in candles]
    lows = [c["low"] for c in candles[-14:]] if len(candles) >= 14 else [c["low"] for c in candles]
    atr_val = sum([h-l for h,l in zip(highs,lows)]) / len(highs) if highs else 0.0
    return [ret1, sma_diff, rsi_last/100.0, macd_last, vol, atr_val/last if last != 0 else 0.0]

# -----------------------
# BACKTEST (simple)
# -----------------------
def backtest_strategy(candles, strategy_fn, initial_cash=10000.0):
    cash = initial_cash; position=None; trades=[]; equity=[]
    for i in range(len(candles)):
        price = candles[i]["close"]
        signal = strategy_fn(candles, i)
        if position:
            # exit logic
            if position["direction"]=="long":
                if price <= position["stop"]:
                    pnl = (position["stop"] - position["entry"]) * position["qty"]; cash += position["qty"]*position["stop"]
                    trades.append({**position, "exit":position["stop"], "exit_idx":i, "pnl":pnl}); position=None
                elif price >= position["tp"]:
                    pnl = (position["tp"] - position["entry"]) * position["qty"]; cash += position["qty"]*position["tp"]
                    trades.append({**position, "exit":position["tp"], "exit_idx":i, "pnl":pnl}); position=None
            else:
                if price >= position["stop"]:
                    pnl = (position["entry"] - position["stop"]) * position["qty"]; cash += position["qty"]*(position["entry"]-position["stop"])
                    trades.append({**position, "exit":position["stop"], "exit_idx":i, "pnl":pnl}); position=None
                elif price <= position["tp"]:
                    pnl = (position["entry"] - position["tp"]) * position["qty"]; cash += position["qty"]*(position["entry"]-position["tp"])
                    trades.append({**position, "exit":position["tp"], "exit_idx":i, "pnl":pnl}); position=None
        if signal and not position:
            dirc = "long" if signal==1 else "short"
            risk_cap = 0.1*(cash if cash>0 else initial_cash)
            qty = risk_cap / price if price>0 else 0.0
            stop_price, stop_pct, vol = calculate_dynamic_stop(price, candles[:i+1], position_type=dirc)
            tp = price + (price - stop_price) * 2 if dirc=="long" else price - (stop_price - price) * 2
            position = {"entry":price, "qty":qty, "direction":dirc, "stop":stop_price, "tp":tp, "entry_idx":i}
            if dirc=="long": cash -= qty*price
            trades.append({**position, "opened_idx":i})
        cur_val = cash + (position["qty"]*price if position and position["direction"]=="long" else (position["qty"]*(position["entry"]-price) if position else 0.0)) if position else cash
        equity.append(cur_val)
    if position:
        price=candles[-1]["close"]
        if position["direction"]=="long":
            pnl=(price-position["entry"])*position["qty"]; cash += position["qty"]*price
        else:
            pnl=(position["entry"]-price)*position["qty"]; cash += position["qty"]*(position["entry"]-price)
        trades.append({**position,"exit":price,"exit_idx":len(candles)-1,"pnl":pnl}); equity.append(cash)
    total_return = (equity[-1]-initial_cash)/initial_cash if initial_cash else 0.0
    returns = [(equity[i]-equity[i-1])/equity[i-1] for i in range(1,len(equity)) if equity[i-1]!=0] if len(equity)>1 else []
    avg_ret = statistics.mean(returns) if returns else 0.0
    std_ret = statistics.pstdev(returns) if len(returns)>1 else 0.0
    sharpe = avg_ret/std_ret if std_ret!=0 else 0.0
    peak=equity[0] if equity else initial_cash; max_dd=0.0
    for v in equity:
        if v>peak: peak=v
        dd = (peak-v)/peak if peak>0 else 0.0
        if dd>max_dd: max_dd=dd
    winrate = len([t for t in trades if t.get("pnl",0)>0])/len(trades) if trades else 0.0
    return {"trades":trades, "equity":equity, "total_return":total_return, "sharpe":sharpe, "max_dd":max_dd, "winrate":winrate, "final_cash":cash}

# -----------------------
# STRATEGIES
# -----------------------
def strategy_sma_cross(candles,i):
    closes=[c["close"] for c in candles[:i+1]]
    if len(closes)<51: return 0
    s20=sum(closes[-20:])/20; s50=sum(closes[-50:])/50
    prev_s20=sum(closes[-21:-1])/20; prev_s50=sum(closes[-51:-1])/50
    if prev_s20 <= prev_s50 and s20 > s50: return 1
    if prev_s20 >= prev_s50 and s20 < s50: return -1
    return 0

def strategy_macd_momentum(candles,i):
    closes=[c["close"] for c in candles[:i+1]]
    if len(closes)<35: return 0
    m,s,h = macd(closes,12,26,9)
    if len(m)<2 or m[-1] is None or s[-1] is None or m[-2] is None or s[-2] is None: return 0
    if m[-1] > s[-1] and m[-2] <= s[-2]: return 1
    if m[-1] < s[-1] and m[-2] >= s[-2]: return -1
    return 0

def strategy_rsi_bounce(candles,i):
    closes=[c["close"] for c in candles[:i+1]]
    if len(closes) < 20: return 0
    r = rsi(closes,14)
    if r[-1] is None or r[-2] is None: return 0
    if r[-1] < 30 and r[-1] > r[-2]: return 1
    if r[-1] > 70 and r[-1] < r[-2]: return -1
    return 0

def strategy_breakout(candles,i):
    window=20
    if i < window: return 0
    recent_highs = [c["high"] for c in candles[i-window:i]]
    recent_lows = [c["low"] for c in candles[i-window:i]]
    if candles[i]["close"] > max(recent_highs): return 1
    if candles[i]["close"] < min(recent_lows): return -1
    return 0

# -----------------------
# ANALYZE & ESTIMATE
# -----------------------
def analyze_and_estimate(candles, perceptron_model):
    markers = detect_markers(candles)
    closes=[c["close"] for c in candles]
    s20 = sma(closes,20); s50 = sma(closes,50)
    macd_line, macd_sig, macd_hist = macd(closes,12,26,9)
    rsi_vals = rsi(closes,14)
    boll = bollinger(closes,20,2.0)
    supports,resistances = detect_support_resistance(candles, window=15)
    reasons=[]; score=0
    last = candles[-1]; prev = candles[-2] if len(candles)>1 else None
    if is_hammer(last): score+=1; reasons.append("Hammer (bullish)")
    if prev and is_bullish_engulfing(prev,last): score+=2; reasons.append("Bullish Engulfing")
    if is_three_white_soldiers(candles): score+=2; reasons.append("3 White Soldiers")
    if is_shooting_star(last): score-=1; reasons.append("Shooting Star (bearish)")
    if prev and is_bearish_engulfing(prev,last): score-=2; reasons.append("Bearish Engulfing")
    if s20 and s50 and s20[-1] and s50[-1]:
        if s20[-1] > s50[-1]: score+=1; reasons.append("SMA20 > SMA50 (bullish)")
        else: score-=1; reasons.append("SMA20 < SMA50 (bearish)")
    if macd_line and macd_sig and macd_line[-1] is not None and macd_sig[-1] is not None:
        if macd_line[-1] > macd_sig[-1]: score+=1; reasons.append("MACD > Signal (momentum up)")
        else: score-=1; reasons.append("MACD < Signal (momentum down)")
    # ML signal
    feat = features_from_candles(candles)
    ml_signal = perceptron_model.predict(feat) if perceptron_model else 0
    base_prob = 0.5 + (score / 12.0)
    ml_bias = 0.12 * (1 if ml_signal == 1 else -1)
    prob = min(max(base_prob + ml_bias, 0.01), 0.99)
    stop_price, stop_pct, vol = calculate_dynamic_stop(closes[-1], candles, position_type="long")
    risk_pct = stop_pct * 100.0
    if prob >= 0.65: rec = "Kaufen (Long empfohlen)"
    elif prob <= 0.35: rec = "Short / Verkaufen (Short empfohlen)"
    else: rec = "Halten / Beobachten"
    success_est = round(prob*100.0,1)
    sents=[]
    if rec.startswith("Kaufen"):
        sents.append(f"Algorithmus erkennt vorrangig bullishe Signale ({len(reasons)} Indikatoren).")
        sents.append(f"Empfohlener Stop-Loss: ca. {round(risk_pct,2)}% unter aktuellem Kurs.")
        sents.append("Tipp: Klein anfangen, Stop setzen, bei Bestätigung nachlegen.")
    elif rec.startswith("Short"):
        sents.append("Mehrere bärische Signale — erhöhte Vorsicht.")
        sents.append(f"Empfohlener Stop-Loss: ca. {round(risk_pct,2)}% oberhalb des Kurses.")
        sents.append("Tipp: Small size oder Absicherung; warte auf Bestätigung.")
    else:
        sents.append("Kein klares Signal — Markt neutral.")
        sents.append("Beobachte Volumen, RSI und SMA-Kreuzungen.")
        sents.append("Tipp: Warte auf klares Setup (Breakout oder bestätigte Candle).")
    return {
        "markers": markers, "sma20": s20, "sma50": s50, "macd":(macd_line,macd_sig,macd_hist),
        "rsi": rsi_vals, "boll": boll, "supports": supports, "resistances": resistances,
        "recommendation": rec, "success_pct": success_est, "risk_pct": round(risk_pct,2),
        "prob": prob, "ml_signal": ml_signal, "reasons": reasons, "summary_sentences": sents
    }

# -----------------------
# ROBOFLOW IMAGE INFERENCE
# -----------------------
def encode_multipart(fields, file_fieldname, filename, file_bytes, content_type):
    """
    Build multipart/form-data body (bytes) for a single file upload and optional fields.
    Returns (content_type_header, body_bytes)
    """
    boundary = '----WebKitFormBoundary' + ''.join(random.choice('0123456789abcdef') for _ in range(16))
    crlf = b'\r\n'
    body = bytearray()
    # file part
    body.extend(b'--' + boundary.encode() + crlf)
    body.extend(f'Content-Disposition: form-data; name="{file_fieldname}"; filename="{filename}"'.encode() + crlf)
    body.extend(f'Content-Type: {content_type}'.encode() + crlf + crlf)
    body.extend(file_bytes + crlf)
    # end
    body.extend(b'--' + boundary.encode() + b'--' + crlf)
    content_type_header = f'multipart/form-data; boundary={boundary}'
    return content_type_header, bytes(body)

def roboflow_detect(image_bytes):
    """
    Send image bytes to Roboflow detection API.
    Endpoint format: https://detect.roboflow.com/{MODEL_PATH}?api_key=KEY
    Model path might be something like "chart-detector/1" — check your Roboflow deployment.
    Returns parsed JSON or None.
    """
    try:
        endpoint = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_PATH}?api_key={urllib.parse.quote(ROBOFLOW_KEY)}"
        # build multipart body
        content_type, body = encode_multipart({}, "file", "upload.png", image_bytes, "image/png")
        req = urllib.request.Request(endpoint, data=body, method="POST")
        req.add_header("Content-Type", content_type)
        req.add_header("User-Agent", "Lumina-Pro-Agent/1.0")
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except Exception as e:
        # Roboflow may respond with an error if model path invalid; return None
        return None

# -----------------------
# IMAGE ANALYSIS (combining Roboflow + heuristics)
# -----------------------
def analyze_uploaded_chart_with_roboflow(uploaded_file):
    """
    1) If Roboflow available, send image and parse detections (boxes, labels)
    2) Fallback to PIL heuristics
    Returns combined result dict: {'rf': rf_json_or_None, 'heuristics': heur, 'final_rec': {...}}
    """
    try:
        bytes_io = uploaded_file.read()
        rf_res = roboflow_detect(bytes_io)
    except Exception:
        rf_res = None
    heur = None
    if PIL_AVAILABLE:
        try:
            heur = analyze_uploaded_chart_local(bytes_io)
        except Exception:
            heur = None
    # combine results
    # simple fusion rules:
    rec = {"recommendation": "Neutral", "reason": [], "risk_pct": None, "prob": None, "expected_move_pct": None}
    # Roboflow analysis strong signals -> convert
    if rf_res and isinstance(rf_res, dict):
        # Roboflow returns 'predictions' list with label/confidence and bbox
        preds = rf_res.get("predictions", [])
        labels = [p.get("class") for p in preds]
        confs = [p.get("confidence", 0.0) for p in preds]
        # heuristics: if many 'bullish_pattern' labels or 'engulfing' -> buy
        buy_like = sum(1 for l in labels if any(k in (l or "").lower() for k in ["bull", "engulf", "hammer", "threewhite", "morning"]))
        sell_like = sum(1 for l in labels if any(k in (l or "").lower() for k in ["bear", "shoot", "evening", "tripletop"]))
        avg_conf = (sum(confs)/len(confs)) if confs else 0.0
        if buy_like > sell_like and buy_like >= 1:
            rec["recommendation"] = "Kaufen (Long)"
            rec["reason"].append(f"Roboflow: {buy_like} bullishe Muster erkannt (avg conf {avg_conf:.2f})")
            rec["prob"] = min(95.0, 50 + buy_like*12 + avg_conf*20)
            rec["risk_pct"] = max(0.5, 8.0 - buy_like)
            rec["expected_move_pct"] = min(12.0, 3 + buy_like*2)
        elif sell_like > buy_like and sell_like >= 1:
            rec["recommendation"] = "Short / Verkaufen"
            rec["reason"].append(f"Roboflow: {sell_like} bärische Muster erkannt (avg conf {avg_conf:.2f})")
            rec["prob"] = min(95.0, 50 + sell_like*12 + avg_conf*20)
            rec["risk_pct"] = max(0.5, 8.0 - sell_like)
            rec["expected_move_pct"] = min(12.0, 3 + sell_like*2)
        else:
            rec["recommendation"] = "Neutral"
            rec["prob"] = min(85.0, 50 + avg_conf*10)
            rec["risk_pct"] = 5.0
            rec["expected_move_pct"] = 0.0
    # combine with heuristics (PIL)
    if heur:
        if heur.get("trend") == "Aufwärtstrend":
            if rec["recommendation"].startswith("Short"):
                rec["reason"].append("Heuristik: Bildtrend zeigt Aufwärtstrend (Konflikt).")
            else:
                rec["recommendation"] = "Kaufen (Long)"
                rec["reason"].append("Heuristik: Bildtrend Aufwärts")
                rec["prob"] = max(rec.get("prob",40), rec.get("prob",0) + 8)
                rec["risk_pct"] = rec.get("risk_pct",5.0) * 0.9
        elif heur.get("trend") == "Abwärtstrend":
            if rec["recommendation"].startswith("Kaufen"):
                rec["reason"].append("Heuristik: Bildtrend zeigt Abwärtstrend (Konflikt).")
            else:
                rec["recommendation"] = "Short / Verkaufen"
                rec["reason"].append("Heuristik: Bildtrend Abwärts")
                rec["prob"] = max(rec.get("prob",40), rec.get("prob",0) + 8)
                rec["risk_pct"] = rec.get("risk_pct",5.0) * 1.1
        # add pattern notes
        for n in heur.get("notes", []): rec["reason"].append("Heuristik: " + n)
    # ensure fields
    if rec.get("prob") is None: rec["prob"] = 50.0
    if rec.get("risk_pct") is None: rec["risk_pct"] = 5.0
    if rec.get("expected_move_pct") is None: rec["expected_move_pct"] = 0.0
    return {"roboflow": rf_res, "heuristics": heur, "final": rec}

def analyze_uploaded_chart_local(image_bytes):
    """
    Local PIL heuristics: detect peaks, brightness, W/M patterns, etc.
    """
    if not PIL_AVAILABLE:
        return None
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    w,h = img.size
    maxw = 1400
    if w > maxw: img = img.resize((maxw, int(h*maxw/w))); w,h = img.size
    img = ImageOps.autocontrast(img, cutoff=1)
    pix = img.load()
    col_sums = [0]*w
    for x in range(w):
        s = 0
        for y in range(h):
            s += 255 - pix[x,y]
        col_sums[x] = s
    # smooth
    smooth = []
    for i in range(w):
        vals = col_sums[max(0,i-3):min(w,i+4)]
        smooth.append(sum(vals)/len(vals))
    avg = sum(smooth)/len(smooth) if len(smooth)>0 else 0
    peaks=[]
    for i in range(2, w-2):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > avg*1.3:
            peaks.append(i)
    minima=[]
    for i in range(2,w-2):
        if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < avg*0.7:
            minima.append(i)
    patterns=[]
    if len(peaks) > 4: patterns.append("Candlestick-Chart erkannt")
    if len(minima) >= 2:
        for j in range(len(minima)-1):
            gap = minima[j+1]-minima[j]
            if 8 < gap < 400:
                patterns.append("Double Bottom / W-Pattern möglich"); break
    left_mean = ImageStat.Stat(img.crop((0,0,w//2,h))).mean[0]
    right_mean = ImageStat.Stat(img.crop((w//2,0,w,h))).mean[0]
    trend = "Seitwärts"
    if right_mean < left_mean - 6: trend = "Abwärtstrend"
    elif right_mean > left_mean + 6: trend = "Aufwärtstrend"
    notes=[]
    if "Double Bottom / W-Pattern möglich" in patterns: notes.append("Möglicher Double-Bottom — Reversal möglich")
    if trend == "Aufwärtstrend": notes.append("Aufwärtstrend erkennbar — Long bias")
    elif trend == "Abwärtstrend": notes.append("Abwärtstrend erkennbar — Short bias")
    if not notes: notes.append("Kein klares Muster, manuelle Prüfung empfohlen")
    return {"patterns": patterns, "trend": trend, "notes": notes, "peaks": len(peaks), "minima": len(minima)}

# -----------------------
# GET CANDLES (live or sim)
# -----------------------
def get_candles_for(symbol: str, interval: str, periods: int, start_price: float, use_live: bool):
    sym = symbol.strip()
    if use_live and ALPHA_KEY and internet_ok():
        fetched = fetch_alpha_intraday(sym, interval=interval, outputsize="compact")
        if fetched and len(fetched) >= periods:
            candles = fetched[-periods:]
            cache_save(sym, interval, candles)
            return candles, True
        cached = cache_load(sym, interval)
        if cached and len(cached) >= periods:
            return cached[-periods:], False
    else:
        cached = cache_load(sym, interval)
        if cached and len(cached) >= periods:
            return cached[-periods:], False
    tf_map = {"1min":1,"5min":5,"15min":15,"30min":30,"60min":60}
    mins = periods * tf_map.get(interval, 5)
    prices = generate_price_walk(sym + "|" + interval, mins, start_price)
    ohlc = prices_to_ohlc(prices, candle_size=tf_map.get(interval, 5))
    if len(ohlc) < periods:
        pad = periods - len(ohlc)
        pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price,"volume":0}
        ohlc = [pad_item]*pad + ohlc
    return ohlc[-periods:], False

# -----------------------
# UI PAGES
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live Analyzer","Bild-Analyse (Roboflow)","Backtest & ML","Portfolio","Einstellungen","Hilfe"])

# quick status
if not internet_ok():
    st.sidebar.error("❌ Keine Internetverbindung — Offline-Fallback aktiv")
elif not ALPHA_KEY:
    st.sidebar.warning("⚠️ Alpha Vantage Key nicht gesetzt — Offline-Fallback aktiv")
else:
    st.sidebar.success("✅ Live-Mode möglich (Alpha Vantage)")

if "model" not in st.session_state:
    st.session_state.model = load_model()

# HOME
if page == "Home":
    st.header("Lumina Pro — Übersicht")
    st.markdown("Kurzübersicht & Quick Analyzer")
    port = load_json(PORTFOLIO_FILE) or []
    st.subheader("Portfolio")
    if not port: st.info("Portfolio leer")
    else:
        for idx,p in enumerate(port): st.write(f"{idx+1}. {p}")
    st.markdown("---")
    st.subheader("Quick Analyzer")
    col1,col2 = st.columns([3,1])
    with col2:
        sym = st.text_input("Symbol", value="AAPL")
        interval = st.selectbox("Intervall", ["1min","5min","15min","30min","60min"], index=1)
        periods = st.slider("Candles", 30, 800, 240, step=10)
        start_price = st.number_input("Startpreis fallback", value=100.0)
        use_live = st.checkbox("Live falls möglich", value=True)
        if st.button("Analysiere jetzt (Quick)"):
            st.session_state["do_quick"] = True
    with col1:
        if st.session_state.get("do_quick", False):
            candles, live_flag = get_candles_for(sym, interval, periods, start_price, use_live)
            analysis = analyze_and_estimate(candles, st.session_state.model)
            stop_price, stop_pct, vol = calculate_dynamic_stop(candles[-1]["close"], candles, position_type="long")
            stop_line = {"price": stop_price, "type":"long", "pct": stop_pct}
            svg = render_candles_svg(candles, markers=analysis["markers"], stop_line=stop_line, sma_periods=(20,50), rsi_vals=analysis["rsi"], boll=analysis["boll"], supports=analysis["supports"], resistances=analysis["resistances"])
            st.components.v1.html(svg, height=640)
            st.markdown(f"**Empfehlung:** {analysis['recommendation']}")
            st.markdown(f"**Erfolgschance:** {analysis['success_pct']}%  •  **Risiko:** {analysis['risk_pct']}%")
            st.markdown("**3-Satz Zusammenfassung:**")
            for s in analysis["summary_sentences"]: st.write("- " + s)
            if live_flag: st.info("Live Daten verwendet (Alpha Vantage).")
            else: st.info("Offline / Cache verwendet.")

# LIVE ANALYZER
elif page == "Live Analyzer":
    st.header("Live Analyzer — Candles & Recommendation")
    left, right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol", value="AAPL")
        interval = st.selectbox("Intervall", ["1min","5min","15min","30min","60min"], index=1)
        periods = st.slider("Kerzen", 30, 800, 240, step=10)
        start_price = st.number_input("Startpreis fallback", value=100.0)
        use_live = st.checkbox("Live via Alpha Vantage (wenn Key & online)", value=True)
        if st.button("Lade & Analysiere"):
            st.session_state["run_live"] = True
        st.markdown("---")
        st.write("Model status:", "geladen" if st.session_state.model else "kein Model")
        if st.button("Train Perceptron (synthetic)"):
            st.info("Training Perceptron (synthetic)...")
            X=[]; y=[]
            for i in range(800):
                base = generate_price_walk("train"+str(i), 300, 100.0)
                ohlc = prices_to_ohlc(base, candle_size=5)
                label = random.choice([1,-1])
                if label==1:
                    for k in range(3):
                        ohlc[-1-k]["open"] = ohlc[-1-k]["close"] * (1 - random.uniform(0.01,0.03))
                        ohlc[-1-k]["close"] = ohlc[-1-k]["open"] * (1 + random.uniform(0.01,0.05))
                else:
                    for k in range(3):
                        ohlc[-1-k]["open"] = ohlc[-1-k]["close"] * (1 + random.uniform(0.01,0.04))
                        ohlc[-1-k]["close"] = ohlc[-1-k]["open"] * (1 - random.uniform(0.01,0.05))
                feat = features_from_candles(ohlc)
                X.append(feat); y.append(label)
            p = SimplePerceptron(len(X[0])); p.train(X,y,epochs=100); save_model(p); st.session_state.model = p; st.success("Perceptron trained.")
    with left:
        if st.session_state.get("run_live", False):
            candles, live_flag = get_candles_for(symbol, interval, periods, start_price, use_live)
            analysis = analyze_and_estimate(candles, st.session_state.model)
            stop_price, stop_pct, vol = calculate_dynamic_stop(candles[-1]["close"], candles, position_type="long")
            svg = render_candles_svg(candles, markers=analysis["markers"], stop_line={"price":stop_price,"type":"long","pct":stop_pct}, sma_periods=(20,50), rsi_vals=analysis["rsi"], boll=analysis["boll"], supports=analysis["supports"], resistances=analysis["resistances"])
            st.components.v1.html(svg, height=640)
            st.subheader("Empfehlung & Kennzahlen")
            st.markdown(f"**Empfehlung:** {analysis['recommendation']}")
            st.markdown(f"**Erfolgschance:** {analysis['success_pct']}%  •  **Risiko:** {analysis['risk_pct']}%")
            st.markdown("**Kurztext (3 Sätze):**")
            for s in analysis["summary_sentences"]: st.write("- " + s)
            st.markdown("**Erkannte Gründe / Muster:**")
            for r in analysis["reasons"]: st.write("- " + r)
            if live_flag: st.info("Live Daten verwendet (Alpha Vantage).")
            else: st.info("Offline / Cache verwendet.")

# IMAGE ANALYSIS (Roboflow)
elif page == "Bild-Analyse (Roboflow)":
    st.header("Chart-Bild-Analyse (Roboflow + Heuristics)")
    st.markdown("Lade ein Chart-Screenshot hoch (z. B. TradingView). Roboflow + lokale Heuristik analysieren Chart und geben Empfehlung.")
    uploaded = st.file_uploader("Bild hochladen (PNG/JPG)", type=["png","jpg","jpeg"])
    if uploaded is None:
        st.info("Kein Bild. Roboflow-Analyse benötigt Internet. Pillow optional für lokale Heuristik.")
        if not PIL_AVAILABLE: st.warning("Pillow fehlt: Bild-Heuristik eingeschränkt. Füge pillow in requirements.txt hinzu.")
    else:
        st.image(uploaded, use_column_width=True)
        if st.button("Analysiere Bild (Roboflow)"):
            with st.spinner("Bild wird an Roboflow gesendet und analysiert..."):
                res = analyze_uploaded_chart_with_roboflow(uploaded)
            if res is None:
                st.error("Analyse fehlgeschlagen.")
            else:
                st.subheader("Ergebnis")
                rf = res.get("roboflow")
                heur = res.get("heuristics")
                final = res.get("final")
                if rf: st.write("Roboflow detections:", rf.get("predictions", []))
                if heur:
                    st.write("Heuristische Analyse:", heur.get("patterns", []), "Trend:", heur.get("trend"))
                    for n in heur.get("notes", []): st.write("- " + n)
                st.markdown("---")
                st.write("**Empfehlung:**", final.get("recommendation"))
                st.write("**Wahrscheinlichkeit:**", f"{final.get('prob'):.1f}%")
                st.write("**Risiko (Stop %):**", f"{final.get('risk_pct'):.2f}%")
                st.write("**Erwartete Bewegung (geschätzt):**", f"{final.get('expected_move_pct'):.2f}%")
                st.write("**Begründung:**")
                for r in final.get("reason", []): st.write("- " + r)

# BACKTEST & ML
elif page == "Backtest & ML":
    st.header("Backtest & ML")
    left,right = st.columns([3,1])
    with right:
        strategy = st.selectbox("Strategie", ["SMA Crossover","MACD Momentum","RSI Bounce","Breakout"])
        initial_cash = st.number_input("Initial Cash", value=10000.0, step=100.0)
        run_bt = st.button("Backtest ausführen")
        if st.button("Train Perceptron (synthetic)"):
            st.info("Training Perceptron (synthetic)...")
            X=[]; y=[]
            for i in range(800):
                base = generate_price_walk("train"+str(i), 300, 100.0)
                ohlc = prices_to_ohlc(base, candle_size=5)
                label = random.choice([1,-1])
                if label==1:
                    for k in range(3):
                        ohlc[-1-k]["open"] = ohlc[-1-k]["close"]*(1-random.uniform(0.01,0.03))
                        ohlc[-1-k]["close"] = ohlc[-1-k]["open"]*(1+random.uniform(0.01,0.05))
                else:
                    for k in range(3):
                        ohlc[-1-k]["open"] = ohlc[-1-k]["close"]*(1+random.uniform(0.01,0.04))
                        ohlc[-1-k]["close"] = ohlc[-1-k]["open"]*(1-random.uniform(0.01,0.05))
                feat = features_from_candles(ohlc)
                X.append(feat); y.append(label)
            p = SimplePerceptron(len(X[0])); p.train(X,y,epochs=120); save_model(p); st.session_state.model = p; st.success("Perceptron trained and saved.")
    with left:
        symbol_bt = st.text_input("Symbol für Backtest", value="AAPL")
        interval_bt = st.selectbox("Interval", ["5min","15min","30min","60min"], index=1)
        periods_bt = st.slider("Kerzen für Backtest", 100, 2000, 600, step=50)
        start_price_bt = st.number_input("Startpreis fallback", value=100.0)
        use_live_bt = st.checkbox("Live falls verfügbar", value=False)
        if run_bt:
            candles_bt, live_flag = get_candles_for(symbol_bt, interval_bt, periods_bt, start_price_bt, use_live_bt)
            strat_fn = {"SMA Crossover":strategy_sma_cross,"MACD Momentum":strategy_macd_momentum,"RSI Bounce":strategy_rsi_bounce,"Breakout":strategy_breakout}[strategy]
            res = backtest_strategy(candles_bt, strat_fn, initial_cash=initial_cash)
            st.subheader("Backtest Ergebnisse")
            st.write(f"Total Return: {res['total_return']*100:.2f}%")
            st.write(f"Sharpe-like: {res['sharpe']:.4f}")
            st.write(f"Max Drawdown: {res['max_dd']*100:.2f}%")
            st.write(f"Winrate: {res['winrate']*100:.2f}%")
            if res.get("trades"):
                st.write("Trades (erste 10):"); 
                for t in res["trades"][:10]: st.write(t)
            st.line_chart(res["equity"])
            if live_flag: st.info("Live data used")
            else: st.info("Offline simulation")

# PORTFOLIO
elif page == "Portfolio":
    st.header("Portfolio")
    port = load_json(PORTFOLIO_FILE) or []
    if not port: st.info("Portfolio empty")
    else:
        for idx,p in enumerate(port): st.write(f"{idx+1}. {p}")
    if st.button("Export Portfolio"):
        st.download_button("Download", data=json.dumps(port, ensure_ascii=False, indent=2), file_name="portfolio.json")
    if st.button("Clear Portfolio"):
        save_json(PORTFOLIO_FILE, []); st.success("Cleared")

# SETTINGS
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Alpha Vantage Key (in code by default). If you want to override for session:")
    api_key_input = st.text_input("Alpha Key (session override)", value=ALPHA_KEY)
    if st.button("Set Alpha Key for session"):
        if api_key_input.strip():
            st.session_state["ALPHA_KEY"] = api_key_input.strip(); st.success("Alpha key set in session (temporary).")
    st.markdown("---")
    st.write("Roboflow model path (current):", ROBOFLOW_MODEL_PATH)
    st.write("Pillow installed:", "Yes" if PIL_AVAILABLE else "No")
    if not PIL_AVAILABLE:
        st.info("Install pillow to improve local image heuristics (add to requirements.txt).")
    if st.button("Clear Cache"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR,f))
            except: pass
        st.success("Cache cleared")

# HELP
elif page == "Hilfe":
    st.header("Hilfe & Hinweise")
    st.markdown("""
    **Wichtig**
    - Alpha Vantage Free Tier: ~5 calls/minute. Use cache to avoid rate limits.
    - Roboflow: you must confirm the model path `ROBOFLOW_MODEL_PATH` matches your deployment. Roboflow returns detections in JSON.
    - Pillow recommended for image heuristics.
    - Recommendations are estimates and **no financial advice**.
    """)
    st.markdown("**requirements.txt** (minimum):")
    st.code("streamlit\npillow\n")

st.markdown("---")
st.caption("Lumina Pro — Deep Analyzer — Built with Alpha Vantage + Roboflow. Recommendations are probabilistic estimates and not financial advice.")
