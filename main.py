# main.py
# Lumina Pro — Deep Analyzer (Live + Offline, Image Analyzer)
# - Uses Alpha Vantage key embedded (or via st.secrets)
# - SVG candlesticks, many indicators, pattern & strategy detection
# - Image analysis (Pillow optional)
#
# Save as main.py and run: streamlit run main.py

import streamlit as st
import json, os, time, random, math, urllib.request, urllib.parse
from datetime import datetime, timedelta
import statistics
from typing import List, Dict, Any

# optional image processing
try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# App config
# -------------------------
st.set_page_config(page_title="Lumina Pro — Deep Analyzer", layout="wide")
st.markdown("""<style>
body { background:#000; color:#e6eef6; }
.stButton>button { background:#111; color:#e6eef6; border:1px solid #222; }
</style>""", unsafe_allow_html=True)
st.title("Lumina Pro — Deep Analyzer")

# -------------------------
# API key (embedded; optionally override with secrets)
# -------------------------
ALPHA_KEY = "22XGVO0TQ1UV167C"  # <-- embedded per request
try:
    if st.secrets and st.secrets.get("api_keys", {}).get("ALPHA_KEY"):
        ALPHA_KEY = st.secrets["api_keys"]["ALPHA_KEY"]
except Exception:
    pass

# -------------------------
# Files & cache
# -------------------------
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

# -------------------------
# Asset lists
# -------------------------
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

KNOWN_SYMBOLS = {"aapl":"AAPL","btc":"BTC-USD","eth":"ETH-USD","googl":"GOOGL"}

# -------------------------
# Utilities
# -------------------------
def human_ts(): return datetime.utcnow().isoformat() + "Z"
def deterministic_seed(s: str) -> int: return abs(hash(s)) % (2**31)

# -------------------------
# Internet / API helpers
# -------------------------
def internet_ok():
    try:
        urllib.request.urlopen("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False

ONLINE = internet_ok() and bool(ALPHA_KEY)

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

def cache_path_for(symbol, interval): return os.path.join(CACHE_DIR, f"{symbol}_{interval}.json")
def cache_save(symbol, interval, candles):
    path=cache_path_for(symbol, interval)
    try:
        out=[]
        for c in candles:
            o=dict(c)
            if isinstance(o.get("t"), datetime): o["t"]=o["t"].isoformat()
            out.append(o)
        save_json(path, {"ts":time.time(),"candles":out})
    except Exception:
        pass

def cache_load(symbol, interval, max_age_seconds=3600*24):
    path=cache_path_for(symbol, interval)
    if not os.path.exists(path): return None
    try:
        obj=load_json(path)
        if not obj: return None
        if time.time()-obj.get("ts",0) > max_age_seconds: return None
        out=[]
        for c in obj.get("candles", []):
            o=dict(c)
            if isinstance(o.get("t"), str):
                try: o["t"]=datetime.fromisoformat(o["t"])
                except: o["t"]=datetime.utcnow()
            out.append(o)
        return out
    except Exception:
        return None

# -------------------------
# Offline generator & ohlc builder
# -------------------------
def generate_price_walk(seed: str, steps: int, start_price: float = 100.0):
    rnd = random.Random(deterministic_seed(seed))
    price = float(start_price)
    series=[]
    for _ in range(steps):
        drift = (rnd.random() - 0.49)*0.003
        shock = (rnd.random() - 0.5)*0.02
        price = max(0.01, price*(1+drift+shock))
        series.append(round(price,6))
    return series

def prices_to_ohlc(prices, candle_size=1):
    ohlc=[]
    for i in range(0,len(prices),candle_size):
        chunk = prices[i:i+candle_size]
        if not chunk: continue
        o=chunk[0]; c=chunk[-1]; h=max(chunk); l=min(chunk)
        ohlc.append({"t":None,"open":o,"high":h,"low":l,"close":c,"volume":0})
    now_dt=datetime.utcnow()
    minutes=candle_size
    for i in range(len(ohlc)):
        ohlc[i]["t"]= now_dt - timedelta(minutes=(len(ohlc)-1-i)*minutes)
    return ohlc

# -------------------------
# INDICATORS: SMA, EMA, MACD, RSI, Bollinger, ATR
# -------------------------
def sma(values, period):
    if not values: return []
    res=[]
    for i in range(len(values)):
        if i+1 < period: res.append(None)
        else: res.append(sum(values[i+1-period:i+1]) / period)
    return res

def ema(values, period):
    res=[]; k=2.0/(period+1.0); ema_prev=None
    for v in values:
        if ema_prev is None: ema_prev=v
        else: ema_prev = v*k + ema_prev*(1-k)
        res.append(ema_prev)
    return res

def macd(values, fast=12, slow=26, signal=9):
    if not values: return [],[],[]
    ef=ema(values,fast); es=ema(values,slow)
    mac=[]
    for a,b in zip(ef,es):
        mac.append((a-b) if (a is not None and b is not None) else None)
    mac_vals=[v for v in mac if v is not None]
    if not mac_vals: return mac, [None]*len(mac), [None]*len(mac)
    sig_vals=ema(mac_vals, signal)
    sig_iter=iter(sig_vals)
    sig_mapped=[]
    for v in mac:
        sig_mapped.append(None if v is None else next(sig_iter))
    hist=[(m-s) if (m is not None and s is not None) else None for m,s in zip(mac,sig_mapped)]
    return mac, sig_mapped, hist

def rsi(values, period=14):
    if len(values) < period+1: return [None]*len(values)
    deltas=[values[i]-values[i-1] for i in range(1,len(values))]
    gains=[d if d>0 else 0 for d in deltas]
    losses=[-d if d<0 else 0 for d in deltas]
    avg_gain=sum(gains[:period])/period
    avg_loss=sum(losses[:period])/period
    res=[None]*period
    for i in range(period, len(deltas)):
        avg_gain=(avg_gain*(period-1)+gains[i])/period
        avg_loss=(avg_loss*(period-1)+losses[i])/period
        rs = avg_gain/avg_loss if avg_loss!=0 else float('inf')
        val = 100 - (100/(1+rs))
        res.append(round(val,2))
    return res

def bollinger(values, period=20, mult=2.0):
    res=[]
    for i in range(len(values)):
        if i+1 < period: res.append((None,None,None))
        else:
            window = values[i+1-period:i+1]
            m=sum(window)/period; std=statistics.pstdev(window)
            res.append((round(m,6), round(m+mult*std,6), round(m-mult*std,6)))
    return res

def atr(candles, period=14):
    trs=[]
    for i in range(1, len(candles)):
        tr = max(candles[i]["high"] - candles[i]["low"], abs(candles[i]["high"] - candles[i-1]["close"]), abs(candles[i]["low"] - candles[i-1]["close"]))
        trs.append(tr)
    if not trs: return [None]*len(candles)
    res=[None]* (1 + len(trs))
    # simple SMA of TR
    for i in range(period, len(trs)+1):
        window = trs[i-period:i]
        res.append(sum(window)/period)
    # ensure length matches
    return res[:len(candles)]

# -------------------------
# Pattern detection
# -------------------------
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
    return prev and (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])

def is_bearish_engulfing(prev, cur):
    return prev and (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def is_three_white_soldiers(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"]>a["open"]) and (b["close"]>b["open"]) and (c["close"]>c["open"]) and (b["close"]>a["close"]) and (c["close"]>b["close"])

def is_morning_star(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"] < a["open"]) and (is_doji(b) or (b["close"] < b["open"])) and (c["close"] > c["open"]) and c["close"] > (a["close"] + a["open"]) /2

def is_evening_star(candles):
    if len(candles) < 3: return False
    a,b,c = candles[-3], candles[-2], candles[-1]
    return (a["close"] > a["open"]) and (is_doji(b) or (b["close"] > b["open"])) and (c["close"] < c["open"]) and c["close"] < (a["close"] + a["open"]) /2

def detect_markers(candles):
    markers=[]
    for i in range(1,len(candles)):
        cur=candles[i]; prev=candles[i-1]
        if is_bullish_engulfing(prev, cur): markers.append({"idx":i,"type":"buy","reason":"Bullish Engulfing"})
        if is_bearish_engulfing(prev, cur): markers.append({"idx":i,"type":"sell","reason":"Bearish Engulfing"})
        if is_hammer(cur): markers.append({"idx":i,"type":"buy","reason":"Hammer"})
        if is_shooting_star(cur): markers.append({"idx":i,"type":"sell","reason":"Shooting Star"})
    if is_three_white_soldiers(candles): markers.append({"idx":len(candles)-1,"type":"buy","reason":"Three White Soldiers"})
    if is_morning_star(candles): markers.append({"idx":len(candles)-1,"type":"buy","reason":"Morning Star"})
    if is_evening_star(candles): markers.append({"idx":len(candles)-1,"type":"sell","reason":"Evening Star"})
    # deduplicate
    seen=set(); uniq=[]
    for m in markers:
        k=(m["idx"],m["type"],m["reason"])
        if k not in seen:
            seen.add(k); uniq.append(m)
    return uniq

# -------------------------
# Support / Resistance helpers (naive)
# -------------------------
def detect_support_resistance(candles, window=20):
    closes=[c["close"] for c in candles]
    supports=[]; resistances=[]
    # local minima/maxima
    for i in range(2,len(closes)-2):
        window_vals = closes[max(0,i-window):i+window]
        v = closes[i]
        if v == min(window_vals):
            supports.append((i,v))
        if v == max(window_vals):
            resistances.append((i,v))
    # reduce to top 3
    supports = sorted(supports, key=lambda x: x[1])[:3]
    resistances = sorted(resistances, key=lambda x: -x[1])[:3]
    return supports, resistances

# -------------------------
# Stop-loss (adaptive)
# -------------------------
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

# -------------------------
# SVG Candles renderer (enhanced)
# -------------------------
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
    # grid and labels
    for i in range(6):
        y=margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{y}" x2="{width_px-margin}" y2="{y}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{y+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')
    # candles
    for idx,c in enumerate(candles):
        x_center = margin + idx*spacing + spacing/2
        x_left = x_center - candle_w/2
        y_open = y_pos(c["open"]); y_close=y_pos(c["close"])
        y_high=y_pos(c["high"]); y_low=y_pos(c["low"])
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
    if sma1: svg.append(polyline(sma1, "#66ccff", 1.8))
    if sma2: svg.append(polyline(sma2, "#ffcc66", 1.8))
    # boll shading
    if boll:
        pts_up=[]; pts_low=[]
        for i,b in enumerate(boll):
            if b[1] is None: pts_up.append(None)
            else: pts_up.append(f'{margin+i*spacing+spacing/2},{y_pos(b[1])}')
            if b[2] is None: pts_low.append(None)
            else: pts_low.append(f'{margin+i*spacing+spacing/2},{y_pos(b[2])}')
        def segs(pts):
            segs=[]; cur=[]
            for p in pts:
                if p is None:
                    if cur: segs.append(cur); cur=[]
                else: cur.append(p)
            if cur: segs.append(cur)
            return segs
        ups = segs(pts_up); lows = segs(pts_low)
        for u,l in zip(ups,lows):
            poly = " ".join(u + l[::-1])
            svg.append(f'<polygon points="{poly}" fill="#223333" opacity="0.2"/>')
    # markers
    used_counts={}
    for m in markers:
        i=m["idx"]
        if i<0 or i>=n: continue
        count=used_counts.get(i,0); used_counts[i]=count+1
        x_center = margin + i*spacing + spacing/2
        c = candles[i]
        y_high=y_pos(c["high"]); y_low=y_pos(c["low"])
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
    # supports/resistances
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
                x=margin+i*spacing+spacing/2; y = rsi_top + rsi_h - (v/100.0)*rsi_h
                cur.append(f"{x},{y}")
        if cur: pts.append(" ".join(cur))
        for s in pts:
            svg.append(f'<polyline points="{s}" fill="none" stroke="#ff66cc" stroke-width="1.2" />')
        y30 = rsi_top + rsi_h - 0.3*rsi_h; y70 = rsi_top + rsi_h - 0.7*rsi_h
        svg.append(f'<line x1="{margin}" y1="{y30}" x2="{width_px-margin}" y2="{y30}" stroke="#333" stroke-dasharray="3,3" />')
        svg.append(f'<line x1="{margin}" y1="{y70}" x2="{width_px-margin}" y2="{y70}" stroke="#333" stroke-dasharray="3,3" />')
    # x labels
    for i in range(0,n,max(1,n//10)):
        x=margin + i*spacing + spacing/2
        t=""
        if candles[i].get("t"):
            try: t=candles[i]["t"].strftime("%Y-%m-%d %H:%M")
            except: t=str(candles[i]["t"])
        svg.append(f'<text x="{x-40}" y="{height_px-6}" font-size="11" fill="#9aa6b2">{t}</text>')
    svg.append('</svg>')
    return "\n".join(svg)

# -------------------------
# Simple perceptron
# -------------------------
class SimplePerceptron:
    def __init__(self, n):
        self.n = n
        self.weights = [random.uniform(-0.01,0.01) for _ in range(n)]
        self.bias = random.uniform(-0.01,0.01)
        self.lr = 0.01
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
    def export(self): return {"n":self.n, "weights":self.weights, "bias":self.bias}
    def load(self,obj): self.n=obj["n"]; self.weights=obj["weights"]; self.bias=obj["bias"]

def load_model():
    m = load_json(MODEL_FILE)
    if m:
        try:
            p = SimplePerceptron(m.get("n",6)); p.load(m); return p
        except Exception:
            pass
    return SimplePerceptron(6)
def save_model(p): save_json(MODEL_FILE, p.export())

# -------------------------
# Features
# -------------------------
def features_from_candles(candles):
    closes=[c["close"] for c in candles]
    if len(closes)<60:
        while len(closes)<60: closes.insert(0, closes[0] if closes else 100.0)
    last=closes[-1]
    ret1=(closes[-1]-closes[-2])/closes[-2] if closes[-2]!=0 else 0.0
    sma20=sum(closes[-20:])/20
    sma50=sum(closes[-50:])/50 if len(closes)>=50 else sma20
    sma_diff=(sma20-sma50)/last if last!=0 else 0.0
    rsi_vals=rsi(closes,14); rsi_last = rsi_vals[-1] if rsi_vals and rsi_vals[-1] is not None else 50.0
    macd_line, sig, hist = macd(closes,12,26,9); macd_last = macd_line[-1] if macd_line and macd_line[-1] is not None else 0.0
    vol = statistics.pstdev([(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]) if len(closes)>1 else 0.0
    highs=[c["high"] for c in candles[-14:]] if len(candles)>=14 else [c["high"] for c in candles]
    lows=[c["low"] for c in candles[-14:]] if len(candles)>=14 else [c["low"] for c in candles]
    atr_val = sum([h-l for h,l in zip(highs,lows)])/len(highs) if highs else 0.0
    return [ret1, sma_diff, rsi_last/100.0, macd_last, vol, atr_val/last if last!=0 else 0.0]

# -------------------------
# Backtest engine (simple)
# -------------------------
def backtest_strategy(candles, strategy_fn, initial_cash=10000.0):
    cash=initial_cash; position=None; trades=[]; equity=[]
    for i in range(len(candles)):
        price=candles[i]["close"]
        signal=strategy_fn(candles,i)
        if position:
            if position["direction"]=="long":
                if price <= position["stop"]:
                    pnl=(position["stop"]-position["entry"])*position["qty"]; cash += position["qty"]*position["stop"]
                    trades.append({**position,"exit":position["stop"],"exit_idx":i,"pnl":pnl}); position=None
                elif price >= position["tp"]:
                    pnl=(position["tp"]-position["entry"])*position["qty"]; cash += position["qty"]*position["tp"]
                    trades.append({**position,"exit":position["tp"],"exit_idx":i,"pnl":pnl}); position=None
            else:
                if price >= position["stop"]:
                    pnl=(position["entry"]-position["stop"])*position["qty"]; cash += position["qty"]*(position["entry"]-position["stop"])
                    trades.append({**position,"exit":position["stop"],"exit_idx":i,"pnl":pnl}); position=None
                elif price <= position["tp"]:
                    pnl=(position["entry"]-position["tp"])*position["qty"]; cash += position["qty"]*(position["entry"]-position["tp"])
                    trades.append({**position,"exit":position["tp"],"exit_idx":i,"pnl":pnl}); position=None
        if signal and not position:
            dirc = "long" if signal==1 else "short"
            risk_cap = 0.1*(cash if cash>0 else initial_cash)
            qty = risk_cap / price if price>0 else 0.0
            stop_price, stop_pct, vol = calculate_dynamic_stop(price, candles[:i+1], position_type=dirc)
            tp = price + (price-stop_price)*2 if dirc=="long" else price - (stop_price-price)*2
            position = {"entry":price,"qty":qty,"direction":dirc,"stop":stop_price,"tp":tp,"entry_idx":i}
            if dirc=="long": cash -= qty*price
            trades.append({**position,"opened_idx":i})
        cur_val = cash + (position["qty"]*price if position and position["direction"]=="long" else (position["qty"]*(position["entry"]-price) if position else 0.0)) if position else cash
        equity.append(cur_val)
    if position:
        price=candles[-1]["close"]
        if position["direction"]=="long":
            pnl=(price-position["entry"])*position["qty"]; cash += position["qty"]*price
        else:
            pnl=(position["entry"]-price)*position["qty"]; cash += position["qty"]*(position["entry"]-price)
        trades.append({**position,"exit":price,"exit_idx":len(candles)-1,"pnl":pnl}); equity.append(cash)
    # metrics
    total_return = (equity[-1]-initial_cash)/initial_cash if initial_cash else 0.0
    returns = [(equity[i]-equity[i-1])/equity[i-1] for i in range(1,len(equity)) if equity[i-1]!=0] if len(equity)>1 else []
    avg_ret = statistics.mean(returns) if returns else 0.0
    std_ret = statistics.pstdev(returns) if len(returns)>1 else 0.0
    sharpe = avg_ret/std_ret if std_ret!=0 else 0.0
    peak=equity[0] if equity else initial_cash; max_dd=0.0
    for v in equity:
        if v>peak: peak=v
        dd=(peak-v)/peak if peak>0 else 0.0
        if dd>max_dd: max_dd=dd
    winrate = len([t for t in trades if t.get("pnl",0)>0])/len(trades) if trades else 0.0
    return {"trades":trades,"equity":equity,"total_return":total_return,"sharpe":sharpe,"max_dd":max_dd,"winrate":winrate,"final_cash":cash}

# -------------------------
# Strategies
# -------------------------
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
    if r[-1] is None: return 0
    if r[-1] < 30 and r[-1] > r[-2]: return 1
    if r[-1] > 70 and r[-1] < r[-2]: return -1
    return 0

def strategy_breakout(candles,i):
    # breakout above recent high or below recent low (window)
    window=20
    if i < window: return 0
    recent = [c["high"] for c in candles[i-window:i]]
    recent_low = [c["low"] for c in candles[i-window:i]]
    if candles[i]["close"] > max(recent): return 1
    if candles[i]["close"] < min(recent_low): return -1
    return 0

# -------------------------
# Analysis & estimate + 3-sentence summary
# -------------------------
def analyze_and_estimate(candles, perceptron_model):
    markers = detect_markers(candles)
    closes=[c["close"] for c in candles]
    s20 = sma(closes,20); s50=sma(closes,50)
    macd_line, macd_sig, macd_hist = macd(closes,12,26,9)
    rsi_vals = rsi(closes,14)
    boll = bollinger(closes,20,2.0)
    supports,resistances = detect_support_resistance(candles, window=15)
    reasons=[]
    score=0
    last=candles[-1]; prev=candles[-2] if len(candles)>1 else None
    if is_hammer(last): score+=1; reasons.append("Hammer (Bullish)")
    if prev and is_bullish_engulfing(prev,last): score+=2; reasons.append("Bullish Engulfing")
    if is_three_white_soldiers(candles): score+=2; reasons.append("Three White Soldiers")
    if is_shooting_star(last): score-=1; reasons.append("Shooting Star (Bearish)")
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
    ml_bias = 0.12 * (1 if ml_signal==1 else -1)
    prob = min(max(base_prob + ml_bias, 0.01), 0.99)
    stop_price, stop_pct, vol = calculate_dynamic_stop(closes[-1], candles, position_type="long")
    risk_pct = stop_pct*100.0
    if prob >= 0.65: rec = "Kaufen (Long empfohlen)"
    elif prob <= 0.35: rec = "Short / Verkaufen (Short empfohlen)"
    else: rec = "Halten / Beobachten"
    success_est = round(prob*100.0,1)
    # build 3-sentence summary
    sents=[]
    if rec.startswith("Kaufen"):
        sents.append(f"Algorithmen erkennen vorrangig bullishe Signale ({len(reasons)} Indikatoren).")
        sents.append(f"Empfohlener Stop-Loss: ca. {round(risk_pct,2)}% unter dem Kurs.")
        sents.append("Tipp: Erst kleine Position, Stop setzen und bei Bestätigung (Volumen, Follow-through) nachlegen.")
    elif rec.startswith("Short"):
        sents.append("Mehrere bärische Signale erkannt — erhöhte Vorsicht.")
        sents.append(f"Empfohlener Stop-Loss: ca. {round(risk_pct,2)}% oberhalb des Kurses.")
        sents.append("Tipp: Small size oder Absicherung; warte auf Bestätigung für Short-Entry.")
    else:
        sents.append("Kein klares Signal — Markt neutral.")
        sents.append("Beobachte Volumen, RSI und SMA-Kreuzungen für klares Setup.")
        sents.append("Tipp: Warte auf Breakout oder bestätigte Candle-Formation.")
    return {
        "markers": markers, "sma20": s20, "sma50": s50, "macd":(macd_line,macd_sig,macd_hist),
        "rsi":rsi_vals, "boll":boll, "supports":supports, "resistances":resistances,
        "recommendation": rec, "success_pct": success_est, "risk_pct": round(risk_pct,2),
        "prob": prob, "ml_signal":ml_signal, "reasons":reasons, "summary_sentences": sents
    }

# -------------------------
# Image Analyzer (advanced heuristics) - uses PIL when available
# -------------------------
def analyze_uploaded_chart(image_bytes):
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(image_bytes).convert("L")  # grayscale
        w,h = img.size
        maxw = 1400
        if w > maxw:
            img = img.resize((maxw, int(h*maxw/w)))
            w,h = img.size
        # enhance contrast a bit
        img = ImageOps.autocontrast(img, cutoff=1)
        # detect edges and vertical projection
        edges = img.filter(ImageFilter.FIND_EDGES)
        pix = img.load()
        col_sums=[0]*w
        for x in range(w):
            s=0
            for y in range(h):
                s += 255 - pix[x,y]
            col_sums[x]=s
        # smooth projection
        smooth = []
        for i in range(w):
            vals = col_sums[max(0,i-3):min(w,i+4)]
            smooth.append(sum(vals)/len(vals))
        # peaks detection
        peaks=[]
        avg = sum(smooth)/len(smooth) if len(smooth)>0 else 0
        for i in range(2,w-2):
            if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > avg*1.3:
                peaks.append(i)
        # minima detection
        minima=[]
        for i in range(2,w-2):
            if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < avg*0.7:
                minima.append(i)
        patterns=[]
        if len(peaks)>4:
            patterns.append("Candlestick-Chart erkannt")
        # look for W / M patterns (double bottom/top)
        if len(minima) >= 2:
            for j in range(len(minima)-1):
                gap = minima[j+1]-minima[j]
                if 8 < gap < 400:
                    patterns.append("Double Bottom / W-Pattern möglich")
                    break
        # trend heuristic: compare average brightness left vs right
        left = ImageStat.Stat(img.crop((0,0,w//2,h))).mean[0]
        right = ImageStat.Stat(img.crop((w//2,0,w,h))).mean[0]
        trend = "Seitwärts"
        if right < left - 6:
            trend = "Abwärtstrend"
        elif right > left + 6:
            trend = "Aufwärtstrend"
        # volume detection (if lower area exists)
        # return structure
        rec_notes=[]
        if "Double Bottom / W-Pattern möglich" in patterns:
            rec_notes.append("Möglicher Double-Bottom — Reversal Long möglich.")
        if trend=="Aufwärtstrend":
            rec_notes.append("Chart zeigt Aufwärtstrend — Long bias.")
        elif trend=="Abwärtstrend":
            rec_notes.append("Chart zeigt Abwärtstrend — Short bias.")
        if not rec_notes:
            rec_notes.append("Keine klare Struktur — manuelle Überprüfung empfohlen.")
        return {"patterns":patterns, "trend":trend, "notes":rec_notes, "peaks":len(peaks), "minima":len(minima)}
    except Exception:
        return None

# -------------------------
# Helper to get candles (live or sim)
# -------------------------
def get_candles_for(symbol: str, interval: str, periods: int, start_price: float, use_live: bool):
    sym = symbol.strip()
    if sym.lower() in KNOWN_SYMBOLS: sym = KNOWN_SYMBOLS[sym.lower()]
    if use_live and ALPHA_KEY and ONLINE:
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
    # fallback simulation
    tf_map = {"1min":1,"5min":5,"15min":15,"30min":30,"60min":60}
    mins = periods * tf_map.get(interval,5)
    prices = generate_price_walk(sym + "|" + interval, mins, start_price)
    ohlc = prices_to_ohlc(prices, candle_size=tf_map.get(interval,5))
    if len(ohlc) < periods:
        pad = periods - len(ohlc)
        pad_item = ohlc[0] if ohlc else {"open":start_price,"high":start_price,"low":start_price,"close":start_price,"volume":0}
        ohlc = [pad_item]*pad + ohlc
    return ohlc[-periods:], False

# -------------------------
# Page UI
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live Analyzer","Bild-Analyse","Backtest & ML","Portfolio","Einstellungen","Hilfe"])

# quick status
if not internet_ok():
    st.sidebar.error("❌ Keine Internetverbindung — Offline-Fallback aktiv")
elif not ALPHA_KEY:
    st.sidebar.warning("⚠️ API-Key nicht gesetzt — Offline-Fallback aktiv")
else:
    st.sidebar.success("✅ Live-Mode möglich (Alpha Vantage)")

# Ensure session model
if "model" not in st.session_state:
    st.session_state.model = load_model()

# -------------------------
# Home
# -------------------------
if page == "Home":
    st.header("Lumina Pro — Übersicht")
    st.markdown("Kurzübersicht deines Portfolios, schnelle Analyse und Aktionen.")
    # quick portfolio summary
    port = load_json(PORTFOLIO_FILE) or []
    st.subheader("Portfolio (gespeichert)")
    if not port:
        st.info("Portfolio leer.")
    else:
        for p in port:
            st.write(p)
    st.markdown("---")
    st.subheader("Quick Analyzer")
    col1,col2 = st.columns([3,1])
    with col2:
        sym_quick = st.text_input("Symbol", value="AAPL")
        interval_q = st.selectbox("Intervall", ["1min","5min","15min","30min","60min"], index=1)
        periods_q = st.slider("Candles", 30, 600, 240, step=10)
        start_price_q = st.number_input("Startpreis fallback", value=100.0)
        live_q = st.checkbox("Live falls möglich", value=True)
        if st.button("Analysiere jetzt (Quick)"):
            st.session_state["do_quick"]=True
    with col1:
        if st.session_state.get("do_quick", False):
            candles, live_flag = get_candles_for(sym_quick, interval_q, periods_q, start_price_q, live_q)
            model = st.session_state.model
            analysis = analyze_and_estimate(candles, model)
            stop_auto, stop_pct_auto, vol = calculate_dynamic_stop([c["close"] for c in candles][-1], candles, position_type="long") if False else (None,None,None)
            stop_line={"price":None,"type":"long","pct":0}
            svg = render_candles_svg(candles, markers=analysis["markers"], stop_line=stop_line, sma_periods=(20,50), rsi_vals=analysis["rsi"], boll=analysis["boll"], supports=analysis["supports"], resistances=analysis["resistances"])
            st.components.v1.html(svg, height=600)
            st.markdown(f"**Empfehlung:** {analysis['recommendation']}")
            st.markdown(f"**Erfolgsschätzung:** {analysis['success_pct']}% | Risiko: {analysis['risk_pct']}%")
            st.markdown("**Kurzbeschreibung:**")
            for s in analysis["summary_sentences"]:
                st.write("- " + s)
            st.markdown("---")
            if live_flag: st.info("Live data used.")
            else: st.info("Offline / cache used.")

# -------------------------
# Live Analyzer page
# -------------------------
elif page == "Live Analyzer":
    st.header("Live Analyzer — Candle Analysis mit Empfehlung")
    left, right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol oder Asset-Name", value="AAPL")
        interval = st.selectbox("Intervall", ["1min","5min","15min","30min","60min"], index=1)
        periods = st.slider("Kerzen", 30, 800, 240, step=10)
        start_price = st.number_input("Startpreis fallback", value=100.0)
        use_live = st.checkbox("Live via Alpha Vantage (wenn Key & online)", value=True)
        run = st.button("Lade & Analysiere")
        st.markdown("---")
        st.write("Model status:", "geladen" if st.session_state.model else "keins")
        if st.button("Train Perceptron (synthetic)"):
            st.info("Training Perceptron (synthetic dataset)...")
            X=[]; y=[]
            for i in range(700):
                base = generate_price_walk("train"+str(i), 300, 100.0)
                ohlc = prices_to_ohlc(base, candle_size=5)
                label = random.choice([1,-1])
                if label==1:
                    for k in range(3):
                        ohlc[-1-k]["open"]=ohlc[-1-k]["close"]*(1-random.uniform(0.01,0.03))
                        ohlc[-1-k]["close"]=ohlc[-1-k]["open"]*(1+random.uniform(0.01,0.05))
                else:
                    for k in range(3):
                        ohlc[-1-k]["open"]=ohlc[-1-k]["close"]*(1+random.uniform(0.01,0.04))
                        ohlc[-1-k]["close"]=ohlc[-1-k]["open"]*(1-random.uniform(0.01,0.05))
                feat = features_from_candles(ohlc)
                X.append(feat); y.append(label)
            p = SimplePerceptron(len(X[0])); p.train(X,y,epochs=80); save_model(p); st.session_state.model=p; st.success("Perceptron trained.")
    with left:
        if run:
            candles, live_flag = get_candles_for(symbol, interval, periods, start_price, use_live)
            model = st.session_state.model
            analysis = analyze_and_estimate(candles, model)
            stop_price, stop_pct, vol = calculate_dynamic_stop(candles[-1]["close"], candles, position_type="long")
            stop_line = {"price":stop_price, "type":"long", "pct":stop_pct}
            supports,resistances = analysis["supports"], analysis["resistances"]
            svg = render_candles_svg(candles, markers=analysis["markers"], stop_line=stop_line, sma_periods=(20,50), rsi_vals=analysis["rsi"], boll=analysis["boll"], supports=supports, resistances=resistances)
            st.components.v1.html(svg, height=640)
            st.subheader("Empfehlung & Erklärung")
            st.markdown(f"**Empfehlung:** {analysis['recommendation']}")
            st.markdown(f"**Erfolgschance (Schätzung):** **{analysis['success_pct']}%**")
            st.markdown(f"**Risiko (empf. Stop-Loss Abstand):** **{analysis['risk_pct']}%**")
            st.markdown("**Kurze Erklärung (3 Sätze):**")
            for s in analysis["summary_sentences"]: st.write("- " + s)
            st.markdown("**Erkannte Gründe / Muster:**")
            for r in analysis["reasons"]: st.write("- " + r)
            if live_flag: st.info("Live Daten verwendet (Alpha Vantage).")
            else: st.info("Offline / Cache verwendet.")

# -------------------------
# Bild-Analyse page
# -------------------------
elif page == "Bild-Analyse":
    st.header("Chart-Bild-Analyse — Upload & Analyse")
    st.markdown("Lade ein Chart-Screenshot hoch. Die App analysiert Muster, Trend und gibt Empfehlungen.")
    uploaded = st.file_uploader("Bild hochladen (PNG/JPG)", type=["png","jpg","jpeg"])
    if uploaded is None:
        st.info("Kein Bild hochgeladen. Wenn Pillow fehlt, installiere `pillow` in requirements.txt für automatische Analyse.")
        if not PIL_AVAILABLE:
            st.warning("Pillow nicht installiert — automatische Bildanalyse deaktiviert.")
            st.markdown("Füge in requirements.txt: `pillow`")
    else:
        st.image(uploaded, use_column_width=True)
        if not PIL_AVAILABLE:
            st.error("Automatische Bildanalyse nicht verfügbar (Pillow fehlt). Bitte installiere pillow.")
            tag = st.selectbox("Manuelle Muster-Auswahl", ["Kein Muster","Hammer","Doji","Engulfing (bull)","Engulfing (bear)","Double Top","Double Bottom","Head & Shoulders"])
            notes = st.text_area("Notiz (optional)")
            if st.button("Analysiere manuell"):
                map_rec = {
                    "Hammer":("Kaufen","Hammer an Unterstützung, mögliches Reversal","Stop unter Low setzen"),
                    "Doji":("Beobachten","Indecision candle — Volumen prüfen","Warte auf Bestätigung"),
                    "Engulfing (bull)":("Kaufen","Bullish Engulfing — starkes Kaufsignal","Stop unter Hoch des vorherigen Lows"),
                    "Engulfing (bear)":("Short","Bearish Engulfing — Verkaufssignal","Stop über Hoch setzen"),
                    "Double Top":("Short","Double Top — Trendwende möglich","Warte auf Nackenbruch"),
                    "Double Bottom":("Kaufen","Double Bottom — Reversal möglich","Stop unter Support"),
                    "Head & Shoulders":("Short","H&S — in der Regel bärisch","Stop oberhalb der rechten Schulter"),
                    "Kein Muster":("Halten","Kein eindeutiges Muster","Weitere Daten/Analyse notwendig")
                }
                rec, expl, tip = map_rec.get(tag, ("Halten","Keine Info","Keine Aktion"))
                st.success(f"Empfehlung: {rec}")
                st.write(expl); st.write(tip)
        else:
            res = analyze_uploaded_chart(uploaded)
            if not res:
                st.error("Bildanalyse fehlgeschlagen.")
            else:
                st.subheader("Analyse Ergebnis")
                st.write("Erkannte Muster:", res.get("patterns", []))
                st.write("Trend:", res.get("trend"))
                st.write("Hinweise:")
                for n in res.get("notes", []): st.write("- " + n)
                st.write(f"Peaks detected: {res.get('peaks')}, minima: {res.get('minima')}")

# -------------------------
# Backtest & ML
# -------------------------
elif page == "Backtest & ML":
    st.header("Backtest & Model Training")
    left,right = st.columns([3,1])
    with right:
        strategy = st.selectbox("Strategie", ["SMA Crossover","MACD Momentum","RSI Bounce","Breakout"])
        initial_cash = st.number_input("Initial Cash", value=10000.0, step=100.0)
        run_bt = st.button("Backtest ausführen")
        if st.button("Train Perceptron (synthetic)"):
            st.info("Training Perceptron mit synthetischen Mustern...")
            X=[]; y=[]
            for i in range(800):
                base = generate_price_walk("train"+str(i), 300, 100.0)
                ohlc = prices_to_ohlc(base, candle_size=5)
                label = random.choice([1,-1])
                if label==1:
                    for k in range(3):
                        ohlc[-1-k]["open"]=ohlc[-1-k]["close"]*(1-random.uniform(0.01,0.03))
                        ohlc[-1-k]["close"]=ohlc[-1-k]["open"]*(1+random.uniform(0.01,0.05))
                else:
                    for k in range(3):
                        ohlc[-1-k]["open"]=ohlc[-1-k]["close"]*(1+random.uniform(0.01,0.04))
                        ohlc[-1-k]["close"]=ohlc[-1-k]["open"]*(1-random.uniform(0.01,0.05))
                feat = features_from_candles(ohlc)
                X.append(feat); y.append(label)
            p = SimplePerceptron(len(X[0])); p.train(X,y,epochs=100); save_model(p); st.session_state.model = p
            st.success("Perceptron trained and saved.")
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
            st.subheader("Backtest Result")
            st.write(f"Total Return: {res['total_return']*100:.2f}%")
            st.write(f"Sharpe-like: {res['sharpe']:.4f}")
            st.write(f"Max Drawdown: {res['max_dd']*100:.2f}%")
            st.write(f"Winrate: {res['winrate']*100:.2f}%")
            if res.get("trades"):
                st.write("Trades (first 10):")
                for t in res["trades"][:10]: st.write(t)
            st.line_chart(res["equity"])
            if live_flag: st.info("Live data used")
            else: st.info("Offline simulation")

# -------------------------
# Portfolio
# -------------------------
elif page == "Portfolio":
    st.header("Portfolio")
    port = load_json(PORTFOLIO_FILE) or []
    if not port: st.info("Portfolio empty")
    else:
        for idx,p in enumerate(port):
            st.write(f"{idx+1}. {p}")
    if st.button("Export Portfolio"):
        st.download_button("Download", data=json.dumps(port, ensure_ascii=False, indent=2), file_name="portfolio.json")
    if st.button("Clear Portfolio"):
        save_json(PORTFOLIO_FILE, []); st.success("Cleared")

# -------------------------
# Settings
# -------------------------
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Alpha Vantage Key (optional override):")
    key_input = st.text_input("Alpha Key", value=ALPHA_KEY if ALPHA_KEY else "")
    if st.button("Speichere Key (in Memory for this session)"):
        if key_input.strip():
            ALPHA_KEY = key_input.strip()
            st.session_state["ALPHA_KEY"] = ALPHA_KEY
            st.success("Key stored in session. For persistent storage, use streamlit secrets.")
    st.markdown("---")
    st.write("Pillow installed:", "Yes" if PIL_AVAILABLE else "No")
    if not PIL_AVAILABLE:
        st.info("Install pillow in requirements.txt to enable image analysis.")
    if st.button("Clear Cache"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR,f))
            except: pass
        st.success("Cache cleared")

# -------------------------
# Help
# -------------------------
elif page == "Hilfe":
    st.header("Hilfe & Hinweise")
    st.markdown("""
    **Wichtige Hinweise**
    - Alpha Vantage free tier: ~5 calls/min, 500 calls/day. The app caches responses in `.cache_av`.
    - Erfolgswahrscheinlichkeit ist eine Schätzung — KEINE Anlageberatung.
    - Für Bildanalyse installiere `pillow` (optional) in `requirements.txt`.
    - Wenn du API-Keys sicher verwahren willst, verwende Streamlit Secrets.
    """)
    st.markdown("**Empfohlene next steps:**")
    st.write("- Fees & Slippage in Backtest einbauen")
    st.write("- Höhere Qualität ML (scikit-learn / lightGBM) — benötigt zusätzliche libs")
    st.write("- Bessere Bildanalyse mit CNN (erfordert TensorFlow/PyTorch)")

st.markdown("---")
st.caption("Lumina Pro — Deep Analyzer — Live via Alpha Vantage (if configured); offline fallback available. Recommendations are probabilistic estimates and not financial advice.")
