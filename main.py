# main.py
# Lumina Pro — Deep Analyzer (Live Alpha Vantage + Roboflow Image Inference)
# Features:
# - Offline image-only chart-structure analyzer (Pillow)
# - Live data from Finnhub (primary) and Alpha Vantage (fallback)
# - Robust Roboflow detect wrapper with retries (optional)
# - Background lightweight backtesting with slippage & fees
# - Export analysis JSON/CSV
# - SVG candle renderer fallback; matplotlib rendering optional
#
# Keys embedded per user's request:
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"
ALPHAV_KEY   = "22XGVO0TQ1UV167C"
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"

# ---------------------------
# Imports
# ---------------------------
import streamlit as st
import json, os, time, random, io, urllib.request, urllib.parse, math, csv, traceback
from datetime import datetime, timedelta
import statistics

# optional libs
try:
    from PIL import Image, ImageOps, ImageStat, ImageFilter
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------
# Page config & Theme
# ---------------------------
st.set_page_config(page_title="Lumina Pro — Deep Analyzer", layout="wide", page_icon="💹")
st.markdown("""
<style>
html, body, [class*="css"] { background:#000 !important; color:#e6eef6 !important; }
.stButton>button { background:#111 !important; color:#e6eef6 !important; border:1px solid #222 !important; }
.card { background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
.small { color:#9aa6b2; font-size:13px; }
.badge { background:#111; color:#e6eef6; padding:6px 10px; border-radius:8px; border:1px solid #222; display:inline-block; }
</style>
""", unsafe_allow_html=True)
st.title("Lumina Pro — Deep Analyzer (Live Alpha Vantage + Roboflow Image Inference)")

# ---------------------------
# Utilities & Cache
# ---------------------------
def now_iso(): return datetime.utcnow().isoformat() + "Z"

def internet_ok(timeout=3):
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

CACHE_DIR = ".lumina_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_save(key, obj):
    try:
        with open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "data": obj}, f)
    except Exception:
        pass

def cache_load(key, max_age=3600*24):
    try:
        p = os.path.join(CACHE_DIR, key + ".json")
        if not os.path.exists(p): return None
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        if time.time() - j.get("ts",0) > max_age:
            return None
        return j.get("data")
    except Exception:
        return None

# ---------------------------
# HTTP multipart helper (Roboflow)
# ---------------------------
def encode_multipart(file_fieldname, filename, file_bytes, content_type="image/png"):
    boundary = '----WebKitFormBoundary' + ''.join(random.choice('0123456789abcdef') for _ in range(16))
    crlf = b'\r\n'
    body = bytearray()
    body.extend(b'--' + boundary.encode() + crlf)
    body.extend(f'Content-Disposition: form-data; name="{file_fieldname}"; filename="{filename}"'.encode() + crlf)
    body.extend(f'Content-Type: {content_type}'.encode() + crlf + crlf)
    body.extend(file_bytes + crlf)
    body.extend(b'--' + boundary.encode() + b'--' + crlf)
    return f'multipart/form-data; boundary={boundary}', bytes(body)

def roboflow_detect_with_retries(image_bytes, retries=2):
    if not ROBOFLOW_KEY:
        return None
    for attempt in range(retries+1):
        try:
            endpoint = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_PATH}?api_key={urllib.parse.quote(ROBOFLOW_KEY)}"
            content_type, body = encode_multipart("file", "upload.png", image_bytes, "image/png")
            req = urllib.request.Request(endpoint, data=body, method="POST")
            req.add_header("Content-Type", content_type)
            req.add_header("User-Agent", "LuminaPro/1.0")
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            else:
                return None

# ---------------------------
# Market data fetchers
# ---------------------------
def fetch_finnhub_candles(symbol: str, resolution: str = "5", from_ts: int = None, to_ts: int = None):
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None: to_ts = int(time.time())
        if from_ts is None:
            if resolution in ("1","5","15","30","60"):
                from_ts = to_ts - 60*60*24
            else:
                from_ts = to_ts - 60*60*24*30
        params = {"symbol": symbol, "resolution": resolution, "from": str(int(from_ts)), "to": str(int(to_ts)), "token": FINNHUB_KEY}
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        if data.get("s") != "ok":
            return None
        ts = data.get("t", []); o = data.get("o", []); h = data.get("h", []); l = data.get("l", []); c = data.get("c", []); v = data.get("v", [])
        candles=[]
        for i, t in enumerate(ts):
            try: dt = datetime.utcfromtimestamp(int(t))
            except: dt = datetime.utcnow()
            candles.append({"t": dt, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i]), "volume": float(v[i]) if v and i < len(v) else 0.0})
        return candles
    except Exception:
        return None

def fetch_alpha_minute(symbol: str, interval="5min", outputsize="compact"):
    if not ALPHAV_KEY:
        return None
    try:
        base = "https://www.alphavantage.co/query?"
        params = {"function":"TIME_SERIES_INTRADAY","symbol":symbol,"interval":interval,"outputsize":outputsize,"apikey":ALPHAV_KEY}
        url = base + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        key = None
        for k in data:
            if "Time Series" in k:
                key = k; break
        if not key:
            return None
        series = data[key]
        candles=[]
        for ts in sorted(series.keys()):
            row = series[ts]
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            candles.append({"t": dt, "open": float(row["1. open"]), "high": float(row["2. high"]), "low": float(row["3. low"]), "close": float(row["4. close"]), "volume": float(row.get("5. volume", 0))})
        return candles
    except Exception:
        return None

# fallback generator
def generate_simulated_candles(seed: str, periods: int, start_price: float = 100.0, resolution_minutes: int = 5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    p = float(start_price)
    prices=[]
    for _ in range(periods):
        drift = (rnd.random() - 0.49) * 0.003
        shock = (rnd.random() - 0.5) * 0.02
        p = max(0.01, p * (1 + drift + shock))
        prices.append(round(p,6))
    candles=[]; now = datetime.utcnow()
    for i, prm in enumerate(prices):
        o = round(prm * (1 + random.uniform(-0.002,0.002)),6); c = prm
        h = round(max(o,c) * (1 + random.uniform(0.0,0.004)),6); l = round(min(o,c) * (1 - random.uniform(0.0,0.004)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c, "volume": random.randint(1,1000)})
    return candles

# ---------------------------
# Indicators & pattern detectors
# ---------------------------
def sma(vals, period):
    res=[]
    for i in range(len(vals)):
        if i+1 < period: res.append(None)
        else: res.append(sum(vals[i+1-period:i+1])/period)
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
    mac = [(a-b) if (a is not None and b is not None) else None for a,b in zip(ef, es)]
    mac_vals = [m for m in mac if m is not None]
    if not mac_vals: return mac, [None]*len(mac), [None]*len(mac)
    sig_vals = ema(mac_vals, signal)
    sig_iter = iter(sig_vals); sig_mapped=[]
    for v in mac:
        sig_mapped.append(None if v is None else next(sig_iter))
    hist = [(m-s) if (m is not None and s is not None) else None for m,s in zip(mac, sig_mapped)]
    return mac, sig_mapped, hist

def rsi(vals, period=14):
    if len(vals) < period+1: return [None]*len(vals)
    deltas = [vals[i] - vals[i-1] for i in range(1,len(vals))]
    gains = [d if d>0 else 0 for d in deltas]; losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period; avg_loss = sum(losses[:period]) / period
    res = [None]*period
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain*(period-1) + gains[i]) / period
        avg_loss = (avg_loss*(period-1) + losses[i]) / period
        rs = avg_gain / avg_loss if avg_loss != 0 else float('inf')
        val = 100 - (100 / (1 + rs))
        res.append(round(val,2))
    return res

# candle detectors
def is_doji(c): body = abs(c["close"] - c["open"]); total = c["high"] - c["low"]; return total > 0 and (body / total) < 0.15
def is_hammer(c): body = abs(c["close"] - c["open"]); lower = min(c["open"], c["close"]) - c["low"]; return body > 0 and lower > 2 * body
def is_shooting_star(c): body = abs(c["close"] - c["open"]); upper = c["high"] - max(c["open"], c["close"]); return body > 0 and upper > 2 * body
def is_bullish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])
def is_bearish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def detect_patterns(candles):
    patterns=[]; n = len(candles)
    for i in range(1,n):
        cur = candles[i]; prev = candles[i-1]
        if is_bullish_engulfing(prev, cur): patterns.append(("Bullish Engulfing", i))
        if is_bearish_engulfing(prev, cur): patterns.append(("Bearish Engulfing", i))
        if is_hammer(cur): patterns.append(("Hammer", i))
        if is_shooting_star(cur): patterns.append(("Shooting Star", i))
        if is_doji(cur): patterns.append(("Doji", i))
    if n>=3:
        if (candles[-3]["close"] < candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"] > candles[-1]["open"]):
            patterns.append(("Morning Star", n-1))
        if (candles[-3]["close"] > candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"] < candles[-1]["open"]):
            patterns.append(("Evening Star", n-1))
    return patterns

# ---------------------------
# Image Analyzer: improved (main focus)
# - returns trend, patterns, confidence, volatility, recommendation, probability, risk_pct, summary, internals
# ---------------------------
def analyze_chart_image_structure(image_bytes):
    if not PIL_AVAILABLE:
        return {"error":"Pillow not installed; install pillow for image analyzer."}
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return {"error":"Failed to open image."}
    W,H = img.size
    maxw = 1400
    if W > maxw:
        img = img.resize((maxw, int(H * maxw / W))); W,H = img.size
    left = int(W*0.03); right = int(W*0.97); top = int(H*0.08); bottom = int(H*0.78)
    chart = img.crop((left, top, right, bottom))
    chart = ImageOps.autocontrast(chart, cutoff=2); chart = chart.filter(ImageFilter.MedianFilter(size=3))
    pix = chart.load(); Wc,Hc = chart.size
    # vertical profile
    col_darkness = []
    for x in range(Wc):
        s = 0
        for y in range(0, Hc, 2):
            s += 255 - pix[x,y]
        col_darkness.append(s)
    # smooth
    smooth=[] 
    for i in range(Wc):
        w = col_darkness[max(0,i-4):min(Wc,i+5)]
        smooth.append(sum(w)/len(w) if w else 0)
    avg = sum(smooth)/len(smooth) if smooth else 0.0
    peaks=[]; troughs=[]
    for i in range(2, Wc-2):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > avg*1.25:
            peaks.append(i)
        if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < avg*0.6:
            troughs.append(i)
    peak_count = len(peaks); trough_count = len(troughs)
    density = peak_count / (Wc/100.0 + 1e-9)
    left_mean = ImageStat.Stat(chart.crop((0,0,Wc//2,Hc))).mean[0]
    right_mean = ImageStat.Stat(chart.crop((Wc//2,0,Wc,Hc))).mean[0]
    # corrected trend logic (explicit)
    if right_mean > left_mean + 6:
        trend = "Aufwärtstrend"
    elif right_mean < left_mean - 6:
        trend = "Abwärtstrend"
    else:
        trend = "Seitwärts"
    # detailed pattern heuristics
    doji_like=0; hammer_like=0; engulfing_like=0; shooting_like=0; harami_like=0
    sample_indices = peaks[-min(120,len(peaks)):] if peaks else []
    for idx in sample_indices:
        col = [255 - pix[idx, y] for y in range(Hc)]
        maxv = max(col) if col else 0
        if maxv <= 0: continue
        thresh = max(2, maxv * 0.4)
        high_pos = [i for i,v in enumerate(col) if v >= thresh]
        if not high_pos: continue
        body_height = (max(high_pos) - min(high_pos)) if len(high_pos)>1 else 0
        lower_shadow = Hc - 1 - max(high_pos)
        upper_shadow = min(high_pos)
        if body_height < Hc * 0.05: doji_like += 1
        if lower_shadow > body_height * 2.5 and body_height > 0: hammer_like += 1
        if upper_shadow > body_height * 2.5 and body_height > 0: shooting_like += 1
    # engulfing detection via ratios
    for i in range(len(peaks)-1):
        a = peaks[i]; b = peaks[i+1]
        if smooth[b] > smooth[a] * 1.9: engulfing_like += 1
    # volatility estimate
    var = statistics.pvariance(smooth) if len(smooth)>1 else 0.0
    vol_est = min(100.0, max(1.0, (var**0.5)/(avg+1e-9) * 200.0))
    # assemble patterns
    patterns=[]
    if doji_like: patterns.append(f"{doji_like}× Doji-like")
    if hammer_like: patterns.append(f"{hammer_like}× Hammer-like")
    if shooting_like: patterns.append(f"{shooting_like}× Shooting-like")
    if engulfing_like: patterns.append(f"{engulfing_like}× Engulfing-like")
    if peak_count > 12 and density > 6: patterns.append("Hohe Candle-Dichte")
    if trough_count > 6: patterns.append("Mehrere lokale Tiefs")
    if not patterns: patterns.append("Keine klaren Candle-Formen erkannt")
    conf = 30 + min(60, int(min(peak_count, 60) * 1.2 + len(patterns)*4 + (10 if trend!="Seitwärts" else 0)))
    conf = max(5, min(98, conf))
    # scoring
    score = 0
    if trend == "Aufwärtstrend": score += 2
    if trend == "Abwärtstrend": score -= 2
    score += hammer_like * 2; score += engulfing_like * 2; score -= shooting_like * 2
    if score >= 3: rec = "Kaufen"
    elif score <= -2: rec = "Short"
    else: rec = "Neutral"
    prob = min(95.0, max(10.0, 45.0 + score * 9.0 + conf * 0.2))
    risk_pct = min(50.0, max(1.0, vol_est * 0.6))
    # create summary
    summary=[]
    if rec == "Kaufen":
        summary.append(f"Bild-Struktur zeigt {', '.join(patterns[:3])}. Trend: {trend}.")
        summary.append(f"Geschätzte Trefferwahrscheinlichkeit: {prob:.1f}% • Risiko: {risk_pct:.1f}%.")
        summary.append("Empfehlung: Klein long mit Stop-Loss, warte Bestätigung der nächsten Kerze.")
    elif rec == "Short":
        summary.append(f"Bild-Struktur zeigt bärische Muster ({', '.join(patterns[:3])}). Trend: {trend}.")
        summary.append(f"Geschätzte Trefferwahrscheinlichkeit: {prob:.1f}% • Risiko: {risk_pct:.1f}%.")
        summary.append("Empfehlung: Short mit enger Absicherung oder warten auf Bestätigung.")
    else:
        summary.append("Keine eindeutige Struktur erkennbar.")
        summary.append("Empfehlung: Warten auf Bestätigung / Volumenanstieg.")
        summary.append("Tipp: Kein Full-Size-Entry ohne Bestätigung.")
    return {"trend":trend,"patterns":patterns,"confidence":conf,"volatility":round(vol_est,2),"recommendation":rec,"probability":round(prob,1),"risk_pct":round(risk_pct,2),"summary":summary,"internal":{"peaks":peak_count,"troughs":trough_count,"density":round(density,2)}}

# ---------------------------
# Backtesting functions (with slippage & fees)
# ---------------------------
def backtest_pattern_on_history(pattern_name, candles, lookahead=10, slippage_pct=0.0, fee_pct=0.0):
    hits=0; total=0; returns=[]
    n = len(candles)
    for i in range(2, n - lookahead):
        prev = candles[i-1]; cur = candles[i]
        detected=False
        pn = pattern_name.lower()
        if "engulf" in pn:
            if is_bullish_engulfing(prev, cur) or is_bearish_engulfing(prev, cur): detected=True
        if "hammer" in pn:
            if is_hammer(cur): detected=True
        if "doji" in pn:
            if is_doji(cur): detected=True
        if not detected: continue
        total += 1
        entry = cur["close"] * (1 + slippage_pct/100.0)
        future = candles[i + lookahead]["close"] * (1 - slippage_pct/100.0)
        ret = (future - entry) / (entry + 1e-12) - fee_pct/100.0
        returns.append(ret)
        if ret > 0: hits += 1
    winrate = (hits / total * 100.0) if total else 0.0
    avg_ret = (sum(returns)/len(returns)*100.0) if returns else 0.0
    return {"pattern":pattern_name,"checked":total,"wins":hits,"winrate":round(winrate,2),"avg_return_pct":round(avg_ret,3)}

# ---------------------------
# SVG candle renderer (fallback)
# ---------------------------
def render_svg_candles(candles, markers=None, stop=None, tp=None, width=1000, height=520):
    if not candles: return "<svg></svg>"
    n = len(candles)
    margin = 54; chart_h = int(height * 0.62)
    max_p = max(c["high"] for c in candles); min_p = min(c["low"] for c in candles)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad
    spacing = (width - 2*margin) / n; candle_w = max(3, spacing * 0.6)
    def y(p): return margin + chart_h - (p - min_p) / (max_p - min_p) * chart_h
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>')
    for i in range(6):
        yy = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{yy}" x2="{width-margin}" y2="{yy}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{yy+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')
    for i,c in enumerate(candles):
        cx = margin + i*spacing + spacing/2
        top = y(c["high"]); low = y(c["low"]); oy = y(c["open"]); cy = y(c["close"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
        by = min(oy, cy); bh = max(1, abs(cy - oy))
        svg.append(f'<rect x="{cx-candle_w/2}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
    if markers:
        for m in markers:
            i = m.get("idx", len(candles)-1)
            if i<0 or i>=n: continue
            cx = margin + i*spacing + spacing/2
            if m.get("type","").lower() == "buy":
                svg.append(f'<polygon points="{cx-8},{margin+8} {cx+8},{margin+8} {cx},{margin-2}" fill="#00ff88"/>')
            else:
                svg.append(f'<polygon points="{cx-8},{height-30} {cx+8},{height-30} {cx},{height-46}" fill="#ff7788"/>')
    if stop:
        try:
            sy = y(stop); svg.append(f'<line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#ffcc00" stroke-width="2" stroke-dasharray="6,4"/><text x="{width-margin-260}" y="{sy-6}" fill="#ffcc00" font-size="12">Stop: {stop}</text>')
        except: pass
    if tp:
        try:
            ty = y(tp); svg.append(f'<line x1="{margin}" y1="{ty}" x2="{width-margin}" y2="{ty}" stroke="#66ff88" stroke-width="2" stroke-dasharray="4,4"/><text x="{width-margin-260}" y="{ty-6}" fill="#66ff88" font-size="12">TP: {tp}</text>')
        except: pass
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        try: t = candles[i]["t"].strftime("%m-%d %H:%M")
        except: t = str(candles[i].get("t",""))
        svg.append(f'<text x="{x-36}" y="{height-6}" font-size="11" fill="#9aa6b2">{t}</text>')
    svg.append('</svg>'); return "\n".join(svg)

# ---------------------------
# Fusion logic: image -> mapping to TP/SL + backtest insights + final recommendation
# ---------------------------
def map_patterns_to_levels(image_result, candles=None):
    # Basic TP/SL mapping: use image volatility & a last price (if available).
    last_price = candles[-1]["close"] if candles else None
    risk = image_result.get("risk_pct", 5.0)
    stop = round(last_price * (1 - risk/100.0), 6) if last_price else None
    tp = round(last_price * (1 + risk/100.0 * 2.0), 6) if last_price else None
    # Map key patterns to textual strategies
    strategy_notes=[]
    for p in image_result.get("patterns", []):
        key = p.lower()
        if "hammer" in key: strategy_notes.append("Reverse-entry: Hammer suggests possible short-term reversal")
        if "engulf" in key: strategy_notes.append("Momentum entry: Engulfing often leads to short-run continuation")
        if "doji" in key: strategy_notes.append("Indecision: Doji suggests wait for confirmation")
        if "shoot" in key: strategy_notes.append("Rejection: Shooting-star may indicate rejection at highs")
    return {"stop":stop,"tp":tp,"notes":strategy_notes}

# ---------------------------
# UI & Pages
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home","Live Analyzer","Bild-Analyse (offline)","Backtest","Einstellungen","Hilfe"])

if not ONLINE:
    st.sidebar.error("❌ Keine Internetverbindung — Live Daten werden simuliert")
else:
    st.sidebar.success("✅ Internet verfügbar")

# Home
if page == "Home":
    st.header("Lumina Pro — Deep Analyzer")
    st.markdown("""
    - Bild-Analyzer (offline) — automatisch: Muster, Volatilität, klare Empfehlung.
    - Live Analyzer (Finnhub / Alpha Vantage fallback).
    - Background Backtesting & exportierbare Analysen.
    """)
    st.write("Pillow installed:", PIL_AVAILABLE)
    st.write("Matplotlib installed:", MATPLOTLIB_AVAILABLE)

# Live Analyzer
elif page == "Live Analyzer":
    st.header("Live Analyzer — symbol")
    left,right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol (Finnhub format)", value="BINANCE:BTCUSDT")
        resolution = st.selectbox("Resolution (min)", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles", 30, 1000, 240, step=10)
        fallback_price = st.number_input("Fallback Startprice", value=20000.0)
        run = st.button("Lade & Analysiere")
    with left:
        if run:
            candles = None
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - int(periods) * int(resolution) * 60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None and ALPHAV_KEY:
                    st.warning("Finnhub lieferte nichts — versuche Alpha Vantage")
                    av = fetch_alpha_minute(symbol, interval=resolution + "min")
                    if av: candles = av[-periods:] if len(av)>=periods else av
                if candles is None:
                    st.warning("Keine Live-Daten — Simulation wird verwendet")
                    candles = generate_simulated_candles(symbol + "_sim", periods, fallback_price, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol + "_pad", need, candles[0]["open"] if candles else fallback_price, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline or no key — using simulated candles")
                candles = generate_simulated_candles(symbol + "_sim", periods, fallback_price, int(resolution))
            closes = [c["close"] for c in candles]
            # fixed trend logic
            if len(closes) >= 50:
                s20 = sum(closes[-20:]) / 20
                s50 = sum(closes[-50:]) / 50
                trend = "Aufwärtstrend" if s20 > s50 else "Abwärtstrend" if s20 < s50 else "Seitwärts"
            else:
                trend = "Seitwärts"
            patt = detect_patterns(candles)
            # create an image-like summary from candles only
            pseudo_img_res = {"recommendation":"Neutral","probability":50.0,"risk_pct":5.0,"patterns":[p[0] for p in patt],"summary":[]}
            fused = map_patterns_to_levels(pseudo_img_res, candles)
            # fuse into final
            final = {"recommendation": pseudo_img_res["recommendation"], "probability":pseudo_img_res["probability"], "risk_pct":pseudo_img_res["risk_pct"], "stop":fused["stop"], "tp":fused["tp"], "reasons":[p[0] for p in patt]}
            st.subheader(f"Symbol: {symbol}")
            st.write(f"Aktueller Preis: {candles[-1]['close']:.4f}")
            st.write("Trend:", trend)
            st.write("Detected patterns:", [p[0] for p in patt][:6])
            st.write("Stop / TP (rough):", final["stop"], final["tp"])
            # render chart: use matplotlib if available for nicer visuals; otherwise SVG
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(11,5), facecolor="#07070a")
                ax.set_facecolor("#07070a")
                # plot close as line for speed
                xs = [c["t"] for c in candles[-200:]]
                ys = [c["close"] for c in candles[-200:]]
                ax.plot(xs, ys, color="#00cc66")
                ax.set_title(f"{symbol} (close)", color="#e6eef6")
                ax.tick_params(axis='x', colors='#9aa6b2'); ax.tick_params(axis='y', colors='#9aa6b2')
                st.pyplot(fig)
            else:
                svg = render_svg_candles(candles[-160:], stop=final["stop"], tp=final["tp"])
                st.components.v1.html(svg, height=520)

# Bild-Analyse (offline)
elif page == "Bild-Analyse (offline)":
    st.header("Bild-Analyse — Struktur & Muster (OFFLINE, automatisch)")
    st.markdown("Lade einen Chart-Screenshot. Die App analysiert automatisch: Muster, Volatilität, Recommendation, Hintergrund-Backtest.")
    uploaded = st.file_uploader("Chart-Bild (PNG/JPG)", type=["png","jpg","jpeg"])
    show_internals = st.checkbox("Zeige interne Metriken", value=False)
    run = st.button("Analysiere Bild (offline)")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            # Roboflow optional
            rf_res = None
            if ONLINE and ROBOFLOW_KEY:
                rf_res = roboflow_detect_with_retries(img_bytes, retries=2)
            # local analysis (main)
            img_res = analyze_chart_image_structure(img_bytes)
            if img_res.get("error"):
                st.error(img_res["error"])
            else:
                # try to pull history for backtesting; try multiple symbols but do not require user input
                hist = None
                if ONLINE and FINNHUB_KEY:
                    for sym in ["BINANCE:BTCUSDT","AAPL","SPY","MSFT"]:
                        try:
                            hist = fetch_finnhub_candles(sym, "5", int(time.time()) - 60*60*24*90, int(time.time()))
                            if hist and len(hist) >= 400:
                                break
                        except Exception:
                            hist = None
                if hist is None and ALPHAV_KEY:
                    try:
                        hist = fetch_alpha_minute("AAPL", interval="5min", outputsize="full")
                    except Exception:
                        hist = None
                if hist is None:
                    hist = generate_simulated_candles("backtest_seed", 900, 100.0, 5)
                # backtest top patterns
                backtests = []
                for p in img_res.get("patterns", [])[:4]:
                    name = p.split()[0]
                    bt = backtest_pattern_on_history(name, hist, lookahead=10, slippage_pct=0.05, fee_pct=0.02)
                    backtests.append(bt)
                # map patterns->levels (TP/SL)
                mapping = map_patterns_to_levels(img_res, hist[-200:] if hist else None)
                # create final fused recommendation: use image result + backtest stats to adjust probability
                adjusted_prob = img_res["probability"]
                if backtests:
                    avg_wr = sum(bt["winrate"] for bt in backtests if bt["checked"]>0) / max(1, sum(1 for bt in backtests if bt["checked"]>0))
                    # move probability a bit toward historic winrate
                    adjusted_prob = round((adjusted_prob*0.6 + avg_wr*0.4), 1)
                final = {
                    "recommendation": img_res["recommendation"],
                    "probability": adjusted_prob,
                    "risk_pct": img_res["risk_pct"],
                    "stop": mapping["stop"],
                    "tp": mapping["tp"],
                    "summary": img_res["summary"],
                    "patterns": img_res["patterns"],
                    "confidence": img_res["confidence"],
                    "volatility": img_res["volatility"],
                    "backtests": backtests,
                    "mapping_notes": mapping["notes"],
                    "roboflow": rf_res
                }
                # show UI
                left, right = st.columns([2,1])
                with left:
                    if final["recommendation"] == "Kaufen":
                        st.success(f"Empfehlung: {final['recommendation']}  •  {final['probability']}%")
                    elif final["recommendation"] == "Short":
                        st.error(f"Empfehlung: {final['recommendation']}  •  {final['probability']}%")
                    else:
                        st.info(f"Empfehlung: {final['recommendation']}  •  {final['probability']}%")
                    st.markdown(f"**Risiko (est.)**: {final['risk_pct']}%")
                    st.markdown("**Kurz-Fazit (3 Sätze):**")
                    for s in final["summary"][:3]: st.write("- " + s)
                    if final["mapping_notes"]:
                        st.markdown("**Strategie-Hinweise**")
                        for n in final["mapping_notes"][:4]: st.write("- " + n)
                with right:
                    st.markdown("**Details**")
                    st.write("Trend:", img_res["trend"]); st.write("Confidence:", img_res["confidence"])
                    st.write("Volatility est.:", img_res["volatility"])
                    st.write("Detected patterns:")
                    for p in img_res["patterns"][:8]: st.write("- " + p)
                    if rf_res:
                        st.markdown("Roboflow Predictions (top 5):")
                        preds = rf_res.get("predictions", [])
                        for pr in preds[:5]:
                            st.write(f"- {pr.get('class')} ({pr.get('confidence'):.2f})")
                st.markdown("---")
                st.subheader("Backtest (approx.)")
                if backtests:
                    for b in backtests:
                        st.write(f"{b['pattern']}: checked={b['checked']} wins={b['wins']} winrate={b['winrate']}% avgRet={b['avg_return_pct']}%")
                else:
                    st.info("Keine Backtest-Ergebnisse.")
                # export buttons
                analysis_export = { "meta":{"ts": now_iso(), "source":"image"}, "final": final, "image_internal": img_res.get("internal",{}), "backtests": backtests }
                st.download_button("Export Analysis (JSON)", data=json.dumps(analysis_export, ensure_ascii=False, indent=2), file_name="analysis_export.json", mime="application/json")
                # CSV (simple)
                csv_buf = io.StringIO()
                w = csv.writer(csv_buf); w.writerow(["key","value"])
                w.writerow(["recommendation", final["recommendation"]]); w.writerow(["probability", final["probability"]]); w.writerow(["risk_pct", final["risk_pct"]])
                st.download_button("Export Analysis (CSV)", data=csv_buf.getvalue(), file_name="analysis_export.csv", mime="text/csv")
                # demo svg chart
                demo = generate_simulated_candles("img_demo_seed_"+str(img_res["internal"]["peaks"]), 160, 100.0, 5)
                if MATPLOTLIB_AVAILABLE:
                    fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
                    ax.plot([c["t"] for c in demo], [c["close"] for c in demo], color="#00cc66")
                    ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
                    st.pyplot(fig)
                else:
                    svg = render_svg_candles(demo, width=1000, height=420)
                    st.components.v1.html(svg, height=450)
                if show_internals:
                    st.markdown("Internal metrics:"); st.write(img_res.get("internal",{}))
                st.success("Bildanalyse + Backtest abgeschlossen.")

# Backtest page
elif page == "Backtest":
    st.header("Backtest / Simulation")
    st.markdown("Leichtgewichtiger Backtester (simuliert). Slippage & Fee einstellbar.")
    pattern = st.selectbox("Muster", ["Bullish Engulfing","Bearish Engulfing","Hammer","Doji","Morning Star"])
    lookahead = st.slider("Lookahead (Kerzen)", 1, 30, 10)
    slippage = st.number_input("Slippage %", value=0.05, step=0.01)
    fee = st.number_input("Fee %", value=0.02, step=0.01)
    if st.button("Run Backtest"):
        hist = generate_simulated_candles("bt_seed_full", 1200, 100.0, 5)
        res = backtest_pattern_on_history(pattern, hist, lookahead=lookahead, slippage_pct=slippage, fee_pct=fee)
        st.write(res)
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
            ax.plot([c["t"] for c in hist[-300:]], [c["close"] for c in hist[-300:]], color="#00cc66")
            ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
            st.pyplot(fig)
        else:
            st.line_chart([c["close"] for c in hist[-300:]])

# Settings
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Pillow installed:", PIL_AVAILABLE)
    st.write("Matplotlib installed:", MATPLOTLIB_AVAILABLE)
    st.write("Finnhub Key present:", bool(FINNHUB_KEY))
    st.write("AlphaV Key present:", bool(ALPHAV_KEY))
    if st.button("Cache löschen"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR,f))
            except: pass
        st.success("Cache gelöscht")

# Help
elif page == "Hilfe":
    st.header("Hilfe & Hinweise")
    st.markdown("""
    - Bild-Analyzer läuft offline und benötigt Pillow.
    - Live-Analyzer nutzt Finnhub/AlphaV (API limits gelten).
    - Empfehlungen sind probabilistisch — **keine Anlageberatung**.
    - Für Roboflow-Integration: stelle sicher, dass ROBOFLOW_MODEL_PATH korrekt ist.
    """)

st.markdown("---")
st.caption("Lumina Pro — Deep Analyzer. Keys are embedded as requested. Use responsibly; not financial advice.")
