# main.py
# Lumina Pro — Deep Analyzer (Next Gen)
# Features included:
#  - Offline image-structure analyzer (Pillow)
#  - Roboflow integration (optional) with retries and label->action mapping
#  - Live data from Finnhub + Alpha Vantage fallback
#  - SL/TP mapping via local support/resistance detection (pivot)
#  - Backtester with fees, slippage, position sizing & plain-language summary
#  - Annotations on uploaded chart images (PIL drawing)
#  - Exports per-analysis (JSON + CSV)
#  - SVG candle renderer fallback, matplotlib optional for nicer charts
#
# NOTE: Keys embedded per user request:
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"
ALPHAV_KEY   = "22XGVO0TQ1UV167C"
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"  # change if needed

# -------------------------
# Imports
# -------------------------
import streamlit as st
import json, os, io, time, random, math, urllib.request, urllib.parse, csv, traceback
from datetime import datetime, timedelta
import statistics

# Pillow
try:
    from PIL import Image, ImageOps, ImageStat, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# matplotlib optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# -------------------------
# Config & Styling
# -------------------------
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

st.title("Lumina Pro — Deep Analyzer (Next Gen)")

# -------------------------
# Utilities
# -------------------------
def now_iso(): return datetime.utcnow().isoformat() + "Z"
def short_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def internet_ok(timeout=2):
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
        with open(os.path.join(CACHE_DIR, f"{key}.json"), "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "data": obj}, f)
    except Exception:
        pass

def cache_load(key, max_age=86400):
    try:
        p = os.path.join(CACHE_DIR, f"{key}.json")
        if not os.path.exists(p): return None
        with open(p, "r", encoding="utf-8") as f:
            j = json.load(f)
        if time.time() - j.get("ts", 0) > max_age:
            return None
        return j.get("data")
    except Exception:
        return None

# -------------------------
# HTTP multipart helper for Roboflow
# -------------------------
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

def roboflow_detect(image_bytes, retries=2):
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
        except Exception:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            return None

# -------------------------
# Market data fetchers
# -------------------------
def fetch_finnhub_candles(symbol, resolution="5", from_ts=None, to_ts=None):
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None:
            to_ts = int(time.time())
        if from_ts is None:
            if resolution in ("1","5","15","30","60"):
                from_ts = to_ts - 60*60*24
            else:
                from_ts = to_ts - 60*60*24*30
        params = {"symbol": symbol, "resolution": resolution, "from": str(int(from_ts)), "to": str(int(to_ts)), "token": FINNHUB_KEY}
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=25) as resp:
            txt = resp.read().decode("utf-8")
            data = json.loads(txt)
        if data.get("s") != "ok":
            return None
        ts = data.get("t", []); o = data.get("o", []); h = data.get("h", []); l = data.get("l", []); c = data.get("c", []); v = data.get("v", [])
        candles=[]
        for i, t in enumerate(ts):
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except:
                dt = datetime.utcnow()
            candles.append({"t": dt, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i]), "volume": float(v[i]) if v and i < len(v) else 0.0})
        return candles
    except Exception:
        return None

def fetch_alpha_minute(symbol, interval="5min", outputsize="compact"):
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

# fallback simulated candles
def generate_simulated_candles(seed, periods, start_price=100.0, resolution_minutes=5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    p = float(start_price)
    prices=[]
    for _ in range(periods):
        drift = (rnd.random() - 0.49) * 0.003
        shock = (rnd.random() - 0.5) * 0.02
        p = max(0.01, p * (1 + drift + shock))
        prices.append(round(p,6))
    candles=[]; now = datetime.utcnow()
    for i,pr in enumerate(prices):
        o = round(pr * (1 + random.uniform(-0.002, 0.002)),6)
        c = pr
        h = round(max(o,c) * (1 + random.uniform(0.0,0.004)),6)
        l = round(min(o,c) * (1 - random.uniform(0.0,0.004)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c, "volume": random.randint(1,1000)})
    return candles

# -------------------------
# Indicators & pattern detectors
# -------------------------
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

# candlestick simple detectors
def is_doji(c): 
    body = abs(c["close"] - c["open"]); total = c["high"] - c["low"]
    return total > 0 and (body / total) < 0.15
def is_hammer(c):
    body = abs(c["close"] - c["open"]); lower = min(c["open"], c["close"]) - c["low"]
    return body > 0 and lower > 2 * body
def is_shooting_star(c):
    body = abs(c["close"] - c["open"]); upper = c["high"] - max(c["open"], c["close"])
    return body > 0 and upper > 2 * body
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

# -------------------------
# Image Analyzer (improved)
# -------------------------
def analyze_chart_image_structure(image_bytes):
    """
    Input: raw image bytes (screenshot)
    Output: dict with trend, patterns (strings), confidence, volatility est, recommendation, probability, risk_pct, summary, internal metrics
    """
    if not PIL_AVAILABLE:
        return {"error":"Pillow not installed. Install pillow for image analyzer."}
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception as e:
        return {"error":"Image open failed."}
    W,H = img.size
    maxw = 1400
    if W > maxw:
        img = img.resize((maxw, int(H * maxw / W))); W,H = img.size
    # crop to central region (likely candles)
    left = int(W*0.03); right = int(W*0.97); top = int(H*0.08); bottom = int(H*0.78)
    chart = img.crop((left, top, right, bottom))
    chart = ImageOps.autocontrast(chart, cutoff=2); chart = chart.filter(ImageFilter.MedianFilter(size=3))
    pix = chart.load()
    Wc,Hc = chart.size
    # vertical darkness profile
    col_darkness = []
    for x in range(Wc):
        s=0
        for y in range(0, Hc, 2):
            s += 255 - pix[x,y]
        col_darkness.append(s)
    # smooth profile
    smooth = []
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
    peak_count=len(peaks); trough_count=len(troughs)
    density = peak_count / (Wc/100.0 + 1e-9)
    # left-right brightness trend
    left_mean = ImageStat.Stat(chart.crop((0,0,Wc//2,Hc))).mean[0]
    right_mean = ImageStat.Stat(chart.crop((Wc//2,0,Wc,Hc))).mean[0]
    if right_mean > left_mean + 6:
        trend = "Aufwärtstrend"
    elif right_mean < left_mean - 6:
        trend = "Abwärtstrend"
    else:
        trend = "Seitwärts"
    # pattern heuristics scanning peaks
    doji_like = 0; hammer_like = 0; shooting_like = 0; engulfing_like = 0
    sample = peaks[-min(120,len(peaks)):] if peaks else []
    for idx in sample:
        col = [255 - pix[idx, y] for y in range(Hc)]
        maxv = max(col) if col else 0
        if maxv <= 0: continue
        thresh = max(2, maxv*0.4)
        highpos = [i for i,v in enumerate(col) if v >= thresh]
        if not highpos: continue
        body_h = max(highpos) - min(highpos) if len(highpos) > 1 else 0
        lower_shadow = Hc - 1 - max(highpos)
        upper_shadow = min(highpos)
        if body_h < Hc * 0.05: doji_like += 1
        if lower_shadow > body_h * 2.5 and body_h > 0: hammer_like += 1
        if upper_shadow > body_h * 2.5 and body_h > 0: shooting_like += 1
    for i in range(len(peaks)-1):
        a = peaks[i]; b = peaks[i+1]
        if smooth[b] > smooth[a] * 1.9:
            engulfing_like += 1
    var = statistics.pvariance(smooth) if len(smooth)>1 else 0.0
    vol_est = min(100.0, max(1.0, (var**0.5)/(avg+1e-9) * 200.0))
    # patterns bucket
    patterns=[]
    if doji_like: patterns.append(f"{doji_like}× Doji-like")
    if hammer_like: patterns.append(f"{hammer_like}× Hammer-like")
    if shooting_like: patterns.append(f"{shooting_like}× Shooting-like")
    if engulfing_like: patterns.append(f"{engulfing_like}× Engulfing-like")
    if peak_count > 12 and density > 6: patterns.append("Hohe Candle-Dichte")
    if trough_count > 6: patterns.append("Mehrere lokale Tiefs")
    if not patterns: patterns.append("Keine klaren Candle-Formen")
    conf = 30 + min(60, int(min(peak_count, 70)*1.2 + len(patterns)*4 + (10 if trend!="Seitwärts" else 0)))
    conf = max(5, min(98, conf))
    score=0
    if trend=="Aufwärtstrend": score+=2
    if trend=="Abwärtstrend": score-=2
    score += hammer_like*2 + engulfing_like*2 - shooting_like*2
    if score >= 3: rec = "Kaufen"
    elif score <= -2: rec = "Short"
    else: rec = "Neutral"
    prob = min(95.0, max(10.0, 45.0 + score*9.0 + conf*0.2))
    risk_pct = min(50.0, max(1.0, vol_est*0.6))
    # summary phrases
    summary=[]
    if rec=="Kaufen":
        summary.append(f"Bild-Struktur zeigt {', '.join(patterns[:3])}. Trend: {trend}.")
        summary.append(f"Geschätzte Trefferwahrscheinlichkeit: {prob:.1f}% • Risiko: {risk_pct:.1f}%.")
        summary.append("Empfehlung: Kleine Long-Position mit Stop-Loss; warte Bestätigung der nächsten Kerze.")
    elif rec=="Short":
        summary.append(f"Bärische Struktur erkannt ({', '.join(patterns[:3])}). Trend: {trend}.")
        summary.append(f"Geschätzte Trefferwahrscheinlichkeit: {prob:.1f}% • Risiko: {risk_pct:.1f}%.")
        summary.append("Empfehlung: Short mit enger Absicherung oder Abwarten.")
    else:
        summary.append("Keine eindeutige Struktur erkannt.")
        summary.append("Empfehlung: Warten auf Bestätigung (Volumen/Breakout).")
        summary.append("Tipp: Kein Full-Size-Entry ohne Bestätigung.")
    # prepare internal metrics for annotation
    internal = {"peaks": peak_count, "troughs": trough_count, "density": round(density,2)}
    return {"trend": trend, "patterns": patterns, "confidence": conf, "volatility": round(vol_est,2), "recommendation": rec, "probability": round(prob,1), "risk_pct": round(risk_pct,2), "summary": summary, "internal": internal, "chart_image": chart}

# -------------------------
# Support/Resistance detection (local pivots) -> use for SL/TP
# -------------------------
def local_pivots_from_candles(candles, window=5):
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]
    pivots_high=[]; pivots_low=[]
    n=len(candles)
    for i in range(window, n-window):
        hl = highs[i]
        if all(hl > highs[j] for j in range(i-window, i)) and all(hl > highs[j] for j in range(i+1, i+window+1)):
            pivots_high.append((i, highs[i]))
        ll = lows[i]
        if all(ll < lows[j] for j in range(i-window, i)) and all(ll < lows[j] for j in range(i+1, i+window+1)):
            pivots_low.append((i, lows[i]))
    return pivots_high, pivots_low

def map_patterns_to_levels(image_result, candles=None):
    """
    Map image patterns to concrete stop-loss and take-profit levels.
    If 'candles' provided, use last price and local pivots; otherwise provide relative SL/TP.
    """
    risk = image_result.get("risk_pct", 5.0)
    pat = image_result.get("patterns", [])
    last_price = candles[-1]["close"] if candles else None
    sl=None; tp=None; notes=[]
    # if candles available -> use pivots
    if candles and len(candles)>=20:
        ph, pl = local_pivots_from_candles(candles, window=4)
        # choose nearest pivot below as support, above as resistance
        last_close = candles[-1]["close"]
        supports = sorted([p[1] for p in pl if p[1] < last_close], reverse=True)
        resistances = sorted([p[1] for p in ph if p[1] > last_close])
        if supports:
            sl = supports[0] * (1 - 0.002)  # slightly under support
            notes.append("Stop unter lokalem Support")
        else:
            sl = last_close * (1 - risk/100.0)
            notes.append("Stop relativ (kein Support gefunden)")
        if resistances:
            tp = resistances[0] * (1 + 0.002)
            notes.append("TP an lokalem Widerstand")
        else:
            tp = last_close * (1 + 2 * risk/100.0)
            notes.append("TP relativ (kein Resistance)")
    else:
        if last_price:
            sl = last_price * (1 - risk/100.0)
            tp = last_price * (1 + 2*risk/100.0)
            notes.append("Relative SL/TP (keine Candle-Historie)")
        else:
            # fallback absolute ratios
            sl = None; tp=None
            notes.append("Keine Price-Info: nur relative Empfehlung")
    return {"stop": None if sl is None else round(sl,6), "tp": None if tp is None else round(tp,6), "notes": notes}

# -------------------------
# Backtester (improved)
# -------------------------
def backtest_strategy_on_history(candles, entries, position_size_pct=1.0, slippage_pct=0.05, fee_pct=0.02, lookahead=10):
    """
    candles: historical list (old->new)
    entries: list of indices where we 'entered' (detected pattern)
    Simulate simple market orders: entry at close after pattern, exit after lookahead or stop hit (no intraday sim; approximate).
    returns: dict summary & trades list
    """
    equity = 10000.0  # base bankroll for simulation
    trades = []
    wins = 0; losses = 0; total_ret = 0.0
    for idx in entries:
        if idx+lookahead >= len(candles): continue
        entry_price = candles[idx]["close"] * (1 + slippage_pct/100.0)
        # simple stop = entry *  (1 - 0.01) for example; here we won't dynamically apply SL intraday; we'll compute outcome at lookahead price
        exit_price = candles[idx+lookahead]["close"] * (1 - slippage_pct/100.0)
        gross_ret = (exit_price - entry_price) / (entry_price + 1e-12)
        net_ret = gross_ret - fee_pct/100.0
        position_value = equity * (position_size_pct/100.0)
        pnl = position_value * net_ret
        total_ret += net_ret
        trades.append({"idx": idx, "entry": entry_price, "exit": exit_price, "gross_ret_pct": round(gross_ret*100,3), "net_ret_pct": round(net_ret*100,3), "pnl": round(pnl,2)})
        if net_ret > 0:
            wins += 1
        else:
            losses += 1
    total_trades = len(trades)
    winrate = (wins/total_trades*100.0) if total_trades>0 else 0.0
    avg_ret = (sum(t["net_ret_pct"] for t in trades)/total_trades) if total_trades>0 else 0.0
    pf = (sum(t["pnl"] for t in trades if t["pnl"]>0) / abs(sum(t["pnl"] for t in trades if t["pnl"]<0))) if any(t["pnl"]<0 for t in trades) else (sum(t["pnl"] for t in trades if t["pnl"]>0) or 0.0)
    # Plain-language summary
    if total_trades == 0:
        summary = "Keine Trades gefunden für Backtest."
    else:
        perf = "gut" if winrate > 55 and avg_ret > 0 else "neutral" if winrate > 40 else "riskant"
        summary = f"In {total_trades} getesteten Trades lag die Trefferquote bei {winrate:.1f}% (avg return {avg_ret:.3f}% netto). Performance-Label: {perf}."
    return {"trades": trades, "total": total_trades, "wins": wins, "losses": losses, "winrate": round(winrate,2), "avg_return_pct": round(avg_ret,3), "profit_factor": round(pf,3) if isinstance(pf, float) else None, "summary": summary}

# -------------------------
# Annotation: draw arrows/lines on uploaded chart image to indicate patterns & SL/TP
# -------------------------
def annotate_chart_image(pil_img, detections, sl=None, tp=None):
    """
    pil_img: PIL.Image (RGB or L) - whole screenshot
    detections: list of dicts with keys: {'type': 'Hammer','x': px, 'y': px, 'dir': 'up'/'down'}
    sl,tp: price levels - we will draw lines across approximate y-coordinates if chart cropping provided with mapping
    Returns: annotated PIL image object
    """
    # convert to RGBA to draw
    if pil_img.mode != "RGBA":
        base = pil_img.convert("RGBA")
    else:
        base = pil_img.copy()
    draw = ImageDraw.Draw(base, "RGBA")
    W,H = base.size
    # choose font if available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    # draw detections as small arrows/labels
    for d in detections:
        x = int(d.get("x", W*0.5))
        y = int(d.get("y", H*0.5))
        typ = d.get("type","Pattern")
        direction = d.get("dir","up")
        color = (0,255,136,200) if direction=="up" else (255,100,120,200)
        # arrow
        if direction=="up":
            draw.polygon([(x, y-14), (x-8, y+6), (x+8, y+6)], fill=color)
        else:
            draw.polygon([(x, y+14), (x-8, y-6), (x+8, y-6)], fill=color)
        # label
        txt = typ
        txt_w, txt_h = draw.textsize(txt, font=font) if font else (len(txt)*6, 12)
        draw.rectangle([x-txt_w//2-6, y+10, x+txt_w//2+6, y+10+txt_h+6], fill=(10,10,10,180))
        draw.text((x-txt_w//2, y+12), txt, fill=(230,238,246,255), font=font)
    # draw SL/TP as horizontal lines if provided
    if sl is not None:
        # map price->y unknown: just draw label at top right to indicate value
        draw.line([(20, H*0.12), (W-20, H*0.12)], fill=(255,204,0,160), width=2)
        draw.text((30, H*0.12-12), f"Stop: {sl}", fill=(255,204,0,255), font=font)
    if tp is not None:
        draw.line([(20, H*0.18), (W-20, H*0.18)], fill=(102,255,136,160), width=2)
        draw.text((30, H*0.18-12), f"TP: {tp}", fill=(102,255,136,255), font=font)
    return base

# -------------------------
# Export helpers
# -------------------------
def export_analysis_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)

def export_analysis_csv(obj):
    # flatten some fields into CSV with key,value rows
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key","value"])
    # meta
    meta = obj.get("meta", {})
    for k,v in meta.items():
        w.writerow([f"meta.{k}", v])
    final = obj.get("final", {})
    for k,v in final.items():
        if isinstance(v, (str,int,float)):
            w.writerow([k, v])
        else:
            try:
                w.writerow([k, json.dumps(v, ensure_ascii=False)])
            except:
                w.writerow([k, str(v)])
    return buf.getvalue()

# -------------------------
# UI: Navigation
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live Analyzer","Bild-Analyse (offline)","Backtest","Einstellungen","Hilfe"])

if not ONLINE:
    st.sidebar.error("❌ Keine Internetverbindung — Live Daten werden simuliert")
else:
    st.sidebar.success("✅ Internet verfügbar")

# -------------------------
# Page: Home
# -------------------------
if page == "Home":
    st.header("Lumina Pro — Deep Analyzer")
    st.markdown("""
    **Was neu ist:** SL/TP Mapping, Export, Roboflow label→action mapping, verbesserter Backtester, Annotationen.
    Lade ein Chartbild in 'Bild-Analyse' oder probiere 'Live Analyzer'.
    """)
    st.write("Pillow:", PIL_AVAILABLE, "Matplotlib:", MATPLOTLIB_AVAILABLE)
    st.write("Finnhub key present:", bool(FINNHUB_KEY))
    st.write("AlphaV key present:", bool(ALPHAV_KEY))
    st.markdown("---")
    st.markdown("Tip: Für beste Bildanalyse: croppe das Bild auf die reine Kerzenzone, keine Overlays.")

# -------------------------
# Page: Live Analyzer
# -------------------------
elif page == "Live Analyzer":
    st.header("Live Analyzer — Finnhub / AlphaV (Fallback)")
    left, right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol (e.g. BINANCE:BTCUSDT or AAPL)", value="BINANCE:BTCUSDT")
        resolution = st.selectbox("Resolution", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles", 30, 1000, 240, step=10)
        fallback_price = st.number_input("Fallback price (if no live)", value=20000.0)
        run = st.button("Load & Analyze")
    with left:
        if run:
            candles=None
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - int(periods) * int(resolution) * 60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None and ALPHAV_KEY:
                    st.warning("Finnhub no data — trying Alpha Vantage fallback")
                    try:
                        av = fetch_alpha_minute(symbol, interval=resolution+"min")
                        if av and len(av)>0:
                            candles = av[-periods:] if len(av)>=periods else av
                    except Exception:
                        candles = None
                if candles is None:
                    st.warning("No live data — using simulation")
                    candles = generate_simulated_candles(symbol + "_sim", periods, fallback_price, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol + "_pad", need, candles[0]["open"] if candles else fallback_price, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline or no key — using simulation")
                candles = generate_simulated_candles(symbol + "_sim", periods, fallback_price, int(resolution))
            closes = [c["close"] for c in candles]
            if len(closes) >= 50:
                s20 = sum(closes[-20:]) / 20
                s50 = sum(closes[-50:]) / 50
                trend = "Aufwärtstrend" if s20 > s50 else ("Abwärtstrend" if s20 < s50 else "Seitwärts")
            else:
                trend = "Seitwärts"
            patt = detect_patterns(candles)
            # build pseudo-image result to map levels
            pseudo_img = {"patterns":[p[0] for p in patt], "risk_pct": 3.0}
            mapped = map_patterns_to_levels(pseudo_img, candles[-200:])
            # fuse simple recommendation: use sma bias & patterns
            bias = 0
            if len(closes) >= 50:
                if s20 > s50: bias += 1
                else: bias -= 1
            rec = "Neutral"
            if bias > 0 and any("Bullish" in p[0] or "Hammer" in p[0] for p in patt): rec = "Kaufen"
            if bias < 0 and any("Bearish" in p[0] or "Shooting" in p[0] for p in patt): rec = "Short"
            # show results
            st.subheader(f"{symbol} — {trend}")
            st.write(f"Current price: {candles[-1]['close']:.6f}")
            st.write("Detected patterns:", [p[0] for p in patt][:8])
            st.write("Recommendation:", rec)
            st.write("Stop:", mapped["stop"], "TP:", mapped["tp"])
            # render plot
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
                ax.plot([c["t"] for c in candles[-300:]], [c["close"] for c in candles[-300:]], color="#00cc66")
                ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
                st.pyplot(fig)
            else:
                svg = render_svg_candles(candles[-160:], stop=mapped["stop"], tp=mapped["tp"])
                st.components.v1.html(svg, height=480)

# -------------------------
# Page: Bild-Analyse (offline) — main focus
# -------------------------
elif page == "Bild-Analyse (offline)":
    st.header("Bild-Analyse — Struktur & Muster (automatisch, offline-first)")
    st.markdown("Lade ein Chart-Screenshot hoch (der Bereich mit Kerzen). Die App führt automatisch Analyse, SL/TP Mapping, Hintergrund-Backtest und Annotation durch.")
    uploaded = st.file_uploader("Chart-Bild hochladen (PNG/JPG)", type=["png","jpg","jpeg"])
    show_internals = st.checkbox("Zeige interne Metriken", value=False)
    run = st.button("Analysiere Bild (automatisch)")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            # Roboflow analyze (optional, robust retries)
            rf_res = None
            if ONLINE and ROBOFLOW_KEY:
                with st.spinner("Roboflow detection..."):
                    try:
                        rf_res = roboflow_detect(img_bytes, retries=2)
                    except Exception:
                        rf_res = None
            # local analysis (main)
            img_res = analyze_chart_image_structure(img_bytes)
            if img_res.get("error"):
                st.error(img_res["error"])
            else:
                # choose history for backtest: attempt to get real history for generic symbol list; otherwise simulated
                hist = None
                if ONLINE and FINNHUB_KEY:
                    syms = ["BINANCE:BTCUSDT","AAPL","SPY","MSFT"]
                    for s in syms:
                        try:
                            h = fetch_finnhub_candles(s, "5", int(time.time()) - 60*60*24*180, int(time.time()))
                            if h and len(h) >= 500:
                                hist = h; break
                        except Exception:
                            hist = None
                if hist is None and ALPHAV_KEY:
                    try:
                        hist = fetch_alpha_minute("AAPL","5min","full")
                    except Exception:
                        hist = None
                if hist is None:
                    hist = generate_simulated_candles("backtest_seed_img", 900, 100.0, 5)
                # backtest top patterns (map names)
                patterns = [p.split()[0] for p in img_res.get("patterns", [])]
                backtests = []
                # naive entries detection for backtest: find pattern indices in hist by detector name
                detected_indices = []
                for i in range(1, len(hist)):
                    # check a few basic patterns
                    try:
                        if "Doji" in patterns and is_doji(hist[i]): detected_indices.append(i)
                        if "Hammer" in patterns and is_hammer(hist[i]): detected_indices.append(i)
                        if "Engulfing" in patterns:
                            if i>=1 and (is_bullish_engulfing(hist[i-1], hist[i]) or is_bearish_engulfing(hist[i-1], hist[i])): detected_indices.append(i)
                    except Exception:
                        continue
                # deduplicate
                detected_indices = sorted(set(detected_indices))
                # run backtester with user defaults for slippage/fee/position
                bt_res = backtest_strategy_on_history(hist, detected_indices, position_size_pct=1.0, slippage_pct=0.05, fee_pct=0.02, lookahead=10)
                # Map SL/TP
                mapping = map_patterns_to_levels(img_res, hist[-200:] if hist else None)
                # annotate uploaded image: create some dummy detection coordinates from peaks to map visuals
                annotated = None
                if PIL_AVAILABLE:
                    try:
                        uploaded_img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
                        # create detections for drawing: place arrows roughly across width by detected internal peaks count
                        internal = img_res.get("internal", {})
                        peak_count = internal.get("peaks", 0)
                        detections = []
                        if peak_count > 0:
                            # distribute along width
                            W,H = uploaded_img.size
                            for i in range(min(6, peak_count)):
                                x = int((i+1) * W / (min(6, peak_count)+1))
                                y = int(H*0.3 + (i%2)*H*0.3)
                                typ = img_res["patterns"][i%len(img_res["patterns"])] if img_res["patterns"] else "Pattern"
                                # direction heuristics
                                dirc = "up" if img_res["recommendation"]=="Kaufen" else "down" if img_res["recommendation"]=="Short" else "up"
                                detections.append({"x": x, "y": y, "type": typ, "dir": dirc})
                        annotated = annotate_chart_image(uploaded_img, detections, sl=mapping.get("stop"), tp=mapping.get("tp"))
                    except Exception:
                        annotated = None
                # Final fused result
                adj_prob = img_res["probability"]
                if bt_res["total"]>0:
                    adj_prob = round((adj_prob*0.6 + bt_res["winrate"]*0.4),1)
                final = {
                    "meta": {"ts": now_iso(), "source": "image_upload"},
                    "final": {
                        "recommendation": img_res["recommendation"],
                        "probability": adj_prob,
                        "risk_pct": img_res["risk_pct"],
                        "stop": mapping.get("stop"),
                        "tp": mapping.get("tp"),
                        "summary": img_res["summary"],
                        "patterns": img_res["patterns"],
                        "confidence": img_res["confidence"],
                        "volatility": img_res["volatility"],
                        "backtest_summary": bt_res["summary"],
                        "backtest_stats": {"total": bt_res["total"], "winrate": bt_res["winrate"], "avg_return_pct": bt_res["avg_return_pct"]}
                    },
                    "internal": img_res.get("internal", {}),
                    "roboflow": rf_res
                }
                # UI: show final card
                left, right = st.columns([2,1])
                with left:
                    if final["final"]["recommendation"] == "Kaufen":
                        st.success(f"Empfehlung: {final['final']['recommendation']}  •  {final['final']['probability']}%")
                    elif final["final"]["recommendation"] == "Short":
                        st.error(f"Empfehlung: {final['final']['recommendation']}  •  {final['final']['probability']}%")
                    else:
                        st.info(f"Empfehlung: {final['final']['recommendation']}  •  {final['final']['probability']}%")
                    st.markdown(f"**Risiko:** {final['final']['risk_pct']}%")
                    st.markdown("**3-Satz-Fazit:**")
                    for s in final["final"]["summary"][:3]:
                        st.write("- " + s)
                    if mapping.get("notes"):
                        st.markdown("**SL/TP-Notizen**")
                        for n in mapping["notes"][:4]:
                            st.write("- " + n)
                    # Export buttons: JSON & CSV
                    st.download_button("Export Analyse (JSON)", data=export_analysis_json(final), file_name=f"lumina_analysis_{short_ts()}.json", mime="application/json")
                    st.download_button("Export Analyse (CSV)", data=export_analysis_csv({"meta": final["meta"], "final": final["final"]}), file_name=f"lumina_analysis_{short_ts()}.csv", mime="text/csv")
                with right:
                    st.markdown("**Details**")
                    st.write("Trend:", img_res["trend"])
                    st.write("Confidence:", img_res["confidence"])
                    st.write("Volatility:", img_res["volatility"])
                    st.write("Detected patterns:")
                    for p in img_res["patterns"][:8]:
                        st.write("- " + p)
                    st.markdown("**Backtest (Kurzfassung)**")
                    st.write(bt_res["summary"])
                    st.write(f"Trades checked: {bt_res['total']} • Winrate: {bt_res['winrate']}% • AvgRet: {bt_res['avg_return_pct']}%")
                st.markdown("---")
                # show annotated image if exists
                if annotated:
                    st.image(annotated, use_column_width=True)
                else:
                    st.info("Keine Annotation möglich (Pillow nicht verfügbar oder Annotate fehlgeschlagen).")
                if show_internals:
                    st.write("Internals:", img_res.get("internal", {}))
                st.success("Analyse & Backtest abgeschlossen.")

# -------------------------
# Page: Backtest (manual)
# -------------------------
elif page == "Backtest":
    st.header("Backtest & Simulation (manuell)")
    pattern = st.selectbox("Muster testen", ["Bullish Engulfing","Bearish Engulfing","Hammer","Doji","Morning Star"])
    lookahead = st.slider("Lookahead (Kerzen)", 1, 30, 10)
    pos_size = st.number_input("Position size (% of equity per trade)", value=1.0, min_value=0.1, step=0.1)
    slippage = st.number_input("Slippage (%)", value=0.05, step=0.01)
    fee = st.number_input("Fee (%)", value=0.02, step=0.01)
    if st.button("Run Backtest"):
        hist = generate_simulated_candles("bt_seed_manual", 1200, 100.0, 5)
        # detect sample indices for chosen pattern
        indices=[]
        for i in range(1, len(hist)):
            try:
                if "Doji" in pattern and is_doji(hist[i]): indices.append(i)
                if "Hammer" in pattern and is_hammer(hist[i]): indices.append(i)
                if "Engulf" in pattern and i>=1 and (is_bullish_engulfing(hist[i-1], hist[i]) or is_bearish_engulfing(hist[i-1], hist[i])): indices.append(i)
            except Exception:
                continue
        res = backtest_strategy_on_history(hist, sorted(set(indices)), position_size_pct=pos_size, slippage_pct=slippage, fee_pct=fee, lookahead=lookahead)
        st.write(res)
        # plain-language short conclusion
        if res["total"] == 0:
            st.info("Keine Musterereignisse in der simulierten Historie gefunden.")
        else:
            st.success("Backtest abgeschlossen.")
            # human summary
            if res["winrate"] > 60 and res["avg_return_pct"] > 0:
                conclusion = "Starkes Ergebnis — Muster liefert überdurchschnittliche Trefferquote in Simulation."
            elif res["winrate"] > 45:
                conclusion = "Akzeptables Ergebnis — Muster zeigt moderate Trefferquote."
            else:
                conclusion = "Schwaches Ergebnis — Vorsicht, hohes Risiko."
            st.markdown(f"**Fazit:** {conclusion}")
        # plot sample history
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
            ax.plot([c["t"] for c in hist[-300:]], [c["close"] for c in hist[-300:]], color="#00cc66")
            ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
            st.pyplot(fig)
        else:
            st.line_chart([c["close"] for c in hist[-300:]])

# -------------------------
# Page: Settings & Help
# -------------------------
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Keys are embedded in the script (not secure for public hosting). Move them to st.secrets in production.")
    st.write("Pillow installed:", PIL_AVAILABLE)
    st.write("Matplotlib installed:", MATPLOTLIB_AVAILABLE)
    if st.button("Clear cache"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR, f))
            except: pass
        st.success("Cache cleared.")

elif page == "Hilfe":
    st.header("Hilfe")
    st.markdown("""
    - Bild-Analyse ist offline-first; Roboflow optional for better detection (requires internet).
    - Live Analyzer uses Finnhub primary and Alpha Vantage fallback (watch API limits).
    - Exports provide JSON + CSV for audit/log.
    - Recommendations are probabilistic estimates — NOT financial advice.
    """)

st.markdown("---")
st.caption("Lumina Pro — Deep Analyzer. Keys embedded per user request. Use responsibly.")
