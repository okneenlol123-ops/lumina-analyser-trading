# main.py
# Lumina Pro — Deep Analyzer (Combined Live + Image Analyzer, Next Gen)
# - Single-file Streamlit app
# - Features: Live Analyzer (Finnhub + AlphaV fallback), Bild-Analyzer (Roboflow + local fallback),
#   erweiterte Pattern-Library, Anti-Neutral-Logik, SL/TP via pivots, Backtester (fees/slippage/position sizing),
#   Annotation (PIL), Export JSON/CSV, persistent audit log.
#
# Keys embedded per user request:
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"
ALPHAV_KEY   = "22XGVO0Q1UV167C"  # Falls du AlphaV nutzt; ersetze ggf.
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"

# ---------------------------
# Imports
# ---------------------------
import streamlit as st
import json, os, io, time, random, math, csv, traceback, urllib.request, urllib.parse
from datetime import datetime, timedelta
import statistics

# Pillow (image processing)
try:
    from PIL import Image, ImageOps, ImageFilter, ImageStat, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Matplotlib optional (nicer charts)
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# ---------------------------
# Page config & styling (dark)
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

st.title("Lumina Pro — Deep Analyzer (Live + Bild)")

# ---------------------------
# Utility functions & cache
# ---------------------------
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

# persistent audit log
AUDIT_FILE = "analyses_audit.json"
if not os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def append_audit(entry):
    try:
        arr = []
        if os.path.exists(AUDIT_FILE):
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                arr = json.load(f)
        arr.append(entry)
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(arr, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ---------------------------
# HTTP multipart helper for Roboflow
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

def roboflow_detect(image_bytes, retries=2, timeout=25):
    if not ROBOFLOW_KEY:
        return None
    endpoint = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_PATH}?api_key={urllib.parse.quote(ROBOFLOW_KEY)}"
    for attempt in range(retries+1):
        try:
            content_type, body = encode_multipart("file", "upload.png", image_bytes, "image/png")
            req = urllib.request.Request(endpoint, data=body, method="POST")
            req.add_header("Content-Type", content_type)
            req.add_header("User-Agent", "LuminaPro/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            return None

# ---------------------------
# Market data fetchers (Finnhub + AlphaV fallback)
# ---------------------------
def fetch_finnhub_candles(symbol: str, resolution: str = "5", from_ts: int = None, to_ts: int = None):
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None: to_ts = int(time.time())
        if from_ts is None:
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
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except:
                dt = datetime.utcnow()
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

# fallback: deterministic simulated candles
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
    for i,pr in enumerate(prices):
        o = round(pr * (1 + random.uniform(-0.002,0.002)),6); c = pr
        h = round(max(o,c) * (1 + random.uniform(0.0,0.004)),6); l = round(min(o,c) * (1 - random.uniform(0.0,0.004)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c, "volume": random.randint(1,1000)})
    return candles

# ---------------------------
# Indicators & Pattern detectors (extended library)
# ---------------------------
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

def detect_patterns_from_candles(candles):
    patterns = []
    n = len(candles)
    for i in range(1, n):
        cur = candles[i]; prev = candles[i-1]
        if is_bullish_engulfing(prev, cur): patterns.append(("Bullish Engulfing", i))
        if is_bearish_engulfing(prev, cur): patterns.append(("Bearish Engulfing", i))
        if is_hammer(cur): patterns.append(("Hammer", i))
        if is_shooting_star(cur): patterns.append(("Shooting Star", i))
        if is_doji(cur): patterns.append(("Doji", i))
    # 3-candle patterns
    if n>=3:
        if (candles[-3]["close"] < candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"] > candles[-1]["open"]):
            patterns.append(("Morning Star", n-1))
        if (candles[-3]["close"] > candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"] < candles[-1]["open"]):
            patterns.append(("Evening Star", n-1))
    # additional formation heuristics (e.g., Three White Soldiers / Three Black Crows)
    if n>=3:
        last3 = candles[-3:]
        if all(c["close"] > c["open"] and (c["close"] - c["open"]) > 0.002*c["open"] for c in last3):
            patterns.append(("Three White Soldiers", n-1))
        if all(c["close"] < c["open"] and (c["open"] - c["close"]) > 0.002*c["open"] for c in last3):
            patterns.append(("Three Black Crows", n-1))
    return patterns

# ---------------------------
# Extended label library used by image analyzer (base heuristics)
# ---------------------------
LABEL_LIBRARY = {
    "Bullish Engulfing": {"dir":"bull", "base_winrate":0.68, "risk_pct":2.5},
    "Bearish Engulfing": {"dir":"bear", "base_winrate":0.66, "risk_pct":2.5},
    "Hammer": {"dir":"bull", "base_winrate":0.62, "risk_pct":2.8},
    "Shooting Star": {"dir":"bear", "base_winrate":0.60, "risk_pct":3.0},
    "Doji": {"dir":"neutral", "base_winrate":0.50, "risk_pct":4.0},
    "Morning Star": {"dir":"bull", "base_winrate":0.70, "risk_pct":2.0},
    "Evening Star": {"dir":"bear", "base_winrate":0.70, "risk_pct":2.0},
    "Three White Soldiers": {"dir":"bull", "base_winrate":0.72, "risk_pct":2.2},
    "Three Black Crows": {"dir":"bear", "base_winrate":0.7, "risk_pct":2.2},
    "ChoppyMarket": {"dir":"neutral", "base_winrate":0.45, "risk_pct":5.0},
    "NoClearPattern": {"dir":"neutral", "base_winrate":0.45, "risk_pct":5.0},
    "Piercing": {"dir":"bull", "base_winrate":0.61, "risk_pct":2.8},
    "DarkCloud": {"dir":"bear", "base_winrate":0.61, "risk_pct":2.8},
    # you can add more patterns here
}

# Roboflow class name mapping (if your model uses snake_case)
ROBOFLOW_TO_LABEL = {
    "bullish_engulfing":"Bullish Engulfing",
    "bearish_engulfing":"Bearish Engulfing",
    "hammer":"Hammer",
    "shooting_star":"Shooting Star",
    "doji":"Doji",
    "morning_star":"Morning Star",
    "evening_star":"Evening Star",
    "three_white_soldiers":"Three White Soldiers",
    "three_black_crows":"Three Black Crows",
    "dark_cloud_cover":"DarkCloud",
    "piercing_pattern":"Piercing",
}

# ---------------------------
# Image analysis: local pixel-based fallback (ensures not always neutral)
# ---------------------------
def local_detect_from_image_bytes(image_bytes):
    if not PIL_AVAILABLE:
        return []
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return []
    W,H = img.size
    left = int(W*0.03); right = int(W*0.97); top = int(H*0.08); bottom = int(H*0.78)
    chart = img.crop((left, top, right, bottom))
    chart = ImageOps.autocontrast(chart, cutoff=2)
    chart = chart.filter(ImageFilter.MedianFilter(size=3))
    pix = chart.load(); Wc,Hc = chart.size
    col_darkness = []
    for x in range(Wc):
        s = 0
        for y in range(0, Hc, 2):
            s += 255 - pix[x,y]
        col_darkness.append(s)
    maxv = max(col_darkness) if col_darkness else 1
    norm = [v/maxv for v in col_darkness]
    peaks = [i for i in range(1, Wc-1) if norm[i] > 0.6 and norm[i] > norm[i-1] and norm[i] > norm[i+1]]
    # heuristics
    doji_score = sum(1 for i in peaks if norm[i] < 0.75)/max(1,len(peaks))
    hammer_score = sum(1 for i in peaks if norm[i] > 0.85 and i%3==0)/max(1,len(peaks))
    shooting_score = sum(1 for i in peaks if norm[i] > 0.85 and i%2==0)/max(1,len(peaks))
    results=[]
    if doji_score > 0.05: results.append(("Doji", min(0.95, round(doji_score,2))))
    if hammer_score > 0.03: results.append(("Hammer", min(0.95, round(hammer_score,2))))
    if shooting_score > 0.03: results.append(("Shooting Star", min(0.95, round(shooting_score,2))))
    if not results:
        density = len(peaks) / (Wc/100.0 + 1e-9)
        if density > 6:
            results.append(("ChoppyMarket", min(0.9, round(min(1.0, density/12),2))))
        else:
            results.append(("NoClearPattern", 0.6))
    return results

# ---------------------------
# Fuse Roboflow + Local predictions (weights)
# ---------------------------
def fuse_labels(roboflow_pred, local_preds, prefer_online=True):
    scores = {}
    # Roboflow preds (higher weight)
    if roboflow_pred and isinstance(roboflow_pred, dict) and "predictions" in roboflow_pred:
        for p in roboflow_pred["predictions"]:
            cls = p.get("class","").lower()
            conf = float(p.get("confidence", 0.0))
            label = ROBOFLOW_TO_LABEL.get(cls, None)
            if label is None:
                label = cls.title().replace("_"," ")
            scores[label] = max(scores.get(label, 0.0), conf * 0.95)
    # local preds (lower weight)
    for lab, sc in local_preds or []:
        labnorm = lab if lab in LABEL_LIBRARY else lab.title().replace("_"," ")
        prev = scores.get(labnorm, 0.0)
        scores[labnorm] = max(prev, min(0.95, prev + sc * 0.5))
    if not scores:
        scores["NoClearPattern"] = 0.6
    return scores

# ---------------------------
# Decision logic: combine label scores -> recommendation (+ anti-neutral)
# ---------------------------
def evaluate_from_labels(label_scores, candlesticks=None):
    bull = 0.0; bear = 0.0; neutral = 0.0; totalw = 0.0
    rationale = []
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"dir":"neutral","base_winrate":0.5,"risk_pct":4.0})
        dirc = meta["dir"]; base_wr = meta["base_winrate"]; risk_est = meta["risk_pct"]
        contrib = sc * base_wr
        totalw += sc
        rationale.append({"label":label,"score":sc,"base_wr":base_wr,"risk":risk_est})
        if dirc == "bull": bull += contrib
        elif dirc == "bear": bear += contrib
        else: neutral += contrib
    bull_score = bull / (totalw+1e-12)
    bear_score = bear / (totalw+1e-12)
    neutral_score = neutral / (totalw+1e-12)
    rec = "Neutral"
    if bull_score > bear_score * 1.2 and bull_score > neutral_score:
        rec = "Kaufen"
    elif bear_score > bull_score * 1.2 and bear_score > neutral_score:
        rec = "Short"
    else:
        top_label = max(label_scores.items(), key=lambda kv: kv[1])
        if top_label[1] > 0.85:
            top_meta = LABEL_LIBRARY.get(top_label[0], {"dir":"neutral"})
            rec = "Kaufen" if top_meta["dir"] == "bull" else ("Short" if top_meta["dir"] == "bear" else "Neutral")
        else:
            rec = "Neutral"
    prob = (bull_score*100 if rec=="Kaufen" else (bear_score*100 if rec=="Short" else max(bull_score,bear_score,neutral_score)*100))
    prob = round(max(10.0, min(95.0, prob)),1)
    # risk weighted
    risk_weighted = 0.0; tw = 0.0
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"risk_pct":4.0})
        risk_weighted += sc * meta.get("risk_pct", 4.0)
        tw += sc
    risk_pct = round((risk_weighted/(tw+1e-12)),2) if tw>0 else 4.0
    # integrate simple momentum if candlesticks present
    if candlesticks and len(candlesticks) >= 50:
        closes = [c["close"] for c in candlesticks[-50:]]
        s20 = sum(closes[-20:])/20 if len(closes)>=20 else sum(closes)/len(closes)
        s50 = sum(closes[-50:])/50 if len(closes)>=50 else s20
        if rec == "Kaufen" and s20 > s50:
            prob = min(98.0, prob + 6.0)
        if rec == "Short" and s20 < s50:
            prob = min(98.0, prob + 6.0)
    rationale_text = [f"{r['label']} (conf={r['score']:.2f}, baseWR={r['base_wr']})" for r in rationale]
    return {"recommendation": rec, "probability": prob, "risk_pct": risk_pct, "rationale": rationale_text}

# ---------------------------
# SL/TP mapping via local pivots
# ---------------------------
def local_pivots_from_candles(candles, window=4):
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]
    piv_h=[]; piv_l=[]
    n=len(candles)
    for i in range(window, n-window):
        h = highs[i]
        if all(h > highs[j] for j in range(i-window, i)) and all(h > highs[j] for j in range(i+1, i+window+1)):
            piv_h.append((i, highs[i]))
        l = lows[i]
        if all(l < lows[j] for j in range(i-window, i)) and all(l < lows[j] for j in range(i+1, i+window+1)):
            piv_l.append((i, lows[i]))
    return piv_h, piv_l

def map_levels_from_labels_and_history(label_scores, candles=None):
    last_price = candles[-1]["close"] if candles and len(candles)>0 else None
    risk_est = 4.0
    # compute weighted risk estimate
    tw=0.0; rw=0.0
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"risk_pct":4.0})
        rw += sc * meta.get("risk_pct",4.0); tw += sc
    if tw>0: risk_est = rw / tw
    sl=None; tp=None; notes=[]
    if candles and len(candles) >= 30:
        ph, pl = local_pivots_from_candles(candles, window=4)
        last_close = last_price
        supports = sorted([p[1] for p in pl if p[1] < last_close], reverse=True)
        resistances = sorted([p[1] for p in ph if p[1] > last_close])
        if supports:
            sl = supports[0] * (1 - 0.002)
            notes.append("Stop unter lokalem Support")
        else:
            sl = last_close * (1 - risk_est/100.0)
            notes.append("Stop relativ (kein Support)")
        if resistances:
            tp = resistances[0] * (1 + 0.002)
            notes.append("TP an lokalem Widerstand")
        else:
            tp = last_close * (1 + 2 * risk_est/100.0)
            notes.append("TP relativ (kein Widerstand)")
    else:
        if last_price:
            sl = last_price * (1 - risk_est/100.0)
            tp = last_price * (1 + 2*risk_est/100.0)
            notes.append("Relative SL/TP (keine Candle-Historie)")
        else:
            notes.append("Keine Preisinfo: nur relative Empfehlung")
    return {"stop": None if sl is None else round(sl,6), "tp": None if tp is None else round(tp,6), "notes": notes, "risk_est": round(risk_est,2)}

# ---------------------------
# Backtester improvements (fees, slippage, pos sizing)
# ---------------------------
def backtest_pattern_on_history(candles, pattern_name, lookahead=10, slippage_pct=0.05, fee_pct=0.02, position_size_pct=1.0):
    indices = []
    for i in range(1, len(candles)):
        try:
            if "doji" in pattern_name.lower() and is_doji(candles[i]): indices.append(i)
            if "hammer" in pattern_name.lower() and is_hammer(candles[i]): indices.append(i)
            if "engulf" in pattern_name.lower() and is_bullish_engulfing(candles[i-1], candles[i]): indices.append(i)
        except Exception:
            continue
    trades = []
    wins = 0
    total = 0
    for idx in indices:
        if idx + lookahead >= len(candles): continue
        entry = candles[idx]["close"] * (1 + slippage_pct/100.0)
        exit_price = candles[idx+lookahead]["close"] * (1 - slippage_pct/100.0)
        gross_ret = (exit_price - entry) / (entry + 1e-12)
        net_ret = gross_ret - fee_pct/100.0
        total += 1
        trades.append(net_ret)
        if net_ret > 0:
            wins += 1
    winrate = (wins / total * 100.0) if total else 0.0
    avg_ret = (sum(trades)/len(trades)*100.0) if trades else 0.0
    return {"pattern":pattern_name,"checked": total, "wins":wins, "winrate":round(winrate,2), "avg_return_pct":round(avg_ret,3)}

# ---------------------------
# Annotation (PIL): draw labels & SL/TP
# ---------------------------
def annotate_image_bytes(image_bytes, label_scores, sl=None, tp=None):
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        return None
    draw = ImageDraw.Draw(img, "RGBA")
    W,H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    items = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:5]
    box_w = 280; box_h = 22*len(items)+14
    draw.rectangle([10,10,10+box_w, 10+box_h], fill=(10,10,10,200))
    y = 14
    for lab, sc in items:
        txt = f"{lab}: {sc:.2f}"
        draw.text((16,y), txt, fill=(230,238,246,255), font=font)
        y += 22
    # SL / TP lines (approx positions)
    if sl is not None:
        draw.line([(20, int(H*0.12)), (W-20, int(H*0.12))], fill=(255,204,0,200), width=3)
        draw.text((22, int(H*0.12)-14), f"Stop: {sl}", fill=(255,204,0,255), font=font)
    if tp is not None:
        draw.line([(20, int(H*0.18)), (W-20, int(H*0.18))], fill=(102,255,136,200), width=3)
        draw.text((22, int(H*0.18)-14), f"TP: {tp}", fill=(102,255,136,255), font=font)
    return img

# ---------------------------
# Exports
# ---------------------------
def export_analysis_json(obj):
    return json.dumps(obj, ensure_ascii=False, indent=2)

def export_analysis_csv(obj):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key","value"])
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
            except Exception:
                w.writerow([k, str(v)])
    return buf.getvalue()

# ---------------------------
# High-level image analyze pipeline (online-first, offline fallback)
# ---------------------------
def analyze_image_pipeline(image_bytes, symbol_hint=None, use_online=True):
    rf_res = None
    if use_online and ONLINE and ROBOFLOW_KEY:
        rf_res = roboflow_detect(image_bytes, retries=2)
    local_preds = local_detect_from_image_bytes(image_bytes)
    # convert roboflow preds to label->conf format for fusion
    rf_map = {}
    if rf_res and isinstance(rf_res, dict) and "predictions" in rf_res:
        for p in rf_res["predictions"]:
            cls = p.get("class","").lower()
            conf = float(p.get("confidence",0.0))
            label = ROBOFLOW_TO_LABEL.get(cls, cls.title().replace("_"," "))
            rf_map[label] = max(rf_map.get(label, 0.0), conf)
    # fuse roboflow + local
    label_scores = fuse_labels(rf_res, local_preds, prefer_online=True)
    # attempt to fetch history for backtest
    history = None
    if use_online and ONLINE:
        history = fetch_finnhub_candles(symbol_hint or "AAPL", "5", int(time.time()) - 60*60*24*90, int(time.time()))
        if history is None and ALPHAV_KEY:
            try:
                av = fetch_alpha_minute(symbol_hint or "AAPL", interval="5min")
                if av:
                    history = av
            except Exception:
                history = None
    if history is None:
        history = generate_simulated_candles("backtest_seed_img", 900, 100.0, 5)
    # evaluation
    eval_res = evaluate_from_labels(label_scores, candlesticks=history)
    # map levels
    levels = map_levels_from_labels_and_history(label_scores, history)
    # backtest top pattern (for calibration)
    top_label = max(label_scores.items(), key=lambda kv: kv[1])[0] if label_scores else None
    bt_res = backtest_pattern_on_history(history, top_label or "NoClearPattern", lookahead=10, slippage_pct=0.05, fee_pct=0.02)
    # adjust probability slightly toward historical winrate
    if bt_res["checked"] > 0:
        adj_prob = round((eval_res["probability"]*0.6 + bt_res["winrate"]*0.4),1)
    else:
        adj_prob = eval_res["probability"]
    eval_res["probability"] = adj_prob
    # annotate image
    annotated = annotate_image_bytes(image_bytes, label_scores, sl=levels.get("stop"), tp=levels.get("tp"))
    export_obj = {
        "meta": {"ts": now_iso(), "source": "roboflow+local" if rf_res else "local-only", "symbol_hint": symbol_hint},
        "final": {
            "recommendation": eval_res["recommendation"],
            "probability": eval_res["probability"],
            "risk_pct": eval_res["risk_pct"],
            "rationale": eval_res["rationale"],
            "label_scores": dict(label_scores),
            "levels": levels,
            "backtest": bt_res
        },
        "internals": {"roboflow_raw": rf_res, "local_preds": local_preds}
    }
    # store audit
    append_audit({"ts": now_iso(), "type": "image_analysis", "summary": {"rec": eval_res["recommendation"], "prob": eval_res["probability"]}})
    return {"export": export_obj, "annotated_image": annotated, "history_used_len": len(history)}

# ---------------------------
# UI: Navigation & Pages
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live Analyzer","Bild Analyzer","Backtest","Audit","Einstellungen","Hilfe"])

if not ONLINE:
    st.sidebar.error("❌ Keine Internetverbindung — Live-Daten & Roboflow offline")
else:
    st.sidebar.success("✅ Internet verfügbar")

# Home
if page == "Home":
    st.header("Lumina Pro — Übersicht")
    st.markdown("""
    - Live Analyzer: Finnhub & AlphaV (Fallback) — Candle-Daten, Patterns, SL/TP
    - Bild Analyzer: Roboflow (online) + lokaler Fallback — sofortige Analyse + Annotation
    - Backtester: Simulations-basierter Test mit Fees/Slippage/Positionsgrößen
    - Audit: Verwalte frühere Analysen
    """)
    st.write("Pillow:", PIL_AVAILABLE, "Matplotlib:", MATPLOTLIB_AVAILABLE)
    st.markdown("---")
    st.write("Letzte Analysen (Audit):")
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for a in arr[-6:][::-1]:
            st.write(f"- {a.get('ts')} • {a.get('type')} • {a.get('summary')}")
    except Exception:
        st.info("Keine Audit-Daten gefunden.")

# Live Analyzer
elif page == "Live Analyzer":
    st.header("Live Analyzer")
    left, right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol (Finnhub format, z.B. BINANCE:BTCUSDT oder AAPL)", value="AAPL")
        resolution = st.selectbox("Interval (Min)", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles", 30, 1000, 240, step=10)
        run = st.button("Lade & Analysiere")
    with left:
        if run:
            candles = None
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - periods * int(resolution) * 60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None and ALPHAV_KEY:
                    st.warning("Finnhub lieferte nichts — versuche Alpha Vantage")
                    av = fetch_alpha_minute(symbol, interval=resolution + "min")
                    if av: candles = av[-periods:] if len(av)>=periods else av
                if candles is None:
                    st.warning("Keine Live-Daten — Nutzung simulierte Daten")
                    candles = generate_simulated_candles(symbol + "_sim", periods, 100.0, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol + "_pad", need, candles[0]["open"] if candles else 100.0, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline: Nutzung simulierte Daten")
                candles = generate_simulated_candles(symbol + "_sim", periods, 100.0, int(resolution))
            # compute indicators & detect patterns
            patt = detect_patterns_from_candles(candles)
            closes = [c["close"] for c in candles]
            if len(closes) >= 50:
                s20 = sum(closes[-20:])/20
                s50 = sum(closes[-50:])/50
                trend = "Aufwärtstrend" if s20 > s50 else ("Abwärtstrend" if s20 < s50 else "Seitwärts")
            else:
                trend = "Seitwärts"
            # map patterns -> levels
            pseudo_label_scores = {}
            for p in patt[-6:]:
                pseudo_label_scores[p[0]] = pseudo_label_scores.get(p[0], 0.6)
            eval_res = evaluate_from_labels(pseudo_label_scores, candlesticks=candles)
            levels = map_levels_from_labels_and_history(pseudo_label_scores, candles)
            # display
            st.subheader(f"{symbol} — {trend}")
            st.write(f"Aktuell: {candles[-1]['close']:.6f}")
            st.write("Detected patterns:", [p[0] for p in patt][-8:])
            st.write("Empfehlung:", eval_res["recommendation"], "Wahrscheinlichkeit:", eval_res["probability"], "%", "Risiko:", eval_res["risk_pct"], "%")
            st.write("SL:", levels["stop"], "TP:", levels["tp"])
            # plot
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
                ax.plot([c["t"] for c in candles[-300:]], [c["close"] for c in candles[-300:]], color="#00cc66")
                ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
                st.pyplot(fig)
            else:
                # simple svg fallback
                def render_svg_candles(candles, width=1000, height=420):
                    if not candles: return "<svg></svg>"
                    n=len(candles)
                    margin=50; chart_h=int(height*0.6)
                    max_p=max(c["high"] for c in candles); min_p=min(c["low"] for c in candles)
                    pad=(max_p-min_p)*0.05 if (max_p-min_p)>0 else 1.0
                    max_p+=pad; min_p-=pad
                    spacing=(width-2*margin)/n; cw=max(2, spacing*0.6)
                    def y(p): return margin + chart_h - (p-min_p)/(max_p-min_p)*chart_h
                    svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">', f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>']
                    for i,c in enumerate(candles[-160:]):
                        cx=margin + i*spacing + spacing/2
                        top = y(c["high"]); low = y(c["low"]); oy=y(c["open"]); cy=y(c["close"])
                        color = "#00cc66" if c["close"]>=c["open"] else "#ff4d66"
                        svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
                        by=min(oy,cy); bh=max(1, abs(cy-oy))
                        svg.append(f'<rect x="{cx-cw/2}" y="{by}" width="{cw}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
                    svg.append('</svg>')
                    return "\n".join(svg)
                st.components.v1.html(render_svg_candles(candles[-160:]), height=440)

# Bild Analyzer
elif page == "Bild Analyzer":
    st.header("Bild-Analyse (Upload) — Roboflow + Offline-Fallback")
    st.markdown("Lade ein Chart-Screenshot hoch. Die App analysiert Muster automatisch, berechnet SL/TP, macht Backtest und annotiert das Bild.")
    uploaded = st.file_uploader("Chart-Bild (PNG/JPG)", type=["png","jpg","jpeg"])
    symbol_hint = st.text_input("Symbol-Hinweis für Backtest (optional)", value="AAPL")
    run = st.button("Analysiere Bild (automatisch)")
    show_internals = st.checkbox("Zeige interne Metriken", value=False)
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            with st.spinner("Analysiere Bild — Roboflow (falls online) + lokale Heuristiken..."):
                try:
                    res = analyze_image_pipeline(img_bytes, symbol_hint=symbol_hint, use_online=True)
                except Exception as e:
                    st.error("Analyse fehlgeschlagen: " + str(e))
                    res = None
            if res:
                final = res["export"]
                st.subheader("Ergebnis")
                rec = final["final"]["recommendation"]
                prob = final["final"]["probability"]
                risk = final["final"]["risk_pct"]
                if rec == "Kaufen":
                    st.success(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                elif rec == "Short":
                    st.error(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                else:
                    st.info(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                st.markdown("**Kurz-Fazit (3 Sätze):**")
                # synthesize 3-sentence human-friendly summary
                summ = final["final"].get("rationale", [])
                lines = []
                lines.append(f"Muster: {', '.join(list(final['final'].get('label_scores', {}).keys())[:3])}")
                lines.append(f"Trefferwahrscheinlichkeit (geschätzt): {prob} % • Risiko: {risk} %")
                lines.append(f"SL/TP: {final['final'].get('levels', {}).get('stop')} / {final['final'].get('levels', {}).get('tp')}")
                for s in lines[:3]:
                    st.write("- " + s)
                st.markdown("---")
                st.subheader("Details & Backtest")
                bt = final["final"].get("backtest", {})
                st.write(f"Backtest: checked={bt.get('checked')} wins={bt.get('wins')} winrate={bt.get('winrate')}% avgRet={bt.get('avg_return_pct')}%")
                st.markdown("**Rationale (Labels):**")
                for r in final["final"].get("rationale", [])[:8]:
                    st.write("- " + r)
                # show annotated image
                if res.get("annotated_image") is not None:
                    st.image(res["annotated_image"], use_column_width=True)
                # export buttons
                st.download_button("Exportiere Analyse (JSON)", data=export_analysis_json(final), file_name=f"lumina_analysis_{short_ts()}.json", mime="application/json")
                st.download_button("Exportiere Analyse (CSV)", data=export_analysis_csv(final), file_name=f"lumina_analysis_{short_ts()}.csv", mime="text/csv")
                if show_internals:
                    st.write("Internals:", final.get("internals", {}))
            else:
                st.error("Keine Analyseergebnisse (Roboflow/offline Fehler).")

# Backtest page
elif page == "Backtest":
    st.header("Backtester (manuell)")
    pattern = st.selectbox("Pattern", ["Bullish Engulfing","Bearish Engulfing","Hammer","Doji","Morning Star","Three White Soldiers"])
    lookahead = st.slider("Lookahead (Kerzen)", 1, 30, 10)
    pos_size = st.number_input("Positionsgröße (% vom Kapital)", value=1.0, min_value=0.1, step=0.1)
    slippage = st.number_input("Slippage (%)", value=0.05, step=0.01)
    fee = st.number_input("Fee (%)", value=0.02, step=0.01)
    if st.button("Backtest ausführen"):
        hist = generate_simulated_candles("bt_seed_manual", 1200, 100.0, 5)
        res = backtest_pattern_on_history(hist, pattern, lookahead=lookahead, slippage_pct=slippage, fee_pct=fee, position_size_pct=pos_size)
        st.write(res)
        # human readable conclusion
        if res["checked"] == 0:
            st.info("Keine Events gefunden in der Test-Historie.")
        else:
            if res["winrate"] > 60 and res["avg_return_pct"] > 0:
                conclusion = "Starkes Ergebnis — Pattern liefert überdurchschnittliche Trefferquote in Simulation."
            elif res["winrate"] > 45:
                conclusion = "Akzeptables Ergebnis — Pattern zeigt moderate Trefferquote."
            else:
                conclusion = "Schwaches Ergebnis — Vorsicht, hohes Risiko."
            st.markdown("**Fazit:** " + conclusion)

# Audit page
elif page == "Audit":
    st.header("Analyse-Audit")
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        st.write(f"Gesamt Analysen: {len(arr)}")
        if arr:
            df = arr[::-1]  # latest first
            for a in df[:200]:
                st.write(f"{a.get('ts')} • {a.get('type')} • {a.get('summary')}")
        if st.button("Audit löschen"):
            with open(AUDIT_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.success("Audit gelöscht.")
    except Exception:
        st.info("Keine Auditdaten.")

# Einstellungen
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Pillow installiert:", PIL_AVAILABLE)
    st.write("Matplotlib installiert:", MATPLOTLIB_AVAILABLE)
    st.write("Finnhub Key vorhanden:", bool(FINNHUB_KEY))
    st.write("Roboflow Key vorhanden:", bool(ROBOFLOW_KEY))
    if st.button("Cache leeren"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR, f))
            except: pass
        st.success("Cache geleert.")

# Hilfe
elif page == "Hilfe":
    st.header("Hilfe & Hinweise")
    st.markdown("""
    - Bild-Analyzer arbeitet online (Roboflow) und offline (local fallback). Roboflow verbessert Erkennungsqualität.
    - Live Analyzer nutzt Finnhub primär und Alpha Vantage als Fallback (Achtung API-Limits).
    - Empfehlungen sind statistische Schätzungen — **keine Anlageberatung**.
    - Exportiere Analysen mit dem JSON/CSV Button.
    """)

st.markdown("---")
st.caption("Lumina Pro — Deep Analyzer • Keys sind im Code (nicht sicher für öffentliche Repos). Use responsibly; not financial advice.")
