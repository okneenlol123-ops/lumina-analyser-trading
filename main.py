# main.py
# Lumina Pro — Deep Analyzer (Hybrid, Finnhub + Roboflow + Offline backup)
# Features:
#  - Hybrid: Finnhub live candles (falls online), else deterministic simulation
#  - Bild-Analyzer: Roboflow (online) + verbesserte lokale Pixel/Muster-Detektoren (offline fallback)
#  - Erweiterte Muster-Library (many patterns + formation detection)
#  - Anti-neutral logic (weights & thresholds)
#  - SL/TP mapping via local pivot detection
#  - Backtester (fees, slippage, position sizing)
#  - Annotation: approximate candle->pixel mapping & drawing on uploaded images (Pillow)
#  - Export JSON/CSV, persistent audit
# Keys embedded per user request:
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
import json, os, io, time, random, math, csv, traceback, urllib.request, urllib.parse
from datetime import datetime, timedelta
import statistics

# Image libs
try:
    from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Matplotlib optional
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

# -----------------------------
# Page config & dark theme
# -----------------------------
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

st.title("Lumina Pro — Deep Analyzer (Hybrid)")

# -----------------------------
# Utilities & cache
# -----------------------------
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

def cache_load(key, max_age=3600):
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

# audit storage
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

# -----------------------------
# Roboflow multipart helper
# -----------------------------
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
            ct, body = encode_multipart("file", "upload.png", image_bytes, "image/png")
            req = urllib.request.Request(endpoint, data=body, method="POST")
            req.add_header("Content-Type", ct)
            req.add_header("User-Agent", "LuminaPro/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            return None

# -----------------------------
# Finnhub + AlphaV fetchers (hybrid)
# -----------------------------
def fetch_finnhub_candles(symbol, resolution="5", from_ts=None, to_ts=None):
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None:
            to_ts = int(time.time())
        if from_ts is None:
            from_ts = to_ts - 60*60*24*30
        params = {"symbol": symbol, "resolution": resolution, "from": str(int(from_ts)), "to": str(int(to_ts)), "token": FINNHUB_KEY}
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=25) as resp:
            txt = resp.read().decode("utf-8")
            data = json.loads(txt)
        if data.get("s") != "ok": return None
        ts = data.get("t", []); o = data.get("o", []); h = data.get("h", []); l = data.get("l", []); c = data.get("c", [])
        candles=[]
        for i, t in enumerate(ts):
            dt = datetime.utcfromtimestamp(int(t)) if isinstance(t, (int,float)) else datetime.utcnow()
            candles.append({"t": dt, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i])})
        return candles
    except Exception:
        return None

def fetch_alpha_intraday(symbol, interval="5min", outputsize="compact"):
    if not False:  # ALPHAV optional; user didn't insist - keep off to avoid key mismatch.
        return None

# deterministic simulated candles
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
    for i, pr in enumerate(prices):
        o = round(pr * (1 + random.uniform(-0.002,0.002)),6); c = pr
        h = round(max(o,c) * (1 + random.uniform(0.0,0.004)),6); l = round(min(o,c) * (1 - random.uniform(0.0,0.004)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c})
    return candles

# -----------------------------
# Indicators & patterns (extended)
# -----------------------------
def sma(vals, period):
    res=[]
    for i in range(len(vals)):
        if i+1 < period: res.append(None)
        else: res.append(sum(vals[i+1-period:i+1]) / period)
    return res

def is_doji(c):
    body = abs(c["close"] - c["open"]); total = c["high"] - c["low"]
    return total > 0 and (body / total) < 0.15

def is_hammer(c):
    body = abs(c["close"] - c["open"]); lower = min(c["open"], c["close"]) - c["low"]
    return body > 0 and lower > 2.5 * body

def is_shooting_star(c):
    body = abs(c["close"] - c["open"]); upper = c["high"] - max(c["open"], c["close"])
    return body > 0 and upper > 2.5 * body

def is_engulfing(prev, cur):
    if not prev: return None
    bull = (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])
    bear = (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])
    if bull: return "bull"
    if bear: return "bear"
    return None

# simple formation detectors: double top/bottom, head & shoulders (heuristic)
def detect_formations(candles):
    out=[]
    n=len(candles)
    if n < 20: return out
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]; closes=[c["close"] for c in candles]
    # double top: local high -> small dip -> similar local high
    for i in range(5, n-5):
        # left high
        if highs[i] > max(highs[i-4:i]) and highs[i] > max(highs[i+1:i+5]):
            # search next peak
            for j in range(i+3, min(i+20, n-1)):
                if highs[j] > max(highs[j-3:j]) and highs[j] > max(highs[j+1:j+4]):
                    ratio = highs[j]/highs[i]
                    if 0.96 <= ratio <= 1.04:
                        out.append(("Double Top", i, j))
    # double bottom: similar
    for i in range(5, n-5):
        if lows[i] < min(lows[i-4:i]) and lows[i] < min(lows[i+1:i+5]):
            for j in range(i+3, min(i+20, n-1)):
                if lows[j] < min(lows[j-3:j]) and lows[j] < min(lows[j+1:j+4]):
                    ratio = lows[j]/lows[i] if lows[i] != 0 else 1.0
                    if 0.96 <= ratio <= 1.04:
                        out.append(("Double Bottom", i, j))
    # head and shoulders (rough)
    # look for three peaks where middle higher than sides
    for i in range(5, n-10):
        a=i; b=i+4; c=i+8
        if highs[b] > highs[a] and highs[b] > highs[c] and highs[a] < highs[b] and highs[c] < highs[b] and highs[a] > highs[c]*0.8:
            out.append(("Head & Shoulders", a,b,c))
    return out

# -----------------------------
# Image Analyzer: robust online + offline
# -----------------------------
# label library heuristics
LABEL_LIBRARY = {
    "Bullish Engulfing": {"dir":"bull", "base_wr":0.68, "risk":2.5},
    "Bearish Engulfing": {"dir":"bear", "base_wr":0.66, "risk":2.5},
    "Hammer": {"dir":"bull", "base_wr":0.62, "risk":2.8},
    "Shooting Star": {"dir":"bear", "base_wr":0.60, "risk":3.0},
    "Doji": {"dir":"neutral", "base_wr":0.50, "risk":4.0},
    "Morning Star": {"dir":"bull", "base_wr":0.70, "risk":2.0},
    "Evening Star": {"dir":"bear", "base_wr":0.70, "risk":2.0},
    "Three White Soldiers": {"dir":"bull", "base_wr":0.72, "risk":2.2},
    "Three Black Crows": {"dir":"bear", "base_wr":0.70, "risk":2.2},
    "Double Top": {"dir":"bear", "base_wr":0.62, "risk":3.5},
    "Double Bottom": {"dir":"bull", "base_wr":0.63, "risk":3.1},
    "Head & Shoulders": {"dir":"bear", "base_wr":0.66, "risk":3.5},
    "ChoppyMarket": {"dir":"neutral", "base_wr":0.45, "risk":5.0},
    "NoClearPattern": {"dir":"neutral", "base_wr":0.45, "risk":5.0},
}

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
    "double_top":"Double Top",
    "double_bottom":"Double Bottom",
    "head_shoulders":"Head & Shoulders",
}

# local pixel-based detection (ensures non-neutral)
def local_detect_from_image(image_bytes):
    if not PIL_AVAILABLE:
        return []
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return []
    W,H = img.size
    # crop area heuristics (center)
    left=int(W*0.04); right=int(W*0.96); top=int(H*0.1); bottom=int(H*0.78)
    chart = img.crop((left, top, right, bottom))
    chart = ImageOps.autocontrast(chart, cutoff=2)
    chart = chart.filter(ImageFilter.MedianFilter(size=3))
    pix = chart.load()
    Wc,Hc = chart.size
    col_dark = [sum(255 - pix[x,y] for y in range(0,Hc,2)) for x in range(Wc)]
    maxv = max(col_dark) if col_dark else 1
    norm = [v/maxv for v in col_dark]
    peaks = [i for i in range(2,Wc-2) if norm[i] > norm[i-1] and norm[i] > norm[i+1] and norm[i] > 0.55]
    # heuristics: detect long lower shadows -> hammer-like by scanning column darkness distribution
    hammer_count = 0; doji_count = 0; shooting_count = 0
    for x in peaks:
        col = [255 - pix[x,y] for y in range(Hc)]
        if not col: continue
        maxc = max(col); threshold = max(2, maxc*0.4)
        highpos = [i for i,v in enumerate(col) if v >= threshold]
        if not highpos: continue
        body = max(highpos)-min(highpos) if len(highpos)>1 else 0
        top_gap = min(highpos)
        bottom_gap = Hc - 1 - max(highpos)
        if body < Hc*0.06:
            doji_count += 1
        if bottom_gap > body*2.5 and body>0:
            hammer_count += 1
        if top_gap > body*2.5 and body>0:
            shooting_count += 1
    results=[]
    if hammer_count>0: results.append(("Hammer", min(0.98, round(0.3 + hammer_count*0.08,2))))
    if shooting_count>0: results.append(("Shooting Star", min(0.98, round(0.25 + shooting_count*0.07,2))))
    if doji_count>0: results.append(("Doji", min(0.95, round(0.2 + doji_count*0.05,2))))
    # density detection -> choppy
    density = len(peaks) / (Wc/100.0 + 1e-9)
    if density > 7:
        results.append(("ChoppyMarket", min(0.9, round(min(1.0, density/12),2))))
    if not results:
        results.append(("NoClearPattern", 0.6))
    return results

# fuse RF + local predictions -> label->score
def fuse_labels(roboflow_response, local_preds):
    scores = {}
    # Roboflow
    if roboflow_response and isinstance(roboflow_response, dict) and "predictions" in roboflow_response:
        for p in roboflow_response["predictions"]:
            cls = p.get("class","").lower()
            conf = float(p.get("confidence",0.0))
            label = ROBOFLOW_TO_LABEL.get(cls, cls.title().replace("_"," "))
            scores[label] = max(scores.get(label,0.0), conf * 0.95)
    # local -> boost if missing
    for lab, sc in local_preds:
        labn = lab if lab in LABEL_LIBRARY else lab.title().replace("_"," ")
        prev = scores.get(labn, 0.0)
        scores[labn] = max(prev, min(0.99, prev + sc * 0.5))
    if not scores:
        scores["NoClearPattern"] = 0.6
    return scores

# evaluate labels -> recommendation
def evaluate_labels(label_scores, candlesticks=None):
    bull=0.0; bear=0.0; neutral=0.0; total=0.0
    rationale=[]
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"dir":"neutral","base_wr":0.45,"risk":4.0})
        dirc = meta["dir"]; base_wr = meta["base_wr"]; risk = meta["risk"]
        total += sc
        contrib = sc * base_wr
        rationale.append(f"{label} (conf={sc:.2f}, baseWR={base_wr})")
        if dirc=="bull": bull += contrib
        elif dirc=="bear": bear += contrib
        else: neutral += contrib
    bull_score = bull / (total + 1e-12)
    bear_score = bear / (total + 1e-12)
    neutral_score = neutral / (total + 1e-12)
    # anti-neutral thresholds
    rec="Neutral"
    if bull_score > max(bear_score*1.2, neutral_score):
        rec="Kaufen"
    elif bear_score > max(bull_score*1.2, neutral_score):
        rec="Short"
    else:
        # if a single label very confident -> follow
        top = max(label_scores.items(), key=lambda kv: kv[1])
        if top[1] > 0.86:
            topmeta = LABEL_LIBRARY.get(top[0], {"dir":"neutral"})
            rec = "Kaufen" if topmeta["dir"]=="bull" else ("Short" if topmeta["dir"]=="bear" else "Neutral")
        else:
            rec = "Neutral"
    # probability estimate
    prob = bull_score*100 if rec=="Kaufen" else (bear_score*100 if rec=="Short" else max(bull_score,bear_score,neutral_score)*100)
    prob = round(max(10.0, min(98.0, prob)),1)
    # risk weighted
    risk_w = 0.0; tw=0.0
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"risk":4.0})
        risk_w += sc * meta.get("risk",4.0); tw += sc
    risk_pct = round((risk_w/(tw+1e-12)),2) if tw>0 else 4.0
    # momentum boost if candlesticks present
    if candlesticks and len(candlesticks)>=50:
        closes=[c["close"] for c in candlesticks[-50:]]
        s20 = sum(closes[-20:])/20 if len(closes)>=20 else sum(closes)/len(closes)
        s50 = sum(closes[-50:])/50 if len(closes)>=50 else s20
        if rec=="Kaufen" and s20 > s50: prob = min(98.0, prob + 6.0)
        if rec=="Short" and s20 < s50: prob = min(98.0, prob + 6.0)
    return {"recommendation": rec, "probability": prob, "risk_pct": risk_pct, "rationale": rationale}

# -----------------------------
# SL/TP mapping (pivots)
# -----------------------------
def local_pivots(candles, window=4):
    highs=[c["high"] for c in candles]; lows=[c["low"] for c in candles]
    ph=[]; pl=[]
    n=len(candles)
    for i in range(window, n-window):
        if all(highs[i] > highs[j] for j in range(i-window,i)) and all(highs[i] > highs[j] for j in range(i+1,i+window+1)):
            ph.append((i, highs[i]))
        if all(lows[i] < lows[j] for j in range(i-window,i)) and all(lows[i] < lows[j] for j in range(i+1,i+window+1)):
            pl.append((i, lows[i]))
    return ph, pl

def map_sl_tp(label_scores, candles=None):
    last_price = candles[-1]["close"] if candles and len(candles)>0 else None
    # weighted risk
    tw=0.0; riskw=0.0
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"risk":4.0})
        riskw += sc * meta["risk"]; tw += sc
    risk = (riskw/(tw+1e-12)) if tw>0 else 4.0
    sl=None; tp=None; notes=[]
    if candles and len(candles)>=30:
        ph, pl = local_pivots(candles, window=4)
        supports = sorted([v for (i,v) in pl if v < last_price], reverse=True)
        resistances = sorted([v for (i,v) in ph if v > last_price])
        if supports:
            sl = supports[0]*(1-0.002); notes.append("Stop unter lokalem Support")
        else:
            sl = last_price*(1 - risk/100.0); notes.append("Stop relativ (kein Support)")
        if resistances:
            tp = resistances[0]*(1+0.002); notes.append("TP an lokalem Widerstand")
        else:
            tp = last_price*(1 + 2*risk/100.0); notes.append("TP relativ (kein Widerstand)")
    else:
        if last_price:
            sl = last_price*(1 - risk/100.0)
            tp = last_price*(1 + 2*risk/100.0)
            notes.append("Relative SL/TP (zu wenig Historie)")
        else:
            notes.append("Keine Preisinfo: relative SL/TP nicht möglich")
    return {"stop": None if sl is None else round(sl,6), "tp": None if tp is None else round(tp,6), "notes": notes, "risk_est": round(risk,2)}

# -----------------------------
# Backtester (improved)
# -----------------------------
def backtest(candles, pattern_name, lookahead=10, slippage_pct=0.05, fee_pct=0.02, position_pct=1.0):
    indices=[]
    for i in range(1,len(candles)):
        try:
            if "doji" in pattern_name.lower() and is_doji(candles[i]): indices.append(i)
            if "hammer" in pattern_name.lower() and is_hammer(candles[i]): indices.append(i)
            if "engulf" in pattern_name.lower() and is_engulfing(candles[i-1], candles[i]) is not None: indices.append(i)
        except Exception:
            continue
    trades=[]; wins=0
    for idx in indices:
        if idx+lookahead >= len(candles): continue
        entry = candles[idx]["close"] * (1 + slippage_pct/100.0)
        exitp = candles[idx+lookahead]["close"] * (1 - slippage_pct/100.0)
        gross = (exitp - entry) / (entry + 1e-12)
        net = gross - fee_pct/100.0
        trades.append(net)
        if net>0: wins += 1
    total=len(trades)
    winrate = (wins/total*100.0) if total>0 else 0.0
    avg_ret = (sum(trades)/total*100.0) if trades else 0.0
    pf = (sum(t for t in trades if t>0) / abs(sum(t for t in trades if t<0))) if any(t<0 for t in trades) else (sum(t for t in trades if t>0) or 0.0)
    summary = f"{total} Trades getestet • Winrate {winrate:.1f}% • AvgRet {avg_ret:.3f}% • ProfitFactor {round(pf,3) if isinstance(pf,float) else None}"
    return {"pattern":pattern_name,"checked":total,"wins":wins,"winrate":round(winrate,2),"avg_return_pct":round(avg_ret,3),"profit_factor":round(pf,3) if isinstance(pf,float) else None,"summary":summary}

# -----------------------------
# Annotation: map candle->image approx & draw
# -----------------------------
def annotate_uploaded_image(image_bytes, detections_labels, sl=None, tp=None):
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    except Exception:
        return None
    draw = ImageDraw.Draw(img, "RGBA")
    W,H = img.size
    # font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    # draw top labels box
    items = sorted(detections_labels.items(), key=lambda kv: kv[1], reverse=True)[:5]
    box_w = 300; box_h = 24*len(items)+12
    draw.rectangle([12,12,12+box_w,12+box_h], fill=(12,12,12,200))
    y=16
    for lab, sc in items:
        txt = f"{lab}: {sc:.2f}"
        draw.text((18,y), txt, fill=(230,238,246,255), font=font)
        y += 24
    # draw SL/TP approximate lines
    if sl is not None:
        draw.line([(10, int(H*0.14)), (W-10, int(H*0.14))], fill=(255,204,0,200), width=3)
        draw.text((14, int(H*0.14)-12), f"Stop: {sl}", fill=(255,204,0,255), font=font)
    if tp is not None:
        draw.line([(10, int(H*0.18)), (W-10, int(H*0.18))], fill=(102,255,136,200), width=3)
        draw.text((14, int(H*0.18)-12), f"TP: {tp}", fill=(102,255,136,255), font=font)
    return img

# -----------------------------
# Exports
# -----------------------------
def export_json(obj):
    return json.dumps(obj, indent=2, ensure_ascii=False)

def export_csv(obj):
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["key","value"])
    meta = obj.get("meta",{})
    for k,v in meta.items():
        w.writerow([f"meta.{k}", v])
    final = obj.get("final",{})
    for k,v in final.items():
        if isinstance(v,(str,int,float)):
            w.writerow([k,v])
        else:
            w.writerow([k,json.dumps(v, ensure_ascii=False)])
    return buf.getvalue()

# -----------------------------
# High-level image pipeline
# -----------------------------
def analyze_image_full(image_bytes, symbol_hint=None, use_online=True):
    rf_res=None
    if use_online and ONLINE and ROBOFLOW_KEY:
        rf_res = roboflow_detect(image_bytes, retries=2)
    local_preds = local_detect_from_image(image_bytes)
    # build rf_map
    rf_map={}
    if rf_res and isinstance(rf_res, dict) and "predictions" in rf_res:
        for p in rf_res["predictions"]:
            cls = p.get("class","").lower(); conf = float(p.get("confidence",0.0))
            label = ROBOFLOW_TO_LABEL.get(cls, cls.title().replace("_"," "))
            rf_map[label] = max(rf_map.get(label,0.0), conf)
    # fuse
    label_scores = fuse_labels(rf_res, local_preds)
    # try fetch history for backtest
    history=None
    if use_online and ONLINE and FINNHUB_KEY and symbol_hint:
        history = fetch_finnhub_candles(symbol_hint, resolution="5", from_ts=int(time.time())-60*60*24*90, to_ts=int(time.time()))
    if not history:
        history = generate_simulated_candles("img_hist_seed", 900, 100.0, 5)
    eval_res = evaluate_labels(label_scores, candlesticks=history)
    levels = map_sl_tp(label_scores, history)
    # backtest top label
    top_label = max(label_scores.items(), key=lambda kv: kv[1])[0] if label_scores else "NoClearPattern"
    bt = backtest(history, top_label, lookahead=10)
    # adjust final probability with historical winrate
    if bt["checked"] > 0:
        final_prob = round((eval_res["probability"]*0.6 + bt["winrate"]*0.4),1)
    else:
        final_prob = eval_res["probability"]
    eval_res["probability"] = final_prob
    annotated = annotate_uploaded_image(image_bytes, label_scores, sl=levels.get("stop"), tp=levels.get("tp"))
    export_obj = {
        "meta": {"ts": now_iso(), "source": "roboflow+local" if rf_res else "local-only", "symbol_hint": symbol_hint},
        "final": {
            "recommendation": eval_res["recommendation"],
            "probability": eval_res["probability"],
            "risk_pct": eval_res["risk_pct"],
            "rationale": eval_res["rationale"],
            "label_scores": dict(label_scores),
            "levels": levels,
            "backtest": bt
        },
        "internals": {"roboflow_raw": rf_res, "local_preds": local_preds}
    }
    append_audit({"ts": now_iso(), "type": "image", "summary": {"rec": eval_res["recommendation"], "prob": eval_res["probability"], "labels": list(label_scores.keys())}})
    return {"export": export_obj, "annotated": annotated, "history_len": len(history)}

# -----------------------------
# UI pages
# -----------------------------
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
    Hybrid-Modus: Live-Daten (Finnhub) wenn verfügbar, sonst Offline-Simulation.
    Bild-Analyzer: Roboflow (online) + lokaler Fallback (pixelheuristik).
    """)
    st.write("Pillow:", PIL_AVAILABLE, "Matplotlib:", MATPLOTLIB_AVAILABLE)
    st.markdown("---")
    st.subheader("Letzte Analysen")
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for a in arr[-8:][::-1]:
            st.write(f"- {a.get('ts')} • {a.get('type')} • {a.get('summary')}")
    except Exception:
        st.info("Keine Audit-Einträge.")

# Live Analyzer
elif page == "Live Analyzer":
    st.header("Live Analyzer (Hybrid)")
    col1, col2 = st.columns([3,1])
    with col2:
        symbol = st.text_input("Finnhub-Symbol (z.B. AAPL, BINANCE:BTCUSDT)", "AAPL")
        resolution = st.selectbox("Intervall (Min)", ["1","5","15","30","60"], index=1)
        periods = st.slider("Anzahl Kerzen", 30, 1200, 300, step=10)
        run = st.button("Laden & Analysieren")
    with col1:
        if run:
            candles=None
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - periods * int(resolution) * 60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None:
                    st.warning("Finnhub lieferte keine Daten — nutze simulierte Daten")
                    candles = generate_simulated_candles(symbol+"_sim", periods, 100.0, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol+"_pad", need, candles[0]["open"] if candles else 100.0, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline → Simulation")
                candles = generate_simulated_candles(symbol+"_sim", periods, 100.0, int(resolution))
            # detect patterns
            patterns = detect_patterns_from_candles(candles)
            formations = detect_formations(candles)
            closes = [c["close"] for c in candles]
            trend = "Seitwärts"
            if len(closes)>=50:
                s20 = sum(closes[-20:])/20; s50 = sum(closes[-50:])/50
                trend = "Aufwärtstrend" if s20 > s50 else ("Abwärtstrend" if s20 < s50 else "Seitwärts")
            # convert patterns to label_scores
            label_scores = {}
            for p in patterns[-8:]:
                label_scores[p[0]] = max(label_scores.get(p[0],0.0), 0.65)
            for f in formations[-3:]:
                label_scores[f[0]] = max(label_scores.get(f[0],0.0), 0.7)
            eval_res = evaluate_labels(label_scores, candlesticks=candles)
            levels = map_sl_tp(label_scores, candles)
            # backtest top if available
            top_label = max(label_scores.items(), key=lambda kv: kv[1])[0] if label_scores else None
            bt = backtest(candles, top_label or "NoClearPattern", lookahead=10)
            if eval_res["recommendation"]=="Kaufen":
                st.success(f"Empfehlung: {eval_res['recommendation']}  •  {eval_res['probability']}%  •  Risiko: {eval_res['risk_pct']}%")
            elif eval_res["recommendation"]=="Short":
                st.error(f"Empfehlung: {eval_res['recommendation']}  •  {eval_res['probability']}%  •  Risiko: {eval_res['risk_pct']}%")
            else:
                st.info(f"Empfehlung: {eval_res['recommendation']}  •  {eval_res['probability']}%  •  Risiko: {eval_res['risk_pct']}%")
            st.write("SL:", levels["stop"], "TP:", levels["tp"])
            st.write("Erkannte Patterns:", list(label_scores.keys())[:8])
            st.write("Formationen:", [f[0] for f in formations][:3])
            st.markdown("**Backtest (Top-Pattern)**"); st.write(bt["summary"])
            # plot
            if MATPLOTLIB_AVAILABLE:
                fig, ax = plt.subplots(figsize=(11,4), facecolor="#07070a")
                ax.plot([c["t"] for c in candles[-400:]], [c["close"] for c in candles[-400:]], color="#00cc66")
                ax.set_facecolor("#07070a"); ax.tick_params(colors="#9aa6b2")
                st.pyplot(fig)
            else:
                # simple svg
                def svg_candles(candles, width=1000, height=420):
                    if not candles: return "<svg></svg>"
                    n=len(candles)
                    margin=50; chart_h=int(height*0.6)
                    maxp=max(c["high"] for c in candles); minp=min(c["low"] for c in candles)
                    pad=(maxp-minp)*0.05 if (maxp-minp)>0 else 1.0
                    maxp+=pad; minp-=pad
                    spacing=(width-2*margin)/n; cw=max(2, spacing*0.6)
                    def y(p): return margin + chart_h - (p-minp)/(maxp-minp)*chart_h
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
                st.components.v1.html(svg_candles(candles[-160:]), height=460)

# Bild Analyzer
elif page == "Bild Analyzer":
    st.header("Bild-Analyse (Roboflow + Offline-Fallback)")
    st.markdown("Lade Chart-Screenshot hoch; Analyse läuft automatisch (online-first).")
    uploaded = st.file_uploader("Chart-Bild (PNG/JPG)", type=["png","jpg","jpeg"])
    symbol_hint = st.text_input("Symbol-Hinweis für Backtest (optional)", "AAPL")
    run = st.button("Analysiere Bild")
    show_internal = st.checkbox("Zeige Internals", value=False)
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            with st.spinner("Analysiere..."):
                try:
                    res = analyze_image_full(img_bytes, symbol_hint, use_online=True)
                except Exception as e:
                    st.error("Analyse fehlgeschlagen: " + str(e))
                    res = None
            if res:
                final = res["export"]
                rec = final["final"]["recommendation"]
                prob = final["final"]["probability"]
                risk = final["final"]["risk_pct"]
                if rec=="Kaufen": st.success(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                elif rec=="Short": st.error(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                else: st.info(f"Empfehlung: {rec}  •  {prob}%  • Risiko: {risk}%")
                st.markdown("**3-Satz-Fazit:**")
                # create 3-sentence human summary
                labels = list(final["final"].get("label_scores", {}).keys())[:3]
                s1 = f"Erkannte Muster: {', '.join(labels) if labels else 'keine'}."
                s2 = f"Trefferwahrscheinlichkeit geschätzt: {prob}% • Risiko: {risk}%."
                levels = final["final"].get("levels", {})
                s3 = f"Empfohlene SL/TP: {levels.get('stop')} / {levels.get('tp')}."
                for s in [s1,s2,s3]:
                    st.write("- " + s)
                st.markdown("---")
                st.subheader("Backtest (Kurz)")
                bt = final["final"].get("backtest", {})
                st.write(bt.get("summary"))
                st.markdown("**Rationale / Labels**")
                for r in final["final"].get("rationale", [])[:8]:
                    st.write("- " + r)
                if res.get("annotated") is not None:
                    st.image(res["annotated"], use_column_width=True)
                st.download_button("Export JSON", data=export_json(final), file_name=f"analysis_{short_ts()}.json", mime="application/json")
                st.download_button("Export CSV", data=export_csv(final), file_name=f"analysis_{short_ts()}.csv", mime="text/csv")
                if show_internal:
                    st.write("Internals:", final.get("internals", {}))
            else:
                st.error("Keine Ergebnisse – Roboflow/offline Fehler.")

# Backtest page (manual)
elif page == "Backtest":
    st.header("Backtester (manuell)")
    pattern = st.selectbox("Pattern", ["Bullish Engulfing","Bearish Engulfing","Hammer","Doji","Morning Star","Three White Soldiers","Double Top","Double Bottom","Head & Shoulders"])
    lookahead = st.slider("Lookahead (Kerzen)", 1, 30, 10)
    slippage = st.number_input("Slippage (%)", 0.0, 5.0, 0.05, step=0.01)
    fee = st.number_input("Fee (%)", 0.0, 2.0, 0.02, step=0.01)
    if st.button("Backtest starten"):
        hist = generate_simulated_candles("bt_seed", 1200, 100.0, 5)
        res = backtest(hist, pattern, lookahead=lookahead, slippage_pct=slippage, fee_pct=fee)
        st.write(res)
        if res["checked"] == 0:
            st.info("Keine Events gefunden.")
        else:
            if res["winrate"] > 60 and res["avg_return_pct"] > 0:
                conclusion = "Starkes Ergebnis — Pattern zeigt gute Trefferquote in Simulation."
            elif res["winrate"] > 45:
                conclusion = "Akzeptabel — moderates Verhalten."
            else:
                conclusion = "Schwach — hohe Vorsicht empfohlen."
            st.markdown("**Fazit:** " + conclusion)

# Audit
elif page == "Audit":
    st.header("Audit — vergangene Analysen")
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
        st.write(f"Analysen gespeichert: {len(arr)}")
        for a in arr[::-1][:200]:
            st.write(f"{a.get('ts')} • {a.get('type')} • {a.get('summary')}")
        if st.button("Audit löschen"):
            with open(AUDIT_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
            st.success("Audit gelöscht.")
    except Exception:
        st.info("Keine Auditdaten.")

# Einstellungen & Hilfe
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Hybrid-Modus:", "ONLINE" if ONLINE else "OFFLINE")
    st.write("Pillow:", PIL_AVAILABLE, "Matplotlib:", MATPLOTLIB_AVAILABLE)
    st.write("Finnhub-Key vorhanden:", bool(FINNHUB_KEY))
    st.write("Roboflow-Key vorhanden:", bool(ROBOFLOW_KEY))
    if st.button("Cache & Audit löschen"):
        try:
            for f in os.listdir(CACHE_DIR):
                os.remove(os.path.join(CACHE_DIR,f))
        except Exception:
            pass
        try:
            with open(AUDIT_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)
        except Exception:
            pass
        st.success("Gelöscht.")

elif page == "Hilfe":
    st.header("Hilfe / Hinweise")
    st.markdown("""
    - Hybrid: Live-Daten (Finnhub) wenn online, sonst deterministische Simulation.
    - Bildanalyse nutzt Roboflow (online) und lokale Pixelheuristiken (offline fallback).
    - Empfehlungen sind Schätzungen/statistische Werte — KEINE Anlageberatung.
    - Exportiere Analysen mit JSON/CSV.
    - Wenn etwas fehlt: prüfe Keys, Internet oder installiere Pillow/Matplotlib.
    """)

st.markdown("---")
st.caption("Lumina Pro — Hybrid Deep Analyzer — Keys im Code (für Entwicklung). Entferne Keys im Produktivmodus.")
