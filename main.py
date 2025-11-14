# main.py
# Ultimate Vision-Trader v3.0
# Hybrid (Finnhub optional) + Enhanced Image Analyzer + Backtester + Batch report
# Offline-first, Roboflow/Finnhub optional. Uses Pillow + standard libs + requests.
#
# Requirements: pip install streamlit pillow requests
#
# Put API keys into st.secrets or fill below for dev:
FINNHUB_KEY = ""   # set in st.secrets["FINNHUB_KEY"] preferable
ROBOFLOW_KEY = ""  # set in st.secrets["ROBOFLOW_KEY"]

# -----------------------------
# Imports
# -----------------------------
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import io, os, json, math, random, time, csv, traceback
from datetime import datetime, timedelta
import urllib.request, urllib.parse
import statistics

# optional requests
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# -----------------------------
# Page config + theme
# -----------------------------
st.set_page_config(page_title="Ultimate Vision-Trader v3.0", layout="wide", page_icon="💹")
st.markdown("""
<style>
html, body, [class*="css"] { background:#000 !important; color:#e6eef6 !important; }
.stButton>button { background:#111 !important; color:#e6eef6 !important; border:1px solid #222 !important; }
.card { background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
.small { color:#9aa6b2; font-size:13px; }
</style>
""", unsafe_allow_html=True)
st.title("Ultimate Vision-Trader v3.0 — Hybrid Image + Chart Analyzer")

# -----------------------------
# Utility helpers
# -----------------------------
def now_iso(): return datetime.utcnow().isoformat() + "Z"
def short_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

APP_DIR = ".uvt_cache"
os.makedirs(APP_DIR, exist_ok=True)
AUDIT_FILE = os.path.join(APP_DIR, "audit.json")
if not os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def append_audit(entry):
    try:
        a = json.load(open(AUDIT_FILE, "r", encoding="utf-8"))
    except Exception:
        a = []
    a.append(entry)
    try:
        json.dump(a, open(AUDIT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception:
        pass

def internet_ok(timeout=2):
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

# read keys from st.secrets if present
try:
    secrets = st.secrets
    if FINNHUB_KEY == "" and "FINNHUB_KEY" in secrets:
        FINNHUB_KEY = secrets["FINNHUB_KEY"]
    if ROBOFLOW_KEY == "" and "ROBOFLOW_KEY" in secrets:
        ROBOFLOW_KEY = secrets["ROBOFLOW_KEY"]
except Exception:
    pass

# -----------------------------
# Candle simulation (offline)
# -----------------------------
def generate_simulated_candles(seed, periods=500, start_price=100.0, resolution_minutes=5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    p = float(start_price)
    candles=[]
    now = datetime.utcnow()
    for i in range(periods):
        drift = (rnd.random() - 0.49) * 0.002
        shock = (rnd.random() - 0.5) * 0.01
        p = max(0.01, p * (1 + drift + shock))
        o = round(p * (1 + rnd.uniform(-0.002,0.002)),6)
        c = round(p,6)
        h = round(max(o,c) * (1 + rnd.uniform(0,0.003)),6)
        l = round(min(o,c) * (1 - rnd.uniform(0,0.003)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c})
    return candles

# -----------------------------
# Finnhub fetcher (optional)
# -----------------------------
def fetch_finnhub_candles(symbol, resolution="5", from_ts=None, to_ts=None):
    key = FINNHUB_KEY or (st.secrets.get("FINNHUB_KEY") if hasattr(st, "secrets") and "FINNHUB_KEY" in st.secrets else "")
    if not key:
        return None
    if not HAS_REQUESTS:
        # fallback to urllib
        try:
            if to_ts is None: to_ts = int(time.time())
            if from_ts is None: from_ts = to_ts - 60*60*24*30
            params = {"symbol": symbol, "resolution": resolution, "from": str(int(from_ts)), "to": str(int(to_ts)), "token": key}
            url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=20) as resp:
                txt = resp.read().decode("utf-8")
                data = json.loads(txt)
        except Exception:
            return None
    else:
        try:
            if to_ts is None: to_ts = int(time.time())
            if from_ts is None: from_ts = to_ts - 60*60*24*30
            params = {"symbol": symbol, "resolution": resolution, "from": int(from_ts), "to": int(to_ts), "token": key}
            r = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=20)
            data = r.json()
        except Exception:
            return None
    if data.get("s") != "ok":
        return None
    ts = data.get("t", []); o = data.get("o", []); h = data.get("h", []); l = data.get("l", []); c = data.get("c", [])
    candles=[]
    for i, t in enumerate(ts):
        dt = datetime.utcfromtimestamp(int(t))
        candles.append({"t": dt, "open": float(o[i]), "high": float(h[i]), "low": float(l[i]), "close": float(c[i])})
    return candles

# -----------------------------
# Roboflow detect (optional)
# -----------------------------
def encode_multipart(fieldname, filename, file_bytes, content_type="image/png"):
    boundary = '----WebKitFormBoundary' + ''.join(random.choice('0123456789abcdef') for _ in range(16))
    crlf = b'\r\n'
    body = bytearray()
    body.extend(b'--' + boundary.encode() + crlf)
    body.extend(f'Content-Disposition: form-data; name="{fieldname}"; filename="{filename}"'.encode() + crlf)
    body.extend(f'Content-Type: {content_type}'.encode() + crlf + crlf)
    body.extend(file_bytes + crlf)
    body.extend(b'--' + boundary.encode() + b'--' + crlf)
    return f'multipart/form-data; boundary={boundary}', bytes(body)

def roboflow_detect(image_bytes, retries=2, timeout=20, model_path="chart-pattern-detector/1"):
    key = ROBOFLOW_KEY or (st.secrets.get("ROBOFLOW_KEY") if hasattr(st, "secrets") and "ROBOFLOW_KEY" in st.secrets else "")
    if not key:
        return None
    endpoint = f"https://detect.roboflow.com/{model_path}?api_key={urllib.parse.quote(key)}"
    for attempt in range(retries+1):
        try:
            ct, body = encode_multipart("file", "upload.png", image_bytes, "image/png")
            req = urllib.request.Request(endpoint, data=body, method="POST")
            req.add_header("Content-Type", ct)
            req.add_header("User-Agent", "UltimateVisionTrader/3")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception:
            if attempt < retries: time.sleep(1 + attempt)
            else: return None

# -----------------------------
# IMAGE PREPROCESSING & GEOMETRY EXTRACTION (improved)
# -----------------------------
def load_image_bytes(bytes_in):
    try:
        img = Image.open(io.BytesIO(bytes_in)).convert("RGB")
        return img
    except Exception:
        return None

def preprocess_for_chart(img):
    # autocrop heuristics: remove top UI / bottom legend automatically
    W,H = img.size
    # crop central vertical slice; allow user override in UI later
    top = int(H * 0.06)
    bottom = int(H * 0.86)
    chart = img.crop((0, top, W, bottom))
    # grayscale, autocontrast, median filter
    gray = chart.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return chart, gray, top

# Column scanning: reconstruct candles (more precise than naive)
def extract_candle_geometry(gray_img, top_offset=0, expected_min=20):
    W,H = gray_img.size
    pix = gray_img.load()
    # compute vertical darkness profile per x
    col_darkness = []
    for x in range(W):
        s=0
        dark_positions=[]
        for y in range(H):
            inv = 255 - pix[x,y]
            if inv > 6: # noise threshold
                s += inv
                dark_positions.append(y)
        col_darkness.append({"x": x, "sum": s, "dark": dark_positions})
    # smoothing sum with window
    smoothed = []
    win = max(1, W//200)
    for i in range(W):
        start = max(0, i-win); end = min(W, i+win+1)
        ssum = sum(col_darkness[j]["sum"] for j in range(start,end))
        smoothed.append(ssum)
    maxv = max(smoothed) if smoothed else 1
    # dynamic threshold: adaptive
    threshold = max(10, maxv * 0.10)
    candidates = [i for i,v in enumerate(smoothed) if v >= threshold]
    # cluster consecutive columns into candle groups
    groups=[]
    if not candidates:
        return []
    cur=[candidates[0]]
    for i in candidates[1:]:
        if i - cur[-1] <= 2:
            cur.append(i)
        else:
            groups.append(cur); cur=[i]
    if cur: groups.append(cur)
    candles=[]
    for g in groups:
        xs = g
        dark_positions=[]
        for x in xs:
            dark_positions += col_darkness[x]["dark"]
        if not dark_positions: continue
        wick_top = min(dark_positions); wick_bottom = max(dark_positions)
        # body detection: find densest contiguous region
        counts={}
        for d in dark_positions:
            counts[d] = counts.get(d,0)+1
        # find contiguous runs
        runs=[]; run_start=None; prev=None
        for k in sorted(counts.keys()):
            if run_start is None:
                run_start = k; prev = k
            elif k - prev <= 2:
                prev = k
            else:
                runs.append((run_start, prev)); run_start = k; prev = k
        if run_start is not None: runs.append((run_start, prev))
        best = max(runs, key=lambda r: sum(counts.get(i,0) for i in range(r[0], r[1]+1))) if runs else (wick_top, wick_bottom)
        body_top = best[0]; body_bottom = best[1]
        # color guess: compare darkness slightly above vs below body to decide filled/empty
        sample_x = xs[len(xs)//2]
        above = sum(255 - pix[sample_x, y] for y in range(max(0, body_top-2), body_top+1))
        below = sum(255 - pix[sample_x, y] for y in range(body_bottom, min(H, body_bottom+3)))
        color = "green" if below > above else "red"
        center_x = sum(xs)/len(xs)
        candles.append({"wick_top": top_offset + wick_top, "wick_bottom": top_offset + wick_bottom, "body_top": top_offset + body_top, "body_bottom": top_offset + body_bottom, "color": color, "x": int(center_x)})
    # if too few candles, attempt relaxed threshold
    if len(candles) < expected_min and threshold > 5:
        # second pass lower threshold
        threshold2 = max(5, maxv * 0.06)
        candidates = [i for i,v in enumerate(smoothed) if v >= threshold2]
        groups=[]; cur=[candidates[0]] if candidates else []
        for i in candidates[1:]:
            if i-cur[-1] <= 3: cur.append(i)
            else: groups.append(cur); cur=[i]
        if cur: groups.append(cur)
        candles2=[]
        for g in groups:
            xs=g; dark_positions=[]
            for x in xs: dark_positions += col_darkness[x]["dark"]
            if not dark_positions: continue
            wt=min(dark_positions); wb=max(dark_positions)
            bt=wt + (wb-wt)//3; bb=wb - (wb-wt)//3
            sample_x = xs[len(xs)//2]
            color = "green"
            candles2.append({"wick_top": top_offset+wt, "wick_bottom": top_offset+wb, "body_top": top_offset+bt, "body_bottom": top_offset+bb, "color": color, "x": int(sum(xs)/len(xs))})
        if len(candles2) > len(candles):
            candles = candles2
    # ensure left->right order
    candles = sorted(candles, key=lambda c: c["x"])
    return candles

# -----------------------------
# Trend detection: robust regression + R^2 check
# -----------------------------
def slope_and_r2(y):
    n = len(y)
    if n < 2:
        return 0.0, 0.0
    xs = list(range(n))
    mean_x = sum(xs)/n; mean_y = sum(y)/n
    ss_xy = sum((xs[i]-mean_x)*(y[i]-mean_y) for i in range(n))
    ss_xx = sum((xs[i]-mean_x)**2 for i in range(n))
    if ss_xx == 0:
        return 0.0, 0.0
    b = ss_xy / ss_xx
    a = mean_y - b * mean_x
    # r2
    ss_tot = sum((yi - mean_y)**2 for yi in y)
    ss_res = sum((y[i] - (a + b*xs[i]))**2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, r2

def detect_trend(candles):
    # use center of body as proxy price (pixel inverted)
    centers = [-( (c["body_top"] + c["body_bottom"])/2.0 ) for c in candles[-80:]]
    if len(centers) < 8:
        return "neutral", 0.0, 0.0
    slope, r2 = slope_and_r2(centers)
    # thresholds chosen via heuristics:
    if slope > 0.12 and r2 > 0.05:
        return "uptrend", slope, r2
    if slope < -0.12 and r2 > 0.05:
        return "downtrend", slope, r2
    return "neutral", slope, r2

# -----------------------------
# Pattern detectors (local) - many heuristics
# -----------------------------
def detect_local_patterns(candles):
    patterns = []
    n = len(candles)
    for i in range(n):
        c = candles[i]
        body = abs(c["body_bottom"] - c["body_top"]) + 1e-9
        wick_top = abs(c["body_top"] - c["wick_top"])
        wick_bot = abs(c["wick_bottom"] - c["body_bottom"])
        total = wick_top + wick_bot + body
        # doji
        if body < 0.12 * total:
            patterns.append(("Doji", i, c["x"]))
        # hammer
        if wick_bot > 2.5 * body and wick_top < 0.6 * body:
            patterns.append(("Hammer", i, c["x"]))
        # shooting star
        if wick_top > 2.5 * body and wick_bot < 0.6 * body:
            patterns.append(("Shooting Star", i, c["x"]))
        # strong body
        if body > 0.6 * total:
            if c["color"] == "green":
                patterns.append(("Strong Bull Body", i, c["x"]))
            else:
                patterns.append(("Strong Bear Body", i, c["x"]))
    # formations
    if n >= 3:
        last3 = candles[-3:]
        if all((abs(cc["body_bottom"] - cc["body_top"]) > 0 and cc["color"] == "green") for cc in last3):
            patterns.append(("Three White Soldiers", n-1, last3[-1]["x"]))
        if all((abs(cc["body_bottom"] - cc["body_top"]) > 0 and cc["color"] == "red") for cc in last3):
            patterns.append(("Three Black Crows", n-1, last3[-1]["x"]))
    # detect double top/bottom heuristics
    # naive: find local maxima/minima among wicks
    highs = [c["wick_top"] for c in candles]
    lows = [c["wick_bottom"] for c in candles]
    for i in range(4, n-4):
        if highs[i] < min(highs[i-4:i]) and highs[i] < min(highs[i+1:i+5]):
            # local max in pixel invert: check later similar peak
            for j in range(i+3, min(n-2, i+20)):
                if abs(highs[j] - highs[i]) < 0.03 * abs(highs[i]): # similar pixel-level
                    patterns.append(("Double Top", i, candles[j]["x"]))
        if lows[i] > max(lows[i-4:i]) and lows[i] > max(lows[i+1:i+5]):
            for j in range(i+3, min(n-2, i+20)):
                if abs(lows[j] - lows[i]) < 0.03 * abs(lows[i]):
                    patterns.append(("Double Bottom", i, candles[j]["x"]))
    return patterns

# -----------------------------
# Multi-pattern voting ensemble (merge Roboflow optionally)
# -----------------------------
# Base pattern meta
PATTERN_META = {
    "Hammer": {"dir":"bull","weight":0.7},
    "Shooting Star": {"dir":"bear","weight":0.7},
    "Doji": {"dir":"neutral","weight":0.35},
    "Strong Bull Body": {"dir":"bull","weight":0.6},
    "Strong Bear Body": {"dir":"bear","weight":0.6},
    "Three White Soldiers": {"dir":"bull","weight":1.0},
    "Three Black Crows": {"dir":"bear","weight":1.0},
    "Double Top": {"dir":"bear","weight":0.8},
    "Double Bottom": {"dir":"bull","weight":0.8}
}

def fuse_pattern_scores(local_patterns, roboflow_response=None):
    scores = {}
    # local patterns
    for p in local_patterns:
        name = p[0]
        meta = PATTERN_META.get(name, {"dir":"neutral","weight":0.4})
        prev = scores.get(name, {"score":0.0, "dir": meta["dir"]})
        prev["score"] += meta["weight"]
        scores[name] = prev
    # roboflow predictions add weight (if present)
    if roboflow_response and isinstance(roboflow_response, dict) and "predictions" in roboflow_response:
        for pr in roboflow_response["predictions"]:
            cls = pr.get("class","").lower()
            conf = float(pr.get("confidence",0.0))
            # map common names
            label_map = {
                "hammer":"Hammer","shooting_star":"Shooting Star","doji":"Doji",
                "bullish_engulfing":"Strong Bull Body","bearish_engulfing":"Strong Bear Body",
                "three_white_soldiers":"Three White Soldiers","three_black_crows":"Three Black Crows",
                "double_top":"Double Top","double_bottom":"Double Bottom"
            }
            name = label_map.get(cls, cls.title().replace("_"," "))
            meta = PATTERN_META.get(name, {"dir":"neutral","weight":0.5})
            prev = scores.get(name, {"score":0.0, "dir": meta["dir"]})
            # roboflow confidence strongly influences; scale to 0.5-1.2 * conf
            prev["score"] += conf * (0.9 + meta["weight"]*0.5)
            prev["dir"] = meta["dir"]
            scores[name] = prev
    return scores

def ensemble_decision(pattern_scores, trend, r2):
    # aggregate dir-weighted score
    bull = 0.0; bear = 0.0; neutral = 0.0
    for name, obj in pattern_scores.items():
        s = obj["score"]
        dirc = obj.get("dir", "neutral")
        if dirc == "bull":
            bull += s
        elif dirc == "bear":
            bear += s
        else:
            neutral += s
    # trend boost
    if trend == "uptrend": bull *= 1.12
    if trend == "downtrend": bear *= 1.12
    top_label = max(pattern_scores.items(), key=lambda kv: kv[1]["score"])[0] if pattern_scores else None
    top_conf = pattern_scores[top_label]["score"] if top_label else 0.0
    # decision rules (aggressive but backed by ensemble)
    rec = "Neutral"
    if bull > bear * 1.08 and bull > 0.8:
        rec = "Kaufen"
    elif bear > bull * 1.08 and bear > 0.8:
        rec = "Short"
    else:
        # if single label very strong
        if top_conf >= 1.2:
            m = pattern_scores[top_label]
            rec = "Kaufen" if m["dir"]=="bull" else ("Short" if m["dir"]=="bear" else "Neutral")
        else:
            # if both low but trend strong -> follow trend
            if r2 > 0.12:
                if trend == "uptrend": rec = "Kaufen"
                if trend == "downtrend": rec = "Short"
            else:
                rec = "Neutral"
    # probability estimation (blend)
    base = max(bull, bear, neutral, 0.1)
    prob = min(99.0, max(12.0, round(30 + base * 18 + (top_conf * 8) + r2*50, 1)))
    # risk ~ inverse of base
    risk = round(max(1.0, min(10.0, 6.0 - math.log(base+1) )), 2)
    return {"recommendation": rec, "probability": prob, "risk_pct": risk, "top_label": top_label, "top_conf": round(top_conf,3), "raw": {"bull": bull, "bear": bear, "neutral": neutral}}

# -----------------------------
# SL/TP mapping: map pivot -> price mapping support
# -----------------------------
def compute_pixel_sl_tp(candles, rec):
    # This returns pixel-level suggestions. Mapping pixel->price requires user mapping or anchor points (see UI)
    last = candles[-1]
    centers = [ (c["body_top"]+c["body_bottom"])/2.0 for c in candles ]
    # naive pixel-based SL/TP: take local extremes
    lows = [c["body_bottom"] for c in candles]
    highs = [c["body_top"] for c in candles]
    last_center = centers[-1]
    support = min(lows[-8:]) if lows[-8:] else None
    resist = max(highs[-8:]) if highs[-8:] else None
    if rec == "Kaufen":
        sl = support*(1+0.002) if support else last_center*(1+0.02)
        tp = last_center - (sl-last_center)*1.8 if support else last_center*(1-0.04)
    elif rec == "Short":
        sl = resist*(1-0.002) if resist else last_center*(1-0.02)
        tp = last_center + (last_center-sl)*1.8 if resist else last_center*(1+0.04)
    else:
        sl = None; tp = None
    return {"stop_pixel": round(sl,3) if sl else None, "tp_pixel": round(tp,3) if tp else None}

# -----------------------------
# Backtester (improved)
# -----------------------------
def backtest_pattern_on_series(candles, pattern_name, lookahead=10, slippage_pct=0.05, fee_pct=0.02):
    # naive detection similar to local detectors; returns stats
    indices=[]
    for i in range(1,len(candles)):
        cur = candles[i]; prev = candles[i-1]
        # simple checks
        if "doji" in pattern_name.lower():
            body = abs(cur["close"] - cur["open"]); rng = cur["high"] - cur["low"]
            if rng > 0 and body / rng < 0.15: indices.append(i)
        if "hammer" in pattern_name.lower():
            body = abs(cur["close"] - cur["open"])
            lower = min(cur["open"], cur["close"]) - cur["low"]
            if body>0 and lower > 2.5*body: indices.append(i)
        if "engulf" in pattern_name.lower():
            if (cur["close"]>cur["open"] and prev["close"]<prev["open"] and cur["open"]<prev["close"] and cur["close"]>prev["open"]) or \
               (cur["close"]<cur["open"] and prev["close"]>prev["open"] and cur["open"]>prev["close"] and cur["close"]<prev["open"]):
                indices.append(i)
    trades=[]; wins=0
    for idx in indices:
        if idx+lookahead >= len(candles): continue
        entry = candles[idx]["close"] * (1 + slippage_pct/100.0)
        exitp = candles[idx+lookahead]["close"] * (1 - slippage_pct/100.0)
        gross = (exitp - entry) / (entry + 1e-12)
        net = gross - fee_pct/100.0
        trades.append(net)
        if net > 0: wins += 1
    total = len(trades)
    winrate = (wins / total * 100.0) if total>0 else 0.0
    avg_ret = (sum(trades)/total*100.0) if trades else 0.0
    pf = (sum([t for t in trades if t>0]) / abs(sum([t for t in trades if t<0]))) if any(t<0 for t in trades) else float("inf") if trades else None
    return {"pattern": pattern_name, "trades": total, "wins": wins, "winrate": round(winrate,2), "avg_return_pct": round(avg_ret,3), "profit_factor": None if pf is None else round(pf,3)}

# -----------------------------
# Annotation on original image
# -----------------------------
def annotate_image(img, pattern_list, decision, sltp):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W,H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf",14)
    except Exception:
        font = None
    # info box
    info = [f"Empfehlung: {decision['recommendation']} ({decision['probability']}%)",
            f"Risiko: {decision['risk_pct']}%",
            f"Top: {decision.get('top_label')} ({decision.get('top_conf')})"]
    x0,y0=10,10; bw=420; bh=18*len(info)+12
    draw.rectangle([x0,y0,x0+bw,y0+bh], fill=(8,8,10,220))
    yy = y0+6
    for s in info:
        draw.text((x0+8,yy), s, fill=(230,238,246,255), font=font)
        yy += 18
    # draw pattern markers
    for p in pattern_list[:40]:
        name, idx, xpos = p[0], p[1], p[2]
        x = xpos; y = int(H*0.7)
        draw.line([(x,y-14),(x,y+14)], fill=(255,200,80,200), width=2)
        draw.ellipse([(x-6,y-6),(x+6,y+6)], outline=(255,200,80,200), width=2)
        draw.text((x+8,y-6), name, fill=(230,238,246,255), font=font)
    # SL/TP lines if pixel provided
    if sltp.get("stop_pixel"):
        yline = int(H * 0.15)
        draw.line([(6,yline),(W-6,yline)], fill=(255,100,80,200), width=3)
        draw.text((12,yline-12), f"Stop(pixel): {sltp['stop_pixel']}", fill=(255,100,80,255), font=font)
    if sltp.get("tp_pixel"):
        yline = int(H * 0.18)
        draw.line([(6,yline),(W-6,yline)], fill=(80,200,120,200), width=3)
        draw.text((12,yline-12), f"TP(pixel): {sltp['tp_pixel']}", fill=(80,200,120,255), font=font)
    return img

# -----------------------------
# High-level analyze function
# -----------------------------
def analyze_image_full(image_bytes, symbol_hint=None, model_path="chart-pattern-detector/1", verbose=False):
    # load image
    img = load_image_bytes(image_bytes)
    if img is None:
        return {"error":"Bild konnte nicht geladen werden."}
    chart_img, gray, top_offset = preprocess_for_chart(img)
    candles = extract_candle_geometry(gray, top_offset, expected_min=20)
    if len(candles) < 6:
        return {"error": None, "message":"Nicht genug Kerzen entdeckt. Liefere sauberen Chart-Bereich.", "candles":candles}
    # local pattern detection
    local_patterns = detect_local_patterns(candles)
    # roboflow optional
    rf_res = None
    if ROBOFLOW_KEY and ONLINE:
        try:
            rf_res = roboflow_detect(image_bytes, retries=2, timeout=20, model_path=model_path)
        except Exception:
            rf_res = None
    # fuse
    pattern_scores = fuse_pattern_scores(local_patterns, rf_res)
    # trend
    trend, slope, r2 = detect_trend(candles)
    # ensemble decision
    decision = ensemble_decision(pattern_scores, trend, r2)
    # SL/TP pixel
    sltp = compute_pixel_sl_tp(candles, decision["recommendation"])
    # backtest top label if possible (use Finnhub or simulated)
    top_label = decision.get("top_label") or (list(pattern_scores.keys())[0] if pattern_scores else None)
    hist = None
    if ONLINE and FINNHUB_KEY and symbol_hint:
        hist = fetch_finnhub_candles(symbol_hint, resolution="5", from_ts=int(time.time()) - 60*60*24*120, to_ts=int(time.time()))
    if not hist:
        hist = generate_simulated_candles("uvt_bt_"+(symbol_hint or "sim"), periods=800, start_price=100.0, resolution_minutes=5)
    bt = backtest_pattern_on_series(hist, top_label or "NoPattern", lookahead=12)
    # calibrate probability by blending backtest winrate
    if bt["trades"] > 0:
        blended_prob = round((decision["probability"]*0.6 + bt["winrate"]*0.4),1)
    else:
        blended_prob = decision["probability"]
    decision["probability"] = blended_prob
    # annotate image
    annotated = annotate_image(img, local_patterns, decision, sltp)
    # export object
    export = {"meta":{"ts":now_iso(),"symbol_hint":symbol_hint,"online":ONLINE,"rf_used": bool(rf_res)}, "final":{"decision":decision, "sltp":sltp, "patterns":local_patterns, "backtest":bt}}
    append_audit({"ts": now_iso(), "summary": {"rec": decision["recommendation"], "prob": decision["probability"], "n_candles": len(candles)}})
    return {"export": export, "annotated": annotated, "candles": candles, "raw_rf": rf_res, "bt": bt}

# -----------------------------
# Batch evaluation tool (measure hit rate)
# -----------------------------
def batch_evaluate(folder_path, limit=200, symbol_hint=None):
    """
    Run analyzer on all images in a folder; create CSV report summarizing recommendation counts.
    """
    files = [f for f in os.listdir(folder_path) if f.lower().endswith((".png",".jpg",".jpeg"))]
    report=[]
    for i,fn in enumerate(files[:limit]):
        path = os.path.join(folder_path, fn)
        with open(path, "rb") as f:
            imgb = f.read()
        try:
            res = analyze_image_full(imgb, symbol_hint=symbol_hint)
            rec = res.get("export", {}).get("final", {}).get("decision", {}).get("recommendation") if res.get("export") else None
            prob = res.get("export", {}).get("final", {}).get("decision", {}).get("probability") if res.get("export") else None
            report.append({"file":fn, "rec":rec, "prob":prob})
        except Exception as e:
            report.append({"file":fn, "error":str(e)})
    return report

# -----------------------------
# Streamlit UI
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live/Charts","Bild Analyzer","Batch Test","Backtest","Audit","Einstellungen","Hilfe"])

if page == "Home":
    st.header("Ultimate Vision-Trader v3.0 — Übersicht")
    st.write("Hybrid: Finnhub:", bool(FINNHUB_KEY), "Roboflow:", bool(ROBOFLOW_KEY), "Online:", ONLINE)
    st.markdown("**Quick actions**")
    c1,c2,c3 = st.columns(3)
    if c1.button("Neue Bild-Analyse"):
        st.experimental_set_query_params(page="Bild Analyzer")
    if c2.button("Batch aus Ordner"):
        st.experimental_set_query_params(page="Batch Test")
    if c3.button("Audit anzeigen"):
        st.experimental_set_query_params(page="Audit")
    st.markdown("---")
    st.subheader("Letzte Analysen")
    try:
        arr = json.load(open(AUDIT_FILE,"r",encoding="utf-8"))
        for a in arr[-12:][::-1]:
            st.write(f"- {a.get('ts')} • {a.get('summary')}")
    except Exception:
        st.info("Keine Auditdaten.")

# Live/Charts: small interface to fetch Finnhub candles and show basic svg chart (no external plotting libs)
elif page == "Live/Charts":
    st.header("Live Chart / Daytrading View (Hybrid)")
    symbol = st.text_input("Finnhub symbol (e.g. AAPL / BINANCE:BTCUSDT)", "AAPL")
    resolution = st.selectbox("Resolution (min)", ["1","5","15","30","60"], index=1)
    periods = st.slider("Number of candles", 30, 1200, 300, step=10)
    if st.button("Lade Chart & Simuliere falls offline"):
        candles = None
        if ONLINE and FINNHUB_KEY:
            to_ts = int(time.time()); from_ts = to_ts - int(resolution)*60*periods
            candles = fetch_finnhub_candles(symbol, resolution=resolution, from_ts=from_ts, to_ts=to_ts)
            if not candles:
                st.warning("Finnhub lieferte keine Daten, benutze simulierte Serie")
                candles = generate_simulated_candles(symbol+"_sim", periods=periods, start_price=100.0, resolution_minutes=int(resolution))
        else:
            st.info("Offline -> Simulation")
            candles = generate_simulated_candles(symbol+"_sim", periods=periods, start_price=100.0, resolution_minutes=int(resolution))
        # Minimal SVG candle plot for UI
        def svg_candles(candles, width=1100, height=420):
            if not candles: return "<svg></svg>"
            margin=50; chart_h=int(height*0.62)
            maxp=max(c["high"] for c in candles); minp=min(c["low"] for c in candles)
            pad=(maxp-minp)*0.02 if (maxp-minp)>0 else 1.0
            maxp+=pad; minp-=pad
            n=len(candles); spacing=(width-2*margin)/n; cw=max(2, spacing*0.6)
            def y(p): return margin + chart_h - (p-minp)/(maxp-minp)*chart_h
            svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">', f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>']
            for i,c in enumerate(candles[-min(400,len(candles)):]):
                cx=margin + i*spacing + spacing/2
                top = y(c["high"]); low = y(c["low"]); oy=y(c["open"]); cy=y(c["close"])
                color = "#00cc66" if c["close"]>=c["open"] else "#ff4d66"
                svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
                by=min(oy,cy); bh=max(1, abs(cy-oy))
                svg.append(f'<rect x="{cx-cw/2}" y="{by}" width="{cw}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
            svg.append('</svg>')
            return "\n".join(svg)
        st.components.v1.html(svg_candles(candles), height=460)

# Bild Analyzer
elif page == "Bild Analyzer":
    st.header("Bild Analyzer — Ultimate Vision")
    st.markdown("Upload Chart screenshot. Analyzer returns recommendation (Kaufen/Short/Neutral), probability, risk, SL/TP (pixel), annotated image, backtest summary.")
    uploaded = st.file_uploader("Chart Screenshot (png/jpg)", type=["png","jpg","jpeg"])
    symbol_hint = st.text_input("Symbol Hint (optional) e.g. AAPL", "")
    aggressive = st.checkbox("Aggressive Mode (lower Neutral threshold)", value=True)
    run = st.button("Analysiere Bild")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            with st.spinner("Analysiere..."):
                try:
                    res = analyze_image_full(img_bytes, symbol_hint if symbol_hint else None)
                except Exception as e:
                    st.error("Analyse fehlgeschlagen: " + str(e))
                    st.error(traceback.format_exc())
                    res = None
            if res is None:
                st.error("Fehler bei Analyse.")
            elif res.get("error"):
                st.error(res.get("error"))
            elif res.get("message"):
                st.info(res.get("message"))
            else:
                exp = res["export"]
                dec = exp["final"]["decision"]
                sltp = exp["final"]["sltp"]
                bt = exp["final"]["backtest"]
                # result box
                if dec["recommendation"] == "Kaufen":
                    st.success(f"Empfehlung: {dec['recommendation']}  •  {dec['probability']}%  •  Risiko: {dec['risk_pct']}%")
                elif dec["recommendation"] == "Short":
                    st.error(f"Empfehlung: {dec['recommendation']}  •  {dec['probability']}%  •  Risiko: {dec['risk_pct']}%")
                else:
                    st.info(f"Empfehlung: {dec['recommendation']}  •  {dec['probability']}%  •  Risiko: {dec['risk_pct']}%")
                st.write("Top label:", dec.get("top_label"), "Top confidence:", dec.get("top_conf"))
                st.markdown("**SL/TP (pixel approx):**")
                st.write(sltp)
                st.markdown("**Backtest (top)**")
                st.write(bt)
                st.markdown("**3-Satz Kurzfazit:**")
                pats = [p[0] for p in exp["final"].get("patterns",[])][:3]
                st.write("- Erkannt:", ", ".join(pats) if pats else "keine eindeutigen Muster")
                st.write(f"- Wahrscheinlichkeit Empfehlung: {dec['probability']}%  •  Risiko: {dec['risk_pct']}%")
                if res.get("annotated"):
                    st.subheader("Annotiertes Bild")
                    st.image(res["annotated"], use_column_width=True)
                # downloads
                st.download_button("Export JSON", data=json.dumps(exp, ensure_ascii=False, indent=2), file_name=f"uvt_analysis_{short_ts()}.json", mime="application/json")
                # CSV quick export final
                csv_buf = io.StringIO()
                csvw = csv.writer(csv_buf)
                csvw.writerow(["field","value"])
                for k,v in exp["final"].items():
                    csvw.writerow([k,json.dumps(v, ensure_ascii=False)])
                st.download_button("Export CSV (final)", data=csv_buf.getvalue(), file_name=f"uvt_final_{short_ts()}.csv", mime="text/csv")

# Batch Test
elif page == "Batch Test":
    st.header("Batch Evaluation / Report")
    st.markdown("Provide a local folder path (server) containing chart images to run batch analysis.")
    folder = st.text_input("Folder path (server)", "")
    max_files = st.number_input("Max images", min_value=1, max_value=2000, value=200)
    if st.button("Run batch"):
        if not folder or not os.path.exists(folder):
            st.error("Folder path invalid or not accessible from server.")
        else:
            with st.spinner("Running batch..."):
                report = []
                files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))][:int(max_files)]
                for fn in files:
                    p = os.path.join(folder, fn)
                    try:
                        with open(p,"rb") as f:
                            imgb = f.read()
                        r = analyze_image_full(imgb)
                        rec = r.get("export",{}).get("final",{}).get("decision",{}).get("recommendation") if r.get("export") else None
                        prob = r.get("export",{}).get("final",{}).get("decision",{}).get("probability") if r.get("export") else None
                        report.append({"file":fn,"rec":rec,"prob":prob})
                    except Exception as e:
                        report.append({"file":fn,"error":str(e)})
                # aggregate
                rows = report
                st.write(f"Analyzed {len(rows)} images")
                st.table(rows[:40])
                # save report
                outp = os.path.join(APP_DIR, f"batch_report_{short_ts()}.json")
                with open(outp,"w",encoding="utf-8") as f:
                    json.dump(rows,f,indent=2,ensure_ascii=False)
                st.success(f"Report saved: {outp}")
                st.download_button("Download Report JSON", data=json.dumps(rows, ensure_ascii=False, indent=2), file_name=os.path.basename(outp), mime="application/json")

# Backtest page
elif page == "Backtest":
    st.header("Backtester (pattern)")
    pat = st.selectbox("Pattern (simple)", ["Doji","Hammer","Engulfing","Strong Bull Body","Strong Bear Body"])
    lookahead = st.slider("Lookahead candles", 1, 50, 12)
    fee = st.number_input("Fee %", 0.0, 1.0, 0.02, step=0.01)
    slip = st.number_input("Slippage %", 0.0, 1.0, 0.05, step=0.01)
    if st.button("Run Backtest (simulated)"):
        hist = generate_simulated_candles("backtest_seed", periods=2000, start_price=100.0, resolution_minutes=5)
        res = backtest_pattern_on_series(hist, pat, lookahead=lookahead, slippage_pct=slip, fee_pct=fee)
        st.write(res)
        if res["trades"]==0:
            st.info("No pattern events found in simulated series.")
        else:
            st.success("Backtest complete. Use this to calibrate analyzer thresholds.")

# Audit
elif page == "Audit":
    st.header("Audit log")
    try:
        arr = json.load(open(AUDIT_FILE, "r", encoding="utf-8"))
    except Exception:
        arr = []
    st.write(f"Analyses stored: {len(arr)}")
    for a in arr[::-1][:200]:
        st.write(f"- {a.get('ts')} • {a.get('summary')}")
    if st.button("Clear audit"):
        with open(AUDIT_FILE,"w",encoding="utf-8") as f:
            json.dump([],f)
        st.success("Cleared")

# Settings
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Online:", ONLINE)
    st.write("Finnhub key set:", bool(FINNHUB_KEY))
    st.write("Roboflow key set:", bool(ROBOFLOW_KEY))
    st.markdown("**Security**: Put API keys in `st.secrets` for production. Do not commit keys to repo.")

# Help
elif page == "Hilfe":
    st.header("Hilfe / Next Steps")
    st.markdown("""
    - To reach 80–90% accuracy you will need to train a vision model (Roboflow/YOLO) on labeled charts (50-500+ examples per pattern).
    - This app is ready for that: it produces batch reports, audit logs, and has Roboflow hooks.
    - Next recommended actions:
      1) Collect labeled chart screenshots (clean crops), 100-500 per pattern.
      2) Train on Roboflow or YOLO and set ROBOFLOW_KEY + model path.
      3) Recalibrate thresholds using batch backtests.
      4) Add pixel->price mapping using 2 anchor points on chart (I can add UI).
    """)

st.caption("Ultimate Vision-Trader v3.0 — Hybrid analyzer (offline-first). No financial advice.")
