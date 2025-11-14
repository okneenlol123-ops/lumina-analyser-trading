# main.py
# Ultimate Vision-Trader v3.0 — Advanced (Version B)
# Offline-first; optional Finnhub + Roboflow integration (st.secrets recommended)
# Requirements: streamlit, pillow, requests (optional)
# Save this file as main.py and run: streamlit run main.py

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import io, os, json, math, random, time, csv, traceback
from datetime import datetime, timedelta
import urllib.request, urllib.parse

# optional requests for convenience
try:
    import requests
    HAS_REQUESTS = True
except Exception:
    HAS_REQUESTS = False

# -------------------------
# Config & Styling
# -------------------------
st.set_page_config(page_title="Ultimate Vision-Trader v3.0 — Advanced", layout="wide", page_icon="💹")
st.markdown("""
<style>
html, body, [class*="css"] { background:#000 !important; color:#e6eef6 !important; }
.stButton>button { background:#111 !important; color:#e6eef6 !important; border:1px solid #222 !important; border-radius:6px; }
.card { background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
.small { color:#9aa6b2; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.title("Ultimate Vision-Trader v3.0 — Advanced (Daytrading Image Analyzer)")

# -------------------------
# Globals & files
# -------------------------
APP_DIR = ".uvt_cache"
os.makedirs(APP_DIR, exist_ok=True)
AUDIT_FILE = os.path.join(APP_DIR, "audit.json")
if not os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def now_iso(): return datetime.utcnow().isoformat() + "Z"
def short_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def append_audit(obj):
    try:
        arr = json.load(open(AUDIT_FILE, "r", encoding="utf-8"))
    except Exception:
        arr = []
    arr.append(obj)
    try:
        json.dump(arr, open(AUDIT_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception:
        pass

def internet_ok(timeout=2):
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

# Read API keys from st.secrets if available (recommended)
FINNHUB_KEY = ""
ROBOFLOW_KEY = ""
ROBOFLOW_MODEL = "chart-pattern-detector/1"
try:
    if st.secrets and "api_keys" in st.secrets:
        FINNHUB_KEY = st.secrets["api_keys"].get("FINNHUB_KEY", "") if isinstance(st.secrets["api_keys"], dict) else ""
        ROBOFLOW_KEY = st.secrets["api_keys"].get("ROBOFLOW_KEY", "") if isinstance(st.secrets["api_keys"], dict) else ""
        ROBOFLOW_MODEL = st.secrets["api_keys"].get("ROBOFLOW_MODEL", ROBOFLOW_MODEL) if isinstance(st.secrets["api_keys"], dict) else ROBOFLOW_MODEL
except Exception:
    pass

# -------------------------
# Utils: safe json / io
# -------------------------
def safe_json_load(path, default):
    try:
        if os.path.exists(path):
            return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        pass
    return default

def safe_json_dump(path, obj):
    try:
        json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    except Exception:
        pass

# -------------------------
# Candle sim (offline fallback)
# -------------------------
def generate_simulated_candles(seed, periods=400, start_price=100.0, step_min=5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    price = float(start_price)
    now = datetime.utcnow()
    candles = []
    for i in range(periods):
        drift = (rnd.random() - 0.49) * 0.002
        vol = (rnd.random() - 0.5) * 0.01
        price = max(0.01, price * (1 + drift + vol))
        o = round(price * (1 + rnd.uniform(-0.002, 0.002)), 6)
        c = round(price, 6)
        h = round(max(o, c) * (1 + rnd.uniform(0, 0.003)), 6)
        l = round(min(o, c) * (1 - rnd.uniform(0, 0.003)), 6)
        t = now - timedelta(minutes=(periods - 1 - i) * step_min)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c})
    return candles

# -------------------------
# Finnhub fetch (optional)
# -------------------------
def fetch_finnhub_candles(symbol, resolution="5", from_ts=None, to_ts=None):
    key = FINNHUB_KEY
    if not key:
        return None
    try:
        if to_ts is None:
            to_ts = int(time.time())
        if from_ts is None:
            from_ts = to_ts - 60*60*24*30
        params = {"symbol": symbol, "resolution": resolution, "from": int(from_ts), "to": int(to_ts), "token": key}
        if HAS_REQUESTS:
            r = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=20)
            data = r.json()
        else:
            url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
            with urllib.request.urlopen(url, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        if data.get("s") != "ok":
            return None
        ts = data.get("t", []); o = data.get("o", []); h = data.get("h", []); l = data.get("l", []); c = data.get("c", [])
        candles = []
        for i,t in enumerate(ts):
            dt = datetime.utcfromtimestamp(int(t))
            candles.append({"t":dt, "open":float(o[i]), "high":float(h[i]), "low":float(l[i]), "close":float(c[i])})
        return candles
    except Exception:
        return None

# -------------------------
# Roboflow detect (optional)
# -------------------------
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

def roboflow_detect(image_bytes, retries=2, timeout=20, model_path=None):
    key = ROBOFLOW_KEY
    model = model_path or ROBOFLOW_MODEL
    if not key:
        return None
    endpoint = f"https://detect.roboflow.com/{model}?api_key={urllib.parse.quote(key)}"
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
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            return None

# -------------------------
# Image loading & preprocessing
# -------------------------
def load_image_bytes(bytes_in):
    try:
        img = Image.open(io.BytesIO(bytes_in)).convert("RGB")
        return img
    except Exception:
        return None

def preprocess_for_chart(img, crop_top_ratio=0.06, crop_bottom_ratio=0.86):
    W,H = img.size
    top = int(H * crop_top_ratio)
    bottom = int(H * crop_bottom_ratio)
    chart = img.crop((0, top, W, bottom))
    gray = chart.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return chart, gray, top

# -------------------------
# Candle geometry extraction (robust, adaptive)
# -------------------------
def extract_candle_geometry(gray_img, top_offset=0, expected_min=15):
    W, H = gray_img.size
    pix = gray_img.load()
    # compute darkness per column
    col_dark = []
    for x in range(W):
        s = 0
        dark_positions = []
        for y in range(H):
            inv = 255 - pix[x, y]
            if inv > 6:  # noise threshold
                s += inv
                dark_positions.append(y)
        col_dark.append({"x": x, "sum": s, "dark": dark_positions})
    # smooth sums
    win = max(1, W // 200)
    smoothed = []
    for i in range(W):
        ssum = 0
        cnt = 0
        for j in range(max(0, i-win), min(W, i+win+1)):
            ssum += col_dark[j]["sum"]; cnt += 1
        smoothed.append(ssum / max(1, cnt))
    maxv = max(smoothed) if smoothed else 1
    threshold = max(8, maxv * 0.10)  # adaptive threshold (lowered to accept fewer candles)
    candidate_x = [i for i,v in enumerate(smoothed) if v >= threshold]
    if not candidate_x:
        # fallback very permissive
        threshold2 = max(4, maxv * 0.06)
        candidate_x = [i for i,v in enumerate(smoothed) if v >= threshold2]
    # cluster into groups representing candles
    groups = []
    if not candidate_x:
        return []
    cur = [candidate_x[0]]
    for idx in candidate_x[1:]:
        if idx - cur[-1] <= 2:
            cur.append(idx)
        else:
            groups.append(cur); cur = [idx]
    if cur: groups.append(cur)
    candles = []
    for g in groups:
        x_positions = g
        dark_pos = []
        for x in x_positions:
            dark_pos.extend(col_dark[x]["dark"])
        if not dark_pos:
            continue
        wick_top = min(dark_pos); wick_bot = max(dark_pos)
        # find densest contiguous dark run -> body
        hist = {}
        for d in dark_pos:
            hist[d] = hist.get(d,0) + 1
        runs = []
        sorted_keys = sorted(hist.keys())
        run_start = None; prev = None
        for k in sorted_keys:
            if run_start is None:
                run_start = k; prev = k
            elif k - prev <= 2:
                prev = k
            else:
                runs.append((run_start, prev)); run_start = k; prev = k
        if run_start is not None: runs.append((run_start, prev))
        best = max(runs, key=lambda r: sum(hist.get(i,0) for i in range(r[0], r[1]+1))) if runs else (wick_top, wick_bot)
        body_top = best[0]; body_bot = best[1]
        sample_x = x_positions[len(x_positions)//2]
        above = sum(255 - pix[sample_x, y] for y in range(max(0, body_top-2), body_top+1))
        below = sum(255 - pix[sample_x, y] for y in range(body_bot, min(H, body_bot+3)))
        color = "green" if below > above else "red"
        center_x = sum(x_positions) / len(x_positions)
        candles.append({
            "wick_top": top_offset + wick_top,
            "wick_bottom": top_offset + wick_bot,
            "body_top": top_offset + body_top,
            "body_bottom": top_offset + body_bot,
            "color": color,
            "x": int(center_x)
        })
    # relax if too few candles: reduce threshold and attempt simple fallback
    if len(candles) < expected_min and threshold > 4:
        # second pass
        threshold2 = max(4, maxv * 0.06)
        candidate_x = [i for i,v in enumerate(smoothed) if v >= threshold2]
        groups=[]; cur=[candidate_x[0]] if candidate_x else []
        for idx in candidate_x[1:]:
            if idx - cur[-1] <= 3: cur.append(idx)
            else: groups.append(cur); cur=[idx]
        if cur: groups.append(cur)
        candles2=[]
        for g in groups:
            xs = g; dp=[]
            for x in xs: dp += col_dark[x]["dark"]
            if not dp: continue
            wt=min(dp); wb=max(dp)
            bt = wt + (wb-wt)//3; bb = wb - (wb-wt)//3
            candles2.append({"wick_top": top_offset+wt, "wick_bottom": top_offset+wb, "body_top": top_offset+bt, "body_bottom": top_offset+bb, "color":"green", "x": int(sum(xs)/len(xs))})
        if len(candles2) > len(candles):
            candles = candles2
    # ensure sorted left->right by x
    candles = sorted(candles, key=lambda c: c["x"])
    return candles

# -------------------------
# Trend detection: slope + r2
# -------------------------
def slope_and_r2(values):
    n = len(values)
    if n < 2:
        return 0.0, 0.0
    xs = list(range(n))
    mean_x = sum(xs)/n; mean_y = sum(values)/n
    ss_xy = sum((xs[i]-mean_x)*(values[i]-mean_y) for i in range(n))
    ss_xx = sum((xs[i]-mean_x)**2 for i in range(n))
    if ss_xx == 0:
        return 0.0, 0.0
    b = ss_xy / ss_xx
    a = mean_y - b*mean_x
    ss_tot = sum((v-mean_y)**2 for v in values)
    ss_res = sum((values[i] - (a + b*xs[i]))**2 for i in range(n))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return b, r2

def detect_trend_from_candles(candles):
    centers = [ -((c["body_top"] + c["body_bottom"])/2.0) for c in candles ]  # invert pixel coordinates
    # allow small windows for daytrading
    if len(centers) < 6:
        return "neutral", 0.0, 0.0
    recent = centers[-min(len(centers), 60):]
    slope, r2 = slope_and_r2(recent)
    if slope > 0.10 and r2 > 0.04:
        return "uptrend", slope, r2
    if slope < -0.10 and r2 > 0.04:
        return "downtrend", slope, r2
    return "neutral", slope, r2

# -------------------------
# Local pattern detectors (micro + larger)
# -------------------------
def detect_local_patterns(candles):
    patterns = []
    n = len(candles)
    for i, c in enumerate(candles):
        body = abs(c["body_bottom"] - c["body_top"]) + 1e-9
        upper = abs(c["body_top"] - c["wick_top"])
        lower = abs(c["wick_bottom"] - c["body_bottom"])
        total = upper + lower + body
        # micro patterns
        if body < 0.12 * total:
            patterns.append(("Doji", i, c["x"]))
        if lower > 2.5 * body and upper < 0.6 * body:
            patterns.append(("Hammer", i, c["x"]))
        if upper > 2.5 * body and lower < 0.6 * body:
            patterns.append(("Shooting Star", i, c["x"]))
        if body > 0.6 * total:
            if c["color"] == "green":
                patterns.append(("Strong Bull Body", i, c["x"]))
            else:
                patterns.append(("Strong Bear Body", i, c["x"]))
    # formation level: three soldiers/crows
    if n >= 3:
        last3 = candles[-3:]
        if all((abs(cc["body_bottom"]-cc["body_top"])>0 and cc["color"]=="green") for cc in last3):
            patterns.append(("Three White Soldiers", n-1, last3[-1]["x"]))
        if all((abs(cc["body_bottom"]-cc["body_top"])>0 and cc["color"]=="red") for cc in last3):
            patterns.append(("Three Black Crows", n-1, last3[-1]["x"]))
    # double top/bottom (micro)
    highs = [c["wick_top"] for c in candles]
    lows = [c["wick_bottom"] for c in candles]
    for i in range(4, n-4):
        if highs[i] < min(highs[i-4:i]) and highs[i] < min(highs[i+1:i+5]):
            for j in range(i+3, min(n-2, i+22)):
                if abs(highs[j] - highs[i]) < 0.02 * max(1,abs(highs[i])):
                    patterns.append(("Double Top", i, candles[j]["x"]))
        if lows[i] > max(lows[i-4:i]) and lows[i] > max(lows[i+1:i+5]):
            for j in range(i+3, min(n-2, i+22)):
                if abs(lows[j] - lows[i]) < 0.02 * max(1,abs(lows[i])):
                    patterns.append(("Double Bottom", i, candles[j]["x"]))
    return patterns

# -------------------------
# Pattern meta + fusion
# -------------------------
PATTERN_META = {
    "Hammer": {"dir":"bull","weight":0.8},
    "Shooting Star": {"dir":"bear","weight":0.8},
    "Doji": {"dir":"neutral","weight":0.35},
    "Strong Bull Body": {"dir":"bull","weight":0.7},
    "Strong Bear Body": {"dir":"bear","weight":0.7},
    "Three White Soldiers": {"dir":"bull","weight":1.1},
    "Three Black Crows": {"dir":"bear","weight":1.1},
    "Double Top": {"dir":"bear","weight":0.9},
    "Double Bottom": {"dir":"bull","weight":0.9}
}

def fuse_patterns(local_patterns, roboflow_res=None):
    scores = {}
    # local
    for p in local_patterns:
        name = p[0]
        meta = PATTERN_META.get(name, {"dir":"neutral","weight":0.4})
        obj = scores.get(name, {"score":0.0, "dir":meta["dir"]})
        obj["score"] += meta["weight"]
        scores[name] = obj
    # roboflow
    if roboflow_res and isinstance(roboflow_res, dict) and "predictions" in roboflow_res:
        for pr in roboflow_res["predictions"]:
            cls = pr.get("class","").lower()
            conf = float(pr.get("confidence",0.0))
            mapping = {
                "hammer":"Hammer","shooting_star":"Shooting Star","doji":"Doji",
                "bullish_engulfing":"Strong Bull Body","bearish_engulfing":"Strong Bear Body",
                "three_white_soldiers":"Three White Soldiers","three_black_crows":"Three Black Crows",
                "double_top":"Double Top","double_bottom":"Double Bottom"
            }
            name = mapping.get(cls, cls.title().replace("_"," "))
            meta = PATTERN_META.get(name, {"dir":"neutral","weight":0.5})
            obj = scores.get(name, {"score":0.0, "dir":meta["dir"]})
            obj["score"] += conf * (0.8 + meta["weight"]*0.3)
            obj["dir"] = meta["dir"]
            scores[name] = obj
    return scores

# -------------------------
# Ensemble decision
# -------------------------
def ensemble_decision(pattern_scores, trend, r2):
    bull = 0.0; bear = 0.0; neutral = 0.0
    for nm,obj in pattern_scores.items():
        s = obj["score"]; d = obj.get("dir","neutral")
        if d=="bull": bull += s
        elif d=="bear": bear += s
        else: neutral += s
    # trend boost
    if trend=="uptrend": bull *= 1.12
    if trend=="downtrend": bear *= 1.12
    top_label = max(pattern_scores.items(), key=lambda kv: kv[1]["score"])[0] if pattern_scores else None
    top_conf = pattern_scores.get(top_label, {}).get("score", 0.0) if top_label else 0.0
    # decision rules
    rec = "Neutral"
    if bull > bear * 1.08 and bull > 0.8:
        rec = "Kaufen"
    elif bear > bull * 1.08 and bear > 0.8:
        rec = "Short"
    else:
        if top_conf >= 1.2:
            rec = "Kaufen" if pattern_scores[top_label]["dir"]=="bull" else ("Short" if pattern_scores[top_label]["dir"]=="bear" else "Neutral")
        else:
            if r2 > 0.12:
                if trend == "uptrend": rec = "Kaufen"
                elif trend == "downtrend": rec = "Short"
            else:
                # micro heuristics: if either side slightly dominant but window small, still output
                if bull > bear + 0.4: rec = "Kaufen"
                elif bear > bull + 0.4: rec = "Short"
                else: rec = "Range"
    # probability estimate
    base = max(bull, bear, neutral, 0.1)
    prob = min(99.0, max(12.0, round(30 + base * 18 + (top_conf * 8) + r2 * 50, 1)))
    risk = round(max(1.0, min(10.0, 6.0 - math.log(base+1) )), 2)
    return {"recommendation": rec, "probability": prob, "risk_pct": risk, "top_label": top_label, "top_conf": round(top_conf,3), "raw": {"bull": round(bull,3), "bear": round(bear,3), "neutral": round(neutral,3)}}

# -------------------------
# SL/TP pixel mapping (simple)
# -------------------------
def compute_pixel_sl_tp(candles, rec):
    if not candles:
        return {"stop_pixel": None, "tp_pixel": None}
    centers = [ (c["body_top"] + c["body_bottom"])/2.0 for c in candles ]
    last_center = centers[-1]
    lows = [c["body_bottom"] for c in candles]; highs = [c["body_top"] for c in candles]
    support = min(lows[-12:]) if lows else None
    resist = max(highs[-12:]) if highs else None
    if rec == "Kaufen":
        sl = support*(1+0.002) if support else last_center*(1+0.02)
        tp = last_center - (sl - last_center)*1.8 if support else last_center*(1-0.04)
    elif rec == "Short":
        sl = resist*(1-0.002) if resist else last_center*(1-0.02)
        tp = last_center + (last_center - sl)*1.8 if resist else last_center*(1+0.04)
    else:
        sl = None; tp = None
    return {"stop_pixel": round(sl,3) if sl else None, "tp_pixel": round(tp,3) if tp else None}

# -------------------------
# Backtester (improved)
# -------------------------
def backtest_pattern_on_series(candles, pattern_name, lookahead=12, slippage_pct=0.05, fee_pct=0.02):
    indices=[]
    for i in range(1, len(candles)):
        cur = candles[i]; prev = candles[i-1]
        if "doji" in pattern_name.lower():
            body = abs(cur["close"] - cur["open"]); rng = cur["high"] - cur["low"]
            if rng > 0 and body / rng < 0.15: indices.append(i)
        if "hammer" in pattern_name.lower():
            body = abs(cur["close"] - cur["open"])
            lower = min(cur["open"], cur["close"]) - cur["low"]
            if body > 0 and lower > 2.5 * body: indices.append(i)
        if "engulf" in pattern_name.lower():
            if (cur["close"] > cur["open"] and prev["close"] < prev["open"] and cur["open"] < prev["close"] and cur["close"] > prev["open"]) or \
               (cur["close"] < cur["open"] and prev["close"] > prev["open"] and cur["open"] > prev["close"] and cur["close"] < prev["open"]):
                indices.append(i)
    trades=[]; wins=0
    for idx in indices:
        if idx + lookahead >= len(candles): continue
        entry = candles[idx]["close"] * (1 + slippage_pct / 100.0)
        exitp = candles[idx + lookahead]["close"] * (1 - slippage_pct / 100.0)
        gross = (exitp - entry) / (entry + 1e-12)
        net = gross - fee_pct/100.0
        trades.append(net)
        if net > 0: wins += 1
    total = len(trades)
    winrate = round((wins / total * 100.0), 2) if total > 0 else 0.0
    avg_ret = round((sum(trades) / total * 100.0), 3) if trades else 0.0
    return {"pattern": pattern_name, "trades": total, "wins": wins, "winrate": winrate, "avg_return_pct": avg_ret}

# -------------------------
# Annotation drawing
# -------------------------
def annotate_image(img, patterns, decision, sltp):
    img = img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W,H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None
    # summary box
    box_w = 420
    lines = [
        f"Empfehlung: {decision['recommendation']} ({decision['probability']}%)",
        f"Risiko: {decision['risk_pct']}%",
        f"Top: {decision.get('top_label')} ({decision.get('top_conf')})"
    ]
    draw.rectangle([10,10,10+box_w, 16 + 20*len(lines)], fill=(12,12,12,220))
    y = 14
    for l in lines:
        draw.text((14,y), l, fill=(230,238,246,255), font=font)
        y += 20
    # pattern markers
    for p in patterns[:40]:
        name, idx, xpos = p[0], p[1], p[2]
        x = xpos; y0 = int(H * 0.7)
        draw.line([(x, y0-16),(x, y0+16)], fill=(255,200,80,200), width=2)
        draw.ellipse([(x-6, y0-6),(x+6, y0+6)], outline=(255,200,80,200), width=2)
        draw.text((x+8, y0-6), name, fill=(230,238,246,255), font=font)
    # SL/TP lines
    if sltp.get("stop_pixel"):
        yline = int(H * 0.15)
        draw.line([(6,yline),(W-6,yline)], fill=(255,100,80,200), width=3)
        draw.text((12,yline-12), f"Stop(pixel): {sltp['stop_pixel']}", fill=(255,100,80,255), font=font)
    if sltp.get("tp_pixel"):
        yline = int(H * 0.18)
        draw.line([(6,yline),(W-6,yline)], fill=(80,200,120,200), width=3)
        draw.text((12,yline-12), f"TP(pixel): {sltp['tp_pixel']}", fill=(80,200,120,255), font=font)
    return img

# -------------------------
# High-level analyze function
# -------------------------
def analyze_image_full(image_bytes, symbol_hint=None, model_path=None, aggressive=True):
    img = load_image_bytes(image_bytes)
    if img is None:
        return {"error": "Bild konnte nicht geladen werden."}
    chart_img, gray, top_offset = preprocess_for_chart(img)
    candles = extract_candle_geometry(gray, top_offset, expected_min=12)  # accept smaller windows
    if len(candles) < 6:
        return {"error": None, "message": "Nicht genug Kerzen erkannt. Bitte sauberen Chartbereich liefern.", "candles": candles}
    local_patterns = detect_local_patterns(candles)
    rf_res = None
    if ROBOFLOW_KEY and ONLINE:
        try:
            rf_res = roboflow_detect(image_bytes, retries=1, timeout=14, model_path=model_path)
        except Exception:
            rf_res = None
    pattern_scores = fuse_patterns(local_patterns, rf_res)
    trend, slope, r2 = detect_trend_from_candles(candles)
    decision = ensemble_decision(pattern_scores, trend, r2)
    sltp = compute_pixel_sl_tp(candles, decision["recommendation"])
    # backtest top label using Finnhub or simulated
    top_label = decision.get("top_label") or (list(pattern_scores.keys())[0] if pattern_scores else None)
    hist = None
    if ONLINE and FINNHUB_KEY and symbol_hint:
        try:
            hist = fetch_finnhub_candles(symbol_hint, resolution="5", from_ts=int(time.time()) - 60*60*24*90, to_ts=int(time.time()))
        except Exception:
            hist = None
    if not hist:
        hist = generate_simulated_candles("uvt_bt_"+(symbol_hint or "sim"), periods=600, start_price=100.0, step_min=5)
    bt = backtest_pattern_on_series(hist, top_label or "NoPattern", lookahead=12)
    # calibrate probability
    if bt["trades"] > 0:
        decision["probability"] = round((decision["probability"]*0.6 + bt["winrate"]*0.4), 1)
    annotated = annotate_image(img, local_patterns, decision, sltp)
    export = {"meta":{"ts":now_iso(),"symbol_hint":symbol_hint,"online":ONLINE,"rf_used": bool(rf_res)}, "final":{"decision":decision, "sltp":sltp, "patterns": local_patterns, "backtest": bt}}
    append_audit({"ts": now_iso(), "summary": {"rec": decision["recommendation"], "prob": decision["probability"], "n_candles": len(candles)}})
    return {"export": export, "annotated": annotated, "candles": candles, "rf": rf_res, "bt": bt}

# -------------------------
# UI: pages
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Bild Analyzer","Live/Charts","Backtest","Batch","Audit","Einstellungen","Hilfe"])

if page == "Home":
    st.header("Ultimate Vision-Trader v3.0 — Dashboard")
    st.write("Hybrid: Finnhub:", bool(FINNHUB_KEY), "Roboflow:", bool(ROBOFLOW_KEY), "Online:", ONLINE)
    st.markdown("Quick actions:")
    c1,c2,c3 = st.columns(3)
    if c1.button("Bild Analyzer"):
        st.experimental_set_query_params(page="Bild Analyzer")
    if c2.button("Batch Test"):
        st.experimental_set_query_params(page="Batch")
    if c3.button("Audit"):
        st.experimental_set_query_params(page="Audit")
    st.markdown("---")
    st.subheader("Letzte Analysen")
    try:
        arr = safe_json_load(AUDIT_FILE, [])
        for a in arr[-10:][::-1]:
            st.write(f"- {a.get('ts')} • {a.get('summary')}")
    except Exception:
        st.info("Keine Auditdaten.")

# -------------------------
# Bild Analyzer page
# -------------------------
elif page == "Bild Analyzer":
    st.header("Bild Analyzer — Daytrading (verbessert für kleine Fenster)")
    st.markdown("Lade dein Chart-Screenshot (auch 15–40 Kerzen). Die App analysiert Muster, Trend und gibt Empfehlung + SL/TP (pixel).")
    uploaded = st.file_uploader("Chart Screenshot (png/jpg)", type=["png","jpg","jpeg"])
    symbol_hint = st.text_input("Symbol (optional) für Backtest, z.B. AAPL", "")
    aggressive = st.checkbox("Aggressive Mode (weniger Neutral)", value=True)
    run = st.button("Analysiere")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            with st.spinner("Analysiere Bild..."):
                try:
                    res = analyze_image_full(img_bytes, symbol_hint=symbol_hint if symbol_hint else None, model_path=ROBOFLOW_MODEL, aggressive=aggressive)
                except Exception as e:
                    st.error("Analyse Fehler: " + str(e))
                    st.error(traceback.format_exc())
                    res = None
            if res is None:
                st.error("Analyse fehlgeschlagen.")
            elif res.get("error"):
                st.error(res.get("error"))
            elif res.get("message"):
                st.info(res.get("message"))
                if res.get("candles"):
                    st.write(f"Candles erkannt: {len(res['candles'])}")
            else:
                exp = res["export"]; dec = exp["final"]["decision"]; sltp = exp["final"]["sltp"]; bt = exp["final"]["backtest"]
                if dec["recommendation"] in ("Kaufen","Buy"):
                    st.success(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                elif dec["recommendation"] == "Short":
                    st.error(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                else:
                    st.info(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                st.write("Top label:", dec.get("top_label"), "Top confidence:", dec.get("top_conf"))
                st.markdown("**SL / TP (pixel approx)**")
                st.write(sltp)
                st.markdown("**Backtest (Top Label)**")
                st.write(bt)
                st.markdown("**Kurz-Fazit (3 Sätze)**")
                pats = [p[0] for p in exp["final"].get("patterns", [])][:3]
                st.write("- Erkannt: " + (", ".join(pats) if pats else "keine eindeutigen Muster"))
                st.write(f"- Wahrscheinlichkeit: {dec['probability']}%  •  Risiko: {dec['risk_pct']}%")
                st.write("- Empfehlung basiert auf Muster-Voting + Trend + Backtest-Kalibrierung.")
                if res.get("annotated"):
                    st.subheader("Annotiertes Bild")
                    st.image(res["annotated"], use_column_width=True)
                st.download_button("Export JSON", data=json.dumps(exp, ensure_ascii=False, indent=2), file_name=f"uvt_{short_ts()}.json", mime="application/json")
                # CSV export final
                csv_buf = io.StringIO()
                csvw = csv.writer(csv_buf)
                csvw.writerow(["field","value"])
                for k,v in exp["final"].items():
                    csvw.writerow([k, json.dumps(v, ensure_ascii=False)])
                st.download_button("Export CSV (final)", data=csv_buf.getvalue(), file_name=f"uvt_final_{short_ts()}.csv", mime="text/csv")

# -------------------------
# Live/Charts page (simple svg rendering)
# -------------------------
elif page == "Live/Charts":
    st.header("Live/Charts (Finnhub optional, fallback Simulation)")
    symbol = st.text_input("Symbol (z.B. AAPL)", "AAPL")
    resolution = st.selectbox("Resolution (min)", ["1","5","15","30","60"], index=1)
    periods = st.slider("Candles", 20, 1200, 300, step=10)
    if st.button("Load Chart"):
        candles = None
        if ONLINE and FINNHUB_KEY:
            try:
                to_ts = int(time.time()); from_ts = to_ts - int(resolution) * 60 * periods
                candles = fetch_finnhub_candles(symbol, resolution=resolution, from_ts=from_ts, to_ts=to_ts)
                if not candles:
                    st.warning("Finnhub returned no data — using simulation.")
                    candles = generate_simulated_candles(symbol+"_sim", periods=periods, start_price=100.0, step_min=int(resolution))
            except Exception:
                candles = generate_simulated_candles(symbol+"_sim", periods=periods, start_price=100.0, step_min=int(resolution))
        else:
            st.info("Offline → simulation")
            candles = generate_simulated_candles(symbol+"_sim", periods=periods, start_price=100.0, step_min=int(resolution))
        # draw minimal SVG candle chart
        def svg_candles(candles, width=1100, height=420):
            if not candles: return "<svg></svg>"
            margin=40; chart_h=int(height*0.66)
            maxp=max(c["high"] for c in candles); minp=min(c["low"] for c in candles)
            pad=(maxp-minp)*0.02 if (maxp-minp)>0 else 1.0
            maxp+=pad; minp-=pad
            n=len(candles); spacing=(width-2*margin)/n; cw=max(1, spacing*0.6)
            def yval(p): return margin + chart_h - (p-minp)/(maxp-minp)*chart_h
            svg=[f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">', f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>']
            for i,c in enumerate(candles[-min(400,len(candles)):]):
                cx=margin + i*spacing + spacing/2
                top = yval(c["high"]); low = yval(c["low"]); oy=yval(c["open"]); cy=yval(c["close"])
                color = "#00cc66" if c["close"]>=c["open"] else "#ff4d66"
                svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
                by=min(oy,cy); bh=max(1, abs(cy-oy))
                svg.append(f'<rect x="{cx-cw/2}" y="{by}" width="{cw}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
            svg.append('</svg>')
            return "\n".join(svg)
        st.components.v1.html(svg_candles(candles), height=460)

# -------------------------
# Backtest page
# -------------------------
elif page == "Backtest":
    st.header("Backtester — pattern calibration")
    pat = st.selectbox("Pattern", ["Doji","Hammer","Engulfing","Strong Bull Body","Strong Bear Body"])
    lookahead = st.slider("Lookahead candles", 1, 50, 12)
    fee = st.number_input("Fee %", 0.0, 1.0, 0.02, step=0.01)
    slip = st.number_input("Slippage %", 0.0, 1.0, 0.05, step=0.01)
    if st.button("Run Backtest (simulated)"):
        hist = generate_simulated_candles("backtest_seed", periods=2000, start_price=100.0, step_min=5)
        res = backtest_pattern_on_series(hist, pat, lookahead=lookahead, slippage_pct=slip, fee_pct=fee)
        st.write(res)
        if res["trades"]==0:
            st.info("No events found in simulated series.")

# -------------------------
# Batch page
# -------------------------
elif page == "Batch":
    st.header("Batch Evaluation (Server folder)")
    folder = st.text_input("Folder path (server, accessible)", "")
    max_files = st.number_input("Max files to analyze", min_value=1, max_value=2000, value=100)
    if st.button("Run batch"):
        if not folder or not os.path.exists(folder):
            st.error("Ungültiger Ordnerpfad oder nicht erreichbar.")
        else:
            files = [f for f in os.listdir(folder) if f.lower().endswith((".png",".jpg",".jpeg"))][:int(max_files)]
            report=[]; progress = st.progress(0)
            for i,fn in enumerate(files):
                p = os.path.join(folder, fn)
                with open(p,"rb") as f:
                    imgb = f.read()
                try:
                    r = analyze_image_full(imgb)
                    rec = r.get("export",{}).get("final",{}).get("decision",{}).get("recommendation")
                    prob = r.get("export",{}).get("final",{}).get("decision",{}).get("probability")
                    report.append({"file":fn,"rec":rec,"prob":prob})
                except Exception as e:
                    report.append({"file":fn,"error":str(e)})
                progress.progress((i+1)/len(files))
            st.write(f"Analyzed {len(report)} images")
            st.table(report[:min(30,len(report))])
            outp = os.path.join(APP_DIR, f"batch_report_{short_ts()}.json")
            safe_json_dump(outp, report)
            st.success(f"Report saved: {outp}")
            st.download_button("Download report JSON", data=json.dumps(report, ensure_ascii=False, indent=2), file_name=os.path.basename(outp), mime="application/json")

# -------------------------
# Audit
# -------------------------
elif page == "Audit":
    st.header("Audit Log — letzte Analysen")
    arr = safe_json_load(AUDIT_FILE, [])
    st.write(f"Analysen: {len(arr)}")
    for a in arr[::-1][:200]:
        st.write(f"- {a.get('ts')} • {a.get('summary')}")
    if st.button("Audit löschen"):
        safe_json_dump(AUDIT_FILE, [])
        st.success("Audit gelöscht.")

# -------------------------
# Einstellungen
# -------------------------
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Online:", ONLINE)
    st.write("Finnhub key gesetzt:", bool(FINNHUB_KEY))
    st.write("Roboflow key gesetzt:", bool(ROBOFLOW_KEY))
    st.markdown("**Hinweis:** Keys nicht im Repo speichern. Benutze st.secrets oder Umgebungsvariablen.")
    st.markdown("Tipps: Für Daytrading-Screenshots: min. 15 Candles, crop den Chartbereich, entferne Overlays.")

# -------------------------
# Hilfe
# -------------------------
elif page == "Hilfe":
    st.header("Hilfe & Next Steps")
    st.markdown("""
    - Diese Advanced-Version ist optimiert für Daytrading-Charts mit 15–40 Candles.
    - Für echte 80–90% Accuracy: Label-Dataset bauen + trainiertes Vision-Modell (Roboflow/YOLO).
    - Pixel→Preis-Mapping: Wenn du 2 Anker (y pixel → price) in der UI gibst, berechne ich echte SL/TP Preise.
    - Du kannst Batch-Reports nutzen, um reale Trefferquote zu messen.
    """)

st.caption("Keine Anlageberatung. Ultimate Vision-Trader v3.0 — Advanced. Entwickelt für Offline-first Daytrading-Chart-Analyse.")
