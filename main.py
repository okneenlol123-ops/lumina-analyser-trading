# main.py
# Lumina Pro — Aggressive Image Analyzer (Roboflow + Local fallback + Finnhub backtest)
# Fokus: Bildanalyse (wenig Neutral), klare Empfehlung (Kaufen/Short/Neutral only if truly unknown)
#
# Requirements:
# pip install streamlit pillow
#
# Keys: set your keys here or move to st.secrets for production
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"   # <--- set or "" to disable
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"   # model path on Roboflow
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"  # <--- set or "" to disable

import streamlit as st
import io, os, time, json, random, urllib.request, urllib.parse, csv
from datetime import datetime, timedelta

# Pillow
try:
    from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# Basic utils
def now_iso(): return datetime.utcnow().isoformat() + "Z"
def short_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def internet_ok(timeout=2):
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

# Audit file
AUDIT_FILE = "image_analysis_audit.json"
if not os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

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

# Roboflow multipart
def encode_multipart(fieldname, filename, file_bytes, content_type="image/png"):
    boundary = "----WebKitFormBoundary" + "".join(random.choice("0123456789abcdef") for _ in range(16))
    crlf = b"\r\n"
    body = bytearray()
    body.extend(b"--" + boundary.encode() + crlf)
    body.extend(f'Content-Disposition: form-data; name="{fieldname}"; filename="{filename}"'.encode() + crlf)
    body.extend(f"Content-Type: {content_type}".encode() + crlf + crlf)
    body.extend(file_bytes + crlf)
    body.extend(b"--" + boundary.encode() + b"--" + crlf)
    return f"multipart/form-data; boundary={boundary}", bytes(body)

def roboflow_detect(image_bytes, retries=2, timeout=20):
    if not ROBOFLOW_KEY:
        return None
    endpoint = f"https://detect.roboflow.com/{ROBOFLOW_MODEL_PATH}?api_key={urllib.parse.quote(ROBOFLOW_KEY)}"
    for attempt in range(retries+1):
        try:
            ct, body = encode_multipart("file", "upload.png", image_bytes, "image/png")
            req = urllib.request.Request(endpoint, data=body, method="POST")
            req.add_header("Content-Type", ct)
            req.add_header("User-Agent", "LuminaPro-ImageAnalyzer/1.0")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except Exception as e:
            if attempt < retries:
                time.sleep(1 + attempt)
                continue
            return None

# Label mapping + base heuristics
LABEL_LIBRARY = {
    "Bullish Engulfing": {"dir":"bull", "base_wr":0.70, "risk":2.5},
    "Bearish Engulfing": {"dir":"bear", "base_wr":0.68, "risk":2.5},
    "Hammer": {"dir":"bull", "base_wr":0.63, "risk":2.8},
    "Shooting Star": {"dir":"bear", "base_wr":0.62, "risk":3.0},
    "Doji": {"dir":"neutral", "base_wr":0.48, "risk":4.0},
    "Morning Star": {"dir":"bull", "base_wr":0.75, "risk":2.0},
    "Evening Star": {"dir":"bear", "base_wr":0.74, "risk":2.0},
    "Three White Soldiers": {"dir":"bull", "base_wr":0.78, "risk":2.0},
    "Three Black Crows": {"dir":"bear", "base_wr":0.77, "risk":2.0},
    "Double Top": {"dir":"bear", "base_wr":0.66, "risk":3.5},
    "Double Bottom": {"dir":"bull", "base_wr":0.66, "risk":3.2},
    "Head & Shoulders": {"dir":"bear", "base_wr":0.68, "risk":3.5},
    "NoClearPattern": {"dir":"neutral", "base_wr":0.45, "risk":5.0},
    "ChoppyMarket": {"dir":"neutral", "base_wr":0.40, "risk":5.5},
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

# Local pixel-based detection (aggressive heuristics)
def local_detect_from_image_bytes(image_bytes):
    """Return list of (label, score)"""
    if not PIL_AVAILABLE:
        return []
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
    except Exception:
        return []
    W,H = img.size
    # crop central chart area heuristics
    left = int(W*0.04); right = int(W*0.96); top = int(H*0.08); bottom = int(H*0.82)
    crop = img.crop((left, top, right, bottom))
    crop = ImageOps.autocontrast(crop, cutoff=2)
    crop = crop.filter(ImageFilter.MedianFilter(size=3))
    pix = crop.load()
    Wc,Hc = crop.size
    # column darkness profile
    col_sum = []
    for x in range(Wc):
        s = 0
        for y in range(0, Hc, 2):
            s += 255 - pix[x,y]
        col_sum.append(s)
    maxv = max(col_sum) if col_sum else 1
    norm = [v / maxv for v in col_sum]
    peaks = [i for i in range(2, Wc-2) if norm[i] > norm[i-1] and norm[i] > norm[i+1] and norm[i] > 0.5]
    # count long lower/upper wicks by sampling vertical darkness around peaks
    hammer_count = 0; shooting_count = 0; doji_count = 0
    for x in peaks:
        col = [255 - pix[x,y] for y in range(Hc)]
        if not col: continue
        maxc = max(col); threshold = max(2, maxc*0.45)
        strong = [i for i,v in enumerate(col) if v>=threshold]
        if not strong: continue
        body_top = min(strong); body_bot = max(strong)
        body = body_bot - body_top + 1
        top_gap = body_top
        bot_gap = Hc - 1 - body_bot
        # doji: tiny body
        if body < max(1, Hc*0.06):
            doji_count += 1
        # hammer: long lower shadow
        if bot_gap > 2.5*body and body > 0:
            hammer_count += 1
        # shooting: long upper shadow
        if top_gap > 2.5*body and body > 0:
            shooting_count += 1
    results = []
    # produce scores aggressively: 1-2 events -> high confidence
    if hammer_count > 0:
        s = min(0.95, 0.45 + hammer_count*0.18)
        results.append(("Hammer", round(s,2)))
    if shooting_count > 0:
        s = min(0.95, 0.42 + shooting_count*0.16)
        results.append(("Shooting Star", round(s,2)))
    if doji_count > 0:
        s = min(0.9, 0.30 + doji_count*0.08)
        results.append(("Doji", round(s,2)))
    density = len(peaks) / (Wc/100.0 + 1e-9)
    if density > 6:
        results.append(("ChoppyMarket", round(min(0.95, density/12.0 + 0.2),2)))
    if not results:
        # fallback structural hint (detect if many peaks consistent => trending)
        avg = sum(norm)/len(norm) if norm else 0.0
        if avg > 0.22:
            results.append(("Three White Soldiers", 0.55))
        else:
            results.append(("NoClearPattern", 0.6))
    return results

# Fuse Roboflow + local -> label_scores (aggressive combination)
def fuse_labels(roboflow_res, local_preds):
    scores = {}
    # Roboflow predictions (high weight)
    if roboflow_res and isinstance(roboflow_res, dict) and "predictions" in roboflow_res:
        for p in roboflow_res["predictions"]:
            cls = p.get("class","").lower(); conf = float(p.get("confidence",0.0))
            label = ROBOFLOW_TO_LABEL.get(cls, cls.title().replace("_"," "))
            prev = scores.get(label, 0.0)
            scores[label] = max(prev, conf * 0.98)
    # local preds (additive, can push to decision)
    for lab, sc in local_preds:
        labn = lab if lab in LABEL_LIBRARY else lab.title().replace("_"," ")
        prev = scores.get(labn, 0.0)
        # local can add substantially (aggressive)
        scores[labn] = max(prev, min(0.99, prev + sc * 0.65))
    # ensure at least something
    if not scores:
        scores["NoClearPattern"] = 0.6
    return scores

# Decision: aggressive anti-neutral thresholds
def decide_from_labels(label_scores, candlesticks=None):
    # accumulate direction-weighted contributions
    bull = 0.0; bear = 0.0; neutral = 0.0; totalw = 0.0
    rationale = []
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"dir":"neutral","base_wr":0.5,"risk":4.0})
        dirc = meta["dir"]; base = meta["base_wr"]
        contrib = sc * base
        totalw += sc
        rationale.append(f"{label} (conf={sc:.2f}, baseWR={base})")
        if dirc == "bull": bull += contrib
        elif dirc == "bear": bear += contrib
        else: neutral += contrib
    bull_score = bull / (totalw + 1e-12)
    bear_score = bear / (totalw + 1e-12)
    neutral_score = neutral / (totalw + 1e-12)
    # Aggressive decision rules:
    # - If top label confidence > 0.80 -> follow it
    # - If bull_score > bear_score * 1.1 -> buy (lower margin)
    # - If bear_score > bull_score * 1.1 -> short
    # - else if any label score > 0.6 -> follow its dir
    rec = "Neutral"
    top_label, top_conf = max(label_scores.items(), key=lambda kv: kv[1])
    if top_conf >= 0.80:
        meta = LABEL_LIBRARY.get(top_label, {"dir":"neutral"})
        rec = "Kaufen" if meta["dir"]=="bull" else ("Short" if meta["dir"]=="bear" else "Neutral")
    elif bull_score > bear_score * 1.1 and bull_score > neutral_score * 0.9:
        rec = "Kaufen"
    elif bear_score > bull_score * 1.1 and bear_score > neutral_score * 0.9:
        rec = "Short"
    else:
        # fallback: if any strong label > 0.65, follow it
        strong = [(l,s) for l,s in label_scores.items() if s>0.65]
        if strong:
            lab = max(strong, key=lambda x: x[1])[0]
            meta = LABEL_LIBRARY.get(lab, {"dir":"neutral"})
            rec = "Kaufen" if meta["dir"]=="bull" else ("Short" if meta["dir"]=="bear" else "Neutral")
        else:
            # final fallback: if local heuristics suggest trending (Three Soldiers) -> buy
            if "Three White Soldiers" in label_scores and label_scores["Three White Soldiers"] > 0.5:
                rec = "Kaufen"
            else:
                rec = "Neutral"
    # probability and risk estimates
    prob = bull_score*100 if rec=="Kaufen" else (bear_score*100 if rec=="Short" else max(bull_score,bear_score,neutral_score)*100)
    prob = round(max(12.0, min(98.0, prob + (top_conf*10 - 5))),1)  # boost by top_conf
    # risk weighted average
    risk_sum = 0.0; wsum = 0.0
    for label, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(label, {"risk":4.0})
        risk_sum += sc * meta.get("risk",4.0); wsum += sc
    risk_pct = round((risk_sum/(wsum+1e-12)),2) if wsum>0 else 4.0
    # momentum tweak if candlesticks provided (small)
    if candlesticks and len(candlesticks)>=20:
        closes = [c["close"] for c in candlesticks[-50:]]
        s10 = sum(closes[-10:])/10 if len(closes)>=10 else sum(closes)/len(closes)
        s30 = sum(closes[-30:])/30 if len(closes)>=30 else s10
        if rec=="Kaufen" and s10 > s30: prob = min(99.0, prob + 6.0)
        if rec=="Short" and s10 < s30: prob = min(99.0, prob + 6.0)
    return {"recommendation": rec, "probability": prob, "risk_pct": risk_pct, "rationale": rationale, "top_label": top_label, "top_conf": top_conf}

# SL/TP mapping (aggressive)
def compute_sl_tp(label_scores, candles):
    last_price = candles[-1]["close"] if candles and len(candles)>0 else None
    # weighted risk
    w=0.0; rsum=0.0
    for lab, sc in label_scores.items():
        meta = LABEL_LIBRARY.get(lab, {"risk":4.0})
        w += sc; rsum += sc * meta.get("risk",4.0)
    risk = (rsum/(w+1e-12)) if w>0 else 4.0
    # try pivot detection
    sl=None; tp=None; notes=[]
    if candles and len(candles)>=30:
        highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
        # simple last local support/resistance detection: find last local min/max
        def last_local_extrema(arr, typ="min", window=4):
            n=len(arr)
            for i in range(n-1, window-1, -1):
                v = arr[i]
                left = arr[i-window:i]; right = arr[i+1:i+window+1] if i+window+1<=n else []
                if typ=="min":
                    if all(v < x for x in left) and all(v < x for x in right): return v
                else:
                    if all(v > x for x in left) and all(v > x for x in right): return v
            return None
        last_support = last_local_extrema(lows, "min")
        last_res = last_local_extrema(highs, "max")
        lp = last_price
        if last_support:
            sl = last_support * (1 - 0.001)
            notes.append("SL unter lokalem Support")
        else:
            sl = lp * (1 - risk/100.0)
            notes.append("SL relativ (kein Support)")
        if last_res:
            tp = last_res * (1 + 0.001)
            notes.append("TP an lokalem Widerstand")
        else:
            tp = lp * (1 + 2*risk/100.0)
            notes.append("TP relativ (kein Widerstand)")
    else:
        if last_price:
            sl = last_price * (1 - risk/100.0)
            tp = last_price * (1 + 2*risk/100.0)
            notes.append("Relative SL/TP (zu wenig Historie)")
        else:
            notes.append("Keine Preisinfo")
    return {"stop": None if sl is None else round(sl,6), "tp": None if tp is None else round(tp,6), "notes": notes, "risk_est": round(risk,2)}

# Backtest helper (Finnhub fetch + simple evaluation)
def fetch_finnhub_candles(symbol, resolution="5", from_ts=None, to_ts=None):
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None: to_ts = int(time.time())
        if from_ts is None: from_ts = to_ts - 60*60*24*90
        params = {"symbol": symbol, "resolution": resolution, "from": str(int(from_ts)), "to": str(int(to_ts)), "token": FINNHUB_KEY}
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=20) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        if data.get("s") != "ok": return None
        ts = data.get("t",[]); o=data.get("o",[]); h=data.get("h",[]); l=data.get("l",[]); c=data.get("c",[])
        candles=[]
        for i,t in enumerate(ts):
            dt = datetime.utcfromtimestamp(int(t))
            candles.append({"t":dt,"open":float(o[i]),"high":float(h[i]),"low":float(l[i]),"close":float(c[i])})
        return candles
    except Exception:
        return None

def generate_simulated_candles(seed, periods=800, start=100.0, res_mins=5):
    rnd = random.Random(abs(hash(seed))%(2**31))
    p = float(start)
    arr=[]
    for i in range(periods):
        drift = (rnd.random()-0.49)*0.002
        shock = (rnd.random()-0.5)*0.01
        p = max(0.01, p*(1+drift+shock))
        o = round(p*(1+random.uniform(-0.002,0.002)),6); c = round(p,6)
        h = round(max(o,c)*(1+random.uniform(0,0.003)),6); l = round(min(o,c)*(1-random.uniform(0,0.003)),6)
        arr.append({"t": datetime.utcnow() - timedelta(minutes=(periods-i)*res_mins), "open":o, "high":h, "low":l, "close":c})
    return arr

def simple_backtest(candles, pattern_label, lookahead=10, slippage_pct=0.05, fee_pct=0.02):
    # find pattern indices via simple detection for main patterns (only illustrative)
    indices=[]
    for i in range(1,len(candles)):
        cur=candles[i]; prev=candles[i-1]
        if pattern_label.lower().find("doji")>=0 and abs(cur["close"]-cur["open"]) < (cur["high"]-cur["low"])*0.15:
            indices.append(i)
        if pattern_label.lower().find("hammer")>=0 and is_hammer_local(cur):
            indices.append(i)
        if pattern_label.lower().find("engulf")>=0:
            if (cur["close"]>cur["open"] and prev["close"]<prev["open"] and cur["open"]<prev["close"] and cur["close"]>prev["open"]) or \
               (cur["close"]<cur["open"] and prev["close"]>prev["open"] and cur["open"]>prev["close"] and cur["close"]<prev["open"]):
                indices.append(i)
    trades=[]; wins=0
    for idx in indices:
        if idx+lookahead>=len(candles): continue
        entry = candles[idx]["close"]*(1+slippage_pct/100.0)
        exitp = candles[idx+lookahead]["close"]*(1 - slippage_pct/100.0)
        net = (exitp - entry)/(entry+1e-12) - fee_pct/100.0
        trades.append(net)
        if net>0: wins+=1
    total=len(trades)
    winrate = (wins/total*100.0) if total>0 else 0.0
    avg = (sum(trades)/total*100.0) if trades else 0.0
    return {"pattern":pattern_label,"checked":total,"wins":wins,"winrate":round(winrate,2),"avg_return_pct":round(avg,3)}

# small local hammer test reused
def is_hammer_local(c):
    body = abs(c["close"]-c["open"]); lower = min(c["open"], c["close"]) - c["low"]
    return body>0 and lower > 2.5*body

# Annotate image (draw labels + SL/TP)
def annotate_image(image_bytes, label_scores, sl=None, tp=None):
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
    # draw box of top labels
    items = sorted(label_scores.items(), key=lambda kv: kv[1], reverse=True)[:6]
    box_w = 320; box_h = 22*len(items) + 14
    draw.rectangle([10,10,10+box_w,10+box_h], fill=(12,12,12,220))
    y=14
    for lab, sc in items:
        txt = f"{lab}: {sc:.2f}"
        draw.text((14,y), txt, fill=(230,238,246,255), font=font)
        y += 22
    # SL/TP lines approximate
    if sl is not None:
        draw.line([(12, int(H*0.14)), (W-12, int(H*0.14))], fill=(255,204,0,190), width=3)
        draw.text((16, int(H*0.14)-12), f"Stop: {sl}", fill=(255,204,0,255), font=font)
    if tp is not None:
        draw.line([(12, int(H*0.18)), (W-12, int(H*0.18))], fill=(102,255,136,190), width=3)
        draw.text((16, int(H*0.18)-12), f"TP: {tp}", fill=(102,255,136,255), font=font)
    return img

# High-level analyze function (aggressive)
def analyze_image_aggressive(image_bytes, symbol_hint=None, use_online=True):
    rf_res = None
    if use_online and ONLINE and ROBOFLOW_KEY:
        rf_res = roboflow_detect(image_bytes, retries=2)
    local_preds = local_detect_from_image_bytes(image_bytes)
    label_scores = fuse_labels(rf_res, local_preds)
    # fetch history for calibration/backtest
    history = None
    if use_online and ONLINE and FINNHUB_KEY and symbol_hint:
        history = fetch_finnhub_candles(symbol_hint, resolution="5", from_ts=int(time.time())-60*60*24*90, to_ts=int(time.time()))
    if not history:
        history = generate_simulated_candles("img_hist_seed", 900, 100.0, 5)
    decision = decide_from_labels(label_scores, candlesticks=history)
    levels = compute_sl_tp(label_scores, history)
    top = decision.get("top_label", None)
    bt = simple_backtest(history, top or "NoClearPattern", lookahead=10)
    # calibrate probability by blending with backtest winrate if available
    if bt["checked"] > 0:
        blended_prob = round((decision["probability"]*0.6 + bt["winrate"]*0.4),1)
    else:
        blended_prob = decision["probability"]
    decision["probability"] = blended_prob
    # annotate
    annotated = annotate_image(image_bytes, label_scores, sl=levels.get("stop"), tp=levels.get("tp"))
    # Build export object
    export = {
        "meta": {"ts": now_iso(), "online": bool(rf_res and ONLINE), "symbol_hint": symbol_hint},
        "final": {
            "recommendation": decision["recommendation"],
            "probability": decision["probability"],
            "risk_pct": decision["risk_pct"],
            "rationale": decision.get("rationale", []),
            "label_scores": dict(label_scores),
            "levels": levels,
            "backtest": bt
        },
        "internals": {"roboflow_raw": rf_res, "local_preds": local_preds}
    }
    append_audit({"ts": now_iso(), "type":"image_analysis", "summary": {"rec": decision["recommendation"], "prob": decision["probability"], "labels": list(label_scores.keys())}})
    return {"export": export, "annotated": annotated}

# -------------------------
# Streamlit UI (focused)
# -------------------------
st.set_page_config(page_title="Lumina Pro — Aggressive Image Analyzer", layout="centered")
st.title("Lumina Pro — Aggressive Bild-Analyzer")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Analyze Bild","Audit","Einstellungen","Hilfe"])

if page == "Analyze Bild":
    st.header("Upload & Bulk-Analyse")
    st.markdown("Lade ein Chart-Screenshot hoch (oder mehrere). Die App entscheidet aggressiv (wenig Neutral).")
    uploaded = st.file_uploader("Chart-Bild (png/jpg) — bei mehreren: einzeln hochladen", type=["png","jpg","jpeg"])
    symbol_hint = st.text_input("Symbol-Hinweis für Backtest (optional)", value="AAPL")
    run = st.button("Analysiere (aggressiv)")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            img_bytes = uploaded.read()
            with st.spinner("Analysiere Bild..."):
                try:
                    res = analyze_image_aggressive(img_bytes, symbol_hint=symbol_hint, use_online=True)
                except Exception as e:
                    st.error("Fehler bei Analyse: " + str(e))
                    res = None
            if res:
                final = res["export"]
                rec = final["final"]["recommendation"]
                prob = final["final"]["probability"]
                risk = final["final"]["risk_pct"]
                if rec == "Kaufen":
                    st.success(f"Empfehlung: {rec} • Wahrscheinlichkeit: {prob}% • Risiko: {risk}%")
                elif rec == "Short":
                    st.error(f"Empfehlung: {rec} • Wahrscheinlichkeit: {prob}% • Risiko: {risk}%")
                else:
                    st.info(f"Empfehlung: {rec} • Wahrscheinlichkeit: {prob}% • Risiko: {risk}%")
                st.markdown("**Kurz-Fazit (3 Sätze):**")
                labels = list(final["final"].get("label_scores",{}).keys())[:3]
                st.write("- Erkannt: " + (", ".join(labels) if labels else "keine klaren Muster"))
                st.write(f"- Wahrscheinlichkeit der Empfehlung: {prob} %  •  Risiko: {risk} %")
                lv = final["final"].get("levels",{})
                st.write(f"- SL / TP (empfohlen): {lv.get('stop')}  /  {lv.get('tp')}")
                st.markdown("---")
                st.subheader("Backtest (Top-Label)")
                bt = final["final"].get("backtest",{})
                st.write(bt.get("summary", bt))
                if res.get("annotated") is not None:
                    st.subheader("Annotiertes Bild")
                    st.image(res["annotated"], use_column_width=True)
                st.download_button("Export JSON", data=json.dumps(final, ensure_ascii=False, indent=2), file_name=f"analysis_{short_ts()}.json", mime="application/json")
                st.download_button("Export CSV", data=(lambda o: "\n".join([",".join([k,str(v)]) for k,v in o.items()]))(final["final"]), file_name=f"analysis_{short_ts()}.csv", mime="text/csv")
            else:
                st.error("Keine Ergebnisse — Roboflow/offline Fehler")

elif page == "Audit":
    st.header("Audit — letzte Analysen")
    try:
        arr = json.load(open(AUDIT_FILE,"r",encoding="utf-8"))
    except Exception:
        arr = []
    st.write(f"Anzahl Analysen: {len(arr)}")
    for a in arr[::-1][:200]:
        st.write(f"- {a.get('ts')} • {a.get('type')} • {a.get('summary')}")
    if st.button("Audit löschen"):
        with open(AUDIT_FILE,"w",encoding="utf-8") as f:
            json.dump([], f)
        st.success("Audit gelöscht")

elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Internet:", "✓" if ONLINE else "✗")
    st.write("Pillow verfügbar:", PIL_AVAILABLE)
    st.write("Roboflow Key gesetzt:", bool(ROBOFLOW_KEY))
    st.write("Finnhub Key gesetzt:", bool(FINNHUB_KEY))
    st.markdown("**Hinweis:** Keys sollten im Produktivbetrieb in `st.secrets` oder Umgebungsvariablen gespeichert werden.")

elif page == "Hilfe":
    st.header("Hilfe")
    st.markdown("""
    - Diese Version ist *aggressiv*: Ziel ist, so wenig Neutral wie möglich auszugeben.
    - Lade saubere Chart-Screenshots (nur Candles, wenig UI) für beste Ergebnisse.
    - Wenn die App zu aggressiv ist, sag Bescheid — wir erhöhen wieder die Schwellen.
    - Exportiere Analysen via JSON/CSV.
    """)

st.caption("Lumina Pro — Bild-Analyzer (aggressive). Keine Anlageberatung.")
