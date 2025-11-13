# main.py
# Lumina Pro — Advanced Aggressive Image Analyzer (offline-capable)
# Uses Pillow + pure Python (no OpenCV / no numpy / no sklearn)
# Requirements: pip install streamlit pillow

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import io, os, json, math, statistics, time
from datetime import datetime

# -------------------------
# Config / UI styling
# -------------------------
st.set_page_config(page_title="Lumina Pro — Image Analyzer (Aggressive)", layout="centered")
st.markdown("""
<style>
html, body, [class*="css"] { background:#000 !important; color:#e6eef6 !important; }
.stButton>button { background:#111 !important; color:#e6eef6 !important; border:1px solid #222 !important; }
.card { background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
.small { color:#9aa6b2; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.title("Lumina Pro — Aggressive Bild-Analyzer (Candle Geometry + Regression + Voting)")

# -------------------------
# Audit / persistence
# -------------------------
AUDIT_FILE = "image_analyzer_audit.json"
if not os.path.exists(AUDIT_FILE):
    with open(AUDIT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def append_audit(entry):
    try:
        a = []
        if os.path.exists(AUDIT_FILE):
            with open(AUDIT_FILE, "r", encoding="utf-8") as f:
                a = json.load(f)
        a.append(entry)
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump(a, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def now_iso(): return datetime.utcnow().isoformat() + "Z"
def short_ts(): return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

# -------------------------
# Helpers: image preprocessing
# -------------------------
def load_image_bytes(bytes_in):
    try:
        img = Image.open(io.BytesIO(bytes_in)).convert("RGB")
        return img
    except Exception:
        return None

def preprocess_image_for_chart(img: Image.Image):
    """
    - Convert to grayscale
    - Autocontrast
    - Median filter to reduce noise
    - Return processed grayscale image
    """
    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    return gray

# -------------------------
# Candle geometry extraction
# -------------------------
def extract_candles_from_grayscale_image(gray_img, expected_min_candles=8):
    """
    Heuristics:
    - Crop central area of image (remove UI/top/bottom)
    - Scan columns (step) and find dark pixel clusters indicating wicks/bodies
    - For each detected cluster produce wick_top, wick_bottom, body_top, body_bottom, color_guess
    Returns list of candle dicts (left-to-right)
    """

    W, H = gray_img.size
    # crop central area heuristics (tweakable)
    left = int(W * 0.03)
    right = int(W * 0.97)
    top = int(H * 0.08)
    bottom = int(H * 0.82)
    chart = gray_img.crop((left, top, right, bottom))
    Wc, Hc = chart.size
    pixels = chart.load()

    # column step roughly: 1 or more depending on width
    # choose small step to get many columns, but cluster later
    step = max(1, Wc // 300)  # aim ~300 columns max
    col_profiles = []
    for x in range(0, Wc, step):
        # compute darkness metric per column (sum of inverted pixel)
        s = 0
        dark_positions = []
        for y in range(Hc):
            val = pixels[x, y]  # 0 black .. 255 white
            inv = 255 - val
            if inv > 8:  # ignore tiny noise (threshold)
                s += inv
                dark_positions.append(y)
        col_profiles.append({"x": x, "sum": s, "dark_pos": dark_positions})

    # find columns that likely contain candles by thresholding sum
    if not col_profiles:
        return []

    max_sum = max(c["sum"] for c in col_profiles) or 1
    # threshold quite permissive (aggressive)
    threshold = max(10, max_sum * 0.12)
    candidate_cols = [c for c in col_profiles if c["sum"] >= threshold]

    # cluster adjacent columns into candle groups
    candles = []
    i = 0
    while i < len(candidate_cols):
        group = [candidate_cols[i]]
        j = i + 1
        while j < len(candidate_cols) and candidate_cols[j]["x"] - candidate_cols[j-1]["x"] <= step * 2:
            group.append(candidate_cols[j]); j += 1
        # aggregate group into single candle
        all_dark = []
        for g in group:
            all_dark.extend(g["dark_pos"])
        if not all_dark:
            i = j; continue
        wick_top = min(all_dark)
        wick_bottom = max(all_dark)
        # body detection: find densest contiguous dark region inside (heuristic)
        # build histogram of dark positions
        hist = {}
        for p in all_dark:
            hist[p] = hist.get(p, 0) + 1
        # find longest contiguous run with high counts → body region
        runs = []
        sorted_keys = sorted(hist.keys())
        run_start = None; run_prev = None
        for k in sorted_keys:
            if run_start is None:
                run_start = k; run_prev = k
            elif k - run_prev <= 2:
                run_prev = k
            else:
                runs.append((run_start, run_prev)); run_start = k; run_prev = k
        if run_start is not None:
            runs.append((run_start, run_prev))
        # choose run with max total count (likely body)
        best_run = None; best_score = -1
        for (a,b) in runs:
            score = sum(hist.get(y,0) for y in range(a,b+1))
            if score > best_score:
                best_score = score; best_run = (a,b)
        if best_run:
            body_top = best_run[0]; body_bottom = best_run[1]
        else:
            # fallback: narrow body in middle
            body_mid = (wick_top + wick_bottom) // 2
            body_top = body_mid; body_bottom = body_mid
        # determine color guess: compare average darkness above vs below body to infer filled/empty
        above_dark = sum(255 - pixels[group[0]["x"], y] for y in range(max(0, body_top-2), body_top+1)) if body_top>0 else 0
        below_dark = sum(255 - pixels[group[0]["x"], y] for y in range(body_bottom, min(Hc, body_bottom+3)))
        color = "green" if below_dark > above_dark else "red"
        # convert positions back to original image coordinates (global)
        global_wick_top = top + wick_top
        global_wick_bottom = top + wick_bottom
        candle = {
            "wick_top": global_wick_top,
            "wick_bottom": global_wick_bottom,
            "body_top": top + body_top,
            "body_bottom": top + body_bottom,
            "color": color,
            "group_x": left + group[len(group)//2]["x"]
        }
        candles.append(candle)
        i = j

    # if too few candles detected, try relaxing threshold (second pass)
    if len(candles) < expected_min_candles and threshold > 5:
        # reduce threshold and rerun a simple fallback pass
        threshold2 = max(5, max_sum * 0.06)
        candidate_cols = [c for c in col_profiles if c["sum"] >= threshold2]
        # simple one-col-per-candle mapping
        candles2 = []
        for c in candidate_cols:
            if not c["dark_pos"]: continue
            wick_top = min(c["dark_pos"]); wick_bottom = max(c["dark_pos"])
            body_top = wick_top + (wick_bottom - wick_top)//3
            body_bottom = wick_bottom - (wick_bottom - wick_top)//3
            color = "green" if True else "red"
            candles2.append({
                "wick_top": top + wick_top,
                "wick_bottom": top + wick_bottom,
                "body_top": top + body_top,
                "body_bottom": top + body_bottom,
                "color": color,
                "group_x": left + c["x"]
            })
        # prefer richer set if larger
        if len(candles2) > len(candles):
            candles = candles2

    # final guard: sort by group_x (left->right)
    candles = sorted(candles, key=lambda c: c["group_x"])
    return candles

# -------------------------
# Linear regression (simple) for trend detection
# -------------------------
def linear_regression_slope(y_values):
    """
    Compute slope of y = a + b*x with x = 0..n-1
    returns slope (b). We use simple OLS formula.
    """
    n = len(y_values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = (n - 1) / 2.0
    mean_y = sum(y_values) / n
    num = sum((xs[i] - mean_x) * (y_values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    slope = num / den
    return slope

def detect_trend_from_candles(candles):
    """
    Use center-of-body as proxy "close" for geometry-based regression.
    Return: ("uptrend"/"downtrend"/"neutral", slope)
    """
    if len(candles) < 8:
        return "neutral", 0.0
    closes = []
    for c in candles:
        body_center = (c["body_top"] + c["body_bottom"]) / 2.0
        # invert Y because pixel coordinates increase downward
        closes.append(-body_center)
    # take last N (max 50)
    closes = closes[-50:]
    slope = linear_regression_slope(closes)
    # define slope thresholds heuristically (pixel-based)
    # normalized by image height approx: but we keep raw slope and interpret relatively
    if slope > 0.25:
        return "uptrend", slope
    elif slope < -0.25:
        return "downtrend", slope
    else:
        return "neutral", slope

# -------------------------
# Multi-pattern voting system
# -------------------------
def pattern_votes_from_candles(candles):
    """
    Count weighted votes for bullish and bearish across recent candles.
    Returns bull_score, bear_score and list of detected patterns.
    """
    bull = 0.0
    bear = 0.0
    patterns = []

    # inspect recent N candles
    window = min(len(candles), 40)
    for c in candles[-window:]:
        body = abs(c["body_bottom"] - c["body_top"]) + 1e-9
        wick_top_len = abs(c["body_top"] - c["wick_top"])
        wick_bot_len = abs(c["wick_bottom"] - c["body_bottom"])
        total_wick = wick_top_len + wick_bot_len + 1e-9

        # Hammer: long lower wick, small body
        if wick_bot_len > 2.5 * body and wick_top_len < 0.6 * body:
            bull += 1.2
            patterns.append(("Hammer", c["group_x"]))
        # Shooting star: long upper wick
        if wick_top_len > 2.5 * body and wick_bot_len < 0.6 * body:
            bear += 1.2
            patterns.append(("Shooting Star", c["group_x"]))
        # Doji: tiny body relative to wicks
        if body < 0.12 * total_wick:
            bull += 0.15; bear += 0.15
            patterns.append(("Doji", c["group_x"]))
        # Strong body (engulfing-like): body large relative to wick
        if body > 0.6 * total_wick:
            if c["color"] == "green":
                bull += 0.8; patterns.append(("Strong Bull Body", c["group_x"]))
            else:
                bear += 0.8; patterns.append(("Strong Bear Body", c["group_x"]))

    # additional formation heuristics: three white soldiers / three crows
    if len(candles) >= 3:
        last3 = candles[-3:]
        if all((abs(cc["body_bottom"] - cc["body_top"]) > 0 and cc["color"] == "green") for cc in last3):
            bull += 1.5
            patterns.append(("Three White Soldiers", last3[-1]["group_x"]))
        if all((abs(cc["body_bottom"] - cc["body_top"]) > 0 and cc["color"] == "red") for cc in last3):
            bear += 1.5
            patterns.append(("Three Black Crows", last3[-1]["group_x"]))

    return bull, bear, patterns

# -------------------------
# Decision engine (aggressive)
# -------------------------
def decide_from_votes(bull, bear, trend, patterns, top_confidence_est=0.0):
    """
    Aggressive thresholds:
    - If bull > bear * 1.1 and bull > 1.0 -> BUY
    - If bear > bull * 1.1 and bear > 1.0 -> SHORT
    - If any strong pattern (Three Soldiers, Hammer with trend) -> follow
    - else neutral only if both scores small
    """
    rec = "Neutral"
    # apply trend bias
    if trend == "uptrend":
        bull *= 1.12
    if trend == "downtrend":
        bear *= 1.12

    if bull > bear * 1.1 and bull > 0.9:
        rec = "Kaufen"
    elif bear > bull * 1.1 and bear > 0.9:
        rec = "Short"
    else:
        # check powerful single patterns
        strong_names = [p[0] for p in patterns]
        if any(x in strong_names for x in ("Three White Soldiers","Morning Star")) and bull > 0.5:
            rec = "Kaufen"
        elif any(x in strong_names for x in ("Three Black Crows","Evening Star")) and bear > 0.5:
            rec = "Short"
        else:
            # if either vote is moderate and top_confidence_est is decent -> follow
            if bull > 0.6 and top_confidence_est > 0.6:
                rec = "Kaufen"
            elif bear > 0.6 and top_confidence_est > 0.6:
                rec = "Short"
            else:
                rec = "Neutral"

    # compute probability estimate roughly
    base = max(bull, bear, 0.1)
    prob = min(99.0, max(12.0, round(30.0 + base * 25.0 + top_confidence_est * 20.0, 1)))
    # risk estimate rough
    risk = round(3.0 + (1.0 / (1.0 + base)) * 6.0, 2)
    return {"recommendation": rec, "probability": prob, "risk_pct": risk}

# -------------------------
# SL/TP mapping (simple)
# -------------------------
def compute_sl_tp_from_candles(candles, recommendation):
    """
    Very simple SL/TP:
    - find last local support / resistance from wick extremes
    - if none, use relative percentages based on risk
    """
    if not candles:
        return {"stop": None, "tp": None, "notes": ["keine Kerzen"]}

    last = candles[-1]
    last_center = (last["body_top"] + last["body_bottom"]) / 2.0
    # build lists of recent wick tops and bottoms
    highs = [c["wick_top"] for c in candles]  # note: pixel coordinates -> inverted later; we use differences
    lows = [c["wick_bottom"] for c in candles]
    # simple detection: last local min below last_center
    supports = [l for l in lows if l > last_center]  # pixel coords: larger means lower visually -> support below current level
    resistances = [h for h in highs if h < last_center]
    # convert to price-like numbers by inverting pixel coords (higher on screen -> lower pixel)
    # here we'll compute relative percentages using distances
    stop=None; tp=None; notes=[]
    if recommendation == "Kaufen":
        if supports:
            stop = supports[-1]  # closest support (pixel)
            notes.append("Stop unter lokalem Support (pixel-basiert)")
            tp = last_center - (stop - last_center) * 1.8  # mirror
        else:
            # relative percent method
            stop = last_center + abs(last_center)*0.02
            tp = last_center - abs(last_center)*0.04
            notes.append("Relative SL/TP (kein Support)")
    elif recommendation == "Short":
        if resistances:
            stop = resistances[-1]
            notes.append("Stop über lokalem Widerstand (pixel-basiert)")
            tp = last_center + (last_center - stop) * 1.8
        else:
            stop = last_center - abs(last_center)*0.02
            tp = last_center + abs(last_center)*0.04
            notes.append("Relative SL/TP (kein Widerstand)")
    else:
        notes.append("Keine SL/TP (Neutral)")
    # return pixel-based numbers — user should map to price context; for UI we display only 'relative' explanation
    return {"stop": None if stop is None else round(stop,2), "tp": None if tp is None else round(tp,2), "notes": notes}

# -------------------------
# Annotate image (draw markers)
# -------------------------
def annotate_image_with_results(img_original, candles, patterns, decision, sltp):
    """
    Draw bounding markers and label boxes on a copy of the original image.
    """
    img = img_original.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    W,H = img.size
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        font = None

    # draw small top-left info box
    box_w = 360
    lines = [
        f"Empfehlung: {decision['recommendation']}  ({decision['probability']}%)",
        f"Risiko: {decision['risk_pct']}%",
        f"SL: {sltp.get('stop')}  TP: {sltp.get('tp')}"
    ]
    draw.rectangle([8,8,8+box_w, 14 + 20*len(lines)], fill=(10,10,12,220))
    y = 12
    for l in lines:
        draw.text((12,y), l, fill=(230,238,246,255), font=font)
        y += 20

    # draw markers for patterns (x positions)
    for p in patterns:
        name, xpix = p[0], p[1]
        # draw small arrow/marker at approximate x position
        x = xpix; y0 = int(H*0.68)
        draw.line([(x, y0-18), (x, y0+18)], fill=(255,200,80,200), width=2)
        draw.ellipse([(x-6, y0-6), (x+6, y0+6)], outline=(255,200,80,200), width=2)
        # label
        draw.text((x+8, y0-6), name, fill=(230,238,246,255), font=font)
    return img

# -------------------------
# Main analyze pipeline (high-level)
# -------------------------
def analyze_image_bytes(bytes_img):
    img = load_image_bytes(bytes_img)
    if img is None:
        return {"error": "Bild konnte nicht geladen werden."}
    gray = preprocess_image_for_chart(img)
    candles = extract_candles_from_grayscale_image(gray)
    # if too few candles, return neutral informing user to upload cleaner image
    if len(candles) < 6:
        return {"error": None, "message": "Nicht genug Kerzen gefunden. Bitte saubere Chartaufnahme (mehr Kerzen, weniger UI).", "candles": candles}

    trend, slope = detect_trend_from_candles(candles)
    bull, bear, patterns = pattern_votes_from_candles(candles)
    # estimate top_conf (we use highest pattern weight from simple heuristics)
    top_conf = 0.0
    # derive top_conf from strongest local pattern signal heuristic:
    if patterns:
        top_conf = 0.7
    # decide
    decision = decide_from_votes(bull, bear, trend, patterns, top_confidence_est=top_conf)
    sltp = compute_sl_tp_from_candles(candles, decision["recommendation"])
    annotated = annotate_image_with_results(img, candles, patterns, decision, sltp)
    export = {
        "meta": {"ts": now_iso(), "candles_detected": len(candles), "trend": trend, "slope": round(slope,4)},
        "final": {
            "recommendation": decision["recommendation"],
            "probability": decision["probability"],
            "risk_pct": decision["risk_pct"],
            "patterns": [p[0] for p in patterns[:8]],
            "sltp": sltp
        }
    }
    # audit
    append_audit({"ts": now_iso(), "summary": {"rec": decision["recommendation"], "prob": decision["probability"], "n_candles": len(candles)}})
    return {"export": export, "annotated": annotated, "candles": candles, "decision": decision, "sltp": sltp}

# -------------------------
# Streamlit UI
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Analyze Bild","Audit","Einstellungen","Hilfe"])

if page == "Analyze Bild":
    st.header("Bild-Analyzer — Neue (aggressive) Logik")
    st.markdown("Lade ein Chart-Screenshot hoch (nur Kerzen-Bereich, wenig UI). Die App erkennt Kerzen geometrisch, bestimmt Trend per Regression und nutzt Multi-Pattern-Voting für eine klare Empfehlung.")
    uploaded = st.file_uploader("Bild hochladen (png/jpg)", type=["png","jpg","jpeg"])
    run = st.button("Analysiere Bild")
    if uploaded:
        st.image(uploaded, use_column_width=True)
        if run:
            bytes_img = uploaded.read()
            with st.spinner("Analysiere Bild (Geometrie, Trend, Voting)..."):
                try:
                    res = analyze_image_bytes(bytes_img)
                except Exception as e:
                    st.error("Analyse fehlgeschlagen: " + str(e))
                    res = None
            if res is None:
                st.error("Unbekannter Fehler.")
            elif res.get("error"):
                st.error(res["error"])
            elif res.get("message"):
                st.info(res["message"])
                if res.get("candles"):
                    st.write(f"Candles erkannt: {len(res['candles'])}")
            else:
                export = res["export"]
                dec = res["decision"]
                sltp = res["sltp"]
                # show short result header
                if dec["recommendation"] == "Kaufen":
                    st.success(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                elif dec["recommendation"] == "Short":
                    st.error(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                else:
                    st.info(f"Empfehlung: {dec['recommendation']} • {dec['probability']}% • Risiko: {dec['risk_pct']}%")
                st.markdown("**Kurz-Fazit (3 Sätze)**")
                pats = export["final"].get("patterns", [])[:3]
                s1 = "Erkannte Muster: " + (", ".join(pats) if pats else "keine eindeutigen Muster")
                s2 = f"Geschätzte Trefferwahrscheinlichkeit: {dec['probability']}% • Risiko: {dec['risk_pct']}%"
                s3 = f"Empfohlene SL/TP (relativ / pixel-basiert): {sltp.get('stop')} / {sltp.get('tp')}"
                st.write("- " + s1); st.write("- " + s2); st.write("- " + s3)
                st.markdown("---")
                st.subheader("Annotiertes Bild")
                if res.get("annotated"):
                    st.image(res["annotated"], use_column_width=True)
                st.markdown("---")
                st.subheader("Export & Download")
                st.download_button("Export JSON", data=json.dumps(export, ensure_ascii=False, indent=2), file_name=f"analysis_{short_ts()}.json", mime="application/json")
                # CSV flatten quick
                csv_buf = io.StringIO()
                csv_buf.write("key,value\n")
                for k,v in export["final"].items():
                    if isinstance(v, (str,int,float)):
                        csv_buf.write(f"{k},{v}\n")
                    else:
                        csv_buf.write(f"{k},\"{json.dumps(v, ensure_ascii=False)}\"\n")
                st.download_button("Export CSV", data=csv_buf.getvalue(), file_name=f"analysis_{short_ts()}.csv", mime="text/csv")

elif page == "Audit":
    st.header("Audit — letzte Analysen")
    try:
        with open(AUDIT_FILE, "r", encoding="utf-8") as f:
            arr = json.load(f)
    except Exception:
        arr = []
    st.write(f"Analysen gespeichert: {len(arr)}")
    for a in arr[::-1][:200]:
        st.write(f"- {a.get('ts')} • {a.get('summary')}")
    if st.button("Audit löschen"):
        with open(AUDIT_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
        st.success("Audit gelöscht.")

elif page == "Einstellungen":
    st.header("Einstellungen")
    st.write("Offline-Analyzer (keine externen Keys benötigt)")
    st.write("Pillow verfügbar:", True)
    st.markdown("Tipps: Lade möglichst reine Chartbilder (nur Kerzen, wenig overlay), mindestens 20-40 Kerzen sichtbar für beste Ergebnisse.")

elif page == "Hilfe":
    st.header("Hilfe")
    st.markdown("""
    - Dieser Analyzer ist aggressiv und versucht, so wenig Neutral wie möglich auszugeben.
    - Wenn dein Bild trotzdem 'Neutral' liefert: 
      - Stelle sicher, dass der Kerzenbereich vollständig sichtbar ist (oben und unten nicht abgeschnitten).
      - Entferne UI-Elemente / Legenden.
      - Liefere mehr Kerzen (mind. 20).
    - Exportiere Analysen per JSON/CSV.
    - Hinweis: Pixel-basierte SL/TP sind relativ — für echte Preise benötigst du Mapping Chartpixel→Preis.
    """)

st.caption("Lumina Pro — Aggressive Image Analyzer — keine Anlageberatung.")
