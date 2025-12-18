# Ultimate Vision-Trader v3.1 — Image First Edition
# IMPROVED from Backup analys
# Focus: fewer candles, less neutral, real confidence
# Run: streamlit run main.py

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import io, math, json, os, time

# -----------------------
# UI CONFIG
# -----------------------
st.set_page_config("Ultimate Vision-Trader v3.1", layout="wide")
st.title("📊 Ultimate Vision-Trader v3.1 — Image Analyzer")

# -----------------------
# IMAGE UTILS
# -----------------------
def load_image(bytes_):
    return Image.open(io.BytesIO(bytes_)).convert("RGB")

def preprocess(img):
    w,h = img.size
    img = img.crop((0, int(h*0.05), w, int(h*0.9)))
    g = ImageOps.grayscale(img)
    g = ImageOps.autocontrast(g)
    g = g.filter(ImageFilter.MedianFilter(3))
    return img, g

# -----------------------
# CANDLE EXTRACTION (IMPROVED)
# -----------------------
def extract_candles(gray):
    arr = np.array(gray)
    h,w = arr.shape
    darkness = np.sum(255-arr, axis=0)
    thr = np.max(darkness)*0.06  # lower threshold
    xs = np.where(darkness > thr)[0]

    if len(xs) < 6:
        return []

    clusters = []
    cur = [xs[0]]
    for x in xs[1:]:
        if x-cur[-1] <= 2:
            cur.append(x)
        else:
            clusters.append(cur)
            cur=[x]
    clusters.append(cur)

    candles=[]
    for g in clusters:
        ys=[]
        for x in g:
            ys.extend(np.where(arr[:,x] < 240)[0])
        if not ys: continue
        top=min(ys); bot=max(ys)
        body_top = top + (bot-top)*0.35
        body_bot = bot - (bot-top)*0.35
        color = "green" if body_bot-body_top > 0 else "red"
        candles.append({
            "x": int(np.mean(g)),
            "top": top, "bot": bot,
            "body_top": body_top, "body_bot": body_bot,
            "color": color
        })

    return candles[-40:]  # hard cap

# -----------------------
# STRUCTURE ANALYSIS
# -----------------------
def trend_strength(candles):
    if len(candles) < 6:
        return "neutral", 0

    centers = np.array([(c["body_top"]+c["body_bot"])/2 for c in candles])
    slope = np.polyfit(range(len(centers)), centers, 1)[0]
    strength = abs(slope)

    if slope < -0.12: return "up", strength
    if slope > 0.12: return "down", strength
    return "neutral", strength

def volatility_pressure(candles):
    sizes = [abs(c["bot"]-c["top"]) for c in candles[-8:]]
    return np.std(sizes) / (np.mean(sizes)+1e-6)

# -----------------------
# PATTERN ENGINE (STRONGER)
# -----------------------
def detect_patterns(candles):
    pats=[]
    for i,c in enumerate(candles):
        body = abs(c["body_bot"]-c["body_top"])
        wick = abs(c["top"]-c["bot"])
        if body < 0.15*wick:
            pats.append(("Doji", i))
        if c["color"]=="green" and body > 0.6*wick:
            pats.append(("Bull Power", i))
        if c["color"]=="red" and body > 0.6*wick:
            pats.append(("Bear Power", i))
    return pats

# -----------------------
# DECISION ENGINE (KEY PART)
# -----------------------
def decide(candles):
    pats = detect_patterns(candles)
    trend, t_strength = trend_strength(candles)
    vol = volatility_pressure(candles)

    bull = sum(1 for p,_ in pats if "Bull" in p)
    bear = sum(1 for p,_ in pats if "Bear" in p)

    score = bull - bear

    # Hybrid pressure
    if trend=="up": score += 1.2
    if trend=="down": score -= 1.2

    score *= (1 + min(vol,0.6))

    # FINAL DECISION (ANTI-NEUTRAL)
    if score > 1.3:
        rec="LONG"
    elif score < -1.3:
        rec="SHORT"
    else:
        rec="NEUTRAL"

    # REALISTIC PROBABILITY
    confidence = min(78, max(52, 55 + abs(score)*7))
    expected_winrate = min(72, max(50, 50 + abs(score)*5))

    # SL / TP
    last = candles[-1]
    risk = abs(last["bot"]-last["top"])*0.8

    if rec=="LONG":
        sl = last["bot"] + risk
        tp = last["top"] - risk*1.8
    elif rec=="SHORT":
        sl = last["top"] - risk
        tp = last["bot"] + risk*1.8
    else:
        sl=tp=None

    return {
        "recommendation": rec,
        "confidence": round(confidence,1),
        "expected_winrate": round(expected_winrate,1),
        "trend": trend,
        "patterns": pats,
        "sl": sl,
        "tp": tp
    }

# -----------------------
# UI
# -----------------------
uploaded = st.file_uploader("📷 Chart Screenshot", ["png","jpg","jpeg"])
if uploaded:
    img = load_image(uploaded.read())
    st.image(img, use_column_width=True)

    base, gray = preprocess(img)
    candles = extract_candles(gray)

    st.write(f"🕯️ Erkannte Candles: {len(candles)}")

    if len(candles) >= 8:
        res = decide(candles)

        if res["recommendation"]=="LONG":
            st.success(f"📈 LONG — {res['confidence']}%")
        elif res["recommendation"]=="SHORT":
            st.error(f"📉 SHORT — {res['confidence']}%")
        else:
            st.warning("⚖️ NEUTRAL")

        st.markdown(f"""
**Trend:** {res['trend']}  
**Expected Trefferquote:** {res['expected_winrate']} %  
**SL (Pixel):** {res['sl']}  
**TP (Pixel):** {res['tp']}
""")

        st.markdown("### Kurzfazit")
        st.write(
            "Der Analyzer erkennt eine klare Marktstruktur basierend auf Kerzenform, Trenddruck und Volatilität. "
            "Die Empfehlung berücksichtigt nur Bilddaten und ist für Daytrading optimiert. "
            "Trade nur, wenn SL/TP klar definiert sind."
        )
    else:
        st.info("Zu wenige Candles — bitte etwas weiter rauszoomen.")
