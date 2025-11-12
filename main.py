# main.py
# Lumina Pro — Upgraded Daytrading Analyzer + Offline Image-Structure Analyzer
# - Offline image analyzer: erkennt Strukturen & Muster nur aus dem Bild (Pillow)
# - Live analyzer: Finnhub integration for real candles (Key embedded)
# - Dark theme, SVG candle renderer, decision logic, feedback card like screenshot
#
# Requirements:
#   pip install streamlit pillow
#
# Keys (embedded per user request)
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"

import streamlit as st
import json, os, time, random, io, urllib.request, urllib.parse, math
from datetime import datetime, timedelta
import statistics

# PIL for image heuristics
try:
    from PIL import Image, ImageOps, ImageStat, ImageFilter
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# -------------------------
# App config & basic style
# -------------------------
st.set_page_config(page_title="Lumina Pro — Daytrading & Image Analyzer", layout="wide", page_icon="💹")
st.markdown("""
<style>
html, body, [class*="css"] { background:#000 !important; color:#e6eef6 !important; }
.stButton>button { background:#111 !important; color:#e6eef6 !important; border:1px solid #222 !important; }
.card { background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
.small { color:#9aa6b2; font-size:13px; }
.badge { background:#111; color:#e6eef6; padding:6px 10px; border-radius:8px; border:1px solid #222; display:inline-block; }
</style>
""", unsafe_allow_html=True)

st.title("Lumina Pro — Daytrading Analyzer (Live) & Offline Image-Structure Analyzer")

# -------------------------
# Utilities
# -------------------------
def now_iso(): return datetime.utcnow().isoformat() + "Z"

def internet_ok(timeout=3):
    try:
        urllib.request.urlopen("https://www.google.com", timeout=timeout)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

# cache dir
CACHE_DIR = ".lumina_cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR, exist_ok=True)

def cache_save(key, obj):
    try:
        with open(os.path.join(CACHE_DIR, key + ".json"), "w", encoding="utf-8") as f:
            json.dump({"ts": time.time(), "data": obj}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def cache_load(key, max_age=3600*24):
    try:
        path = os.path.join(CACHE_DIR, key + ".json")
        if not os.path.exists(path): return None
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if time.time() - obj.get("ts",0) > max_age:
            return None
        return obj.get("data")
    except Exception:
        return None

# -------------------------
# Finnhub candles fetcher
# -------------------------
def fetch_finnhub_candles(symbol: str, resolution: str = "5", from_ts: int = None, to_ts: int = None):
    """
    Finnhub candle fetcher.
    symbol: e.g. 'AAPL' or 'BINANCE:BTCUSDT' (depends on Finnhub allowance)
    resolution: '1','5','15','30','60','D'
    from_ts,to_ts: unix timestamps (int)
    returns: list of candles dicts or None
    """
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None: to_ts = int(time.time())
        if from_ts is None:
            if resolution in ("1","5","15","30","60"):
                from_ts = to_ts - 60*60*24  # last 24 hours by default
            else:
                from_ts = to_ts - 60*60*24*30
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": str(int(from_ts)),
            "to": str(int(to_ts)),
            "token": FINNHUB_KEY
        }
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=25) as resp:
            raw = resp.read().decode("utf-8")
            data = json.loads(raw)
        if data.get("s") != "ok":
            return None
        ts = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        vols = data.get("v", [])
        candles=[]
        for i, t in enumerate(ts):
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except Exception:
                dt = datetime.utcnow()
            candles.append({
                "t": dt,
                "open": float(opens[i]),
                "high": float(highs[i]),
                "low": float(lows[i]),
                "close": float(closes[i]),
                "volume": float(vols[i]) if vols and i < len(vols) else 0.0
            })
        return candles
    except Exception:
        return None

# -------------------------
# Offline candle simulator
# -------------------------
def generate_simulated_candles(seed: str, periods: int, start_price: float = 100.0, resolution_minutes: int = 5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    p = float(start_price)
    prices=[]
    for _ in range(periods):
        drift = (rnd.random() - 0.49) * 0.003
        shock = (rnd.random() - 0.5) * 0.02
        p = max(0.01, p * (1 + drift + shock))
        prices.append(round(p,6))
    candles=[]
    now = datetime.utcnow()
    for i, prm in enumerate(prices):
        o = round(prm * (1 + random.uniform(-0.002,0.002)),6)
        c = prm
        h = round(max(o,c) * (1 + random.uniform(0.0,0.004)),6)
        l = round(min(o,c) * (1 - random.uniform(0.0,0.004)),6)
        t = now - timedelta(minutes=(periods - 1 - i) * resolution_minutes)
        candles.append({"t": t, "open": o, "high": h, "low": l, "close": c, "volume": random.randint(1,1000)})
    return candles

# -------------------------
# Indicators & patterns for candles
# -------------------------
def sma(values, period):
    res=[]
    for i in range(len(values)):
        if i+1 < period: res.append(None)
        else: res.append(sum(values[i+1-period:i+1])/period)
    return res

def ema(values, period):
    res=[]; k = 2.0/(period+1.0); prev=None
    for v in values:
        if prev is None: prev = v
        else: prev = v * k + prev * (1-k)
        res.append(prev)
    return res

def macd(values, fast=12, slow=26, signal=9):
    if not values: return [],[],[]
    ef = ema(values, fast); es = ema(values, slow)
    mac = [(a-b) if (a is not None and b is not None) else None for a,b in zip(ef, es)]
    mac_vals = [m for m in mac if m is not None]
    if not mac_vals: return mac, [None]*len(mac), [None]*len(mac)
    sig_vals = ema(mac_vals, signal)
    sig_iter = iter(sig_vals)
    sig_mapped=[]
    for v in mac:
        sig_mapped.append(None if v is None else next(sig_iter))
    hist = [(m-s) if (m is not None and s is not None) else None for m,s in zip(mac, sig_mapped)]
    return mac, sig_mapped, hist

def rsi(values, period=14):
    if len(values) < period+1: return [None]*len(values)
    deltas = [values[i] - values[i-1] for i in range(1,len(values))]
    gains = [d if d>0 else 0 for d in deltas]
    losses = [-d if d<0 else 0 for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
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
    if not prev: return False
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])

def is_bearish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def detect_patterns(candles):
    patterns=[]
    n = len(candles)
    for i in range(1,n):
        cur=candles[i]; prev=candles[i-1]
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
# Offline image analyzer (structure-only)
# -------------------------
def analyze_chart_image_structure(image_bytes):
    """
    Offline-only image analyzer: analyses chart structure and returns:
    - trend: 'Aufwärtstrend'|'Abwärtstrend'|'Seitwärts'
    - patterns: list of strings (e.g. 'Doji-like','Hammer-like','Engulfing-like')
    - confidence: 0-100 estimate
    - volatility_estimate: 0-100
    - quick_recommendation: 'Kaufen'|'Short'|'Neutral'
    - summary_sentences: list of short strings (3)
    This function DOES NOT try to identify the symbol or exact candles count.
    """
    if not PIL_AVAILABLE:
        return {"error":"Pillow not installed; offline image analysis unavailable."}
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")  # grayscale
    except Exception as e:
        return {"error":"Image open failed."}
    # resize for speed
    W,H = img.size
    maxw = 1400
    if W > maxw:
        img = img.resize((maxw, int(H*maxw/W)))
        W,H = img.size
    # crop to likely chart area (exclude header/footer)
    left = int(W*0.03); right = int(W*0.97)
    top = int(H*0.08); bottom = int(H*0.78)
    chart = img.crop((left, top, right, bottom))
    chart = ImageOps.autocontrast(chart, cutoff=2)
    chart = chart.filter(ImageFilter.MedianFilter(size=3))
    pix = chart.load()
    Wc,Hc = chart.size
    # compute vertical darkness profile (candles/wicks produce peaks)
    col_darkness = [0]*Wc
    for x in range(Wc):
        s=0
        for y in range(Hc):
            s += 255 - pix[x,y]
        col_darkness[x] = s
    # smooth profile
    smooth = []
    for i in range(Wc):
        window = col_darkness[max(0,i-4):min(Wc,i+5)]
        smooth.append(sum(window)/len(window) if window else 0)
    avg = sum(smooth)/len(smooth) if smooth else 0
    # detect peaks (candles), troughs
    peaks=[]; troughs=[]
    for i in range(2, Wc-2):
        if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > avg*1.25:
            peaks.append(i)
        if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < avg*0.6:
            troughs.append(i)
    # basic metrics
    peak_count = len(peaks)
    trough_count = len(troughs)
    density = peak_count / (Wc/100.0)  # peaks per 100px
    # left vs right brightness trend
    left_mean = ImageStat.Stat(chart.crop((0,0,Wc//2,Hc))).mean[0]
    right_mean = ImageStat.Stat(chart.crop((Wc//2,0,Wc,Hc))).mean[0]
    trend = "Seitwärts"
    if right_mean < left_mean - 6:  # darker right => more candles/wicks => downward bias
        trend = "Abwärtstrend"
    elif right_mean > left_mean + 6:
        trend = "Aufwärtstrend"
    # detect "doji-like" by scanning narrow vertical lines with low body height
    # simple heuristic: find narrow dark columns with low vertical variance
    doji_like = 0
    hammer_like = 0
    engulfing_like = 0
    for idx in peaks[-min(40, len(peaks)):]:
        col = [255 - pix[idx, y] for y in range(Hc)]
        # body estimation: high intensity region length
        thresh = max(1, max(col)*0.4)
        high_positions = [i for i,v in enumerate(col) if v >= thresh]
        if not high_positions: continue
        body_height = max(high_positions) - min(high_positions) if len(high_positions)>1 else 0
        # if body small relative to height -> doji-like
        if body_height < Hc*0.06:
            doji_like += 1
        # if long lower shadow -> hammer-like: check distance from bottom to body
        bottom = Hc - 1
        lower_shadow = bottom - max(high_positions) if high_positions else 0
        upper_shadow = min(high_positions) - 0 if high_positions else 0
        if lower_shadow > body_height * 2.5 and body_height > 0:
            hammer_like += 1
    # engulfing-like heuristic: look for adjacent peaks with different darkness ratio
    if len(peaks) >= 2:
        for i in range(len(peaks)-1):
            a = peaks[i]; b = peaks[i+1]
            val_a = smooth[a]; val_b = smooth[b]
            if val_b > val_a * 1.8:
                engulfing_like += 1
    # volatility estimate from variance of smooth
    variance = statistics.pvariance(smooth) if len(smooth)>1 else 0.0
    vol_est = min(100.0, max(1.0, variance**0.5 / (avg+1e-9) * 200.0))
    # build patterns list
    patterns=[]
    if doji_like > 0: patterns.append(f"{doji_like}× Doji-like")
    if hammer_like > 0: patterns.append(f"{hammer_like}× Hammer-like")
    if engulfing_like > 0: patterns.append(f"{engulfing_like}× Engulfing-like")
    if peak_count > 12 and density > 6: patterns.append("Hohe Candle-Dichte")
    if trough_count > 6: patterns.append("Mehrere lokale Tiefs erkannt")
    if not patterns: patterns.append("Keine klaren Candle-Formen erkannt")
    # confidence heuristic
    conf = 30 + min(60, int(min(peak_count, 40) * 1.2 + len(patterns)*4 + (10 if trend != "Seitwärts" else 0)))
    conf = max(5, min(95, conf))
    # recommendation logic (structure-only)
    score = 0
    if trend == "Aufwärtstrend": score += 2
    if trend == "Abwärtstrend": score -= 2
    score += (doji_like * 0)  # neutral-ish
    score += (hammer_like * 2)
    score += (engulfing_like * 2)
    # penalize high volatility for risky
    risk_pct = min(50.0, max(1.0, vol_est * 0.6))
    if score >= 3:
        rec = "Kaufen"
    elif score <= -2:
        rec = "Short"
    else:
        rec = "Neutral"
    # estimate success probability
    prob = min(95.0, max(10.0, 40.0 + score*10 + conf*0.25))
    # summary sentences
    summ=[]
    if rec == "Kaufen":
        summ.append(f"Strukturanalyse zeigt {', '.join(patterns[:3])}. Trend: {trend}.")
        summ.append(f"Schätzung Erfolg: {prob:.1f}% • Risiko (Volatilität): {risk_pct:.1f}%.")
        summ.append("Tipp: Klein einsteigen, Stop-Loss setzen; warte Bestätigung der nächsten Kerze.")
    elif rec == "Short":
        summ.append(f"Strukturanalyse liefert bärische Hinweise ({', '.join(patterns[:3])}). Trend: {trend}.")
        summ.append(f"Schätzung Erfolg: {prob:.1f}% • Risiko: {risk_pct:.1f}%.")
        summ.append("Tipp: Absichern, Small-Size, Stop oberhalb lokaler Widerstände.")
    else:
        summ.append(f"Keine eindeutige Struktur — Muster: {', '.join(patterns[:2])}. Trend: {trend}.")
        summ.append("Warte auf Bestätigung durch weitere Kerzen/Volumen.")
        summ.append("Tipp: Kein Full-Size Entry ohne Bestätigung.")
    return {
        "trend": trend,
        "patterns": patterns,
        "confidence": conf,
        "volatility": round(vol_est,2),
        "recommendation": rec,
        "probability": round(prob,1),
        "risk_pct": round(risk_pct,2),
        "summary": summ,
        "internal": {"peak_count": peak_count, "trough_count": trough_count, "density": round(density,2)}
    }

# -------------------------
# SVG renderer for candles (for Live/Sim charts)
# -------------------------
def render_svg_candles(candles, markers=None, stop=None, tp=None, width=1000, height=520):
    if not candles: return "<svg></svg>"
    n = len(candles)
    margin = 54
    chart_h = int(height * 0.62)
    max_p = max(c["high"] for c in candles)
    min_p = min(c["low"] for c in candles)
    pad = (max_p - min_p) * 0.08 if (max_p - min_p) > 0 else 1.0
    max_p += pad; min_p -= pad
    spacing = (width - 2*margin) / n
    candle_w = max(3, spacing * 0.6)
    def y(p): return margin + chart_h - (p - min_p) / (max_p - min_p) * chart_h
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>')
    # grid lines and labels
    for i in range(6):
        yy = margin + i*(chart_h/5)
        price_label = round(max_p - i*(max_p-min_p)/5, 6)
        svg.append(f'<line x1="{margin}" y1="{yy}" x2="{width-margin}" y2="{yy}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{yy+4}" font-size="11" fill="#9aa6b2">{price_label}</text>')
    # candles
    for i,c in enumerate(candles):
        cx = margin + i*spacing + spacing/2
        top = y(c["high"]); low = y(c["low"]); open_y = y(c["open"]); close_y = y(c["close"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
        by = min(open_y, close_y); bh = max(1, abs(close_y - open_y))
        svg.append(f'<rect x="{cx-candle_w/2}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
    # markers
    if markers:
        for m in markers:
            i = m.get("idx", len(candles)-1)
            if i<0 or i>=n: continue
            cx = margin + i*spacing + spacing/2
            if m.get("type","").lower() == "buy":
                svg.append(f'<polygon points="{cx-8},{margin+8} {cx+8},{margin+8} {cx},{margin-2}" fill="#00ff88"/>')
            else:
                svg.append(f'<polygon points="{cx-8},{height-30} {cx+8},{height-30} {cx},{height-46}" fill="#ff7788"/>')
    # stop / tp
    if stop:
        try:
            sy = y(stop)
            svg.append(f'<line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#ffcc00" stroke-width="2" stroke-dasharray="6,4"/>')
            svg.append(f'<text x="{width-margin-260}" y="{sy-6}" fill="#ffcc00" font-size="12">Stop: {stop}</text>')
        except Exception:
            pass
    if tp:
        try:
            ty = y(tp)
            svg.append(f'<line x1="{margin}" y1="{ty}" x2="{width-margin}" y2="{ty}" stroke="#66ff88" stroke-width="2" stroke-dasharray="4,4"/>')
            svg.append(f'<text x="{width-margin-260}" y="{ty-6}" fill="#66ff88" font-size="12">TP: {tp}</text>')
        except Exception:
            pass
    # x labels (sparse)
    for i in range(0, n, max(1, n//10)):
        x = margin + i*spacing + spacing/2
        t = ""
        try:
            t = candles[i]["t"].strftime("%m-%d %H:%M")
        except:
            t = str(candles[i].get("t",""))
        svg.append(f'<text x="{x-36}" y="{height-6}" font-size="11" fill="#9aa6b2">{t}</text>')
    svg.append('</svg>')
    return "\n".join(svg)

# -------------------------
# Decision fusion for image + live
# -------------------------
def fuse_image_and_market(image_result, candles):
    """
    Combines the offline image structural analysis with market candles (if available)
    to produce a robust recommendation and parameters (stop, tp, prob, risk).
    If candles is None, uses only image_result.
    """
    # base from image
    rec = {
        "recommendation": image_result.get("recommendation", "Neutral"),
        "prob": image_result.get("probability", 50.0),
        "risk_pct": image_result.get("risk_pct", 5.0),
        "stop": None,
        "tp": None,
        "reasons": image_result.get("patterns", [])[:6],
        "summary": image_result.get("summary", [])
    }
    last_price = None
    if candles:
        last_price = candles[-1]["close"]
        closes = [c["close"] for c in candles]
        # adjust prob with simple market indicators
        s20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes)/len(closes)
        s50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else s20
        macd_line, macd_sig, _ = macd(closes)
        bias = 0
        if s20 > s50: bias += 1
        else: bias -= 1
        if macd_line and macd_sig and macd_line[-1] is not None and macd_sig[-1] is not None:
            if macd_line[-1] > macd_sig[-1]:
                bias += 1
            else:
                bias -= 1
        rec["prob"] = max(5.0, min(95.0, rec["prob"] + bias * 6.0))
        # recompute risk using recent volatility
        if len(closes) >= 10:
            returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
            vol = statistics.pstdev(returns) if len(returns) > 1 else 0.02
        else:
            vol = 0.02
        rec["risk_pct"] = min(50.0, max(0.5, vol * 100 * 2.5))
        # stop/tp based on last price
        rec["stop"] = round(last_price * (1 - rec["risk_pct"]/100.0), 6) if last_price else None
        rec["tp"] = round(last_price * (1 + (rec["risk_pct"]/100.0) * 2.0), 6) if last_price else None
        # reasons: add indicator reasons
        if s20 > s50: rec["reasons"].append("SMA20 > SMA50 (bullish)")
        else: rec["reasons"].append("SMA20 < SMA50 (bearish)")
        if macd_line and macd_sig and macd_line[-1] is not None and macd_sig[-1] is not None:
            if macd_line[-1] > macd_sig[-1]:
                rec["reasons"].append("MACD > Signal")
            else:
                rec["reasons"].append("MACD < Signal")
    else:
        # set stop/tp relative to a nominal value if no candles
        rec["stop"] = None
        rec["tp"] = None
    return rec

# -------------------------
# UI Layout & Pages
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite", ["Home","Live Analyzer","Bild-Analyse (offline)","Portfolio","Einstellungen","Hilfe"])

# quick status
if not ONLINE:
    st.sidebar.error("❌ Keine Internetverbindung — Live Daten werden simuliert")
else:
    st.sidebar.success("✅ Internet vorhanden (Finnhub live möglich)")

# Home
if page == "Home":
    st.header("Übersicht")
    st.markdown("""
    **Lumina Pro** — Daytrading Analyzer mit Offline-Bildanalyse.
    - Bild-Analyzer: erkennt Strukturen/Muster nur aus dem Bild (offline, keine Symbol-Erkennung).
    - Live Analyzer: holt Kerzen von Finnhub (falls Internet & Key).
    """)
    st.markdown("Schnellaktionen:")
    col1,col2,col3 = st.columns(3)
    col1.button("Zur Bild-Analyse", key="goto_img")
    col2.button("Live Analyzer", key="goto_live")
    col3.button("Einstellungen", key="goto_set")
    st.markdown("---")
    st.write("Tip: Lade klare Chart-Screenshots (Kernausschnitt mit Candles sichtbar) für die beste Bild-Analyse.")

# Live Analyzer page
elif page == "Live Analyzer":
    st.header("Live Analyzer — Symbol (Finnhub)")
    left,right = st.columns([3,1])
    with right:
        symbol = st.text_input("Finnhub Symbol", value="BINANCE:BTCUSDT")
        resolution = st.selectbox("Resolution (min)", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles", 30, 800, 240, step=10)
        fallback_price = st.number_input("Fallback Price (if no live)", value=20000.0)
        run = st.button("Lade & Analysiere Symbol")
    with left:
        if run:
            candles = None
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - int(periods) * int(resolution) * 60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None:
                    st.warning("Finnhub lieferte keine Daten — Simulation wird verwendet.")
                    candles = generate_simulated_candles(symbol, periods, fallback_price, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol + "_pad", need, candles[0]["open"] if candles else fallback_price, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline or Finnhub key missing — simulation used.")
                candles = generate_simulated_candles(symbol, periods, fallback_price, int(resolution))
            # analyze candles
            closes = [c["close"] for c in candles]
            s20 = sum(closes[-20:])/20 if len(closes)>=20 else sum(closes)/len(closes)
            s50 = sum(closes[-50:])/50 if len(closes)>=50 else s20
            patt = detect_patterns(candles)
            heur = {"trend": "Aufwärtstrend" if s20 > s50 else "Abwärtstrend" if s20 < s50 else "Seitwärts", "notes": []}
            rec = fuse_image_and_market({"recommendation":"Neutral","probability":50.0,"risk_pct":5.0,"patterns":[],"summary":[]}, candles)
            st.subheader(f"{symbol} — Live Analysis")
            st.markdown(f"**Aktueller Preis:** {candles[-1]['close']:.2f}")
            if rec["recommendation"].lower().startswith("kaufen"): st.success(rec["recommendation"])
            elif rec["recommendation"].lower().startswith("short"): st.error(rec["recommendation"])
            else: st.info(rec["recommendation"])
            st.markdown(f"Wahrscheinlichkeit: **{rec['prob']}%**  •  Risiko: **{rec['risk_pct']}%**")
            st.markdown("**Begründung:**")
            for r in rec.get("reasons", [])[:8]: st.write("- " + r)
            st.markdown("**Kurz:**")
            for s in rec.get("summary", [])[:3]: st.write("- " + s)
            svg = render_svg_candles(candles[-160:], stop=rec.get("stop"), tp=rec.get("tp"))
            st.components.v1.html(svg, height=540)

# Bild-Analyse (offline-only structure)
elif page == "Bild-Analyse (offline)":
    st.header("Bild-Analyse — Struktur & Muster (offline)")
    st.markdown("Lade ein Chart-Screenshot hoch (nur Strukturanalyse — kein Symbol, keine Candle-Anzahl nötig). Die Analyse läuft **offline** (Pillow).")
    uploaded = st.file_uploader("Chart-Bild hochladen (PNG/JPG)", type=["png","jpg","jpeg"])
    col1,col2 = st.columns([3,1])
    with col2:
        show_internal = st.checkbox("Zeige interne Metriken (Peak counts)", value=False)
        analyze_btn = st.button("Analysiere Bild (offline)")
    with col1:
        if uploaded is None:
            st.info("Bitte lade ein Chartbild hoch. Tipp: nur Kerzenbereich, klares Bild.")
        else:
            st.image(uploaded, use_column_width=True)
            if analyze_btn:
                bytes_img = uploaded.read()
                if not PIL_AVAILABLE:
                    st.error("Pillow nicht installiert — installiere pillow in requirements.txt")
                else:
                    with st.spinner("Analysiere Bild offline..."):
                        img_res = analyze_chart_image_structure(bytes_img)
                    if img_res.get("error"):
                        st.error(img_res["error"])
                    else:
                        # top feedback card like screenshot
                        left2, right2 = st.columns([2,1])
                        current_est = "--"
                        left2.markdown(f"### Empfehlung: ")
                        rec = img_res["recommendation"]
                        if rec == "Kaufen":
                            left2.success(f"{rec}  •  {img_res['probability']}%")
                        elif rec == "Short":
                            left2.error(f"{rec}  •  {img_res['probability']}%")
                        else:
                            left2.info(f"{rec}  •  {img_res['probability']}%")
                        left2.markdown(f"**Risiko (Volatility est.)**: {img_res['risk_pct']}%")
                        left2.markdown("**3-Satz Zusammenfassung:**")
                        for s in img_res["summary"]:
                            left2.write("- " + s)
                        right2.markdown("**Details**")
                        right2.write("Trend: " + img_res["trend"])
                        right2.write("Confidence: " + str(img_res["confidence"]) + "%")
                        right2.write("Volatility est.: " + str(img_res["volatility"]) + "%")
                        right2.write("Patterns: ")
                        for p in img_res["patterns"][:6]:
                            right2.write("- " + p)
                        if show_internal:
                            st.markdown("**Interne Metriken**")
                            st.write(img_res.get("internal", {}))
                        # no market candles here; just present an illustrative (simulated) svg chart based on the structural profile
                        demo_candles = generate_simulated_candles("img_demo_seed", 160, start_price=100.0, resolution_minutes=5)
                        svg = render_svg_candles(demo_candles, width=1000, height=420)
                        st.components.v1.html(svg, height=450)
                        st.success("Bildanalyse abgeschlossen (offline).")

# Portfolio (light) - keep simple offline
elif page == "Portfolio":
    st.header("Portfolio (offline)")
    st.markdown("Einfaches lokales Portfolio (JSON gespeichert).")
    PORTFILE = "portfolio_lp.json"
    if not os.path.exists(PORTFILE):
        with open(PORTFILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    with open(PORTFILE, "r", encoding="utf-8") as f:
        port = json.load(f)
    st.write("Positionen:", len(port))
    if port:
        for i,p in enumerate(port):
            st.markdown(f"- **{p.get('name')}** • Menge: {p.get('qty')} • Kauf: {p.get('buy_price')}")
    st.markdown("---")
    with st.form("addpos"):
        n = st.text_input("Name")
        qty = st.number_input("Menge", value=1.0, step=0.1)
        bp = st.number_input("Kaufpreis", value=100.0, step=0.01)
        cat = st.selectbox("Kategorie", ["ETF","Aktie","Krypto"])
        if st.form_submit_button("Hinzufügen"):
            port.append({"name":n,"qty":qty,"buy_price":bp,"category":cat,"added": now_iso()})
            with open(PORTFILE, "w", encoding="utf-8") as f:
                json.dump(port, f, indent=2, ensure_ascii=False)
            st.success("Position hinzugefügt. Reload Seite zum Aktualisieren.")

# Einstellungen
elif page == "Einstellungen":
    st.header("Einstellungen")
    st.markdown("API Keys sind direkt im Code. Für mehr Sicherheit: verschiebe sie in Streamlit secrets.")
    st.write("Finnhub Key gesetzt:", bool(FINNHUB_KEY))
    st.write("Pillow installiert:", PIL_AVAILABLE)
    if st.button("Cache leeren"):
        for f in os.listdir(CACHE_DIR):
            try: os.remove(os.path.join(CACHE_DIR, f))
            except: pass
        st.success("Cache geleert")

# Hilfe
elif page == "Hilfe":
    st.header("Hilfe")
    st.markdown("""
    **Wichtig**
    - Bild-Analyzer arbeitet **offline** und nutzt nur das Bild (keine Symbol-Erkennung).
    - Live Analyzer braucht Finnhub Key (im Code). Finnhub Free Tier hat Rate Limits.
    - Installiere `pillow` in requirements.txt für beste Bildfunktionalität.
    - Empfehlungen sind probabilistische Schätzungen — **keine Anlageberatung**.
    """)
    st.markdown("Tipps zur Bildaufnahme:")
    st.write("- Croppe das Chart so, dass nur Kerzenbereich sichtbar ist (ohne UI-Header).")
    st.write("- Hohe Auflösung hilft der Struktur-Erkennung.")
    st.write("- Vermeide Overlay-Annotations (zu viele Linien) — reduziert Erkennungsqualität.")

# footer
st.markdown("---")
st.caption("Lumina Pro — Offline Image Analyzer + Finnhub Live Analyzer. Empfehlungen sind Schätzungen und keine Finanzberatung.")

# End of file
