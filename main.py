# main.py
# Lumina Pro — Bild-Analyzer + Live Finnhub Integration (kombiniert: Roboflow + lokale Heuristik)
# Keys (eingetragen auf Nutzerwunsch)
FINNHUB_KEY = "d49pi19r01qlaebikhvgd49pi19r01qlaebiki00"
ROBOFLOW_KEY = "rf_54FHs4sk2XhtAQly4YNOSTjo75B2"
ROBOFLOW_MODEL_PATH = "chart-pattern-detector/1"  # ggf. anpassen

import streamlit as st
import json, os, time, random, io, urllib.request, urllib.parse, math
from datetime import datetime, timedelta
import statistics

# Pillow optional (für lokale Bildheuristiken)
try:
    from PIL import Image, ImageFilter, ImageOps, ImageStat
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# App Config & Style (Dark)
st.set_page_config(page_title="Lumina Pro — Bild+Finnhub Analyzer", layout="wide")
st.markdown("""
<style>
body {background:#000; color:#e6eef6;}
.stButton>button {background:#111; color:#e6eef6; border:1px solid #222;}
.card {background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px;}
.small {color:#9aa6b2; font-size:13px;}
</style>
""", unsafe_allow_html=True)
st.title("Lumina Pro — Bild-Analyzer + Live Finnhub")

# -------------------------
# Utilities
# -------------------------
def internet_ok():
    try:
        urllib.request.urlopen("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False

ONLINE = internet_ok()

def now_iso(): return datetime.utcnow().isoformat() + "Z"

# -------------------------
# Roboflow multipart helper
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
    content_type_header = f'multipart/form-data; boundary={boundary}'
    return content_type_header, bytes(body)

def roboflow_detect(image_bytes):
    """
    Sendet Bild an Roboflow Detect API. Erwartet: JSON mit "predictions".
    Modellpfad anpassen falls nötig.
    """
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
        return None

# -------------------------
# PIL-Heuristik (falls Pillow installiert)
# -------------------------
def analyze_image_local(image_bytes):
    if not PIL_AVAILABLE:
        return None
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        w,h = img.size
        # crop center area where candles are likely
        cx1, cy1 = int(w*0.05), int(h*0.08)
        cx2, cy2 = int(w*0.95), int(h*0.7)
        chart = img.crop((cx1, cy1, cx2, cy2))
        chart = ImageOps.autocontrast(chart, cutoff=2)
        pix = chart.load()
        W,H = chart.size
        col_sums = [0]*W
        for x in range(W):
            s=0
            for y in range(H):
                s += 255 - pix[x,y]
            col_sums[x] = s
        # smoothing
        smooth=[]
        for i in range(W):
            vals = col_sums[max(0,i-3):min(W,i+4)]
            smooth.append(sum(vals)/len(vals))
        avg = sum(smooth)/len(smooth) if smooth else 0
        peaks=[]; minima=[]
        for i in range(2,W-2):
            if smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1] and smooth[i] > avg*1.25:
                peaks.append(i)
            if smooth[i] < smooth[i-1] and smooth[i] < smooth[i+1] and smooth[i] < avg*0.75:
                minima.append(i)
        left_mean = ImageStat.Stat(chart.crop((0,0,W//2,H))).mean[0]
        right_mean = ImageStat.Stat(chart.crop((W//2,0,W,H))).mean[0]
        trend="Seitwärts"
        if right_mean > left_mean + 6: trend="Aufwärtstrend"
        elif right_mean < left_mean - 6: trend="Abwärtstrend"
        notes=[]
        if peaks: notes.append(f"{len(peaks)} Kerzenleisten erkannt")
        if minima and len(minima)>=2: notes.append("Möglicher Double Bottom erkannt")
        if trend!="Seitwärts": notes.append("Heuristik: " + trend)
        if not notes: notes.append("Keine starke Struktur erkannt")
        return {"patterns": peaks, "minima": minima, "trend": trend, "notes": notes}
    except Exception:
        return None

# -------------------------
# Finnhub candles fetcher
# -------------------------
def fetch_finnhub_candles(symbol: str, resolution: str = "5", from_ts: int = None, to_ts: int = None):
    """
    Finnhub candle fetcher.
    symbol: 'AAPL' or 'BINANCE:BTCUSDT' (depends on Finnhub subscription)
    resolution: '1','5','15','30','60','D'
    from_ts,to_ts: unix timestamps (int)
    returns list of candles dicts or None
    """
    if not FINNHUB_KEY:
        return None
    try:
        if to_ts is None: to_ts = int(time.time())
        if from_ts is None:
            if resolution in ("1","5","15","30","60"):
                from_ts = to_ts - 60*60*24  # 24h default
            else:
                from_ts = to_ts - 60*60*24*30  # 30 days
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": str(int(from_ts)),
            "to": str(int(to_ts)),
            "token": FINNHUB_KEY
        }
        url = "https://finnhub.io/api/v1/stock/candle?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=20) as resp:
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
        for i,t in enumerate(ts):
            try:
                dt = datetime.utcfromtimestamp(int(t))
            except:
                dt = datetime.utcnow()
            candles.append({"t": dt, "open": float(opens[i]), "high": float(highs[i]), "low": float(lows[i]), "close": float(closes[i]), "volume": float(vols[i])})
        return candles
    except Exception:
        return None

# -------------------------
# Offline fallback generator
# -------------------------
def generate_simulated_candles(seed: str, periods: int, start_price: float = 100.0, resolution_minutes: int = 5):
    rnd = random.Random(abs(hash(seed)) % (2**31))
    price = float(start_price)
    prices=[]
    for _ in range(periods):
        drift = (rnd.random() - 0.49)*0.005
        shock = (rnd.random() - 0.5)*0.03
        price = max(0.01, price*(1+drift+shock))
        prices.append(round(price,6))
    # convert to OHLC per candle (simple: uses each price as close; small spread)
    candles=[]
    for i,p in enumerate(prices):
        o = max(0.01, round(p*(1+random.uniform(-0.002,0.002)),6))
        c = p
        h = max(o,c) * (1 + random.uniform(0,0.003))
        l = min(o,c) * (1 - random.uniform(0,0.003))
        candles.append({"t": datetime.utcnow() - timedelta(minutes=(periods-i)*resolution_minutes), "open": o, "high": round(h,6), "low": round(l,6), "close": c, "volume": random.randint(1,1000)})
    return candles

# -------------------------
# Indicators & patterns (subset)
# -------------------------
def sma(values, period):
    res=[]
    for i in range(len(values)):
        if i+1 < period: res.append(None)
        else: res.append(sum(values[i+1-period:i+1])/period)
    return res

def macd(values, fast=12, slow=26, signal=9):
    def ema(vals,p):
        res=[]; k=2/(p+1); prev=None
        for v in vals:
            if prev is None: prev=v
            else: prev=v*k + prev*(1-k)
            res.append(prev)
        return res
    if not values: return [],[],[]
    ef = ema(values, fast); es = ema(values, slow)
    mac = [(a-b) if (a is not None and b is not None) else None for a,b in zip(ef,es)]
    mac_vals=[m for m in mac if m is not None]
    if not mac_vals: 
        return mac, [None]*len(mac), [None]*len(mac)
    sig_vals = ema(mac_vals, signal)
    sig_iter = iter(sig_vals)
    sig_mapped=[]
    for v in mac:
        sig_mapped.append(None if v is None else next(sig_iter))
    hist=[(m-s) if (m is not None and s is not None) else None for m,s in zip(mac,sig_mapped)]
    return mac, sig_mapped, hist

def rsi(values, period=14):
    if len(values) < period+1: return [None]*len(values)
    deltas=[values[i]-values[i-1] for i in range(1,len(values))]
    gains=[d if d>0 else 0 for d in deltas]
    losses=[-d if d<0 else 0 for d in deltas]
    avg_gain=sum(gains[:period])/period
    avg_loss=sum(losses[:period])/period
    res=[None]*period
    for i in range(period, len(deltas)):
        avg_gain=(avg_gain*(period-1)+gains[i])/period
        avg_loss=(avg_loss*(period-1)+losses[i])/period
        rs = avg_gain/avg_loss if avg_loss!=0 else float('inf')
        res.append(100 - (100/(1+rs)))
    return res

# simple candle pattern detectors (robust)
def is_doji(c): 
    body = abs(c["close"] - c["open"])
    total = c["high"] - c["low"]
    return total > 0 and (body/total) < 0.15

def is_hammer(c):
    body = abs(c["close"] - c["open"]); lower = min(c["open"], c["close"]) - c["low"]
    return body > 0 and lower > 2*body

def is_shooting_star(c):
    body = abs(c["close"] - c["open"]); upper = c["high"] - max(c["open"], c["close"])
    return body > 0 and upper > 2*body

def is_bullish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] > cur["open"]) and (prev["close"] < prev["open"]) and (cur["open"] < prev["close"]) and (cur["close"] > prev["open"])

def is_bearish_engulfing(prev, cur):
    if not prev: return False
    return (cur["close"] < cur["open"]) and (prev["close"] > prev["open"]) and (cur["open"] > prev["close"]) and (cur["close"] < prev["open"])

def detect_patterns(candles):
    patterns=[]
    n=len(candles)
    for i in range(1,n):
        cur=candles[i]; prev=candles[i-1]
        if is_bullish_engulfing(prev, cur): patterns.append(("Bullish Engulfing", i))
        if is_bearish_engulfing(prev, cur): patterns.append(("Bearish Engulfing", i))
        if is_hammer(cur): patterns.append(("Hammer", i))
        if is_shooting_star(cur): patterns.append(("Shooting Star", i))
        if is_doji(cur): patterns.append(("Doji", i))
    # multi-candle patterns
    if n>=3 and (candles[-3]["close"]<candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"]>candles[-1]["open"]):
        patterns.append(("Morning Star", n-1))
    if n>=3 and (candles[-3]["close"]>candles[-3]["open"]) and is_doji(candles[-2]) and (candles[-1]["close"]<candles[-1]["open"]):
        patterns.append(("Evening Star", n-1))
    return patterns

# -------------------------
# Decision logic: combine image results + live candles
# -------------------------
def build_recommendation_from_image_and_market(rf_res, heur_res, candles):
    """
    rf_res: roboflow json or None
    heur_res: local heuristics dict or None
    candles: list of markets candles (recent)
    returns: dict with recommendation, prob%, risk%, stop, takeprofit, summary_sentences
    """
    # base scoring
    score = 0
    reasons=[]
    # image signals
    buy_votes=0; sell_votes=0; avg_conf=0.0
    if rf_res and isinstance(rf_res, dict):
        preds = rf_res.get("predictions", [])
        if preds:
            avg_conf = sum(p.get("confidence",0) for p in preds)/len(preds)
        for p in preds:
            label = (p.get("class") or "").lower()
            if any(k in label for k in ["bull","engulf","hammer","morning","threewhite","support"]): buy_votes += 1
            if any(k in label for k in ["bear","shoot","evening","top","resistance"]): sell_votes += 1
    # heuristics
    if heur_res:
        trend = heur_res.get("trend","Seitwärts")
        if trend == "Aufwärtstrend": buy_votes += 1; reasons.append("Heuristik: Aufwärtstrend")
        if trend == "Abwärtstrend": sell_votes += 1; reasons.append("Heuristik: Abwärtstrend")
        for n in heur_res.get("notes", []): reasons.append("Heuristik: " + n)
    # candle patterns from market candles
    patt = detect_patterns(candles) if candles else []
    for name,idx in patt:
        ln = name.lower()
        if any(k in ln for k in ["bull","hammer","morning","three"]): buy_votes += 1; reasons.append("Muster: "+name)
        if any(k in ln for k in ["bear","shoot","evening","top"]): sell_votes += 1; reasons.append("Muster: "+name)
    # indicator-based bias
    closes = [c["close"] for c in candles] if candles else []
    bias = 0
    if len(closes) >= 50:
        s20 = sum(closes[-20:])/20; s50 = sum(closes[-50:])/50
        if s20 > s50: bias += 1; reasons.append("SMA20 > SMA50")
        else: bias -= 1; reasons.append("SMA20 < SMA50")
    macd_line, macd_sig, macd_hist = macd(closes) if closes else ([],[],[])
    if macd_line and macd_sig and macd_line[-1] is not None and macd_sig[-1] is not None:
        if macd_line[-1] > macd_sig[-1]: bias += 1; reasons.append("MACD > Signal")
        else: bias -= 1; reasons.append("MACD < Signal")
    # final votes
    net_votes = buy_votes - sell_votes + bias
    # compute probability estimate
    base_prob = 50.0 + net_votes*8.0 + (avg_conf*15.0 if avg_conf else 0.0)
    prob = max(5.0, min(95.0, base_prob))
    # risk estimation based on volatility
    if len(closes) >= 20:
        returns = [(closes[i]-closes[i-1])/closes[i-1] for i in range(1,len(closes))]
        vol = statistics.pstdev(returns) if len(returns)>1 else 0.02
    else:
        vol = 0.02
    risk_pct = min(20.0, max(0.5, vol*100*2.5))  # eg. vol*250 ~= percent
    # stop and tp calculation (simple)
    last_price = closes[-1] if closes else None
    if last_price:
        stop_loss = round(last_price * (1 - risk_pct/100.0), 6)
        take_profit = round(last_price * (1 + (risk_pct/100.0)*2.0), 6)
    else:
        stop_loss = None; take_profit = None
    # recommendation text
    if prob >= 65:
        rec = "Kaufen (Long empfohlen)"
    elif prob <= 35:
        rec = "Short / Verkaufen empfohlen"
    else:
        rec = "Neutral / Beobachten"
    # three-sentence summary
    summary=[]
    if rec.startswith("Kaufen"):
        summary.append(f"Bild- & Marktmuster ergeben eine bullishe Tendenz (Score {net_votes:+d}).")
        summary.append(f"Empfohlener Stop-Loss: {round(risk_pct,2)}% unter Kurs -> {stop_loss if stop_loss else 'n/a'}.")
        summary.append("Tipp: Kleine Position, Stop setzen, bei Follow-through nachlegen.")
    elif rec.startswith("Short"):
        summary.append(f"Signal zeigt bärische Tendenz (Score {net_votes:+d}).")
        summary.append(f"Empfohlener Stop-Loss: {round(risk_pct,2)}% oberhalb des Kurses.")
        summary.append("Tipp: Absicherung oder Small-Size Short, warte auf Bestätigung.")
    else:
        summary.append("Kein klares Signal — Markt neutral.")
        summary.append("Volumen & weitere Kerzen abwarten; bei Bestätigung handeln.")
        summary.append("Tipp: Setze kein Full-Size Entry ohne zusätzlicher Bestätigung.")
    return {
        "recommendation": rec, "prob": round(prob,1), "risk_pct": round(risk_pct,2),
        "stop_loss": stop_loss, "take_profit": take_profit, "reasons": reasons, "summary": summary
    }

# -------------------------
# SVG Candles Renderer (simplified, good for UI)
# -------------------------
def render_svg_candles(candles, markers=None, stop=None, tp=None, width=1000, height=520):
    if not candles:
        return "<svg></svg>"
    n = len(candles)
    margin=50; chart_h = int(height*0.65)
    max_p = max(c["high"] for c in candles); min_p = min(c["low"] for c in candles)
    pad = (max_p-min_p)*0.06 if (max_p-min_p)>0 else 1.0
    max_p += pad; min_p -= pad
    spacing = (width - 2*margin) / n
    candle_w = max(3, spacing*0.6)
    def y(p): return margin + chart_h - (p-min_p)/(max_p-min_p)*chart_h
    svg = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#07070a"/>')
    # grid lines
    for i in range(6):
        yy = margin + i*(chart_h/5)
        price = round(max_p - i*(max_p-min_p)/5,6)
        svg.append(f'<line x1="{margin}" y1="{yy}" x2="{width-margin}" y2="{yy}" stroke="#151515" stroke-width="1"/>')
        svg.append(f'<text x="{8}" y="{yy+4}" font-size="11" fill="#9aa6b2">{price}</text>')
    # candles
    for i,c in enumerate(candles):
        cx = margin + i*spacing + spacing/2
        top = y(c["high"]); low = y(c["low"]); open_y = y(c["open"]); close_y = y(c["close"])
        color = "#00cc66" if c["close"] >= c["open"] else "#ff4d66"
        svg.append(f'<line x1="{cx}" y1="{top}" x2="{cx}" y2="{low}" stroke="#888" stroke-width="1"/>')
        by = min(open_y, close_y); bh = max(1, abs(close_y-open_y))
        svg.append(f'<rect x="{cx-candle_w/2}" y="{by}" width="{candle_w}" height="{bh}" fill="{color}" stroke="{color}" rx="1" ry="1"/>')
    # markers
    if markers:
        for m in markers:
            i = m.get("idx", len(candles)-1)
            if i < 0 or i >= n: continue
            cx = margin + i*spacing + spacing/2
            if m.get("type","").lower()=="buy":
                svg.append(f'<polygon points="{cx-8},{margin+8} {cx+8},{margin+8} {cx},{margin-2}" fill="#00ff88"/>')
            else:
                svg.append(f'<polygon points="{cx-8},{height-30} {cx+8},{height-30} {cx},{height-46}" fill="#ff7788"/>')
    # stop / tp lines
    if stop:
        sy = y(stop)
        svg.append(f'<line x1="{margin}" y1="{sy}" x2="{width-margin}" y2="{sy}" stroke="#ffcc00" stroke-width="2" stroke-dasharray="6,4"/>')
        svg.append(f'<text x="{width-margin-260}" y="{sy-4}" fill="#ffcc00" font-size="12">Stop: {stop}</text>')
    if tp:
        ty = y(tp)
        svg.append(f'<line x1="{margin}" y1="{ty}" x2="{width-margin}" y2="{ty}" stroke="#66ff88" stroke-width="2" stroke-dasharray="4,4"/>')
        svg.append(f'<text x="{width-margin-260}" y="{ty-4}" fill="#66ff88" font-size="12">TP: {tp}</text>')
    svg.append('</svg>')
    return "\n".join(svg)

# -------------------------
# UI: Pages
# -------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home","Bild-Analyse + Live","Live Analyzer","Hilfe"])

# HOME
if page == "Home":
    st.header("Lumina Pro — Übersicht")
    st.markdown("Upload eines Chart-Bildes auf 'Bild-Analyse + Live' oder nutze 'Live Analyzer' für direkte Symbolanalyse.")
    st.write("Roboflow:", "online" if ROBOFLOW_KEY else "kein key")
    st.write("Finnhub:", "online" if FINNHUB_KEY else "kein key")
    if not PIL_AVAILABLE:
        st.warning("Pillow nicht installiert — lokale Bildheuristik begrenzt. Installiere `pillow` in requirements.txt.")

# Bild-Analyse + Live (kombiniert)
elif page == "Bild-Analyse + Live":
    st.header("Bild-Analyse + Live-Market-Data (Roboflow + Finnhub)")
    st.markdown("Lade ein Chart-Screenshot hoch, die App kombiniert Roboflow-Detections + lokale Heuristik und holt Live-Kerzen für das gewählte Symbol.")
    uploaded = st.file_uploader("Chart Bild hochladen (PNG/JPG)", type=["png","jpg","jpeg"])
    col1,col2 = st.columns([3,1])
    with col2:
        symbol = st.text_input("Symbol für Live-Daten (Finnhub)", value="BINANCE:BTCUSDT")
        resolution = st.selectbox("Resolution (min)", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles (Anzahl)", min_value=30, max_value=600, value=240, step=10)
        start_price = st.number_input("Fallback Startpreis (bei Offline)", value=20000.0)
        do_analyze = st.button("Analysiere Bild + Live")
    with col1:
        if uploaded is None:
            st.info("Bitte Bild hochladen.")
        else:
            st.image(uploaded, use_column_width=True)
            if do_analyze:
                image_bytes = uploaded.read()
                st.info("1) Bild wird an Roboflow gesendet (falls erreichbar) ...")
                rf = None
                if ONLINE and ROBOFLOW_KEY:
                    rf = roboflow_detect(image_bytes)
                    if rf is None:
                        st.warning("Roboflow-Antwort fehlgeschlagen oder Modellpfad prüfen.")
                    else:
                        st.success(f"Roboflow: {len(rf.get('predictions',[]))} Detections")
                else:
                    st.info("Roboflow nicht erreichbar — überspringe Remote-Analyse.")
                st.info("2) Lokale Heuristik wird angewendet ...")
                heur = analyze_image_local(image_bytes) if PIL_AVAILABLE else None
                if heur:
                    st.write("Lokale Heuristik:", heur.get("trend"), heur.get("notes"))
                else:
                    st.info("Lokale Heuristik nicht verfügbar (Pillow fehlt).")
                st.info("3) Hole Live-Kerzen von Finnhub (oder simuliere fallback) ...")
                candles = None
                if ONLINE and FINNHUB_KEY:
                    to_ts = int(time.time())
                    # from_ts approx periods*resolution
                    from_ts = to_ts - int(periods) * int(resolution) * 60
                    candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                    if candles is None:
                        st.warning("Finnhub hat keine Daten zurückgegeben — fallback simulation wird genutzt.")
                        candles = generate_simulated_candles(symbol, periods, start_price, int(resolution))
                    else:
                        if len(candles) < periods:
                            # pad if less
                            need = periods - len(candles)
                            pad = generate_simulated_candles(symbol+"pad", need, start_price, int(resolution))
                            candles = pad + candles
                else:
                    st.info("Offline oder Finnhub-Key fehlt — benutze Simulation.")
                    candles = generate_simulated_candles(symbol, periods, start_price, int(resolution))
                st.success("Daten bereit — kombiniere Bild und Markt")
                rec = build_recommendation_from_image_and_market(rf, heur, candles)
                # display visual like screenshot
                st.subheader("Empfehlung")
                # current price
                current_price = candles[-1]["close"] if candles else start_price
                cols = st.columns([2,1,1,1])
                cols[0].markdown(f"**Current Price**\n\n### {current_price:.2f}")
                # recommendation badge
                if rec["recommendation"].lower().startswith("kaufen"):
                    cols[1].success(rec["recommendation"])
                elif rec["recommendation"].lower().startswith("short"):
                    cols[1].error(rec["recommendation"])
                else:
                    cols[1].info(rec["recommendation"])
                # stop/tp
                cols[2].markdown(f"**Stop Loss**\n\n{rec['stop_loss'] if rec['stop_loss'] else 'n/a'}")
                cols[3].markdown(f"**Take Profit**\n\n{rec['take_profit'] if rec['take_profit'] else 'n/a'}")
                st.markdown("---")
                st.markdown(f"**Wahrscheinlichkeit (geschätzt):** {rec['prob']}%  •  **Risiko (Stop Abstand):** {rec['risk_pct']}%")
                st.markdown("**Begründung / Gründe (Kurz):**")
                for r in (rec.get("reasons") or []): st.write("- " + r)
                st.markdown("**3 Sätze Zusammenfassung:**")
                for s in rec.get("summary", []): st.write("- " + s)
                # draw candles SVG
                svg = render_svg_candles(candles[-120:], markers=None, stop=rec.get("stop_loss"), tp=rec.get("take_profit"), width=1000, height=520)
                st.components.v1.html(svg, height=540)
                st.success("Analyse abgeschlossen.")
# LIVE ANALYZER (symbol-based)
elif page == "Live Analyzer":
    st.header("Live Analyzer — Symbolanalyse via Finnhub")
    left,right = st.columns([3,1])
    with right:
        symbol = st.text_input("Symbol (Finnhub)", value="BINANCE:BTCUSDT")
        resolution = st.selectbox("Resolution", ["1","5","15","30","60"], index=1)
        periods = st.slider("Candles", 30, 600, 240, step=10)
        do_live = st.button("Lade & Analysiere Symbol")
    with left:
        if do_live:
            if ONLINE and FINNHUB_KEY:
                to_ts = int(time.time()); from_ts = to_ts - int(periods)*int(resolution)*60
                candles = fetch_finnhub_candles(symbol, resolution, from_ts, to_ts)
                if candles is None:
                    st.warning("Finnhub lieferte keine Daten — Simulation wird genutzt.")
                    candles = generate_simulated_candles(symbol, periods, 100.0, int(resolution))
                elif len(candles) < periods:
                    need = periods - len(candles)
                    pad = generate_simulated_candles(symbol+"pad", need, candles[0]["open"] if candles else 100.0, int(resolution))
                    candles = pad + candles
            else:
                st.info("Offline oder Finnhub-Key fehlt — Simulation wird verwendet.")
                candles = generate_simulated_candles(symbol, periods, 100.0, int(resolution))
            # analyze patterns on candles
            patt = detect_patterns(candles)
            closes = [c["close"] for c in candles]
            heur = {"trend": "Neutral", "notes": []}
            if len(closes) >= 20:
                s20 = sum(closes[-20:])/20; s50 = sum(closes[-50:])/50 if len(closes)>=50 else s20
                heur["trend"] = "Aufwärts" if s20 > s50 else "Abwärts" if s20 < s50 else "Seitwärts"
            # build simple rec from candles alone
            rf_none=None
            rec = build_recommendation_from_image_and_market(rf_none, heur, candles)
            st.subheader("Ergebnis")
            st.markdown(f"**Symbol:** {symbol}  •  **Preis:** {candles[-1]['close']:.2f}")
            if rec["recommendation"].lower().startswith("kaufen"): st.success(rec["recommendation"])
            elif rec["recommendation"].lower().startswith("short"): st.error(rec["recommendation"])
            else: st.info(rec["recommendation"])
            st.write(f"Wahrscheinlichkeit: {rec['prob']}%  • Risiko: {rec['risk_pct']}%")
            st.write("Kurzbegründung:")
            for s in rec["summary"]: st.write("- " + s)
            svg = render_svg_candles(candles[-120:], markers=None, stop=rec.get("stop_loss"), tp=rec.get("take_profit"))
            st.components.v1.html(svg, height=540)

# HELP
elif page == "Hilfe":
    st.header("Hilfe & Hinweise")
    st.markdown("""
    **Wichtige Hinweise**
    - Finnhub: Du hast deinen Key im Code gesetzt. Finnhub free-tier hat Limitierungen.
    - Roboflow: Prüfe `ROBOFLOW_MODEL_PATH`, falls Roboflow keine Detections liefert.
    - Pillow (optional): Installiere `pillow` für bessere lokale Bildheuristik.
    - Empfehlungen sind Schätzungen und KEINE Anlageberatung.
    """)
    st.markdown("**Empfohlene Updates:**")
    st.write("- Robustere Roboflow-Model-Integration (prüfe Modelllabels)")
    st.write("- Gebühren & Slippage in Backtests einbauen")
    st.write("- Optional: Speicherung von Analysen in JSON für Audit")

st.markdown("---")
st.caption("Lumina Pro — Bild + Live Analyzer — Empfehlungen sind probabilistische Schätzungen, keine Anlageberatung.")
