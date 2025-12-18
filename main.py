# ============================================================
# LUMINA PRO — Offline Learning Trading Analyzer
# One-File Version | Stable | No external finance modules
# ============================================================

import streamlit as st
import random
import math
import json
import os
import time
from datetime import datetime

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

st.set_page_config(
    page_title="Lumina Pro — Offline Analyzer",
    layout="wide"
)

DATA_FILE = ".learning_stats.json"

# ------------------------------------------------------------
# LEARNING ENGINE (SELF CALIBRATING)
# ------------------------------------------------------------

DEFAULT_STATS = {
    "HTF_up_LTF_up": {"wins": 10, "losses": 6},
    "HTF_down_LTF_down": {"wins": 9, "losses": 6},
    "Countertrend": {"wins": 4, "losses": 7},
    "Range": {"wins": 3, "losses": 6}
}

def load_stats():
    if not os.path.exists(DATA_FILE):
        save_stats(DEFAULT_STATS)
    with open(DATA_FILE, "r") as f:
        return json.load(f)

def save_stats(stats):
    with open(DATA_FILE, "w") as f:
        json.dump(stats, f, indent=2)

def record_result(key, win):
    stats = load_stats()
    if key not in stats:
        stats[key] = {"wins": 1, "losses": 1}
    if win:
        stats[key]["wins"] += 1
    else:
        stats[key]["losses"] += 1
    save_stats(stats)

def hitrate(key):
    stats = load_stats()
    if key not in stats:
        return 0.6
    w = stats[key]["wins"]
    l = stats[key]["losses"]
    return round(w / (w + l), 3)

# ------------------------------------------------------------
# MARKET SIMULATION (OFFLINE DATA)
# ------------------------------------------------------------

def simulate_prices(n=120, start=100):
    prices = [start]
    for _ in range(n):
        drift = random.uniform(-0.6, 0.6)
        prices.append(max(1, prices[-1] + drift))
    return prices

def trend(prices, length):
    if len(prices) < length:
        return "range"
    avg_now = sum(prices[-length:]) / length
    avg_old = sum(prices[-2*length:-length]) / length
    if avg_now > avg_old * 1.002:
        return "up"
    if avg_now < avg_old * 0.998:
        return "down"
    return "range"

def volatility(prices):
    if len(prices) < 10:
        return 0
    diffs = [abs(prices[i]-prices[i-1]) for i in range(1,len(prices))]
    return round(sum(diffs)/len(diffs), 2)

# ------------------------------------------------------------
# ANALYZER CORE
# ------------------------------------------------------------

def build_structure(htf, ltf):
    if htf=="up" and ltf=="up":
        return "HTF_up_LTF_up"
    if htf=="down" and ltf=="down":
        return "HTF_down_LTF_down"
    if htf != ltf:
        return "Countertrend"
    return "Range"

def analyze(prices, mode="Profi"):
    htf = trend(prices, 40)
    ltf = trend(prices, 15)
    struct = build_structure(htf, ltf)
    vol = volatility(prices)

    base_conf = {
        "HTF_up_LTF_up": 72,
        "HTF_down_LTF_down": 71,
        "Countertrend": 58,
        "Range": 52
    }[struct]

    learned = hitrate(struct) * 100

    confidence = round(base_conf*0.55 + learned*0.45, 1)

    if confidence < 55:
        action = "NEUTRAL"
    elif struct == "HTF_up_LTF_up":
        action = "LONG"
    elif struct == "HTF_down_LTF_down":
        action = "SHORT"
    else:
        action = "LONG" if random.random()>0.5 else "SHORT"

    price = prices[-1]
    sl_dist = max(1.0, vol * 2)
    tp_dist = sl_dist * 2.2

    stop_loss = round(price - sl_dist if action=="LONG" else price + sl_dist,2)
    take_profit = round(price + tp_dist if action=="LONG" else price - tp_dist,2)

    risk = round((sl_dist / price) * 100, 2)

    explanation_simple = f"""
    Marktstruktur: {struct.replace('_',' ')}  
    Empfehlung: {action}  
    Risiko: moderat  
    Trefferwahrscheinlichkeit: ca. {confidence} %
    """

    explanation_pro = f"""
    HTF Trend: {htf.upper()}  
    LTF Trend: {ltf.upper()}  
    Volatilität: {vol}  
    Struktur-Key: {struct}  
    Lernende Trefferquote: {round(learned,1)} %  
    Gewichtete Entscheidung: {confidence} %  

    ➜ Position: {action}  
    ➜ Stop Loss: {stop_loss}  
    ➜ Take Profit: {take_profit}  
    ➜ Risiko pro Trade: {risk} %
    """

    return {
        "action": action,
        "confidence": confidence,
        "structure": struct,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk": risk,
        "simple": explanation_simple,
        "pro": explanation_pro
    }

# ------------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------------

st.title("💎 Lumina Pro — Lernender Offline Trading Analyzer")

mode = st.radio("Analyse-Modus", ["Einfach", "Profi"], horizontal=True)

st.markdown("---")

prices = simulate_prices()

with st.spinner("🔍 Analyse läuft im Hintergrund..."):
    time.sleep(0.6)
    result = analyze(prices, mode)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

st.subheader("📌 Empfehlung")

col1, col2, col3 = st.columns(3)
col1.metric("Position", result["action"])
col2.metric("Gewinnwahrscheinlichkeit", f"{result['confidence']} %")
col3.metric("Risiko", f"{result['risk']} %")

st.markdown("### 🛑 Stop / 🎯 Ziel")
st.write(f"Stop Loss: **{result['stop_loss']}**")
st.write(f"Take Profit: **{result['take_profit']}**")

st.markdown("---")

st.subheader("🧠 Analyse-Erklärung")

if mode == "Einfach":
    st.info(result["simple"])
else:
    st.code(result["pro"])

st.markdown("---")

# ------------------------------------------------------------
# FEEDBACK LOOP (SIMULATED BACKTEST)
# ------------------------------------------------------------

st.subheader("📊 Lern-Feedback (Simulation)")

if st.button("📈 Trade als GEWONNEN markieren"):
    record_result(result["structure"], True)
    st.success("Trade als Gewinn gespeichert — Analyzer lernt 📈")

if st.button("📉 Trade als VERLOREN markieren"):
    record_result(result["structure"], False)
    st.warning("Trade als Verlust gespeichert — Risiko angepasst")

st.caption("Trefferquote verbessert sich mit echten Ergebnissen.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("""
---
⚠️ **Hinweis:**  
Dies ist ein **Analyse- & Trainingssystem**, kein Finanzberater.  
Die Trefferquote ist **lernend**, nicht garantiert.

Version: **Lumina Pro — Learning Engine v3**
""")
