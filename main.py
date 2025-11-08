# main.py
# Offline Finance App — Daytrading mit echten Candlesticks, Empfehlung & Risiko-Estimates
# Hinweise: Für echte Candlesticks installiere `plotly` in requirements.txt
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# Optional: Plotly für echte Candlestick-Charts (empfohlen)
PLOTLY_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ------------------------
# App-Konfiguration
# ------------------------
st.set_page_config(page_title="Offline Finance — Daytrading Pro", page_icon="💹", layout="wide")
APP_FOLDER = "."
PROFILES_FILE = os.path.join(APP_FOLDER, "profiles.json")
DEFAULT_PROFILE = "default"

# ------------------------
# Themes
# ------------------------
THEMES = {
    "Dark Black": {"bg":"#000000","fg":"#e6eef6","card":"#070707","accent":"#00ff88","muted":"#9aa6b2"},
    "Midnight Blue": {"bg":"#020428","fg":"#dbe9ff","card":"#08112a","accent":"#6fb3ff","muted":"#a7bfdc"},
    "Neon Green": {"bg":"#001100","fg":"#eafff0","card":"#022002","accent":"#39ff7a","muted":"#8fdca4"}
}
def apply_theme_css(name):
    t = THEMES.get(name, THEMES["Dark Black"])
    css = f"""
    <style>
    html, body, [class*="css"] {{background:{t['bg']} !important; color:{t['fg']} !important;}}
    .stButton>button {{background:{t['card']}; color:{t['fg']}; border:1px solid #222; border-radius:6px;}}
    .card {{background:{t['card']}; padding:12px; border-radius:10px; border:1px solid #111; margin-bottom:12px;}}
    .small {{color:{t['muted']}; font-size:13px;}}
    .metric {{color:{t['accent']}; font-weight:700;}}
    a {{color:{t['accent']}}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

apply_theme_css("Dark Black")

# ------------------------
# JSON Hilfsfunktionen
# ------------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------------
# Profile / Dateien sicherstellen
# ------------------------
if not os.path.exists(PROFILES_FILE):
    save_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": DEFAULT_PROFILE})
profiles_meta = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": DEFAULT_PROFILE})
if "active" not in profiles_meta:
    profiles_meta["active"] = profiles_meta.get("profiles",[DEFAULT_PROFILE])[0]

def profile_file(profile, name):
    return os.path.join(APP_FOLDER, f"{name}_{profile}.json")

def ensure_profile_files(profile):
    defaults = {"settings": {}, "portfolio": [], "history": [], "notifications": [], "watchlist": []}
    for k, v in defaults.items():
        p = profile_file(profile, k)
        if not os.path.exists(p):
            save_json(p, v)

ensure_profile_files(profiles_meta["active"])

# ------------------------
# Session Init & Laden
# ------------------------
st.session_state.setdefault("profile", profiles_meta["active"])
st.session_state.setdefault("series_cache", {})
st.session_state.setdefault("alerts_sent", set())

# load profile files
st.session_state.settings = load_json(profile_file(st.session_state.profile, "settings"), {})
st.session_state.portfolio = load_json(profile_file(st.session_state.profile, "portfolio"), [])
st.session_state.history = load_json(profile_file(st.session_state.profile, "history"), [])
st.session_state.notifications = load_json(profile_file(st.session_state.profile, "notifications"), [])
st.session_state.watchlist = load_json(profile_file(st.session_state.profile, "watchlist"), [])

# sensible defaults
st.session_state.settings.setdefault("goal", 10000.0)
st.session_state.settings.setdefault("theme", "Dark Black")
apply_theme_css(st.session_state.settings.get("theme","Dark Black"))

# ------------------------
# Auth (single-owner PBKDF2)
# ------------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def save_profile_state():
    save_json(profile_file(st.session_state.profile, "settings"), st.session_state.settings)
    save_json(profile_file(st.session_state.profile, "portfolio"), st.session_state.portfolio)
    save_json(profile_file(st.session_state.profile, "history"), st.session_state.history)
    save_json(profile_file(st.session_state.profile, "notifications"), st.session_state.notifications)
    save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
    pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
    if st.session_state.profile not in pm.get("profiles", []):
        pm.setdefault("profiles", []).append(st.session_state.profile)
    pm["active"] = st.session_state.profile
    save_json(PROFILES_FILE, pm)

def setup_or_login():
    st.sidebar.title("Profil & Login")
    st.sidebar.markdown("---")
    st.sidebar.write(f"Aktives Profil: **{st.session_state.profile}**")
    newp = st.sidebar.text_input("Neues Profil", value="", key="newp")
    if st.sidebar.button("Erstelle Profil") and newp.strip():
        pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
        if newp in pm.get("profiles", []):
            st.sidebar.error("Profil existiert bereits.")
        else:
            pm.setdefault("profiles", []).append(newp); pm["active"]=newp; save_json(PROFILES_FILE, pm)
            ensure_profile_files(newp)
            st.sidebar.success("Profil erstellt — bitte Seite neu laden")
    pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
    profs = pm.get("profiles", [DEFAULT_PROFILE])
    sel = st.sidebar.selectbox("Wechsle Profil", profs, index=profs.index(st.session_state.profile) if st.session_state.profile in profs else 0)
    if st.sidebar.button("Profil wechseln"):
        if sel != st.session_state.profile:
            save_profile_state(); pm["active"]=sel; save_json(PROFILES_FILE, pm); st.sidebar.info("Profil gewechselt — neu laden")
    st.sidebar.markdown("---")
    auth = st.session_state.settings.get("auth")
    if not auth:
        st.sidebar.subheader("Passwort setzen")
        p1 = st.sidebar.text_input("Passwort", type="password", key="p1")
        p2 = st.sidebar.text_input("Wiederholen", type="password", key="p2")
        if st.sidebar.button("Setze Passwort"):
            if not p1 or p1 != p2:
                st.sidebar.error("Passwörter leer oder ungleich")
            else:
                salt = os.urandom(16); dk = derive_key(p1, salt)
                st.session_state.settings["auth"] = {"salt": binascii.hexlify(salt).decode(), "key": binascii.hexlify(dk).decode(), "iterations":200_000, "dklen":72}
                save_profile_state(); st.sidebar.success("Passwort gesetzt — neu laden"); st.stop()
        st.stop()
    else:
        st.sidebar.subheader("Login")
        pwd = st.sidebar.text_input("Passwort", type="password", key="loginpwd")
        if st.sidebar.button("Einloggen"):
            try:
                salt = binascii.unhexlify(auth["salt"])
                dk = derive_key(pwd, salt, iterations=auth.get("iterations",200_000), dklen=auth.get("dklen",72))
                if binascii.hexlify(dk).decode() == auth["key"]:
                    st.session_state.auth_ok = True; st.sidebar.success("Erfolgreich eingeloggt")
                else:
                    st.sidebar.error("Falsches Passwort")
            except Exception:
                st.sidebar.error("Auth fehlerhaft")
    if not st.session_state.get("auth_ok", False):
        st.stop()

setup_or_login()

# ------------------------
# Asset Universe (erweitert suchbar)
# ------------------------
ASSETS = [
    # ETFs (Beispiele)
    {"id":"ETF_DE","category":"ETF","name":"iShares DAX ETF","symbol":"DAX.DE","div_yield":0.02},
    {"id":"ETF_US","category":"ETF","name":"SP500 ETF","symbol":"SPY","div_yield":0.015},
    {"id":"ETF_WW","category":"ETF","name":"MSCI World ETF","symbol":"IWDA","div_yield":0.012},
    {"id":"ETF_EU","category":"ETF","name":"Europa ETF","symbol":"VGK","div_yield":0.013},
    # Stocks
    {"id":"ST_AAPL","category":"Aktie","name":"Apple Inc.","symbol":"AAPL","div_yield":0.006},
    {"id":"ST_MSFT","category":"Aktie","name":"Microsoft Corp.","symbol":"MSFT","div_yield":0.008},
    {"id":"ST_NVDA","category":"Aktie","name":"NVIDIA Corp.","symbol":"NVDA","div_yield":0.0},
    {"id":"ST_TSLA","category":"Aktie","name":"Tesla Inc.","symbol":"TSLA","div_yield":0.0},
    {"id":"ST_SAP","category":"Aktie","name":"SAP SE","symbol":"SAP.DE","div_yield":0.03},
    # Crypto
    {"id":"CR_BTC","category":"Krypto","name":"Bitcoin","symbol":"BTC","div_yield":0.0},
    {"id":"CR_ETH","category":"Krypto","name":"Ethereum","symbol":"ETH","div_yield":0.0},
    {"id":"CR_SOL","category":"Krypto","name":"Solana","symbol":"SOL","div_yield":0.0},
]
ASSETS_BY_ID = {a["id"]: a for a in ASSETS}

# ------------------------
# Deterministische OHLC-Generator (Minute base)
# ------------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def generate_minute_series(asset_id: str, minutes: int = 1440, start_price: float = 100.0):
    key = f"min_{asset_id}_{minutes}_{int(start_price)}"
    if key in st.session_state.series_cache:
        return st.session_state.series_cache[key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    series = []
    now = datetime.utcnow()
    for i in range(minutes):
        drift = (rnd.random() - 0.5) * 0.0008
        vol = (rnd.random() - 0.5) * 0.006
        price = max(0.0001, price * (1 + drift + vol))
        ts = (now - timedelta(minutes=minutes - i - 1)).isoformat()
        series.append({"ts": ts, "price": round(price, 6)})
    st.session_state.series_cache[key] = series
    return series

def minutes_to_ohlc(minutes_series, candle_minutes=5):
    if not minutes_series:
        return []
    candles = []
    chunk = []
    start_ts = None
    for i, p in enumerate(minutes_series, start=1):
        if start_ts is None:
            start_ts = p["ts"]
        chunk.append(p["price"])
        if i % candle_minutes == 0:
            o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk); t = start_ts
            candles.append({"t": t, "open": round(o,6), "high": round(h,6), "low": round(l,6), "close": round(c,6)})
            chunk = []; start_ts = None
    if chunk:
        o = chunk[0]; c = chunk[-1]; h = max(chunk); l = min(chunk); t = start_ts or minutes_series[-1]["ts"]
        candles.append({"t": t, "open": round(o,6), "high": round(h,6), "low": round(l,6), "close": round(c,6)})
    return candles

# Mapping timeframe label -> candle_minutes
TIMEFRAMES = {"1m":1,"5m":5,"10m":10,"30m":30,"1h":60,"3h":180,"12h":720,"1d":1440}

def build_candles(asset_id, timeframe_label="5m", periods=200, start_price=100.0):
    if timeframe_label not in TIMEFRAMES:
        timeframe_label = "5m"
    candle_m = TIMEFRAMES[timeframe_label]
    minutes_needed = periods * candle_m
    min_series = generate_minute_series(asset_id, minutes=minutes_needed, start_price=start_price)
    candles = minutes_to_ohlc(min_series, candle_minutes=candle_m)
    # ensure length = periods (pad if necessary)
    if len(candles) < periods:
        last = candles[-1] if candles else {"t": datetime.utcnow().isoformat(), "open":start_price,"high":start_price,"low":start_price,"close":start_price}
        while len(candles) < periods:
            candles.insert(0, last)  # pad front to reach length
    return candles[-periods:]  # return oldest->newest trimmed

# ------------------------
# Indicators + Recommendation + Risk + Profit Estimate
# ------------------------
def sma(series_vals, window):
    if not series_vals or len(series_vals) < window:
        return None
    return sum(series_vals[-window:]) / window

def calc_returns_from_candles(candles):
    # returns list of returns between consecutive closes
    rets = []
    for i in range(1, len(candles)):
        prev = candles[i-1]["close"]
        cur = candles[i]["close"]
        if prev > 0:
            rets.append((cur - prev) / prev)
    return rets

def risk_label_from_vol(vol):
    # vol = stddev of returns (per candle). thresholds tuned for intraday scaled values
    if vol < 0.0008:
        return "Niedrig"
    elif vol < 0.0025:
        return "Mittel"
    else:
        return "Hoch"

def recommendation_from_candles(candles):
    # simple rule: SMA(short) > SMA(long) + positive momentum -> Kaufen
    closes = [c["close"] for c in candles]
    sma_short = sma(closes, min(10, len(closes)))
    sma_long = sma(closes, min(50, len(closes)))
    rets = calc_returns_from_candles(candles[-50:]) if len(closes) >= 2 else []
    momentum = mean(rets) if rets else 0.0
    vol = stdev(rets) if len(rets) > 1 else 0.0
    label = risk_label_from_vol(vol)
    # decision logic
    if sma_short is None or sma_long is None:
        rec = "Unklar"
    else:
        if sma_short > sma_long and momentum > 0:
            rec = "Kaufen"
        else:
            rec = "Nicht kaufen"
    return {"recommendation": rec, "sma_short": sma_short, "sma_long": sma_long, "momentum": momentum, "vol": vol, "risk_label": label}

def profit_estimate(candles, horizon_candles=12, invest_amount=1000.0):
    """
    Schätzt realistischen Gewinn über horizon_candles Kerzen.
    - berechnet durchschnittliche Rendite pro Kerze und StdDev
    - annualisiert nicht; wir simulieren horizon direkt: expected return = mean_return_per_candle * horizon
    - liefert conservative_range: (mean - std)*horizon ... (mean + std)*horizon in %
    """
    rets = calc_returns_from_candles(candles)
    if not rets:
        return {"expected_pct": 0.0, "low_pct": 0.0, "high_pct": 0.0, "expected_eur": 0.0, "low_eur": 0.0, "high_eur": 0.0}
    mu = mean(rets)  # avg per candle
    sigma = stdev(rets) if len(rets) > 1 else 0.0
    # expected total return (approx additive for small returns)
    expected_pct = mu * horizon_candles
    low_pct = (mu - sigma) * horizon_candles
    high_pct = (mu + sigma) * horizon_candles
    expected_eur = invest_amount * expected_pct
    low_eur = invest_amount * low_pct
    high_eur = invest_amount * high_pct
    return {"expected_pct": expected_pct, "low_pct": low_pct, "high_pct": high_pct, "expected_eur": expected_eur, "low_eur": low_eur, "high_eur": high_eur, "mu": mu, "sigma": sigma}

# ------------------------
# Portfolio ops (minimal)
# ------------------------
def add_position(category, asset_id, name, qty, buy_price, note=""):
    item = {"id": f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
            "category": category, "asset_id": asset_id, "name": name, "qty": float(qty), "buy_price": float(buy_price),
            "note": note, "added_at": datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "add", "item": item})
    save_profile_state()
    st.success("Position hinzugefügt.")

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p.get("id") != item_id]
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "remove", "item":{"id": item_id}})
    save_profile_state()
    st.experimental_rerun()

def portfolio_snapshot():
    tot_v = 0.0; tot_c = 0.0; rows = []
    for item in st.session_state.portfolio:
        # use last close of minute series (1d => last minute)
        min_series = generate_minute_series(item["asset_id"], minutes=1440, start_price=item.get("buy_price",100.0))
        cur = min_series[-1]["price"] if min_series else item.get("buy_price", 0.0)
        qty = float(item.get("qty", 0.0)); value = cur * qty; cost = float(item.get("buy_price",0.0)) * qty
        pnl = value - cost; pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct})
        tot_v += value; tot_c += cost
    return {"rows": rows, "total_value": tot_v, "total_cost": tot_c}

# ------------------------
# UI: Sidebar + Navigation
# ------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home","Marktplatz","Daytrading","Portfolio","Simulation","Wissensbasis","Einstellungen","Export/Import"])
st.sidebar.markdown("---")
if st.sidebar.button("Snapshot speichern"):
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "snapshot": portfolio_snapshot()})
    save_profile_state(); st.sidebar.success("Snapshot gespeichert")
st.sidebar.markdown("---")
st.sidebar.subheader("Theme")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.settings.get("theme","Dark Black")))
if st.sidebar.button("Theme anwenden"):
    st.session_state.settings["theme"] = theme_choice; apply_theme_css(theme_choice); save_profile_state(); st.experimental_rerun()
st.sidebar.markdown("---")
st.sidebar.subheader("Asset-Suche")
query = st.sidebar.text_input("Suche nach Name oder Symbol (z.B. BTC, AAPL, SPY)")
if query:
    q = query.lower()
    found = [a for a in ASSETS if q in a["name"].lower() or q in a["symbol"].lower() or q in a["id"].lower()]
    st.sidebar.write(f"Treffer: {len(found)}")
    for a in found[:15]:
        st.sidebar.write(f"- {a['name']} ({a['symbol']})")
        if st.sidebar.button(f"Open Daytrading {a['id']}", key=f"open_{a['id']}"):
            st.session_state.setdefault("daytrading_target", {"asset_id": a["id"], "timeframe":"5m"}); st.experimental_rerun()

# ------------------------
# Page: Home
# ------------------------
if page == "Home":
    st.title("Dashboard — Übersicht")
    snap = portfolio_snapshot()
    st.metric("Portfolio Wert", f"{snap['total_value']:.2f} €", delta=f"{(snap['total_value'] - snap['total_cost']):+.2f} €")
    st.write(f"Eingezahlt: {snap['total_cost']:.2f} €  •  Ziel: {st.session_state.settings.get('goal',10000.0):.2f} €")
    st.markdown("---")
    st.subheader("Schnelle Watchlist")
    for wid in st.session_state.watchlist:
        a = ASSETS_BY_ID.get(wid)
        if a:
            st.write(f"- {a['name']} ({a['symbol']})")
    st.markdown("---")
    st.subheader("Letzte Aktionen")
    for h in st.session_state.history[-8:][::-1]:
        st.write(f"{h.get('timestamp')} — {h.get('action','snapshot' if 'snapshot' in h else 'history')}")

# ------------------------
# Page: Marktplatz (übersicht + Suche)
# ------------------------
elif page == "Marktplatz":
    st.title("Marktplatz — Durchsuche Assets")
    st.markdown("Suche Assets, öffne Daytrading-Candles, füge zur Watchlist oder zum Portfolio hinzu.")
    q = st.text_input("Filter (Name oder Symbol) — leer = alle")
    assets_to_show = ASSETS
    if q:
        ql = q.lower()
        assets_to_show = [a for a in ASSETS if ql in a["name"].lower() or ql in a["symbol"].lower() or ql in a["id"].lower()]
    for a in assets_to_show:
        st.subheader(f"{a['name']} ({a['symbol']})")
        cols = st.columns([3,1])
        with cols[0]:
            # small preview: last 120 close prices
            min_series = generate_minute_series(a["id"], minutes=1440, start_price=100.0 + (abs(hash(a['id'])) % 1000)/10)
            last120 = min_series[-120:]
            if PLOTLY_AVAILABLE:
                fig = go.Figure(data=go.Scatter(x=[p["ts"] for p in last120], y=[p["price"] for p in last120], mode="lines", line=dict(color="lightblue")))
                fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=180, paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"], plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"], font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart({p["ts"]: p["price"] for p in last120})
        with cols[1]:
            cur = last120[-1]["price"] if last120 else 0.0
            st.write(f"Aktuell: {cur:.2f} €")
            if st.button("Open Daytrading", key=f"dt_{a['id']}"):
                st.session_state["daytrading_target"] = {"asset_id": a["id"], "timeframe": "5m"}
                st.experimental_rerun()
            if st.button("Zur Watchlist", key=f"wl_{a['id']}"):
                if a["id"] not in st.session_state.watchlist:
                    st.session_state.watchlist.append(a["id"]); save_profile_state(); st.success("Zur Watchlist")
                else:
                    st.info("Schon in Watchlist")
            with st.form(key=f"addform_{a['id']}", clear_on_submit=False):
                qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                bp = st.number_input("Kaufpreis", min_value=0.0001, value=float(cur) if cur>0 else 100.0, step=0.01, key=f"bp_{a['id']}")
                note = st.text_area("Notiz", value="", key=f"note_{a['id']}", height=60)
                if st.form_submit_button("Zum Portfolio hinzufügen"):
                    add_position(a["category"], a["id"], a["name"], qty, bp, note)

# ------------------------
# Page: Daytrading (Candlesticks + Recommendation)
# ------------------------
elif page == "Daytrading":
    st.title("Daytrading — Candlestick Chart")
    # if opened from Marktplatz or sidebar search, use target
    target = st.session_state.get("daytrading_target", None)
    asset_ids = [a["id"] + " — " + a["name"] for a in ASSETS]
    sel = st.selectbox("Wähle Asset", asset_ids, index=asset_ids.index(target["asset_id"] + " — " + ASSETS_BY_ID[target["asset_id"]]["name"]) if target else 0)
    asset_id = sel.split(" — ")[0]
    timeframe = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=list(TIMEFRAMES.keys()).index(target["timeframe"]) if target else list(TIMEFRAMES.keys()).index("5m"))
    periods = st.number_input("Anzahl Kerzen", min_value=20, max_value=2000, value=200, step=10)
    start_price = st.number_input("Startpreis (Sim)", min_value=0.01, value=100.0, step=0.1)
    invest_amount = st.number_input("Simulierter Investitionsbetrag (€)", min_value=1.0, value=1000.0, step=1.0)

    if not PLOTLY_AVAILABLE:
        st.warning("Plotly nicht installiert — Candlestick-Charts sind nur mit Plotly verfügbar. Installiere 'plotly' in requirements.txt für beste Darstellung.")
    # Generate candles
    candles = build_candles(asset_id, timeframe_label=timeframe, periods=int(periods), start_price=float(start_price))
    # Plot
    st.markdown("### Candlestick")
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Candlestick(
            x=[c["t"] for c in candles],
            open=[c["open"] for c in candles],
            high=[c["high"] for c in candles],
            low=[c["low"] for c in candles],
            close=[c["close"] for c in candles],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=480,
                          paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                          plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                          font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart({c["t"]: c["close"] for c in candles})

    # Indicators & Recommendation
    rec = recommendation_from_candles(candles)
    st.markdown("### Analyse & Empfehlung")
    st.write(f"**Empfehlung:** {rec['recommendation']}")
    st.write(f"**SMA short:** {rec['sma_short']:.6f}  •  **SMA long:** {rec['sma_long']:.6f}" if rec['sma_short'] and rec['sma_long'] else "Nicht genug Daten für SMA")
    st.write(f"**Momentum (durchschn. Rendite pro Kerze):** {rec['momentum']:.6f}")
    st.write(f"**Volatilität (StdDev der Renditen):** {rec['vol']:.6f}  •  **Risiko:** {rec['risk_label']}")

    # Profit estimate for several horizons (in candles)
    st.markdown("### Realistischer Gewinn-Estimate (konservativ)")
    hmap = {"Kurz (12 Kerzen)":12, "Mittel (60 Kerzen)":60, "Lang (240 Kerzen)":240}
    est_rows = []
    for label, hc in hmap.items():
        pe = profit_estimate(candles, horizon_candles=hc, invest_amount=float(invest_amount))
        est_rows.append({"Horizon": label, "Erwartung (%)": round(pe["expected_pct"]*100,3), "Niedrig (%)": round(pe["low_pct"]*100,3), "Hoch (%)": round(pe["high_pct"]*100,3), "Erwartung (€)": round(pe["expected_eur"],2)})
    st.table(est_rows)

    # Quick actions: add to portfolio / watchlist
    cols = st.columns(3)
    with cols[0]:
        if st.button("Zur Watchlist hinzufügen"):
            if asset_id not in st.session_state.watchlist:
                st.session_state.watchlist.append(asset_id); save_profile_state(); st.success("Zur Watchlist")
            else:
                st.info("Bereits in Watchlist")
    with cols[1]:
        if st.button("Als Position hinzufügen (1 Einheit)"):
            add_position(ASSETS_BY_ID[asset_id]["category"], asset_id, ASSETS_BY_ID[asset_id]["name"], 1.0, candles[-1]["close"], note="Added from Daytrading")
    with cols[2]:
        if st.button("Daten exportieren (CSV)"):
            # produce simple CSV from candles
            csv_lines = "t,open,high,low,close\n" + "\n".join([f'{c["t"]},{c["open"]},{c["high"]},{c["low"]},{c["close"]}' for c in candles])
            st.download_button("Download CSV", data=csv_lines, file_name=f"{asset_id}_{timeframe}_candles.csv", mime="text/csv")

# ------------------------
# Page: Portfolio
# ------------------------
elif page == "Portfolio":
    st.title("Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Portfolio leer — füge Positionen im Marktplatz hinzu.")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €  •  Eingezahlt: {snap['total_cost']:.2f} €")
        st.progress(min(snap['total_value'] / float(st.session_state.settings.get("goal",10000.0)), 1.0))
        st.markdown("---")
        for r in snap["rows"]:
            item = r["item"]
            with st.expander(f"{item['name']} — Wert: {r['value']:.2f} €  •  PnL: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)"):
                st.write(f"Menge: {item['qty']} • Kaufpreis: {item['buy_price']:.2f} €")
                # small preview candlestick
                candles = build_candles(item["asset_id"], timeframe_label="30m", periods=120, start_price=item.get("buy_price",100.0))
                if PLOTLY_AVAILABLE:
                    fig = go.Figure(data=[go.Candlestick(x=[c["t"] for c in candles], open=[c["open"] for c in candles], high=[c["high"] for c in candles], low=[c["low"] for c in candles], close=[c["close"] for c in candles], increasing_line_color='green', decreasing_line_color='red')])
                    fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=300)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart({c["t"]: c["close"] for c in candles})
                if st.button("Entfernen", key=f"rm_{item['id']}"):
                    remove_position(item['id'])

# ------------------------
# Page: Simulation (einfache Tagesserie)
# ------------------------
elif page == "Simulation":
    st.title("Simulation")
    days = st.selectbox("Zeitraum (Tage)", [7,30,90,180,365], index=2)
    # produce daily aggregated portfolio value
    def simulate_daily(days):
        snap = portfolio_snapshot()
        if not snap["rows"]:
            return []
        dates = [(datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat() for i in range(days)]
        combined = [0.0] * days
        for r in snap["rows"]:
            asset = r["item"]
            minutes = days * 1440
            series_min = generate_minute_series(asset["asset_id"], minutes=minutes, start_price=asset.get("buy_price",100.0))
            for i in range(days):
                idx = (i+1)*1440 - 1
                if idx < len(series_min):
                    combined[i] += series_min[idx]["price"] * asset.get("qty",0.0)
        return [{"date": dates[i], "value": round(combined[i],4)} for i in range(days)]
    hist = simulate_daily(days)
    if hist:
        st.line_chart({h["date"]: h["value"] for h in hist})
    else:
        st.info("Keine Daten — Portfolio leer.")

# ------------------------
# Wissensbasis (erweitert)
# ------------------------
elif page == "Wissensbasis":
    st.title("Wissensbasis")
    topics = {
        "ETF_vs_Aktie": "ETFs bündeln... (ausführlicher Text mit Beispielen und Stil).",
        "Technische_Analyse": "Technische Analyse: SMA, EMA, RSI, MACD — Einsatz und Grenzen.",
        "Risk_Management": "Positionsgröße, Stop-Loss, Diversifikation — praktische Regeln.",
        "Krypto_Basics": "Krypto-Grundlagen, Security, Positionierung, Volatilität."
    }
    for k,t in topics.items():
        with st.expander(k.replace("_"," ")):
            st.write(t)

# ------------------------
# Export / Import
# ------------------------
elif page == "Export/Import":
    st.title("Export / Import")
    out = {"profile": st.session_state.profile, "settings": st.session_state.settings, "portfolio": st.session_state.portfolio, "history": st.session_state.history, "notifications": st.session_state.notifications}
    st.download_button("Export JSON", data=json.dumps(out, ensure_ascii=False, indent=2), file_name=f"export_{st.session_state.profile}.json", mime="application/json")
    up = st.file_uploader("Import JSON", type=["json"])
    if up:
        try:
            obj = json.loads(up.read().decode("utf-8"))
            st.session_state.settings.update(obj.get("settings", {}))
            inc = obj.get("portfolio", [])
            if isinstance(inc, list) and inc:
                st.session_state.portfolio.extend(inc)
            st.session_state.history.extend(obj.get("history", []))
            st.session_state.notifications.extend(obj.get("notifications", []))
            save_profile_state(); st.success("Importiert — Seite neu laden")
        except Exception as e:
            st.error(f"Import fehlgeschlagen: {e}")

# ------------------------
# Einstellungen
# ------------------------
elif page == "Einstellungen":
    st.title("Einstellungen")
    st.subheader("Theme")
    cur = st.session_state.settings.get("theme","Dark Black")
    choice = st.selectbox("Wähle Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(cur))
    if st.button("Anwenden"):
        st.session_state.settings["theme"] = choice; apply_theme_css(choice); save_profile_state(); st.experimental_rerun()
    st.markdown("---")
    st.subheader("Allgemein")
    goal = st.number_input("Finanzziel (€)", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Speichere"):
        st.session_state.settings["goal"] = float(goal); save_profile_state(); st.success("Gespeichert")

# ------------------------
# Footer & persist
# ------------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Deterministische Candlestick-Simulation • Klassische Farben (grün/up, rot/down)</div>", unsafe_allow_html=True)
save_profile_state()
