# main.py
# Offline Finance App with Daytrading Candlestick Charts (classic colors)
# Author: generated (offline deterministic simulation)
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# Optional libs
PLOTLY_AVAILABLE = False
PANDAS_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# ----------------------
# Basic App Config
# ----------------------
st.set_page_config(page_title="Offline Finance — Daytrading", page_icon="💹", layout="wide")
APP_FOLDER = "."
PROFILES_FILE = os.path.join(APP_FOLDER, "profiles.json")
DEFAULT_PROFILE = "default"

# ----------------------
# Themes
# ----------------------
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

# ----------------------
# JSON helpers
# ----------------------
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

# ----------------------
# Ensure profile files
# ----------------------
if not os.path.exists(PROFILES_FILE):
    save_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": DEFAULT_PROFILE})

profiles_meta = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": DEFAULT_PROFILE})
if "active" not in profiles_meta:
    profiles_meta["active"] = profiles_meta.get("profiles",[DEFAULT_PROFILE])[0]

def profile_file(profile, name):
    return os.path.join(APP_FOLDER, f"{name}_{profile}.json")

def ensure_profile(profile):
    files = {"settings":{}, "portfolio":[], "history":[], "notifications":[], "watchlist":[]}
    for k,v in files.items():
        p = profile_file(profile, k)
        if not os.path.exists(p):
            save_json(p, v)

ensure_profile(profiles_meta["active"])

# ----------------------
# Session init
# ----------------------
st.session_state.setdefault("profile", profiles_meta["active"])
st.session_state.setdefault("series_cache", {})
st.session_state.setdefault("alerts_sent", set())

# load profile data
st.session_state.settings = load_json(profile_file(st.session_state.profile, "settings"), {})
st.session_state.portfolio = load_json(profile_file(st.session_state.profile, "portfolio"), [])
st.session_state.history = load_json(profile_file(st.session_state.profile, "history"), [])
st.session_state.notifications = load_json(profile_file(st.session_state.profile, "notifications"), [])
st.session_state.watchlist = load_json(profile_file(st.session_state.profile, "watchlist"), [])

st.session_state.settings.setdefault("goal", 10000.0)
st.session_state.settings.setdefault("theme", "Dark Black")
apply_theme_css(st.session_state.settings.get("theme","Dark Black"))

# ----------------------
# Authentication (single-owner)
# ----------------------
def derive_key(pwd: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", pwd.encode("utf-8"), salt, iterations, dklen)

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
    newp = st.sidebar.text_input("Neues Profil (optional)", value="", key="newp")
    if st.sidebar.button("Erstelle Profil") and newp.strip():
        pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
        if newp in pm.get("profiles", []):
            st.sidebar.error("Profil existiert bereits.")
        else:
            pm.setdefault("profiles", []).append(newp); pm["active"]=newp; save_json(PROFILES_FILE, pm)
            ensure_profile(newp)
            st.sidebar.success("Profil erstellt — neu laden")
    pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
    profs = pm.get("profiles",[DEFAULT_PROFILE])
    sel = st.sidebar.selectbox("Wechsle Profil", profs, index=profs.index(st.session_state.profile) if st.session_state.profile in profs else 0)
    if st.sidebar.button("Profil anwenden"):
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
                save_profile_state(); st.sidebar.success("Passwort gesetzt — bitte neu einloggen"); st.stop()
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

# ----------------------
# Assets
# ----------------------
ASSETS = [
    {"id":"ETF_DE","category":"ETF","name":"Deutschland ETF","symbol":"DE.ETF","div_yield":0.015},
    {"id":"ETF_US","category":"ETF","name":"USA ETF","symbol":"US.ETF","div_yield":0.012},
    {"id":"CR_BTC","category":"Krypto","name":"Bitcoin","symbol":"BTC","div_yield":0.0},
    {"id":"ST_AAPL","category":"Aktie","name":"Apple","symbol":"AAPL","div_yield":0.006},
]

ASSETS_BY_ID = {a["id"]: a for a in ASSETS}

# ----------------------
# Deterministic minute-series generator (OHLC)
# ----------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def generate_minute_prices(asset_id: str, minutes: int = 1440, start_price: float = 100.0):
    """
    Generate per-minute prices (close) deterministically and return list of dicts:
    [{"ts": datetime, "price": float}, ...] length = minutes
    """
    key = f"min_{asset_id}_{minutes}_{int(start_price)}"
    if key in st.session_state.series_cache:
        return st.session_state.series_cache[key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    series = []
    now = datetime.utcnow()
    for i in range(minutes):
        # simulate small drift & volatility per minute
        drift = (rnd.random() - 0.5) * 0.0004  # small drift
        vol = (rnd.random() - 0.5) * 0.005    # minute vol
        price = max(0.01, price * (1 + drift + vol))
        ts = (now - timedelta(minutes=minutes - i - 1))
        series.append({"ts": ts.isoformat(), "price": round(price, 6)})
    st.session_state.series_cache[key] = series
    return series

def to_ohlc_from_minute(series_min, candle_minutes=1):
    """
    Aggregate minute-level close prices into OHLC candles with given candle_minutes.
    series_min: list of {"ts":iso, "price":float}
    returns list of {"t":iso_start, "open","high","low","close"}
    """
    if not series_min:
        return []
    candles = []
    chunk = []
    for i, p in enumerate(series_min, start=1):
        chunk.append(p["price"])
        if i % candle_minutes == 0:
            open_p = chunk[0]
            close_p = chunk[-1]
            high_p = max(chunk)
            low_p = min(chunk)
            t = p["ts"]
            candles.append({"t": t, "open": round(open_p,6), "high": round(high_p,6), "low": round(low_p,6), "close": round(close_p,6)})
            chunk = []
    # leftover chunk -> make final candle
    if chunk:
        open_p = chunk[0]
        close_p = chunk[-1]
        high_p = max(chunk)
        low_p = min(chunk)
        t = series_min[-1]["ts"]
        candles.append({"t": t, "open": round(open_p,6), "high": round(high_p,6), "low": round(low_p,6), "close": round(close_p,6)})
    return candles

# ----------------------
# Portfolio ops
# ----------------------
def save_profile_state_wrapper():
    save_profile_state()

def add_position(category, asset_id, name, qty, buy_price, note=""):
    item = {"id": f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
            "category": category, "asset_id": asset_id, "name": name, "qty": float(qty), "buy_price": float(buy_price),
            "note": note, "added_at": datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action":"add", "item": item})
    save_profile_state_wrapper()
    st.success(f"{name} hinzugefügt.")

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p.get("id") != item_id]
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action":"remove", "item":{"id": item_id}})
    save_profile_state_wrapper()
    st.success("Position entfernt.")
    st.experimental_rerun()

# ----------------------
# Snapshot & analytics (simple)
# ----------------------
def portfolio_snapshot():
    tot_v = 0.0; tot_c = 0.0; rows = []
    for item in st.session_state.portfolio:
        if not isinstance(item, dict): continue
        series = generate_minute_prices(item["asset_id"], 60*24, start_price=item.get("buy_price",100.0))
        cur = series[-1]["price"]
        qty = float(item.get("qty",0.0))
        value = cur * qty
        cost = item.get("buy_price",0.0) * qty
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct})
        tot_v += value; tot_c += cost
    return {"rows": rows, "total_value": tot_v, "total_cost": tot_c}

# ----------------------
# Simulate OHLC + Candlestick plotting helper
# ----------------------
# Map user timeframe label to candle_minutes
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "10m": 10,
    "30m": 30,
    "1h": 60,
    "3h": 180,
    "12h": 720,
    "1d": 1440
}

def build_candles_for_asset(asset_id, timeframe_label="5m", periods=200, start_price=100.0):
    """
    Build 'periods' candles at chosen timeframe.
    Implementation: generate minutes = periods * candle_minutes, from generate_minute_prices, then aggregate.
    Returns list of candles oldest->newest.
    """
    if timeframe_label not in TIMEFRAMES:
        timeframe_label = "5m"
    candle_m = TIMEFRAMES[timeframe_label]
    minutes_needed = periods * candle_m
    minute_series = generate_minute_prices(asset_id, minutes=minutes_needed, start_price=start_price)
    candles = to_ohlc_from_minute(minute_series, candle_minutes=candle_m)
    return candles

def plot_candles(candles, title="Candles", show_volume=False):
    """
    candles: list of {"t","open","high","low","close"}
    If plotly available -> candlestick, else fallback to two-line chart (open/close)
    """
    if not candles:
        st.info("Keine Candles verfügbar.")
        return
    if PLOTLY_AVAILABLE:
        fig = go.Figure(data=[go.Candlestick(
            x=[c["t"] for c in candles],
            open=[c["open"] for c in candles],
            high=[c["high"] for c in candles],
            low=[c["low"] for c in candles],
            close=[c["close"] for c in candles],
            increasing_line_color='green', decreasing_line_color='red'
        )])
        fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=420,
                          paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                          plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                          font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        # fallback: show close price line and colored bars for direction
        dates = [c["t"] for c in candles]
        closes = [c["close"] for c in candles]
        opens = [c["open"] for c in candles]
        st.line_chart({d: closes[i] for i,d in enumerate(dates)})

# ----------------------
# Sidebar & Navigation
# ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home","Marktplatz","Daytrading","Portfolio","Simulation","Einstellungen"])
st.sidebar.markdown("---")
if st.sidebar.button("Snapshot speichern"):
    # simple snapshot
    snap = portfolio_snapshot()
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "snapshot": snap})
    save_profile_state_wrapper()
    st.sidebar.success("Snapshot gespeichert")
st.sidebar.markdown("---")
st.sidebar.subheader("Theme")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(st.session_state.settings.get("theme","Dark Black")))
if st.sidebar.button("Theme anwenden"):
    st.session_state.settings["theme"] = theme_choice
    apply_theme_css(theme_choice)
    save_profile_state_wrapper()
    st.experimental_rerun()
st.sidebar.markdown("---")
st.sidebar.subheader("Schnellsuche")
q = st.sidebar.text_input("Asset suchen")
if q:
    res = [a for a in ASSETS if q.lower() in a["name"].lower() or q.lower() in a["symbol"].lower()]
    for r in res:
        st.sidebar.write(f"- {r['name']} ({r['symbol']})")

# ----------------------
# HOME
# ----------------------
if page == "Home":
    st.title("Dashboard")
    snap = portfolio_snapshot()
    st.metric("Portfolio Wert", f"{snap['total_value']:.2f} €", delta=f"{(snap['total_value'] - snap['total_cost']):+.2f} €")
    st.write(f"Eingezahlt: {snap['total_cost']:.2f} € • Ziel: {st.session_state.settings.get('goal',10000.0):.2f} €")
    st.markdown("---")
    st.subheader("Watchlist")
    for wid in st.session_state.watchlist:
        a = ASSETS_BY_ID.get(wid)
        if a:
            st.write(f"- {a['name']} ({a['symbol']})")
    st.markdown("---")
    st.subheader("Letzte Aktivitäten")
    for h in st.session_state.history[-10:][::-1]:
        st.write(f"{h.get('timestamp')} — {h.get('action','snapshot' if 'snapshot' in h else 'history')}")
    st.markdown("---")
    st.subheader("Offline News")
    rnd = random.Random(deterministic_seed("NEWS"+st.session_state.profile))
    for i in range(5):
        k = rnd.choice(["Zinsen","Inflation","Tech","Krypto","Rohstoffe"])
        st.write(f"- {k}: Marktentwicklung {rnd.randint(-5,5)}%")

# ----------------------
# MARKTPLATZ
# ----------------------
elif page == "Marktplatz":
    st.title("Marktplatz")
    st.markdown("Simulierte Assets — candlesticks im Daytrading-Modus verfügbar.")
    intervals = {"1 Monat":30, "3 Monate":90, "6 Monate":180, "1 Jahr":365}
    for a in ASSETS:
        st.subheader(f"{a['name']} — {a['symbol']}")
        cols = st.columns([3,1])
        with cols[0]:
            # show a daily line chart (fallback) and provide Daytrading button
            daily_series = generate_minute_prices(a["id"], minutes=1440, start_price=100.0 + (abs(hash(a['id'])) % 1000)/10)
            # show last 120 closes as line for overview
            last120 = daily_series[-120:]
            st.line_chart({p["ts"]: p["price"] for p in last120})
            st.markdown("Daytrading: Candles -> klick auf 'Öffne Daytrading'.")
        with cols[1]:
            cur = daily_series[-1]["price"]
            st.write(f"Aktuell: {cur:.2f} €")
            if st.button("Öffne Daytrading", key=f"dt_{a['id']}"):
                # set session to open daytrading for this asset
                st.session_state.setdefault("daytrading_asset", a["id"])
                st.session_state.setdefault("daytrading_timeframe", "5m")
                st.experimental_rerun()
            if st.button("Zur Watchlist", key=f"wl_{a['id']}"):
                if a["id"] not in st.session_state.watchlist:
                    st.session_state.watchlist.append(a["id"]); save_profile_state_wrapper(); st.success("Zur Watchlist")
                else:
                    st.info("Schon in Watchlist")

# ----------------------
# DAYTRADING PAGE
# ----------------------
elif page == "Daytrading":
    st.title("Daytrading — Candlestick Charts (klassisch)")
    asset_choice = st.selectbox("Asset", [a["id"]+" — "+a["name"] for a in ASSETS])
    asset_id = asset_choice.split(" — ")[0]
    tf = st.selectbox("Timeframe", list(TIMEFRAMES.keys()), index=list(TIMEFRAMES.keys()).index("5m"))
    periods = st.number_input("Anzahl Kerzen", min_value=20, max_value=2000, value=200, step=10)
    start_price = st.number_input("Startpreis (sim)", min_value=0.01, value=100.0, step=0.1)
    if st.button("Generiere Candles"):
        with st.spinner("Erzeuge Candles..."):
            candles = build_candles_for_asset(asset_id, timeframe_label=tf, periods=int(periods), start_price=float(start_price))
            plot_candles(candles, title=f"{asset_id} — {tf} ({periods} Kerzen)")
    st.markdown("---")
    st.write("Schnell: vordefinierte Buttons")
    row = st.columns(4)
    if row[0].button("1m, 200 Kerzen"):
        candles = build_candles_for_asset(asset_id, "1m", 200, start_price)
        plot_candles(candles, title=f"{asset_id} 1m")
    if row[1].button("5m, 200 Kerzen"):
        candles = build_candles_for_asset(asset_id, "5m", 200, start_price)
        plot_candles(candles, title=f"{asset_id} 5m")
    if row[2].button("30m, 200 Kerzen"):
        candles = build_candles_for_asset(asset_id, "30m", 200, start_price)
        plot_candles(candles, title=f"{asset_id} 30m")
    if row[3].button("1d, 200 Kerzen"):
        candles = build_candles_for_asset(asset_id, "1d", 200, start_price)
        plot_candles(candles, title=f"{asset_id} 1d")

# ----------------------
# PORTFOLIO
# ----------------------
elif page == "Portfolio":
    st.title("Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Portfolio leer.")
    else:
        st.markdown(f"Gesamtwert: {snap['total_value']:.2f} €  •  Eingezahlt: {snap['total_cost']:.2f} €")
        for r in snap["rows"]:
            item = r["item"]
            with st.expander(f"{item['name']} — {r['value']:.2f} € ({r['pnl']:+.2f} €)"):
                st.write(f"Menge: {item['qty']} • Kaufpreis: {item['buy_price']:.2f} €")
                # enable small candlestick preview: build last 120 candles 5m
                candles = build_candles_for_asset(item["asset_id"], timeframe_label="5m", periods=120, start_price=item.get("buy_price",100.0))
                if PLOTLY_AVAILABLE:
                    plot_candles(candles, title=f"{item['name']} — 5m preview")
                else:
                    st.line_chart({c["t"]: c["close"] for c in candles})
                if st.button("Entfernen", key=f"rm_{item['id']}"):
                    remove_position(item['id'])

# ----------------------
# SIMULATION (historical overview)
# ----------------------
elif page == "Simulation":
    st.title("Simulation")
    days = st.selectbox("Zeitraum (Tage)", [7,30,90,180,365], index=2)
    # produce aggregated daily value from minute series (safe)
    def simulate_portfolio_daily(days):
        snap = portfolio_snapshot()
        if not snap["rows"]:
            return []
        dates = [(datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat() for i in range(days)]
        combined = [0.0] * days
        for r in snap["rows"]:
            asset = r["item"]
            minutes = days * 1440
            series_min = generate_minute_prices(asset["asset_id"], minutes=minutes, start_price=asset.get("buy_price",100.0))
            # take each day's last minute as close
            for i in range(days):
                idx = (i+1)*1440 - 1
                if idx < len(series_min):
                    combined[i] += series_min[idx]["price"] * asset.get("qty",0.0)
        return [{"date": dates[i], "value": round(combined[i],4)} for i in range(days)]
    hist = simulate_portfolio_daily(days)
    if hist:
        st.line_chart({h["date"]: h["value"] for h in hist})
    else:
        st.info("Keine Daten — Portfolio leer.")

# ----------------------
# SETTINGS
# ----------------------
elif page == "Einstellungen":
    st.title("Einstellungen")
    st.subheader("Theme")
    curr = st.session_state.settings.get("theme","Dark Black")
    choice = st.selectbox("Wähle Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(curr))
    if st.button("Anwenden"):
        st.session_state.settings["theme"] = choice
        apply_theme_css(choice)
        save_profile_state_wrapper()
        st.experimental_rerun()
    st.markdown("---")
    st.subheader("App Optionen")
    goal = st.number_input("Finanzziel", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Speichere Einstellungen"):
        st.session_state.settings["goal"] = float(goal)
        save_profile_state_wrapper()
        st.success("Gespeichert")

# ----------------------
# Final save & footer
# ----------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Candlestick-Simulation • Classic colors (green up, red down) • Optional: install plotly for best charts</div>", unsafe_allow_html=True)
save_profile_state_wrapper()
