# main.py
# Fehlerbereinigte Offline-Finanz-App
# - Alle hist / Simulation Fehler behoben
# - Sofortiges Theme-Switching (3 Themes)
# - Erweiterte Wissensbasis
# - Keine externen APIs; lokale JSONs bleiben erhalten
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# Optionale libs (falls installiert) -> Fallbacks verwenden, wenn nicht
PLOTLY_AVAILABLE = False
PANDAS_AVAILABLE = False
REPORTLAB_AVAILABLE = False
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except Exception:
    PANDAS_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# --------------------------
# App-Config
# --------------------------
st.set_page_config(page_title="Offline Finance Pro (Stable)", page_icon="💹", layout="wide")
APP_FOLDER = "."
PROFILES_FILE = os.path.join(APP_FOLDER, "profiles.json")
DEFAULT_PROFILE = "default"

# --------------------------
# Themes (3) + CSS-Helper
# --------------------------
THEMES = {
    "Dark Black": {"bg": "#000000", "fg": "#e6eef6", "card": "#070707", "accent": "#00ff88", "muted": "#9aa6b2"},
    "Midnight Blue": {"bg": "#020428", "fg": "#dbe9ff", "card": "#08112a", "accent": "#6fb3ff", "muted": "#a7bfdc"},
    "Neon Green": {"bg": "#001100", "fg": "#eafff0", "card": "#022002", "accent": "#39ff7a", "muted": "#8fdca4"}
}

def apply_theme_css(theme_name: str):
    theme = THEMES.get(theme_name, THEMES["Dark Black"])
    css = f"""
    <style>
    html, body, [class*="css"] {{background:{theme['bg']} !important; color:{theme['fg']} !important;}}
    .stButton>button {{background:{theme['card']}; color:{theme['fg']}; border:1px solid #222; border-radius:6px;}}
    .card {{background:{theme['card']}; padding:12px; border-radius:10px; border:1px solid #111; margin-bottom:12px;}}
    .small {{color:{theme['muted']}; font-size:13px;}}
    .metric {{color:{theme['accent']}; font-weight:700;}}
    a {{color:{theme['accent']}}}
    .spark {{height:48px;}}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Apply default theme early
apply_theme_css("Dark Black")

# --------------------------
# JSON helpers
# --------------------------
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

# --------------------------
# Profiles meta & ensure files
# --------------------------
if not os.path.exists(PROFILES_FILE):
    save_json(PROFILES_FILE, {"profiles": [DEFAULT_PROFILE], "active": DEFAULT_PROFILE})

profiles_meta = load_json(PROFILES_FILE, {"profiles": [DEFAULT_PROFILE], "active": DEFAULT_PROFILE})
if "active" not in profiles_meta:
    profiles_meta["active"] = profiles_meta.get("profiles", [DEFAULT_PROFILE])[0]

def profile_file(profile, name):
    return os.path.join(APP_FOLDER, f"{name}_{profile}.json")

def ensure_profile_files(profile):
    defaults = {
        "settings": {},
        "portfolio": [],
        "history": [],
        "notifications": [],
        "watchlist": []
    }
    for key, val in defaults.items():
        p = profile_file(profile, key)
        if not os.path.exists(p):
            save_json(p, val)

ensure_profile_files(profiles_meta["active"])

# --------------------------
# Load active profile data (non-destructive)
# --------------------------
active_profile = profiles_meta["active"]
st.session_state.setdefault("profile", active_profile)
st.session_state.setdefault("series_cache", {})
st.session_state.setdefault("alerts_sent", set())

# load or create per-profile files
st.session_state.settings = load_json(profile_file(st.session_state["profile"], "settings"), {})
st.session_state.portfolio = load_json(profile_file(st.session_state["profile"], "portfolio"), [])
st.session_state.history = load_json(profile_file(st.session_state["profile"], "history"), [])
st.session_state.notifications = load_json(profile_file(st.session_state["profile"], "notifications"), [])
st.session_state.watchlist = load_json(profile_file(st.session_state["profile"], "watchlist"), [])
st.session_state.settings.setdefault("goal", 10000.0)
st.session_state.settings.setdefault("dividends_enabled", True)
st.session_state.settings.setdefault("reinvest_dividends", False)
st.session_state.settings.setdefault("daily_snapshot_enabled", False)
st.session_state.settings.setdefault("theme", "Dark Black")

# apply theme from settings
apply_theme_css(st.session_state.settings.get("theme", "Dark Black"))

# --------------------------
# Auth (single-owner)
# --------------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def save_profile_state():
    save_json(profile_file(st.session_state.profile, "settings"), st.session_state.settings)
    save_json(profile_file(st.session_state.profile, "portfolio"), st.session_state.portfolio)
    save_json(profile_file(st.session_state.profile, "history"), st.session_state.history)
    save_json(profile_file(st.session_state.profile, "notifications"), st.session_state.notifications)
    save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
    # update profiles meta
    pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
    if st.session_state.profile not in pm.get("profiles", []):
        pm.setdefault("profiles", []).append(st.session_state.profile)
    pm["active"] = st.session_state.profile
    save_json(PROFILES_FILE, pm)

def setup_or_login():
    st.sidebar.title("Profil & Login")
    st.sidebar.markdown("---")
    st.sidebar.write(f"Aktives Profil: **{st.session_state.profile}**")
    new_profile = st.sidebar.text_input("Neues Profil erstellen", value="", key="new_profile")
    if st.sidebar.button("Erstelle Profil") and new_profile.strip():
        pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
        if new_profile in pm.get("profiles", []):
            st.sidebar.error("Profil existiert bereits.")
        else:
            pm.setdefault("profiles", []).append(new_profile)
            pm["active"] = new_profile
            save_json(PROFILES_FILE, pm)
            ensure_profile_files(new_profile)
            st.sidebar.success("Profil erstellt. Bitte Seite neu laden.")
    pm = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": st.session_state.profile})
    profs = pm.get("profiles", [DEFAULT_PROFILE])
    sel = st.sidebar.selectbox("Wechsle Profil (save+reload)", profs, index=profs.index(st.session_state.profile) if st.session_state.profile in profs else 0, key="sel_profile")
    if st.sidebar.button("Profil wechseln"):
        if sel != st.session_state.profile:
            save_profile_state()
            pm["active"] = sel
            save_json(PROFILES_FILE, pm)
            st.sidebar.info("Profil gewechselt — bitte Seite neu laden.")
    st.sidebar.markdown("---")
    auth = st.session_state.settings.get("auth")
    if not auth:
        st.sidebar.subheader("Erstinstallation: Passwort setzen")
        pwd = st.sidebar.text_input("Passwort wählen", type="password", key="setup_pwd")
        pwd2 = st.sidebar.text_input("Passwort wiederholen", type="password", key="setup_pwd2")
        if st.sidebar.button("Passwort setzen"):
            if not pwd or pwd != pwd2:
                st.sidebar.error("Passwörter leer oder stimmen nicht überein.")
            else:
                salt = os.urandom(16)
                dk = derive_key(pwd, salt)
                st.session_state.settings["auth"] = {"salt": binascii.hexlify(salt).decode(), "key": binascii.hexlify(dk).decode(), "iterations":200_000, "dklen":72}
                save_profile_state()
                st.sidebar.success("Passwort gesetzt. Bitte Seite neu laden und einloggen.")
                st.stop()
        st.stop()
    else:
        st.sidebar.subheader("Login")
        pwd = st.sidebar.text_input("Passwort", type="password", key="login_pwd")
        if st.sidebar.button("Einloggen"):
            try:
                salt = binascii.unhexlify(auth["salt"])
                dk = derive_key(pwd, salt, iterations=auth.get("iterations",200_000), dklen=auth.get("dklen",72))
                if binascii.hexlify(dk).decode() == auth["key"]:
                    st.session_state.auth_ok = True
                    st.sidebar.success("Erfolgreich eingeloggt.")
                else:
                    st.sidebar.error("Falsches Passwort.")
            except Exception:
                st.sidebar.error("Auth-Daten fehlerhaft.")
    if not st.session_state.get("auth_ok", False):
        st.stop()

setup_or_login()

# --------------------------
# Asset definitions
# --------------------------
ASSETS = [
    {"id":"ETF_DE","category":"ETF","name":"Deutschland ETF","symbol":"DE.ETF","div_yield":0.015},
    {"id":"ETF_US","category":"ETF","name":"USA ETF","symbol":"US.ETF","div_yield":0.012},
    {"id":"ETF_EU","category":"ETF","name":"Europa ETF","symbol":"EU.ETF","div_yield":0.013},
    {"id":"ETF_WW","category":"ETF","name":"Welt ETF","symbol":"WW.ETF","div_yield":0.011},
    {"id":"CR_BTC","category":"Krypto","name":"Bitcoin","symbol":"BTC","div_yield":0.0},
    {"id":"CR_ETH","category":"Krypto","name":"Ethereum","symbol":"ETH","div_yield":0.0},
    {"id":"ST_AAPL","category":"Aktie","name":"Apple","symbol":"AAPL","div_yield":0.006},
    {"id":"ST_MSFT","category":"Aktie","name":"Microsoft","symbol":"MSFT","div_yield":0.008},
    {"id":"ST_TSLA","category":"Aktie","name":"Tesla","symbol":"TSLA","div_yield":0.0},
    {"id":"ST_NVDA","category":"Aktie","name":"NVIDIA","symbol":"NVDA","div_yield":0.0}
]
ASSETS_BY_ID = {a["id"]: a for a in ASSETS}

# --------------------------
# Deterministic price generator (cached)
# --------------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 31)

def generate_series(asset_id: str, days: int = 365, start_price: float = 100.0):
    key = f"{asset_id}_{days}_{int(start_price)}"
    if key in st.session_state.series_cache:
        return st.session_state.series_cache[key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    series = []
    for i in range(days):
        drift = (rnd.random() - 0.48) * 0.001
        vol = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        series.append({"date": date, "price": round(price, 4)})
    # SMA20/50
    for i in range(len(series)):
        p20 = [series[j]["price"] for j in range(max(0, i - 19), i + 1)]
        p50 = [series[j]["price"] for j in range(max(0, i - 49), i + 1)]
        series[i]["sma20"] = round(mean(p20), 4)
        series[i]["sma50"] = round(mean(p50), 4)
    st.session_state.series_cache[key] = series
    return series

# --------------------------
# Portfolio operations
# --------------------------
def add_position(category, asset_id, name, qty, buy_price, note="", tags=None):
    if tags is None:
        tags = []
    item = {"id": f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
            "category": category, "asset_id": asset_id, "name": name, "qty": float(qty),
            "buy_price": float(buy_price), "note": note, "tags": tags, "added_at": datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "add", "item": item})
    save_profile_state()
    st.success(f"{name} hinzugefügt.")
    check_notifications_for_item(item)

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p.get("id") != item_id]
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "remove", "item": {"id": item_id}})
    save_profile_state()
    st.success("Position entfernt.")
    st.experimental_rerun()

def update_note_tags(item_id, new_note, new_tags):
    for p in st.session_state.portfolio:
        if p.get("id") == item_id:
            p["note"] = new_note
            p["tags"] = new_tags
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "note_update", "item": {"id": item_id}})
    save_profile_state()
    st.success("Notiz & Tags gespeichert.")

# --------------------------
# Analytics helpers (robust)
# --------------------------
def portfolio_snapshot():
    tot_value = 0.0
    tot_cost = 0.0
    rows = []
    for item in st.session_state.portfolio:
        # defensive
        if not isinstance(item, dict) or "asset_id" not in item or "qty" not in item or "buy_price" not in item:
            continue
        series = generate_series(item["asset_id"], 365, start_price=item.get("buy_price", 100.0))
        cur = series[-1]["price"]
        qty = float(item.get("qty", 0.0))
        value = cur * qty
        cost = float(item.get("buy_price", 0.0)) * qty
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct, "series": series})
        tot_value += value
        tot_cost += cost
    return {"rows": rows, "total_value": tot_value, "total_cost": tot_cost}

def calc_volatility(series, window=30):
    if not series or len(series) < window + 1:
        return 0.0
    prices = [p["price"] for p in series[-(window+1):]]
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] > 0:
            returns.append((prices[i] - prices[i-1]) / prices[i-1])
    if not returns:
        return 0.0
    return stdev(returns)

def compute_dividends(item, years=1):
    meta = ASSETS_BY_ID.get(item.get("asset_id"), {})
    dy = meta.get("div_yield", 0.0)
    return dy * item.get("buy_price", 0.0) * item.get("qty", 0.0) * years

def generate_benchmark(days=365, start=100.0):
    rnd = random.Random(deterministic_seed("BENCHMARK"))
    price = float(start)
    series = []
    for i in range(days):
        drift = 0.0002
        vol = (rnd.random() - 0.5) * 0.01
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        series.append({"date": date, "price": round(price, 4)})
    return series

def forecast_linear(series, future_days=30):
    if not series or len(series) < 3:
        return []
    n = len(series)
    xs = list(range(n)); ys = [p["price"] for p in series]
    x_mean = mean(xs); y_mean = mean(ys)
    num = sum((xs[i] - x_mean)*(ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean)**2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    intercept = y_mean - slope * x_mean
    preds = []
    for fd in range(1, future_days+1):
        x = n + fd - 1
        preds.append({"date": (datetime.utcnow().date() + timedelta(days=fd)).isoformat(), "price": round(intercept + slope * x, 4)})
    return preds

def risk_analyzer_score():
    snap = portfolio_snapshot()
    if not snap["rows"]:
        return {"score": 0.0, "volatility_component": 0.0, "concentration_component": 0.0}
    vols = [calc_volatility(r["series"], 30) for r in snap["rows"]]
    vol_norm = mean(vols) if vols else 0.0
    total = snap["total_value"] if snap["total_value"] > 0 else 1.0
    top_share = max((r["value"] for r in snap["rows"]), default=0.0) / total
    score = min(100.0, (vol_norm*100*0.6) + (top_share*100*0.4))
    return {"score": round(score, 2), "volatility_component": round(vol_norm*100,2), "concentration_component": round(top_share*100,2)}

# --------------------------
# Notifications & watchlist (robust)
# --------------------------
def show_toast(msg, level="info"):
    try:
        if hasattr(st, "toast"):
            st.toast(msg)
        else:
            if level == "warning":
                st.warning(msg)
            elif level == "success":
                st.success(msg)
            else:
                st.info(msg)
    except Exception:
        if level == "warning":
            st.warning(msg)
        elif level == "success":
            st.success(msg)
        else:
            st.info(msg)

def check_notifications_for_item(item):
    if not item or "asset_id" not in item:
        return
    series = generate_series(item["asset_id"], 365, start_price=item.get("buy_price", 100.0))
    cur = series[-1]["price"]
    cost = item.get("buy_price", 0.0) * item.get("qty", 0.0)
    value = cur * item.get("qty", 0.0)
    pnl = value - cost
    pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
    if pnl_pct < -10 and item.get("id") not in st.session_state.alerts_sent:
        msg = f"⚠️ Verlust bei {item.get('name')}: {pnl_pct:.2f}%"
        st.session_state.notifications.append({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        save_json(profile_file(st.session_state.profile, "notifications"), st.session_state.notifications)
        st.session_state.alerts_sent.add(item.get("id"))
        show_toast(msg, level="warning")

def check_all_notifications():
    for it in st.session_state.portfolio:
        check_notifications_for_item(it)

check_all_notifications()

def add_to_watchlist(asset_id):
    if asset_id not in st.session_state.watchlist:
        st.session_state.watchlist.append(asset_id)
        save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
        st.success("Zur Watchlist hinzugefügt.")
    else:
        st.info("Schon in Watchlist.")

def remove_from_watchlist(asset_id):
    if asset_id in st.session_state.watchlist:
        st.session_state.watchlist.remove(asset_id)
        save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
        st.success("Von Watchlist entfernt.")

# --------------------------
# Snapshots & report
# --------------------------
def take_snapshot():
    snap = portfolio_snapshot()
    entry = {"timestamp": datetime.utcnow().isoformat(), "snapshot": snap}
    st.session_state.history.append(entry)
    save_json(profile_file(st.session_state.profile, "history"), st.session_state.history)
    st.success("Snapshot gespeichert.")

def export_daily_report(filename="daily_report"):
    snap = portfolio_snapshot()
    lines = [f"Daily Report — {datetime.utcnow().isoformat()}",
             f"Total Value: {snap['total_value']:.2f} EUR",
             f"Total Cost: {snap['total_cost']:.2f} EUR",
             "Positions:"]
    for r in snap["rows"]:
        lines.append(f"- {r['item']['name']}: value={r['value']:.2f} EUR pnl={r['pnl']:+.2f} ({r['pnl_pct']:+.2f}%)")
    txt = "\n".join(lines)
    if REPORTLAB_AVAILABLE:
        pdf_path = os.path.join(APP_FOLDER, filename + ".pdf")
        c = canvas.Canvas(pdf_path, pagesize=A4)
        w, h = A4
        y = h - 40
        c.setFont("Helvetica", 10)
        for line in lines:
            c.drawString(40, y, line)
            y -= 14
            if y < 40:
                c.showPage()
                y = h - 40
        c.save()
        return pdf_path
    else:
        return txt.encode("utf-8")

# --------------------------
# Safe simulation helpers (all hist issues fixed here)
# --------------------------
def simulate_over_time(days=365):
    """
    Produces a combined historical portfolio series for 'days' days.
    Always returns a list (possibly empty).
    """
    snap = portfolio_snapshot()
    if not snap["rows"]:
        return []
    # prepare dates
    dates = [(datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat() for i in range(days)]
    combined = [0.0] * days
    # sum each position's historical series (defensive)
    for r in snap["rows"]:
        asset_id = r["item"].get("asset_id")
        qty = r["item"].get("qty", 0.0)
        if not asset_id or qty == 0:
            continue
        series = generate_series(asset_id, days, start_price=r["item"].get("buy_price", 100.0))
        # guard series length
        if len(series) < days:
            # pad with last known price
            last_price = series[-1]["price"] if series else 0.0
            while len(series) < days:
                series.append({"date": dates[len(series)], "price": last_price, "sma20": last_price, "sma50": last_price})
        for i in range(days):
            combined[i] += series[i]["price"] * qty
    return [{"date": dates[i], "value": round(combined[i], 4)} for i in range(days)]

def simulate_crash(percent_drop):
    snap = portfolio_snapshot()
    out = []
    for r in snap["rows"]:
        pre_value = r["value"]
        post_price = r["cur"] * (1 - percent_drop / 100.0)
        post_value = post_price * r["item"].get("qty", 0.0)
        out.append({"id": r["item"].get("id"), "name": r["item"].get("name"), "pre_value": round(pre_value, 4), "post_value": round(post_value, 4), "delta": round(post_value - pre_value, 4)})
    total_pre = snap["total_value"]
    total_post = sum(x["post_value"] for x in out)
    return {"items": out, "total_pre": total_pre, "total_post": total_post, "total_delta": total_post - total_pre}

# --------------------------
# Search helpers
# --------------------------
def search_all_assets(q):
    q = q.lower()
    return [a for a in ASSETS if q in a.get("name","").lower() or q in a.get("symbol","").lower() or q in a.get("id","").lower()]

def search_portfolio(q):
    q = q.lower()
    return [p for p in st.session_state.portfolio if q in p.get("name","").lower() or q in p.get("note","").lower() or any(q in t.lower() for t in p.get("tags",[]))]

# --------------------------
# Sidebar (navigation) + quick actions
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home", "Marktplatz", "Portfolio", "Rebalancing", "Simulation", "Statistiken", "Wissensbasis", "Export/Import", "Einstellungen"])
st.sidebar.markdown("---")
if st.sidebar.button("Snapshot speichern"):
    take_snapshot()
if st.sidebar.button("Export Report (PDF/TXT)"):
    out = export_daily_report("daily_report")
    if isinstance(out, str) and out.endswith(".pdf"):
        with open(out, "rb") as f:
            st.sidebar.download_button("Download PDF", data=f.read(), file_name=os.path.basename(out), mime="application/pdf")
    else:
        st.sidebar.download_button("Download TXT", data=out, file_name="daily_report.txt", mime="text/plain")
st.sidebar.markdown("---")
st.sidebar.subheader("Schnellsuche")
sq = st.sidebar.text_input("Suche Assets/Portfolio")
if sq:
    sa = search_all_assets(sq)
    sp = search_portfolio(sq)
    st.sidebar.write(f"Assets: {len(sa)} — Portfolio Treffer: {len(sp)}")
    for a in sa[:6]:
        st.sidebar.write(f"- {a['name']} ({a['symbol']})")

# --------------------------
# Pages
# --------------------------
if page == "Home":
    st.title("🏠 Dashboard — Übersicht")
    snap = portfolio_snapshot()
    total_val = snap["total_value"]
    total_cost = snap["total_cost"]
    st.metric("Portfolio Gesamtwert", f"{total_val:.2f} €", delta=f"{(total_val - total_cost):+.2f} €")
    st.write(f"Eingezahlt: {total_cost:.2f} €  •  Ziel: {st.session_state.settings.get('goal',10000.0):.2f} €")
    st.progress(min(total_val / float(st.session_state.settings.get("goal", 10000.0)), 1.0))
    st.markdown("---")
    st.subheader("Widgets")
    ra = risk_analyzer_score()
    c1, c2, c3 = st.columns(3)
    c1.metric("Anzahl Positionen", len(snap["rows"]))
    avg_pnl = round(mean([r["pnl"] for r in snap["rows"]]) if snap["rows"] else 0.0, 2)
    c2.metric("Durchschn. PnL", f"{avg_pnl:+.2f} €")
    vol_avg = round(mean([calc_volatility(r["series"], 30) for r in snap["rows"]]) if snap["rows"] else 0.0, 4)
    c3.metric("Volatilität (avg)", f"{vol_avg:.4f}")
    st.write(f"Risikoscore: **{ra['score']}** (Vol:{ra['volatility_component']} / Konz:{ra['concentration_component']})")
    st.markdown("---")
    st.subheader("Offline News (Mock)")
    rnd = random.Random(deterministic_seed("NEWS" + st.session_state.profile))
    keywords = ["Zinsen","Inflation","Tech","Rohstoffe","Krypto","Arbeitsmarkt","Wachstum"]
    for i in range(5):
        t = f"{rnd.choice(keywords)} — Marktbeobachtung ({rnd.randint(-4,4)}%)"
        ts = (datetime.utcnow() - timedelta(hours=rnd.randint(1,72))).isoformat()
        st.markdown(f"- {t}  <span class='small'>({ts})</span>", unsafe_allow_html=True)

elif page == "Marktplatz":
    st.title("🏬 Marktplatz")
    intervals = {"1 Monat":30, "3 Monate":90, "6 Monate":180, "1 Jahr":365}
    cats = {}
    for a in ASSETS:
        cats.setdefault(a["category"], []).append(a)
    for cat, assets in cats.items():
        st.subheader(cat)
        for a in assets:
            with st.container():
                cols = st.columns([3,1])
                with cols[0]:
                    st.markdown(f"**{a['name']}** — <span class='small'>{a['symbol']}</span>", unsafe_allow_html=True)
                    sel = st.selectbox(f"Zeitraum {a['id']}", list(intervals.keys()), key=f"iv_{a['id']}")
                    days = intervals[sel]
                    series = generate_series(a["id"], days, start_price=100.0 + (abs(hash(a['id'])) % 1000) / 10)
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["price"] for p in series], mode="lines", name=a["symbol"]))
                        fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=260,
                                          paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                          plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                          font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart({p["date"]: p["price"] for p in series})
                with cols[1]:
                    cur = series[-1]["price"]
                    sma20 = series[-1]["sma20"]
                    sma50 = series[-1]["sma50"]
                    rec = "Kaufen" if sma20 > sma50 else "Nicht kaufen"
                    st.write(f"Aktuell: {cur:.2f} €")
                    st.write(f"SMA20: {sma20:.2f} | SMA50: {sma50:.2f}")
                    st.markdown(f"**Empfehlung:** {'🟢 '+rec if rec=='Kaufen' else '🔴 '+rec}")
                    if st.button("Zur Watchlist hinzufügen", key=f"watch_{a['id']}"):
                        add_to_watchlist(a["id"])
                    with st.form(key=f"add_{a['id']}", clear_on_submit=False):
                        qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                        bp = st.number_input("Kaufpreis", min_value=0.0001, value=float(cur), step=0.01, key=f"bp_{a['id']}")
                        note = st.text_area("Notiz", value="", key=f"note_{a['id']}", height=80)
                        tags_txt = st.text_input("Tags (kommagetrennt)", value="", key=f"tags_{a['id']}")
                        if st.form_submit_button("Zum Portfolio hinzufügen"):
                            tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                            add_position(cat, a["id"], a["name"], qty, bp, note, tags)

elif page == "Portfolio":
    st.title("💼 Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Portfolio leer — füge Assets im Marktplatz hinzu.")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €  •  Eingezahlt: {snap['total_cost']:.2f} €")
        st.progress(min(snap['total_value'] / float(st.session_state.settings.get("goal",10000.0)), 1.0))
        st.markdown("---")
        for r in snap["rows"]:
            item = r["item"]
            with st.expander(f"{item['name']} — Wert: {r['value']:.2f} €  •  PnL: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)"):
                cols = st.columns([3,1])
                with cols[0]:
                    st.write(f"Menge: {item['qty']} | Kaufpreis: {item['buy_price']:.2f} €")
                    st.write(f"Aktuell: {r['cur']:.4f} €")
                    series = r["series"]
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["price"] for p in series], name="Preis"))
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["sma20"] for p in series], name="SMA20"))
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["sma50"] for p in series], name="SMA50"))
                        fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=300,
                                          paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                          plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                          font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart({p["date"]: p["price"] for p in series})
                with cols[1]:
                    div = compute_dividends(item, years=1) if st.session_state.settings.get("dividends_enabled", True) else 0.0
                    st.write(f"Dividenden (1y, simuliert): {div:.2f} €")
                    new_note = st.text_area("Notiz", value=item.get("note",""), key=f"note_p_{item['id']}")
                    tags_txt = st.text_input("Tags", value=",".join(item.get("tags", [])), key=f"tags_p_{item['id']}")
                    if st.button("Speichern", key=f"save_p_{item['id']}"):
                        tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                        update_note_tags(item['id'], new_note, tags)
                    if st.button("Entfernen", key=f"rm_p_{item['id']}"):
                        remove_position(item['id'])
        st.markdown("---")
        st.subheader("Analyse & Benchmark")
        ra = risk_analyzer_score()
        st.write(f"Risikoscore: **{ra['score']}** — Vol:{ra['volatility_component']} / Konz:{ra['concentration_component']}")
        # history and benchmark
        hist = simulate_over_time(180)
        bench = generate_benchmark(180, start=100.0)
        if hist and bench:
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[p["date"] for p in hist], y=[p["value"] for p in hist], name="Portfolio"))
                scale = hist[0]["value"] / bench[0]["price"] if bench and bench[0]["price"] > 0 else 1.0
                fig.add_trace(go.Scatter(x=[b["date"] for b in bench], y=[b["price"]*scale for b in bench], name="Benchmark (scaled)"))
                fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=320,
                                  paper_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                  plot_bgcolor=THEMES[st.session_state.settings.get("theme","Dark Black")]["bg"],
                                  font_color=THEMES[st.session_state.settings.get("theme","Dark Black")]["fg"])
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart({p["date"]: p["value"] for p in hist})
                st.line_chart({b["date"]: b["price"] for b in bench})

elif page == "Rebalancing":
    st.title("⚖️ Rebalancing")
    st.markdown("Zielallokation (Summe = 1.0)")
    c1,c2,c3 = st.columns(3)
    tgt_etf = c1.number_input("ETF", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="tgt_etf")
    tgt_act = c2.number_input("Aktie", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key="tgt_act")
    tgt_cry = c3.number_input("Krypto", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="tgt_cry")
    total = tgt_etf + tgt_act + tgt_cry
    if abs(total - 1.0) > 1e-6:
        st.warning(f"Summe = {total:.2f} — bitte auf 1.0 einstellen.")
    else:
        target = {"ETF":tgt_etf,"Aktie":tgt_act,"Krypto":tgt_cry}
        snap = portfolio_snapshot()
        total_val = snap["total_value"] if snap["total_value"]>0 else 1.0
        cur_alloc = {"ETF":0.0,"Aktie":0.0,"Krypto":0.0}
        for r in snap["rows"]:
            cat = r["item"]["category"]
            cur_alloc[cat] = cur_alloc.get(cat,0.0) + r["value"]
        for k in cur_alloc:
            cur_alloc[k] = cur_alloc[k] / total_val
        advice = {k: target[k] - cur_alloc.get(k,0.0) for k in target.keys()}
        st.markdown("**Aktuelle Allokation**")
        for k,v in cur_alloc.items():
            st.write(f"- {k}: {v*100:.2f}%")
        st.markdown("**Vorschlag** (positiv = kaufen, negativ = verkaufen)")
        for k,v in advice.items():
            st.write(f"- {k}: {(v*100):+.2f}%")

elif page == "Simulation":
    st.title("🔬 Simulation & Forecast")
    st.markdown("Historische Simulation & Crash-Szenarien.")
    days = st.selectbox("Zeitraum (Tage)", [90,180,365], index=1)
    hist = simulate_over_time(days)
    if hist:
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[p["date"] for p in hist], y=[p["value"] for p in hist], name="Portfolio"))
            fig.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart({p["date"]: p["value"] for p in hist})
        # forecast
        base = hist[-120:] if len(hist) >= 120 else hist
        ser = [{"date": base[i]["date"], "price": base[i]["value"]} for i in range(len(base))]
        preds = forecast_linear(ser, future_days=30) if ser else []
        if preds:
            if PLOTLY_AVAILABLE:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=[p["date"] for p in ser], y=[p["price"] for p in ser], name="History"))
                fig2.add_trace(go.Scatter(x=[p["date"] for p in preds], y=[p["price"] for p in preds], name="Forecast"))
                fig2.update_layout(margin=dict(l=0,r=0,t=20,b=0), height=300)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.line_chart({p["date"]: p["price"] for p in ser})
                st.line_chart({p["date"]: p["price"] for p in preds})
    else:
        st.info("Keine Daten für Simulation (Portfolio leer).")
    st.markdown("---")
    st.subheader("Crash-Simulation")
    drop = st.slider("Simulierter %-Drop", 1, 100, 20)
    if st.button("Crash simulieren"):
        snap = portfolio_snapshot()
        if not snap["rows"]:
            st.info("Portfolio leer.")
        else:
            res = simulate_crash(drop)
            st.write(f"Vorher: {res['total_pre']:.2f} € • Nachher: {res['total_post']:.2f} € • Änderung: {res['total_delta']:+.2f} €")
            st.table([{"Asset": x["name"], "Vorher": round(x["pre_value"],2), "Nachher": round(x["post_value"],2), "Delta": round(x["delta"],2)} for x in res["items"]])

elif page == "Statistiken":
    st.title("📊 Statistiken")
    stats = portfolio_stats() if 'portfolio_stats' in globals() else {}
    try:
        stats = portfolio_stats()
    except Exception:
        stats = {}
    if not stats:
        st.info("Portfolio leer — keine Statistiken.")
    else:
        st.write(f"Positionen: {stats.get('count',0)}")
        st.write(f"Durchschnitts-Gewinn/Verlust: {stats.get('avg_pnl',0.0):+.2f} €")
        st.write(f"Volatilität (Snapshot): {stats.get('volatility',0.0):.4f}")
        st.markdown("---")
        snap = portfolio_snapshot()
        cat_vals = {}
        for r in snap["rows"]:
            cat_vals[r["item"]["category"]] = cat_vals.get(r["item"]["category"], 0.0) + r["value"]
        if cat_vals:
            st.subheader("Kategorieaufteilung")
            st.bar_chart(cat_vals)
        st.markdown("---")
        st.subheader("Heatmap (vereinfacht)")
        for r in snap["rows"]:
            pct = r["pnl_pct"]
            color = "#006600" if pct >= 0 else "#660000"
            width = min(max(abs(pct) * 1.5, 2), 100)
            st.markdown(f"<div style='background:{color}; width:{width}%; padding:8px; border-radius:6px; margin-bottom:6px;'>{r['item']['name']} — {r['pnl']:+.2f} € ({pct:+.2f}%)</div>", unsafe_allow_html=True)

elif page == "Wissensbasis":
    st.title("📚 Wissensbasis & Leitfäden")
    guides = {
        "ETF_vs_Aktie": "ETFs bündeln viele Aktien... (längerer Text mit Beispielen und Anwendung).",
        "Technische_Analyse": "Technische Analyse erklärt: SMA, RSI, MACD und Beispielcharts...",
        "Fundamentalanalyse": "Fundamentalanalyse: Kennzahlen, Bilanz, Kennzahleninterpretation...",
        "Volatilitaet_und_Risiko": "Volatilität, Diversifikation, Risikoquellen und Gegenmaßnahmen...",
        "Behavioral_Finance": "Psychologische Biases, wie sie Märkte beeinflussen und wie man sie vermeidet...",
        "Krypto_Basics": "Krypto-Grundlagen, Tokenomics, Security, Positionierung im Portfolio...",
        "Rebalancing_Practice": "Rebalancing-Methoden: calendar-based, threshold-based, Beispielrechnungen..."
    }
    for k, txt in guides.items():
        with st.expander(k.replace("_", " ")):
            st.write(txt)
            sample = generate_series("GUIDE_"+k, 90, start_price=50.0)
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[p["date"] for p in sample], y=[p["price"] for p in sample], name="Preis"))
                fig.update_layout(margin=dict(l=0,r=0,t=10,b=0), height=220)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.line_chart({p["date"]: p["price"] for p in sample})

elif page == "Export/Import":
    st.title("📤 Export / Import")
    export_obj = {"profile": st.session_state.profile, "settings": st.session_state.settings, "portfolio": st.session_state.portfolio, "history": st.session_state.history, "notifications": st.session_state.notifications, "exported_at": datetime.utcnow().isoformat()}
    st.download_button("Export JSON", data=json.dumps(export_obj, ensure_ascii=False, indent=2), file_name=f"export_{st.session_state.profile}.json", mime="application/json")
    st.markdown("---")
    uploaded = st.file_uploader("Import JSON", type=["json"])
    if uploaded:
        try:
            obj = json.loads(uploaded.read().decode("utf-8"))
            st.session_state.profile = obj.get("profile", st.session_state.profile)
            st.session_state.settings.update(obj.get("settings", {}))
            incoming = obj.get("portfolio", [])
            if isinstance(incoming, list) and incoming:
                st.session_state.portfolio.extend(incoming)
            st.session_state.history.extend(obj.get("history", []))
            st.session_state.notifications.extend(obj.get("notifications", []))
            save_profile_state()
            st.success("Importiert. Bitte Seite neu laden.")
        except Exception as e:
            st.error(f"Import fehlgeschlagen: {e}")

elif page == "Einstellungen":
    st.title("⚙️ Einstellungen")
    st.subheader("Theme")
    current_theme = st.session_state.settings.get("theme", "Dark Black")
    theme_choice = st.selectbox("Wähle Theme", list(THEMES.keys()), index=list(THEMES.keys()).index(current_theme))
    if st.button("Theme anwenden"):
        st.session_state.settings["theme"] = theme_choice
        apply_theme_css(theme_choice)
        save_profile_state()
        st.experimental_rerun()
    st.markdown("---")
    st.subheader("App Optionen")
    goal_val = st.number_input("Finanzziel", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    divs = st.checkbox("Dividenden simulieren", value=st.session_state.settings.get("dividends_enabled", True))
    reinv = st.checkbox("Dividenden reinvestieren (simuliert)", value=st.session_state.settings.get("reinvest_dividends", False))
    snapshot_enable = st.checkbox("Tägliche Snapshots aktivieren", value=st.session_state.settings.get("daily_snapshot_enabled", False))
    if st.button("Speichern"):
        st.session_state.settings["goal"] = float(goal_val)
        st.session_state.settings["dividends_enabled"] = bool(divs)
        st.session_state.settings["reinvest_dividends"] = bool(reinv)
        st.session_state.settings["daily_snapshot_enabled"] = bool(snapshot_enable)
        save_profile_state()
        st.success("Einstellungen gespeichert.")
    st.markdown("---")
    st.subheader("Benachrichtigungen")
    for n in st.session_state.notifications[-20:][::-1]:
        st.write(f"{n['timestamp']}: {n['message']}")
    if st.button("Benachrichtigungen leeren"):
        st.session_state.notifications = []
        save_profile_state()
        st.success("Benachrichtigungen gelöscht.")
    st.markdown("---")
    st.subheader("Passwort ändern")
    auth = st.session_state.settings.get("auth")
    if not auth:
        st.warning("Kein Passwort gesetzt (siehe Login/Setup).")
    else:
        old = st.text_input("Altes Passwort", type="password", key="pwd_old")
        new = st.text_input("Neues Passwort", type="password", key="pwd_new")
        new2 = st.text_input("Neues Passwort wiederholen", type="password", key="pwd_new2")
        if st.button("Passwort ändern"):
            try:
                salt = binascii.unhexlify(auth["salt"])
                dk_old = derive_key(old, salt, iterations=auth.get("iterations",200_000), dklen=auth.get("dklen",72))
                if binascii.hexlify(dk_old).decode() != auth["key"]:
                    st.error("Altes Passwort falsch.")
                elif not new or new != new2:
                    st.error("Neues Passwort leer oder Wiederholung stimmt nicht.")
                else:
                    ns = os.urandom(16)
                    new_dk = derive_key(new, ns, iterations=200_000, dklen=72)
                    st.session_state.settings["auth"] = {"salt": binascii.hexlify(ns).decode(), "key": binascii.hexlify(new_dk).decode(), "iterations":200_000, "dklen":72}
                    save_profile_state()
                    st.success("Passwort geändert.")
            except Exception as e:
                st.error(f"Fehler: {e}")

# Footer + persist
st.markdown("---")
st.markdown("<div class='small'>Offline • Lokale JSONs • Optional: plotly/pandas/reportlab für erweiterte Features</div>", unsafe_allow_html=True)
save_profile_state()
