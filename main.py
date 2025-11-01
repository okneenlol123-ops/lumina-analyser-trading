# main.py
# Advanced Offline-Finanz-App — inkl. features requested (profiles, watchlist, dividends, PDF, plotly fallback)
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# Optional libraries (plotly, pandas, reportlab). Use fallbacks if not installed.
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

# ---------------------------
# Config / filenames
# ---------------------------
st.set_page_config(page_title="Offline Finance Pro", page_icon="💹", layout="wide")
APP_FOLDER = "."  # single-folder mode as requested
PROFILES_FILE = os.path.join(APP_FOLDER, "profiles.json")
DEFAULT_PROFILE = "default"

# Utility to resolve profile-specific filepaths
def profile_file(profile, name):
    return os.path.join(APP_FOLDER, f"{name}_{profile}.json")

# ---------------------------
# Styling & theme toggle
# ---------------------------
DEFAULT_CSS = """
<style>
html, body, [class*="css"] {background:#000 !important; color:#e6eef6 !important;}
.stButton>button {background:#111; color:#e6eef6; border:1px solid #222; border-radius:6px;}
.card {background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px;}
.small {color:#9aa6b2; font-size:13px;}
.toast {background:#222; color:#e6eef6; padding:8px; border-radius:6px; border:1px solid #444; margin-bottom:6px;}
</style>
"""
st.markdown(DEFAULT_CSS, unsafe_allow_html=True)

# ---------------------------
# Helpers: JSON load/save
# ---------------------------
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

# ---------------------------
# Profiles: multiple offline profiles
# ---------------------------
if not os.path.exists(PROFILES_FILE):
    save_json(PROFILES_FILE, {"profiles": [DEFAULT_PROFILE], "active": DEFAULT_PROFILE})

profiles_meta = load_json(PROFILES_FILE, {"profiles":[DEFAULT_PROFILE], "active": DEFAULT_PROFILE})
if "active" not in profiles_meta:
    profiles_meta["active"] = profiles_meta.get("profiles", [DEFAULT_PROFILE])[0]
active_profile = profiles_meta["active"]

# ensure profile-specific files exist
def ensure_profile_files(profile):
    defaults = {
        f"profile_settings": {},
        f"portfolio": [],
        f"history": [],
        f"notifications": [],
        f"watchlist": []
    }
    for key, val in defaults.items():
        pfile = profile_file(profile, key)
        if not os.path.exists(pfile):
            save_json(pfile, val)

ensure_profile_files(active_profile)

# load profile data
def load_profile(profile):
    settings = load_json(profile_file(profile, "profile_settings"), {})
    portfolio = load_json(profile_file(profile, "portfolio"), [])
    history = load_json(profile_file(profile, "history"), [])
    notifications = load_json(profile_file(profile, "notifications"), [])
    watchlist = load_json(profile_file(profile, "watchlist"), [])
    return {"settings": settings, "portfolio": portfolio, "history": history, "notifications": notifications, "watchlist": watchlist}

profile_data = load_profile(active_profile)

# Save profile back
def save_profile(profile, data):
    save_json(profile_file(profile, "profile_settings"), data.get("settings", {}))
    save_json(profile_file(profile, "portfolio"), data.get("portfolio", []))
    save_json(profile_file(profile, "history"), data.get("history", []))
    save_json(profile_file(profile, "notifications"), data.get("notifications", []))
    save_json(profile_file(profile, "watchlist"), data.get("watchlist", []))

# store in session
if "profile" not in st.session_state:
    st.session_state.profile = active_profile
    st.session_state.settings = profile_data["settings"]
    st.session_state.portfolio = profile_data["portfolio"]
    st.session_state.history = profile_data["history"]
    st.session_state.notifications = profile_data["notifications"]
    st.session_state.watchlist = profile_data["watchlist"]
    st.session_state.series_cache = {}
    st.session_state.alerts_sent = set()
    st.session_state.auth_ok = False
    st.session_state.dark_theme = st.session_state.settings.get("dark_theme", True)

# ---------------------------
# Optional: feature toggles
# ---------------------------
# set defaults if not present
st.session_state.settings.setdefault("goal", 10000.0)
st.session_state.settings.setdefault("dividends_enabled", True)
st.session_state.settings.setdefault("reinvest_dividends", False)
st.session_state.settings.setdefault("daily_snapshot_enabled", False)
st.session_state.settings.setdefault("dark_theme", True)

# ---------------------------
# Auth: single-owner per app (profile-level)
# ---------------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def ensure_auth_for_profile(profile):
    sett = st.session_state.settings
    if not sett.get("auth"):
        return False
    return True

# Setup / login UI
def setup_or_login():
    sett = st.session_state.settings
    if not sett.get("auth"):
        st.header("Erstinstallation: Eigentümer-Passwort setzen (Profil: %s)" % st.session_state.profile)
        pwd = st.text_input("Passwort wählen", type="password", key="setup_pwd")
        pwd2 = st.text_input("Passwort wiederholen", type="password", key="setup_pwd2")
        if st.button("Passwort setzen"):
            if not pwd or pwd != pwd2:
                st.error("Passwörter leer oder stimmen nicht überein.")
            else:
                salt = os.urandom(16)
                dk = derive_key(pwd, salt)
                sett["auth"] = {"salt": binascii.hexlify(salt).decode(), "key": binascii.hexlify(dk).decode(), "iterations": 200_000, "dklen": 72}
                st.session_state.settings = sett
                save_profile(st.session_state.profile, {
                    "settings": st.session_state.settings,
                    "portfolio": st.session_state.portfolio,
                    "history": st.session_state.history,
                    "notifications": st.session_state.notifications,
                    "watchlist": st.session_state.watchlist
                })
                st.success("Passwort gesetzt. Seite neu laden und einloggen.")
    else:
        st.header("Login (Profil: %s)" % st.session_state.profile)
        pwd = st.text_input("Passwort", type="password", key="login_pwd")
        if st.button("Einloggen"):
            auth = sett.get("auth")
            try:
                salt = binascii.unhexlify(auth["salt"])
                dk = derive_key(pwd, salt, iterations=auth.get("iterations", 200_000), dklen=auth.get("dklen", 72))
                if binascii.hexlify(dk).decode() == auth["key"]:
                    st.session_state.auth_ok = True
                    st.success("Erfolgreich eingeloggt.")
                else:
                    st.error("Falsches Passwort.")
            except Exception:
                st.error("Auth-Daten fehlerhaft.")
    # stop if not logged in
    if not st.session_state.auth_ok:
        st.stop()

# run auth
setup_or_login()

# ---------------------------
# Asset universe & metadata (dividend yields simulated)
# ---------------------------
ASSETS = [
    {"id": "ETF_DE", "category": "ETF", "name": "Deutschland ETF", "symbol": "DE.ETF", "div_yield": 0.015},
    {"id": "ETF_US", "category": "ETF", "name": "USA ETF", "symbol": "US.ETF", "div_yield": 0.012},
    {"id": "ETF_EU", "category": "ETF", "name": "Europa ETF", "symbol": "EU.ETF", "div_yield": 0.013},
    {"id": "ETF_WW", "category": "ETF", "name": "Welt ETF", "symbol": "WW.ETF", "div_yield": 0.011},
    {"id": "CR_BTC", "category": "Krypto", "name": "Bitcoin", "symbol": "BTC", "div_yield": 0.0},
    {"id": "CR_ETH", "category": "Krypto", "name": "Ethereum", "symbol": "ETH", "div_yield": 0.0},
    {"id": "ST_AAPL", "category": "Aktie", "name": "Apple", "symbol": "AAPL", "div_yield": 0.006},
    {"id": "ST_TSLA", "category": "Aktie", "name": "Tesla", "symbol": "TSLA", "div_yield": 0.0},
    {"id": "ST_MSFT", "category": "Aktie", "name": "Microsoft", "symbol": "MSFT", "div_yield": 0.008},
    {"id": "ST_NVDA", "category": "Aktie", "name": "NVIDIA", "symbol": "NVDA", "div_yield": 0.0},
]
ASSETS_BY_ID = {a["id"]: a for a in ASSETS}

# ---------------------------
# Deterministic price generation (cached)
# ---------------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2 ** 31)

def generate_series(asset_id: str, days: int = 365, start_price: float = 100.0):
    cache_key = f"{asset_id}_{days}_{int(start_price)}"
    if cache_key in st.session_state.series_cache:
        return st.session_state.series_cache[cache_key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    series = []
    for i in range(days):
        drift = (rnd.random() - 0.48) * 0.001
        vol = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        series.append({"date": date, "price": round(price, 4)})
    # SMA
    for i in range(len(series)):
        p20 = [series[j]["price"] for j in range(max(0, i - 19), i + 1)]
        p50 = [series[j]["price"] for j in range(max(0, i - 49), i + 1)]
        series[i]["sma20"] = round(mean(p20), 4)
        series[i]["sma50"] = round(mean(p50), 4)
    st.session_state.series_cache[cache_key] = series
    return series

# ---------------------------
# Portfolio operations (per profile)
# ---------------------------
def save_profile_state():
    data = {
        "settings": st.session_state.settings,
        "portfolio": st.session_state.portfolio,
        "history": st.session_state.history,
        "notifications": st.session_state.notifications,
        "watchlist": st.session_state.watchlist
    }
    save_profile(st.session_state.profile, data)
    # also update global profiles meta
    profiles_meta = load_json(PROFILES_FILE, {"profiles": [DEFAULT_PROFILE], "active": st.session_state.profile})
    if st.session_state.profile not in profiles_meta.get("profiles", []):
        profiles_meta.setdefault("profiles", []).append(st.session_state.profile)
    profiles_meta["active"] = st.session_state.profile
    save_json(PROFILES_FILE, profiles_meta)

def add_position(category, asset_id, name, qty, buy_price, note="", tags=None):
    if tags is None:
        tags = []
    item = {
        "id": f"{asset_id}_{len(st.session_state.portfolio) + 1}_{int(datetime.utcnow().timestamp())}",
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "qty": float(qty),
        "buy_price": float(buy_price),
        "note": note,
        "tags": tags,
        "added_at": datetime.utcnow().isoformat()
    }
    st.session_state.portfolio.append(item)
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "add", "item": item})
    save_profile_state()
    st.success(f"{name} hinzugefügt.")
    check_notifications_for_item(item)

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "remove", "item": {"id": item_id}})
    save_profile_state()
    st.success("Position entfernt.")
    st.experimental_rerun()

def update_note_tags(item_id, new_note, new_tags):
    for p in st.session_state.portfolio:
        if p["id"] == item_id:
            p["note"] = new_note
            p["tags"] = new_tags
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": "note_update", "item": {"id": item_id, "note": new_note, "tags": new_tags}})
    save_profile_state()
    st.success("Notiz und Tags gespeichert.")

# ---------------------------
# Analytics: snapshot, volatility, dividends, benchmark, forecast, risk analyzer
# ---------------------------
def portfolio_snapshot():
    tot_value = 0.0
    tot_cost = 0.0
    rows = []
    for item in st.session_state.portfolio:
        series = generate_series(item["asset_id"], 365, start_price=item["buy_price"] if item["buy_price"] > 0 else 100.0)
        cur = series[-1]["price"]
        qty = float(item["qty"])
        value = cur * qty
        cost = float(item["buy_price"]) * qty
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct, "series": series})
        tot_value += value
        tot_cost += cost
    return {"rows": rows, "total_value": tot_value, "total_cost": tot_cost}

def calc_volatility(series, window=30):
    if not series or len(series) < window + 1:
        return 0.0
    prices = [p["price"] for p in series[-(window + 1):]]
    returns = []
    for i in range(1, len(prices)):
        if prices[i - 1] > 0:
            returns.append((prices[i] - prices[i - 1]) / prices[i - 1])
    if not returns:
        return 0.0
    return stdev(returns)

def compute_dividends(item, years=1):
    # simple annual dividend: div_yield * buy_price * qty * years
    meta = ASSETS_BY_ID.get(item["asset_id"], {})
    div_yield = meta.get("div_yield", 0.0)
    return div_yield * item["buy_price"] * item["qty"] * years

def generate_benchmark(days=365, start=100.0):
    # Create a deterministic benchmark series (e.g., MSCI dummy) for comparison
    rnd = random.Random(deterministic_seed("BENCHMARK"))
    price = float(start)
    series = []
    for i in range(days):
        drift = 0.0002  # slight positive drift
        vol = (rnd.random() - 0.5) * 0.01
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        series.append({"date": date, "price": round(price, 4)})
    return series

def forecast_linear(series, future_days=30):
    # simple linear regression (slope intercept) on price vs day index
    if not series or len(series) < 3:
        return []
    n = len(series)
    xs = list(range(n))
    ys = [p["price"] for p in series]
    x_mean = mean(xs); y_mean = mean(ys)
    num = sum((xs[i] - x_mean) * (ys[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    slope = num / den if den != 0 else 0.0
    intercept = y_mean - slope * x_mean
    preds = []
    for fd in range(1, future_days + 1):
        x = n + fd - 1
        preds.append({"date": (datetime.utcnow().date() + timedelta(days=fd)).isoformat(), "price": round(intercept + slope * x, 4)})
    return preds

def risk_analyzer_score():
    # Simple "KI-like" risk score: combine volatility (portfolio level) and concentration
    snap = portfolio_snapshot()
    if not snap["rows"]:
        return {"score": 0.0, "components": {}}
    # volatility: mean of asset volatilities (30d)
    vols = [calc_volatility(r["series"], 30) for r in snap["rows"]]
    vol_norm = mean(vols) if vols else 0.0  # daily vol
    # concentration: share of top position
    total = snap["total_value"] if snap["total_value"] > 0 else 1.0
    top_share = max((r["value"] for r in snap["rows"]), default=0.0) / total
    # score 0..100 (higher = more risky)
    score = min(100.0, (vol_norm * 100 * 0.6) + (top_share * 100 * 0.4))
    return {"score": round(score, 2), "volatility_component": round(vol_norm * 100, 2), "concentration_component": round(top_share * 100, 2)}

# ---------------------------
# Notifications & alerts
# ---------------------------
def show_toast(msg, level="info"):
    try:
        if hasattr(st, "toast"):
            icon = "ℹ️" if level == "info" else ("✅" if level == "success" else "⚠️")
            st.toast(msg, icon=icon)
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
    # compute current pnl_pct
    series = generate_series(item["asset_id"], 365, start_price=item["buy_price"] if item["buy_price"] > 0 else 100.0)
    cur = series[-1]["price"]
    cost = item["buy_price"] * item["qty"]
    value = cur * item["qty"]
    pnl = value - cost
    pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
    # check watchlist thresholds also
    # Alert on >10% loss and not already sent
    if pnl_pct < -10 and item["id"] not in st.session_state.alerts_sent:
        msg = f"⚠️ Verlust bei {item['name']}: {pnl_pct:.2f}%"
        st.session_state.notifications.append({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        save_json(profile_file(st.session_state.profile, "notifications"), st.session_state.notifications)
        st.session_state.alerts_sent.add(item["id"])
        show_toast(msg, level="warning")

def check_all_notifications():
    for item in st.session_state.portfolio:
        check_notifications_for_item(item)

# initial check
check_all_notifications()

# ---------------------------
# Watchlist operations
# ---------------------------
def add_to_watchlist(asset_id):
    if asset_id not in st.session_state.watchlist:
        st.session_state.watchlist.append(asset_id)
        save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
        st.success("Asset zur Watchlist hinzugefügt.")
    else:
        st.info("Asset bereits auf der Watchlist.")

def remove_from_watchlist(asset_id):
    if asset_id in st.session_state.watchlist:
        st.session_state.watchlist.remove(asset_id)
        save_json(profile_file(st.session_state.profile, "watchlist"), st.session_state.watchlist)
        st.success("Asset von Watchlist entfernt.")

# ---------------------------
# Snapshots (historical recording)
# ---------------------------
def take_snapshot():
    snap = portfolio_snapshot()
    entry = {"timestamp": datetime.utcnow().isoformat(), "snapshot": snap}
    st.session_state.history.append(entry)
    save_json(profile_file(st.session_state.profile, "history"), st.session_state.history)
    st.success("Snapshot gespeichert.")

# ---------------------------
# Export report (PDF if available)
# ---------------------------
def export_daily_report(filename="daily_report.txt"):
    snap = portfolio_snapshot()
    lines = []
    lines.append(f"Daily Report — {datetime.utcnow().isoformat()}\n")
    lines.append(f"Total Value: {snap['total_value']:.2f} EUR")
    lines.append(f"Total Cost: {snap['total_cost']:.2f} EUR\n")
    lines.append("Positions:")
    for r in snap["rows"]:
        lines.append(f"- {r['item']['name']}: value={r['value']:.2f} EUR pnl={r['pnl']:+.2f} ({r['pnl_pct']:+.2f}%)")
    txt = "\n".join(lines)
    if REPORTLAB_AVAILABLE:
        # create simple PDF
        pdf_path = os.path.join(APP_FOLDER, filename.replace(".txt", ".pdf"))
        c = canvas.Canvas(pdf_path, pagesize=A4)
        width, height = A4
        y = height - 40
        c.setFont("Helvetica", 12)
        for line in lines:
            c.drawString(40, y, line)
            y -= 16
            if y < 40:
                c.showPage()
                y = height - 40
        c.save()
        return pdf_path
    else:
        # return text download (bytes)
        return txt.encode("utf-8")

# ---------------------------
# UI: Sidebar (profiles, theme, quick actions)
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home", "Marktplatz", "Portfolio", "Rebalancing", "Simulation", "Statistiken", "Wissensbasis", "Export/Import", "Einstellungen"])

st.sidebar.markdown("---")
# profile chooser & management
profiles_meta = load_json(PROFILES_FILE, {"profiles": [DEFAULT_PROFILE], "active": st.session_state.profile})
profiles = profiles_meta.get("profiles", [DEFAULT_PROFILE])
new_profile = st.sidebar.text_input("Neues Profil erstellen", value="", key="new_profile_name")
if st.sidebar.button("Erstelle Profil") and new_profile.strip():
    if new_profile in profiles:
        st.sidebar.error("Profil existiert bereits.")
    else:
        profiles.append(new_profile)
        profiles_meta["profiles"] = profiles
        profiles_meta["active"] = new_profile
        save_json(PROFILES_FILE, profiles_meta)
        # create files and reload
        ensure_profile_files(new_profile)
        st.sidebar.success(f"Profil {new_profile} erstellt. Bitte reload.")
if st.sidebar.selectbox("Aktives Profil", profiles, index=profiles.index(st.session_state.profile) if st.session_state.profile in profiles else 0, key="profile_select") != st.session_state.profile:
    chosen = st.session_state.profile if st.session_state.profile in profiles else profiles[0]
# switch profile button
if st.sidebar.button("Profil wechseln"):
    sel = st.sidebar.session_state.get("profile_select", profiles[0])
    profiles_meta["active"] = sel
    save_json(PROFILES_FILE, profiles_meta)
    # reload into session (simple approach: ask user to reload app)
    st.sidebar.info(f"Profil gewechselt zu {sel}. Bitte Seite neu laden.")

st.sidebar.markdown("---")
if st.sidebar.button("Snapshot jetzt speichern"):
    take_snapshot()
if st.sidebar.button("Export Report (PDF/Text)"):
    out = export_daily_report("daily_report")
    if isinstance(out, str) and out.endswith(".pdf"):
        with open(out, "rb") as f:
            st.sidebar.download_button("Download PDF Report", data=f.read(), file_name=os.path.basename(out), mime="application/pdf")
    else:
        st.sidebar.download_button("Download TXT Report", data=out, file_name="daily_report.txt", mime="text/plain")

st.sidebar.markdown("---")
# Theme toggle
if st.sidebar.checkbox("Dark Theme (UI)", value=st.session_state.dark_theme, key="ui_dark"):
    st.session_state.dark_theme = True
else:
    st.session_state.dark_theme = False
st.session_state.settings["dark_theme"] = st.session_state.dark_theme

# ---------------------------
# Page: Home
# ---------------------------
if page == "Home":
    st.title("Home — Übersicht")
    snap = portfolio_snapshot()
    st.metric("Portfolio Gesamtwert", f"{snap['total_value']:.2f} €", delta=f"{(snap['total_value'] - snap['total_cost']):+.2f} €")
    st.write(f"Eingezahlt: {snap['total_cost']:.2f} €  •  Ziel: {st.session_state.settings.get('goal', 10000.0):.2f} €")
    st.progress(min(snap['total_value'] / float(st.session_state.settings.get("goal", 10000.0)), 1.0))
    st.markdown("---")
    # widgets
    st.subheader("Widgets")
    stats = portfolio_stats()
    if stats:
        c1, c2, c3 = st.columns(3)
        c1.metric("Anzahl Positionen", stats["count"])
        c2.metric("Durchschn. PnL", f"{stats['avg_pnl']:+.2f} €")
        c3.metric("Volatilität (Snapshot)", f"{stats['volatility']:.4f}")
        ra = risk_analyzer_score()
        st.write(f"Risikoscore: **{ra['score']}** (Vol:{ra['volatility_component']} / Konzentration:{ra['concentration_component']})")
    else:
        st.info("Noch keine Positionen — füge welche im Marktplatz hinzu.")
    st.markdown("---")
    st.subheader("Offline News Feed")
    # generate 5 mock news items (deterministic seeded)
    rnd = random.Random(deterministic_seed("NEWS_" + st.session_state.profile))
    news = []
    keywords = ["Zinsen", "Inflation", "Tech", "Rohstoffe", "Krypto", "Zentralbank", "Arbeitsmarkt", "Wachstum"]
    for i in range(5):
        title = f"{rnd.choice(keywords)}: Markt bewegt sich ({rnd.randint(-3,3)}%)"
        link = "https://example.com/news-" + str(i)
        ts = (datetime.utcnow() - timedelta(hours=rnd.randint(1,72))).isoformat()
        news.append({"title": title, "link": link, "time": ts})
    for n in news:
        st.markdown(f"- [{n['title']}]({n['link']})  <span class='small'>– {n['time']}</span>", unsafe_allow_html=True)

# ---------------------------
# Page: Marktplatz
# ---------------------------
elif page == "Marktplatz":
    st.title("Marktplatz")
    st.markdown("Durchsuche Assets, sieh Preise (simuliert), füge zur Watchlist oder Portfolio hinzu.")
    categories = {}
    for a in ASSETS:
        categories.setdefault(a["category"], []).append(a)
    for cat, assets in categories.items():
        st.subheader(cat)
        for a in assets:
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{a['name']}** — <span class='small'>{a['symbol']}</span>", unsafe_allow_html=True)
                days = st.selectbox(f"Zeitraum für {a['id']}", ["1 Monat", "3 Monate", "6 Monate", "1 Jahr"], key=f"iv_{a['id']}")
                mapping = {"1 Monat": 30, "3 Monate": 90, "6 Monate": 180, "1 Jahr": 365}
                series = generate_series(a["id"], mapping[days], start_price=100.0 + (abs(hash(a["id"])) % 1000) / 10)
                if PLOTLY_AVAILABLE:
                    fig = go.Figure()
                    dates = [p["date"] for p in series]
                    prices = [p["price"] for p in series]
                    fig.add_trace(go.Scatter(x=dates, y=prices, mode="lines", name=a["symbol"]))
                    fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=260, paper_bgcolor="black", plot_bgcolor="black", font_color="white")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.line_chart({p["date"]: p["price"] for p in series})
            with cols[1]:
                cur = series[-1]["price"]
                sma20 = series[-1]["sma20"]; sma50 = series[-1]["sma50"]
                rec = "Kaufen" if sma20 > sma50 else "Nicht kaufen"
                st.write(f"Aktuell: {cur:.2f} €")
                st.write(f"SMA20: {sma20:.2f}  SMA50: {sma50:.2f}")
                st.markdown(f"**Empfehlung:** {'🟢 '+rec if rec=='Kaufen' else '🔴 '+rec}")
                # Watchlist + add form
                if st.button("Zur Watchlist hinzufügen", key=f"watch_{a['id']}"):
                    add_to_watchlist(a["id"])
                with st.form(key=f"add_form_{a['id']}", clear_on_submit=False):
                    qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                    bp = st.number_input("Kaufpreis", min_value=0.0001, value=float(cur), step=0.01, key=f"bp_{a['id']}")
                    note = st.text_area("Notiz", value="", key=f"note_{a['id']}", height=80)
                    tags_txt = st.text_input("Tags (kommagetrennt)", value="", key=f"tags_{a['id']}")
                    if st.form_submit_button("Zum Portfolio hinzufügen"):
                        tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                        add_position(cat, a["id"], a["name"], qty, bp, note, tags)

# ---------------------------
# Page: Portfolio
# ---------------------------
elif page == "Portfolio":
    st.title("Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Portfolio leer. Füge Assets im Marktplatz hinzu.")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} € — Eingezahlt: {snap['total_cost']:.2f} €")
        st.progress(min(snap['total_value'] / float(st.session_state.settings.get("goal", 10000.0)), 1.0))
        st.markdown("---")
        for r in snap['rows']:
            item = r['item']
            with st.expander(f"{item['name']} — {r['value']:.2f} € ({r['pnl']:+.2f} €)"):
                cols = st.columns([3,1])
                with cols[0]:
                    st.write(f"Menge: {item['qty']} | Kaufpreis: {item['buy_price']:.2f} €")
                    st.write(f"Aktuell: {r['cur']:.4f} € • Wert: {r['value']:.2f} €")
                    st.write(f"PnL: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)")
                    # draw series (plotly if possible)
                    series = r["series"]
                    if PLOTLY_AVAILABLE:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["price"] for p in series], name="Preis"))
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["sma20"] for p in series], name="SMA20"))
                        fig.add_trace(go.Scatter(x=[p["date"] for p in series], y=[p["sma50"] for p in series], name="SMA50"))
                        fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, paper_bgcolor="black", plot_bgcolor="black", font_color="white")
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.line_chart({p["date"]: p["price"] for p in series})
                with cols[1]:
                    # dividends
                    div = compute_dividends(item, years=1) if st.session_state.settings.get("dividends_enabled", True) else 0.0
                    st.write(f"Geschätzte Dividenden (1 Jahr): {div:.2f} €")
                    # notes/tags
                    new_note = st.text_area("Notiz", value=item.get("note", ""), key=f"note_p_{item['id']}")
                    tags_txt = st.text_input("Tags", value=",".join(item.get("tags", [])), key=f"tags_p_{item['id']}")
                    if st.button("Speichern (Notiz/Tags)", key=f"save_p_{item['id']}"):
                        tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                        update_note_tags(item['id'], new_note, tags)
                    if st.button("Entfernen", key=f"rm_p_{item['id']}"):
                        remove_position(item['id'])
        # summary & analytics
        st.markdown("---")
        st.subheader("Portfolio Analyse")
        ra = risk_analyzer_score()
        st.write(f"Risikoscore: **{ra['score']}** (Vol:{ra['volatility_component']}%, Concentration:{ra['concentration_component']}%)")
        # benchmark comparison (last 180 days)
        bench = generate_benchmark(180, start=100.0)
        hist = simulate_over_time(180)
        if hist:
            # create comparison plot
            if PLOTLY_AVAILABLE:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[p["date"] for p in hist], y=[p["value"] for p in hist], name="Portfolio"))
                fig.add_trace(go.Scatter(x=[b["date"] for b in bench], y=[b["price"] * (hist[0]["value"] / bench[0]["price"]) for b in bench], name="Benchmark (scaled)"))
                fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, paper_bgcolor="black", plot_bgcolor="black", font_color="white")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # fallback: two charts one after another
                st.line_chart({p["date"]: p["value"] for p in hist})
                st.line_chart({b["date"]: b["price"] for b in bench})
        # snap save
        if st.button("Snapshot jetzt speichern"):
            take_snapshot()

# ---------------------------
# Page: Rebalancing
# ---------------------------
elif page == "Rebalancing":
    st.title("Rebalancing Advisor")
    st.markdown("Definiere Zielallokation (Summe = 1.0). Die App gibt Vorschläge.")
    c1, c2, c3 = st.columns(3)
    tgt_etf = c1.number_input("ETF", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="tgt_etf")
    tgt_act = c2.number_input("Aktie", min_value=0.0, max_value=1.0, value=0.3, step=0.05, key="tgt_act")
    tgt_cry = c3.number_input("Krypto", min_value=0.0, max_value=1.0, value=0.2, step=0.05, key="tgt_cry")
    total = tgt_etf + tgt_act + tgt_cry
    if abs(total - 1.0) > 1e-6:
        st.warning("Die Zielallokation muss auf 1.0 summieren.")
    else:
        target = {"ETF": tgt_etf, "Aktie": tgt_act, "Krypto": tgt_cry}
        cur_alloc, advice = rebalance_advice(target)
        st.write("Aktuelle Allokation (geschätzt):")
        for k, v in cur_alloc.items():
            st.write(f"- {k}: {v*100:.2f}%")
        st.write("Empfehlungen (positiv = kaufen, negativ = verkaufen):")
        for k, v in advice.items():
            st.write(f"- {k}: {(v*100):+.2f}%")

# ---------------------------
# Page: Simulation
# ---------------------------
elif page == "Simulation":
    st.title("Simulation & Forecast")
    st.markdown("Historische Portfolio-Entwicklung (simuliert) und einfache Prognose")
    days = st.selectbox("Zeitraum", [90, 180, 365, 365 * 3], index=2)
    hist = simulate_over_time(days)
    if hist:
        if PLOTLY_AVAILABLE:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[p["date"] for p in hist], y=[p["value"] for p in hist], name="Portfolio"))
            fig.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=350, paper_bgcolor="black", plot_bgcolor="black", font_color="white")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.line_chart({p["date"]: p["value"] for p in hist})
        # forecast: linear projection from portfolio (last 120 days)
        base_series = hist[-120:] if len(hist) >= 120 else hist
        # use last position values as series for linear forecast: map to simple numeric
        prices = [p["value"] for p in base_series]
        # build simple series for forecast function (re-using forecast_linear)
        # create a dummy series format
        ser = [{"date": base_series[i]["date"], "price": base_series[i]["value"]} for i in range(len(base_series))]
        preds = forecast_linear(ser, future_days=30)
        if preds:
            if PLOTLY_AVAILABLE:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=[p["date"] for p in ser], y=[p["price"] for p in ser], name="History"))
                fig2.add_trace(go.Scatter(x=[p["date"] for p in preds], y=[p["price"] for p in preds], name="Forecast"))
                fig2.update_layout(margin=dict(l=0, r=0, t=20, b=0), height=300, paper_bgcolor="black", plot_bgcolor="black", font_color="white")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.line_chart({p["date"]: p["price"] for p in ser})
                st.line_chart({p["date"]: p["price"] for p in preds})
    else:
        st.info("Portfolio leer — keine Simulation verfügbar.")
    st.markdown("---")
    st.subheader("Crash-Simulation")
    drop = st.slider("Simulierter %-Drop", 1, 100, 20)
    if st.button("Crash simulieren"):
        if not st.session_state.portfolio:
            st.info("Portfolio leer.")
        else:
            res = simulate_crash(drop)
            st.write(f"Vorher: {res['total_pre']:.2f} €  •  Nachher: {res['total_post']:.2f} €  •  Änderung: {res['total_delta']:+.2f} €")
            st.table([{"Asset": x["name"], "Vorher": round(x["pre_value"], 2), "Nachher": round(x["post_value"], 2), "Delta": round(x["delta"], 2)} for x in res["items"]])

# ---------------------------
# Page: Statistiken
# ---------------------------
elif page == "Statistiken":
    st.title("Statistiken")
    stats = portfolio_stats()
    if not stats:
        st.info("Portfolio leer — keine Statistiken.")
    else:
        st.write(f"Positionen: {stats['count']}")
        st.write(f"Durchschn. PnL: {stats['avg_pnl']:+.2f} €")
        st.write(f"Volatilität: {stats['volatility']:.4f}")
        st.markdown("---")
        # category breakdown
        snap = portfolio_snapshot()
        cat_vals = {}
        for r in snap["rows"]:
            cat_vals[r["item"]["category"]] = cat_vals.get(r["item"]["category"], 0.0) + r["value"]
        st.bar_chart(cat_vals)
        st.markdown("---")
        # heatmap-like
        st.write("Heatmap (grün = Gewinn, rot = Verlust)")
        for r in snap["rows"]:
            pct = r["pnl_pct"]
            color = "#006600" if pct >= 0 else "#660000"
            width = min(max(abs(pct) * 1.5, 2), 100)
            st.markdown(f"<div style='background:{color}; width:{width}%; padding:8px; border-radius:6px; margin-bottom:6px;'>{r['item']['name']} — {r['pnl']:+.2f} € ({pct:+.2f}%)</div>", unsafe_allow_html=True)

# ---------------------------
# Page: Wissensbasis / Asset Wiki
# ---------------------------
elif page == "Wissensbasis":
    st.title("Wissensbasis & Asset Wiki")
    st.write("Offline-Leitfäden und Asset-Profile.")
    guides = st.session_state.guides or {}
    st.subheader("Guides")
    for k, v in guides.items():
        with st.expander(k.replace("_", " ")):
            st.write(v)
    st.subheader("Asset Wiki")
    for a in ASSETS:
        with st.expander(f"{a['name']} ({a['symbol']})"):
            st.write(f"Kategorie: {a['category']}")
            st.write(f"Symbol: {a['symbol']}")
            st.write(f"Simulierter Dividendenrendite: {a.get('div_yield', 0.0) * 100:.2f}%")
            st.write("Kurzbeschreibung (offline):")
            st.write(f"{a['name']} ist ein Beispiel-Asset in der Offline-Demo. Informationen sind fiktiv.")

# ---------------------------
# Page: Export / Import
# ---------------------------
elif page == "Export/Import":
    st.title("Export / Import")
    st.markdown("Exportiere dein Profil (Portfolio & Einstellungen) oder importiere ein Backup.")
    export_obj = {
        "profile": st.session_state.profile,
        "settings": st.session_state.settings,
        "portfolio": st.session_state.portfolio,
        "history": st.session_state.history,
        "notifications": st.session_state.notifications,
        "exported_at": datetime.utcnow().isoformat()
    }
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2)
    st.download_button("Export JSON", data=export_json, file_name=f"export_{st.session_state.profile}.json", mime="application/json")
    st.markdown("---")
    uploaded = st.file_uploader("Import JSON", type=["json"])
    if uploaded:
        try:
            obj = json.loads(uploaded.read().decode("utf-8"))
            if obj.get("profile") and obj.get("portfolio") is not None:
                st.session_state.profile = obj.get("profile")
                st.session_state.settings = obj.get("settings", {})
                st.session_state.portfolio = obj.get("portfolio", [])
                st.session_state.history = obj.get("history", [])
                st.session_state.notifications = obj.get("notifications", [])
                save_profile_state()
                st.success("Importiert. Bitte Seite neu laden.")
        except Exception as e:
            st.error(f"Import fehlgeschlagen: {e}")

# ---------------------------
# Page: Einstellungen
# ---------------------------
elif page == "Einstellungen":
    st.title("Einstellungen")
    st.subheader("Profil & App")
    st.write(f"Aktives Profil: {st.session_state.profile}")
    if st.button("Profil exportieren (backup)"):
        save_profile_state()
        st.success("Profil gespeichert (lokales Backup).")
    st.markdown("---")
    st.subheader("App-Einstellungen")
    goal_val = st.number_input("Finanzziel", min_value=0.0, value=float(st.session_state.settings.get("goal", 10000.0)), step=100.0)
    div_enabled = st.checkbox("Dividenden berücksichtigen (simuliert)", value=st.session_state.settings.get("dividends_enabled", True))
    reinvest = st.checkbox("Dividenden automatisch reinvestieren (simuliert)", value=st.session_state.settings.get("reinvest_dividends", False))
    if st.button("Speichern"):
        st.session_state.settings["goal"] = float(goal_val)
        st.session_state.settings["dividends_enabled"] = bool(div_enabled)
        st.session_state.settings["reinvest_dividends"] = bool(reinvest)
        save_profile_state()
        st.success("Einstellungen gespeichert.")
    st.markdown("---")
    st.subheader("Cache & Daten")
    if st.button("Chart-Cache löschen"):
        st.session_state.series_cache = {}
        st.success("Cache gelöscht.")
    if st.button("Alle Benachrichtigungen leeren"):
        st.session_state.notifications = []
        save_profile_state()
        st.success("Benachrichtigungen gelöscht.")
    st.markdown("---")
    st.subheader("Eigentümer: Passwort ändern")
    auth = st.session_state.settings.get("auth")
    if not auth:
        st.warning("Kein Passwort gesetzt für dieses Profil.")
    else:
        old = st.text_input("Altes Passwort", type="password", key="pwd_old")
        new = st.text_input("Neues Passwort", type="password", key="pwd_new")
        new2 = st.text_input("Neues Passwort wiederholen", type="password", key="pwd_new2")
        if st.button("Passwort ändern"):
            try:
                salt = binascii.unhexlify(auth["salt"])
                dk_old = derive_key(old, salt, iterations=auth.get("iterations", 200_000), dklen=auth.get("dklen", 72))
                if binascii.hexlify(dk_old).decode() != auth["key"]:
                    st.error("Altes Passwort falsch.")
                elif not new or new != new2:
                    st.error("Neues Passwort leer oder Wiederholung stimmt nicht.")
                else:
                    new_salt = os.urandom(16)
                    new_dk = derive_key(new, new_salt, iterations=200_000, dklen=72)
                    st.session_state.settings["auth"] = {"salt": binascii.hexlify(new_salt).decode(), "key": binascii.hexlify(new_dk).decode(), "iterations": 200_000, "dklen": 72}
                    save_profile_state()
                    st.success("Passwort geändert.")
            except Exception as e:
                st.error(f"Fehler: {e}")

# ---------------------------
# Footer & final save
# ---------------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Local JSON storage • Optional: plotly/pandas/reportlab for enhanced features</div>", unsafe_allow_html=True)

# Save profile state at end of request to persist changes
save_profile_state()
