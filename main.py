# main.py
# Offline-Finanz-App — Black Pro — komplette Version
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# -----------------------
# Config & Files
# -----------------------
st.set_page_config(page_title="Finanz-Platform (Offline Pro)", page_icon="💹", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"
GUIDES_FILE = "guides.json"
NOTIFICATIONS_FILE = "notifications.json"

# -----------------------
# Styling (Black theme)
# -----------------------
st.markdown("""
<style>
html, body, [class*="css"] {background:#000 !important; color:#e6eef6 !important;}
.stButton>button {background:#111; color:#e6eef6; border:1px solid #222; border-radius:6px;}
.card {background:#070707; padding:14px; border-radius:10px; border:1px solid #111; margin-bottom:12px;}
.small {color:#9aa6b2; font-size:13px;}
.gain {background:linear-gradient(90deg,#00ff88,#007744); height:10px; border-radius:6px;}
.loss {background:linear-gradient(90deg,#ff4466,#770022); height:10px; border-radius:6px;}
.badge {background:#111; color:#e6eef6; padding:4px 8px; border-radius:6px; border:1px solid #222; display:inline-block;}
.spark {height:48px;}
.toast {background:#222; color:#e6eef6; padding:8px; border-radius:6px; border:1px solid #444; margin-bottom:6px;}
.link {color:#6fb3ff; text-decoration:underline;}
</style>
""", unsafe_allow_html=True)

# -----------------------
# JSON helpers
# -----------------------
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

# -----------------------
# Ensure persistence files exist (create defaults)
# -----------------------
if not os.path.exists(SETTINGS_FILE):
    save_json(SETTINGS_FILE, {})
if not os.path.exists(GUIDES_FILE):
    initial_guides = {
        "ETF_vs_Aktie": (
            "ETFs (Exchange Traded Funds) bündeln viele Aktien oder Anleihen. "
            "Sie bilden einen Index oder Sektor ab. Vorteil: breite Diversifikation "
            "und niedrige Kosten. Nachteil: du kaufst den Markt, nicht einzelne "
            "Top-Performer."
        ),
        "Volatilitaet": (
            "Volatilität ist die Schwankungsbreite der Renditen. Sie wird oft "
            "als Standardabweichung der Renditen gemessen. Höhere Volatilität "
            "bedeutet mehr Unsicherheit — kann Chancen und Risiken vergrössern."
        ),
        "Rebalancing": (
            "Rebalancing bringt die Aufteilung deines Portfolios zurück auf die "
            "gewünschte Zielallokation. Beispielsweise: 50% ETFs, 30% Aktien, 20% Krypto. "
            "Regelmässiges Rebalancing reduziert Risiko durch Gewinnmitnahme."
        ),
        "Diversifikation": (
            "Diversifikation verteilt Kapital über Regionen, Branchen und Assetklassen. "
            "Sie hilft, idiosynkratisches Risiko einzelner Titel zu reduzieren."
        ),
        "Crash_Simulation": (
            "Mit einer Crash-Simulation kannst du sehen, wie stark ein einmaliger "
            "Preisrückgang (z.B. -20%) dein Portfolio beeinflusst. Nützlich für Stress-Tests."
        )
    }
    save_json(GUIDES_FILE, initial_guides)
if not os.path.exists(PORTFOLIO_FILE):
    save_json(PORTFOLIO_FILE, [])
if not os.path.exists(HISTORY_FILE):
    save_json(HISTORY_FILE, [])
if not os.path.exists(NOTIFICATIONS_FILE):
    save_json(NOTIFICATIONS_FILE, [])

# -----------------------
# Session state defaults
# -----------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_json(PORTFOLIO_FILE, [])
if "settings" not in st.session_state:
    st.session_state.settings = load_json(SETTINGS_FILE, {})
if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE, [])
if "guides" not in st.session_state:
    st.session_state.guides = load_json(GUIDES_FILE, {})
if "notifications" not in st.session_state:
    st.session_state.notifications = load_json(NOTIFICATIONS_FILE, [])
if "series_cache" not in st.session_state:
    st.session_state.series_cache = {}
if "alerts_sent" not in st.session_state:
    st.session_state.alerts_sent = set()
if "favorites" not in st.session_state:
    st.session_state.favorites = set()
if "auth_ok" not in st.session_state:
    st.session_state.auth_ok = False

# -----------------------
# Security: PBKDF2 (72 bytes)
# -----------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def setup_password_flow():
    settings = st.session_state.settings or {}
    if not settings.get("auth"):
        st.header("Erstinstallation: Eigentümer-Passwort setzen")
        st.info("Ein Passwort wird lokal und sicher (PBKDF2) gespeichert. Nur ein Besitzer.")
        pwd = st.text_input("Passwort wählen", type="password")
        pwd2 = st.text_input("Passwort wiederholen", type="password")
        if st.button("Passwort setzen"):
            if not pwd or pwd != pwd2:
                st.error("Passwörter leer oder stimmen nicht überein.")
            else:
                salt = os.urandom(16)
                dk = derive_key(pwd, salt)
                settings["auth"] = {
                    "salt": binascii.hexlify(salt).decode(),
                    "key": binascii.hexlify(dk).decode(),
                    "iterations": 200_000,
                    "dklen": 72
                }
                st.session_state.settings = settings
                save_json(SETTINGS_FILE, settings)
                st.success("Passwort gesetzt. Bitte neu laden und einloggen.")
                st.stop()
        st.stop()
    return True

def login_flow():
    auth = st.session_state.settings.get("auth", {})
    st.header("Login (Eigentümer)")
    pwd = st.text_input("Passwort", type="password")
    if st.button("Einloggen"):
        if not auth:
            st.error("Auth-Daten fehlen.")
            st.stop()
        salt = binascii.unhexlify(auth["salt"])
        dk = derive_key(pwd, salt, iterations=auth.get("iterations", 200_000), dklen=auth.get("dklen", 72))
        if binascii.hexlify(dk).decode() == auth["key"]:
            st.session_state.auth_ok = True
            st.success("Erfolgreich eingeloggt.")
            return True
        else:
            st.error("Falsches Passwort.")
            return False
    st.stop()
    return False

# gating
if not setup_password_flow():
    st.stop()
if not st.session_state.auth_ok:
    if not login_flow():
        st.stop()

# -----------------------
# Asset definitions
# -----------------------
ETFS = [
    {"id": "ETF_DE", "name": "Deutschland ETF", "symbol": "DE.ETF"},
    {"id": "ETF_US", "name": "USA ETF", "symbol": "US.ETF"},
    {"id": "ETF_EU", "name": "Europa ETF", "symbol": "EU.ETF"},
    {"id": "ETF_AS", "name": "Asien ETF", "symbol": "AS.ETF"},
    {"id": "ETF_WW", "name": "Welt ETF", "symbol": "WW.ETF"},
]
CRYPTOS = [
    {"id": "CR_BTC", "name": "Bitcoin", "symbol": "BTC"},
    {"id": "CR_ETH", "name": "Ethereum", "symbol": "ETH"},
    {"id": "CR_SOL", "name": "Solana", "symbol": "SOL"},
]
STOCKS = [
    {"id": "ST_AAPL", "name": "Apple", "symbol": "AAPL"},
    {"id": "ST_TSLA", "name": "Tesla", "symbol": "TSLA"},
    {"id": "ST_MSFT", "name": "Microsoft", "symbol": "MSFT"},
    {"id": "ST_NVDA", "name": "NVIDIA", "symbol": "NVDA"},
    {"id": "ST_SAP", "name": "SAP", "symbol": "SAP"},
]

ALL_ASSETS = ETFS + CRYPTOS + STOCKS

# -----------------------
# Deterministic price series generator
# -----------------------
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
    # SMA20/50
    for i in range(len(series)):
        p20 = [series[j]["price"] for j in range(max(0, i - 19), i + 1)]
        p50 = [series[j]["price"] for j in range(max(0, i - 49), i + 1)]
        series[i]["sma20"] = round(mean(p20), 4)
        series[i]["sma50"] = round(mean(p50), 4)
    st.session_state.series_cache[cache_key] = series
    return series

# -----------------------
# Portfolio operations
# -----------------------
def save_portfolio():
    save_json(PORTFOLIO_FILE, st.session_state.portfolio)

def save_history(action, item):
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": action, "item": item})
    save_json(HISTORY_FILE, st.session_state.history)

def add_position(category: str, asset_id: str, name: str, qty: float, buy_price: float, note: str = "", tags=None):
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
    save_portfolio()
    save_history("add", item)
    st.success(f"{name} hinzugefügt.")
    check_notifications_for_item(item)

def remove_position(item_id: str):
    before = len(st.session_state.portfolio)
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    save_portfolio()
    save_history("remove", {"id": item_id})
    st.success("Position entfernt.")
    # update notifications state
    st.experimental_rerun()

def update_note(item_id: str, new_note: str, tags=None):
    if tags is None:
        tags = []
    for p in st.session_state.portfolio:
        if p["id"] == item_id:
            p["note"] = new_note
            p["tags"] = tags
    save_portfolio()
    save_history("note_update", {"id": item_id, "note": new_note, "tags": tags})
    st.success("Notiz & Tags gespeichert.")

# -----------------------
# Analytics helpers
# -----------------------
def current_price_for(item: dict):
    base = 100.0
    if item["category"].lower().startswith("krypto"):
        base = 1000.0
    elif item["category"].lower().startswith("akt"):
        base = 50.0
    elif item["asset_id"].startswith("ETF"):
        base = 120.0
    series = generate_series(item["asset_id"], 365, start_price=item["buy_price"] if item["buy_price"] > 0 else base)
    return series[-1]["price"]

def portfolio_snapshot():
    tot_value = 0.0
    tot_cost = 0.0
    rows = []
    for item in st.session_state.portfolio:
        cur = current_price_for(item)
        qty = float(item["qty"])
        value = cur * qty
        cost = float(item["buy_price"]) * qty
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct})
        tot_value += value
        tot_cost += cost
    return {"rows": rows, "total_value": tot_value, "total_cost": tot_cost}

def calc_volatility_label(series):
    if not series or len(series) < 31:
        return "Unbekannt", 0.0
    prices = [p["price"] for p in series[-31:]]
    returns = []
    for i in range(1, len(prices)):
        prev = prices[i - 1]
        curr = prices[i]
        if prev > 0:
            returns.append((curr - prev) / prev)
    if len(returns) < 10:
        return "Unbekannt", 0.0
    vol = stdev(returns)
    if vol < 0.01:
        return "Niedrig", vol
    elif vol < 0.03:
        return "Mittel", vol
    else:
        return "Hoch", vol

def rebalance_advice(target_alloc):
    snap = portfolio_snapshot()
    total = snap["total_value"] if snap["total_value"] > 0 else 1.0
    # current allocation by category
    cur_alloc = {}
    for k in target_alloc.keys():
        cur_alloc[k] = 0.0
    for r in snap["rows"]:
        cat = r["item"]["category"]
        cur_alloc[cat] = cur_alloc.get(cat, 0.0) + r["value"]
    for k in cur_alloc.keys():
        cur_alloc[k] = cur_alloc[k] / total
    advice = {}
    for k, v in target_alloc.items():
        advice[k] = v - cur_alloc.get(k, 0.0)
    return cur_alloc, advice

def simulate_over_time(days=365):
    if not st.session_state.portfolio:
        return []
    combined = [0.0] * days
    dates = [(datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat() for i in range(days)]
    for item in st.session_state.portfolio:
        series = generate_series(item["asset_id"], days, start_price=item["buy_price"] if item["buy_price"] > 0 else 100.0)
        for i in range(days):
            combined[i] += series[i]["price"] * item["qty"]
    return [{"date": dates[i], "value": round(combined[i], 4)} for i in range(days)]

def simulate_crash(percent_drop):
    snap = portfolio_snapshot()
    out = []
    for r in snap["rows"]:
        post_price = r["cur"] * (1 - percent_drop / 100.0)
        post_value = post_price * r["item"]["qty"]
        out.append({"id": r["item"]["id"], "name": r["item"]["name"], "pre_value": r["value"], "post_value": post_value, "delta": post_value - r["value"]})
    total_pre = snap["total_value"]
    total_post = sum(x["post_value"] for x in out)
    return {"items": out, "total_pre": total_pre, "total_post": total_post, "total_delta": total_post - total_pre}

def portfolio_stats():
    snap = portfolio_snapshot()
    rows = snap["rows"]
    if not rows:
        return {}
    pnls = [r["pnl"] for r in rows]
    values = [r["value"] for r in rows]
    avg = mean(pnls) if pnls else 0.0
    vol = stdev([r["cur"] for r in rows]) if len(rows) > 1 else 0.0
    best = max(rows, key=lambda x: x["pnl"])
    worst = min(rows, key=lambda x: x["pnl"])
    return {"avg_pnl": avg, "volatility": vol, "best": best, "worst": worst, "count": len(rows)}

# -----------------------
# Notifications / Alerts
# -----------------------
def show_toast(msg, level="info"):
    # st.toast exists only in newer Streamlit; fallback to st.info/warning
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
        # ultimate fallback
        if level == "warning":
            st.warning(msg)
        elif level == "success":
            st.success(msg)
        else:
            st.info(msg)

def check_notifications_for_item(item):
    # compute current pnl_pct if not present
    cur = current_price_for(item)
    cost = item["buy_price"] * item["qty"]
    value = cur * item["qty"]
    pnl = value - cost
    pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
    # Alert only if >10% loss and not already alerted for this item id in this session
    if pnl_pct < -10 and item["id"] not in st.session_state.alerts_sent:
        msg = f"⚠️ Verlust bei {item['name']}: {pnl_pct:.2f}%"
        st.session_state.notifications.append({"timestamp": datetime.utcnow().isoformat(), "message": msg})
        save_json(NOTIFICATIONS_FILE, st.session_state.notifications)
        st.session_state.alerts_sent.add(item["id"])
        show_toast(msg, level="warning")

def check_all_notifications_once():
    # run at startup to populate notifications
    for item in st.session_state.portfolio:
        check_notifications_for_item(item)

# Run initial notification check
check_all_notifications_once()

# -----------------------
# Search / Helpers
# -----------------------
def search_portfolio(query):
    query = query.lower()
    results = []
    for p in st.session_state.portfolio:
        if query in p["name"].lower() or query in p.get("note", "").lower() or any(query in t.lower() for t in p.get("tags", [])):
            results.append(p)
    return results

def search_all_assets(query):
    query = query.lower()
    results = []
    for a in ALL_ASSETS:
        if query in a["name"].lower() or query in a.get("symbol", "").lower() or query in a["id"].lower():
            results.append(a)
    return results

# -----------------------
# Sidebar navigation & quick actions
# -----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", [
    "Home",
    "Marktplatz",
    "Portfolio",
    "Rebalancing",
    "Simulation",
    "Statistiken",
    "Wissensbasis",
    "Export/Import",
    "Einstellungen"
])
st.sidebar.markdown("---")

st.sidebar.subheader("Schnellaktionen")
if st.sidebar.button("Portfolio exportieren"):
    export_obj = {"portfolio": st.session_state.portfolio, "settings": st.session_state.settings, "exported_at": datetime.utcnow().isoformat()}
    st.sidebar.download_button("Download JSON", data=json.dumps(export_obj, ensure_ascii=False, indent=2), file_name="portfolio_export.json", mime="application/json")
st.sidebar.markdown("---")

# quick search in sidebar
st.sidebar.write("Suche (Assets/Portfolio):")
side_search = st.sidebar.text_input("Suchbegriff", value="", key="side_search")
if side_search:
    results_a = search_all_assets(side_search)
    results_p = search_portfolio(side_search)
    st.sidebar.markdown(f"**Assets ({len(results_a)})**")
    for r in results_a:
        st.sidebar.write(f"- {r['name']} ({r.get('symbol','')})")
    st.sidebar.markdown(f"**Portfolio ({len(results_p)})**")
    for r in results_p:
        st.sidebar.write(f"- {r['name']}")

st.sidebar.write("Offline • Daten lokal in JSON")
st.sidebar.markdown("---")

# -----------------------
# Page: Home
# -----------------------
if page == "Home":
    st.title("🏠 Dashboard — Übersicht")
    snap = portfolio_snapshot()
    st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €   •   **Eingezahlt:** {snap['total_cost']:.2f} €")
    goal_val = float(st.session_state.settings.get("goal", 10000.0))
    prog = min(snap['total_value'] / goal_val if goal_val > 0 else 0.0, 1.0)
    st.progress(prog)
    st.markdown(f"Fortschritt zum Ziel ({goal_val:.2f} €): {prog*100:.2f}%")
    st.markdown("---")

    # Alerts
    if st.session_state.notifications:
        st.subheader("🔔 Benachrichtigungen")
        # show the last 5 notifications
        for n in st.session_state.notifications[-5:][::-1]:
            st.markdown(f"<div class='toast'>{n['timestamp']} — {n['message']}</div>", unsafe_allow_html=True)
    else:
        st.info("Keine Benachrichtigungen.")

    st.markdown("---")
    # Analyzer quick insights
    st.subheader("📈 Analyzer Insights")
    stats = portfolio_stats()
    if not stats:
        st.info("Portfolio leer — keine Insights verfügbar.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Anzahl Positionen", stats['count'])
        col2.metric("Durchschn. PnL", f"{stats['avg_pnl']:+.2f} €")
        col3.metric("Volatilität", f"{stats['volatility']:.4f}")
        st.markdown("**Top / Flop**")
        st.write(f"🏆 Best: {stats['best']['item']['name']} ({stats['best']['pnl']:+.2f} €)")
        st.write(f"📉 Worst: {stats['worst']['item']['name']} ({stats['worst']['pnl']:+.2f} €)")

    st.markdown("---")
    # mini portfolio historical
    st.subheader("Historische Portfolio-Entwicklung (simuliert)")
    hist = simulate_over_time(days=180)
    if hist:
        st.line_chart({p["date"]: p["value"] for p in hist})
    else:
        st.info("Keine historischen Daten (Portfolio leer).")

    st.markdown("---")
    # quick search box in Home
    st.subheader("Schnellsuche")
    q = st.text_input("Asset / Notiz suchen in Portfolio oder Marktplatz", key="home_search")
    if q:
        pa = search_portfolio(q)
        aa = search_all_assets(q)
        st.write(f"Portfolio Treffer: {len(pa)} — Marktplatz Treffer: {len(aa)}")
        for p in pa:
            st.markdown(f"- **{p['name']}** ({p['category']}) — Notiz: {p.get('note','')}")
        for a in aa:
            st.markdown(f"- {a['name']} ({a.get('symbol','')}) — id: {a['id']}")

# -----------------------
# Page: Marktplatz
# -----------------------
elif page == "Marktplatz":
    st.title("🏬 Marktplatz — ETFs, Kryptowährungen, Aktien")
    st.markdown("Wähle eine Kategorie, passe den Zeitraum an und füge Positionen deinem Portfolio hinzu.")
    intervals = {"1 Monat": 30, "3 Monate": 90, "6 Monate": 180, "1 Jahr": 365, "5 Jahre": 365 * 5}

    def show_assets_block(title, assets, currency_symbol="€"):
        st.subheader(title)
        for a in assets:
            with st.container():
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"### {a['name']} <span class='small'>({a.get('symbol','')})</span>", unsafe_allow_html=True)
                    sel = st.selectbox(f"Zeitraum {a['id']}", list(intervals.keys()), key=f"iv_{a['id']}")
                    days = intervals[sel]
                    series = generate_series(a['id'], days, start_price=100.0 + (abs(hash(a['id'])) % 1000) / 10)
                    # main chart
                    st.line_chart({p["date"]: p["price"] for p in series})
                    # SMA markers (display last values)
                    last = series[-1]
                    st.markdown(f"<div class='small'>Aktuell: {last['price']:.2f} {currency_symbol}  •  SMA20: {last['sma20']:.2f}  •  SMA50: {last['sma50']:.2f}</div>", unsafe_allow_html=True)
                    # sparkline
                    spark = series[-30:] if len(series) >= 30 else series
                    st.markdown("<div class='spark'>", unsafe_allow_html=True)
                    st.line_chart({p["date"]: p["price"] for p in spark})
                    st.markdown("</div>", unsafe_allow_html=True)
                with cols[1]:
                    cur = series[-1]["price"]
                    sma20 = series[-1]["sma20"]
                    sma50 = series[-1]["sma50"]
                    rec = "Kaufen" if sma20 > sma50 else "Nicht kaufen"
                    st.markdown(f"**Aktuell:** {cur:.2f} {currency_symbol}")
                    st.markdown(f"**Empfehlung:** {'🟢 ' + rec if rec == 'Kaufen' else '🔴 ' + rec}")
                    risk_label, vol = calc_volatility_label(series)
                    st.markdown(f"**Risiko:** {risk_label} (Vol: {vol:.4f})")
                    # add form
                    with st.form(key=f"add_{a['id']}", clear_on_submit=False):
                        qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                        buy_price = st.number_input("Kaufpreis pro Einheit", min_value=0.0001, value=float(cur), step=0.01, key=f"bp_{a['id']}")
                        note = st.text_area("Notiz (optional)", value="", key=f"note_{a['id']}", height=80)
                        tags_txt = st.text_input("Tags (kommagetrennt)", key=f"tags_{a['id']}")
                        if st.form_submit_button("Zu Portfolio hinzufügen"):
                            tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                            add_position(title, a['id'], a['name'], qty, buy_price, note, tags)

    show_assets_block("ETFs", ETFS, "€")
    show_assets_block("Kryptowährungen", CRYPTOS, "$")
    show_assets_block("Aktien", STOCKS, "€")

# -----------------------
# Page: Portfolio
# -----------------------
elif page == "Portfolio":
    st.title("💼 Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Dein Portfolio ist leer. Füge Assets im Marktplatz hinzu.")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €   •   **Eingezahlt:** {snap['total_cost']:.2f} €")
        goal_val = float(st.session_state.settings.get("goal", 10000.0))
        st.progress(min(snap['total_value'] / goal_val if goal_val > 0 else 0.0, 1.0))
        st.markdown(f"Fortschritt: {min(snap['total_value'] / goal_val if goal_val > 0 else 0.0, 1.0) * 100:.2f}%")
        st.markdown("---")
        # portfolio table + expandable details
        for r in snap["rows"]:
            item = r["item"]
            with st.expander(f"{item['name']} — Wert: {r['value']:.2f} €  •  PnL: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)"):
                cols = st.columns([3, 1])
                with cols[0]:
                    st.write(f"**{item['name']}** — Kategorie: {item['category']}")
                    st.write(f"Menge: {item['qty']} • Kaufpreis: {item['buy_price']:.2f} €")
                    st.write(f"Aktuell: {r['cur']:.4f} €  •  Wert: {r['value']:.2f} €")
                    st.write("Notiz:")
                    st.write(item.get("note", "—"))
                    # small history
                    small = generate_series(item['asset_id'], 60, start_price=item['buy_price'])
                    st.line_chart({p["date"]: p["price"] for p in small})
                with cols[1]:
                    rec = "Halten" if r['pnl'] >= 0 else "Verkaufen"
                    if r['pnl'] >= 0:
                        st.success(f"Empfehlung: {rec}")
                    else:
                        st.error(f"Empfehlung: {rec}")
                    new_note = st.text_area("Notiz bearbeiten", value=item.get("note", ""), key=f"note_edit_{item['id']}", height=120)
                    tags_txt = st.text_input("Tags (kommagetrennt)", value=",".join(item.get("tags", [])), key=f"tags_edit_{item['id']}")
                    if st.button("Notiz & Tags speichern", key=f"save_note_{item['id']}"):
                        tags = [t.strip() for t in tags_txt.split(",") if t.strip()]
                        update_note(item['id'], new_note, tags)
                    if st.button("Position entfernen", key=f"remove_{item['id']}"):
                        remove_position(item['id'])
        st.markdown("---")
        # Portfolio summary table
        st.subheader("Portfolio Zusammenfassung")
        st.write(f"Positionen: {len(snap['rows'])}  •  Gesamtwert: {snap['total_value']:.2f} €  •  Eingezahlt: {snap['total_cost']:.2f} €")

# -----------------------
# Page: Rebalancing
# -----------------------
elif page == "Rebalancing":
    st.title("⚖️ Rebalancing")
    st.markdown("Lege eine Zielallokation pro Kategorie fest (Summe = 1.0). Die App gibt einfache Kauf/Verkauf-Empfehlungen.")
    defaults = {"ETF": 0.5, "Aktie": 0.3, "Krypto": 0.2}
    col1, col2, col3 = st.columns(3)
    tgt_etf = col1.number_input("ETF", min_value=0.0, max_value=1.0, value=float(defaults["ETF"]), step=0.05, key="tgt_etf")
    tgt_act = col2.number_input("Aktie", min_value=0.0, max_value=1.0, value=float(defaults["Aktie"]), step=0.05, key="tgt_act")
    tgt_crp = col3.number_input("Krypto", min_value=0.0, max_value=1.0, value=float(defaults["Krypto"]), step=0.05, key="tgt_crp")
    total = tgt_etf + tgt_act + tgt_crp
    if abs(total - 1.0) > 1e-6:
        st.warning(f"Zielallokation sumiert zu {total:.2f} — bitte auf 1.0 einstellen.")
    else:
        target = {"ETF": tgt_etf, "Aktie": tgt_act, "Krypto": tgt_crp}
        cur_alloc, advice = rebalance_advice(target)
        st.markdown("**Aktuelle Allokation (geschätzt)**")
        for k, v in cur_alloc.items():
            st.write(f"- {k}: {v*100:.2f}%")
        st.markdown("**Empfehlungen (positiv = kaufen, negativ = verkaufen)**")
        for k, v in advice.items():
            st.write(f"- {k}: {(v*100):+.2f}%")

# -----------------------
# Page: Simulation
# -----------------------
elif page == "Simulation":
    st.title("🔬 Simulation & Crash-Test")
    st.markdown("Historische Entwicklung (simuliert) und Crash-Simulation (einmaliger %-Drop).")
    days = st.selectbox("Zeitraum (Tage)", [90, 180, 365, 365*3], index=2)
    hist = simulate_over_time(days)
    if hist:
        st.line_chart({p["date"]: p["value"] for p in hist})
    else:
        st.info("Portfolio leer — keine Simulation möglich.")
    st.markdown("---")
    st.subheader("Crash-Simulation")
    drop = st.slider("Simulierter %-Drop", 1, 100, 15)
    if st.button("Crash simulieren"):
        if not st.session_state.portfolio:
            st.info("Portfolio leer — nichts zu simulieren.")
        else:
            res = simulate_crash(drop)
            st.write(f"Vorher: {res['total_pre']:.2f} €  •  Nachher: {res['total_post']:.2f} €  •  Änderung: {res['total_delta']:+.2f} €")
            st.table([{"Asset": x["name"], "Vorher": round(x["pre_value"], 2), "Nachher": round(x["post_value"], 2), "Delta": round(x["delta"], 2)} for x in res["items"]])

# -----------------------
# Page: Statistiken
# -----------------------
elif page == "Statistiken":
    st.title("📊 Statistiken & Heatmap")
    stats = portfolio_stats()
    if not stats:
        st.info("Keine Statistikdaten — Portfolio leer.")
    else:
        st.markdown("**Übersicht**")
        st.write(f"Anzahl Positionen: {stats['count']}")
        st.write(f"Durchschnittlicher Gewinn/Verlust: {stats['avg_pnl']:+.2f} €")
        st.write(f"Volatilität (Snapshot): {stats['volatility']:.4f}")
        st.write(f"Best / Worst: {stats['best']['item']['name']} ({stats['best']['pnl']:+.2f} €) / {stats['worst']['item']['name']} ({stats['worst']['pnl']:+.2f} €)")
        st.markdown("---")
        snap = portfolio_snapshot()
        # value per category bar chart
        cat_vals = {}
        for r in snap['rows']:
            cat_vals[r['item']['category']] = cat_vals.get(r['item']['category'], 0.0) + r['value']
        st.bar_chart(cat_vals)
        st.markdown("---")
        # heatmap-like bars (colored divs)
        st.write("Heatmap (grün=gut, rot=schlecht)")
        for r in snap['rows']:
            pct = r['pnl_pct']
            color = "#006600" if pct >= 0 else "#660000"
            width = min(max(abs(pct) * 1.5, 2), 100)
            st.markdown(f"<div style='background:{color}; width:{width}%; padding:8px; border-radius:6px; margin-bottom:6px;'>{r['item']['name']} — {r['pnl']:+.2f} € ({pct:+.2f}%)</div>", unsafe_allow_html=True)

# -----------------------
# Page: Wissensbasis (erweitert)
# -----------------------
elif page == "Wissensbasis":
    st.title("📘 Wissensbasis — Lernen & Beispiele")
    guides = st.session_state.guides or {}
    st.write("Die Wissensbasis ist offline. Du kannst die Texte editieren und Beispiel-Graphs anzeigen.")
    # allow adding/editing guides
    if st.button("Neuen Guide hinzufügen"):
        new_key = f"Guide_{int(datetime.utcnow().timestamp())}"
        guides[new_key] = "Neuer Leitfaden ... Text hier bearbeiten."
        st.session_state.guides = guides
        save_json(GUIDES_FILE, guides)
    for k, text in guides.items():
        with st.expander(k.replace("_", " ")):
            st.write(text)
            if st.button(f"Bearbeiten: {k}", key=f"edit_{k}"):
                new = st.text_area("Neuen Text eingeben", value=text, key=f"txt_{k}", height=200)
                guides[k] = new
                st.session_state.guides = guides
                save_json(GUIDES_FILE, guides)
                st.success("Guide gespeichert.")
            # show an example colored graph for each guide to illustrate
            sample_series = generate_series("EX_" + k, 90, start_price=50.0)
            st.line_chart({p["date"]: p["price"] for p in sample_series})
            # highlight SMA example
            st.markdown("**Beispiel: SMA20 vs SMA50**")
            st.markdown(f"Letzte SMA20: {sample_series[-1]['sma20']} — SMA50: {sample_series[-1]['sma50']}")

# -----------------------
# Page: Export / Import
# -----------------------
elif page == "Export/Import":
    st.title("📤 Export & 📥 Import")
    st.write("Exportiere dein Portfolio + Einstellungen oder importiere ein Backup.")
    export_obj = {
        "portfolio": st.session_state.portfolio,
        "settings": st.session_state.settings,
        "history": st.session_state.history,
        "exported_at": datetime.utcnow().isoformat()
    }
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2)
    st.download_button("Exportiere Backup (JSON)", data=export_json, file_name="portfolio_backup.json", mime="application/json")
    st.markdown("---")
    uploaded = st.file_uploader("Importiere Backup (JSON)", type=["json"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8")
            obj = json.loads(raw)
            if "portfolio" in obj and isinstance(obj["portfolio"], list):
                st.session_state.portfolio = obj["portfolio"]
                save_portfolio()
            if "settings" in obj and isinstance(obj["settings"], dict):
                st.session_state.settings = obj["settings"]
                save_json(SETTINGS_FILE, st.session_state.settings)
            if "history" in obj and isinstance(obj["history"], list):
                st.session_state.history = obj["history"]
                save_json(HISTORY_FILE, st.session_state.history)
            st.success("Import erfolgreich. Seite wird neu geladen.")
            st.experimental_rerun()
        except Exception as e:
            st.error(f"Import fehlgeschlagen: {e}")

# -----------------------
# Page: Einstellungen
# -----------------------
elif page == "Einstellungen":
    st.title("⚙️ Einstellungen")
    st.write("Verwaltung: Zielbetrag, Cache, Passwort, Benachrichtigungen.")
    # Goal
    goal_val = float(st.session_state.settings.get("goal", 10000.0))
    new_goal = st.number_input("Finanzziel (gesamt)", min_value=0.0, value=goal_val, step=100.0)
    if st.button("Ziel speichern"):
        st.session_state.settings["goal"] = float(new_goal)
        save_json(SETTINGS_FILE, st.session_state.settings)
        st.success("Ziel gespeichert.")
    st.markdown("---")
    # Cache clear
    if st.button("Chart-Cache löschen"):
        st.session_state.series_cache = {}
        st.success("Chart-Cache geleert.")
    st.markdown("---")
    # Notifications list
    st.subheader("Benachrichtigungen (letzte 20)")
    for n in st.session_state.notifications[-20:][::-1]:
        st.write(f"{n['timestamp']}: {n['message']}")
    if st.button("Benachrichtigungen leeren"):
        st.session_state.notifications = []
        save_json(NOTIFICATIONS_FILE, st.session_state.notifications)
        st.success("Benachrichtigungen geleert.")
    st.markdown("---")
    # Password change
    st.subheader("Eigentümer: Passwort ändern")
    auth = st.session_state.settings.get("auth")
    if not auth:
        st.warning("Auth nicht gesetzt — erstelle ein Passwort auf dem Login-Bildschirm.")
    else:
        old = st.text_input("Altes Passwort", type="password", key="chg_old")
        new = st.text_input("Neues Passwort", type="password", key="chg_new")
        new2 = st.text_input("Neues Passwort wiederholen", type="password", key="chg_new2")
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
                    st.session_state.settings["auth"] = {
                        "salt": binascii.hexlify(new_salt).decode(),
                        "key": binascii.hexlify(new_dk).decode(),
                        "iterations": 200_000,
                        "dklen": 72
                    }
                    save_json(SETTINGS_FILE, st.session_state.settings)
                    st.success("Passwort geändert.")
            except Exception as e:
                st.error(f"Fehler: {e}")

# -----------------------
# Footer
# -----------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Daten lokal in portfolio.json / settings.json / history.json / notifications.json • Keine Internet-APIs nötig • Deterministische Simulationen</div>", unsafe_allow_html=True)
