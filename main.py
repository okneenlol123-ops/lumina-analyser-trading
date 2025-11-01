# main.py
# Offline-Finanz-App — Black Edition — Single Owner, Passwort mit 72-Byte PBKDF2
import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# -------------------
# Config & Files
# -------------------
st.set_page_config(page_title="Finanz-Platform (Offline, Pro)", page_icon="💹", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"
GUIDES_FILE = "guides.json"

# -------------------
# Styling (Black + subtle animations)
# -------------------
st.markdown("""
<style>
html, body, [class*="css"] {background:#000 !important; color:#e6eef6 !important;}
.stButton>button {background:#111; color:#e6eef6; border:1px solid #222; border-radius:6px;}
.card {background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px;}
.small {color:#9aa6b2; font-size:13px;}
.gain {background:linear-gradient(90deg,#00ff88, #007744); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(0,255,136,0.08);}
.loss {background:linear-gradient(90deg,#ff4466,#770022); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(255,68,102,0.06);}
.badge {background:#111; color:#e6eef6; padding:4px 8px; border-radius:6px; border:1px solid #222; display:inline-block;}
.spark {height:48px;}
@keyframes glow {0% { box-shadow:0 0 6px rgba(0,255,136,0.04);}50%{box-shadow:0 0 18px rgba(0,255,136,0.10);}100%{box-shadow:0 0 6px rgba(0,255,136,0.04);}}
</style>
""", unsafe_allow_html=True)

# -------------------
# Utility: JSON load/save
# -------------------
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

# -------------------
# Initialize persistent files
# -------------------
if not os.path.exists(SETTINGS_FILE):
    # empty settings; password must be initialised
    save_json(SETTINGS_FILE, {})

if not os.path.exists(GUIDES_FILE):
    # basic offline guides (editable)
    guides = {
        "ETF_vs_Aktie": "ETFs bündeln viele Aktien. Einfacher Diversifikationsmechanismus.",
        "Volatilitaet": "Volatilität misst die Schwankungsbreite der Renditen. Hohe Volatilität = höheres Risiko.",
        "Rebalancing": "Rebalancing stellt die Zielallokation wieder her, z. B. 50/30/20."
    }
    save_json(GUIDES_FILE, guides)

if not os.path.exists(PORTFOLIO_FILE):
    save_json(PORTFOLIO_FILE, [])

if not os.path.exists(HISTORY_FILE):
    save_json(HISTORY_FILE, [])

# -------------------
# Session state init
# -------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_json(PORTFOLIO_FILE, [])

if "settings" not in st.session_state:
    st.session_state.settings = load_json(SETTINGS_FILE, {})

if "history" not in st.session_state:
    st.session_state.history = load_json(HISTORY_FILE, [])

if "series_cache" not in st.session_state:
    st.session_state.series_cache = {}

# -------------------
# Security: Single-owner password handling
# - PBKDF2-HMAC-SHA256 with dklen=72 (72 bytes)
# - store salt (hex) and derived key (hex) in settings.json
# -------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    # returns raw bytes
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def setup_password_flow():
    settings = st.session_state.settings or {}
    if not settings.get("auth"):
        st.header("Erstinstallation: Eigentümer-Passwort setzen")
        st.info("Du bist der einzige Besitzer. Dein Passwort wird sicher (PBKDF2) gespeichert.")
        pwd = st.text_input("Passwort wählen", type="password")
        pwd2 = st.text_input("Passwort wiederholen", type="password")
        if st.button("Passwort setzen"):
            if not pwd or pwd != pwd2:
                st.error("Passwörter leer oder stimmen nicht überein.")
                return False
            salt = os.urandom(16)
            dk = derive_key(pwd, salt, iterations=200_000, dklen=72)
            settings["auth"] = {
                "salt": binascii.hexlify(salt).decode(),
                "key": binascii.hexlify(dk).decode(),
                "iterations": 200_000,
                "dklen": 72
            }
            st.session_state.settings = settings
            save_json(SETTINGS_FILE, settings)
            st.success("Passwort gesetzt. Bitte neu einloggen.")
            return False
        return False
    return True

def login_flow():
    settings = st.session_state.settings
    auth = settings.get("auth", {})
    if not auth:
        return False
    st.header("Login (Eigentümer)")
    pwd = st.text_input("Passwort", type="password")
    if st.button("Einloggen"):
        salt = binascii.unhexlify(auth["salt"])
        iterations = auth.get("iterations", 200_000)
        dklen = auth.get("dklen", 72)
        dk = derive_key(pwd, salt, iterations=iterations, dklen=dklen)
        if binascii.hexlify(dk).decode() == auth["key"]:
            st.session_state.auth_ok = True
            st.success("Erfolgreich eingeloggt.")
            return True
        else:
            st.error("Falsches Passwort.")
            return False
    return False

# Authentication gating
auth_ready = setup_password_flow()
if not auth_ready:
    # password setup page handles its own UI; stop here
    st.stop()

# require login unless session already auth_ok
if not st.session_state.get("auth_ok", False):
    ok = login_flow()
    if not ok:
        st.stop()

# -------------------
# Asset lists (examples)
# -------------------
ETFS = [
    {"id":"ETF_DE","name":"Deutschland"},
    {"id":"ETF_US","name":"USA"},
    {"id":"ETF_EU","name":"Europa"}
]
CRYPTOS = [
    {"id":"CR_BTC","name":"Bitcoin"},
    {"id":"CR_ETH","name":"Ethereum"},
    {"id":"CR_SOL","name":"Solana"}
]
STOCKS = [
    {"id":"ST_AAPL","name":"Apple"},
    {"id":"ST_TSLA","name":"Tesla"},
    {"id":"ST_MSFT","name":"Microsoft"}
]

# -------------------
# Deterministic price series generator (cached)
# -------------------
def deterministic_seed(s:str)->int:
    return int(hashlib.sha256(s.encode()).hexdigest(),16) % (2**31)

def generate_series(asset_id, days=365, start_price=100.0):
    key = f"{asset_id}_{days}"
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
        series.append({"date":date,"price":round(price,4)})
    # add SMA20/50
    for i in range(len(series)):
        pvals20 = [series[j]["price"] for j in range(max(0,i-19), i+1)]
        pvals50 = [series[j]["price"] for j in range(max(0,i-49), i+1)]
        series[i]["sma20"] = round(mean(pvals20),4)
        series[i]["sma50"] = round(mean(pvals50),4)
    st.session_state.series_cache[key] = series
    return series

# -------------------
# Portfolio operations (persist)
# -------------------
def save_portfolio_state():
    save_json(PORTFOLIO_FILE, st.session_state.portfolio)

def save_history_event(action, item):
    hist = st.session_state.history
    hist.append({"timestamp":datetime.utcnow().isoformat(),"action":action,"item":item})
    save_json(HISTORY_FILE, hist)

def add_position(category, asset_id, name, qty, buy_price, note=""):
    item = {
        "id": f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "qty": float(qty),
        "buy_price": float(buy_price),
        "note": note,
        "added_at": datetime.utcnow().isoformat()
    }
    st.session_state.portfolio.append(item)
    save_portfolio_state()
    save_history_event("add", item)
    st.success(f"{name} hinzugefügt.")

def remove_position(item_id):
    before = len(st.session_state.portfolio)
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    save_portfolio_state()
    save_history_event("remove", {"id":item_id})
    st.success("Position entfernt.")
    st.experimental_rerun()

def update_note(item_id, new_note):
    for p in st.session_state.portfolio:
        if p["id"] == item_id:
            p["note"] = new_note
    save_portfolio_state()
    save_history_event("note_update", {"id":item_id,"note":new_note})
    st.success("Notiz gespeichert.")

# -------------------
# Analytics helpers: Value, PnL, rebalancing, historical portfolio sim, crash sim, stats
# -------------------
def current_price_for(item):
    # derive base start price from id for determinism
    base = 100.0
    if item["category"].lower().startswith("krypto"):
        base = 1000.0
    elif item["category"].lower().startswith("akt"):
        base = 50.0
    else:
        base = 120.0
    series = generate_series(item["asset_id"], 365, start_price=item["buy_price"] if item["buy_price"]>0 else base)
    return series[-1]["price"]

def portfolio_snapshot():
    tot_value = 0.0
    tot_cost = 0.0
    rows = []
    for item in st.session_state.portfolio:
        cur = current_price_for(item)
        qty = item["qty"]
        value = cur * qty
        cost = item["buy_price"] * qty
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
        rows.append({"item":item,"cur":cur,"value":value,"cost":cost,"pnl":pnl,"pnl_pct":pnl_pct})
        tot_value += value
        tot_cost += cost
    return {"rows":rows,"total_value":tot_value,"total_cost":tot_cost}

def rebalance_advice(target_alloc):
    """
    target_alloc: dict category -> fraction (sum to 1)
    returns: current allocation and simple advice list
    """
    snap = portfolio_snapshot()
    total = snap["total_value"] if snap["total_value"]>0 else 1.0
    cur_alloc = {}
    for c in target_alloc:
        cur_alloc[c] = 0.0
    for r in snap["rows"]:
        cat = r["item"]["category"]
        cur_alloc[cat] = cur_alloc.get(cat,0.0) + r["value"]
    for k in cur_alloc:
        cur_alloc[k] = cur_alloc[k]/total
    # advice: difference
    advice = {}
    for k,v in target_alloc.items():
        diff = v - cur_alloc.get(k,0.0)
        advice[k] = diff  # positive -> need to buy more, negative -> sell
    return cur_alloc, advice

def simulate_portfolio_over_time():
    """
    Returns timeseries of portfolio value using stored portfolio and series.
    We'll take each position series (365 days) and compute historical combined value.
    """
    if not st.session_state.portfolio:
        return []
    days = 365
    combined = [0.0]*days
    dates = []
    for i in range(days):
        dates.append((datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat())
    for item in st.session_state.portfolio:
        series = generate_series(item["asset_id"], days, start_price=item["buy_price"] if item["buy_price"]>0 else 100.0)
        for i in range(days):
            combined[i] += series[i]["price"] * item["qty"]
    return [{"date":dates[i],"value":round(combined[i],4)} for i in range(days)]

def simulate_crash(percent_drop):
    """
    Simulate instantaneous drop by percent_drop (%) for all assets and return snapshot after drop.
    """
    snap = portfolio_snapshot()
    out = []
    for r in snap["rows"]:
        post_price = r["cur"] * (1 - percent_drop/100.0)
        post_value = post_price * r["item"]["qty"]
        out.append({"id":r["item"]["id"], "name":r["item"]["name"], "pre_value":r["value"], "post_value":post_value, "delta":post_value - r["value"]})
    total_pre = snap["total_value"]
    total_post = sum(x["post_value"] for x in out)
    return {"items":out,"total_pre":total_pre,"total_post":total_post, "total_delta": total_post - total_pre}

def portfolio_statistics():
    snap = portfolio_snapshot()
    rows = snap["rows"]
    if not rows:
        return {}
    pnls = [r["pnl"] for r in rows]
    values = [r["value"] for r in rows]
    avg_pnl = mean(pnls) if pnls else 0.0
    vol = stdev([r["cur"] for r in rows]) if len(rows)>1 else 0.0
    best = max(rows, key=lambda x: x["pnl"])
    worst = min(rows, key=lambda x: x["pnl"])
    return {"avg_pnl":avg_pnl,"volatility":vol,"best":best,"worst":worst,"count":len(rows)}

# -------------------
# UI: Sidebar navigation + quick actions
# -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Marktplatz","Portfolio","Rebalancing","Simulation","Statistiken","Wissensbasis","Export/Import","Einstellungen"])
st.sidebar.markdown("---")
st.sidebar.subheader("Schnellaktionen")
if st.sidebar.button("Portfolio exportieren"):
    export_obj = {"portfolio": st.session_state.portfolio, "settings": st.session_state.settings, "exported_at": datetime.utcnow().isoformat()}
    st.sidebar.download_button("Download JSON", data=json.dumps(export_obj, ensure_ascii=False, indent=2), file_name="portfolio_export.json", mime="application/json")
st.sidebar.markdown("---")
st.sidebar.write("Offline • Alles lokal gespeichert")

# -------------------
# Page: Marktplatz
# -------------------
def show_assets(assets, category_label, currency_symbol="€"):
    st.subheader(category_label)
    intervals = {"1 Monat":30,"3 Monate":90,"6 Monate":180,"1 Jahr":365}
    for a in assets:
        cols = st.columns([3,1])
        with cols[0]:
            st.markdown(f"**{a['name']}**")
            days = st.selectbox(f"Zeitraum {a['id']}", list(intervals.keys()), key=f"mk_{a['id']}")
            series = generate_series(a["id"], intervals[days], start_price=100.0)
            st.line_chart({p["date"]:p["price"] for p in series})
            # sparkline small
            last = series[-20:] if len(series)>=20 else series
            st.markdown("<div class='spark'>", unsafe_allow_html=True)
            st.line_chart({p["date"]:p["price"] for p in last})
            st.markdown("</div>", unsafe_allow_html=True)
        with cols[1]:
            cur = series[-1]["price"]
            sma20 = series[-1]["sma20"]
            sma50 = series[-1]["sma50"]
            st.write(f"**Aktuell:** {cur:.2f} {currency_symbol}")
            st.write(f"SMA20: {sma20:.2f} | SMA50: {sma50:.2f}")
            risk_label, vol = calc_volatility_label(series)
            rec = "Kaufen" if sma20 > sma50 else "Nicht kaufen"
            st.markdown(f"**Empfehlung:** {'🟢 ' + rec if rec=='Kaufen' else '🔴 ' + rec}")
            st.markdown(f"**Risiko:** {risk_label} ({vol:.4f})")
            # add form
            with st.form(key=f"add_{a['id']}"):
                qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                buy_price = st.number_input("Kaufpreis", min_value=0.0001, value=float(cur), step=0.01, key=f"bp_{a['id']}")
                note = st.text_area("Notiz", value="", key=f"note_{a['id']}", height=80)
                if st.form_submit_button("Hinzufügen"):
                    add_position(category_label, a["id"], f"{a['name']}", qty, buy_price, note)

# volatility helper (reused)
def calc_volatility_label(series):
    if not series or len(series)<31:
        return "Unbekannt", 0.0
    prices = [p["price"] for p in series[-31:]]
    returns = []
    for i in range(1,len(prices)):
        if prices[i-1]>0:
            returns.append((prices[i]-prices[i-1])/prices[i-1])
    if len(returns)<10:
        return "Unbekannt", 0.0
    vol = stdev(returns)
    if vol<0.01: lbl="Niedrig"
    elif vol<0.03: lbl="Mittel"
    else: lbl="Hoch"
    return lbl, vol

if page=="Marktplatz":
    st.title("Marktplatz")
    st.markdown("Wähle Assets, prüfe Graphen, füge Positionen mit Notiz hinzu.")
    show_assets(ETFS,"ETF","€")
    show_assets(CRYPTOS,"Krypto","$")
    show_assets(STOCKS,"Aktie","€")

# -------------------
# Page: Portfolio
# -------------------
elif page=="Portfolio":
    st.title("Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Portfolio leer.")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €  •  Eingezahlt: {snap['total_cost']:.2f} €")
        # Goal progress
        goal = float(st.session_state.settings.get("goal",10000.0))
        progress = min(snap['total_value']/goal if goal>0 else 0.0, 1.0)
        st.progress(progress)
        st.markdown(f"Fortschritt: {progress*100:.2f}%")
        st.markdown("---")
        for r in snap["rows"]:
            item = r["item"]
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            cols = st.columns([2,1])
            with cols[0]:
                st.write(f"**{item['name']}** ({item['category']})")
                st.write(f"Menge: {item['qty']} • Kaufpreis: {item['buy_price']:.2f} €")
                st.write(f"Aktuell: {r['cur']:.4f} €  •  Wert: {r['value']:.2f} €")
                st.write(f"Gewinn/Verlust: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)")
                small_series = generate_series(item["asset_id"], 40, start_price=item['buy_price'])
                st.line_chart({p["date"]:p["price"] for p in small_series})
            with cols[1]:
                rec = "Halten" if r['pnl']>=0 else "Verkaufen"
                if r['pnl']>=0:
                    st.success(f"{rec}")
                else:
                    st.error(f"{rec}")
                new_note = st.text_area("Notiz", value=item.get("note",""), key=f"note_edit_{item['id']}", height=80)
                if st.button("Speichern", key=f"save_note_{item['id']}"):
                    update_note(item['id'], new_note)
                if st.button("Entfernen", key=f"remove_{item['id']}"):
                    remove_position(item['id'])
            st.markdown("</div>", unsafe_allow_html=True)

# -------------------
# Page: Rebalancing
# -------------------
elif page=="Rebalancing":
    st.title("Rebalancing")
    st.markdown("Lege Zielallokation fest (Summe = 1.0). Die App gibt Vorschläge, wie du ausgleichen kannst.")
    # default targets
    defaults = {"ETF":0.5,"Aktie":0.3,"Krypto":0.2}
    tgt_ETF = st.number_input("ETF (z.B. 0.5)", min_value=0.0, max_value=1.0, value=float(defaults["ETF"]), step=0.05, key="t1")
    tgt_Aktie = st.number_input("Aktie (z.B. 0.3)", min_value=0.0, max_value=1.0, value=float(defaults["Aktie"]), step=0.05, key="t2")
    tgt_Krypto = st.number_input("Krypto (z.B. 0.2)", min_value=0.0, max_value=1.0, value=float(defaults["Krypto"]), step=0.05, key="t3")
    total = tgt_ETF + tgt_Aktie + tgt_Krypto
    if abs(total - 1.0) > 1e-6:
        st.warning(f"Zielallokation sumiert zu {total:.2f} — bitte auf 1.0 bringen.")
    else:
        target_alloc = {"ETF":tgt_ETF,"Aktie":tgt_Aktie,"Krypto":tgt_Krypto}
        cur_alloc, advice = rebalance_advice(target_alloc)
        st.markdown("**Aktuelle Allokation:**")
        for k,v in cur_alloc.items():
            st.write(f"{k}: {v*100:.2f}%")
        st.markdown("**Empfehlungen (positiv = kaufen, negativ = verkaufen)**")
        for k,v in advice.items():
            st.write(f"{k}: {v*100:+.2f}%")

# -------------------
# Page: Simulation (historical + crash)
# -------------------
elif page=="Simulation":
    st.title("Simulationen")
    st.markdown("Historische Portfolioentwicklung (basierend auf simulierten Serien) & Crash-Simulation.")
    hist = simulate_portfolio_over_time()
    if hist:
        st.line_chart({p["date"]:p["value"] for p in hist})
        st.markdown("**Crash-Simulation**")
        drop = st.slider("Simulierter %-Drop", 1, 100, 10)
        if st.button("Crash simulieren"):
            res = simulate_crash(drop)
            st.write(f"Vorher: {res['total_pre']:.2f} € • Nachher: {res['total_post']:.2f} € • Änderung: {res['total_delta']:+.2f} €")
            st.table([{"Asset":x["name"], "Vorher":x["pre_value"], "Nachher":x["post_value"], "Delta":x["delta"]} for x in res["items"]])
    else:
        st.info("Portfolio leer — füge zuerst Positionen hinzu.")

# -------------------
# Page: Statistiken & Heatmap
# -------------------
elif page=="Statistiken":
    st.title("Statistiken & Heatmap")
    stats = portfolio_statistics()
    if not stats:
        st.info("Keine Statistikdaten (Portfolio leer).")
    else:
        st.write(f"Anzahl Positionen: {stats['count']}")
        st.write(f"Durchschn. Gewinn/Verlust: {stats['avg_pnl']:+.2f} €")
        st.write(f"Volatilität (Portfolio-Snapshot): {stats['volatility']:.4f}")
        st.write(f"Bester: {stats['best']['item']['name']} ({stats['best']['pnl']:+.2f} €)")
        st.write(f"Schlechtester: {stats['worst']['item']['name']} ({stats['worst']['pnl']:+.2f} €)")
        st.markdown("---")
        # simple heatmap by pnl_pct ranges (no external libs; use colored divs)
        snap = portfolio_snapshot()
        rows = snap["rows"]
        st.write("Heatmap (grün=gut, rot=schlecht):")
        for r in rows:
            pct = r["pnl_pct"]
            color = "#006600" if pct>=0 else "#660000"
            width = min(max(abs(pct)*1.5,2),100)
            st.markdown(f"<div style='background:{color}; width:{width}%; padding:8px; border-radius:6px; margin-bottom:6px;'>{r['item']['name']} — {r['pnl']:+.2f} € ({pct:+.2f}%)</div>", unsafe_allow_html=True)

# -------------------
# Page: Wissensbasis
# -------------------
elif page=="Wissensbasis":
    st.title("Wissensbasis (offline)")
    guides = load_json(GUIDES_FILE, {})
    for k,txt in guides.items():
        st.markdown(f"**{k.replace('_',' ')}**")
        st.write(txt)
        st.markdown("---")

# -------------------
# Page: Export/Import
# -------------------
elif page=="Export/Import":
    st.title("Export / Import")
    st.write("Exportiere dein Portfolio & Einstellungen oder importiere ein Backup (JSON).")
    export_obj = {"portfolio": st.session_state.portfolio, "settings": st.session_state.settings, "exported_at": datetime.utcnow().isoformat()}
    st.download_button("Export als JSON", data=json.dumps(export_obj, ensure_ascii=False, indent=2), file_name="portfolio_export.json", mime="application/json")
    uploaded = st.file_uploader("Importiere JSON (portfolio_export.json)", type=["json"])
    if uploaded:
        try:
            raw = uploaded.read().decode("utf-8")
            obj = json.loads(raw)
            if "portfolio" in obj:
                st.session_state.portfolio = obj["portfolio"]
                save_portfolio_state()
            if "settings" in obj:
                st.session_state.settings = obj["settings"]
                save_json(SETTINGS_FILE, st.session_state.settings)
            st.success("Importiert.")
        except Exception as e:
            st.error("Fehler beim Import. Datei prüfen.")

# -------------------
# Page: Einstellungen
# -------------------
elif page=="Einstellungen":
    st.title("Einstellungen")
    st.write("Verwalte Ziel, Cache und Eigentümer-Konto.")
    goal = st.number_input("Finanzziel (gesamt)", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Speichere Ziel"):
        st.session_state.settings["goal"] = float(goal)
        save_json(SETTINGS_FILE, st.session_state.settings)
        st.success("Ziel gespeichert.")
    st.markdown("---")
    if st.button("Cache (Series) löschen"):
        st.session_state.series_cache = {}
        st.success("Cache gelöscht.")
    st.markdown("---")
    st.write("Eigentümerkontrolle:")
    if st.button("Passwort ändern"):
        # simple flow: require old pwd then set new
        old = st.text_input("Altes Passwort", type="password", key="chg_old")
        new = st.text_input("Neues Passwort", type="password", key="chg_new")
        new2 = st.text_input("Neues Passwort wiederholen", type="password", key="chg_new2")
        if st.button("Bestätige Passwortänderung"):
            auth = st.session_state.settings.get("auth")
            if not auth:
                st.error("Auth fehlt.")
            else:
                salt = binascii.unhexlify(auth["salt"])
                dk = derive_key(old, salt, iterations=auth.get("iterations",200_000), dklen=auth.get("dklen",72))
                if binascii.hexlify(dk).decode() != auth["key"]:
                    st.error("Altes Passwort falsch.")
                elif new != new2 or not new:
                    st.error("Neues Passwort leer oder Wiederholung falsch.")
                else:
                    new_salt = os.urandom(16)
                    new_dk = derive_key(new, new_salt, iterations=200_000, dklen=72)
                    st.session_state.settings["auth"] = {"salt":binascii.hexlify(new_salt).decode(), "key":binascii.hexlify(new_dk).decode(), "iterations":200_000, "dklen":72}
                    save_json(SETTINGS_FILE, st.session_state.settings)
                    st.success("Passwort geändert.")

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Daten: portfolio.json / settings.json / history.json • Deterministische Simulationen</div>", unsafe_allow_html=True)
