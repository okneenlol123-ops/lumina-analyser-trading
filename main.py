# main.py
# Offline‑Finanz‑App — Black Edition v5 — Home, Suche, Pop‑ups, Wissensbasis, Benachrichtigungen

import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# ------------------- CONFIG & FILES -------------------
st.set_page_config(page_title="Offline Portfolio Pro+", page_icon="💹", layout="wide")
PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"
GUIDES_FILE = "guides.json"

# ------------------- STYLING -------------------
st.markdown("""
<style>
html, body, [class*="css"] {background:#000 !important; color:#e6eef6 !important;}
.stButton>button {background:#111; color:#e6eef6; border:1px solid #222; border-radius:6px;}
.card {background:#070707; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px;}
.small {color:#9aa6b2; font-size:13px;}
.gain {background:linear-gradient(90deg,#00ff88,#007744); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(0,255,136,0.08); animation: glow 2s infinite;}
.loss {background:linear-gradient(90deg,#ff4466,#770022); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(255,68,102,0.06);}
@keyframes glow {0% {box-shadow:0 0 6px rgba(0,255,136,0.04);}50%{box-shadow:0 0 18px rgba(0,255,136,0.10);}100%{box-shadow:0 0 6px rgba(0,255,136,0.04);}}
.searchbox {background:#111; padding:4px; border-radius:6px;}
.link {color:#4eaaff; text-decoration:underline;}
</style>
""", unsafe_allow_html=True)

# ------------------- JSON HELPERS -------------------
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

# ------------------- INIT FILES -------------------
if not os.path.exists(SETTINGS_FILE): save_json(SETTINGS_FILE, {})
if not os.path.exists(GUIDES_FILE):
    guides = {
        "ETF_vs_Aktie": "Exchange Traded Funds (ETFs) … (ausführliche Erklärung) ETFs bestehen aus vielen Einzelwerten und erlauben breite Diversifikation ohne Einzelrisiko.",
        "Volatilitaet": "Volatilität ist Maßstab für Schwankungen. Eine hohe tägliche Volatilität zeigt höhere Unsicherheit und höheres Risiko.",
        "Rebalancing": "Rebalancing bedeutet, regelmäßig die Aufteilung der Vermögenswerte zu prüfen und ggf. umzuschichten, um z. B. 50 % in ETFs, 30 % Aktien, 20 % Krypto zu halten.",
        "Risikoarten": "Es gibt verschiedene Risikoarten: Marktrisiko, Liquiditätsrisiko, Währungsrisiko, Konzentrationsrisiko. Jede Asset‑Kategorie trägt andere Risiken.",
        "ETF_Typen": "Index‑ETFs, Dividenden‑ETFs, Sektor‑ETFs. Jeder Typ folgt einer anderen Strategie und hat anderes Chance‑/Risiko‑Profil.",
        "Diversifikation": "Diversifikation reduziert Risiko: Kombination verschiedener Asset‑Klassen, Regionen, Branchen hilft Schwankungen abzufedern.",
        "Crash_Simulation": "Mit einer Crash‑Simulation kann man sehen, wie das Portfolio reagiert, wenn alle Kurse z. B. ‑20 % fallen.",
        "Suchfunktion": "Mit der Suchfunktion im Marktplatz & Portfolio findest du Assets, Notizen oder Tags schnell wieder."
    }
    save_json(GUIDES_FILE, guides)
if not os.path.exists(PORTFOLIO_FILE): save_json(PORTFOLIO_FILE, [])
if not os.path.exists(HISTORY_FILE): save_json(HISTORY_FILE, [])

# ------------------- SESSION STATE -------------------
if "portfolio" not in st.session_state: st.session_state.portfolio = load_json(PORTFOLIO_FILE, [])
if "settings" not in st.session_state: st.session_state.settings = load_json(SETTINGS_FILE, {})
if "history" not in st.session_state: st.session_state.history = load_json(HISTORY_FILE, [])
if "series_cache" not in st.session_state: st.session_state.series_cache = {}

# ------------------- SECURITY SINGLE OWNER PASSWORD -------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def setup_password_flow():
    settings = st.session_state.settings or {}
    if not settings.get("auth"):
        st.header("Erst‑Installation – Eigentümer‑Passwort setzen")
        pwd = st.text_input("Passwort wählen", type="password")
        pwd2 = st.text_input("Passwort wiederholen", type="password")
        if st.button("Passwort setzen"):
            if not pwd or pwd != pwd2:
                st.error("Passwörter leer oder stimmen nicht überein.")
                return False
            salt = os.urandom(16)
            dk = derive_key(pwd, salt)
            settings["auth"] = {"salt": binascii.hexlify(salt).decode(), "key": binascii.hexlify(dk).decode()}
            st.session_state.settings = settings
            save_json(SETTINGS_FILE, settings)
            st.success("Passwort gesetzt – bitte neu einloggen.")
            return False
        return False
    return True

def login_flow():
    auth = st.session_state.settings.get("auth", {})
    st.header("Login (Eigentümer)")
    pwd = st.text_input("Passwort", type="password")
    if st.button("Einloggen"):
        salt = binascii.unhexlify(auth["salt"])
        dk = derive_key(pwd, salt)
        if binascii.hexlify(dk).decode() == auth["key"]:
            st.session_state.auth_ok = True
            st.success("Erfolgreich eingeloggt.")
            return True
        else:
            st.error("Falsches Passwort.")
            return False
    return False

if not setup_password_flow():
    st.stop()
if not st.session_state.get("auth_ok", False):
    if not login_flow():
        st.stop()

# ------------------- ASSETS -------------------
ETFS = [{"id":"ETF_DE","name":"Deutschland"},{"id":"ETF_US","name":"USA"},{"id":"ETF_EU","name":"Europa"}]
CRYPTOS = [{"id":"CR_BTC","name":"Bitcoin"},{"id":"CR_ETH","name":"Ethereum"},{"id":"CR_SOL","name":"Solana"}]
STOCKS = [{"id":"ST_AAPL","name":"Apple"},{"id":"ST_TSLA","name":"Tesla"},{"id":"ST_MSFT","name":"Microsoft"}]

# ------------------- PRICE SIMULATION -------------------
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
        series.append({"date": date, "price": round(price, 4)})
    # SMA20/SMA50
    for i in range(len(series)):
        p20 = [series[j]["price"] for j in range(max(0, i-19), i+1)]
        p50 = [series[j]["price"] for j in range(max(0, i-49), i+1)]
        series[i]["sma20"] = round(mean(p20), 4)
        series[i]["sma50"] = round(mean(p50), 4)
    st.session_state.series_cache[key] = series
    return series

# ------------------- SEARCH FUNCTION -------------------
def search_portfolio(query):
    results = []
    for item in st.session_state.portfolio:
        if query.lower() in item["name"].lower() or query.lower() in item.get("note","").lower():
            results.append(item)
    return results

# ------------------- PORTFOLIO HELPERS -------------------
def save_portfolio_state(): save_json(PORTFOLIO_FILE, st.session_state.portfolio)
def save_history_event(action, item):
    st.session_state.history.append({"timestamp": datetime.utcnow().isoformat(), "action": action, "item": item})
    save_json(HISTORY_FILE, st.session_state.history)

def add_position(category, asset_id, name, qty, buy_price, note=""):
    item = {"id": f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
            "category": category, "asset_id": asset_id, "name": name,
            "qty": float(qty), "buy_price": float(buy_price),
            "note": note, "added_at": datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    save_portfolio_state()
    save_history_event("add", item)
    st.success(f"{name} hinzugefügt.")

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    save_portfolio_state()
    save_history_event("remove", {"id": item_id})
    st.experimental_rerun()

def update_note(item_id, new_note):
    for p in st.session_state.portfolio:
        if p["id"] == item_id:
            p["note"] = new_note
    save_portfolio_state()
    save_history_event("note_update", {"id": item_id, "note": new_note})
    st.success("Notiz gespeichert.")

def current_price_for(item):
    base = 100.0 if not item["category"].lower().startswith("krypto") else 1000.0
    series = generate_series(item["asset_id"], 365, start_price=item["buy_price"] if item["buy_price"]>0 else base)
    return series[-1]["price"]

def portfolio_snapshot():
    tot_value = tot_cost = 0.0
    rows = []
    for item in st.session_state.portfolio:
        cur = current_price_for(item)
        value = cur * item["qty"]
        cost = item["buy_price"] * item["qty"]
        pnl = value - cost
        pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0
        rows.append({"item": item, "cur": cur, "value": value, "cost": cost, "pnl": pnl, "pnl_pct": pnl_pct})
        tot_value += value; tot_cost += cost
    return {"rows": rows, "total_value": tot_value, "total_cost": tot_cost}

# ------------------- NOTIFICATIONS -------------------
def check_notifications():
    notifications = ["Testnachricht 1", "Testnachricht 2"]  # Beispiel
    for msg in notifications:
        try:
            st.toast(msg, icon="🔔")  # funktioniert in neueren Streamlit
        except Exception:
            st.info(f"🔔 {msg}")  # Fallback für ältere Version
check_notifications()
# ------------------- UI: Sidebar Navigation -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Home","Marktplatz","Portfolio","Statistiken","Wissensbasis","Export/Import","Einstellungen"])
st.sidebar.markdown("---")

# ---------- PAGE: Home ----------
if page == "Home":
    st.header("🏠 Home – Übersicht")
    snap = portfolio_snapshot()
    st.metric("Gesamtwert", f"{snap['total_value']:.2f} €", delta=f"{(snap['total_value']-snap['total_cost']):+.2f} €")
    cols = st.columns(3)
    cat_vals = {"ETF":0.0,"Krypto":0.0,"Aktie":0.0}
    for r in snap["rows"]:
        cat_vals[r["item"]["category"]] = cat_vals.get(r["item"]["category"],0.0) + r["value"]
    cols[0].metric("ETFs", f"{cat_vals['ETF']:.2f} €")
    cols[1].metric("Kryptos", f"{cat_vals['Krypto']:.2f} €")
    cols[2].metric("Aktien", f"{cat_vals['Aktie']:.2f} €")

    st.subheader("Wirtschafts‑News")
    news = [
        {"title":"Aktienmärkte starten positiv","link":"https://example.com/news1"},
        {"title":"Zinspolitik der Zentralbank","link":"https://example.com/news2"},
        {"title":"Technologie‑Sektor wächst weiter","link":"https://example.com/news3"},
        {"title":"Rohstoffe im Aufwind","link":"https://example.com/news4"},
        {"title":"Krypto‑Markt zeigt Volatilität","link":"https://example.com/news5"}
    ]
    for n in news:
        st.markdown(f"- [{n['title']}]({n['link']})", unsafe_allow_html=True)

    st.subheader("Schnell‑Statistiken")
    st.write("#### Gewinn/Verlust Top/Flop")
    if snap["rows"]:
        best = max(snap["rows"], key=lambda x: x["pnl_pct"])
        worst = min(snap["rows"], key=lambda x: x["pnl_pct"])
        st.write(f"🏆 Best: {best['item']['name']} (+{best['pnl_pct']:.2f}%)")
        st.write(f"☹️ Worst: {worst['item']['name']} ({worst['pnl_pct']:.2f}%)")

# ---------- PAGE: Marktplatz ----------
elif page == "Marktplatz":
    st.header("🏬 Marktplatz")
    intervals = {"1M":30,"3M":90,"6M":180,"1J":365,"5J":365*5}
    def show_assets(assets, cat_label, symbol):
        st.subheader(cat_label)
        for a in assets:
            cols = st.columns([3,1])
            with cols[0]:
                st.markdown(f"**{a['name']}**")
                days = st.selectbox(f"Zeitraum {a['id']}", list(intervals.keys()), key=f"iv_{a['id']}")
                series = generate_series(a["id"], intervals[days], start_price=100.0)
                st.line_chart({p["date"]:p["price"] for p in series})
            with cols[1]:
                cur = series[-1]["price"]
                rec = "Kaufen" if series[-1]["price"] > series[-1]["sma20"] else "Nicht kaufen"
                st.write(f"Aktuell: {cur:.2f} {symbol}")
                st.write(f"Empfehlung: {rec}")
                with st.form(key=f"add_{a['id']}"):
                    qty = st.number_input("Menge", value=1.0, step=0.1)
                    bp = st.number_input("Kaufpreis", value=float(cur), step=0.01)
                    note = st.text_area("Notiz", height=80)
                    if st.form_submit_button("Hinzufügen"):
                        add_position(cat_label, a["id"], a["name"], qty, bp, note)
    show_assets(ETFS,"ETFs","€")
    show_assets(CRYPTOS,"Kryptos","$")
    show_assets(STOCKS,"Aktien","€")

# ---------- PAGE: Portfolio ----------
elif page == "Portfolio":
    st.header("💼 Portfolio")
    snap = portfolio_snapshot()
    q = st.text_input("🔍 Suche im Portfolio (Name oder Notiz)")
    if q:
        results = search_portfolio(q)
        st.write(f"Suchergebnisse für '{q}': {len(results)} Treffer")
        for item in results:
            st.write(item["name"], " • Notiz:", item.get("note",""))
    if not snap["rows"]:
        st.info("Kein Portfolio‑Eintrag vorhanden.")
    else:
        for r in snap["rows"]:
            item = r["item"]
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write(f"**{item['name']}** ({item['category']})")
            st.write(f"Menge: {item['qty']} | Kaufpreis: {item['buy_price']:.2f} € | Aktuell: {r['cur']:.2f} €")
            st.write(f"Wert: {r['value']:.2f} € | Gewinn/Verlust: {r['pnl']:+.2f} € ({r['pnl_pct']:+.2f}%)")
            st.write("Notiz:", item.get("note",""))
            if st.button(f"Entfernen {item['name']}", key=f"rm_{item['id']}"):
                remove_position(item["id"])
            st.markdown("</div>", unsafe_allow_html=True)

# ---------- PAGE: Statistiken ----------
elif page == "Statistiken":
    st.header("📊 Statistiken & Heatmap")
    snap = portfolio_snapshot()
    if not snap["rows"]:
        st.info("Keine Daten zur Statistik.")
    else:
        st.bar_chart({r["item"]["name"]: r["value"] for r in snap["rows"]})
        st.write("Heatmap (farbige Balken für Gewinn/Verlust):")
        for r in snap["rows"]:
            pct = r["pnl_pct"]
            color = "#00aa00" if pct>=0 else "#aa0000"
            width = min(abs(pct)*2,100)
            st.markdown(f"<div style='background:{color}; width:{width}%; padding:6px; border-radius:4px; margin-bottom:4px;'>{r['item']['name']} — {r['pnl_pct']:+.2f}%</div>", unsafe_allow_html=True)

# ---------- PAGE: Wissensbasis ----------
elif page == "Wissensbasis":
    st.header("📘 Wissensbasis")
    guides = load_json(GUIDES_FILE, {})
    for k,v in guides.items():
        st.subheader(k.replace("_"," "))
        st.write(v)
        # Example graph explanation placeholder
        st.markdown("Beispielgraph:") 
        dummy = generate_series("EXAMPLE_"+k, 60, start_price=50.0)
        st.line_chart({p["date"]: p["price"] for p in dummy})
        st.markdown("---")

# ---------- PAGE: Export/Import ----------
elif page == "Export/Import":
    st.header("📤 Export / 📥 Import")
    export_obj = {"portfolio": st.session_state.portfolio, "settings": st.session_state.settings, "exported_at": datetime.utcnow().isoformat()}
    st.download_button("Export als JSON", data=json.dumps(export_obj, ensure_ascii=False, indent=2), file_name="backup_portfolio.json", mime="application/json")
    uploaded = st.file_uploader("Importiere Backup (JSON)", type=["json"])
    if uploaded:
        try:
            obj = json.loads(uploaded.read().decode("utf‑8"))
            if "portfolio" in obj:
                st.session_state.portfolio = obj["portfolio"]
                save_json(PORTFOLIO_FILE, st.session_state.portfolio)
            if "settings" in obj:
                st.session_state.settings = obj["settings"]
                save_json(SETTINGS_FILE, st.session_state.settings)
            st.success("Import erfolgreich.")
            st.experimental_rerun()
        except Exception:
            st.error("Fehler beim Import.")

# ---------- PAGE: Einstellungen ----------
elif page == "Einstellungen":
    st.header("⚙️ Einstellungen")
    goal = st.number_input("Zielbetrag (gesamt)", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Ziel speichern"):
        st.session_state.settings["goal"] = float(goal)
        save_json(SETTINGS_FILE, st.session_state.settings)
        st.success("Ziel gespeichert.")
    st.markdown("---")
    if st.button("Cache leeren"):
        st.session_state.series_cache = {}
        st.success("Chart‑Cache gelöscht.")
    st.markdown("---")
    st.write("📌 Benachrichtigungen werden automatisch angezeigt, wenn Positionen ±10% erreicht haben.")

# ------------------- FOOTER -------------------
st.markdown("---")
st.markdown("<div class='small'>Offline‑Modus • Daten lokal in portfolio.json / settings.json / history.json • Alle Simulationen deterministisch</div>", unsafe_allow_html=True)
