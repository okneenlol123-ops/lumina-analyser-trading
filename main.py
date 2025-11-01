# main.py
# Offline Portfolio App — Black Edition v3 — alle gewünschten Features

import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev

# ------------------- CONFIG -------------------
st.set_page_config(page_title="Offline Portfolio Pro", page_icon="💹", layout="wide")
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
.gain {background:linear-gradient(90deg,#00ff88, #007744); height:10px; border-radius:6px;}
.loss {background:linear-gradient(90deg,#ff4466,#770022); height:10px; border-radius:6px;}
.badge {background:#111; color:#e6eef6; padding:4px 8px; border-radius:6px; border:1px solid #222; display:inline-block;}
.spark {height:48px;}
</style>
""", unsafe_allow_html=True)

# ------------------- JSON HELPERS -------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return default
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# ------------------- INITIALIZATION -------------------
if not os.path.exists(SETTINGS_FILE): save_json(SETTINGS_FILE, {})
if not os.path.exists(GUIDES_FILE):
    guides = {
        "ETF_vs_Aktie": "ETFs bündeln viele Aktien. Einfacher Diversifikationsmechanismus.",
        "Volatilitaet": "Volatilität misst die Schwankungsbreite der Renditen.",
        "Rebalancing": "Stellt Zielallokation wieder her.",
        "Risikoarten": "Marktrisiko, Liquiditätsrisiko, Volatilität",
        "ETF_Typen": "Index, Dividende, Sektor-ETFs",
        "Diversifikation": "Verteilung auf verschiedene Kategorien zur Risikominimierung",
        "Crash_Simulation": "Zeigt Auswirkungen eines Marktrückgangs auf Portfolio"
    }
    save_json(GUIDES_FILE, guides)
if not os.path.exists(PORTFOLIO_FILE): save_json(PORTFOLIO_FILE, [])
if not os.path.exists(HISTORY_FILE): save_json(HISTORY_FILE, [])

# ------------------- SESSION STATE -------------------
if "portfolio" not in st.session_state: st.session_state.portfolio = load_json(PORTFOLIO_FILE, [])
if "settings" not in st.session_state: st.session_state.settings = load_json(SETTINGS_FILE, {})
if "history" not in st.session_state: st.session_state.history = load_json(HISTORY_FILE, [])
if "series_cache" not in st.session_state: st.session_state.series_cache = {}

# ------------------- PASSWORD HANDLING -------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations, dklen)

def setup_password_flow():
    settings = st.session_state.settings or {}
    if not settings.get("auth"):
        st.header("Erstinstallation: Eigentümer-Passwort setzen")
        pwd = st.text_input("Passwort wählen", type="password")
        pwd2 = st.text_input("Passwort wiederholen", type="password")
        if st.button("Passwort setzen"):
            if not pwd or pwd != pwd2: st.error("Passwörter leer oder stimmen nicht überein."); return False
            salt = os.urandom(16)
            dk = derive_key(pwd, salt)
            settings["auth"] = {"salt": binascii.hexlify(salt).decode(), "key": binascii.hexlify(dk).decode()}
            st.session_state.settings = settings
            save_json(SETTINGS_FILE, settings)
            st.success("Passwort gesetzt. Bitte neu einloggen.")
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
        else: st.error("Falsches Passwort."); return False
    return False

auth_ready = setup_password_flow()
if not auth_ready: st.stop()
if not st.session_state.get("auth_ok", False):
    if not login_flow(): st.stop()

# ------------------- ASSET LISTS -------------------
ETFS = [{"id":"ETF_DE","name":"Deutschland"},{"id":"ETF_US","name":"USA"},{"id":"ETF_EU","name":"Europa"},{"id":"ETF_AS","name":"Asien"},{"id":"ETF_WW","name":"Welt"}]
CRYPTOS = [{"id":"CR_BTC","name":"Bitcoin"},{"id":"CR_ETH","name":"Ethereum"},{"id":"CR_SOL","name":"Solana"}]
STOCKS = [{"id":"ST_AAPL","name":"Apple"},{"id":"ST_TSLA","name":"Tesla"},{"id":"ST_MSFT","name":"Microsoft"}]

# ------------------- PRICE SIMULATION -------------------
def deterministic_seed(s:str)->int: return int(hashlib.sha256(s.encode()).hexdigest(),16) % (2**31)

def generate_series(asset_id, days=365, start_price=100.0):
    key = f"{asset_id}_{days}"
    if key in st.session_state.series_cache: return st.session_state.series_cache[key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    series = []
    for i in range(days):
        drift = (rnd.random() - 0.48) * 0.001
        vol = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1)).isoformat()
        series.append({"date":date,"price":round(price,4)})
    st.session_state.series_cache[key] = series
    return series

# ------------------- PORTFOLIO HELPERS -------------------
def save_portfolio_state(): save_json(PORTFOLIO_FILE, st.session_state.portfolio)
def save_history_event(action, item):
    st.session_state.history.append({"timestamp":datetime.utcnow().isoformat(),"action":action,"item":item})
    save_json(HISTORY_FILE, st.session_state.history)

def add_position(category, asset_id, name, qty, buy_price, tags=[], note=""):
    item = {"id":f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
            "category":category,"asset_id":asset_id,"name":name,"qty":float(qty),"buy_price":float(buy_price),
            "tags":tags,"note":note,"added_at":datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    save_portfolio_state()
    save_history_event("add", item)
    st.success(f"{name} hinzugefügt.")

def remove_position(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    save_portfolio_state()
    save_history_event("remove", {"id":item_id})
    st.experimental_rerun()

def update_note(item_id, new_note, tags=[]):
    for p in st.session_state.portfolio:
        if p["id"] == item_id: p["note"] = new_note; p["tags"] = tags
    save_portfolio_state()
    save_history_event("note_update", {"id":item_id,"note":new_note,"tags":tags})
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
        value = cur*item["qty"]
        cost = item["buy_price"]*item["qty"]
        pnl = value - cost
        pnl_pct = (pnl/cost*100) if cost>0 else 0.0
        rec = "Halten"
        if pnl_pct>10: rec="Verkaufen"
        elif pnl_pct<-10: rec="Beobachten"
        rows.append({"item":item,"cur":cur,"value":value,"cost":cost,"pnl":pnl,"pnl_pct":pnl_pct,"recommendation":rec})
        tot_value += value; tot_cost += cost
    return {"rows":rows,"total_value":tot_value,"total_cost":tot_cost}

# ------------------- DASHBOARD -------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seiten", ["Marktplatz","Portfolio","Statistiken","Wissensbasis","Export/Import"])
st.sidebar.markdown("---")

# ---------- Marktplatz ----------
if page=="Marktplatz":
    st.title("Marktplatz")
    for cat, assets in [("ETF",ETFS),("Krypto",CRYPTOS),("Aktie",STOCKS)]:
        st.subheader(cat)
        for a in assets:
            st.markdown(f"**{a['name']}**")
            series = generate_series(a["id"],90)
            st.line_chart({p["date"]:p["price"] for p in series})
            st.markdown(f"Empfehlung: **Halten**")  # offline, einfache Demo
            with st.form(key=f"add_{a['id']}"):
                qty = st.number_input("Menge", value=1.0, step=0.1)
                price = st.number_input("Kaufpreis", value=series[-1]["price"], step=0.01)
                note = st.text_area("Notiz", height=80)
                tags_input = st.text_input("Tags (kommagetrennt)")
                tags = [t.strip() for t in tags_input.split(",") if t.strip()]
                if st.form_submit_button("Hinzufügen"):
                    add_position(cat, a["id"], a["name"], qty, price, tags, note)

# ---------- Portfolio ----------
elif page=="Portfolio":
    st.title("Portfolio")
    snap = portfolio_snapshot()
    if not snap["rows"]: st.info("Portfolio leer")
    else:
        st.markdown(f"**Gesamtwert:** {snap['total_value']:.2f} €  • Eingezahlt: {snap['total_cost']:.2f} €")
        goal = st.session_state.settings.get("goal",10000.0)
        st.progress(min(snap['total_value']/goal,1.0))
        for r in snap["rows"]:
            item = r["item"]
            st.markdown(f"**{item['name']}** ({item['category']})")
            st.markdown(f"Menge: {item['qty']}, Kaufpreis: {item['buy_price']:.2f} €, Aktuell: {r['cur']:.2f} €")
            st.markdown(f"Gewinn/Verlust: {r['pnl']:.2f} € ({r['pnl_pct']:.2f}%), Empfehlung: {r['recommendation']}")
            st.markdown(f"Tags: {', '.join(item['tags'])} | Notiz: {item['note']}")
            if st.button(f"Entfernen {item['name']}", key=f"rm_{item['id']}"): remove_position(item["id"])

# ---------- STATISTIKEN ----------
elif page=="Statistiken":
    st.title("Portfolio Statistiken")
    snap = portfolio_snapshot()
    if not snap["rows"]: st.info("Keine Daten")
    else:
        categories = {"ETF":0,"Krypto":0,"Aktie":0}
        for r in snap["rows"]: categories[r["item"]["category"]] += r["value"]
        st.bar_chart(categories)
        pnl_list = [r["pnl"] for r in snap["rows"]]
        st.markdown("**Gewinn/Verlust pro Asset**")
        for r in snap["rows"]:
            pct = min(max((r["pnl_pct"]+50)/100,0),1)
            color_class = "gain" if r["pnl"]>0 else "loss"
            st.markdown(f"<div class='{color_class}' style='width:{pct*100}%'></div> {r['item']['name']} ({r['pnl']:.2f} €)", unsafe_allow_html=True)

# ---------- Wissensbasis ----------
elif page=="Wissensbasis":
    st.title("Wissensbasis")
    guides = load_json(GUIDES_FILE,{})
    search = st.text_input("Suche")
    for k,v in guides.items():
        if search.lower() in k.lower() or search.lower() in v.lower():
            st.markdown(f"**{k}**")
            st.markdown(v)

# ---------- Export / Import ----------
elif page=="Export/Import":
    st.title("Export / Import")
    if st.button("Portfolio exportieren"): st.download_button("Download JSON", json.dumps(st.session_state.portfolio), file_name="portfolio.json")
    uploaded = st.file_uploader("Portfolio importieren (JSON)")
    if uploaded:
        data = json.load(uploaded)
        st.session_state.portfolio = data
        save_portfolio_state()
        st.success("Portfolio importiert.")
