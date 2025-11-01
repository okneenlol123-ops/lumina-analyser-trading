# main.py (erweitert)
import streamlit as st
import random
import hashlib
import json
import os
from datetime import datetime, timedelta
from statistics import stdev, mean

# --------------------------
# Konfiguration & Styling
# --------------------------
st.set_page_config(page_title="Finanz-Dashboard — Black Pro", page_icon="💹", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  { background-color: #000000 !important; color:#e6eef6 !important; }
    .stButton>button { background-color:#111111; color:#e6eef6; border:1px solid #2f2f2f; border-radius:6px; }
    .stTextInput>div, .stNumberInput>div, .stSelectbox>div { background-color:#000000; color:#e6eef6; }
    .card { background:#0a0a0a; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px; }
    .gain { background: linear-gradient(90deg, #00ff99 0%, #007755 100%); height: 10px; border-radius:4px; box-shadow:0 0 12px rgba(0,255,153,0.12); animation: pulse 2s infinite; }
    .loss { background: linear-gradient(90deg, #ff0044 0%, #770022 100%); height: 10px; border-radius:4px; box-shadow:0 0 12px rgba(255,0,68,0.08); }
    @keyframes pulse { 0% { box-shadow:0 0 6px rgba(0,255,153,0.06);} 50% { box-shadow:0 0 16px rgba(0,255,153,0.14);} 100% { box-shadow:0 0 6px rgba(0,255,153,0.06);} }
    .small { color:#9aa6b2; font-size:12px; }
    .spark { height:48px; }
    .metric { font-weight:700; color:#00ffaa; }
    .note { color:#9aa6b2; font-size:13px; }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Dateinamen / Persistence
# --------------------------
PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"

def load_json_file(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default

def save_json_file(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# --------------------------
# Session state init
# --------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = load_json_file(PORTFOLIO_FILE, [])

if "price_series" not in st.session_state:
    st.session_state.price_series = {}

if "settings" not in st.session_state:
    st.session_state.settings = load_json_file(SETTINGS_FILE, {"goal": 10000.0})

# --------------------------
# Assets (Beispiele)
# --------------------------
ETFS = [
    {"id": "ETF_DE", "name": "Deutschland", "symbol": "DE_ETF"},
    {"id": "ETF_US", "name": "USA", "symbol": "SPY"},
    {"id": "ETF_EU", "name": "Europa", "symbol": "VGK"},
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
]

# --------------------------
# Deterministische Fake-Daten
# --------------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def generate_price_series(asset_id: str, days: int = 365, start_price: float = 100.0):
    key = f"{asset_id}_{days}"
    if key in st.session_state.price_series:
        return st.session_state.price_series[key]
    rnd = random.Random(deterministic_seed(asset_id))
    price = float(start_price)
    prices = []
    for i in range(days):
        drift = (rnd.random() - 0.48) * 0.001
        vol = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1))
        prices.append((date.isoformat(), round(price, 4)))
    series = [{"date": d, "price": p} for d, p in prices]
    st.session_state.price_series[key] = series
    return series

# --------------------------
# Portfolio operations
# --------------------------
def save_portfolio():
    save_json_file(PORTFOLIO_FILE, st.session_state.portfolio)

def add_to_portfolio(category, asset_id, name, qty, buy_price, note=""):
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
    save_portfolio()
    st.success(f"{name} ({qty} × {buy_price}) wurde zum Portfolio hinzugefügt.")

def remove_from_portfolio(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["id"] != item_id]
    save_portfolio()
    st.experimental_rerun()

def update_note(item_id, new_note):
    for p in st.session_state.portfolio:
        if p["id"] == item_id:
            p["note"] = new_note
    save_portfolio()
    st.success("Notiz gespeichert.")

# --------------------------
# Sidebar Navigation
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite wählen", ["Home", "Marktplatz", "Portfolio", "Einstellungen", "Export/Import"])

# --------------------------
# PAGE: Home
# --------------------------
if page == "Home":
    st.header("🏠 Home — Portfolio Übersicht & Wirtschaft-News")
    portfolio = st.session_state.portfolio
    total_value = total_cost = 0.0
    cat_values = {"ETF":0.0, "Krypto":0.0, "Aktie":0.0}
    for item in portfolio:
        series = generate_price_series(item["asset_id"], 365, item["buy_price"])
        cur_price = series[-1]["price"]
        value = cur_price * item["qty"]
        cost = item["buy_price"] * item["qty"]
        total_value += value
        total_cost += cost
        if item["category"] in cat_values:
            cat_values[item["category"]] += value

    st.subheader("Portfolio Übersicht")
    st.metric("Gesamtwert", f"{total_value:,.2f} €", delta=f"{total_value - total_cost:+.2f} €")
    cols = st.columns(3)
    cols[0].metric("ETFs", f"{cat_values['ETF']:,.2f} €")
    cols[1].metric("Kryptos", f"{cat_values['Krypto']:,.2f} €")
    cols[2].metric("Aktien", f"{cat_values['Aktie']:,.2f} €")

    st.subheader("Wirtschafts-News")
    # 5 Offline-News-Beispiele
    news = [
        "📈 Aktienmärkte starten positiv in die Woche.",
        "💹 Euro vs USD zeigt stabile Entwicklung.",
        "🏦 Zentralbank kündigt neue Zinspolitik an.",
        "📊 Tech-Sektor dominiert die Wachstumscharts.",
        "📰 Ölpreise steigen leicht nach geopolitischen Spannungen."
    ]
    for n in news:
        st.markdown(f"- {n}")

# --------------------------
# PAGE: Marktplatz
# --------------------------
elif page == "Marktplatz":
    st.header("🏬 Marktplatz — ETFs, Kryptowährungen & Aktien")
    intervals = {"1 Monat":30, "3 Monate":90, "6 Monate":180, "1 Jahr":365, "5 Jahre":365*5, "10 Jahre":365*10}

    def show_assets_block(title, assets, currency_symbol="€"):
        st.subheader(title)
        for a in assets:
            cols = st.columns([3,1])
            with cols[1]:
                interval_label = st.selectbox(f"Zeitraum für {a['name']}", list(intervals.keys()), key=f"interval_{a['id']}")
                days = intervals[interval_label]
            series = generate_price_series(a["id"], days, start_price=100 + (abs(hash(a["id"])) % 1000) / 10)
            with cols[0]:
                st.markdown(f"**{a['name']} — {a.get('symbol','')}**")
                st.line_chart({p["date"]: p["price"] for p in series})
            cur = series[-1]["price"]
            rec = "Kaufen" if random.random()>0.5 else "Nicht kaufen"
            with cols[1]:
                st.markdown(f"**Aktuell:** {cur:.2f} {currency_symbol}")
                st.markdown(f"**Empfehlung:** {rec}")
                with st.form(key=f"form_add_{a['id']}", clear_on_submit=False):
                    qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                    buy_price = st.number_input("Kaufpreis (pro Einheit)", min_value=0.0001, value=float(cur), step=0.01, key=f"buyprice_{a['id']}")
                    note = st.text_area("Notiz (optional)", value="", key=f"note_{a['id']}", height=80)
                    if st.form_submit_button("Zu Portfolio hinzufügen"):
                        add_to_portfolio(title, a["id"], f"{a['name']} ({a.get('symbol','')})", qty, buy_price, note)
            st.markdown("---")

    show_assets_block("ETFs", ETFS, "€")
    show_assets_block("Kryptowährungen", CRYPTOS, "$")
    show_assets_block("Aktien", STOCKS, "€")

# --------------------------
# PAGE: Portfolio
# --------------------------
elif page == "Portfolio":
    # (Beibehaltung deines bisherigen Portfolio-Codes)
    st.header("💼 Portfolio")
    portfolio = st.session_state.portfolio
    if not portfolio:
        st.info("Dein Portfolio ist leer. Füge Assets im Marktplatz hinzu.")
    else:
        total_value = total_cost = 0.0
        for item in portfolio:
            series = generate_price_series(item["asset_id"], 365, start_price=item["buy_price"])
            cur_price = series[-1]["price"]
            value = cur_price * item["qty"]
            cost = item["buy_price"] * item["qty"]
            total_value += value
            total_cost += cost
        st.write(f"Gesamtwert: {total_value:,.2f} €  • Eingezahlt: {total_cost:,.2f} €")

# --------------------------
# PAGE: Einstellungen
# --------------------------
elif page == "Einstellungen":
    st.header("⚙️ Einstellungen")
    new_goal = st.number_input("Zielbetrag speichern", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Ziel speichern (Hier)"):
        st.session_state.settings["goal"] = float(new_goal)
        save_json_file(SETTINGS_FILE, st.session_state.settings)
        st.success("Ziel gespeichert.")

# --------------------------
# PAGE: Export / Import
# --------------------------
elif page == "Export/Import":
    st.header("📤 Export & 📥 Import")
    export_obj = {
        "portfolio": st.session_state.portfolio,
        "settings": st.session_state.settings,
        "exported_at": datetime.utcnow().isoformat()
    }
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2)
    st.download_button("📤 Exportiere Portfolio als JSON", data=export_json, file_name="portfolio_export.json", mime="application/json")
    uploaded = st.file_uploader("📥 JSON Datei importieren (Portfolio + Settings)", type=["json"])
    if uploaded is not None:
        try:
            raw = uploaded.read().decode("utf-8")
            obj = json.loads(raw)
            if "portfolio" in obj and isinstance(obj["portfolio"], list):
                st.session_state.portfolio = obj["portfolio"]
                save_portfolio()
            if "settings" in obj and isinstance(obj["settings"], dict):
                st.session_state.settings = obj["settings"]
                save_json_file(SETTINGS_FILE, st.session_state.settings)
            st.success("Import erfolgreich.")
            st.experimental_rerun()
        except Exception:
            st.error("Fehler beim Import. Stelle sicher, dass die Datei gültiges JSON enthält.")
