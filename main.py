# main.py
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
    /* BLACK THEME */
    html, body, [class*="css"]  {
        background-color: #000000 !important;
        color: #e6eef6 !important;
    }
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
    """
    Deterministische, wiederholbare Preisreihe (Datum, Preis, SMA20, SMA50)
    Wird in session_state gecached.
    """
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
    # SMA
    sma20 = []
    sma50 = []
    for i in range(len(prices)):
        window20 = [p[1] for p in prices[max(0, i-19): i+1]]
        window50 = [p[1] for p in prices[max(0, i-49): i+1]]
        sma20.append(round(mean(window20), 4))
        sma50.append(round(mean(window50), 4))
    series = []
    for i in range(len(prices)):
        d, p = prices[i]
        series.append({"date": d, "price": p, "sma20": sma20[i], "sma50": sma50[i]})
    st.session_state.price_series[key] = series
    return series

# --------------------------
# Risikoanalyse (Volatilität)
# --------------------------
def calc_volatility_label(series):
    """
    Berechnet Volatilität (StdDev der täglichen Renditen) über 30 Tage und gibt Label.
    Schwellenwerte: low < 0.01, medium < 0.03, high >= 0.03 (these are relative)
    """
    if not series or len(series) < 31:
        return "Unbekannt", 0.0
    prices = [p["price"] for p in series[-31:]]  # last 31 prices -> 30 returns
    returns = []
    for i in range(1, len(prices)):
        prev = prices[i-1]
        curr = prices[i]
        if prev > 0:
            returns.append((curr - prev) / prev)
    if len(returns) < 10:
        return "Unbekannt", 0.0
    vol = stdev(returns)  # daily vol
    # annualized ~ vol * sqrt(252) but we use raw for labels
    if vol < 0.01:
        label = "Niedrig"
    elif vol < 0.03:
        label = "Mittel"
    else:
        label = "Hoch"
    return label, vol

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
# Sidebar (Navigation & Einstellungen)
# --------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Seite wählen", ["Marktplatz", "Portfolio", "Einstellungen", "Export/Import"])

st.sidebar.markdown("---")
st.sidebar.subheader("Ziel-Tracking")
goal = st.sidebar.number_input("Finanzziel (gesamt)", min_value=0.0, value=float(st.session_state.settings.get("goal", 10000.0)), step=100.0)
if st.sidebar.button("Ziel speichern"):
    st.session_state.settings["goal"] = float(goal)
    save_json_file(SETTINGS_FILE, st.session_state.settings)
    st.success("Ziel gespeichert.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Hinweis:** Offline-Version. Daten sind simuliert, deterministisch und bleiben zwischen Sessions erhalten.")

# --------------------------
# Helpers: chart data format
# --------------------------
def to_linechart_dict(series, key="price"):
    # returns dict date->value for st.line_chart
    return {p["date"]: p[key] for p in series}

# --------------------------
# PAGE: Marktplatz
# --------------------------
if page == "Marktplatz":
    st.header("🏬 Marktplatz — ETFs, Kryptowährungen & Aktien")
    st.markdown("Wähle eine Kategorie, schaue den Graphen an und füge Positionen deinem Portfolio hinzu. Unter jedem Graphen: Empfehlung (Kaufen / Nicht kaufen) + Risiko.")
    intervals = {"1 Monat":30, "3 Monate":90, "6 Monate":180, "1 Jahr":365, "5 Jahre":365*5, "10 Jahre":365*10}

    def show_assets_block(title, assets, currency_symbol="€"):
        st.subheader(title)
        for a in assets:
            cols = st.columns([3,1])
            with cols[1]:
                interval_label = st.selectbox(f"Zeitraum für {a['name']}", list(intervals.keys()), key=f"interval_{a['id']}")
                days = intervals[interval_label]
            series = generate_price_series(a["id"], days, start_price=100 + (abs(hash(a["id"])) % 1000) / 10)
            # main graph (last N days)
            with cols[0]:
                st.markdown(f"**{a['name']} — {a.get('symbol','')}**")
                st.line_chart(to_linechart_dict(series, "price"))
                # small sparkline (last 20)
                last20 = series[-20:] if len(series) >= 20 else series
                st.markdown("<div class='spark'>", unsafe_allow_html=True)
                st.line_chart(to_linechart_dict(last20, "price"))
                st.markdown("</div>", unsafe_allow_html=True)
            # right column: stats, risk, recommendation, add form
            cur = series[-1]["price"]
            sma20 = series[-1]["sma20"]
            sma50 = series[-1]["sma50"]
            risk_label, vol = calc_volatility_label(series)
            rec = "Kaufen" if sma20 > sma50 else "Nicht kaufen"
            # show information
            with cols[1]:
                st.write(f"**Aktuell:** {cur:.2f} {currency_symbol}")
                st.write(f"**SMA20:** {sma20:.2f}  |  **SMA50:** {sma50:.2f}")
                st.markdown(f"**Empfehlung:** {'🟢 ' + rec if rec=='Kaufen' else '🔴 ' + rec}")
                st.markdown(f"**Risiko (Volatilität):** {risk_label} ({vol:.4f})")
                # Add to portfolio form with note
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
    st.header("💼 Portfolio")
    portfolio = st.session_state.portfolio
    if not portfolio:
        st.info("Dein Portfolio ist leer. Füge Assets im Marktplatz hinzu.")
    else:
        # summary
        total_value = 0.0
        total_cost = 0.0
        rows = []
        for item in portfolio:
            series = generate_price_series(item["asset_id"], 365, start_price=item["buy_price"])
            cur_price = series[-1]["price"]
            qty = float(item["qty"])
            value = cur_price * qty
            cost = float(item["buy_price"]) * qty
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
            total_value += value
            total_cost += cost
            rows.append((item, cur_price, value, cost, pnl, pnl_pct))
        # goal progress
        goal_val = float(st.session_state.settings.get("goal", 10000.0))
        progress = min(total_value / goal_val if goal_val > 0 else 0.0, 1.0)
        st.markdown(f"**Gesamtwert:** {total_value:,.2f} €  •  Ziel: {goal_val:,.2f} €")
        st.progress(progress)
        st.markdown(f"**Fortschritt:** {progress*100:.2f}%")
        st.markdown("---")
        # show each position with note editable and small bar
        for (item, cur_price, value, cost, pnl, pnl_pct) in rows:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            cols = st.columns([2,1])
            with cols[0]:
                st.write(f"**{item['name']}** ({item['category']})")
                st.write(f"Menge: {item['qty']} • Kaufpreis: {item['buy_price']:.2f} € • Aktuell: {cur_price:.4f} €")
                st.write(f"**Wert:** {value:,.2f} €  |  **Gewinn/Verlust:** {pnl:+.2f} € ({pnl_pct:+.2f}%)")
                # sparkline of last 40
                series = generate_price_series(item["asset_id"], 40, start_price=item["buy_price"])
                st.line_chart(to_linechart_dict(series, "price"))
            with cols[1]:
                # recommendation
                rec = "Halten" if pnl >= 0 else "Verkaufen"
                if pnl >= 0:
                    st.success(f"Empfehlung: {rec}")
                else:
                    st.error(f"Empfehlung: {rec}")
                # note editor
                new_note = st.text_area("Notiz bearbeiten", value=item.get("note",""), key=f"note_edit_{item['id']}", height=80)
                if st.button("Notiz speichern", key=f"save_note_{item['id']}"):
                    update_note(item["id"], new_note)
                # remove
                if st.button("Position entfernen", key=f"remove_{item['id']}"):
                    remove_from_portfolio(item["id"])
            st.markdown("</div>", unsafe_allow_html=True)
        # total summary
        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost * 100) if total_cost > 0 else 0.0
        st.markdown("---")
        st.subheader("Gesamtübersicht")
        st.write(f"Total Wert: **{total_value:,.2f} €**  •  Eingezahlt: **{total_cost:,.2f} €**")
        st.write(f"Gesamtgewinn/Verlust: **{total_pnl:+.2f} €** ({total_pnl_pct:+.2f}%)")
        bar_class = "gain" if total_pnl >= 0 else "loss"
        st.markdown(f"<div class='{bar_class}' style='width:{min(abs(total_pnl_pct)*2,100)}%'></div>", unsafe_allow_html=True)
        if st.button("Portfolio leeren"):
            st.session_state.portfolio = []
            save_portfolio()
            st.experimental_rerun()

# --------------------------
# PAGE: Einstellungen
# --------------------------
elif page == "Einstellungen":
    st.header("⚙️ Einstellungen")
    st.write("Einstellungen & Optionen für diese Offline-App.")
    # goal setting (also in sidebar)
    st.write("Finanzziel:")
    new_goal = st.number_input("Zielbetrag speichern", min_value=0.0, value=float(st.session_state.settings.get("goal",10000.0)), step=100.0)
    if st.button("Ziel speichern (Hier)"):
        st.session_state.settings["goal"] = float(new_goal)
        save_json_file(SETTINGS_FILE, st.session_state.settings)
        st.success("Ziel gespeichert.")
    st.markdown("---")
    st.write("App Daten:")
    if st.button("Session Cache leeren (keine Portfolio-Daten)"):
        st.session_state.price_series = {}
        st.success("Chart-Cache geleert.")
    st.markdown("Die Portfolio-Daten werden in `portfolio.json` im App-Ordner gespeichert.")

# --------------------------
# PAGE: Export / Import
# --------------------------
elif page == "Export/Import":
    st.header("📤 Export & 📥 Import")
    st.write("Exportiere oder importiere dein Portfolio (JSON). Nützlich zum Backup oder Transfer.")
    # export
    export_obj = {
        "portfolio": st.session_state.portfolio,
        "settings": st.session_state.settings,
        "exported_at": datetime.utcnow().isoformat()
    }
    export_json = json.dumps(export_obj, ensure_ascii=False, indent=2)
    st.download_button("📤 Exportiere Portfolio als JSON", data=export_json, file_name="portfolio_export.json", mime="application/json")
    st.markdown("---")
    # import
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
        except Exception as e:
            st.error("Fehler beim Import. Stelle sicher, dass die Datei gültiges JSON enthält.")

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown("<div class='small'>Offline-Modus • Daten sind simuliert (deterministisch) • Portfolio in <code>portfolio.json</code></div>", unsafe_allow_html=True)
