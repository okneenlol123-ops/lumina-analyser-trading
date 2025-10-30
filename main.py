# main.py
import streamlit as st
import random
from datetime import datetime, timedelta
import hashlib
import math

# ---------------------------
# App-Konfiguration & Dark UI
# ---------------------------
st.set_page_config(page_title="Finanz-Dashboard (Offline)", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
    /* Dark background + container */
    .main > div {background-color: #0b0f14 !important; color: #e6eef6;}
    .css-1d391kg {background-color: #0b0f14 !important;}
    .css-18e3th9 {background-color: #0b0f14 !important;}
    .stButton>button {background-color:#111827; color:#e6eef6; border:1px solid #1f2937;}
    .stSelectbox > div {background-color:#0b0f14; color:#e6eef6;}
    .stSlider > div {color:#e6eef6;}
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}
    .card {background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:12px; border-radius:8px;}
    .small {color:#9aa6b2}
    table {width:100%; border-collapse: collapse;}
    th, td {padding:8px 10px; border-bottom: 1px solid rgba(255,255,255,0.04); text-align:left;}
    th {color:#cbd5e1}
    .btn-danger {background:#7f1d1d;color:#fff;padding:6px 10px;border-radius:6px;border:none;}
    .metric {font-weight:700;color:#e6eef6}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Finanz-Dashboard — ETFs, Krypto & Aktien (Offline, Dark)")

st.markdown("**Hinweis:** Dies ist eine Offline-Demo. Kurse sind simuliert. Keine Anlageberatung.")

# ---------------------------
# Session State: Portfolio & Prices
# ---------------------------
if "portfolio" not in st.session_state:
    # portfolio: list of dicts: {id, category, name, qty, buy_price, added_at}
    st.session_state.portfolio = []

if "price_series" not in st.session_state:
    # price_series: dict keyed by asset_id -> list of (date, price, sma20, sma50)
    st.session_state.price_series = {}

# ---------------------------
# Assets (beispielhaft)
# ---------------------------
ETFS = [
    {"id": "ETF_DE", "name": "Deutschland", "symbol": "DE_ETF"},
    {"id": "ETF_US", "name": "USA", "symbol": "SPY"},
    {"id": "ETF_EU", "name": "Europa", "symbol": "VGK"},
    {"id": "ETF_AS", "name": "Asien", "symbol": "AAXJ"},
    {"id": "ETF_WL", "name": "Welt", "symbol": "VT"},
]
CRYPTOS = [
    {"id": "CR_BTC", "name": "Bitcoin", "symbol": "BTC"},
    {"id": "CR_ETH", "name": "Ethereum", "symbol": "ETH"},
    {"id": "CR_SOL", "name": "Solana", "symbol": "SOL"},
    {"id": "CR_BNB", "name": "BNB", "symbol": "BNB"},
    {"id": "CR_ADA", "name": "Cardano", "symbol": "ADA"},
]
STOCKS = [
    {"id": "ST_AAPL", "name": "Apple", "symbol": "AAPL"},
    {"id": "ST_TSLA", "name": "Tesla", "symbol": "TSLA"},
    {"id": "ST_MSFT", "name": "Microsoft", "symbol": "MSFT"},
    {"id": "ST_SIE", "name": "Siemens", "symbol": "SIE"},
    {"id": "ST_AMZN", "name": "Amazon", "symbol": "AMZN"},
]

# ---------------------------
# Helper: deterministic fake price series generator
# ---------------------------
def deterministic_seed(s: str) -> int:
    """Return a deterministic integer seed for a string (used to create consistent simulated series)."""
    h = hashlib.sha256(s.encode()).hexdigest()
    return int(h[:16], 16) % (2**31)

def generate_price_series(asset_id: str, days: int = 3650, start_price: float = 100.0):
    """
    Generate a deterministic, repeatable price series for an asset_id with SMA20 & SMA50.
    Stores result in session_state to keep consistency across reruns.
    """
    key = f"series_{asset_id}_{days}"
    if key in st.session_state.price_series:
        return st.session_state.price_series[key]

    seed = deterministic_seed(asset_id)
    rnd = random.Random(seed)
    prices = []
    price = float(start_price)
    for i in range(days):
        # simulate slow drift + volatility
        drift = (rnd.random() - 0.48) * 0.002  # small drift
        vol = (rnd.random() - 0.5) * 0.03  # daily vol
        price = max(0.01, price * (1 + drift + vol))
        date = (datetime.utcnow().date() - timedelta(days=days - i - 1))
        prices.append([date, round(price, 4)])
    # compute SMA20 & SMA50
    sma20 = []
    sma50 = []
    for i in range(len(prices)):
        window20 = [p[1] for p in prices[max(0, i - 19): i + 1]]
        window50 = [p[1] for p in prices[max(0, i - 49): i + 1]]
        sma20.append(round(sum(window20) / len(window20), 4))
        sma50.append(round(sum(window50) / len(window50), 4))
    series = []
    for i in range(len(prices)):
        d, p = prices[i]
        series.append((d, p, sma20[i], sma50[i]))
    st.session_state.price_series[key] = series
    return series

def get_current_price(asset_id, base_price):
    """Return last price from stored series (generate if missing)."""
    series = generate_price_series(asset_id, days=3650, start_price=base_price)
    return series[-1][1]

# ---------------------------
# Portfolio management functions
# ---------------------------
def add_to_portfolio(category, asset_id, name, qty, buy_price):
    item_id = f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}"
    st.session_state.portfolio.append({
        "item_id": item_id,
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "qty": float(qty),
        "buy_price": float(buy_price),
        "added_at": datetime.utcnow().isoformat()
    })
    st.success(f"{name} hinzugefügt: {qty} @ {buy_price}")

def remove_from_portfolio(item_id):
    st.session_state.portfolio = [p for p in st.session_state.portfolio if p["item_id"] != item_id]
    st.experimental_rerun()

# ---------------------------
# UI Layout: Left = Marketplace, Right = Portfolio
# ---------------------------
col_left, col_right = st.columns([2, 1])

# ---------------------------
# LEFT: Marketplace (categories + graphs)
# ---------------------------
with col_left:
    tabs = st.tabs(["📊 ETFs", "💰 Kryptowährungen", "🏦 Aktien"])
    # common interval selector per asset
    interval_options = ["1 Monat", "3 Monate", "6 Monate", "1 Jahr", "5 Jahre", "10 Jahre"]
    days_map = {"1 Monat": 30, "3 Monate": 90, "6 Monate": 180, "1 Jahr": 365, "5 Jahre": 365*5, "10 Jahre": 365*10}

    # ETFs
    with tabs[0]:
        st.header("ETFs")
        for a in ETFS:
            st.subheader(f"{a['name']} · {a['symbol']}")
            # per-asset controls
            cols = st.columns([3, 1])
            with cols[1]:
                interval = st.selectbox(f"Zeitraum {a['symbol']}", interval_options, key=f"etf_interval_{a['id']}")
            # prepare series (simulate)
            base = 120 + (hash(a['id']) % 80)
            series = generate_price_series(a['id'], days=days_map[interval], start_price=base)
            # plot (streamlit line_chart expects dict: label -> {x: y})
            chart_data = {str(row[0]): row[1] for row in series}
            st.line_chart(chart_data)
            # show current price
            cur_price = series[-1][1]
            st.write(f"**Aktueller Preis:** {cur_price:.4f} €")
            # add form to portfolio
            with st.form(key=f"form_add_{a['id']}", clear_on_submit=False):
                qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                buy_price = st.number_input("Kaufpreis (pro Einheit)", min_value=0.0001, value=float(cur_price), step=0.01, key=f"buyprice_{a['id']}")
                submitted = st.form_submit_button("Zu Portfolio hinzufügen")
                if submitted:
                    add_to_portfolio("ETF", a['id'], f"{a['name']} ({a['symbol']})", qty, buy_price)
            st.markdown("---")

    # Cryptos
    with tabs[1]:
        st.header("Kryptowährungen")
        for a in CRYPTOS:
            st.subheader(f"{a['name']} · {a['symbol']}")
            cols = st.columns([3,1])
            with cols[1]:
                interval = st.selectbox(f"Zeitraum {a['symbol']}", interval_options, key=f"crypto_interval_{a['id']}")
            base = 1000 + (abs(hash(a['id'])) % 50000)
            series = generate_price_series(a['id'], days=days_map[interval], start_price=base)
            chart_data = {str(row[0]): row[1] for row in series}
            st.line_chart(chart_data)
            cur_price = series[-1][1]
            st.write(f"**Aktueller Preis:** {cur_price:.4f} USD")
            with st.form(key=f"form_add_{a['id']}", clear_on_submit=False):
                qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.01, key=f"qty_{a['id']}")
                buy_price = st.number_input("Kaufpreis (pro Einheit)", min_value=0.0001, value=float(cur_price), step=0.01, key=f"buyprice_{a['id']}")
                submitted = st.form_submit_button("Zu Portfolio hinzufügen")
                if submitted:
                    add_to_portfolio("Crypto", a['id'], f"{a['name']} ({a['symbol']})", qty, buy_price)
            st.markdown("---")

    # Stocks
    with tabs[2]:
        st.header("Aktien")
        for a in STOCKS:
            st.subheader(f"{a['name']} · {a['symbol']}")
            cols = st.columns([3,1])
            with cols[1]:
                interval = st.selectbox(f"Zeitraum {a['symbol']}", interval_options, key=f"stock_interval_{a['id']}")
            base = 50 + (abs(hash(a['id'])) % 1500)
            series = generate_price_series(a['id'], days=days_map[interval], start_price=base)
            chart_data = {str(row[0]): row[1] for row in series}
            st.line_chart(chart_data)
            cur_price = series[-1][1]
            st.write(f"**Aktueller Preis:** {cur_price:.4f} €")
            with st.form(key=f"form_add_{a['id']}", clear_on_submit=False):
                qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1, key=f"qty_{a['id']}")
                buy_price = st.number_input("Kaufpreis (pro Einheit)", min_value=0.0001, value=float(cur_price), step=0.01, key=f"buyprice_{a['id']}")
                submitted = st.form_submit_button("Zu Portfolio hinzufügen")
                if submitted:
                    add_to_portfolio("Aktie", a['id'], f"{a['name']} ({a['symbol']})", qty, buy_price)
            st.markdown("---")

# ---------------------------
# RIGHT: Portfolio Übersicht
# ---------------------------
with col_right:
    st.header("Portfolio")
    portfolio = st.session_state.portfolio
    if not portfolio:
        st.info("Dein Portfolio ist leer. Füge Assets aus den Kategorien links hinzu.")
    else:
        # compute values
        total_value = 0.0
        total_cost = 0.0
        # build HTML table manually (no pandas)
        rows_html = []
        header = "<tr><th>Asset</th><th>Kategorie</th><th>Menge</th><th>Kaufpreis</th><th>Aktueller Preis</th><th>Wert</th><th>Gewinn/Verlust</th><th>Empfehlung</th><th>Aktion</th></tr>"
        for item in portfolio:
            # find current price from series
            base = 100.0
            # choose base by category for reasonable defaults
            if item["category"] == "ETF":
                base = 120 + (abs(hash(item["asset_id"])) % 80)
            elif item["category"] == "Crypto":
                base = 1000 + (abs(hash(item["asset_id"])) % 50000)
            else:
                base = 50 + (abs(hash(item["asset_id"])) % 1500)
            cur_price = get_current_price(item["asset_id"], base)
            qty = float(item["qty"])
            value = cur_price * qty
            cost = float(item["buy_price"]) * qty
            pnl = value - cost
            pnl_pct = (pnl / cost * 100) if cost != 0 else 0.0
            total_value += value
            total_cost += cost
            # recommendation simple: if pnl >= 0 -> Halten, else Verkaufen
            recommendation = "Halten" if pnl >= 0 else "Verkaufen"
            # format
            value_s = f"{value:,.4f}"
            cost_s = f"{cost:,.4f}"
            cur_s = f"{cur_price:,.4f}"
            pnl_s = f"{pnl:,.4f} ({pnl_pct:+.2f}%)"
            # remove button id
            btn_key = f"remove_{item['item_id']}"
            # create row html with button (button below table)
            rows_html.append((item, cost_s, cur_s, value_s, pnl_s, recommendation, btn_key))

        # Summary metrics
        st.markdown(f"<div class='card'><div><strong>Gesamtwert:</strong> <span class='metric'>{total_value:,.4f} €</span></div><div class='small'>Eingezahlter Wert: {total_cost:,.4f} €</div></div>", unsafe_allow_html=True)
        st.markdown("---")
        # Render table and action buttons per row
        table_html = "<table>" + header
        for (item, cost_s, cur_s, value_s, pnl_s, recommendation, btn_key) in rows_html:
            table_html += f"<tr><td>{item['name']}</td><td>{item['category']}</td><td>{item['qty']}</td><td>{item['buy_price']}</td><td>{cur_s}</td><td>{value_s}</td><td>{pnl_s}</td><td>{recommendation}</td><td></td></tr>"
        table_html += "</table>"
        st.markdown(table_html, unsafe_allow_html=True)

        # Actions (render remove buttons separately to allow per-row interaction)
        for (item, cost_s, cur_s, value_s, pnl_s, recommendation, btn_key) in rows_html:
            cols = st.columns([3,1])
            with cols[0]:
                st.write(f"**{item['name']}** — {item['qty']} × {item['buy_price']} €")
            with cols[1]:
                if st.button("Entfernen", key=btn_key):
                    remove_from_portfolio(item["item_id"])

        st.markdown("---")
        st.button("Portfolio leeren", on_click=lambda: st.session_state.portfolio.clear())

# ---------------------------
# Footer
# ---------------------------
st.markdown("### ℹ️ Hinweise")
st.markdown("- Demo offline: Preise sind simuliert, aber deterministisch pro Asset (bleiben gleich zwischen Reloads).")
st.markdown("- Empfehlung: einfache Regel (Gewinn ≥ 0 → Halten, Gewinn < 0 → Verkaufen).")
st.markdown("- Portfolio wird nur in dieser Session gespeichert (Streamlit session_state).")
