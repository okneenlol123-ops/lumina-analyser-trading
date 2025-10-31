import streamlit as st
import random
from datetime import datetime, timedelta
import hashlib

# ---------------------------
# App-Konfiguration & Styling
# ---------------------------
st.set_page_config(page_title="Finanz-Dashboard (Black Edition)", page_icon="💹", layout="wide")

st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        background-color: #000000 !important;
        color: #e6eef6 !important;
    }
    .stButton>button {
        background-color:#111111;
        color:#e6eef6;
        border:1px solid #2f2f2f;
        border-radius:6px;
    }
    h1, h2, h3, h4, h5, h6 {color: #ffffff;}
    .card {
        background:#0a0a0a;
        padding:12px;
        border-radius:8px;
        border:1px solid #111;
        margin-bottom:12px;
    }
    .gain {
        background: linear-gradient(90deg, #00ff99 0%, #007755 100%);
        height: 10px; border-radius:4px;
    }
    .loss {
        background: linear-gradient(90deg, #ff0044 0%, #770022 100%);
        height: 10px; border-radius:4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("💹 Finanz-Dashboard – Black Edition (Offline)")
st.markdown("**Hinweis:** Offline-Demo mit simulierten Kursen. Keine Anlageempfehlung.")

# ---------------------------
# Session State
# ---------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

if "price_series" not in st.session_state:
    st.session_state.price_series = {}

# ---------------------------
# Assets
# ---------------------------
ETFS = [
    {"id": "ETF_DE", "name": "Deutschland"},
    {"id": "ETF_US", "name": "USA"},
    {"id": "ETF_EU", "name": "Europa"},
    {"id": "ETF_AS", "name": "Asien"},
    {"id": "ETF_WL", "name": "Welt"},
]
CRYPTOS = [
    {"id": "CR_BTC", "name": "Bitcoin"},
    {"id": "CR_ETH", "name": "Ethereum"},
    {"id": "CR_SOL", "name": "Solana"},
]
STOCKS = [
    {"id": "ST_AAPL", "name": "Apple"},
    {"id": "ST_TSLA", "name": "Tesla"},
    {"id": "ST_MSFT", "name": "Microsoft"},
]

# ---------------------------
# Preis-Simulation
# ---------------------------
def deterministic_seed(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest(), 16) % (2**31)

def generate_price_series(asset_id: str, days: int = 365, start_price: float = 100.0):
    key = f"{asset_id}_{days}"
    if key in st.session_state.price_series:
        return st.session_state.price_series[key]
    rnd = random.Random(deterministic_seed(asset_id))
    prices = []
    price = start_price
    for i in range(days):
        drift = (rnd.random() - 0.48) * 0.001
        vol = (rnd.random() - 0.5) * 0.02
        price = max(0.01, price * (1 + drift + vol))
        prices.append((datetime.utcnow().date() - timedelta(days=days - i - 1), round(price, 3)))
    sma20, sma50 = [], []
    for i in range(len(prices)):
        w20 = [p[1] for p in prices[max(0, i - 19): i + 1]]
        w50 = [p[1] for p in prices[max(0, i - 49): i + 1]]
        sma20.append(sum(w20) / len(w20))
        sma50.append(sum(w50) / len(w50))
    series = [(prices[i][0], prices[i][1], sma20[i], sma50[i]) for i in range(len(prices))]
    st.session_state.price_series[key] = series
    return series

# ---------------------------
# Portfolio-Funktion
# ---------------------------
def add_to_portfolio(category, asset_id, name, qty, buy_price):
    st.session_state.portfolio.append({
        "category": category,
        "asset_id": asset_id,
        "name": name,
        "qty": qty,
        "buy_price": buy_price,
    })
    st.success(f"{name} hinzugefügt.")

# ---------------------------
# Layout
# ---------------------------
col_left, col_right = st.columns([2, 1])
intervals = {"1 Monat": 30, "3 Monate": 90, "1 Jahr": 365, "5 Jahre": 1825}

# --------- Kategorien ----------
def show_category(cat_name, assets, currency="€"):
    st.header(cat_name)
    for a in assets:
        interval = st.selectbox(f"Zeitraum für {a['name']}", intervals.keys(), key=f"{a['id']}_interval")
        days = intervals[interval]
        series = generate_price_series(a['id'], days)
        chart_data = {str(s[0]): s[1] for s in series}
        st.line_chart(chart_data)

        cur_price = series[-1][1]
        sma20 = series[-1][2]
        sma50 = series[-1][3]

        st.write(f"**Aktueller Preis:** {cur_price:.2f} {currency}")

        if sma20 > sma50:
            st.success("📈 Empfehlung: **Kaufen** (Trend positiv)")
        else:
            st.error("📉 Empfehlung: **Nicht kaufen** (Trend schwach)")

        with st.form(key=f"add_{a['id']}"):
            qty = st.number_input("Menge", min_value=0.0, value=1.0, step=0.1)
            buy_price = st.number_input("Kaufpreis", min_value=0.01, value=float(cur_price), step=0.1)
            if st.form_submit_button("Zu Portfolio hinzufügen"):
                add_to_portfolio(cat_name, a["id"], a["name"], qty, buy_price)
        st.markdown("---")

with col_left:
    tabs = st.tabs(["📊 ETFs", "💰 Kryptowährungen", "🏦 Aktien"])
    with tabs[0]:
        show_category("ETF", ETFS)
    with tabs[1]:
        show_category("Krypto", CRYPTOS, "$")
    with tabs[2]:
        show_category("Aktie", STOCKS)

# --------- Portfolio ----------
with col_right:
    st.header("📂 Portfolio")

    if not st.session_state.portfolio:
        st.info("Noch keine Einträge im Portfolio.")
    else:
        total_value, total_cost = 0, 0

        for item in st.session_state.portfolio:
            cur_price = generate_price_series(item["asset_id"], 365, item["buy_price"])[-1][1]
            wert = cur_price * item["qty"]
            pnl = wert - item["qty"] * item["buy_price"]
            pnl_pct = (pnl / (item["qty"] * item["buy_price"])) * 100 if item["buy_price"] > 0 else 0
            total_value += wert
            total_cost += item["qty"] * item["buy_price"]

            color = "gain" if pnl >= 0 else "loss"
            rec = "Halten" if pnl >= 0 else "Verkaufen"

            st.markdown(f"<div class='card'>", unsafe_allow_html=True)
            st.write(f"**{item['name']}** ({item['category']})")
            st.write(f"Menge: {item['qty']} | Kaufpreis: {item['buy_price']:.2f} | Aktuell: {cur_price:.2f}")
            st.write(f"Gewinn/Verlust: **{pnl:+.2f} ({pnl_pct:+.2f}%)** → Empfehlung: **{rec}**")
            st.markdown(f"<div class='{color}' style='width:{min(abs(pnl_pct)*3,100)}%'></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        total_pnl = total_value - total_cost
        total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("💼 Gesamtübersicht")
        st.write(f"**Gesamtwert:** {total_value:.2f} €")
        st.write(f"**Gesamtrendite:** {total_pnl:+.2f} € ({total_pnl_pct:+.2f}%)")

        bar_color = "gain" if total_pnl >= 0 else "loss"
        st.markdown(f"<div class='{bar_color}' style='width:{min(abs(total_pnl_pct)*3,100)}%'></div>", unsafe_allow_html=True)
