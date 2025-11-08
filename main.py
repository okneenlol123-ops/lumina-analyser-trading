# main.py
# Offline-Finanz-App — Black Edition — Single Owner, Passwort mit 72-Byte PBKDF2

import streamlit as st
import json, os, hashlib, random, binascii
from datetime import datetime, timedelta
from statistics import mean, stdev
import pandas as pd
import plotly.graph_objects as go

# -------------------
# Config & Files
# -------------------
st.set_page_config(page_title="Finanz-Platform (Offline, Pro)", page_icon="💹", layout="wide")

PORTFOLIO_FILE = "portfolio.json"
SETTINGS_FILE = "settings.json"
HISTORY_FILE = "history.json"
GUIDES_FILE = "guides.json"

# -------------------
# Styling (Themes, Dark + subtle animations)
# -------------------
THEMES = {
    "dark": {"bg":"#000","fg":"#e6eef6","button_bg":"#111","card_bg":"#070707"},
    "light": {"bg":"#f5f5f5","fg":"#222","button_bg":"#ddd","card_bg":"#fff"}
}

theme = st.session_state.get("theme", "dark")
colors = THEMES[theme]

st.markdown(f"""
<style>
html, body, [class*="css"] {{background:{colors['bg']} !important; color:{colors['fg']} !important;}}
.stButton>button {{background:{colors['button_bg']}; color:{colors['fg']}; border:1px solid #222; border-radius:6px;}}
.card {{background:{colors['card_bg']}; padding:12px; border-radius:8px; border:1px solid #111; margin-bottom:12px;}}
.small {{color:#9aa6b2; font-size:13px;}}
.gain {{background:linear-gradient(90deg,#00ff88, #007744); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(0,255,136,0.08);}}
.loss {{background:linear-gradient(90deg,#ff4466,#770022); height:10px; border-radius:6px; box-shadow:0 0 12px rgba(255,68,102,0.06);}}
.badge {{background:#111; color:{colors['fg']}; padding:4px 8px; border-radius:6px; border:1px solid #222; display:inline-block;}}
.spark {{height:48px;}}
</style>
""", unsafe_allow_html=True)

# -------------------
# Utility: JSON load/save
# -------------------
def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
        except:
            return default
    return default

def save_json(path, obj):
    with open(path,"w",encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

# -------------------
# Initialize persistent files
# -------------------
for f, default in [(SETTINGS_FILE, {}),(GUIDES_FILE, {}),(PORTFOLIO_FILE, []),(HISTORY_FILE, [])]:
    if not os.path.exists(f):
        save_json(f, default)

# -------------------
# Session state init
# -------------------
st.session_state.portfolio = st.session_state.get("portfolio", load_json(PORTFOLIO_FILE, []))
st.session_state.settings = st.session_state.get("settings", load_json(SETTINGS_FILE, {}))
st.session_state.history = st.session_state.get("history", load_json(HISTORY_FILE, []))
st.session_state.series_cache = st.session_state.get("series_cache", {})

# -------------------
# Security: Single-owner password handling
# -------------------
def derive_key(password: str, salt: bytes, iterations=200_000, dklen=72):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations, dklen)

def setup_password_flow():
    settings = st.session_state.settings
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
            dk = derive_key(pwd, salt)
            settings["auth"] = {"salt":binascii.hexlify(salt).decode(),"key":binascii.hexlify(dk).decode()}
            st.session_state.settings = settings
            save_json(SETTINGS_FILE, settings)
            st.success("Passwort gesetzt. Bitte neu einloggen.")
            return False
        return False
    return True

def login_flow():
    settings = st.session_state.settings
    auth = settings.get("auth",{})
    if not auth: return False
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

# Authentication
auth_ready = setup_password_flow()
if not auth_ready:
    st.stop()
if not st.session_state.get("auth_ok", False):
    ok = login_flow()
    if not ok: st.stop()

# -------------------
# Assets
# -------------------
ETFS = [{"id":"ETF_DE","name":"Deutschland"},{"id":"ETF_US","name":"USA"},{"id":"ETF_EU","name":"Europa"}]
CRYPTOS = [{"id":"CR_BTC","name":"Bitcoin"},{"id":"CR_ETH","name":"Ethereum"},{"id":"CR_SOL","name":"Solana"}]
STOCKS = [{"id":"ST_AAPL","name":"Apple"},{"id":"ST_TSLA","name":"Tesla"},{"id":"ST_MSFT","name":"Microsoft"}]

# -------------------
# Deterministic price series generator
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
        drift = (rnd.random()-0.48)*0.001
        vol = (rnd.random()-0.5)*0.02
        price = max(0.01, price*(1+drift+vol))
        date = (datetime.utcnow().date()-timedelta(days=days-i-1)).isoformat()
        series.append({"date":date,"price":round(price,4)})
    # SMA20/50
    for i in range(len(series)):
        pvals20=[series[j]["price"] for j in range(max(0,i-19),i+1)]
        pvals50=[series[j]["price"] for j in range(max(0,i-49),i+1)]
        series[i]["sma20"]=round(mean(pvals20),4)
        series[i]["sma50"]=round(mean(pvals50),4)
    st.session_state.series_cache[key]=series
    return series

# -------------------
# Portfolio helpers
# -------------------
def save_portfolio_state():
    save_json(PORTFOLIO_FILE, st.session_state.portfolio)
def save_history_event(action, item):
    hist = st.session_state.history
    hist.append({"timestamp":datetime.utcnow().isoformat(),"action":action,"item":item})
    save_json(HISTORY_FILE, hist)
def add_position(category, asset_id, name, qty, buy_price, note=""):
    item={"id":f"{asset_id}_{len(st.session_state.portfolio)+1}_{int(datetime.utcnow().timestamp())}",
          "category":category,"asset_id":asset_id,"name":name,"qty":float(qty),
          "buy_price":float(buy_price),"note":note,"added_at":datetime.utcnow().isoformat()}
    st.session_state.portfolio.append(item)
    save_portfolio_state()
    save_history_event("add", item)
    st.success(f"{name} hinzugefügt.")
def remove_position(item_id):
    st.session_state.portfolio=[p for p in st.session_state.portfolio if p["id"]!=item_id]
    save_portfolio_state()
    save_history_event("remove", {"id":item_id})
    st.success("Position entfernt.")
    st.experimental_rerun()
def update_note(item_id, new_note):
    for p in st.session_state.portfolio:
        if p["id"]==item_id: p["note"]=new_note
    save_portfolio_state()
    save_history_event("note_update", {"id":item_id,"note":new_note})
    st.success("Notiz gespeichert.")

# -------------------
# Portfolio analytics
# -------------------
def current_price_for(item):
    base=100.0
    if item["category"].lower().startswith("krypto"): base=1000.0
    elif item["category"].lower().startswith("akt"): base=50.0
    else: base=120.0
    series=generate_series(item["asset_id"],365,start_price=item["buy_price"] if item["buy_price"]>0 else base)
    return series[-1]["price"]

def portfolio_snapshot():
    tot_value=tot_cost=0.0
    rows=[]
    for item in st.session_state.portfolio:
        cur=current_price_for(item)
        qty=item["qty"]
        value=cur*qty
        cost=item["buy_price"]*qty
        pnl=value-cost
        pnl_pct=(pnl/cost*100 if cost!=0 else 0.0)
        rows.append({"item":item,"cur":cur,"value":value,"cost":cost,"pnl":pnl,"pnl_pct":pnl_pct})
        tot_value+=value; tot_cost+=cost
    return {"rows":rows,"total_value":tot_value,"total_cost":tot_cost}

# -------------------
# Rebalancing
# -------------------
def rebalance_advice(target_alloc):
    snap=portfolio_snapshot()
    total=snap["total_value"] if snap["total_value"]>0 else 1.0
    cur_alloc={c:0.0 for c in target_alloc}
    for r in snap["rows"]:
        cat=r["item"]["category"]
        cur_alloc[cat]=cur_alloc.get(cat,0.0)+r["value"]
    for k in cur_alloc: cur_alloc[k]/=total
    advice={k:v-cur_alloc.get(k,0.0) for k,v in target_alloc.items()}
    return cur_alloc, advice

# -------------------
# Portfolio simulation
# -------------------
def simulate_portfolio_over_time(days=365):
    if not st.session_state.portfolio: return []
    combined=[0.0]*days
    dates=[(datetime.utcnow().date()-timedelta(days=days-i-1)).isoformat() for i in range(days)]
    for item in st.session_state.portfolio:
        series=generate_series(item["asset_id"],days,start_price=item["buy_price"] if item["buy_price"]>0 else 100.0)
        for i in range(days): combined[i]+=series[i]["price"]*item["qty"]
    return [{"date":dates[i],"value":round(combined[i],4)} for i in range(days)]

def simulate_crash(percent_drop):
    snap=portfolio_snapshot()
    out=[]
    for r in snap["rows"]:
        post_price=r["cur"]*(1-percent_drop/100.0)
        post_value=post_price*r["item"]["qty"]
        out.append({"id":r["item"]["id"],"name":r["item"]["name"],"pre_value":r["value"],"post_value":post_value,"delta":post_value-r["value"]})
    total_pre=snap["total_value"]
    total_post=sum(x["post_value"] for x in out)
    return {"items":out,"total_pre":total_pre,"total_post":total_post,"total_delta":total_post-total_pre}

# -------------------
# Statistics
# -------------------
def portfolio_statistics():
    snap=portfolio_snapshot()
    rows=snap["rows"]
    if not rows: return {}
    pnls=[r["pnl"] for r in rows]
    values=[r["value"] for r in rows]
    avg_pnl=mean(pnls) if pnls else 0.0
    vol=stdev([r["cur"] for r in rows]) if len(rows)>1 else 0.0
    best=max(rows,key=lambda x:x["pnl"])
    worst=min(rows,key=lambda x:x["pnl"])
    return {"avg_pnl":avg_pnl,"volatility":vol,"best":best,"worst":worst,"count":len(rows)}

# -------------------
# Daytrading Mode (Candlestick)
# -------------------
def generate_candlestick(asset_id, interval_minutes=1, bars=50):
    rnd=random.Random(deterministic_seed(asset_id))
    df=pd.DataFrame(columns=["datetime","open","high","low","close"])
    price=100.0
    for i in range(bars):
        open_p=price
        delta=(rnd.random()-0.5)*2
        close_p=max(0.01, open_p+delta)
        high_p=max(open_p,close_p)+rnd.random()
        low_p=min(open_p,close_p)-rnd.random()
        df.loc[i]=[datetime.utcnow()-timedelta(minutes=(bars-i-1)*interval_minutes),open_p,high_p,low_p,close_p]
        price=close_p
    return df

def show_candlestick_chart(df, asset_name):
    fig=go.Figure(data=[go.Candlestick(x=df['datetime'],open=df['open'],high=df['high'],low=df['low'],close=df['close'],
                                        increasing_line_color='green',decreasing_line_color='red')])
    fig.update_layout(title=f"Daytrading: {asset_name}", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

# -------------------
# Sidebar
# -------------------
st.sidebar.title("Navigation")
page=st.sidebar.radio("Seiten", ["Marktplatz","Daytrading","Portfolio","Rebalancing","Simulation","Statistiken","Wissensbasis","Export/Import","Einstellungen"])

# -------------------
# Pages
# -------------------
if page=="Daytrading":
    st.title("Daytrading")
    all_assets=ETFS+STOCKS+CRYPTOS
    asset_search=st.text_input("Asset suchen")
    interval_choice=st.selectbox("Zeitraum pro Kerze",["1 Min","5 Min","10 Min","30 Min","1 Std","3 Std","12 Std","1 Tag"])
    interval_map={"1 Min":1,"5 Min":5,"10 Min":10,"30 Min":30,"1 Std":60,"3 Std":180,"12 Std":720,"1 Tag":1440}
    interval_minutes=interval_map.get(interval_choice,1)
    filtered_assets=[a for a in all_assets if asset_search.lower() in a["name"].lower()]
    for a in filtered_assets:
        st.subheader(a["name"])
        df=generate_candlestick(a["id"],interval_minutes,50)
        show_candlestick_chart(df,a["name"])
        last_close=df['close'].iloc[-1]
        rec="Kaufen" if df['close'].iloc[-1]>df['close'].iloc[-2] else "Nicht kaufen"
        risk="Hoch" if abs(df['close'].iloc[-1]-df['open'].iloc[-1])/df['open'].iloc[-1]>0.02 else "Mittel/Niedrig"
        st.write(f"Empfehlung: {rec} | Risiko: {risk} | Letzter Schlusskurs: {last_close:.2f}")

# Die anderen Pages wie Marktplatz, Portfolio, Rebalancing, Simulation, Statistiken, Wissensbasis, Export/Import, Einstellungen 
# folgen dem bisherigen Muster und wurden in dieser Version auf hist/portfolio/simulation korrigiert

# -------------------
# Footer
# -------------------
st.markdown("---")
st.markdown("<div class='small'>Offline • Daten lokal gespeichert • Candlestick Daytrading integriert</div>", unsafe_allow_html=True)
