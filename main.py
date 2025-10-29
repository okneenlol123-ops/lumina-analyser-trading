import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="ETF Analyse Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📊 ETF Analyse — Multi-Region")
st.markdown("""
Dieses Dashboard zeigt ETF-Daten aus verschiedenen Regionen (Deutschland, USA, Europa, Asien, Welt)
mit einer einfachen technischen Analyse (SMA 20/50).  
*Hinweis: Keine Anlageberatung — nur technische Demo.*
""")

# ---- API Key Eingabe (optional)
api_key = st.text_input("Alpha Vantage API Key (optional)", type="password", placeholder="z. B. ABCD1234XYZ")
if api_key:
    st.session_state['api_key'] = api_key

# ---- Lade HTML/JS Frontend
html_path = Path(__file__).parent / "etf_frontend.html"
if not html_path.exists():
    st.error("❌ Die Datei `etf_frontend.html` fehlt! Bitte in den Projektordner legen.")
else:
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    # optional: Übergib API-Key dynamisch ans Frontend (als JS-Variable)
    html_content = html_content.replace('const apiKey = "";', f'const apiKey = "{api_key}";')

    # Zeige HTML im Streamlit-Frame
    st.components.v1.html(html_content, height=1200, scrolling=True)
