import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# ---------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------
st.set_page_config(
    page_title="FailSafe AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)
# ---------------------------------------------------------------
# GLOBAL CSS — Pure Neon theme
# ---------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Space+Mono:wght@400;700&family=Rajdhani:wght@300;400;500;600&display=swap');

:root {
    --bg-base:      #050508;
    --bg-card:      #0A0A12;
    --bg-card2:     #0E0E1A;
    --border:       #1C1C30;
    --neon-green:   #39FF14;
    --neon-magenta: #FF2079;
    --neon-cyan:    #00FFFF;
    --neon-yellow:  #FFE600;
    --neon-purple:  #BF5FFF;
    --neon-orange:  #FF6B00;
    --text-primary: #F0F0FF;
    --text-muted:   #7070A0;
    --text-dim:     #3A3A5C;
}

@keyframes neon-pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.7; }
}
@keyframes scan-line {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: 'Rajdhani', sans-serif;
}

[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 40% at 15% 0%,   rgba(57,255,20,0.06)   0%, transparent 55%),
        radial-gradient(ellipse 60% 50% at 85% 100%,  rgba(255,32,121,0.07)  0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 50%  50%,  rgba(0,255,255,0.03)   0%, transparent 65%);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: #060609 !important;
    border-right: 1px solid #1C1C30 !important;
    box-shadow: 2px 0 20px rgba(57,255,20,0.04) !important;
}

h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
h1 { color: var(--text-primary) !important; font-weight: 900 !important; }
h2 { color: var(--text-primary) !important; font-weight: 700 !important; }
h3 { color: var(--neon-cyan)    !important; font-weight: 600 !important; }

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 20px !important;
    position: relative !important;
    overflow: hidden !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 20px rgba(57,255,20,0.15), 0 0 40px rgba(57,255,20,0.06) !important;
    border-color: rgba(57,255,20,0.35) !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--neon-green), var(--neon-cyan), var(--neon-magenta));
    animation: neon-pulse 2.5s ease-in-out infinite;
}
[data-testid="stMetricLabel"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    font-weight: 400 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 24px !important;
    font-weight: 700 !important;
    color: var(--neon-green) !important;
    text-shadow: 0 0 10px rgba(57,255,20,0.5) !important;
}

.stButton > button {
    background: transparent !important;
    color: var(--neon-green) !important;
    border: 1px solid var(--neon-green) !important;
    border-radius: 4px !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    letter-spacing: 0.12em !important;
    padding: 12px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 0 10px rgba(57,255,20,0.2), inset 0 0 10px rgba(57,255,20,0.03) !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: rgba(57,255,20,0.08) !important;
    box-shadow: 0 0 20px rgba(57,255,20,0.4), 0 0 60px rgba(57,255,20,0.15), inset 0 0 20px rgba(57,255,20,0.05) !important;
    color: #ffffff !important;
    border-color: var(--neon-green) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    border: 1px solid #1C1C30 !important;
    border-radius: 4px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 12px !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 0 1px var(--neon-cyan), 0 0 12px rgba(0,255,255,0.2) !important;
}
.stSelectbox > div > div > div { color: var(--text-primary) !important; }

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 4px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-muted) !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    font-size: 11px !important;
    border-radius: 3px !important;
    padding: 8px 18px !important;
    border: none !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.06em !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(57,255,20,0.08) !important;
    color: var(--neon-green) !important;
    border: 1px solid rgba(57,255,20,0.3) !important;
    box-shadow: 0 0 12px rgba(57,255,20,0.15) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--neon-green), var(--neon-cyan)) !important;
    border-radius: 99px !important;
    box-shadow: 0 0 8px rgba(57,255,20,0.4) !important;
}
.stProgress > div > div {
    background: var(--bg-card2) !important;
    border-radius: 99px !important;
    border: 1px solid var(--border) !important;
}

[data-baseweb="tag"] {
    background: rgba(57,255,20,0.1) !important;
    border: 1px solid rgba(57,255,20,0.3) !important;
    border-radius: 3px !important;
    color: var(--neon-green) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    text-shadow: 0 0 6px rgba(57,255,20,0.4) !important;
}

hr { border-color: var(--border) !important; opacity: 0.5 !important; }
.stMarkdown p { color: var(--text-primary) !important; line-height: 1.7 !important; font-family: 'Rajdhani', sans-serif !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--neon-green); border-radius: 99px; box-shadow: 0 0 6px rgba(57,255,20,0.5); }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# HTML HELPERS — Neon Theme
# ---------------------------------------------------------------
def page_hero(icon, title, subtitle, accent="#39FF14"):
    st.markdown(f"""
    <div style="padding:40px 0 32px 0;">
        <div style="font-family:'Space Mono',monospace;font-size:10px;color:{accent};
                    letter-spacing:0.2em;text-transform:uppercase;margin-bottom:12px;
                    text-shadow:0 0 8px {accent}88;">
            // FAILSAFE AI PLATFORM
        </div>
        <div style="display:flex;align-items:center;gap:18px;margin-bottom:12px;">
            <div style="font-size:32px;background:#0A0A12;
                        border:1px solid {accent}55;border-radius:6px;
                        width:60px;height:60px;display:flex;align-items:center;justify-content:center;
                        box-shadow:0 0 16px {accent}33,inset 0 0 16px {accent}11;">
                {icon}
            </div>
            <h1 style="font-family:'Orbitron',monospace;font-size:30px;font-weight:900;
                       color:#F0F0FF;margin:0;letter-spacing:0.06em;line-height:1.1;
                       text-transform:uppercase;">
                {title}
            </h1>
        </div>
        <p style="font-family:'Rajdhani',sans-serif;font-size:16px;color:#7070A0;
                  margin:0;font-weight:400;max-width:620px;line-height:1.6;letter-spacing:0.02em;">
            {subtitle}
        </p>
        <div style="margin-top:24px;height:1px;
                    background:linear-gradient(90deg,{accent}88,{accent}22,transparent);
                    box-shadow:0 0 6px {accent}44;"></div>
    </div>
    """, unsafe_allow_html=True)

def section_header(text, icon="", accent="#39FF14"):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:10px;margin:28px 0 14px 0;">
        <span style="font-size:16px;">{icon}</span>
        <span style="font-family:'Orbitron',monospace;font-size:12px;font-weight:700;
                     color:{accent};letter-spacing:0.12em;text-transform:uppercase;
                     text-shadow:0 0 8px {accent}66;">{text}</span>
        <div style="flex:1;height:1px;background:linear-gradient(90deg,{accent}44,transparent);
                    margin-left:8px;box-shadow:0 0 4px {accent}33;"></div>
    </div>
    """, unsafe_allow_html=True)

def stat_card(label, value, sub="", color="#39FF14", icon=""):
    return f"""
    <div style="background:#0A0A12;border:1px solid #1C1C30;border-radius:6px;
                padding:20px 18px;position:relative;overflow:hidden;
                transition:all 0.2s ease;">
        <div style="position:absolute;top:0;left:0;right:0;height:1px;
                    background:linear-gradient(90deg,transparent,{color},{color}44,transparent);
                    box-shadow:0 0 6px {color}66;"></div>
        <div style="position:absolute;bottom:0;left:0;width:2px;height:40%;
                    background:{color};opacity:0.4;box-shadow:0 0 6px {color};"></div>
        <div style="position:absolute;top:14px;right:14px;font-size:18px;opacity:0.12;">{icon}</div>
        <div style="font-family:'Space Mono',monospace;font-size:9px;color:#7070A0;
                    text-transform:uppercase;letter-spacing:0.14em;margin-bottom:10px;">
            {label}
        </div>
        <div style="font-family:'Orbitron',monospace;font-size:22px;font-weight:700;
                    color:{color};line-height:1;margin-bottom:5px;
                    text-shadow:0 0 12px {color}66;">
            {value}
        </div>
        <div style="font-family:'Rajdhani',sans-serif;font-size:12px;color:#3A3A5C;
                    letter-spacing:0.04em;">{sub}</div>
    </div>"""

def risk_badge(level):
    if "High" in level:
        return '<span style="background:rgba(255,32,121,0.12);border:1px solid rgba(255,32,121,0.5);color:#FF2079;padding:4px 14px;border-radius:3px;font-family:\'Space Mono\',monospace;font-size:10px;font-weight:700;letter-spacing:0.1em;text-shadow:0 0 8px rgba(255,32,121,0.6);box-shadow:0 0 10px rgba(255,32,121,0.15);">▲ HIGH RISK</span>'
    elif "Moderate" in level:
        return '<span style="background:rgba(255,230,0,0.1);border:1px solid rgba(255,230,0,0.45);color:#FFE600;padding:4px 14px;border-radius:3px;font-family:\'Space Mono\',monospace;font-size:10px;font-weight:700;letter-spacing:0.1em;text-shadow:0 0 8px rgba(255,230,0,0.5);box-shadow:0 0 10px rgba(255,230,0,0.12);">◆ MODERATE RISK</span>'
    else:
        return '<span style="background:rgba(57,255,20,0.1);border:1px solid rgba(57,255,20,0.4);color:#39FF14;padding:4px 14px;border-radius:3px;font-family:\'Space Mono\',monospace;font-size:10px;font-weight:700;letter-spacing:0.1em;text-shadow:0 0 8px rgba(57,255,20,0.5);box-shadow:0 0 10px rgba(57,255,20,0.12);">● LOW RISK</span>'

def flag_card(text, kind="red"):
    colors = {
        "red":   ("#FF2079","rgba(255,32,121,0.07)","rgba(255,32,121,0.25)","!"),
        "green": ("#39FF14","rgba(57,255,20,0.05)", "rgba(57,255,20,0.22)", "✓"),
        "warn":  ("#FFE600","rgba(255,230,0,0.06)", "rgba(255,230,0,0.22)",  "?"),
    }
    c, bg, border, ico = colors[kind]
    clean = text.lstrip("🚩✅⚠️ ")
    return f"""
    <div style="background:{bg};border:1px solid {border};border-left:2px solid {c};
                border-radius:4px;padding:12px 16px;margin:5px 0;
                display:flex;align-items:flex-start;gap:12px;
                box-shadow:inset 0 0 20px {c}08;">
        <span style="color:{c};font-size:12px;font-weight:700;line-height:1.6;
                     font-family:'Space Mono',monospace;text-shadow:0 0 6px {c}88;
                     min-width:14px;">{ico}</span>
        <span style="font-family:'Rajdhani',sans-serif;font-size:14px;color:#D0D0F0;
                     line-height:1.6;letter-spacing:0.02em;">{clean}</span>
    </div>"""

def sidebar_logo():
    st.sidebar.markdown("""
    <div style="padding:24px 8px 20px 8px;text-align:center;
                border-bottom:1px solid #1C1C30;margin-bottom:20px;">
        <div style="font-family:'Orbitron',monospace;font-size:18px;font-weight:900;
                    color:#F0F0FF;letter-spacing:0.08em;line-height:1;text-transform:uppercase;">
            🛡 Fail<span style="color:#39FF14;text-shadow:0 0 10px rgba(57,255,20,0.7);">Safe</span>
            <span style="color:#00FFFF;text-shadow:0 0 10px rgba(0,255,255,0.6);">AI</span>
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:8px;color:#3A3A5C;
                    letter-spacing:0.2em;text-transform:uppercase;margin-top:6px;">
            Risk Intelligence Platform
        </div>
        <div style="margin-top:12px;height:1px;
                    background:linear-gradient(90deg,transparent,#39FF1444,transparent);"></div>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------
sidebar_logo()

st.sidebar.markdown(
    '<div style="font-family:\'Space Mono\',monospace;font-size:9px;color:#3A3A5C;'
    'text-transform:uppercase;letter-spacing:0.18em;margin-bottom:8px;">// Navigation</div>',
    unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["📊  Main Dashboard", "🧠  AI Risk Mentor", "🔍  Failure Verification AI"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------
# DATABASE  ← ALL CHANGES ARE IN THIS BLOCK
# ---------------------------------------------------------------
# CHANGE 1: Use sqlite3.connect() with a local file path (no DSN / credentials needed)
DB_PATH = os.path.join(os.getcwd(), "startups.db")

try:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()

    # CHANGE 2: CREATE TABLE IF NOT EXISTS with SQLite types (no schema prefix)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS STARTUPS (
        STARTUP         TEXT PRIMARY KEY,
        INDUSTRY        TEXT,
        MONTHLYREVENUE  INTEGER,
        MONTHLYEXPENSES INTEGER,
        GROWTHRATE      REAL,
        MONTHSRUNWAY    INTEGER,
        TEAMSIZE        INTEGER,
        MARKETRISK      TEXT,
        FUNDING         INTEGER
    )
    """)
    conn.commit()

    # Check if table is empty
    cursor.execute("SELECT COUNT(*) FROM STARTUPS")
    count = cursor.fetchone()[0]

    # CHANGE 3: INSERT uses plain table name (no SYSTEM. prefix) and ? placeholders
    if count == 0:
        cursor.executemany(
            """INSERT INTO STARTUPS
               (STARTUP, INDUSTRY, MONTHLYREVENUE, MONTHLYEXPENSES,
                GROWTHRATE, MONTHSRUNWAY, TEAMSIZE, MARKETRISK, FUNDING)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ('AlphaTech', 'Tech',   50000, 70000, -5.0, 4, 6,  'High',   1000000),
                ('GreenFoods','Food',   80000, 60000, 10.0, 8, 12, 'Low',    1500000),
                ('MediCare',  'Health', 40000, 50000, -8.0, 3, 5,  'Medium',  800000),
            ]
        )
        conn.commit()

    # UI Success message
    st.sidebar.markdown("""
    <div style="background:rgba(57,255,20,0.07);border:1px solid rgba(57,255,20,0.3);
                border-radius:4px;padding:10px 14px;display:flex;align-items:center;gap:8px;
                box-shadow:0 0 12px rgba(57,255,20,0.08);">
        <span style="color:#39FF14;font-size:8px;text-shadow:0 0 6px #39FF14;">●</span>
        <span style="font-family:'Space Mono',monospace;font-size:10px;
                     color:#39FF14;font-weight:700;text-shadow:0 0 6px rgba(57,255,20,0.4);
                     letter-spacing:0.06em;">SQLITE DB ONLINE</span>
    </div>""", unsafe_allow_html=True)

except Exception as e:
    st.sidebar.markdown(f"""
    <div style="background:rgba(255,32,121,0.07);border:1px solid rgba(255,32,121,0.3);
                border-radius:4px;padding:10px 14px;">
        <span style="font-family:'Space Mono',monospace;font-size:10px;color:#FF2079;
                     text-shadow:0 0 6px rgba(255,32,121,0.4);">✗ DB OFFLINE: {e}</span>
    </div>""", unsafe_allow_html=True)
    st.stop()

# ---------------------------------------------------------------
# LOAD & PROCESS DATA
# ---------------------------------------------------------------
try:
    df = pd.read_sql("SELECT * FROM STARTUPS", con=conn)
except Exception as e:
    st.error(f"Database Error: {e}"); st.stop()

df.rename(columns={
    "STARTUP":"Startup","INDUSTRY":"Industry","FUNDING":"Funding (₹)",
    "MONTHLYREVENUE":"Monthly Revenue","MONTHLYEXPENSES":"Monthly Expenses",
    "TEAMSIZE":"Team Size","GROWTHRATE":"Growth Rate (%)","MARKETRISK":"Market Risk",
    "MONTHSRUNWAY":"Months Runway","RISKSCORE":"Risk Score","RISKLEVEL":"Risk Level",
    "PREMIUMAMOUNT":"Premium Amount","ESTIMATEDPAYOUT":"Estimated Payout"
}, inplace=True)

def calculate_risk(row):
    burn  = max(0, row["Monthly Expenses"] - row["Monthly Revenue"])
    rev_d = abs(row["Growth Rate (%)"]) * 1000 if row["Growth Rate (%)"] < 0 else 0
    run_r = 50000 if row["Months Runway"] <= 3 else 20000 if row["Months Runway"] <= 6 else 0
    tm_r  = (10 - row["Team Size"]) * 2000 if row["Team Size"] < 10 else 0
    mkt_r = 40000 if row["Market Risk"] == "High" else 20000 if row["Market Risk"] == "Medium" else 0
    return max(0, burn + rev_d + run_r + tm_r + mkt_r)

def risk_level(score):
    if score > 100000: return "High Risk"
    elif score > 50000: return "Moderate Risk"
    else: return "Low Risk"

def premium_rate(level):
    return 0.12 if "High" in level else 0.10 if "Moderate" in level else 0.08

def payout_rate(level):
    return 0.40 if "High" in level else 0.50 if "Moderate" in level else 0.60

df["Risk Score"]       = df.apply(calculate_risk, axis=1)
df["Risk Level"]       = df["Risk Score"].apply(risk_level)
df["Risk Level Display"] = df["Risk Level"].map({
    "High Risk":"High Risk 🔴","Moderate Risk":"Moderate Risk 🟡","Low Risk":"Low Risk 🟢"})
df["Premium Amount"]   = df["Funding (₹)"] * df["Risk Level"].apply(premium_rate)
df["Estimated Payout"] = df["Funding (₹)"] * df["Risk Level"].apply(payout_rate)

# ---------------------------------------------------------------
# ML MODEL
# ---------------------------------------------------------------
@st.cache_resource
def train_ml_model(data):
    ml = data.copy()
    le_mkt = LabelEncoder()
    ml["Market Risk Enc"] = le_mkt.fit_transform(ml["Market Risk"])
    ind_dum = pd.get_dummies(ml["Industry"], prefix="Industry")
    ml = pd.concat([ml, ind_dum], axis=1)
    fcols = ["Monthly Revenue","Monthly Expenses","Growth Rate (%)","Months Runway",
             "Team Size","Market Risk Enc","Funding (₹)"] + list(ind_dum.columns)
    le_risk = LabelEncoder()
    ml["Risk Label"] = le_risk.fit_transform(ml["Risk Level"])
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(ml[fcols].fillna(0), ml["Risk Label"])
    return clf, le_mkt, le_risk, fcols, list(ind_dum.columns)

clf, le_market, le_risk, feature_cols, industry_cols = train_ml_model(df)

def prepare_features(revenue, expenses, growth, runway, team, market, funding, industry):
    row = {
        "Monthly Revenue": revenue, "Monthly Expenses": expenses,
        "Growth Rate (%)": growth, "Months Runway": runway, "Team Size": team,
        "Market Risk Enc": le_market.transform([market])[0] if market in le_market.classes_ else 0,
        "Funding (₹)": funding,
    }
    for col in industry_cols:
        row[col] = 1 if industry == col.replace("Industry_","") else 0
    return pd.DataFrame([row])[feature_cols].fillna(0)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — MAIN DASHBOARD
# ═══════════════════════════════════════════════════════════════
if page == "📊  Main Dashboard":

    page_hero("📊","Risk Intelligence Dashboard",
              "Real-time financial protection analytics for your startup portfolio.")

    st.sidebar.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#3A3A5C;'
        'text-transform:uppercase;letter-spacing:0.18em;margin:20px 0 8px 0;">//Filters</div>',
        unsafe_allow_html=True)
    industry_filter = st.sidebar.multiselect("Industry", df["Industry"].unique(), default=df["Industry"].unique())
    risk_filter     = st.sidebar.multiselect("Risk Level", df["Risk Level Display"].unique(), default=df["Risk Level Display"].unique())
    filtered_df     = df[(df["Industry"].isin(industry_filter)) & (df["Risk Level Display"].isin(risk_filter))]

    # KPI row
    section_header("Financial Engine","💰","#39FF14")
    total_premium     = filtered_df["Premium Amount"].sum()
    high_risk         = filtered_df[filtered_df["Risk Level"] == "High Risk"]
    expected_failures = int(len(high_risk) * 0.3)
    avg_payout        = filtered_df["Estimated Payout"].mean() if len(filtered_df) > 0 else 0
    expected_payout   = expected_failures * avg_payout
    buffer            = total_premium - expected_payout
    failure_rate      = len(high_risk) / len(filtered_df) if len(filtered_df) > 0 else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    for col, lbl, val, sub, color, ico in [
        (c1,"Total Premium",    f"₹{int(total_premium):,}",    "Collected",        "#00FFFF","💎"),
        (c2,"Exp. Failures",    str(expected_failures),         f"of {len(filtered_df)}","#FF2079","⚠"),
        (c3,"Exp. Payout",      f"₹{int(expected_payout):,}",  "Liability",        "#FFE600","📤"),
        (c4,"System Buffer",    f"₹{int(buffer):,}",           "Safety margin",    "#39FF14","🛡"),
        (c5,"Failure Rate",     f"{int(failure_rate*100)}%",   "High-risk exposure","#BF5FFF","📉"),
    ]:
        col.markdown(stat_card(lbl, val, sub, color, ico), unsafe_allow_html=True)

    # Failure meter
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    mc = "#FF2079" if failure_rate > 0.5 else "#FFE600" if failure_rate > 0.25 else "#39FF14"
    st.markdown(f"""
    <div style="background:#0A0A12;border:1px solid #1C1C30;
                border-radius:6px;padding:20px 24px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <span style="font-family:'Orbitron',monospace;font-size:15px;font-weight:700;color:#F0F0FF;">
                    Portfolio Failure Probability</span>
                <span style="font-family:'Space Mono',monospace;font-size:11px;color:#7070A0;margin-left:12px;">
                    live assessment</span>
            </div>
            <span style="font-family:'Orbitron',monospace;font-size:24px;font-weight:800;color:{mc};">
                {int(failure_rate*100)}%</span>
        </div>
        <div style="background:#0E0E1A;border-radius:99px;height:10px;border:1px solid #1C1C30;overflow:hidden;">
            <div style="width:{int(failure_rate*100)}%;height:100%;
                        background:linear-gradient(90deg,{mc}CC,{mc});border-radius:99px;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;margin-top:6px;">
            <span style="font-family:'Space Mono',monospace;font-size:10px;color:#3A3A5C;">LOW</span>
            <span style="font-family:'Space Mono',monospace;font-size:10px;color:#3A3A5C;">CRITICAL</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Charts
    section_header("Risk Analytics","📊","#00FFFF")
    col1, col2 = st.columns(2)
    for col, title, data, color in [
        (col1,"Risk Score by Startup",    filtered_df.set_index("Startup")["Risk Score"],           "#00FFFF"),
        (col2,"Avg Risk Score by Industry",filtered_df.groupby("Industry")["Risk Score"].mean(),     "#39FF14"),
    ]:
        with col:
            st.markdown(f'<div style="background:#0A0A12;'
                        f'border:1px solid #1C1C30;border-radius:6px;padding:18px 20px 8px;">'
                        f'<div style="font-family:\'Space Mono\',monospace;font-size:10px;'
                        f'color:#7070A0;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px;">'
                        f'{title}</div>', unsafe_allow_html=True)
            st.bar_chart(data, color=color, height=240)
            st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊   Risk Calculator","➕   Add Startup","✏️   Manage Startups"])

    with tab1:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col0, _ = st.columns([2,1])
        with col0:
            user_funding = st.number_input("Total Funding (₹)", min_value=10000, value=1000000, key="rc_fund")
        col1, col2 = st.columns(2)
        with col1:
            user_revenue  = st.number_input("Monthly Revenue (₹)",  min_value=0,   key="rc_rev")
            user_expenses = st.number_input("Monthly Expenses (₹)", min_value=0,   key="rc_exp")
            user_growth   = st.number_input("Growth Rate (%)",       value=0.0,     key="rc_growth")
        with col2:
            user_runway   = st.number_input("Runway (Months)",       min_value=0,   key="rc_runway")
            user_team     = st.number_input("Team Size",             min_value=1,   key="rc_team")
            user_market   = st.selectbox("Market Risk", ["Low","Medium","High"],    key="rc_market")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("⚡  Calculate Risk", key="calc_btn"):
            burn   = max(0, user_expenses - user_revenue)
            rev_d  = abs(user_growth)*1000 if user_growth < 0 else 0
            run_r  = 50000 if user_runway<=3 else 20000 if user_runway<=6 else 0
            team_r = (10-user_team)*2000 if user_team<10 else 0
            mkt_r  = 40000 if user_market=="High" else 20000 if user_market=="Medium" else 0
            score  = max(0, burn+rev_d+run_r+team_r+mkt_r)
            if score>100000: level,pr,po = "High Risk",0.12,0.40
            elif score>50000: level,pr,po = "Moderate Risk",0.10,0.50
            else: level,pr,po = "Low Risk",0.08,0.60
            lc = "#FF2079" if "High" in level else "#FFE600" if "Moderate" in level else "#39FF14"
            st.markdown(f"""
            <div style="background:#0A0A12;border:1px solid {lc}33;
                        border-radius:6px;padding:28px;margin-top:16px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:2px;
                            background:linear-gradient(90deg,{lc},{lc}44);"></div>
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
                    <span style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;color:#F0F0FF;">
                        Calculation Result</span>
                    {risk_badge(level)}
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
                    {stat_card("Risk Score",f"{int(score):,}","composite",lc,"")}
                    {stat_card("Risk Level",level.replace(" Risk",""),"",lc,"")}
                    {stat_card("Premium",f"₹{int(user_funding*pr):,}",f"{int(pr*100)}% of funding","#BF5FFF","")}
                    {stat_card("Est. Payout",f"₹{int(user_funding*po):,}",f"{int(po*100)}% of funding","#39FF14","")}
                </div>
            </div>""", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            new_name     = st.text_input("Startup Name",            key="add_name")
            new_industry = st.text_input("Industry",                key="add_industry")
            new_revenue  = st.number_input("Monthly Revenue (₹)",  min_value=0, key="add_rev")
            new_expenses = st.number_input("Monthly Expenses (₹)", min_value=0, key="add_exp")
        with col2:
            new_growth   = st.number_input("Growth Rate (%)", value=0.0, key="add_growth")
            new_runway   = st.number_input("Runway (Months)", min_value=0, key="add_runway")
            new_team     = st.number_input("Team Size",       min_value=1, key="add_team")
            new_market   = st.selectbox("Market Risk", ["Low","Medium","High"], key="add_market")
            new_funding  = st.number_input("Total Funding (₹)", min_value=10000, value=1000000, key="add_fund")
        if st.button("➕  Add Startup", key="add_btn"):
            if new_name=="" or new_industry=="":
                st.warning("Fill all required fields.")
            elif new_name in df["Startup"].values:
                st.error("Startup already exists.")
            else:
                try:
                    # CHANGE 4: Plain table name (no SYSTEM. prefix), ? placeholders (not :1,:2,...)
                    cursor.execute(
                        """INSERT INTO STARTUPS
                           (STARTUP, INDUSTRY, MONTHLYREVENUE, MONTHLYEXPENSES,
                            GROWTHRATE, MONTHSRUNWAY, TEAMSIZE, MARKETRISK, FUNDING)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (new_name, new_industry, new_revenue, new_expenses,
                         new_growth, new_runway, new_team, new_market, new_funding)
                    )
                    conn.commit()
                    st.success("Startup added successfully.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    with tab3:
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        selected = st.selectbox("Select Startup", df["Startup"].unique(), key="mg_sel")
        sd = df[df["Startup"]==selected].iloc[0]
        col1, col2 = st.columns(2)
        with col1:
            ui = st.text_input("Industry", value=sd["Industry"],  key="upd_ind")
            ur = st.number_input("Monthly Revenue (₹)", value=int(sd["Monthly Revenue"]), key="upd_rev")
            ue = st.number_input("Monthly Expenses (₹)",value=int(sd["Monthly Expenses"]),key="upd_exp")
            ug = st.number_input("Growth Rate (%)", value=float(sd["Growth Rate (%)"]),   key="upd_g")
        with col2:
            un = st.number_input("Runway (Months)",value=int(sd["Months Runway"]),  key="upd_run")
            ut = st.number_input("Team Size",       value=int(sd["Team Size"]),      key="upd_team")
            um = st.selectbox("Market Risk",["Low","Medium","High"],
                              index=["Low","Medium","High"].index(sd["Market Risk"]),key="upd_mkt")
            uf = st.number_input("Total Funding (₹)",value=int(sd["Funding (₹)"]), key="upd_fund")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾  Update", key="upd_btn"):
                try:
                    # CHANGE 5: Plain table name, ? placeholders (not :1,:2,...)
                    cursor.execute(
                        """UPDATE STARTUPS SET
                           INDUSTRY=?, MONTHLYREVENUE=?, MONTHLYEXPENSES=?, GROWTHRATE=?,
                           MONTHSRUNWAY=?, TEAMSIZE=?, MARKETRISK=?, FUNDING=?
                           WHERE STARTUP=?""",
                        (ui, ur, ue, ug, un, ut, um, uf, selected)
                    )
                    conn.commit()
                    st.success("Updated.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        with col2:
            if st.button("🗑️  Delete", key="del_btn"):
                try:
                    # CHANGE 6: Plain table name, ? placeholder (not :1)
                    cursor.execute("DELETE FROM STARTUPS WHERE STARTUP=?", (selected,))
                    conn.commit()
                    st.success("Deleted.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    # Startup spotlight
    section_header("Startup Spotlight","🔍","#00FFFF")
    sel2 = st.selectbox("Select a Startup", df["Startup"].unique(), key="view_sel")
    if sel2:
        s  = df[df["Startup"]==sel2].iloc[0]
        lv = s["Risk Level"]
        lc = "#FF2079" if "High" in lv else "#FFE600" if "Moderate" in lv else "#39FF14"
        st.markdown(f"""
        <div style="background:#0A0A12;border:1px solid #1C1C30;
                    border-radius:6px;padding:24px;margin:8px 0;">
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
                <div>
                    <div style="font-family:'Orbitron',monospace;font-size:20px;font-weight:800;
                                color:#F0F0FF;">{sel2}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:11px;color:#7070A0;margin-top:2px;">
                        {s['Industry']} · {s['Market Risk']} Market Risk</div>
                </div>
                {risk_badge(lv)}
            </div>
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px;">
                {stat_card("Monthly Revenue","₹"+f"{int(s['Monthly Revenue']):,}","","#00FFFF","")}
                {stat_card("Risk Score",f"{int(s['Risk Score']):,}","composite",lc,"")}
                {stat_card("Runway",f"{int(s['Months Runway'])} mo","cash left","#BF5FFF","")}
                {stat_card("Premium","₹"+f"{int(s['Premium Amount']):,}","","#FFE600","")}
                {stat_card("Est. Payout","₹"+f"{int(s['Estimated Payout']):,}","","#39FF14","")}
            </div>
        </div>""", unsafe_allow_html=True)

    section_header("Portfolio Dataset","📋","#7070A0")
    st.dataframe(filtered_df.sort_values("Risk Score",ascending=False),
                 use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — AI RISK MENTOR
# ═══════════════════════════════════════════════════════════════
elif page == "🧠  AI Risk Mentor":

    page_hero("🧠","AI Risk Mentor",
              "Deep-dive into why a startup is at risk and receive AI-driven mitigation strategies.",
              "#BF5FFF")

    st.sidebar.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#3A3A5C;'
        'text-transform:uppercase;letter-spacing:0.18em;margin:20px 0 8px 0;">//Filters</div>',
        unsafe_allow_html=True)
    industry_filter = st.sidebar.multiselect("Industry", df["Industry"].unique(), default=df["Industry"].unique())

    mentor_mode = st.radio("Input Mode",["📂 Database","✍️ Manual"], label_visibility="collapsed")

    if mentor_mode == "📂 Database":
        sel = st.selectbox("Select Startup", df[df["Industry"].isin(industry_filter)]["Startup"].unique())
        row = df[df["Startup"]==sel].iloc[0]
        revenue,expenses,growth = row["Monthly Revenue"],row["Monthly Expenses"],row["Growth Rate (%)"]
        runway,team,market      = row["Months Runway"],row["Team Size"],row["Market Risk"]
        funding,industry        = row["Funding (₹)"],row["Industry"]
    else:
        col1, col2 = st.columns(2)
        with col1:
            revenue  = st.number_input("Monthly Revenue (₹)",  min_value=0, key="m_rev")
            expenses = st.number_input("Monthly Expenses (₹)", min_value=0, key="m_exp")
            growth   = st.number_input("Growth Rate (%)", value=0.0,        key="m_growth")
            funding  = st.number_input("Total Funding (₹)", min_value=10000, value=1000000, key="m_fund")
        with col2:
            runway   = st.number_input("Runway (Months)", min_value=0, key="m_run")
            team     = st.number_input("Team Size",       min_value=1, key="m_team")
            market   = st.selectbox("Market Risk",["Low","Medium","High"],  key="m_mkt")
            industry = st.text_input("Industry", value="Tech",              key="m_ind")

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("🔍  Analyse Risk", key="mentor_btn"):
        burn   = max(0, expenses-revenue)
        rev_d  = abs(growth)*1000 if growth<0 else 0
        run_r  = 50000 if runway<=3 else 20000 if runway<=6 else 0
        tm_r   = (10-team)*2000 if team<10 else 0
        mkt_r  = 40000 if market=="High" else 20000 if market=="Medium" else 0
        score  = max(0, burn+rev_d+run_r+tm_r+mkt_r)
        level  = risk_level(score)
        lc     = "#FF2079" if "High" in level else "#FFE600" if "Moderate" in level else "#39FF14"

        X_in   = prepare_features(revenue,expenses,growth,runway,team,market,funding,industry)
        proba  = clf.predict_proba(X_in)[0]
        risk_proba_dict = {c:round(p*100,1) for c,p in zip(le_risk.classes_,proba)}

        # Assessment header
        st.markdown(f"""
        <div style="background:#0A0A12;border:1px solid {lc}33;
                    border-radius:6px;padding:24px;margin:16px 0;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:2px;
                        background:linear-gradient(90deg,{lc},{lc}44);"></div>
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;">
                <span style="font-family:'Orbitron',monospace;font-size:18px;font-weight:700;color:#F0F0FF;">
                    Risk Assessment</span>
                {risk_badge(level)}
            </div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:14px;">
                {stat_card("Risk Score",f"{int(score):,}","composite",lc,"")}
                {stat_card("Burn Rate",f"₹{int(burn):,}","per month","#FF2079","")}
                {stat_card("Runway",f"{int(runway)} mo","remaining","#FFE600","")}
                {stat_card("Team Size",str(int(team)),"members","#BF5FFF","")}
            </div>
        </div>""", unsafe_allow_html=True)

        # ML probs
        section_header("ML Model Probabilities","🤖","#BF5FFF")
        ph = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
        for cls, pct in risk_proba_dict.items():
            c2 = "#FF2079" if "High" in cls else "#FFE600" if "Moderate" in cls else "#39FF14"
            ph += stat_card(cls.upper(), f"{pct}%","model confidence",c2,"")
        ph += "</div>"
        st.markdown(ph, unsafe_allow_html=True)

        # Risk factors
        section_header("Risk Factors Detected","⚠️","#FF2079")
        factors = []
        if burn>0:     factors.append((f"High Burn Rate — spending ₹{int(burn):,} more than earning monthly.","red"))
        if growth<0:   factors.append((f"Negative Growth — revenue declining at {abs(growth):.1f}% per period.","red"))
        if runway<=3:  factors.append(("Critical Runway — less than 3 months of cash remaining.","red"))
        elif runway<=6:factors.append(("Short Runway — only 3–6 months of runway left.","warn"))
        if team<5:     factors.append((f"Lean Team — only {int(team)} members, execution risk elevated.","warn"))
        if market=="High": factors.append(("High Market Volatility — turbulent market environment.","warn"))
        if factors:
            for txt,kind in factors: st.markdown(flag_card(txt,kind),unsafe_allow_html=True)
        else:
            st.markdown(flag_card("No major risk flags detected at this time.","green"),unsafe_allow_html=True)

        # Mitigations
        section_header("Mitigation Strategies","🛡️","#39FF14")
        mits = []
        if burn>0:
            mits.append("Cut non-essential expenses — audit SaaS, renegotiate vendors, reduce overheads.")
            mits.append("Activate new revenue streams — launch add-ons, upsell customers, explore B2B partnerships.")
        if growth<0:
            mits.append("Revise go-to-market strategy — run customer interviews, identify churn causes, pivot if needed.")
        if runway<=6:
            mits.append("Raise bridge funding immediately — approach angels, government grants, or venture debt.")
            mits.append("Extend runway — defer founder salaries, pause non-critical hiring, target profitability.")
        if team<5:
            mits.append("Strengthen the team — bring in a senior advisor or hire one key operations/sales person.")
        if market=="High":
            mits.append("Diversify market exposure — explore adjacent geographies or verticals.")
        if not mits:
            mits.append("Portfolio looks healthy. Monitor KPIs monthly and maintain a 12-month runway buffer.")
        for m in mits: st.markdown(flag_card(m,"green"),unsafe_allow_html=True)

        # Coverage
        section_header("FailSafe Coverage","💎","#00FFFF")
        pr = premium_rate(level); po = payout_rate(level)
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:14px;">
            {stat_card("Premium Rate",f"{int(pr*100)}%","of total funding","#BF5FFF","")}
            {stat_card("Premium Amount",f"₹{int(funding*pr):,}","annual payment","#00FFFF","")}
            {stat_card("Estimated Payout",f"₹{int(funding*po):,}","on verified failure","#39FF14","")}
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — FAILURE VERIFICATION AI
# ═══════════════════════════════════════════════════════════════
elif page == "🔍  Failure Verification AI":

    page_hero("🔍","Failure Verification AI",
              "Determine whether a startup failure claim is genuine or potentially fraudulent.",
              "#FF2079")

    st.sidebar.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:10px;color:#3A3A5C;'
        'text-transform:uppercase;letter-spacing:0.18em;margin:20px 0 8px 0;">//Filters</div>',
        unsafe_allow_html=True)
    industry_filter = st.sidebar.multiselect("Industry", df["Industry"].unique(), default=df["Industry"].unique())

    claim_mode = st.radio("Input Mode",["📂 Select from Database","✍️ Enter Claim Manually"],
                          label_visibility="collapsed")

    if claim_mode == "📂 Select from Database":
        sel = st.selectbox("Select Startup", df[df["Industry"].isin(industry_filter)]["Startup"].unique())
        row = df[df["Startup"]==sel].iloc[0]
        revenue,expenses,growth = row["Monthly Revenue"],row["Monthly Expenses"],row["Growth Rate (%)"]
        runway,team,market      = row["Months Runway"],row["Team Size"],row["Market Risk"]
        funding,industry        = row["Funding (₹)"],row["Industry"]
        claimed_payout          = row["Estimated Payout"]
    else:
        col1, col2 = st.columns(2)
        with col1:
            revenue  = st.number_input("Monthly Revenue (₹)",  min_value=0, key="fv_rev")
            expenses = st.number_input("Monthly Expenses (₹)", min_value=0, key="fv_exp")
            growth   = st.number_input("Growth Rate (%)", value=0.0,        key="fv_growth")
            funding  = st.number_input("Total Funding (₹)", min_value=10000, value=1000000, key="fv_fund")
        with col2:
            runway   = st.number_input("Runway (Months)", min_value=0, key="fv_run")
            team     = st.number_input("Team Size",       min_value=1, key="fv_team")
            market   = st.selectbox("Market Risk",["Low","Medium","High"],  key="fv_mkt")
            industry = st.text_input("Industry", value="Tech",              key="fv_ind")
        claimed_payout = st.number_input("Claimed Payout Amount (₹)", min_value=0, key="fv_payout")

    claim_description = st.text_area(
        "Describe the Failure Claim",
        placeholder="e.g. We lost all clients due to market crash. Revenue dropped to zero in 2 months...",
        height=110)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    if st.button("🔎  Verify Claim", key="verify_btn"):
        burn  = max(0, expenses-revenue)
        score = calculate_risk({"Monthly Expenses":expenses,"Monthly Revenue":revenue,
                                 "Growth Rate (%)":growth,"Months Runway":runway,
                                 "Team Size":team,"Market Risk":market})
        level = risk_level(score)
        X_in  = prepare_features(revenue,expenses,growth,runway,team,market,funding,industry)
        pred_lv = le_risk.inverse_transform([clf.predict(X_in)[0]])[0]
        proba   = clf.predict_proba(X_in)[0]
        risk_proba_dict = {c:round(p*100,1) for c,p in zip(le_risk.classes_,proba)}

        expected_payout = funding * payout_rate(level)
        payout_ratio    = claimed_payout / expected_payout if expected_payout > 0 else 0

        red_flags, green_flags, suspicious_score = [], [], 0

        if "High" not in level and claimed_payout > 0:
            red_flags.append("Startup risk is Low/Moderate yet a payout is being claimed."); suspicious_score+=30
        if payout_ratio > 1.5:
            red_flags.append(f"Claimed payout (₹{int(claimed_payout):,}) is {payout_ratio:.1f}× expected (₹{int(expected_payout):,})."); suspicious_score+=25
        elif payout_ratio <= 1.2:
            green_flags.append(f"Claimed payout is within acceptable range of expected (₹{int(expected_payout):,}).")
        if growth > 10 and claimed_payout > 0:
            red_flags.append(f"Positive growth ({growth:.1f}%) inconsistent with failure claim."); suspicious_score+=20
        if revenue > expenses and claimed_payout > 0:
            red_flags.append("Revenue exceeds expenses — startup appears profitable, not failing."); suspicious_score+=20
        if runway > 12 and claimed_payout > 0:
            red_flags.append(f"Runway of {int(runway)} months — no imminent cash-out risk."); suspicious_score+=15
        if team >= 10 and burn == 0:
            red_flags.append("Large team with zero burn rate contradicts failure narrative."); suspicious_score+=10
        if burn > 50000:  green_flags.append(f"High burn rate (₹{int(burn):,}/mo) consistent with distress.")
        if runway <= 3:   green_flags.append("Very short runway (≤3 months) supports genuine distress.")
        if growth < -10:  green_flags.append(f"Significant revenue decline ({growth:.1f}%) aligns with failure.")
        if market=="High":green_flags.append("High market risk corroborates external failure factors.")

        if claim_description:
            sus_kw  = ["sudden","overnight","forced","emergency","no choice","all clients left"]
            inc_kw  = ["growing","profitable","expansion","new clients","revenue increased"]
            dl = claim_description.lower()
            fi = [kw for kw in inc_kw if kw in dl]
            fs = [kw for kw in sus_kw if kw in dl]
            if fi: red_flags.append(f"Claim uses growth-positive language: {', '.join(fi)}."); suspicious_score+=15
            if fs: green_flags.append(f"Description uses distress language: {', '.join(fs)}.")

        suspicious_score = min(suspicious_score, 100)
        genuine_score    = 100 - suspicious_score
        if suspicious_score>=60:   verdict,vc,vi = "SUSPICIOUS CLAIM","#FF2079","❌"
        elif suspicious_score>=30: verdict,vc,vi = "NEEDS FURTHER REVIEW","#FFE600","⚠️"
        else:                      verdict,vc,vi = "CLAIM APPEARS GENUINE","#39FF14","✅"

        # Verdict banner
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{vc}15,{vc}08);border:1px solid {vc}44;
                    border-radius:6px;padding:28px;margin:16px 0;position:relative;overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;
                        background:linear-gradient(90deg,{vc},{vc}33);"></div>
            <div style="font-family:'Space Mono',monospace;font-size:10px;color:{vc};
                        letter-spacing:0.12em;text-transform:uppercase;margin-bottom:6px;">
                Verification Result</div>
            <div style="font-family:'Orbitron',monospace;font-size:28px;font-weight:800;
                        color:{vc};letter-spacing:-0.02em;margin-bottom:16px;">{vi} {verdict}</div>
            <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">
                {stat_card("Genuine Score",f"{genuine_score}%","","#39FF14","")}
                {stat_card("Suspicion Score",f"{suspicious_score}%","",vc,"")}
                {stat_card("ML Risk Level",pred_lv,"","#BF5FFF","")}
                {stat_card("Payout Ratio",f"{payout_ratio:.2f}×","vs expected","#FFE600","")}
            </div>
        </div>""", unsafe_allow_html=True)

        # Suspicion meter
        section_header("Suspicion Meter","🎯",vc)
        bc = "#FF2079" if suspicious_score>=60 else "#FFE600" if suspicious_score>=30 else "#39FF14"
        st.markdown(f"""
        <div style="background:#0A0A12;border:1px solid #1C1C30;
                    border-radius:6px;padding:20px 24px;">
            <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
                <span style="font-family:'Space Mono',monospace;font-size:11px;color:#7070A0;
                             text-transform:uppercase;letter-spacing:0.08em;">Suspicion Level</span>
                <span style="font-family:'Orbitron',monospace;font-size:20px;font-weight:800;
                             color:{bc};">{suspicious_score}%</span>
            </div>
            <div style="background:#0E0E1A;border-radius:99px;height:14px;
                        border:1px solid #1C1C30;overflow:hidden;">
                <div style="width:{suspicious_score}%;height:100%;
                            background:linear-gradient(90deg,{bc}BB,{bc});border-radius:99px;"></div>
            </div>
            <div style="display:flex;justify-content:space-between;margin-top:6px;">
                <span style="font-family:'Space Mono',monospace;font-size:9px;color:#3A3A5C;">GENUINE</span>
                <span style="font-family:'Space Mono',monospace;font-size:9px;color:#3A3A5C;">FRAUDULENT</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # ML probs
        section_header("ML Model Probabilities","🤖","#BF5FFF")
        ph = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;">'
        for cls,pct in risk_proba_dict.items():
            c2 = "#FF2079" if "High" in cls else "#FFE600" if "Moderate" in cls else "#39FF14"
            ph += stat_card(cls.upper(), f"{pct}%","confidence",c2,"")
        ph += "</div>"
        st.markdown(ph, unsafe_allow_html=True)

        # Flags side by side
        col1, col2 = st.columns(2)
        with col1:
            section_header("Red Flags","🚩","#FF2079")
            if red_flags:
                for f in red_flags: st.markdown(flag_card(f,"red"),unsafe_allow_html=True)
            else:
                st.markdown(flag_card("No red flags detected.","green"),unsafe_allow_html=True)
        with col2:
            section_header("Genuine Indicators","✅","#39FF14")
            if green_flags:
                for f in green_flags: st.markdown(flag_card(f,"green"),unsafe_allow_html=True)
            else:
                st.markdown(flag_card("No genuine indicators found.","warn"),unsafe_allow_html=True)

        # Payout decision
        section_header("Payout Decision","💰","#00FFFF")
        if suspicious_score>=60:
            rec = f"REJECT payout claim. Suspicious indicators dominate. Initiate full financial audit before any disbursement."
            rk  = "red"
        elif suspicious_score>=30:
            rec = f"CONDITIONAL APPROVAL. Request bank statements, client termination letters, and audited financials."
            rk  = "warn"
        else:
            rec = f"APPROVE payout of ₹{int(expected_payout):,}. Claim is consistent with financial data and risk profile."
            rk  = "green"
        st.markdown(flag_card(rec, rk), unsafe_allow_html=True)

        # Summary table
        section_header("Verification Summary","📋","#7070A0")
        summary = pd.DataFrame({
            "Metric":["Risk Score","Risk Level","Burn Rate","Runway",
                      "Claimed Payout","Expected Payout","Payout Ratio","Suspicion Score","Verdict"],
            "Value": [int(score),level,f"₹{int(burn):,}/mo",f"{int(runway)} months",
                      f"₹{int(claimed_payout):,}",f"₹{int(expected_payout):,}",
                      f"{payout_ratio:.2f}×",f"{suspicious_score}%",verdict]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)