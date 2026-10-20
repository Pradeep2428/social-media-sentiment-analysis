import os
import re
import json
import html
import importlib

import joblib
import pandas as pd
import altair as alt
import streamlit as st

import smart_layer as sl
import behaviour_tests as bt
importlib.reload(sl)  # pick up edits without restarting Streamlit
importlib.reload(bt)

st.set_page_config(
    page_title="SentiLens",
    page_icon="◐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Paths, model, helpers
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
REPORT_PATH = os.path.join(BASE_DIR, "model_report.json")
CLASS_ORDER = ["Positive", "Neutral", "Negative"]
TONE = {  # text colour, highlighter tint
    "Positive": ("#13795B", "#BDE8D6"),
    "Neutral": ("#5B6472", "#DDE1E6"),
    "Negative": ("#C2364A", "#F6CBD1"),
    "Mixed": ("#6A4BC4", "#DCD3F5"),
}


def find_data(name):
    for folder in ("archive", "dataset"):
        path = os.path.join(BASE_DIR, "..", folder, name)
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    return joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)


def load_report():
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH) as f:
            return json.load(f)
    return None


model, vectorizer = load_artifacts()
report = load_report()

_lab = st.query_params.get("lab")
if _lab:  # experiment runner used to produce the evaluation files; not part of the normal app
    import lab_experiments
    importlib.reload(lab_experiments)
    _train = find_data("twitter_training.csv")
    if _lab in ("real_val", "real_test"):
        lab_experiments.real_check(sl, clean_text, _train,
                                   os.path.join(BASE_DIR, f"{_lab}.json"), split=_lab.split("_")[1])
    elif _lab in ("hinglish_dev", "hinglish_test"):
        lab_experiments.hinglish_phase(sl, model, vectorizer, clean_text, BASE_DIR, phase=_lab.split("_")[1])
    st.stop()
esc = html.escape


def H(markup):
    # Markdown ends an HTML block at a blank line, so collapse blank lines first
    st.markdown(re.sub(r"\n\s*\n", "\n", markup), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Visual system
# ---------------------------------------------------------------------------

H("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&family=Schibsted+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
  --paper: #E8EEEA;
  --sheet: #F7FAF8;
  --ink: #17221F;
  --muted: #5A6A64;
  --line: #C9D5CF;
  --cobalt: #2C46C9;
  --cobalt-deep: #1F34A0;
  --pos: #13795B; --pos-hl: #BDE8D6;
  --neg: #C2364A; --neg-hl: #F6CBD1;
  --neu: #5B6472; --neu-hl: #DDE1E6;
  --mix: #6A4BC4; --mix-hl: #DCD3F5;
  --sans: "Schibsted Grotesk", system-ui, -apple-system, "Segoe UI", sans-serif;
  --serif: "Newsreader", Georgia, "Times New Roman", serif;
}

/* ---------- page ---------- */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--paper) !important; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stAppDeployButton"], [data-testid="stSidebarCollapsedControl"], section[data-testid="stSidebar"] { display: none !important; }
.block-container { max-width: 1120px; padding-top: 2.2rem !important; padding-bottom: 4rem !important; }
.stApp { font-family: var(--sans); color: var(--ink); }
.stApp p, .stApp li, .stApp label, .stApp td, .stApp th, .stApp input, .stApp button,
.stApp .mast, .stApp .sheet, .stApp .stats, .stApp .score, .stApp .acc, .stApp .cm, .stApp .push, .stApp .legend {
  font-family: var(--sans); color: var(--ink);
}
.stApp p, .stApp li { font-size: 1rem; line-height: 1.55; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4 { font-family: var(--sans) !important; color: var(--ink) !important; letter-spacing: -0.01em; }
.stApp h3 { font-size: 1.25rem !important; font-weight: 600 !important; margin: 1.6rem 0 .5rem !important; padding: 0 !important; }
.stApp code { color: var(--ink); background: rgba(23,34,31,.07); font-size: .88em; }
.stApp a { color: var(--cobalt); }

/* ---------- masthead ---------- */
.mast { display: flex; align-items: flex-end; justify-content: space-between; gap: 2rem;
        padding-bottom: 1.4rem; border-bottom: 1px solid var(--line); margin-bottom: .4rem; flex-wrap: wrap; }
.mast .brand { display: flex; align-items: center; gap: .9rem; }
.mast .wordmark { font-size: 2.45rem; font-weight: 700; letter-spacing: -0.035em; line-height: 1; }
.mast .tag { color: var(--muted); font-size: 1.05rem; margin-top: .45rem; max-width: 34rem; }
.mast .fact { text-align: right; color: var(--muted); font-size: .9rem; line-height: 1.45; }
.mast .fact b { color: var(--ink); font-weight: 600; font-size: 1.35rem; display: block; }

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] { gap: 2.2rem !important; border-bottom: 1px solid var(--line); background: transparent !important; }
button[data-baseweb="tab"] { background: transparent !important; padding: .9rem 0 .7rem !important; margin-right: 1.4rem !important; }
button[data-baseweb="tab"] p { font-size: 1rem !important; font-weight: 500; color: var(--muted) !important; }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--ink) !important; font-weight: 600; }
[data-baseweb="tab-highlight"] { background: var(--cobalt) !important; height: 3px !important; }
[data-baseweb="tab-border"] { display: none; }
[data-testid="stTabs"] [data-baseweb="tab-panel"] { padding-top: 1.6rem; }
[data-testid="stTabs"] [role="tablist"] { gap: 2.1rem !important; border-bottom: 1px solid var(--line); }
[data-testid="stTab"] p { font-size: 1rem !important; font-weight: 500 !important; color: var(--muted) !important; }
[data-testid="stTab"][aria-selected="true"] p { color: var(--ink) !important; font-weight: 600 !important; }
[data-testid="stTab"]:focus-visible { outline: 3px solid rgba(44,70,201,.45); outline-offset: 4px; border-radius: 4px; }
[data-testid="stTabs"] [role="tabpanel"] { padding-top: 1.4rem; }

/* ---------- inputs ---------- */
[data-testid="stTextArea"] textarea, [data-testid="stTextInput"] input {
  background: var(--sheet) !important; color: var(--ink) !important; border-radius: 10px !important;
  font-family: var(--serif) !important; font-size: 1.2rem !important; line-height: 1.5 !important;
  caret-color: var(--cobalt);
}
[data-testid="stTextInput"] input { font-size: 1.05rem !important; }
[data-testid="stTextArea"] > div > div, [data-testid="stTextInput"] > div > div,
[data-baseweb="textarea"], [data-baseweb="input"], [data-baseweb="base-input"] {
  background: var(--sheet) !important; border-color: var(--line) !important; border-radius: 10px !important;
}
[data-baseweb="textarea"]:focus-within, [data-baseweb="input"]:focus-within { border-color: var(--cobalt) !important; box-shadow: 0 0 0 3px rgba(44,70,201,.18) !important; }
textarea::placeholder, input::placeholder { color: #8A9893 !important; font-style: italic; }
[data-testid="stWidgetLabel"] p { font-weight: 600 !important; font-size: .95rem !important; color: var(--ink) !important; }

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"] {
  background: var(--cobalt) !important; border: none !important; border-radius: 999px !important;
  padding: .55rem 1.5rem !important;
}
[data-testid="stBaseButton-primary"] p { color: #fff !important; font-weight: 600 !important; }
[data-testid="stBaseButton-primary"]:hover { background: var(--cobalt-deep) !important; }
[data-testid="stBaseButton-secondary"] {
  background: transparent !important; border: 1px solid var(--line) !important; border-radius: 999px !important;
  padding: .3rem .9rem !important; min-height: 0 !important;
}
[data-testid="stBaseButton-secondary"] p { color: var(--ink) !important; font-size: .9rem !important; }
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--cobalt) !important; }
[data-testid="stBaseButton-secondary"]:hover p { color: var(--cobalt) !important; }
button:focus-visible { outline: 3px solid rgba(44,70,201,.45) !important; outline-offset: 2px; }
[data-testid="stDownloadButton"] button { border-radius: 999px !important; }

/* ---------- misc widgets ---------- */
[data-testid="stFileUploader"] section { background: var(--sheet) !important; border: 1.5px dashed var(--line) !important; border-radius: 12px !important; }
[data-testid="stFileUploader"] section * { color: var(--muted) !important; }
[data-testid="stFileUploader"] button { background: var(--sheet) !important; border: 1px solid var(--line) !important; }
[data-testid="stExpander"] details { background: var(--sheet) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; }
[data-testid="stExpander"] summary * { color: var(--ink) !important; }
[data-testid="stAlert"] > div { background: var(--sheet) !important; border: 1px solid var(--line) !important; border-radius: 12px !important; }
[data-baseweb="select"] > div { background: var(--sheet) !important; border-color: var(--line) !important; }
[data-testid="stSlider"] [role="slider"] { background: var(--cobalt) !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color: var(--muted) !important; }
[data-testid="stVegaLiteChart"] { background: transparent !important; }

/* ---------- components ---------- */
.lede { color: var(--muted); font-size: 1.02rem; max-width: 42rem; margin: 0 0 1rem; }
.sheet { background: var(--sheet); border: 1px solid var(--line); border-radius: 18px; padding: 1.6rem 1.8rem; }
.subtle { color: var(--muted); font-size: .9rem; }

.specimen { font-family: var(--serif); font-size: 2.05rem; line-height: 1.55; color: var(--ink); margin: .2rem 0 1rem; }
.specimen span.w { padding: 0 .08em; border-radius: 2px; }
.specimen .orig { display: block; font-size: 1.05rem; font-style: italic; color: var(--muted); margin-bottom: .6rem; line-height: 1.5; }
.legend { display: flex; gap: 1.2rem; flex-wrap: wrap; color: var(--muted); font-size: .85rem; }
.legend i { display: inline-block; width: 1.4rem; height: .55rem; border-radius: 2px; margin-right: .35rem; vertical-align: middle; }

.verdict-label { font-size: 3rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; margin: .2rem 0 .3rem; }
.verdict-lean { font-size: 1.05rem; font-weight: 500; color: var(--muted); }
.probbar { display: flex; height: 12px; border-radius: 999px; overflow: hidden; margin: .4rem 0 .5rem; background: var(--neu-hl); }
.probkey { display: flex; justify-content: space-between; font-size: .85rem; color: var(--muted); }
.probkey b { font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }
.compare { margin-top: 1.3rem; padding-top: 1rem; border-top: 1px solid var(--line); font-size: .95rem; color: var(--muted); }
.compare b { color: var(--ink); font-weight: 600; }
.flags { margin-top: .9rem; display: flex; flex-wrap: wrap; gap: .4rem; }
.flag { font-size: .82rem; padding: .18rem .65rem; border-radius: 999px; font-weight: 500; }

.reasons { margin: 0; padding-left: 1.1rem; }
.reasons li { margin: .35rem 0; }
.reasons .ev { color: var(--muted); font-size: .9rem; }

.push { display: grid; grid-template-columns: 9.5rem 1fr; gap: .45rem .9rem; align-items: center; font-size: .92rem; }
.push .term { text-align: right; font-family: var(--serif); font-size: 1.05rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.push .track { position: relative; height: 14px; }
.push .track::before { content: ""; position: absolute; left: 50%; top: -4px; bottom: -4px; width: 1px; background: var(--line); }
.push .bar { position: absolute; top: 0; height: 14px; border-radius: 3px; }
.push-axis { display: grid; grid-template-columns: 9.5rem 1fr; gap: .9rem; font-size: .8rem; color: var(--muted); margin-top: .5rem; }
.push-axis div:last-child { display: flex; justify-content: space-between; }

.clauses { width: 100%; border-collapse: collapse; }
.clauses td { padding: .75rem .2rem; border-top: 1px solid var(--line); vertical-align: top; }
.clauses tr:first-child td { border-top: none; }
.clauses .asp { font-weight: 600; width: 7rem; text-transform: capitalize; }
.clauses .cl { font-family: var(--serif); font-size: 1.08rem; }
.clauses .lab { text-align: right; white-space: nowrap; font-weight: 600; }
.clauses .by { display: block; font-weight: 400; color: var(--muted); font-size: .8rem; }

.score { display: flex; gap: 3rem; align-items: baseline; flex-wrap: wrap; margin: .4rem 0 1.4rem; }
.score .n { font-size: 3.4rem; font-weight: 700; letter-spacing: -0.04em; line-height: 1; }
.score .n small { font-size: 1.4rem; color: var(--muted); font-weight: 500; letter-spacing: 0; }
.score .l { color: var(--muted); font-size: .95rem; margin-top: .35rem; }

table.cases { width: 100%; border-collapse: collapse; font-size: .93rem; }
table.cases th { text-align: left; font-weight: 600; color: var(--muted); font-size: .85rem; padding: .5rem .5rem; border-bottom: 1.5px solid var(--ink); }
table.cases td { padding: .7rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
table.cases td.post { font-family: var(--serif); font-size: 1.04rem; max-width: 26rem; }
table.cases .ok { color: var(--pos); font-weight: 700; }
table.cases .no { color: var(--neg); font-weight: 700; }
table.cases .kind { color: var(--muted); white-space: nowrap; }

table.plain { width: 100%; max-width: 44rem; border-collapse: collapse; font-size: .93rem; font-variant-numeric: tabular-nums; }
table.plain th { text-align: left; color: var(--muted); font-weight: 600; font-size: .85rem; padding: .45rem .5rem; border-bottom: 1.5px solid var(--ink); }
table.plain td { padding: .5rem .5rem; border-bottom: 1px solid var(--line); }
table.plain td.num, table.plain th.num { text-align: right; }

.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem; margin: .6rem 0 1.2rem; }
.stat .v { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; }
.stat .k { color: var(--muted); font-size: .9rem; }

.acc { display: grid; grid-template-columns: 15rem 1fr 4.5rem; gap: .9rem; align-items: center; margin: .7rem 0; }
.acc .k { font-size: .95rem; }
.acc .k small { display: block; color: var(--muted); font-size: .82rem; }
.acc .track { background: rgba(23,34,31,.06); height: 22px; border-radius: 4px; position: relative; overflow: hidden; }
.acc .fill { height: 100%; border-radius: 4px; }
.acc .v { font-weight: 700; font-size: 1.15rem; text-align: right; }

.cm { display: grid; grid-template-columns: 7.5rem repeat(3, 1fr); gap: 4px; font-variant-numeric: tabular-nums; max-width: 34rem; }
.cm div { padding: .65rem .5rem; text-align: center; border-radius: 4px; font-size: .95rem; }
.cm .h { color: var(--muted); font-size: .82rem; background: none; }
.cm .rh { text-align: right; padding-right: .7rem; }

table.caps { width: 100%; border-collapse: collapse; font-size: .93rem; }
table.caps th { text-align: left; font-weight: 600; color: var(--muted); font-size: .85rem; padding: .5rem .5rem; border-bottom: 1.5px solid var(--ink); }
table.caps td { padding: .8rem .5rem; border-bottom: 1px solid var(--line); vertical-align: top; }
table.caps .cap b { display: block; font-weight: 600; }
table.caps .cap span { color: var(--muted); font-size: .85rem; }
table.caps .ex { font-family: var(--serif); font-size: 1rem; max-width: 20rem; }
table.caps .rate { width: 9.5rem; }
.rbar { height: 8px; background: rgba(23,34,31,.08); border-radius: 999px; margin-top: .35rem; overflow: hidden; }
.rbar div { height: 100%; border-radius: 999px; }
.rnum { font-weight: 700; font-size: 1.05rem; }
.kind { font-size: .75rem; color: var(--muted); border: 1px solid var(--line); border-radius: 4px; padding: 0 .3rem; margin-left: .3rem; }
.stApp table.caps, .stApp table.caps td, .stApp table.caps th { border-left: none !important; border-right: none !important; background: transparent !important; }
.stApp table.caps { border: none !important; }
.stApp table.caps .ex { font-family: var(--serif) !important; }
.finding { padding: .1rem 0; margin: .6rem 0; max-width: 44rem; }
.finding b { font-weight: 600; }
.steps { counter-reset: s; list-style: none; padding: 0; margin: 0; }
.steps li { counter-increment: s; display: grid; grid-template-columns: 2rem 1fr; padding: .55rem 0; border-top: 1px solid var(--line); }
.steps li::before { content: counter(s); font-weight: 700; color: var(--cobalt); font-variant-numeric: tabular-nums; }

.stApp .specimen, .stApp .specimen span, .stApp .push .term, .stApp .clauses .cl,
.stApp table.cases td.post { font-family: var(--serif) !important; }
.stApp table.clauses, .stApp table.cases, .stApp table.plain { border: none !important; }
.stApp table.clauses td, .stApp table.cases td, .stApp table.cases th, .stApp table.plain td, .stApp table.plain th {
  border-left: none !important; border-right: none !important; background: transparent !important; }
.stApp table.clauses td { border-bottom: none !important; }
.stApp .specimen .orig { font-family: var(--serif) !important; }
.stApp .specimen span.w { color: var(--ink); }
.stApp .verdict-label { color: inherit; }
/* ---------- browser surfaces ---------- */
::selection { background: rgba(44,70,201,.20); color: var(--ink); }
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #BFCBC5; border-radius: 999px; border: 2px solid var(--paper); }
::-webkit-scrollbar-track { background: transparent; }
.stApp a { text-underline-offset: 3px; }

/* ---------- signature moment: the highlighter sweep ---------- */
.stApp .specimen span.m {
  background-repeat: no-repeat; background-position: 0 0; background-size: 100% 100%;
  animation: sweep 460ms cubic-bezier(.16,1,.3,1) both; animation-delay: var(--d, 0ms);
}
@keyframes sweep { from { background-size: 0% 100%; } to { background-size: 100% 100%; } }
.sw { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; margin-right: .35rem; vertical-align: -1px; }

/* ---------- verdict ---------- */
.verdict-label .pre { font-size: 1rem; font-weight: 500; color: var(--muted); letter-spacing: 0; margin-right: .4rem; }
.correction { margin: .1rem 0 .2rem; font-size: 1.05rem; color: var(--muted); }
.correction s { text-decoration-color: var(--neg); text-decoration-thickness: 2.5px; color: var(--ink); font-weight: 600; margin-right: .35rem; }
.demo-note { font-family: var(--serif); font-size: 1.15rem; line-height: 1.55; max-width: 40rem; }
.demo-note b { font-family: var(--sans); font-weight: 600; }
.stApp .demo-note { font-family: var(--serif) !important; font-size: 1.15rem; color: var(--ink); }
.stApp .demo-note b { font-family: var(--sans) !important; }

/* ---------- headline sentences (serif, the specimen voice) ---------- */
.claim { font-family: var(--serif) !important; font-size: 1.95rem; line-height: 1.3; letter-spacing: -0.01em;
         max-width: 44rem; margin: .4rem 0 1rem; color: var(--ink); text-wrap: balance; }
.claim .c-smart { color: var(--cobalt); font-weight: 500; }
.claim .c-base { color: var(--neu); font-weight: 500; }
.stApp .claim, .stApp .claim span { font-family: var(--serif) !important; }
.stApp p.claim { font-size: 1.95rem !important; line-height: 1.3 !important; }
.dumb .lab.t { margin-top: -19px; transform: translate(-50%, -50%); }

/* ---------- dumbbell chart (stress test) ---------- */
.dumb { display: grid; grid-template-columns: minmax(11rem, 16rem) 1fr; column-gap: 1.6rem; }
.dumb .dname { padding: .75rem 0; border-top: 1px solid var(--line); }
.dumb .dname b { display: block; font-weight: 600; }
.dumb .dname span { display: block; color: var(--muted); font-size: .84rem; line-height: 1.35; }
.dumb .dname em { display: block; font-family: var(--serif); font-style: normal; font-size: .95rem; color: var(--ink); margin-top: .25rem; }
.dumb .dtrack { position: relative; border-top: 1px solid var(--line); min-height: 4.2rem;
  background-image: linear-gradient(to right, rgba(23,34,31,.07) 1px, transparent 1px);
  background-size: 25% 100%; background-position: -0.5px 0; }
.dumb .dline { position: absolute; top: 50%; height: 3px; margin-top: -1.5px; border-radius: 2px; }
.dumb .dot { position: absolute; top: 50%; width: 14px; height: 14px; margin: -7px 0 0 -7px; border-radius: 50%; }
.dumb .dot.base { background: var(--paper); border: 2.5px solid var(--neu); }
.dumb .dot.smart { background: var(--cobalt); box-shadow: 0 1px 3px rgba(31,52,160,.35); }
.dumb .lab { position: absolute; top: 50%; transform: translateY(-50%); font-size: .88rem; font-weight: 700;
  font-variant-numeric: tabular-nums; white-space: nowrap; }
.dumb .lab.l { margin-left: -14px; transform: translate(-100%, -50%); }
.dumb .lab.r { margin-left: 14px; }
.dumb .note { position: absolute; top: 50%; transform: translateY(-50%); font-size: .82rem; color: var(--muted); }
.dumb .axis { grid-column: 2; display: flex; justify-content: space-between; font-size: .78rem; color: var(--muted);
  border-top: 1px solid var(--ink); padding-top: .3rem; font-variant-numeric: tabular-nums; }
.dlegend { display: flex; gap: 1.4rem; font-size: .88rem; color: var(--muted); margin: .2rem 0 .6rem; flex-wrap: wrap; }
.dlegend i { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: .4rem; vertical-align: -1px; }

/* ---------- leakage bar ---------- */
.leak { margin: 1.2rem 0 .4rem; }
.leak .bar { position: relative; height: 46px; border-radius: 6px; background: rgba(23,34,31,.06); overflow: visible; }
.leak .solid { position: absolute; left: 0; top: 0; bottom: 0; background: var(--neu); border-radius: 6px 0 0 6px; }
.leak .hatch { position: absolute; top: 0; bottom: 0; border-radius: 0 6px 6px 0;
  background: repeating-linear-gradient(-45deg, rgba(194,54,74,.55) 0 6px, rgba(194,54,74,.12) 6px 12px); }
.leak .mark { position: absolute; top: -8px; bottom: -8px; width: 3px; margin-left: -1.5px; background: var(--cobalt); border-radius: 2px; }
.leak .keys { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: .9rem; font-size: .9rem; color: var(--muted); }
.leak .keys b { display: block; font-size: 1.35rem; color: var(--ink); font-weight: 700; font-variant-numeric: tabular-nums; }
.leak .ticks { position: relative; height: 1.2rem; font-size: .75rem; color: var(--muted); margin-top: .25rem; }
.leak .ticks span { position: absolute; transform: translateX(-50%); font-variant-numeric: tabular-nums; }

/* ---------- ladder (Hinglish) ---------- */
.ladder { position: relative; height: 9.5rem; margin: 1rem 0 .4rem; }
.ladder .rail { position: absolute; left: 0; right: 0; top: 50%; height: 2px; background: var(--ink); opacity: .25; }
.ladder .gain { position: absolute; top: 50%; height: 4px; margin-top: -1px; background: var(--cobalt); border-radius: 2px; }
.ladder .pt { position: absolute; top: 50%; width: 16px; height: 16px; margin: -7px 0 0 -8px; border-radius: 50%; }
.ladder .tag { position: absolute; transform: translateX(-50%); text-align: center; font-size: .82rem; color: var(--muted); width: 9.5rem; line-height: 1.3; }
.ladder .tag b { display: block; font-size: 1.15rem; color: var(--ink); font-variant-numeric: tabular-nums; }
.ladder .tag.up { bottom: calc(50% + 14px); }
.ladder .tag.down { top: calc(50% + 14px); }
.ladder .tick { position: absolute; top: calc(50% + 5px); font-size: .72rem; color: var(--muted); transform: translateX(-50%); }

/* ---------- fixed vs broke (real tweets) ---------- */
.fb { display: grid; grid-template-columns: 1fr 13rem 1fr; gap: .35rem .8rem; align-items: center; font-size: .9rem; max-width: 46rem; }
.fb .name { text-align: center; }
.fb .l, .fb .r { position: relative; height: 16px; }
.fb .l div { position: absolute; right: 0; top: 0; bottom: 0; background: var(--neg); border-radius: 3px 0 0 3px; }
.fb .r div { position: absolute; left: 0; top: 0; bottom: 0; background: var(--pos); border-radius: 0 3px 3px 0; }
.fb .l span, .fb .r span { position: absolute; top: 50%; transform: translateY(-50%); font-size: .8rem; font-weight: 700; font-variant-numeric: tabular-nums; }
.fb .hd { font-size: .82rem; color: var(--muted); }
.fb .hd.l { text-align: right; } .fb .hd.r { text-align: left; }

@media (prefers-reduced-motion: reduce) {
  .stApp .specimen span.m { animation: none !important; background-size: 100% 100% !important; }
}
@media (max-width: 760px) {
  .stApp p.claim { font-size: 1.45rem !important; }
  .dumb { grid-template-columns: 1fr; }
  .dumb .dname { border-top: 1px solid var(--line); padding-bottom: .1rem; }
  .dumb .dtrack { border-top: none; }
  .dumb .axis { grid-column: 1; }
  .leak .keys { grid-template-columns: 1fr; }
  .fb { grid-template-columns: 1fr 7rem 1fr; }
  .ladder { height: 11rem; }
  .ladder .tag { width: 4.3rem; font-size: .68rem; }
  .ladder .tag b { font-size: .95rem; }
  .mast .fact { text-align: left; }
  .specimen { font-size: 1.5rem; }
  .verdict-label { font-size: 2.4rem; }
  .push, .push-axis { grid-template-columns: 6.5rem 1fr; }
  .acc { grid-template-columns: 1fr 4rem; }
  .acc .track { grid-column: 1 / -1; grid-row: 2; }
  .sheet { padding: 1.2rem; }
  table.cases td.post { max-width: none; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
</style>
""")

LENS_SVG = """<svg width="46" height="46" viewBox="0 0 46 46" aria-hidden="true">
<circle cx="20" cy="20" r="15" fill="none" stroke="#17221F" stroke-width="3.2"/>
<path d="M20 5 A15 15 0 0 1 20 35 Z" fill="#2C46C9"/>
<path d="M31 31 L42 42" stroke="#17221F" stroke-width="4" stroke-linecap="round"/></svg>"""

fact = ""
if report:
    fact = (f'<div class="fact"><b>{report["improved"]["accuracy"]*100:.1f}%</b>'
            f'accuracy on posts the model<br>has never seen</div>')
H(f"""<div class="mast"><div><div class="brand">{LENS_SVG}<div class="wordmark">SentiLens</div></div>
<div class="tag">Reads the tone of a social-media post and shows which words decided it, including sarcasm,
quiet complaints, mixed opinions and Hinglish.</div></div>{fact}</div>""")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def tone(label):
    return TONE.get(label, TONE["Neutral"])


def specimen_html(res):
    scores = res["token_scores"] or {}
    max_abs = max((abs(v) for v in scores.values()), default=1.0) or 1.0
    words, marked = [], 0
    toks = res["cleaned"].split()
    step = min(60, 900 // max(1, sum(1 for w in toks if scores.get(w))))  # whole sweep stays under ~1s
    for w in toks:
        s = scores.get(w, 0.0)
        if s:
            strength = 0.25 + 0.75 * min(abs(s) / max_abs, 1.0)
            rgb = "19,121,91" if s > 0 else "194,54,74"
            # a highlighter stroke over the lower part of the word, swept in left to right
            style = (f"background-image: linear-gradient(transparent 42%, rgba({rgb},{strength*0.45:.2f}) 42%);"
                     f"--d:{120 + marked * step}ms;")
            words.append(f'<span class="w m" style="{style}" title="{s:+.2f}">{esc(w)}</span>')
            marked += 1
        else:
            words.append(f'<span class="w">{esc(w)}</span>')
    orig = ""
    if res["changes"]:
        orig = f'<span class="orig">You wrote: “{esc(res["raw"])}”</span>'
    return f'<div class="specimen">{orig}{" ".join(words)}</div>'


def prob_bar(probs):
    segs, keys = "", ""
    for lab in CLASS_ORDER:
        p = probs.get(lab, 0.0)
        segs += f'<div style="width:{p*100:.2f}%;background:{tone(lab)[0]}"></div>'
        keys += f'<span><i class="sw" style="background:{tone(lab)[0]}"></i>{lab} <b>{p*100:.0f}%</b></span>'
    return f'<div class="probbar">{segs}</div><div class="probkey">{keys}</div>'


def flags_html(res):
    items = []
    if res["sarcasm"]: items.append(("Sarcasm", "Negative"))
    if res["strong_hedge"]: items.append(("Quiet complaint", "Negative"))
    if res["weak_hedge"]: items.append(("Hedging", "Neutral"))
    if res.get("litotes"): items.append(("“Not bad” means good", "Positive"))
    if res["mixed"] and res["final"] == "Mixed": items.append(("Mixed opinion", "Mixed"))
    if res["changes"]: items.append(("Hinglish or emoji translated", "Positive"))
    if res.get("masked"): items.append(("Brand names ignored", "Neutral"))
    if res["uncertain"]: items.append(("Low confidence", "Neutral"))
    if not items:
        return ""
    return '<div class="flags">' + "".join(
        f'<span class="flag" style="background:{tone(t)[1]};color:{tone(t)[0]}">{esc(n)}</span>' for n, t in items
    ) + "</div>"


def verdict_html(res):
    final = res["final"]
    colour = tone(final)[0]
    lean = f'<div class="verdict-lean">leans {res["lean"].lower()}</div>' if res["lean"] and final == "Mixed" else ""
    changed = final != res["base_label"]
    correction = (f'<div class="correction"><s>{res["base_label"]}</s>model alone, {res["base_conf"]*100:.0f}%</div>'
                  if changed else "")
    if changed:
        compare = "SentiLens overruled the model after checking the context. The reasons are listed below."
    else:
        compare = f'The model alone agrees: <b>{res["base_label"]}</b> ({res["base_conf"]*100:.0f}%).'
    return f"""<div class="sheet">
{correction}<div class="verdict-label" style="color:{colour}"><span class="pre">reads as</span>{final}</div>{lean}
<div class="subtle" style="margin-top:1.1rem">Model’s probability estimate</div>
{prob_bar(res["model_probs"])}
{flags_html(res)}
<div class="compare">{compare}</div></div>"""


def push_html(features):
    if not features:
        return ""
    m = max(abs(v) for _, v in features) or 1.0
    rows = ""
    for term, v in features:
        width = abs(v) / m * 50
        left = 50 if v > 0 else 50 - width
        col = TONE["Positive"][0] if v > 0 else TONE["Negative"][0]
        rows += (f'<div class="term" title="{esc(term)}">{esc(term)}</div><div class="track">'
                 f'<div class="bar" style="left:{left:.2f}%;width:{width:.2f}%;background:{col}"></div></div>')
    return (f'<div class="push">{rows}</div><div class="push-axis"><div></div>'
            f'<div><span>toward negative</span><span>toward positive</span></div></div>')


def clauses_html(aspects):
    rows = ""
    for a in aspects:
        col = tone(a["Sentiment"])[0]
        by = "decided by opinion words" if a.get("Decided by") == "opinion words" else f'{a["Confidence"]*100:.0f}% model probability'
        rows += (f'<tr><td class="asp">{esc(a["Aspect"])}</td><td class="cl">{esc(a["Clause"])}</td>'
                 f'<td class="lab" style="color:{col}">{a["Sentiment"]}<span class="by">{by}</span></td></tr>')
    return f'<table class="clauses">{rows}</table>'


def html_table(df, num_cols=()):
    head = "".join(f'<th class="{"num" if c in num_cols else ""}">{esc(str(c))}</th>' for c in df.columns)
    body = ""
    for _, r in df.iterrows():
        body += "<tr>" + "".join(
            f'<td class="{"num" if c in num_cols else ""}">{esc(str(r[c]))}</td>' for c in df.columns) + "</tr>"
    return f'<table class="plain"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def stats_html(items):
    return '<div class="stats">' + "".join(
        f'<div class="stat"><div class="v">{v}</div><div class="k">{esc(k)}</div></div>' for k, v in items) + "</div>"


def confusion_html(cm, labels):
    flat = [v for row in cm for v in row]
    peak = max(flat) or 1
    cells = '<div class="h"></div>' + "".join(f'<div class="h">Predicted {l.lower()}</div>' for l in labels)
    for i, row in enumerate(cm):
        cells += f'<div class="h rh">{labels[i]}</div>'
        for j, v in enumerate(row):
            a = 0.08 + 0.82 * v / peak
            fg = "#fff" if a > 0.5 else "var(--ink)"
            cells += f'<div style="background:rgba(44,70,201,{a:.2f});color:{fg}">{v:,}</div>'
    return f'<div class="cm">{cells}</div>'


def _altair_theme():
    return {"config": {
        "background": "transparent",
        "font": "Schibsted Grotesk",
        "view": {"stroke": None},
        "axis": {"labelColor": "#3E4B47", "titleColor": "#5A6A64", "gridColor": "#D5DFDA",
                 "domainColor": "#C9D5CF", "tickColor": "#C9D5CF", "labelFontSize": 13, "titleFontWeight": 500, "labelLimit": 220},
        "legend": {"labelColor": "#17221F", "titleColor": "#5A6A64", "orient": "top", "labelFontSize": 13},
    }}


if hasattr(alt, "theme") and hasattr(alt.theme, "register"):  # Altair 5.5+
    alt.theme.register("sentilens", enable=True)(_altair_theme)
else:
    alt.themes.register("sentilens", _altair_theme)
    alt.themes.enable("sentilens")
COLOR_SCALE = alt.Scale(domain=CLASS_ORDER, range=[TONE[c][0] for c in CLASS_ORDER])


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_analyze, tab_stress, tab_topics, tab_eval, tab_about = st.tabs(
    ["Analyze a post", "Stress test", "Topics", "Evaluation", "How it works"]
)

EXAMPLES = {
    "Sarcasm": "Great job! Another two-hour delay. Just what I needed.",
    "Quiet complaint": "It works, I guess, but I wouldn't recommend buying it.",
    "Mixed opinion": "The phone has an excellent camera, but the battery life is disappointing.",
    "Hinglish praise": "Yeh phone ekdum mast hai, camera zabardast 🔥",
    "Hinglish rant": "Bilkul bakwas service, paisa barbad",
}


LEGEND = ('<div class="legend"><span><i style="background:rgba(19,121,91,.45)"></i>pushed toward positive</span>'
          '<span><i style="background:rgba(194,54,74,.45)"></i>pushed toward negative</span>'
          '<span>Unmarked words carry no learned weight. Hover a word for its score.</span></div>')


@st.cache_data(show_spinner=False)
def _demo(_model, _vectorizer, key):
    return sl.smart_analyze(_model, _vectorizer, clean_text, EXAMPLES["Sarcasm"])


def demo_analysis(m, v):
    return _demo(m, v, os.path.getmtime(MODEL_PATH))


def _use_example(text):
    st.session_state["post_text"] = text
    st.session_state["run_analysis"] = True


with tab_analyze:
    if model is None or vectorizer is None:
        st.error("Model files are missing. Put sentiment_model.pkl and tfidf_vectorizer.pkl next to app.py, then reload.")
    else:
        text_input = st.text_area(
            "Post or comment",
            key="post_text",
            placeholder="Paste a tweet, review or comment…",
            height=120,
        )
        cols = st.columns([1.3] + [1] * len(EXAMPLES), gap="small")
        run = cols[0].button("Analyze post", type="primary", use_container_width=True)
        for col, (name, text) in zip(cols[1:], EXAMPLES.items()):
            col.button(name, on_click=_use_example, args=(text,), use_container_width=True)

        if st.session_state.pop("run_analysis", False):
            run = True

        if not run:
            demo = demo_analysis(model, vectorizer)
            st.write("")
            left, right = st.columns([1.6, 1], gap="large")
            with left:
                H(specimen_html(demo))
                H(LEGEND)
            with right:
                H(f'<p class="demo-note">This is how SentiLens reads a post. The model alone calls this one '
                  f'<b>{demo["base_label"]}</b>; SentiLens spots the sarcasm and reads it as '
                  f'<b style="color:{tone(demo["final"])[0]}">{demo["final"]}</b>.</p>'
                  f'<p class="subtle">Paste your own post above, or try one of the examples.</p>')

        if run:
            if not text_input.strip():
                st.info("Type or paste a post above, or pick one of the examples.")
            else:
                res = sl.smart_analyze(model, vectorizer, clean_text, text_input)
                st.write("")
                left, right = st.columns([1.6, 1], gap="large")
                with left:
                    H(specimen_html(res))
                    H(LEGEND)
                    if res["changes"]:
                        H('<p class="subtle" style="margin-top:.8rem">Translated before reading: '
                          + ", ".join(f"{esc(a)} → {esc(b)}" for a, b in res["changes"]) + "</p>")
                with right:
                    H(verdict_html(res))

                if res["reasons"]:
                    st.markdown("### Why")
                    items = "".join(f"<li>{esc(r)}</li>" for r in res["reasons"])
                    items += "".join(f'<li class="ev">Evidence: {esc(e)}</li>' for e in res["sarcasm_evidence"])
                    H(f'<ul class="reasons">{items}</ul>')

                c1, c2 = st.columns([1, 1] if res["aspects"] else [1.6, 1], gap="large")
                with c1:
                    if res["top_features"]:
                        st.markdown("### Words that mattered most")
                        H(push_html(res["top_features"]))
                with c2:
                    if res["aspects"]:
                        st.markdown("### Part by part")
                        H(clauses_html(res["aspects"]))


@st.cache_data(show_spinner=False)
def run_behaviour_suite(_model, _vectorizer, cache_key, split="dev"):
    def analyze(text):
        r = sl.smart_analyze(_model, _vectorizer, clean_text, text, with_explanation=False)
        return r["base_label"], r["final"]
    return bt.run_suite(analyze, split)


def rate_cell(rate, colour):
    if rate is None:
        return '<td class="rate"><span class="subtle">can’t say “Mixed”</span></td>'
    return (f'<td class="rate"><span class="rnum">{rate*100:.0f}%</span>'
            f'<div class="rbar"><div style="width:{rate*100:.1f}%;background:{colour}"></div></div></td>')


with tab_stress:
    if model is None or vectorizer is None:
        st.error("Model files are missing.")
    else:
        H('<p class="lede">Instead of one accuracy number, SentiLens is tested skill by skill, the way the '
          'CheckList method (Ribeiro et al., ACL 2020) tests language models. Each skill gets up to 40 generated '
          'posts. Most have an expected label; the last two check that a harmless change, like a different brand '
          'name or a typo, leaves the verdict unchanged.</p>')
        key = "|".join(str(os.path.getmtime(p)) for p in
                       (MODEL_PATH, os.path.join(BASE_DIR, "smart_layer.py"), os.path.join(BASE_DIR, "behaviour_tests.py"),
                        os.path.join(BASE_DIR, "hinglish_lexicon.json"))
                       if os.path.exists(p))
        with st.spinner("Running behaviour tests…"):
            caps = run_behaviour_suite(model, vectorizer, key, "test")
            dev_caps = run_behaviour_suite(model, vectorizer, key, "dev")

        def caps_dumbbell(rows_in):
            out = ""
            for c in rows_in:
                b, sm = c["base"], c["smart"]
                name = (f'<div class="dname"><b>{esc(c["name"])}</b><span>{esc(c["desc"])}</span>'
                        f'<em>“{esc(c["example"])}”</em></div>')
                sx = sm * 100
                if b is None:
                    track = (f'<div class="dot smart" style="left:{sx:.1f}%"></div>'
                             f'<div class="lab l" style="left:{sx:.1f}%;color:var(--cobalt)">{sx:.0f}%</div>'
                             f'<div class="note" style="left:2%">model alone can’t answer “Mixed”</div>')
                else:
                    bx = b * 100
                    lo, hi = min(bx, sx), max(bx, sx)
                    line_col = "var(--cobalt)" if sx >= bx else "var(--neg)"
                    b_side, s_side = ("l", "r") if sx >= bx else ("r", "l")
                    if abs(sx - bx) < 0.5:
                        b_side, s_side = "l", "r"
                    # labels that would run off either edge sit above their dot instead
                    fix = lambda x, side: "t" if (side == "r" and x > 90) or (side == "l" and x < 8) else side
                    b_side, s_side = fix(bx, b_side), fix(sx, s_side)
                    track = (f'<div class="dline" style="left:{lo:.1f}%;width:{hi - lo:.1f}%;background:{line_col}"></div>'
                             f'<div class="dot base" style="left:{bx:.1f}%"></div>'
                             f'<div class="dot smart" style="left:{sx:.1f}%"></div>'
                             f'<div class="lab {b_side}" style="left:{bx:.1f}%;color:var(--neu)">{bx:.0f}%</div>'
                             f'<div class="lab {s_side}" style="left:{sx:.1f}%;color:var(--cobalt)">{sx:.0f}%</div>')
                out += name + f'<div class="dtrack">{track}</div>'
            axis = "".join(f"<span>{t}%</span>" for t in (0, 25, 50, 75, 100))
            return (f'<div class="dlegend"><span><i style="border:2.5px solid var(--neu)"></i>model alone</span>'
                    f'<span><i style="background:var(--cobalt)"></i>SentiLens</span></div>'
                    f'<div class="dumb">{out}<div class="axis">{axis}</div></div>')

        def caps_table(rows_in):
            rows = ""
            for c in rows_in:
                rows += (f'<tr><td class="cap"><b>{esc(c["name"])}<span class="kind">{c["kind"]}</span></b>'
                         f'<span>{esc(c["desc"])}</span></td><td class="ex">{esc(c["example"])}</td>'
                         f'<td>{esc(c["expected"])}</td>'
                         f'{rate_cell(c["base"], TONE["Neutral"][0])}{rate_cell(c["smart"], "var(--cobalt)")}</tr>')
            return (f'<table class="caps"><thead><tr><th>Skill</th><th>Example</th><th>Expected</th>'
                    f'<th>Model alone</th><th>SentiLens</th></tr></thead><tbody>{rows}</tbody></table>')

        def averages(rows_in):
            scored = [c for c in rows_in if c["base"] is not None]
            return (sum(c["base"] for c in scored) / len(scored), sum(c["smart"] for c in scored) / len(scored),
                    len(scored), sum(c["n"] for c in rows_in))

        avg_base, avg_smart, n_scored, total = averages(caps)
        H(f'<p class="claim">On {total} sentences it had never seen, SentiLens passed '
          f'<span class="c-smart">{avg_smart*100:.0f}%</span> of the checks. The model alone passed '
          f'<span class="c-base">{avg_base*100:.0f}%</span>.</p>')
        H(f'<p class="lede">Averages cover the {n_scored} skills both can answer. The held-out sentences use new patterns, '
          f'products, adjectives, brands and Hinglish words that the rules were never tuned on.</p>')
        H(caps_dumbbell(caps))
        st.write("")

        with st.expander("Where SentiLens still fails on the held-out test"):
            for c in caps:
                if c["fail_count"]:
                    st.markdown(f"**{c['name']}**: {c['fail_count']} of {c['n']} failed")
                    H('<ul class="reasons">' + "".join(
                        f'<li><span style="font-family:var(--serif)">{esc(p)}</span> '
                        f'<span class="subtle">got {esc(got)}, expected {esc(exp)}</span></li>'
                        for p, got, exp in c["fails"]) + "</ul>")

        d_base, d_smart, _, d_total = averages(dev_caps)
        with st.expander(f"Practice set used while improving the rules ({d_total} posts): "
                         f"model alone {d_base*100:.0f}%, SentiLens {d_smart*100:.0f}%"):
            st.caption(f"The rules were improved by studying failures on this set, so its score is optimistic. "
                       f"The {(d_smart - avg_smart)*100:.0f}-point drop to the held-out score shows how much they overfit.")
            H(caps_dumbbell(dev_caps))

        st.markdown("### Hand-picked examples")
        H('<p class="lede">Thirteen posts that bag-of-words models usually misread. Each one goes through the '
          'model alone and through SentiLens, and both answers are checked against how a person would read it.</p>')
        rows, base_ok, smart_ok = "", 0, 0
        for kind, text, expected in sl.CHALLENGES:
            r = sl.smart_analyze(model, vectorizer, clean_text, text)
            b_ok, s_ok = r["base_label"] == expected, r["final"] == expected
            base_ok += b_ok
            smart_ok += s_ok
            smart_txt = r["final"] + (f', leans {r["lean"].lower()}' if r["lean"] and r["final"] == "Mixed" else "")
            rows += (f'<tr><td class="kind">{esc(kind)}</td><td class="post">{esc(text)}</td>'
                     f'<td>{expected}</td>'
                     f'<td><span class="{"ok" if b_ok else "no"}">{"✓" if b_ok else "✗"}</span> {r["base_label"]} '
                     f'<span class="subtle">{r["base_conf"]*100:.0f}%</span></td>'
                     f'<td><span class="{"ok" if s_ok else "no"}">{"✓" if s_ok else "✗"}</span> {esc(smart_txt)}</td></tr>')
        n = len(sl.CHALLENGES)
        H(f'<p class="lede">The model alone reads <b>{base_ok} of {n}</b> correctly; SentiLens reads '
          f'<b style="color:var(--cobalt)">{smart_ok} of {n}</b>.</p>')
        H(f'<table class="cases"><thead><tr><th>Case</th><th>Post</th><th>Expected</th><th>Model alone</th>'
          f'<th>SentiLens</th></tr></thead><tbody>{rows}</tbody></table>')
        st.caption("The model alone has only three labels, so it can never answer “Mixed”. These are hand-picked "
                   "demonstration cases that the context rules were written for, not an accuracy estimate.")

        st.markdown("### Try to fool it")
        own = st.text_input("Your own tricky post", placeholder="Wow, thanks for deleting my order again 🙃")
        if own.strip():
            r = sl.smart_analyze(model, vectorizer, clean_text, own)
            col = tone(r["final"])[0]
            H(f'<p style="font-size:1.1rem">Model alone: <b>{r["base_label"]}</b> '
              f'<span class="subtle">({r["base_conf"]*100:.0f}%)</span> &nbsp;&nbsp; SentiLens: '
              f'<b style="color:{col}">{r["final"]}</b></p>')
            if r["reasons"]:
                H('<ul class="reasons">' + "".join(f"<li>{esc(x)}</li>" for x in r["reasons"]) + "</ul>")


with tab_topics:
    H('<p class="lede">Upload a labelled dataset to see which topics people talk about most and how they feel '
      'about each one. The CSV needs <code>entity</code> and <code>sentiment</code> columns; a date column adds a '
      'timeline.</p>')
    uploaded = st.file_uploader("Labelled posts (CSV)", type=["csv"], key="topic_csv")
    sample_path = os.path.join(BASE_DIR, "..", "cleaned_training_data.csv")
    if uploaded is None and os.path.exists(sample_path):
        if st.button("Use the project dataset instead"):
            st.session_state["use_sample"] = True
    source = uploaded if uploaded is not None else (sample_path if st.session_state.get("use_sample") else None)
    if source is None:
        st.caption("Upload a CSV to see topic insights.")
    else:
        try:
            df = pd.read_csv(source)
            if not {"entity", "sentiment"}.issubset(df.columns):
                st.error("This CSV has no `entity` and `sentiment` columns. Rename the topic and label columns and upload it again.")
            else:
                df = df.dropna(subset=["entity", "sentiment"]).copy()
                df["entity"] = df["entity"].astype(str).str.strip()
                df["sentiment"] = df["sentiment"].astype(str).str.strip().str.title()
                df = df[df["sentiment"].isin(CLASS_ORDER)]
                if df.empty:
                    st.warning("No rows are labelled Positive, Neutral or Negative.")
                else:
                    counts = df["entity"].value_counts()
                    overall_pct = df["sentiment"].value_counts(normalize=True) * 100
                    ct = pd.crosstab(df["entity"], df["sentiment"]).reindex(columns=CLASS_ORDER, fill_value=0)
                    pct = ct.div(ct.sum(axis=1).replace(0, 1), axis=0) * 100

                    H(stats_html([
                        ("posts", f"{len(df):,}"),
                        ("topics", f"{df['entity'].nunique():,}"),
                        ("positive", f"{overall_pct.get('Positive', 0):.0f}%"),
                        ("negative", f"{overall_pct.get('Negative', 0):.0f}%"),
                    ]))

                    max_topics = min(32, len(counts))
                    top_n = 1 if max_topics <= 1 else st.slider("Topics to show", 2, max_topics, min(12, max_topics))
                    top = counts.head(top_n).index

                    long = (pct.loc[top].reset_index().melt(id_vars="entity", var_name="Sentiment", value_name="Share"))
                    long["Posts"] = long["entity"].map(counts)
                    chart = (alt.Chart(long).mark_bar(height={"band": 0.72}).encode(
                        y=alt.Y("entity:N", sort=list(top), title=None),
                        x=alt.X("Share:Q", stack="normalize", title="Share of posts", axis=alt.Axis(format="%")),
                        color=alt.Color("Sentiment:N", scale=COLOR_SCALE, sort=CLASS_ORDER, title=None),
                        order=alt.Order("sentiment_order:Q"),
                        tooltip=["entity", "Sentiment", alt.Tooltip("Share:Q", format=".1f"), "Posts"],
                    ).transform_calculate(
                        sentiment_order="datum.Sentiment === 'Positive' ? 0 : datum.Sentiment === 'Neutral' ? 1 : 2"
                    ).properties(height=max(240, 34 * len(top))))
                    st.markdown("### How people feel about each topic")
                    st.caption("Sorted by number of posts, most discussed first.")
                    st.altair_chart(chart, use_container_width=True, theme=None)

                    pos_t = pct["Positive"].idxmax(); neg_t = pct["Negative"].idxmax()
                    st.markdown("### What stands out")
                    H(f'<div class="finding"><b>{esc(counts.idxmax())}</b> is the most discussed topic, with {int(counts.max()):,} posts.</div>'
                      f'<div class="finding"><b>{esc(pos_t)}</b> has the warmest reception: {pct.loc[pos_t, "Positive"]:.0f}% of its posts are positive.</div>'
                      f'<div class="finding"><b>{esc(neg_t)}</b> draws the most criticism: {pct.loc[neg_t, "Negative"]:.0f}% of its posts are negative.</div>'
                      f'<div class="finding"><b>{esc(ct["Negative"].idxmax())}</b> has the most negative posts overall ({int(ct["Negative"].max()):,}).</div>')

                    date_candidates = [c for c in df.columns if c.lower() in {"date", "datetime", "timestamp", "created_at", "createdat", "time"}]
                    if date_candidates:
                        date_col = date_candidates[0]
                        dated = df.copy()
                        dated[date_col] = pd.to_datetime(dated[date_col], errors="coerce", utc=True)
                        dated = dated.dropna(subset=[date_col])
                        if not dated.empty:
                            st.markdown("### Over time")
                            freq_label = st.selectbox("Group by", ["Day", "Week", "Month"], index=1)
                            freq = {"Day": "D", "Week": "W", "Month": "M"}[freq_label]
                            dated["period"] = dated[date_col].dt.to_period(freq).dt.start_time
                            trend = pd.crosstab(dated["period"], dated["sentiment"]).reindex(columns=CLASS_ORDER, fill_value=0)
                            trend_pct = trend.div(trend.sum(axis=1).replace(0, 1), axis=0)
                            tl = trend_pct.reset_index().melt(id_vars="period", var_name="Sentiment", value_name="Share")
                            st.altair_chart(alt.Chart(tl).mark_line(strokeWidth=2.5).encode(
                                x=alt.X("period:T", title=None),
                                y=alt.Y("Share:Q", axis=alt.Axis(format="%"), title="Share of posts"),
                                color=alt.Color("Sentiment:N", scale=COLOR_SCALE, sort=CLASS_ORDER, title=None),
                            ).properties(height=280), use_container_width=True, theme=None)
                            st.download_button("Download timeline (CSV)", trend.to_csv().encode("utf-8"),
                                               file_name="sentiment_over_time.csv", mime="text/csv")

                    summary = ct.copy()
                    summary["Total"] = ct.sum(axis=1)
                    for s in CLASS_ORDER:
                        summary[f"{s} %"] = pct[s].round(1)
                    summary = summary.sort_values("Total", ascending=False)
                    st.markdown("### Topic table")
                    show = summary.head(top_n).reset_index().rename(columns={"entity": "Topic"})
                    H(html_table(show, num_cols=[c for c in show.columns if c != "Topic"]))
                    st.write("")
                    st.download_button("Download topic table (CSV)", summary.to_csv().encode("utf-8"),
                                       file_name="topic_sentiment_summary.csv", mime="text/csv", type="primary")
        except Exception as exc:
            st.error(f"This file could not be read as a CSV ({exc}). Check that it is comma-separated and try again.")


@st.cache_data(show_spinner=False)
def leakage_overlap(train_path, val_path):
    cols = ["id", "entity", "sentiment", "text"]
    tr = pd.read_csv(train_path, names=cols, header=None)
    va = pd.read_csv(val_path, names=cols, header=None)
    va = va[va["sentiment"].isin(CLASS_ORDER)]
    texts = set(tr["text"].map(clean_text))
    return {
        "text_overlap": float(va["text"].map(clean_text).isin(texts).mean()),
        "id_overlap": float(va["id"].isin(set(tr["id"])).mean()),
    }


with tab_eval:
    train_path, val_path = find_data("twitter_training.csv"), find_data("twitter_validation.csv")
    if report:
        o, i = report["original"], report["improved"]
        fair, imp, claimed = o["accuracy"] * 100, i["accuracy"] * 100, 96.5
        H('<p class="claim">The original 96.5% was mostly memory.</p>')
        H(f'<p class="lede">The notebook tested on posts the model had already seen during training. Split fairly, so every '
          f'test post is new, the same model scores {fair:.1f}%. The striped part of the bar is what it had memorised.</p>')
        ticks = "".join(f'<span style="left:{t}%">{t}%</span>' for t in (0, 25, 50, 75, 100))
        H(f"""<div class="leak"><div class="bar">
<div class="solid" style="width:{fair:.1f}%"></div>
<div class="hatch" style="left:{fair:.1f}%;width:{claimed - fair:.1f}%"></div>
<div class="mark" style="left:{imp:.1f}%" title="Improved model {imp:.1f}%"></div></div>
<div class="ticks">{ticks}</div>
<div class="keys">
<div><b>{fair:.1f}%</b><i class="sw" style="background:var(--neu)"></i>learned: the original model on {report['test_rows']:,} unseen posts</div>
<div><b>+{claimed - fair:.1f} pts</b><i class="sw" style="background:rgba(194,54,74,.55)"></i>memorised: only counted because test posts leaked from training</div>
<div><b style="color:var(--cobalt)">{imp:.1f}%</b><i class="sw" style="background:var(--cobalt)"></i>improved model on the same unseen posts</div>
</div></div>""")
        if train_path and val_path:
            ov = leakage_overlap(train_path, val_path)
            H(f'<p class="finding"><b>{ov["text_overlap"]*100:.0f}%</b> of the original validation posts appear word for word '
              f'in the training data, and <b>{ov["id_overlap"]*100:.0f}%</b> share a tweet ID with training rows, because the '
              f'dataset stores several rewordings of each tweet. This is called data leakage.</p>')

        st.markdown("### What changed in the improved model")
        H('<ul class="reasons">'
          '<li>Duplicate posts removed before training, so repeated rewordings no longer dominate.</li>'
          '<li>Classes balanced, so the model stops leaning toward the most common label.</li>'
          '<li>Rare and over-common words filtered, with a full bigram vocabulary instead of a 20,000-word cap.</li>'
          '<li>Settings chosen on a separate validation split; the test posts were used once, at the end.</li></ul>')

        st.markdown("### Per-class F1")
        f1 = pd.DataFrame({
            "Class": CLASS_ORDER,
            "Original": [f'{o["per_class_f1"][c]*100:.1f}%' for c in CLASS_ORDER],
            "Improved": [f'{i["per_class_f1"][c]*100:.1f}%' for c in CLASS_ORDER],
        })
        f1.loc[len(f1)] = ["Macro average", f'{o["macro_f1"]*100:.1f}%', f'{i["macro_f1"]*100:.1f}%']
        H(html_table(f1, num_cols=["Original", "Improved"]))

        st.markdown("### Where the improved model gets confused")
        st.caption("Rows are the true label, columns the prediction. Darker cells hold more posts.")
        H(confusion_html(i["confusion"], report["labels"]))
        H(f'<p class="subtle" style="margin-top:1rem">Neutral is the hardest class: it is often confused with both '
          f'positive and negative. The deployed model is retrained on all {report["deployed_rows"]:,} unique posts '
          f'with the same settings.</p>')
    else:
        st.info("No fair evaluation has been saved yet. model_report.json is missing from the app folder.")


def _load_json(name):
    p = os.path.join(BASE_DIR, name)
    return json.load(open(p)) if os.path.exists(p) else None


with tab_eval:
    real_t, real_v = _load_json("real_test.json"), _load_json("real_val.json")
    if real_t:
        st.write("")
        H(f'<p class="claim">On real tweets, the rules fix about as many posts as they break.</p>')
        H(f'<p class="lede">Synthetic tests are easy to pass, so the rules were also run on the {real_t["n"]:,} unseen tweets '
          f'of the fair test set. They fixed {real_t["fixed"]:,} and broke {real_t["broke"]:,}: accuracy '
          f'{real_t["model_acc"]*100:.2f}% for the model alone, {real_t["sentilens_acc"]*100:.2f}% with SentiLens. '
          f'A “Mixed” verdict counts as its leaning label.</p>')
        RULE_NAMES = {"input change (brand masking / translation)": "brand names hidden",
                      "lexicon override": "opinion words beat unsure model",
                      "neutral fallback": "no opinion words → neutral", "mixed": "mixed opinion",
                      "negated criticism": "“not bad”", "negated praise": "“not good”", "complaint": "quiet complaint",
                      "sarcasm": "sarcasm", "hedging": "hedging"}
        items = sorted(real_t["by_rule"].items(), key=lambda kv: -kv[1]["changed"])
        peak = max(max(v["fixed"], v["broke"]) for _, v in items) or 1
        fb = '<div class="hd l">broke</div><div class="hd name">what changed the verdict</div><div class="hd r">fixed</div>'
        for k, v in items:
            bw, fw = v["broke"] / peak * 100, v["fixed"] / peak * 100
            fb += (f'<div class="l"><div style="width:{bw:.1f}%"></div><span style="right:calc({bw:.1f}% + 6px)">{v["broke"]}</span></div>'
                   f'<div class="name">{esc(RULE_NAMES.get(k, k))}</div>'
                   f'<div class="r"><div style="width:{fw:.1f}%"></div><span style="left:calc({fw:.1f}% + 6px)">{v["fixed"]}</span></div>')
        H(f'<div class="fb">{fb}</div>')
        H('<p class="subtle" style="margin-top:.8rem">Overall the context layer neither helps nor hurts on real tweets: '
          'it fixes about as many posts as it breaks, while clearly helping on the specific cases it targets (see Stress '
          'test and Hinglish). Hiding brand names makes verdicts brand-neutral; it was slightly negative on the '
          'validation split and slightly positive here. The rules were tightened on a separate validation split '
          + (f'({real_v["n"]:,} tweets) ' if real_v else '') + 'to apply only to short posts, and the “no opinion words '
          '→ neutral” fallback is still a net loss on real tweets.</p>')

    hing = _load_json("hinglish_eval.json")
    if hing:
        ref = hing["reference_trained_on_hinglish"]
        m0, m1 = hing["semeval_builtin"]["model_acc"] * 100, hing["semeval_builtin"]["sentilens_acc"] * 100
        m2, m3 = hing["semeval_learned"]["sentilens_acc"] * 100, ref["acc"] * 100
        st.write("")
        H(f'<p class="claim">Learning {hing["lexicon_words"]} Hinglish words lifts accuracy from {m0:.0f}% to {m2:.0f}%.</p>')
        H(f'<p class="lede">Measured on the {hing["test_n"]:,} labelled test tweets of SemEval-2020 Task 9 (SentiMix). The words '
          f'were learned from its 14,000 training tweets, with names and religious, community, gender and political words '
          f'removed. A model trained directly on Hinglish sets the ceiling.</p>')
        lo_ax, hi_ax = 35, 75
        X = lambda v: (v - lo_ax) / (hi_ax - lo_ax) * 100
        pts = [(m0, "model alone", "var(--neu)", "down"), (m1, "built-in dictionary", "#8794d9", "up"),
               (m2, "learned word list", "var(--cobalt)", "down"), (m3, "ceiling: trained on Hinglish", "var(--paper)", "up")]
        lad = f'<div class="rail"></div><div class="gain" style="left:{X(m0):.1f}%;width:{X(m2) - X(m0):.1f}%"></div>'
        for v, name, col, side in pts:
            border = "border:2.5px dashed var(--ink);" if "ceiling" in name else f"background:{col};"
            lad += (f'<div class="pt" style="left:{X(v):.1f}%;{border}"></div>'
                    f'<div class="tag {side}" style="left:{X(v):.1f}%"><b>{v:.1f}%</b>{name}</div>')
        lad += "".join(f'<div class="tick" style="left:{X(t):.1f}%">{t}%</div>' for t in (40, 50, 60, 70))
        H(f'<div class="ladder">{lad}</div>')
        st.caption(f"Axis runs from {lo_ax}% to {hi_ax}%. The {hing['youtube_n']:,} Hinglish YouTube comments below were "
                   f"never used for anything but this final check.")
        tbl = pd.DataFrame([
            {"Dataset": "SemEval test", "Model alone": hing["semeval_builtin"]["model_f1"],
             "Built-in dictionary": hing["semeval_builtin"]["sentilens_f1"], "Learned list": hing["semeval_learned"]["sentilens_f1"]},
            {"Dataset": "YouTube comments", "Model alone": hing["youtube_builtin"]["model_f1"],
             "Built-in dictionary": hing["youtube_builtin"]["sentilens_f1"], "Learned list": hing["youtube_learned"]["sentilens_f1"]},
        ])
        for c in ["Model alone", "Built-in dictionary", "Learned list"]:
            tbl[c] = tbl[c].map(lambda v: f"{v*100:.1f}%")
        st.caption("Macro F1 (balances the three labels)")
        H(html_table(tbl, num_cols=["Model alone", "Built-in dictionary", "Learned list"]))


with tab_about:
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        st.markdown("### What happens to a post")
        H('<ol class="steps">'
          '<li><span><b>Translate</b> Hinglish words and emojis into English cues: a hand-written dictionary (mast → great, 🔥 → amazing) plus 338 opinion words learned from SemEval Hinglish tweets.</span></li>'
          '<li><span><b>Hide brand names</b> the model learned biased opinions about (Amazon, Nvidia, FIFA…).</span></li>'
          '<li><span><b>Clean</b> the text: lowercase, remove links, mentions and symbols.</span></li>'
          '<li><span><b>Vectorise</b> it with TF-IDF over single words and word pairs.</span></li>'
          '<li><span><b>Predict</b> with Logistic Regression, which gives a probability for each label.</span></li>'
          '<li><span><b>Explain</b> each word’s push: its TF-IDF value times the model’s weight for positive minus negative.</span></li>'
          '<li><span><b>Check context</b> with the VADER opinion lexicon: sarcasm, quiet complaints, negation (“not bad”), and mixed opinions. These overrides apply only to short posts, where they were shown not to hurt.</span></li>'
          '<li><span><b>Decide</b> the final label and list the reasons behind it.</span></li></ol>')
    with b:
        st.markdown("### What it can’t do")
        H('<ul class="reasons">'
          '<li>The context checks are hand-written rules. They catch common patterns, not every sarcastic post.</li>'
          '<li>Hinglish support is a small dictionary, not a trained language model.</li>'
          '<li>Training labels describe sentiment toward a brand or game, so general posts are harder.</li>'
          '<li>It reads text you enter or upload. It does not connect to live social media.</li>'
          '<li>Probabilities are the model’s estimate, not a guarantee.</li></ul>')
        st.markdown("### Data")
        H('<p>Twitter Entity Sentiment Analysis dataset (Kaggle): about 74,000 labelled tweets across 32 brands and games.</p>')
