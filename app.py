import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Cancer Detection System", page_icon="🧬", layout="centered")


# =====================================================================
# 1. STYLE  (custom CSS to make Streamlit look like the HTML design)
# =====================================================================
st.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #eef3f6;  --card: #ffffff;  --text: #102a33;  --muted: #5b7079;
    --line: #dde6ea; --field: #f7fafb; --accent: #0e7c74; --accent-dark: #0a5f59;
    --accent-soft: #e2f3f1;
}

/* page background + font */
.stApp { background: var(--bg); }
.stApp, .stApp p, .stApp label, .stApp input, .stApp button,
.stApp div[data-baseweb="select"] * { font-family: 'Plus Jakarta Sans', system-ui, sans-serif; }

/* hide Streamlit's own top bar, menu and footer */
header[data-testid="stHeader"], #MainMenu, footer { display: none; }

/* main column */
.block-container { padding-top: 0 !important; padding-bottom: 3rem; max-width: 640px; }

/* ---------- header banner (full width) ---------- */
.hero {
    position: relative; overflow: hidden;
    width: 100vw; left: 50%; margin-left: -50vw;
    padding: 44px 24px 110px; text-align: center; color: #fff;
    background: linear-gradient(135deg, #0a5f59 0%, #0e7c74 55%, #13a397 100%);
}
.hero::before, .hero::after {
    content: ''; position: absolute; border-radius: 50%; background: rgba(255,255,255,.07);
}
.hero::before { width: 340px; height: 340px; top: -170px; left: -90px; }
.hero::after  { width: 260px; height: 260px; bottom: -150px; right: -60px; }
.hero h1 {
    position: relative; margin: 0; padding: 0; color: #fff;
    font-size: clamp(26px, 4vw, 38px); font-weight: 800; letter-spacing: -.02em;
}
.hero p { position: relative; margin: 10px 0 0; opacity: .85; font-size: 15px; }

/* ---------- form card (overlaps the banner) ---------- */
[data-testid="stForm"] {
    position: relative; z-index: 1;
    margin-top: -90px;
    background: var(--card); border: none; border-radius: 16px; padding: 28px;
    box-shadow: 0 1px 2px rgba(16,42,51,.05), 0 12px 32px -12px rgba(16,42,51,.18);
}
.form-title {
    font-size: 18px; font-weight: 700; color: var(--text);
    padding-bottom: 14px; margin-bottom: 4px; border-bottom: 1px solid var(--line);
}

/* labels */
[data-testid="stWidgetLabel"] p { font-size: 13px !important; font-weight: 600; color: var(--muted); }

/* dropdowns */
div[data-baseweb="select"] > div {
    background: var(--field) !important; border: 1.5px solid var(--line) !important;
    border-radius: 10px !important; min-height: 44px;
}
div[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft); }

/* age box */
[data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: var(--field) !important; border: 1.5px solid var(--line) !important;
    border-radius: 10px !important; min-height: 44px;
}
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 4px var(--accent-soft); }
[data-testid="stNumberInput"] input { background: transparent; }
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none; }

/* predict button */
[data-testid="stFormSubmitButton"] button {
    width: 100%; height: 50px; margin-top: 8px; border: none; border-radius: 12px;
    background: linear-gradient(135deg, var(--accent) 0%, #13a397 100%);
    box-shadow: 0 8px 20px -8px rgba(14,124,116,.7); transition: transform .15s, filter .15s;
}
[data-testid="stFormSubmitButton"] button p { color: #fff; font-size: 16px; font-weight: 700; }
[data-testid="stFormSubmitButton"] button:hover { transform: translateY(-1px); filter: brightness(1.05); border: none; }

/* ---------- result card ---------- */
.result-card {
    position: relative; overflow: hidden; margin-top: 8px;
    background: var(--card); border-radius: 16px; padding: 26px 28px 14px;
    box-shadow: 0 1px 2px rgba(16,42,51,.05), 0 12px 32px -12px rgba(16,42,51,.18);
    animation: rise .4s ease both; color: var(--text);
}
.result-card::before {
    content: ''; position: absolute; inset: 0 0 auto 0; height: 5px;
    background: linear-gradient(90deg, var(--accent), #13a397, #7dd3c8);
}
.result-card h4 {
    margin: 0 0 12px; padding: 0; font-size: 13px; font-weight: 700;
    letter-spacing: .1em; text-transform: uppercase; color: var(--accent);
}
.headline { background: var(--accent-soft); border-radius: 12px; padding: 12px 18px 16px; margin-bottom: 6px; }
.headline span { font-size: 14px; color: var(--muted); }
.headline div  { font-size: 30px; font-weight: 800; letter-spacing: -.02em; color: var(--accent-dark); }
.row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 14px 0; border-bottom: 1px dashed var(--line); font-size: 14px; color: var(--muted);
}
.row:last-child { border-bottom: none; }
.row b { font-size: 18px; color: var(--text); }
.row b.living { color: #15803d; }  .row b.deceased { color: #b42318; }
.row.no-line { border-bottom: none; padding-bottom: 4px; }
.note { font-size: 12px; color: var(--muted); padding-bottom: 12px; border-bottom: 1px dashed var(--line); }
.risk-track { height: 8px; border-radius: 99px; background: #edf2f4; overflow: hidden; margin: -4px 0 14px; }
.risk-fill  { height: 100%; border-radius: 99px; }
.disclaimer { text-align: center; font-size: 12px; color: var(--muted); margin-top: 22px; }

@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
</style>
""")


# =====================================================================
# 2. LOAD MODELS (once, then cached)
# =====================================================================
#  One file per target. Each file holds:
#    'model'   -> the trained model (Pipeline, or CatBoost with native categories)
#    'labels'  -> class names, in the order the model's numbers refer to
#    'columns' -> (only for native CatBoost) which input columns to pass
@st.cache_resource
def load_model(filename):
    return joblib.load(f"models/{filename}.pkl")

organ_bundle    = load_model("organ_catboost")
stage_bundle    = load_model("stage_lightgbm_tnm")
survival_bundle = load_model("survival_catboost")


def predict(bundle, patient):
    """Return (label, probabilities) for one patient."""
    X = patient[bundle["columns"]] if "columns" in bundle else patient
    number = int(np.ravel(bundle["model"].predict(X))[0])     # e.g. 4
    probs  = bundle["model"].predict_proba(X)[0]               # one probability per label
    return bundle["labels"][number], probs


@st.cache_data
def load_options():
    df = pd.read_csv("datasets/cancer_patients_tabular.csv")
    return {col: ["Unknown"] + sorted(df[col].dropna().unique())
            for col in ["histology_code", "tumor_stage", "t_stage", "n_stage", "m_stage"]}

OPTIONS = load_options()
NOT_STAGED = ["Brain", "Bone & Soft Tissue"]


# =====================================================================
# 3. HEADER
# =====================================================================
st.html("""
<div class="hero">
    <h1>Multi-Cancer Detection System</h1>
    <p>Prediction of cancer types, stage and survival percentage</p>
</div>
""")


# =====================================================================
# 4. FORM
# =====================================================================
with st.form("patient_form"):
    st.html('<div class="form-title">Patient details</div>')

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=1, max_value=110, value=None, placeholder="e.g. 55")
    sex = c2.selectbox("Gender", ["Male", "Female", "Unknown"])

    histology = st.selectbox("Histology Code", OPTIONS["histology_code"])

    c1, c2 = st.columns(2)
    tumor_stage = c1.selectbox("Tumor stage", OPTIONS["tumor_stage"])
    t_stage     = c2.selectbox("T_Stage", OPTIONS["t_stage"])

    c1, c2 = st.columns(2)
    n_stage = c1.selectbox("N_Stage", OPTIONS["n_stage"])
    m_stage = c2.selectbox("M_Stage", OPTIONS["m_stage"])

    c1, c2 = st.columns(2)
    radiation = c1.selectbox("Radiation Therapy", ["Unknown", "Yes", "No"])
    returned  = c2.selectbox("Cancer Returned", ["Unknown", "Yes", "No"])

    submitted = st.form_submit_button("Predict", width="stretch")


# =====================================================================
# 5. PREDICTION
# =====================================================================
if submitted:
    if age is None:
        st.warning("Please enter the patient's age.")
        st.stop()

    patient = pd.DataFrame([{
        "age": age, "sex": sex, "histology_code": histology,
        "tumor_stage": tumor_stage, "t_stage": t_stage,
        "n_stage": n_stage, "m_stage": m_stage,
        "radiation_therapy": radiation, "new_tumor_event": returned,
    }])

    # 1. organ
    organ, _ = predict(organ_bundle, patient)
    patient["organ"] = organ

    # 2. stage (uses T, N, M when given)
    if organ in NOT_STAGED:
        stage, note = "Not staged", f"{organ} cancers are not staged as Stage I–IV."
    elif tumor_stage != "Unknown":
        stage, note = tumor_stage, "Taken from your input."
    else:
        stage, _ = predict(stage_bundle, patient)
        note = ("Predicted from T, N, M." if "Unknown" not in (t_stage, n_stage, m_stage)
                else "Predicted by the model (less reliable without T, N, M).")
        patient["tumor_stage"] = stage            # use predicted stage for survival

    # 3. survival + mortality risk
    survival, probs = predict(survival_bundle, patient)
    risk = probs[list(survival_bundle["labels"]).index("DECEASED")]
    pct = round(risk * 100)
    color = "#15803d" if pct < 30 else "#b54708" if pct < 60 else "#b42318"
    risk_html = f"""
        <div class="row"><span>Mortality risk score</span><b>{pct}%</b></div>
        <div class="risk-track"><div class="risk-fill" style="width:{pct}%; background:{color};"></div></div>
    """

    st.html(f"""
    <div class="result-card">
        <h4>Prediction</h4>
        <div class="headline"><span>Cancer type (organ)</span><div>{organ}</div></div>
        <div class="row no-line"><span>Stage</span><b>{stage}</b></div>
        <div class="note">{note}</div>
        <div class="row"><span>Survival</span><b class="{survival.lower()}">{survival}</b></div>
        {risk_html}
    </div>
    """)

st.html('<p class="disclaimer">Educational project trained on TCGA patient data. '
        'Predictions are statistical estimates, not a medical diagnosis.</p>')