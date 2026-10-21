import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Multi-Cancer Detection System", page_icon="🧬", layout="wide")


# ---------------- 1. load the 9 saved files (only once, then cached) ----------------
@st.cache_resource
def load_models(name):
    model   = joblib.load(f"scripts/{name}_model.pkl")
    prep    = joblib.load(f"scripts/{name}_preprocessor.pkl")
    encoder = joblib.load(f"scripts/{name}_label_encoder.pkl")
    return model, prep, encoder

organ_model, organ_prep, organ_encoder = load_models("organ")
stage_model, stage_prep, stage_encoder = load_models("stage")
surv_model,  surv_prep,  surv_encoder  = load_models("survival")


# ---------------- 2. dropdown options from the dataset ----------------
@st.cache_data
def load_options():
    df = pd.read_csv("datasets/cancer_patients_tabular.csv")
    def options(col):
        return ["Unknown"] + sorted(df[col].dropna().unique())
    return {col: options(col) for col in
            ["histology_code", "tumor_stage", "t_stage", "n_stage", "m_stage"]}

OPTIONS = load_options()
NOT_STAGED = ["Brain", "Bone & Soft Tissue"]


# ---------------- 3. page ----------------
st.title("🧬 Multi-Cancer Detection System")
st.caption("Predicts cancer type, stage and survival from patient data")

left, right = st.columns([1.3, 1], gap="large")

# ----- input form -----
with left:
    with st.form("patient_form"):
        st.subheader("Patient details")

        c1, c2 = st.columns(2)
        age = c1.number_input("Age", min_value=1, max_value=110, value=50)
        sex = c2.selectbox("Gender", ["Male", "Female", "Unknown"])

        histology = st.selectbox("Histology code", OPTIONS["histology_code"])

        c1, c2 = st.columns(2)
        tumor_stage = c1.selectbox("Tumor stage", OPTIONS["tumor_stage"])
        t_stage     = c2.selectbox("T stage (tumour size)", OPTIONS["t_stage"])
        n_stage     = c1.selectbox("N stage (lymph nodes)", OPTIONS["n_stage"])
        m_stage     = c2.selectbox("M stage (metastasis)", OPTIONS["m_stage"])

        c1, c2 = st.columns(2)
        radiation = c1.selectbox("Radiation therapy", ["Unknown", "Yes", "No"])
        returned  = c2.selectbox("Cancer returned after treatment", ["Unknown", "Yes", "No"])

        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

# ----- prediction -----
with right:
    st.subheader("Prediction")

    if not submitted:
        st.info("Fill in the patient details and press **Predict**.")
    else:
        patient = pd.DataFrame([{
            "age": age, "sex": sex, "histology_code": histology,
            "tumor_stage": tumor_stage, "t_stage": t_stage,
            "n_stage": n_stage, "m_stage": m_stage,
            "radiation_therapy": radiation, "new_tumor_event": returned,
        }])

        # 1. organ
        X = organ_prep.transform(patient)
        organ = organ_encoder.inverse_transform(organ_model.predict(X))[0]
        patient["organ"] = organ

        # 2. stage
        if organ in NOT_STAGED:
            stage, stage_note = "Not staged", f"{organ} cancers are not staged as Stage I–IV."
        elif tumor_stage != "Unknown":
            stage, stage_note = tumor_stage, "Taken from your input."
        else:
            X = stage_prep.transform(patient)
            stage = stage_encoder.inverse_transform(stage_model.predict(X))[0]
            stage_note = "Predicted by the model."
            patient["tumor_stage"] = stage             # use predicted stage for survival

        # 3. survival
        X = surv_prep.transform(patient)
        survival = surv_encoder.inverse_transform(surv_model.predict(X))[0]

        # ----- show results -----
        st.metric("Cancer type (organ)", organ)
        st.metric("Stage", stage)
        st.caption(stage_note)

        if survival == "DECEASED":
            st.error(f"**Survival:** {survival}")
        else:
            st.success(f"**Survival:** {survival}")

        if hasattr(surv_model, "predict_proba"):
            probs = surv_model.predict_proba(X)[0]
            risk = probs[list(surv_encoder.classes_).index("DECEASED")]
            st.write(f"**Risk of death:** {risk:.0%}")
            st.progress(float(risk))

st.divider()
st.caption("Educational project trained on TCGA patient data. Predictions are statistical estimates, not a medical diagnosis.")