from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__, static_folder='styles')

# ---------------- 1. load the 9 saved files (3 per target) ----------------
def load_models(name):
    model   = joblib.load(f'scripts/{name}_model.pkl')
    prep    = joblib.load(f'scripts/{name}_preprocessor.pkl')
    encoder = joblib.load(f'scripts/{name}_label_encoder.pkl')
    return model, prep, encoder

organ_model, organ_prep, organ_encoder = load_models('organ')
stage_model, stage_prep, stage_encoder = load_models('stage')
surv_model,  surv_prep,  surv_encoder  = load_models('survival')

# ---------------- 2. dropdown lists from the dataset ----------------
df = pd.read_csv('datasets/cancer_patients_tabular.csv')

def get_options(column):
    values = sorted(df[column].dropna().unique())
    return ['Unknown'] + values          # Unknown first = selected by default

histology_codes = get_options('histology_code')
tumor_stages    = get_options('tumor_stage')
t_stages        = get_options('t_stage')
n_stages        = get_options('n_stage')
m_stages        = get_options('m_stage')

# cancers that are not staged as Stage I-IV
NOT_STAGED = ['Brain', 'Bone & Soft Tissue']


# ---------------- 3. the web page ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None

    if request.method == 'POST':
        # --- read the form into a one-row table ---
        patient = pd.DataFrame([{
            'age':               float(request.form['age']),
            'sex':               request.form['sex'],
            'histology_code':    request.form['histology_code'],
            'tumor_stage':       request.form['tumor_stage'],
            't_stage':           request.form['t_stage'],
            'n_stage':           request.form['n_stage'],
            'm_stage':           request.form['m_stage'],
            'radiation_therapy': request.form['radiation_therapy'],
            'new_tumor_event':   request.form['new_tumor_event'],
        }])

        # --- predict organ:  input -> numbers -> predict -> word ---
        X = organ_prep.transform(patient)
        organ = organ_encoder.inverse_transform(organ_model.predict(X))[0]
        patient['organ'] = organ                       # stage & survival need organ

        # --- predict stage ---
        if organ in NOT_STAGED:
            stage = 'Not staged'
        elif patient['tumor_stage'][0] != 'Unknown':
            stage = patient['tumor_stage'][0] + ' (entered)'
        else:
            X = stage_prep.transform(patient)
            stage = stage_encoder.inverse_transform(stage_model.predict(X))[0]
            patient['tumor_stage'] = stage             # use predicted stage for survival

        # --- predict survival ---
        X = surv_prep.transform(patient)
        survival = surv_encoder.inverse_transform(surv_model.predict(X))[0]

        risk = None
        if hasattr(surv_model, 'predict_proba'):       # SVC without probability=True has no %
            probs = surv_model.predict_proba(X)[0]
            deceased = list(surv_encoder.classes_).index('DECEASED')
            risk = round(probs[deceased] * 100)

        result = {'organ': organ, 'stage': stage, 'survival': survival, 'risk': risk}

    return render_template('index.html',
                           histology_codes=histology_codes,
                           tumor_stages=tumor_stages,
                           t_stages=t_stages,
                           n_stages=n_stages,
                           m_stages=m_stages,
                           result=result)


if __name__ == '__main__':
    app.run(debug=True)