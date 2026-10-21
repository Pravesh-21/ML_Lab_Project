# 🧬 Multi-Cancer Detection System

A machine learning system that predicts **cancer type (organ)**, **cancer stage**, and a **survival outcome with a mortality risk score** from clinical patient data, served through an interactive **Streamlit** web app.

> ⚠️ **Disclaimer:** This is an educational project. Predictions are statistical estimates from historical data and are **not** a medical diagnosis.

🔗 **Live demo:** *add your Streamlit Cloud link here*

---

## 📌 Overview

When a patient is known to have a tumour, three questions matter most:

1. **Where is the cancer?** → *organ / cancer type*
2. **How advanced is it?** → *stage (I–IV)*
3. **What is the outlook?** → *survival (Living / Deceased) + mortality risk score*

The project trains **one dedicated model per question** and chains them together, the same way information flows in a clinical workflow:

```
Patient details ──► 🫁 Organ model ──► 📊 Stage model ──► ❤️ Survival model
                     (CatBoost)        (LightGBM)          (CatBoost)
                        │                  │                    │
                    "Breast"           "STAGE II"        "LIVING · 4% risk"
```

The predicted organ is passed to the stage model, and both organ and stage are passed to the survival model.

---

## ✨ Features

- **3 prediction tasks**: organ (23 classes), stage (4 classes), survival (2 classes)
- **9 ML algorithms compared per task**: Logistic Regression, KNN, SVC, Decision Tree, Random Forest, XGBoost, **LightGBM**, **CatBoost**
- **One model file per target**: each `.pkl` contains everything needed to predict (preprocessing + model + label names)
- **Handles missing information**: any field can be left as `Unknown`
- **Smart staging**: uses the entered stage when known, predicts it from T/N/M when not, and reports brain tumours and sarcomas as *Not staged*
- **Mortality risk score**: predicted probability of the *Deceased* class
- **Web interface**: a styled Streamlit app; enter patient details and get all three predictions instantly

---

## 📂 Dataset

| | |
|---|---|
| **Source** | The Cancer Genome Atlas (TCGA) |
| **Patients** | 9,523 real, anonymised cancer patients |
| **Cancer types** | 32 TCGA cancer types, grouped into 23 organs |
| **Format** | CSV, 17 columns |

The dataset combines patient records from **TCGA-Reports** (Kefeli & Tatonetti, 2024) with clinical data from the **TCGA PanCancer Atlas** (via cBioPortal), matched by patient ID.

### Columns

| Column | Description |
|---|---|
| `age` | Age at diagnosis |
| `sex` | Male / Female |
| `race`, `ethnicity` | As recorded by the hospital |
| `histology_code` | ICD-O-3 code for the tumour's cell type (e.g. `8500/3`) |
| `tumor_stage` | Overall stage (Stage I–IV and sub-stages) |
| `t_stage`, `n_stage`, `m_stage` | TNM staging: Tumour size, lymph Nodes, Metastasis |
| `radiation_therapy` | Whether the patient received radiation |
| `new_tumor_event` | Whether the cancer returned after treatment |
| `survival_status` | LIVING / DECEASED |
| `survival_months` | Months of follow-up |
| `cancer_type`, `cancer_name` | TCGA cancer code and full name |
| `organ` | Grouped organ (Brain, Lung, Breast, …) |

---

## ⚙️ Methodology

### 1. Data cleaning
- **Numerical (`age`)**: skewness checked (−0.44); missing values filled with the **median**
- **Categorical**: columns with **< 5 %** missing filled with the **mode**; columns with **> 5 %** missing filled with **`Unknown`**, so the model can learn from missingness (e.g. unstaged brain tumours)
- **Target columns** are never filled; rows with a missing target are dropped for that task
- **Stage labels** simplified to four groups (e.g. `STAGE IIA`, `Stage IIB` → `STAGE II`)

### 2. Outlier analysis
- The IQR method on `age` flagged ~100 patients under 24
- **Kept**, because they are genuine young patients with cancers common at younger ages (brain, thyroid, testicular), not data errors

### 3. Feature selection (per task)

| Task | Target | Inputs | Removed (to avoid leakage) |
|---|---|---|---|
| Organ | `organ` | age, sex, histology, stage, T/N/M, therapy, recurrence | cancer type/name, survival |
| Stage | `stage` (I–IV) | age, sex, histology, **T/N/M**, therapy, recurrence, **organ** | `tumor_stage` |
| Survival | `survival_status` | age, sex, histology, stage, T/N/M, therapy, recurrence, **organ** | `survival_months` |

`race` and `ethnicity` were dropped: they added under 0.5 % accuracy and raise fairness concerns.

### 4. Preprocessing
- `StandardScaler` for `age`; `OneHotEncoder(handle_unknown='ignore')` for categorical features
- **CatBoost (survival)** uses its **native categorical handling** instead of one-hot encoding
- **Stratified 80 / 20** train–test split; all preprocessing fitted on training data only

### 5. Model training and selection
All algorithms were trained on each task, and the best model per task was saved as a **single file** with `joblib`:

```python
{'model': <Pipeline or CatBoost>, 'labels': [...class names...], 'columns': [...]}   # columns: CatBoost only
```

---

## 📊 Results

### Final models

| Task | Best model | Accuracy | Macro F1 | Baseline* |
|---|---|---|---|---|
| Organ (23 classes) | **CatBoost** | **92.7 %** | 0.907 | 11 % |
| Stage (4 classes) | **LightGBM** (with T/N/M) | **96.2 %** | 0.955 | 32 % |
| Survival (2 classes) | **CatBoost** (native categories) | **80.9 %** | 0.765 (ROC-AUC 0.864) | 71 % |

<sub>*Baseline = always predicting the most common class.</sub>

### Gradient-boosting comparison

| Task | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Organ | 92.7 % | 91.5 % | **92.7 %** |
| Stage (with T/N/M) | 95.9 % | **96.2 %** | 96.1 % |
| Survival | 80.1 % | 79.5 % | **80.9 %** (native) |

### Key observations
- **Organ**: the boosting models improved on the earlier Random Forest (91 % → 92.7 %). The weakest classes are Skin and Stomach, which have few training samples.
- **Stage**: without T/N/M, accuracy is only **~51 %**; with T/N/M it rises to **96.2 %**, confirming that TNM descriptors carry most of the staging information.
- **Survival**: hyperparameter tuning, ensembling, ordinal TNM encoding and follow-up-based label filtering gave **no significant gain** (all within ±0.6 %). Performance is limited by the information in the structured features (no treatment, lab or comorbidity data), not by the choice of model.

---

## 🗂️ Project Structure

```
ML_Lab_Project/
├── app.py                          # Streamlit web app
├── requirements.txt
├── .streamlit/
│   └── config.toml                 # app colour theme
├── datasets/
│   └── cancer_patients_tabular.csv
├── models/                         # one file per target
│   ├── organ_catboost.pkl
│   ├── stage_lightgbm_tnm.pkl
│   └── survival_catboost.pkl
└── scripts/
    └── ML_project.ipynb            # cleaning, EDA, training, evaluation
```

---

## 🚀 Getting Started

### 1. Clone and enter the project
```bash
git clone https://github.com/Pravesh-21/ML_Lab_Project.git
cd ML_Lab_Project
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> Use the **same scikit-learn version** listed in `requirements.txt`, otherwise the `.pkl` files may fail to load.

### 4. Run the app
```bash
streamlit run app.py
```
It opens at **http://localhost:8501**.

---

## 🧪 Example Test Cases

| Field | Breast | Brain | Colorectal |
|---|---|---|---|
| Age | 52 | 58 | 65 |
| Gender | Female | Female | Male |
| Histology Code | 8500/3 | 9401/3 | 8140/3 |
| Tumor stage | STAGE IIA | Unknown | Unknown |
| T / N / M | T2 / N0 / M0 | Unknown | T3 / N1 / M1 |
| Radiation Therapy | Yes | Yes | No |
| Cancer Returned | No | Yes | No |
| **Expected organ** | Breast | Brain | Colorectal |
| **Expected stage** | Stage IIA (entered) | Not staged | **Stage IV** (predicted from T/N/M) |
| **Expected survival** | Living (low risk) | Deceased (high risk) | Living (medium risk) |

---

## ⚠️ Limitations

- **All patients have cancer**: the system identifies *which* cancer, not *whether* a person has cancer.
- **`histology_code` is strongly linked to organ**, which contributes substantially to organ accuracy.
- **Stage prediction without T/N/M is unreliable (~51 %)**; the high stage accuracy applies when T/N/M are available.
- **The mortality risk score** is *P(deceased at last follow-up)*. It does not account for follow-up time and is **not** a 5-year survival probability.
- **Survival is capped at ~81 %** by the available features; treatment, laboratory and comorbidity data are not included.
- **Bone cancer is not truly represented**: the "Bone & Soft Tissue" class is TCGA sarcoma (mostly soft tissue).
- **Data comes mainly from US hospitals**, so performance on other populations (e.g. Indian patients) is untested.

### Future work
- Time-to-event survival models (e.g. Cox regression, survival forests)
- Rule-based stage derivation from T/N/M (AJCC)
- Validation on an external, non-TCGA patient cohort

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, pandas, NumPy, scikit-learn |
| Gradient boosting | XGBoost, LightGBM, CatBoost |
| Visualisation | Matplotlib, Seaborn |
| Model storage | joblib |
| Web app | Streamlit (custom CSS) |
| Deployment | Streamlit Community Cloud |

---

## 📚 References

- Kefeli, J. & Tatonetti, N. (2024). *TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models.* Patterns.
- The Cancer Genome Atlas (TCGA) PanCancer Atlas clinical data, via cBioPortal.
- Prokhorenkova, L. et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS.
- Ke, G. et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree.* NeurIPS.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* KDD.

---

## 👥 Authors

- **Pravesh Shrivastava**
- **Aniket Nawkar**

*Made as part of the Machine Learning Lab project.*