# 🧬 Multi-Cancer Detection System

A machine learning system that predicts **cancer type (organ)**, **cancer stage**, and **patient survival** from clinical patient data, served through a simple **Flask** web app.

> ⚠️ **Disclaimer:** This is an educational project. Predictions are statistical estimates from historical data and are **not** a medical diagnosis.

---

## 📌 Overview

When a patient is known to have a tumour, three questions matter most:

1. **Where is the cancer?** → *organ / cancer type*
2. **How advanced is it?** → *stage (I–IV)*
3. **What is the outlook?** → *survival (Living / Deceased) + risk of death %*

This project trains a separate model for each question and chains them together, the same way information flows in a real clinical workflow:

```
Patient details ──► 🫁 Organ model ──► 📊 Stage model ──► ❤️ Survival model
                        │                   │                    │
                    "Breast"            "STAGE II"         "LIVING · 6% risk"
```

The predicted organ is fed into the stage model, and both organ and stage are fed into the survival model.

---

## ✨ Features

- **3 prediction tasks** — organ (23 classes), stage (4 classes), survival (2 classes)
- **6 ML models compared per task** — Logistic Regression, KNN, SVC, Decision Tree, Random Forest, XGBoost
- **Handles missing information** — any field can be left as `Unknown`
- **Smart staging** — brain tumours and sarcomas are reported as *Not staged* (they don't use the Stage I–IV system)
- **Risk score** — probability of death from the survival model
- **Web interface** — enter patient details in a form and get all three predictions instantly

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
| `t_stage`, `n_stage`, `m_stage` | TNM staging — Tumour size, lymph Nodes, Metastasis |
| `radiation_therapy` | Whether the patient received radiation |
| `new_tumor_event` | Whether the cancer returned after treatment |
| `survival_status` | LIVING / DECEASED |
| `survival_months` | Months of follow-up |
| `cancer_type`, `cancer_name` | TCGA cancer code and full name |
| `organ` | Grouped organ (Brain, Lung, Breast, …) |

---

## ⚙️ Methodology

### 1. Data cleaning
- **Numerical (`age`)** — skewness checked (−0.44); missing values filled with the **median**
- **Categorical** — columns with **< 5 %** missing filled with the **mode**; columns with **> 5 %** missing filled with **`Unknown`** (so the model can learn from missingness, e.g. unstaged brain tumours)
- **Target columns** are never filled — rows with a missing target are dropped for that task

### 2. Outlier analysis
- IQR method on `age` flagged ~100 patients under 24
- **Kept**, because they are genuine young patients with cancers common at younger ages (brain, thyroid, testicular), not data errors

### 3. Feature selection (per task)

| Task | Target | Inputs | Removed (to avoid leakage) |
|---|---|---|---|
| Organ | `organ` | age, sex, histology, stage, TNM, therapy, recurrence | cancer type/name, survival |
| Stage | `stage` (I–IV) | age, sex, histology, therapy, recurrence, **organ** | `tumor_stage`, T, N, M |
| Survival | `survival_status` | age, sex, histology, stage, TNM, therapy, recurrence, **organ** | `survival_months` |

`race` and `ethnicity` were dropped because they did not improve accuracy.

### 4. Preprocessing
- `StandardScaler` for `age`
- `OneHotEncoder(handle_unknown='ignore')` for categorical features
- `LabelEncoder` for the target
- **Stratified 80 / 20** train–test split; preprocessing fitted on training data only

### 5. Model training
Six classifiers trained and compared on each task; the best model per task was saved with `joblib`.

---

## 📊 Results

| Task | Best model | Accuracy | Baseline* |
|---|---|---|---|
| Organ (23 classes) | Random Forest | **91 %** (macro F1 0.89) | 11 % |
| Stage (4 classes) | Logistic Regression | **~52 %** | 32 % |
| Survival (2 classes) | Logistic Regression | **~80 %** | 71 % |

<sub>*Baseline = always predicting the most common class.</sub>

**Key observations**
- **Organ** prediction is strong. Weakest classes: Skin (recall 0.50) and Stomach (0.72), mostly due to few training samples.
- **Stage** is hardest, because stage is defined by T/N/M, which must be excluded to avoid leakage. Most errors are between neighbouring stages (e.g. II vs III).
- **Survival** is imbalanced (71 % Living), so **recall for DECEASED** is reported alongside accuracy.

---

## 🗂️ Project Structure

```
ML_Lab_Project/
├── app.py                         # Flask web app
├── datasets/
│   └── cancer_patients_tabular.csv
├── scripts/
│   ├── ML_project.ipynb           # data cleaning, EDA, training
│   ├── organ_model.pkl            # ┐
│   ├── organ_preprocessor.pkl     # │ organ
│   ├── organ_label_encoder.pkl    # ┘
│   ├── stage_model.pkl            # ┐
│   ├── stage_preprocessor.pkl     # │ stage
│   ├── stage_label_encoder.pkl    # ┘
│   ├── survival_model.pkl         # ┐
│   ├── survival_preprocessor.pkl  # │ survival
│   └── survival_label_encoder.pkl # ┘
├── templates/
│   └── index.html                 # web page
└── styles/
    └── style.css                  # styling
```

Each task uses **three files**: the *preprocessor* converts inputs to numbers, the *model* predicts, and the *label encoder* converts the prediction back to a readable label.

---

## 🚀 Getting Started

### 1. Clone and enter the project
```bash
git clone <your-repo-url>
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
pip install flask pandas numpy scikit-learn xgboost joblib matplotlib seaborn
```

> Use the **same scikit-learn version** that was used to train the models, otherwise the `.pkl` files may fail to load.

### 4. Run the web app
```bash
python app.py
```
Open **http://127.0.0.1:5000** in your browser.

> Run the app from the `ML_Lab_Project` folder and open it through Flask (port **5000**), not by opening `index.html` directly or via Live Server.

---

## 🧪 Example Test Cases

| Field | Breast | Brain | Colorectal |
|---|---|---|---|
| Age | 52 | 58 | 65 |
| Gender | Female | Female | Male |
| Histology Code | 8500/3 | 9401/3 | 8140/3 |
| Tumor stage | STAGE IIA | Unknown | STAGE IV |
| T / N / M | T2 / N0 / M0 | Unknown | T3 / N1 / M1 |
| Radiation Therapy | Yes | Yes | No |
| Cancer Returned | No | Yes | No |
| **Expected organ** | Breast | Brain | Colorectal |
| **Expected stage** | Stage II | Not staged | Stage IV |
| **Expected survival** | Living (low risk) | Deceased (high risk) | Living (medium risk) |

---

## ⚠️ Limitations

- **All patients have cancer** — the system identifies *which* cancer, not *whether* a person has cancer.
- **`histology_code` is strongly linked to organ** — it raises organ accuracy from ~73 % to ~91 %.
- **Stage prediction without TNM data is inherently limited.**
- **Bone cancer is not truly represented** — the "Bone & Soft Tissue" class is TCGA sarcoma (mostly soft tissue).
- **Data comes mainly from US hospitals**, so performance on other populations (e.g. Indian patients) is untested.
- Some patients may have short follow-up, so a *Living* label is not a guarantee of long-term survival.

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, pandas, NumPy, scikit-learn, XGBoost |
| Visualisation | Matplotlib, Seaborn |
| Model storage | joblib |
| Web app | Flask, HTML, CSS (Jinja2 templates) |

---

## 📚 References

- Kefeli, J. & Tatonetti, N. (2024). *TCGA-Reports: A machine-readable pathology report resource for benchmarking text-based AI models.* Patterns.
- The Cancer Genome Atlas (TCGA) PanCancer Atlas clinical data, via cBioPortal.

---

## 👥 Authors

- *Pravesh SHrivastava*
- *Aniket Nawkar*

*Made as part of the Machine Learning Lab project.*