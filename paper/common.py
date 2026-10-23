"""Shared data loading and pipeline construction for the paper experiments."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = "datasets/cancer_patients_tabular.csv"
STAGING = ["tumor_stage", "t_stage", "n_stage", "m_stage"]
BASE = ["age", "sex", "histology_code"] + STAGING + ["radiation_therapy", "new_tumor_event"]

FEATURES = {
    "organ": BASE,
    "stage": ["age", "sex", "histology_code", "radiation_therapy", "new_tumor_event", "organ"],
    "survival": BASE + ["organ"],
}


def load_task(task, df=None):
    df = pd.read_csv(DATA) if df is None else df
    if task == "organ":
        d = df.copy()
        y = d["organ"]
    elif task == "stage":
        d = df.copy()
        d["stage"] = d["tumor_stage"].str.upper().str.extract(r"(STAGE IV|STAGE III|STAGE II|STAGE I)")[0]
        d = d.dropna(subset=["stage"])
        y = d["stage"]
    else:
        d = df.dropna(subset=["survival_status"]).copy()
        y = d["survival_status"]
    return d[FEATURES[task]], y, d


def preprocessor(cols):
    """Imputation lives inside the pipeline, so it is fitted on training folds only."""
    num = [c for c in cols if c == "age"]
    mode = [c for c in cols if c in ("sex", "histology_code")]          # < 5 % missing
    unk = [c for c in cols if c not in num + mode]                         # > 5 % missing -> 'Unknown'
    ohe = dict(handle_unknown="ignore")
    parts = []
    if num:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    if mode:
        parts.append(("mode", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                        ("oh", OneHotEncoder(**ohe))]), mode))
    if unk:
        parts.append(("unk", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Unknown")),
                                       ("oh", OneHotEncoder(**ohe))]), unk))
    return ColumnTransformer(parts)


def native_preprocessor(cols):
    """Imputation only, categories left as strings (for learners with native categorical support).

    Age comes first, so categorical columns are at indices 1..len(cols)-1.
    """
    mode = [c for c in cols if c in ("sex", "histology_code")]
    unk = [c for c in cols if c not in ["age"] + mode]
    return ColumnTransformer([("num", SimpleImputer(strategy="median"), ["age"]),
                              ("mode", SimpleImputer(strategy="most_frequent"), mode),
                              ("unk", SimpleImputer(strategy="constant", fill_value="Unknown"), unk)])
