"""How much do target-defining fields inflate performance?  Run: python paper/leakage_demo.py

LR (the paper's main learner) and LightGBM with the configuration of models/stage_lightgbm_tnm.pkl
are each evaluated with and without the leaking fields, over the same 5x3 repeated stratified folds.
"""
import os, sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
sys.path.insert(0, os.path.dirname(__file__))
from common import load_task, preprocessor, STAGING
warnings.filterwarnings("ignore")


def learners():
    out = {"LR": lambda: LogisticRegression(max_iter=3000)}
    try:
        from lightgbm import LGBMClassifier
        out["LGBM"] = lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42,
                                             n_jobs=4, verbose=-1)
    except ImportError:
        print("lightgbm not installed: LR only")
    return out


rows = []
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
for task, leak in [("stage", STAGING[1:]), ("survival", ["survival_months"])]:
    X, y_raw, d = load_task(task)
    y = LabelEncoder().fit_transform(y_raw)
    XL = pd.concat([X, d[leak]], axis=1)
    if task == "survival":                       # follow-up time as a categorical decile, like other fields
        XL["survival_months"] = pd.qcut(XL["survival_months"], 10, duplicates="drop").astype(str)
    for name, make in learners().items():
        for label, XX in [("leakage-free", X), ("with leaking fields", XL)]:
            f1s, aucs = [], []
            for tr, te in cv.split(XX, y):
                m = Pipeline([("pre", preprocessor(list(XX.columns))), ("m", make())]).fit(XX.iloc[tr], y[tr])
                f1s.append(f1_score(y[te], np.ravel(m.predict(XX.iloc[te])), average="macro"))
                if task == "survival":
                    aucs.append(roc_auc_score(y[te] == 0, m.predict_proba(XX.iloc[te])[:, 0]))
            rows.append(dict(task=task, model=name, setting=label, f1=np.mean(f1s), f1_sd=np.std(f1s, ddof=1),
                             auc=np.mean(aucs) if aucs else np.nan))
            print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv(os.path.join(os.path.dirname(__file__), "results", "leakage_demo.csv"), index=False)
