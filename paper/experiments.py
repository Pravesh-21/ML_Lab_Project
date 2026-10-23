"""Experiments for the paper. Run from the project root:  python paper/experiments.py

Outputs (paper/results/):
  cv_folds.csv          per-outer-fold metrics for every task x model (nested CV)
  ablation.csv          organ-model feature-group ablation
  cascade.csv           stage / survival under no / oracle / cascaded organ input
  oof_*.npz             out-of-fold predictions (first repeat) for figures

  python paper/experiments.py nested LGBM CatBoost   re-runs only the named models and merges them in
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss, f1_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from common import DATA, STAGING, load_task, native_preprocessor, preprocessor

warnings.filterwarnings("ignore")
OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)
SEED = 42
K, R = 5, 3

MODELS = {
    "LR": (LogisticRegression(max_iter=3000), {"m__C": [0.1, 1, 10]}),
    "KNN": (KNeighborsClassifier(), {"m__n_neighbors": [5, 15, 31], "m__weights": ["uniform", "distance"]}),
    "SVM": (SVC(probability=False, random_state=SEED), {"m__C": [1, 10]}),
    "DT": (DecisionTreeClassifier(random_state=SEED), {"m__max_depth": [None, 10, 20],
                                                      "m__min_samples_leaf": [1, 5]}),
    "RF": (RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
           {"m__min_samples_leaf": [1, 3], "m__max_features": ["sqrt", 0.2]}),
    "XGB": (XGBClassifier(n_estimators=200, tree_method="hist", random_state=SEED, n_jobs=1),
            {"m__max_depth": [3, 6], "m__learning_rate": [0.1, 0.3]}),
    "LGBM": (None, {"m__num_leaves": [15, 31], "m__learning_rate": [0.05, 0.1]}),
    "CatBoost": (None, None),       # fixed hyperparameters (see catboost_pipeline)
}


def lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=SEED, n_jobs=1, verbose=-1)


def catboost_pipeline(task, cols):
    """Configurations of the project's saved CatBoost models (models/*.pkl).

    Organ (23 classes) uses one-hot inputs, because CatBoost's native per-class target statistics are
    prohibitively slow with 23 classes; stage and survival use native categorical handling.
    """
    from catboost import CatBoostClassifier
    if task == "organ":
        m = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=6, random_seed=SEED, verbose=0,
                               thread_count=-1)
        return Pipeline([("pre", preprocessor(cols)), ("m", m)])
    m = CatBoostClassifier(iterations=800, learning_rate=0.05, depth=6, random_seed=SEED, verbose=0,
                           thread_count=-1, cat_features=list(range(1, len(cols))))
    return Pipeline([("pre", native_preprocessor(cols)), ("m", m)])


class Flat(Pipeline):
    """CatBoost returns predictions as a column vector; flatten them like every other model."""
    def predict(self, X, **kw):
        return np.ravel(super().predict(X, **kw)).astype(int)


def scores(task, y_true, y_pred, score=None, pos=None):
    out = dict(acc=accuracy_score(y_true, y_pred), bacc=balanced_accuracy_score(y_true, y_pred),
               f1=f1_score(y_true, y_pred, average="macro"))
    if task == "survival":
        out["rec_dec"] = recall_score(y_true, y_pred, pos_label=pos)
        if score is not None:
            out["auc"] = roc_auc_score(y_true == pos, score)
    return out


def pos_score(model, X, pos):
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, pos], True
        except AttributeError:
            pass
    return model.decision_function(X) * (1 if pos == 1 else -1), False


def nested_cv(only=None):
    path = os.path.join(OUT, "cv_folds.csv")
    rows = []
    if only and os.path.exists(path):
        old = pd.read_csv(path)
        rows = old[~old.model.isin(only)].to_dict("records")
    for task in ["organ", "stage", "survival"]:
        X, y_raw, _ = load_task(task)
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)
        pos = list(le.classes_).index("DECEASED") if task == "survival" else None
        outer = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED)
        oof = {}
        oof_path = os.path.join(OUT, f"oof_{task}.npz")
        if only and os.path.exists(oof_path):
            z = np.load(oof_path, allow_pickle=True)
            for k in z.files:
                if k.startswith("pred_"):
                    oof[k[5:]] = (z[k], z["score_" + k[5:]])
        todo = {"Majority": (DummyClassifier(strategy="most_frequent"), None), **MODELS}
        if only:
            todo = {k: v for k, v in todo.items() if k in only}
        for name, (est, grid) in todo.items():
            t0 = time.time()
            pred1 = np.empty(len(y), dtype=int)
            score1 = np.full(len(y), np.nan)
            for f, (tr, te) in enumerate(outer.split(X, y)):
                if name == "CatBoost":
                    pipe = Flat(catboost_pipeline(task, list(X.columns)).steps)
                else:
                    pipe = Pipeline([("pre", preprocessor(list(X.columns))),
                                     ("m", lgbm() if name == "LGBM" else clone(est))])
                if grid:
                    gs = GridSearchCV(pipe, grid, scoring="f1_macro", n_jobs=-1,
                                      cv=StratifiedKFold(3, shuffle=True, random_state=SEED))
                    gs.fit(X.iloc[tr], y[tr])
                    model, best = gs.best_estimator_, gs.best_params_
                else:
                    model, best = pipe.fit(X.iloc[tr], y[tr]), {}
                yp = model.predict(X.iloc[te])
                sc, is_prob = (pos_score(model, X.iloc[te], pos) if task == "survival" else (None, False))
                r = dict(task=task, model=name, fold=f, n_test=len(te), n_train=len(tr),
                         params=json.dumps({k[3:]: v for k, v in best.items()}),
                         **scores(task, y[te], yp, sc, pos))
                if task == "survival" and is_prob:
                    r["brier"] = brier_score_loss(y[te] == pos, sc)
                rows.append(r)
                if f < K:                                   # first repeat -> out-of-fold predictions
                    pred1[te] = yp
                    if sc is not None:
                        score1[te] = sc
            oof[name] = (pred1, score1)
            print(f"{task:9s} {name:8s} {time.time()-t0:6.1f}s  "
                  f"f1={np.mean([r['f1'] for r in rows if r['task']==task and r['model']==name]):.3f}", flush=True)
        np.savez(os.path.join(OUT, f"oof_{task}.npz"), y=y, classes=le.classes_,
                 **{f"pred_{k}": v[0] for k, v in oof.items()}, **{f"score_{k}": v[1] for k, v in oof.items()})
        pd.DataFrame(rows).to_csv(path, index=False)


def lr_tuned():
    return GridSearchCV(None, {"m__C": [0.1, 1, 10]}, scoring="f1_macro", n_jobs=-1,
                        cv=StratifiedKFold(3, shuffle=True, random_state=SEED))


def fit_lr(X, y, **lr_kw):
    gs = lr_tuned()
    gs.estimator = Pipeline([("pre", preprocessor(list(X.columns))),
                             ("m", LogisticRegression(max_iter=3000, **lr_kw))])
    return gs.fit(X, y).best_estimator_


def ablation():
    X, y_raw, _ = load_task("organ")
    y = LabelEncoder().fit_transform(y_raw)
    groups = {
        "All features": [],
        "w/o histology code": ["histology_code"],
        "w/o sex": ["sex"],
        "w/o stage & TNM": STAGING,
        "w/o radiation & recurrence": ["radiation_therapy", "new_tumor_event"],
        "Histology code only": [c for c in X.columns if c != "histology_code"],
        "Age + sex only": [c for c in X.columns if c not in ("age", "sex")],
    }
    rows = []
    outer = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED)
    for label, drop in groups.items():
        Xa = X.drop(columns=drop)
        for f, (tr, te) in enumerate(outer.split(Xa, y)):
            m = fit_lr(Xa.iloc[tr], y[tr])
            rows.append(dict(setting=label, fold=f, **scores("organ", y[te], m.predict(Xa.iloc[te]))))
        print("ablation", label, np.mean([r["f1"] for r in rows if r["setting"] == label]).round(3), flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "ablation.csv"), index=False)


def cascade():
    """Effect of feeding the stage / survival models a *predicted* organ instead of the true one."""
    df = pd.read_csv(DATA)
    Xo, yo_raw, _ = load_task("organ", df)
    le_o = LabelEncoder().fit(yo_raw)
    organ_model = lr_tuned()
    organ_model.estimator = Pipeline([("pre", preprocessor(list(Xo.columns))),
                                      ("m", LogisticRegression(max_iter=3000))])
    oof = cross_val_predict(organ_model, Xo, le_o.transform(yo_raw),
                            cv=StratifiedKFold(K, shuffle=True, random_state=SEED + 1))
    df["organ_pred"] = le_o.inverse_transform(oof)
    print("cascade: OOF organ accuracy", (df["organ_pred"] == df["organ"]).mean().round(4), flush=True)

    rows = []
    for task in ["stage", "survival"]:
        X, y_raw, d = load_task(task, df)
        le = LabelEncoder().fit(y_raw)
        y = le.transform(y_raw)
        pos = list(le.classes_).index("DECEASED") if task == "survival" else None
        Xp = X.copy()
        Xp["organ"] = d["organ_pred"].values                 # cascaded input
        X0 = X.drop(columns=["organ"])
        outer = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED)
        for f, (tr, te) in enumerate(outer.split(X, y)):
            settings = {}
            m0 = fit_lr(X0.iloc[tr], y[tr])
            settings["No organ input"] = (m0, X0)
            mt = fit_lr(X.iloc[tr], y[tr])
            settings["Oracle organ (true label)"] = (mt, X)
            settings["Cascade: trained on true, tested on predicted"] = (mt, Xp)
            mp = fit_lr(Xp.iloc[tr], y[tr])
            settings["Cascade: trained and tested on predicted"] = (mp, Xp)
            for label, (m, XX) in settings.items():
                sc = pos_score(m, XX.iloc[te], pos)[0] if task == "survival" else None
                rows.append(dict(task=task, setting=label, fold=f,
                                 **scores(task, y[te], m.predict(XX.iloc[te]), sc, pos)))
        print("cascade", task, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cascade.csv"), index=False)


def survival_extras():
    """Class-weighting trade-off and calibration data for the deployed survival LR."""
    X, y_raw, _ = load_task("survival")
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    pos = list(le.classes_).index("DECEASED")
    rows, prob = [], {}
    for cw in [None, "balanced"]:
        p_all = np.empty(len(y))
        outer = RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED)
        for f, (tr, te) in enumerate(outer.split(X, y)):
            m = fit_lr(X.iloc[tr], y[tr], class_weight=cw)
            p = m.predict_proba(X.iloc[te])[:, pos]
            rows.append(dict(class_weight=str(cw), fold=f, brier=brier_score_loss(y[te] == pos, p),
                             **scores("survival", y[te], m.predict(X.iloc[te]), p, pos)))
            if f < K:
                p_all[te] = p
        prob[str(cw)] = p_all
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "survival_cw.csv"), index=False)
    np.savez(os.path.join(OUT, "surv_prob.npz"), y=(y == pos).astype(int), **prob)
    print("survival extras done", flush=True)


if __name__ == "__main__":
    t = time.time()
    steps = sys.argv[1:] or ["nested", "ablation", "cascade", "survival"]
    if "nested" in steps: nested_cv([a for a in sys.argv[2:]] or None)
    if "ablation" in steps: ablation()
    if "cascade" in steps: cascade()
    if "survival" in steps: survival_extras()
    print(f"ALL DONE in {time.time()-t:.0f}s")
