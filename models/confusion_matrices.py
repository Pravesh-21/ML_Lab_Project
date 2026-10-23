"""Confusion matrices for every model in models/, on the notebook's own held-out test sets.

Run from the project root:  python models/confusion_matrices.py
Rebuilds the notebook's cleaning (median / mode / 'Unknown' filling) and its train_test_split calls
(test_size=0.2, random_state=42) so each model is scored on exactly the patients it never saw.
"""
import os
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "axes.titleweight": "semibold",
                     "axes.titlelocation": "left", "text.color": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2})

# ---------------------------------------------------------------- notebook cleaning
df = pd.read_csv(os.path.join(ROOT, "datasets", "cancer_patients_tabular.csv"))
df["age"] = df["age"].fillna(df["age"].median())
categorical_cols = ["sex", "race", "ethnicity", "histology_code", "tumor_stage", "t_stage", "n_stage",
                    "m_stage", "radiation_therapy", "new_tumor_event"]
for c in categorical_cols:
    if df[c].isnull().mean() * 100 < 5:
        df[c] = df[c].fillna(df[c].mode()[0])
    else:
        df[c] = df[c].fillna("Unknown")
DROP = ["patient_id", "survival_status", "survival_months", "cancer_type", "cancer_name", "ethnicity", "race"]

# ---------------------------------------------------------------- notebook test sets
X1 = df.drop(columns=DROP + ["organ"])
_, X1_test, _, y1_test = train_test_split(X1, df["organ"], test_size=0.2, random_state=42)

df2 = df[df["tumor_stage"] != "Unknown"].copy()
df2["stage"] = df2["tumor_stage"].str.upper().str.extract(r"(STAGE IV|STAGE III|STAGE II|STAGE I)")[0]
df2 = df2.dropna(subset=["stage"])
X2b = df2.drop(columns=DROP + ["tumor_stage", "stage"])          # T, N, M kept, as in the notebook
_, X2b_test, _, y2b_test = train_test_split(X2b, df2["stage"], test_size=0.2, random_state=42,
                                            stratify=df2["stage"])

df3 = df[df["survival_status"].notna() & (df["survival_status"] != "Unknown")].copy()
X3 = df3.drop(columns=DROP)
_, X3_test, _, y3_test = train_test_split(X3, df3["survival_status"], test_size=0.2, random_state=42,
                                          stratify=df3["survival_status"])

TESTS = {
    "organ_catboost": (X1_test, y1_test, "Organ — CatBoost"),
    "stage_lightgbm_tnm": (X2b_test, y2b_test, "Stage — LightGBM (uses T, N, M inputs)"),
    "survival_catboost": (X3_test, y3_test, "Survival — CatBoost"),
}


def predict(bundle, X):
    model = bundle["model"]
    if "columns" in bundle:
        X = X[bundle["columns"]]
    pred = np.ravel(model.predict(X)).astype(int)
    return np.asarray(bundle["labels"])[pred]


def plot(cm, labels, title, subtitle, path):
    n = len(labels)
    size = max(5.4, 0.42 * n + 1.8)
    fig, ax = plt.subplots(figsize=(size, size * 0.95))
    rowsum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, rowsum, out=np.zeros_like(cm, dtype=float), where=rowsum > 0)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, length=0, colors=INK2)
    fs = 10 if n <= 4 else 6.2
    for i in range(n):
        for j in range(n):
            if norm[i, j] >= 0.005:                       # hide cells that round to 0.00
                ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=fs,
                        color="white" if norm[i, j] > 0.5 else INK)
    short = [l.replace("STAGE ", "Stage ") for l in labels]
    ax.set_xticks(range(n), short, rotation=90 if n > 4 else 0)
    ax.set_yticks(range(n), short)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=11, pad=34)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5, color=INK2, va="bottom")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


for name, (X, y, title) in TESTS.items():
    path = os.path.join(HERE, f"{name}.pkl")
    if not os.path.exists(path):
        print(f"skip {name}: file not found")
        continue
    bundle = joblib.load(path)
    labels = list(bundle["labels"])
    pred = predict(bundle, X)
    cm = confusion_matrix(y, pred, labels=labels)
    acc, f1 = accuracy_score(y, pred), f1_score(y, pred, average="macro")
    sub = (f"Test set: {len(y):,} patients · accuracy {acc:.1%} · macro-F1 {f1:.3f}\n"
           f"Row-normalised: each row sums to 1, so the diagonal is the recall of each class")
    out = os.path.join(HERE, f"confusion_{name}.png")
    plot(cm, labels, title, sub, out)
    print(f"{name:20s} n={len(y):5d}  acc={acc:.4f}  macroF1={f1:.4f}  -> {os.path.basename(out)}")
