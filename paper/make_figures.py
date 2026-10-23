"""Figures and LaTeX tables for the paper, built only from paper/results/.  Run: python paper/make_figures.py"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, roc_auc_score

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL = 3.45  # IEEE column width (in)
ORDER = ["Majority", "LR", "KNN", "SVM", "DT", "RF", "XGB", "LGBM", "CatBoost"]
NAMES = {"Majority": "Majority class", "LR": "Logistic regression", "KNN": "$k$-nearest neighbours",
         "SVM": "SVM (RBF)", "DT": "Decision tree", "RF": "Random forest", "XGB": "XGBoost",
         "LGBM": "LightGBM", "CatBoost": "CatBoost$^{\dagger}$"}


def corrected_ttest(a, b, n_train, n_test):
    """Nadeau & Bengio corrected resampled t-test (two-sided)."""
    d = np.asarray(a) - np.asarray(b)
    J = len(d)
    var = np.var(d, ddof=1) * (1 / J + n_test / n_train)
    t = d.mean() / np.sqrt(var) if var > 0 else np.inf
    return t, 2 * stats.t.sf(abs(t), J - 1)


def ms(x, pct=True):
    x = np.asarray(x) * (100 if pct else 1)
    return f"{x.mean():.1f}\\,$\\pm$\\,{x.std(ddof=1):.1f}" if pct else f"{x.mean():.3f}\\,$\\pm$\\,{x.std(ddof=1):.3f}"


summary = {}
cv = pd.read_csv(os.path.join(RES, "cv_folds.csv"))

# ---------------------------------------------------------------- Table: main comparison
lines = []
best = {}
for task in ["organ", "stage", "survival"]:
    t = cv[cv.task == task]
    means = t[t.model != "Majority"].groupby("model")["f1"].mean()
    best[task] = means.idxmax()
    summary[task] = {}
    for m in ORDER:
        r = t[t.model == m].sort_values("fold")
        summary[task][m] = {k: float(r[k].mean()) for k in ["acc", "bacc", "f1", "auc", "rec_dec", "brier"]
                            if k in r and r[k].notna().any()}
        summary[task][m].update({k + "_sd": float(r[k].std(ddof=1)) for k in ["acc", "bacc", "f1", "auc"]
                                 if k in r and r[k].notna().any()})
        summary[task][m]["params"] = r["params"].value_counts().index[0]
    # significance vs best
    b = t[t.model == best[task]].sort_values("fold")
    for m in ORDER:
        if m in ("Majority", best[task]):
            continue
        o = t[t.model == m].sort_values("fold")
        _, p = corrected_ttest(b["f1"].values, o["f1"].values, b["n_train"].mean(), b["n_test"].mean())
        summary[task][m]["p_vs_best"] = float(p)

for m in ORDER:
    cells = [NAMES[m]]
    for task in ["organ", "stage", "survival"]:
        s = summary[task][m]
        r = cv[(cv.task == task) & (cv.model == m)]
        f1 = ms(r["f1"])
        if m == best[task]:
            f1 = "\\textbf{" + f1 + "}"
        elif m != "Majority" and s.get("p_vs_best", 1) < 0.05:
            f1 += "$^{*}$"
        cells += [ms(r["acc"]), f1]
        if task == "survival":
            cells += ["--" if m == "Majority" else ms(r["auc"], pct=False).replace("0.", ".")]
    lines.append(" & ".join(cells) + " \\\\")
    if m == "Majority":
        lines.append("\\midrule")
open(os.path.join(FIG, "table_main.tex"), "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: ablation
ab = pd.read_csv(os.path.join(RES, "ablation.csv"))
order = list(dict.fromkeys(ab.setting))
full = ab[ab.setting == "All features"].sort_values("fold")
lines, summary["ablation"] = [], {}
for s in order:
    r = ab[ab.setting == s].sort_values("fold")
    d = (r["f1"].values - full["f1"].values) * 100
    summary["ablation"][s] = dict(acc=float(r.acc.mean()), f1=float(r.f1.mean()), dF1=float(d.mean()))
    delta = "--" if s == "All features" else f"{d.mean():+.1f}".replace("-", "$-$")
    label = {"All features": "All routine variables", "w/o histology code": "\\quad $-$ morphology code",
             "w/o sex": "\\quad $-$ sex", "w/o stage & TNM": "\\quad $-$ stage and TNM",
             "w/o radiation & recurrence": "\\quad $-$ radiation, recurrence",
             "Histology code only": "Morphology code only", "Age + sex only": "Age and sex only"}[s]
    lines.append(f"{label} & {r['acc'].mean()*100:.1f} & {ms(r['f1'])} & {delta} \\\\")
open(os.path.join(FIG, "table_ablation.tex"), "w").write("\n".join(lines))

# ---------------------------------------------------------------- Table: cascade
cs = pd.read_csv(os.path.join(RES, "cascade.csv"))
lines, summary["cascade"] = [], {}
settings = list(dict.fromkeys(cs.setting))
short = {"No organ input": "(i) No organ feature",
         "Oracle organ (true label)": "(ii) True organ (oracle)",
         "Cascade: trained on true, tested on predicted": "(iii) Cascade, naive",
         "Cascade: trained and tested on predicted": "(iv) Cascade, OOF-consistent"}
for s in settings:
    st = cs[(cs.task == "stage") & (cs.setting == s)]
    sv = cs[(cs.task == "survival") & (cs.setting == s)]
    summary["cascade"][s] = dict(stage_f1=float(st.f1.mean()), stage_acc=float(st.acc.mean()),
                                 surv_f1=float(sv.f1.mean()), surv_auc=float(sv.auc.mean()),
                                 surv_bacc=float(sv.bacc.mean()))
    lines.append(f"{short[s]} & {st['acc'].mean()*100:.1f} & {st['f1'].mean()*100:.1f} & "
                 f"{sv['f1'].mean()*100:.1f} & {sv['auc'].mean():.3f} \\\\")
open(os.path.join(FIG, "table_cascade.tex"), "w").write("\n".join(lines))

# ---------------------------------------------------------------- survival class weighting
cw = pd.read_csv(os.path.join(RES, "survival_cw.csv"), keep_default_na=False)
summary["survival_cw"] = {c: {k: float(g[k].mean()) for k in ["acc", "bacc", "f1", "auc", "rec_dec", "brier"]}
                          for c, g in cw.groupby("class_weight")}

# ---------------------------------------------------------------- Fig: stage confusion matrix
z = np.load(os.path.join(RES, "oof_stage.npz"), allow_pickle=True)
y, cls = z["y"], [c.replace("STAGE ", "") for c in z["classes"]]
p = z[f"pred_{best['stage']}"]
cm = confusion_matrix(y, p, normalize="true")
cm_raw = confusion_matrix(y, p)
err = cm_raw.sum() - np.trace(cm_raw)
adj = sum(cm_raw[i, j] for i in range(4) for j in range(4) if abs(i - j) == 1)
summary["stage_adjacent_error_share"] = float(adj / err)
summary["stage_within_one"] = float((np.trace(cm_raw) + adj) / cm_raw.sum())
fig, ax = plt.subplots(figsize=(COL * 0.62, COL * 0.55))
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > 0.5 else INK)
ax.set_xticks(range(4), cls); ax.set_yticks(range(4), cls)
ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage")
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
fig.savefig(os.path.join(FIG, "stage_cm.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig: organ per-class F1
z = np.load(os.path.join(RES, "oof_organ.npz"), allow_pickle=True)
y, cls = z["y"], list(z["classes"])
p = z[f"pred_{best['organ']}"]
f1c = f1_score(y, p, average=None)
counts = np.bincount(y)
order_ = np.argsort(f1c)
summary["organ_per_class"] = {cls[i]: dict(f1=float(f1c[i]), n=int(counts[i])) for i in range(len(cls))}
fig, ax = plt.subplots(figsize=(COL, 2.75))
ax.barh(range(len(cls)), f1c[order_], color=BLUE, height=0.7)
ax.set_yticks(range(len(cls)), [f"{cls[i]} ({counts[i]})" for i in order_], fontsize=6.3)
for k, i in enumerate(order_):
    ax.text(f1c[i] + 0.01, k, f"{f1c[i]:.2f}", va="center", fontsize=5.8, color=INK2)
ax.set_xlim(0, 1.1); ax.set_xlabel("Out-of-fold F1-score")
ax.grid(axis="y", visible=False); ax.tick_params(length=0)
fig.savefig(os.path.join(FIG, "organ_f1.pdf")); plt.close(fig)

# ---------------------------------------------------------------- Fig: survival ROC + calibration
z = np.load(os.path.join(RES, "oof_survival.npz"), allow_pickle=True)
yd = (z["y"] == list(z["classes"]).index("DECEASED")).astype(int)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(COL, 1.75))
curves = [(best["survival"], ORANGE, "--")] + [(m, c, l) for m, c, l in
                                              [("LR", BLUE, "-"), ("DT", GRAY, ":")] if m != best["survival"]]
for m, col, ls in curves:
    sc = z[f"score_{m}"]
    fpr, tpr, _ = roc_curve(yd, sc)
    a1.plot(fpr, tpr, color=col, lw=1.2, ls=ls, label=f"{m} ({roc_auc_score(yd, sc):.3f})")
a1.plot([0, 1], [0, 1], color=GRID, lw=0.8)
a1.set_xlabel("False-positive rate"); a1.set_ylabel("True-positive rate")
a1.legend(frameon=False, loc="lower right", fontsize=6, handlelength=1.6)
a1.set_title("(a) ROC, DECEASED class", loc="left")
sp = np.load(os.path.join(RES, "surv_prob.npz"))
for key, col, lab in [("None", BLUE, "LR"), ("balanced", ORANGE, "LR, weighted")]:
    fp, mp = calibration_curve(sp["y"], sp[key], n_bins=10, strategy="quantile")
    a2.plot(mp, fp, marker="o", ms=2.5, lw=1.1, color=col, label=lab)
a2.plot([0, 1], [0, 1], color=GRID, lw=0.8)
a2.set_xlabel("Predicted risk"); a2.set_ylabel("Observed frequency")
a2.legend(frameon=False, loc="upper left", fontsize=6, handlelength=1.6)
a2.set_title("(b) Calibration", loc="left")
fig.tight_layout(w_pad=1.2)
fig.savefig(os.path.join(FIG, "survival_roc_cal.pdf")); plt.close(fig)

json.dump({"best": best, **summary}, open(os.path.join(RES, "summary.json"), "w"), indent=1, default=str)
print(json.dumps({"best": best}, indent=1))
