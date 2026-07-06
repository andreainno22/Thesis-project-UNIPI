"""Phase 2 (real test set D2) figures for the P3 results chapter.

Two figures into template_tesi/images/patchCore/:
  - p3_d2_metrics.{pdf,png}      : grouped bars of TPR/FPR/Precision/F1/Accuracy at k=4 vs k=3
  - p3_d2_scores.{pdf,png}       : per-sample normalised scores (obstructed vs hard negative)
                                   at k=3, with the decision threshold and the misses highlighted

Run: conda run -n tesi_env python pipeline3/src/make_phase2_figures.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "pipeline3" / "results"
OUT = ROOT / "template_tesi" / "images" / "patchCore"

C_K4 = "#9e9e9e"   # grey
C_K3 = "#1f77b4"   # blue
C_OK = "#2ca02c"   # green (correct)
C_ERR = "#d62728"  # red (error)

POS = "obstructed"
NEG = "bg_normal"   # hard negatives


def metrics(df):
    d = df[df.test_type.isin([POS, NEG])]
    tp = int(((d.test_type == POS) & (d.is_anomaly == 1)).sum())
    fn = int(((d.test_type == POS) & (d.is_anomaly == 0)).sum())
    fp = int(((d.test_type == NEG) & (d.is_anomaly == 1)).sum())
    tn = int(((d.test_type == NEG) & (d.is_anomaly == 0)).sum())
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if (prec + tpr) else 0.0
    acc = (tp + tn) / (tp + fn + fp + tn)
    return dict(TP=tp, FN=fn, FP=fp, TN=tn, TPR=tpr, FPR=fpr, Precision=prec, F1=f1, Accuracy=acc)


k4 = pd.read_csv(RESULTS / "results_poli_completo.csv")             # k=4
k3 = pd.read_csv(RESULTS / "results_poli_completo_overlap_10_k_3.csv")  # k=3
m4, m3 = metrics(k4), metrics(k3)
print("k=4:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m4.items()})
print("k=3:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m3.items()})

# ---- Figure 1: grouped bar of metrics --------------------------------------
names = ["Recall (TPR)", "FPR", "Precision", "F1", "Accuracy"]
keys = ["TPR", "FPR", "Precision", "F1", "Accuracy"]
v4 = [m4[k] for k in keys]
v3 = [m3[k] for k in keys]
x = np.arange(len(names)); w = 0.38

fig, ax = plt.subplots(figsize=(6.8, 3.6))
b4 = ax.bar(x - w / 2, v4, w, label="$k=4$", color=C_K4)
b3 = ax.bar(x + w / 2, v3, w, label="$k=3$ (chosen)", color=C_K3)
for bars in (b4, b3):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylim(0, 1.05)
ax.set_ylabel("Value")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"p3_d2_metrics.{ext}", dpi=150, bbox_inches="tight")
print("saved p3_d2_metrics.pdf/.png")

# ---- Figure 2: per-sample decision margin at k=3 ---------------------------
# Margin = score - per-camera threshold; the decision boundary is a single line
# at 0 (margin > 0 => flagged), which is exact for per-camera thresholds.
d = k3[k3.test_type.isin([POS, NEG])].copy()
d["margin"] = d["score"] - d["threshold"]
agree = ((d.margin >= 0).astype(int) == d.is_anomaly).mean()
print(f"k=3 margin-boundary check (pred==is_anomaly): {agree:.2%}")

rng = np.random.default_rng(0)
rows = {POS: 1.0, NEG: 0.0}
fig, ax = plt.subplots(figsize=(6.8, 3.0))
for tt in (POS, NEG):
    sub = d[d.test_type == tt]
    y = rows[tt] + (rng.random(len(sub)) - 0.5) * 0.32
    correct = (((sub.is_anomaly == 1) == (tt == POS))).values
    mx = sub["margin"].values
    ax.scatter(mx[correct], y[correct], s=28, color=C_OK,
               edgecolor="white", linewidth=0.4, zorder=3)
    ax.scatter(mx[~correct], y[~correct], s=48, color=C_ERR,
               edgecolor="white", linewidth=0.4, zorder=4, marker="X")
ax.axvline(0.0, color="black", ls="--", lw=1.3)
ax.text(0.0, 1.66, " decision boundary", fontsize=9, va="top")
gated = d[(d.test_type == POS) & (d.score <= 1e-9)]
if len(gated):
    gx = float(gated["margin"].mean())
    ax.annotate(f"obstruction gated to 0 ($n={len(gated)}$)",
                xy=(gx, 1.0), xytext=(gx + 0.3, 0.45), fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", lw=0.7))
ax.set_yticks([0, 1]); ax.set_yticklabels(["Hard negative", "Obstructed"])
ax.set_ylim(-0.5, 1.75)
ax.set_xlabel("Decision margin (score $-$ threshold)")
ax.scatter([], [], color=C_OK, label="correct")
ax.scatter([], [], color=C_ERR, marker="X", label="error (FN / FP)")
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.spines[["top", "right", "left"]].set_visible(False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"p3_d2_scores.{ext}", dpi=150, bbox_inches="tight")
print("saved p3_d2_scores.pdf/.png")
