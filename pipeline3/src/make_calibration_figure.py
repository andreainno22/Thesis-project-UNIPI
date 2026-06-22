"""Calibration-not-discrimination figure.

The score distributions of normal / shadow / obstructed are essentially unchanged
by augmentation; only the per-camera threshold moves. We draw the (shared)
distributions once and overlay the two mean thresholds: without augmentation the
threshold falls BELOW the shadow band (shadows become false positives), with
augmentation it rises ABOVE the shadow band but still below the obstructions.
The obstruction-vs-shadow gap (discrimination, per-camera AUROC 0.996) is identical
in both cases.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from scipy.stats import gaussian_kde
    HAVE_KDE = True
except Exception:
    HAVE_KDE = False

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "pipeline3" / "results"
OUT = ROOT / "template_tesi" / "images" / "patchCore"

C = {"normal": "#2ca02c", "shadow_normal": "#ff7f0e", "obstructed": "#d62728"}
LBL = {"normal": "Normal", "shadow_normal": "Shadow (normal)", "obstructed": "Obstructed"}


def auroc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    r = pd.Series(np.concatenate([pos, neg])).rank().to_numpy()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def per_camera_auroc(df):
    vals = []
    for _, g in df.groupby("reference_id"):
        o = g.loc[g.test_type == "obstructed", "score"].to_numpy()
        s = g.loc[g.test_type == "shadow_normal", "score"].to_numpy()
        if len(o) and len(s):
            vals.append(auroc(o, s))
    return float(np.nanmean(vals))


def kde(v, grid):
    v = np.asarray(v, float)
    if HAVE_KDE and v.std() > 1e-6:
        return gaussian_kde(v)(grid)
    h, e = np.histogram(v, bins=40, density=True)
    return np.interp(grid, 0.5 * (e[:-1] + e[1:]), h, left=0, right=0)


noaug = pd.read_csv(RESULTS / "results_p3_senza_gestione_ombre.csv")
aug = pd.read_csv(RESULTS / "results_p3_pooled.csv")

thr_noaug = noaug["threshold"].mean()
thr_aug = aug["threshold"].mean()
pc = per_camera_auroc(aug)
fpr_noaug = (noaug[noaug.test_type == "shadow_normal"]["score"]
             >= noaug[noaug.test_type == "shadow_normal"]["threshold"]).mean()
fpr_aug = (aug[aug.test_type == "shadow_normal"]["score"]
           >= aug[aug.test_type == "shadow_normal"]["threshold"]).mean()
print(f"thr no-aug={thr_noaug:.3f}  thr aug={thr_aug:.3f}  per-cam AUROC={pc:.3f}  "
      f"FPR {fpr_noaug:.0%}->{fpr_aug:.0%}")

# Distributions are config-invariant; use the augmented run for the curves.
allv = aug["score"].to_numpy()
lo, hi = np.percentile(allv, 0.3), np.percentile(allv, 99.7)
grid = np.linspace(lo, hi, 600)

fig, ax = plt.subplots(figsize=(6.6, 3.6))
for tt in ["normal", "shadow_normal", "obstructed"]:
    y = kde(aug.loc[aug.test_type == tt, "score"].to_numpy(), grid)
    ax.fill_between(grid, y, color=C[tt], alpha=0.25)
    ax.plot(grid, y, color=C[tt], lw=1.8, label=LBL[tt])

ymax = ax.get_ylim()[1]
ax.axvline(thr_noaug, color="black", ls=":", lw=1.5)
ax.axvline(thr_aug, color="black", ls="--", lw=1.5)
ax.text(thr_noaug, ymax * 0.78, f" threshold,\n no aug.\n (FPR {fpr_noaug:.0%})",
        ha="left", va="top", fontsize=8.5)
ax.text(thr_aug, ymax * 0.78, f" threshold,\n with aug.\n (FPR {fpr_aug:.0%})",
        ha="left", va="top", fontsize=8.5)

ax.set_xlabel("Raw anomaly score")
ax.set_ylabel("Density")
ax.set_xlim(lo, hi)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", fontsize=9, frameon=False)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"p3_calibration_shift.{ext}", dpi=150, bbox_inches="tight")
print("saved p3_calibration_shift.pdf/.png")
