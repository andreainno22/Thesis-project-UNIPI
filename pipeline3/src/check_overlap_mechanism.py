"""Why does the shadow/obstructed OVERLAP shrink with augmentation while the
PER-CAMERA discrimination (AUROC) stays flat?

Hypothesis: the aggregate (pooled-over-scenes) overlap is a CROSS-camera effect.
Within each scene obstr > shadow always; but scene A's shadow can sit above scene
B's obstruction, so the pooled histograms overlap. Augmentation pulls shadow
scores down to the normal level -> less cross-scene overlap -> lower FPR, while
the within-scene ranking (per-camera AUROC) is untouched.

We quantify both AUROCs (per-camera and global/pooled) on RAW scores, before
(no augmentation) and after (shadow aug + pooled sigma), and draw a before/after
raw-score distribution figure.
"""
from pathlib import Path
import numpy as np
import pandas as pd
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
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
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


def stats(df, tt):
    v = df.loc[df.test_type == tt, "score"]
    return v.mean(), v.std(), v.quantile(0.02), v.quantile(0.98)


def kde(v, grid):
    v = np.asarray(v, float)
    if HAVE_KDE and v.std() > 1e-6:
        return gaussian_kde(v)(grid)
    h, e = np.histogram(v, bins=40, density=True)
    return np.interp(grid, 0.5 * (e[:-1] + e[1:]), h, left=0, right=0)


configs = [
    ("Before: no augmentation", "results_p3_senza_gestione_ombre.csv"),
    ("After: shadow aug + pooled $\\sigma$", "results_p3_pooled.csv"),
]

dfs = []
print(f"{'config':28s} | per-cam AUROC | GLOBAL AUROC | shadow FPR | shadow mean | obstr mean")
for name, fn in configs:
    df = pd.read_csv(RESULTS / fn)
    dfs.append((name, df))
    pc = per_camera_auroc(df)
    gl = auroc(df.loc[df.test_type == "obstructed", "score"],
              df.loc[df.test_type == "shadow_normal", "score"])
    sh = df[df.test_type == "shadow_normal"]
    fpr = (sh["score"] >= sh["threshold"]).mean()
    sm = df.loc[df.test_type == "shadow_normal", "score"].mean()
    om = df.loc[df.test_type == "obstructed", "score"].mean()
    print(f"{name:28s} | {pc:12.4f} | {gl:11.4f} | {fpr:9.2%} | {sm:10.3f} | {om:.3f}")

# Common x-range across both panels (raw scores)
allv = np.concatenate([d["score"].to_numpy() for _, d in dfs])
lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
grid = np.linspace(lo, hi, 600)

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharex=True, sharey=True)
for ax, (name, df) in zip(axes, dfs):
    for tt in ["normal", "shadow_normal", "obstructed"]:
        v = df.loc[df.test_type == tt, "score"].to_numpy()
        y = kde(v, grid)
        ax.fill_between(grid, y, color=C[tt], alpha=0.25)
        ax.plot(grid, y, color=C[tt], lw=1.7, label=LBL[tt])
    pc = per_camera_auroc(df)
    sh = df[df.test_type == "shadow_normal"]
    fpr = (sh["score"] >= sh["threshold"]).mean()
    ax.set_title(name, fontsize=10)
    ax.text(0.02, 0.97, f"per-camera AUROC(obstr,shadow) = {pc:.3f}\nshadow FPR = {fpr:.0%}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8.5)
    ax.set_xlabel("Raw anomaly score")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Density")
axes[0].legend(loc="center right", fontsize=8.5, frameon=False)
axes[0].set_xlim(lo, hi)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"p3_overlap_before_after.{ext}", dpi=150, bbox_inches="tight")
print("saved p3_overlap_before_after.pdf/.png")
