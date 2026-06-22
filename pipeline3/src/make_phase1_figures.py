"""
Generate the Phase 1 (D1, calibration) figures for the P3 results chapter.

Outputs (PDF for LaTeX + PNG for preview) into template_tesi/images/patchCore/:
  - p3_score_overlap.{pdf,png}   : 4-way normalised-score distributions
                                   (normal / shadow / light / obstructed) + threshold
  - p3_threshold_tradeoff.{pdf,png}: TPR vs shadow-FPR vs clean-FPR as the global
                                     decision threshold sweeps (baseline, no augmentation)

Run:  conda run -n tesi_env python pipeline3/src/make_phase1_figures.py
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
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})

# Colour scheme shared by the two figures
C_NORMAL = "#2ca02c"   # green
C_SHADOW = "#ff7f0e"   # orange
C_LIGHT = "#1f77b4"    # blue (distinct from shadow orange)
C_OBSTR = "#d62728"    # red
C_CLEAN = "#7f7f7f"    # grey


def _kde_curve(values, grid):
    values = np.asarray(values, dtype=float)
    if HAVE_KDE and values.std() > 1e-6:
        return gaussian_kde(values)(grid)
    # Fallback: normalised histogram interpolated onto the grid
    hist, edges = np.histogram(values, bins=40, density=True)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return np.interp(grid, centres, hist, left=0.0, right=0.0)


def fig_score_overlap():
    df = pd.read_csv(RESULTS / "ablation_L2_SL15.csv")

    # Decision boundary in normalised units. normalized_score = (score - mu)/sigma_pop
    # and threshold = mu + k*sigma_pop, hence the boundary is k itself.
    m = (df["score"] - df["mu"]).abs() > 1e-6
    k_vals = ((df.loc[m, "threshold"] - df.loc[m, "mu"]) * df.loc[m, "normalized_score"]
              / (df.loc[m, "score"] - df.loc[m, "mu"]))
    k = float(np.median(k_vals))

    # Consistency check against the stored is_anomaly flag
    pred = (df["normalized_score"] >= k).astype(int)
    agree = float((pred == df["is_anomaly"]).mean())
    print(f"[overlap] decision boundary k = {k:.3f}  | pred==is_anomaly: {agree:.3%}")

    groups = [
        ("normal", "Normal", C_NORMAL),
        ("shadow_normal", "Shadow (normal)", C_SHADOW),
        ("light_normal", "Light (normal)", C_LIGHT),
        ("obstructed", "Obstructed", C_OBSTR),
    ]

    allv = df["normalized_score"].to_numpy()
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    grid = np.linspace(lo, hi, 600)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for key, label, colour in groups:
        v = df.loc[df["test_type"] == key, "normalized_score"].to_numpy()
        if v.size == 0:
            continue
        y = _kde_curve(v, grid)
        ax.fill_between(grid, y, color=colour, alpha=0.25)
        ax.plot(grid, y, color=colour, lw=1.8, label=f"{label} (n={v.size})")

    ax.axvline(k, color="black", ls="--", lw=1.3)
    ymax = ax.get_ylim()[1]
    ax.text(k, ymax * 0.98, f"  decision threshold ($k={k:.0f}$)",
            rotation=90, va="top", ha="left", fontsize=9)

    ax.set_xlabel("Normalised anomaly score")
    ax.set_ylabel("Density")
    ax.set_xlim(lo, hi)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"p3_score_overlap.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[overlap] saved p3_score_overlap.pdf/.png in {OUT}")


def fig_threshold_tradeoff():
    df = pd.read_csv(RESULTS / "results_p3_senza_gestione_ombre.csv")
    s_norm = df.loc[df["test_type"] == "normal", "score"].to_numpy()
    s_shad = df.loc[df["test_type"] == "shadow_normal", "score"].to_numpy()
    s_obst = df.loc[df["test_type"] == "obstructed", "score"].to_numpy()

    lo = float(min(s_norm.min(), s_shad.min(), s_obst.min()))
    hi = float(max(s_norm.max(), s_shad.max(), s_obst.max()))
    ts = np.linspace(lo, hi, 400)

    tpr = np.array([(s_obst >= t).mean() for t in ts]) * 100
    fpr_sh = np.array([(s_shad >= t).mean() for t in ts]) * 100
    fpr_cl = np.array([(s_norm >= t).mean() for t in ts]) * 100

    # Operating points quoted in the text
    for t0 in (4.0, 4.5):
        print(f"[tradeoff] t={t0:.2f}  TPR={ (s_obst>=t0).mean():.3%}  "
              f"shadow-FPR={ (s_shad>=t0).mean():.3%}  clean-FPR={ (s_norm>=t0).mean():.3%}")

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(ts, tpr, color=C_OBSTR, lw=2.0, label="TPR (obstructed)")
    ax.plot(ts, fpr_sh, color=C_SHADOW, lw=2.0, label="Shadow FPR")
    ax.plot(ts, fpr_cl, color=C_CLEAN, lw=1.6, ls="--", label="Clean FPR")

    for t0 in (4.0, 4.5):
        ax.axvline(t0, color="black", lw=0.8, ls=":")
    ax.set_xlabel("Global decision threshold (raw anomaly score)")
    ax.set_ylabel("Rate (%)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-2, 103)
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"p3_threshold_tradeoff.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[tradeoff] saved p3_threshold_tradeoff.pdf/.png in {OUT}")


if __name__ == "__main__":
    fig_score_overlap()
    fig_threshold_tradeoff()
