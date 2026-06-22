"""Check whether adding shadows to the bank changes DISCRIMINATION or only CALIBRATION.

For each config we compute, per reference scene, the AUROC of:
  - obstructed vs shadow_normal   (can the model rank a real obstruction above a shadowed normal?)
  - obstructed vs normal          (main detection task)
and average over scenes. We also report the shadow FPR at the stored threshold.
AUROC is rank-based (invariant to any per-camera affine rescaling / threshold).
"""
from pathlib import Path
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[2] / "pipeline3" / "results"

CONFIGS = {
    "no-aug (baseline)": "results_p3_senza_gestione_ombre.csv",
    "shadow04 (coupled)": "results_p3_shadow04.csv",
    "decoupled k=4": "results_p3_decoupled_k_4.csv",
    "pooled sigma": "results_p3_pooled.csv",
}


def auroc(pos, neg):
    """Rank-based AUROC = P(score_pos > score_neg)."""
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    allv = np.concatenate([pos, neg])
    ranks = pd.Series(allv).rank().to_numpy()
    r_pos = ranks[: len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


for name, fn in CONFIGS.items():
    df = pd.read_csv(RESULTS / fn)
    a_so, a_no = [], []
    for ref, g in df.groupby("reference_id"):
        obst = g.loc[g.test_type == "obstructed", "score"].to_numpy()
        shad = g.loc[g.test_type == "shadow_normal", "score"].to_numpy()
        norm = g.loc[g.test_type == "normal", "score"].to_numpy()
        if len(obst) and len(shad):
            a_so.append(auroc(obst, shad))
        if len(obst) and len(norm):
            a_no.append(auroc(obst, norm))
    # shadow FPR at stored threshold
    sh = df[df.test_type == "shadow_normal"]
    shadow_fpr = (sh["score"] >= sh["threshold"]).mean()
    print(f"{name:22s} | AUROC obstr-vs-shadow = {np.nanmean(a_so):.4f} "
          f"| AUROC obstr-vs-normal = {np.nanmean(a_no):.4f} "
          f"| shadow FPR = {shadow_fpr:6.2%}")
