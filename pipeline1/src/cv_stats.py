"""
Paired statistical tests over the 10-fold CV results (cv_train_eval.py).
====================================================================

For every pair of models we have 10 paired measurements (one per fold, all
models trained/evaluated on the same fold split), so we can run:

  * paired t-test          - parametric, assumes the 10 differences are
                             roughly normal. More powerful when they are.
  * Wilcoxon signed-rank   - non-parametric, only assumes symmetry. Safer on
                             small n, but with n=10 the smallest attainable
                             two-sided p is 2/2^10 = 0.002.

Both are run on the GEMINI metric (external test set), not on the synthetic
CV-fold metric - see cv_train_eval.py for why.

CAVEATS reported alongside the numbers (state these in the thesis):
  1. CV folds share training data (any two 10-fold training sets overlap in
     ~89% of the images), so the differences are NOT independent. The variance
     is underestimated and the Type-I error inflated - the classic Dietterich
     (1998) critique. p-values are therefore OPTIMISTIC; treat them as
     descriptive, not as strict hypothesis tests.
  2. With k models there are k(k-1)/2 pairs, so we also report Holm-corrected
     p-values for multiple comparisons.

Usage:
  python pipeline1/src/cv_stats.py --csv pipeline1/results/cv10_gemini.csv \
      --metric gem_map50
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import shapiro, ttest_rel, wilcoxon


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj.tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="pipeline1/results/cv10_gemini.csv")
    ap.add_argument("--metric", default="gem_map50",
                    choices=["gem_map50", "gem_map", "gem_p", "gem_r",
                             "synth_map50", "synth_map"])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    piv = df.pivot(index="fold", columns="model", values=args.metric).sort_index()
    models = list(piv.columns)
    n = len(piv)

    print(f"metric: {args.metric}   folds: {n}   models: {len(models)}")
    if n < 6:
        print(f"[warn] with n={n} the Wilcoxon signed-rank test can never reach "
              f"p<0.05 (min attainable p = {2 / 2 ** n:.4f}).")
    print()

    print("== per-model summary over folds ==")
    summ = piv.agg(["mean", "std", "min", "max"]).T.sort_values("mean",
                                                                ascending=False)
    summ["mean+-std"] = (summ["mean"].round(4).astype(str) + " +- "
                         + summ["std"].round(4).astype(str))
    print(summ[["mean+-std", "min", "max"]].to_string())
    print()

    rows = []
    for a, b in itertools.combinations(models, 2):
        x, y = piv[a].to_numpy(), piv[b].to_numpy()
        d = x - y
        if np.allclose(d, 0):
            continue
        t_stat, t_p = ttest_rel(x, y)
        try:
            w_stat, w_p = wilcoxon(x, y)
        except ValueError:            # all differences zero
            w_stat, w_p = np.nan, 1.0
        # normality of the differences: justifies (or not) the t-test
        sh_p = shapiro(d).pvalue if n >= 3 else np.nan
        rows.append({
            "model_a": a, "model_b": b,
            "mean_diff": round(float(d.mean()), 5),
            "wins_a": int((d > 0).sum()), "wins_b": int((d < 0).sum()),
            "t_p": t_p, "wilcoxon_p": w_p, "shapiro_p_diff": round(float(sh_p), 4),
        })

    res = pd.DataFrame(rows)
    res["t_p_holm"] = holm(res["t_p"].tolist())
    res["wilcoxon_p_holm"] = holm(res["wilcoxon_p"].tolist())
    for c in ("t_p", "wilcoxon_p", "t_p_holm", "wilcoxon_p_holm"):
        res[c] = res[c].round(4)
    res = res.sort_values("t_p")

    print(f"== pairwise paired tests ({len(res)} pairs) ==")
    print(res.to_string(index=False))
    print()

    sig_t = (res["t_p_holm"] < args.alpha).sum()
    sig_w = (res["wilcoxon_p_holm"] < args.alpha).sum()
    print(f"significant after Holm correction (alpha={args.alpha}): "
          f"t-test {sig_t}/{len(res)}, Wilcoxon {sig_w}/{len(res)}")
    if sig_t == 0 and sig_w == 0:
        print("-> no model pair is distinguishable: the differences seen in the "
              "single-run table are within fold-to-fold noise.")
    print()
    print("CAVEAT: 10-fold training sets overlap ~89% pairwise, so the paired "
          "differences are not independent (Dietterich 1998). p-values are "
          "optimistic - report them as descriptive evidence, not as strict "
          "hypothesis tests.")

    if args.out_csv:
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
