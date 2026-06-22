from pathlib import Path
import numpy as np, pandas as pd
RESULTS = Path(__file__).resolve().parents[2] / "pipeline3" / "results"
configs = {
    "no-aug": "results_p3_senza_gestione_ombre.csv",
    "shadow04 coupled": "results_p3_shadow04.csv",
    "decoupled k=3": "results_p3_decoupled.csv",
    "decoupled k=4": "results_p3_decoupled_k_4.csv",
    "pooled": "results_p3_pooled.csv",
}
print(f"{'config':18s} | normal | shadow | obstr | gap(o-s) | mean_thr | shadowFPR | shadow>thr-margin")
for name, fn in configs.items():
    d = pd.read_csv(RESULTS / fn)
    nm = d.loc[d.test_type=='normal','score'].mean()
    sm = d.loc[d.test_type=='shadow_normal','score'].mean()
    om = d.loc[d.test_type=='obstructed','score'].mean()
    thr = d['threshold'].mean()
    sh = d[d.test_type=='shadow_normal']
    fpr = (sh['score']>=sh['threshold']).mean()
    margin = (sh['score'] - sh['threshold']).mean()  # how far shadows sit above their own threshold
    print(f"{name:18s} | {nm:6.3f} | {sm:6.3f} | {om:6.3f} | {om-sm:8.3f} | {thr:8.3f} | {fpr:8.2%} | {margin:+.3f}")
