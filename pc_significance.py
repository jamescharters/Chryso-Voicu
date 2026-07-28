"""
pc_significance.py — Nagy-style Bootstrap Distance Impostors with statistical
likelihood and confidence intervals (NEW file).

Nagy's BDI records, at each bootstrap iteration, the difference between the query's
distance to the impostors and its distance to the candidate (positive = candidate
closer). The statistical likelihood that the candidate is the author is the fraction
of that bootstrapped difference distribution above zero. Because we keep the whole
distribution we can attach a 95% bootstrap confidence interval to every verdict --
the rigour the win-fraction alone did not report.

Outputs paper/tables/groundtruth.tex (the ground-truth table, now with CIs).
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

from pc_verify import load_matrix

CHANNELS_OK = True
ITERS, FEAT_FRAC, N_IMP, TGT = 1000, 0.5, 25, 5
SEVERIAN, CHRYSOSTOM = "pta0001", "pta0002"
# ground-truth focus set: (PC, CPG, scholarship, true author if in candidate set else None)
GT = [
    ("PC1", "4410", "Chrysostom (genuine)", CHRYSOSTOM),
    ("PC21", "4564", "Severian (proposed)", SEVERIAN),
    ("PC9", "4215", "contains Severian work", SEVERIAN),
    ("PC4", "4606--12", "Apollinaris (absent)", None),
    ("PC12", "2082/83", "Anomoean (absent)", None),
]


def _cos(a, B):
    a = a / (np.linalg.norm(a) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Bn @ a


def bdi_diffs(qv, target, impostors, rng):
    """Bootstrapped distance-difference distribution (d_impostor - d_candidate)."""
    dim = qv.shape[1]; fk = int(dim * FEAT_FRAC)
    diffs = []
    for q in qv:
        for _ in range(ITERS):
            fm = rng.choice(dim, fk, replace=False)
            ts = target[rng.choice(len(target), min(TGT, len(target)), replace=False)]
            im = impostors[rng.choice(len(impostors), min(N_IMP, len(impostors)), replace=False)]
            diffs.append(_cos(q[fm], im[:, fm]).min() - _cos(q[fm], ts[:, fm]).min())
    return np.asarray(diffs)


def sl_ci(diffs, rng, n_boot=500):
    """Statistical likelihood (fraction > 0) with a 95% bootstrap CI."""
    sl = float((diffs > 0).mean())
    n = len(diffs)
    boot = [float((diffs[rng.randint(0, n, n)] > 0).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return sl, float(lo), float(hi)


def main():
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()
    docs = {a: Zref[au == a] for a in set(au)}
    rng = np.random.RandomState(0)

    rows = []
    print(f"{'PC':>5} {'CPG':>9} {'SL_Severian':>22} {'SL_Chrysostom':>22}  scholarship")
    print("-" * 90)
    for pcid, cpg, schol, truth in GT:
        qv = Zpc[pc.index[pc["author"] == pcid].to_numpy()]
        s_sev, lo_s, hi_s = sl_ci(bdi_diffs(qv, docs[SEVERIAN], Zref[au != SEVERIAN], rng), rng)
        s_chr, lo_c, hi_c = sl_ci(bdi_diffs(qv, docs[CHRYSOSTOM], Zref[au != CHRYSOSTOM], rng), rng)
        rows.append((pcid, cpg, schol, s_sev, lo_s, hi_s, s_chr, lo_c, hi_c))
        print(f"{pcid:>5} {cpg:>9} {s_sev:6.2f} [{lo_s:.2f},{hi_s:.2f}]      "
              f"{s_chr:6.2f} [{lo_c:.2f},{hi_c:.2f}]   {schol}")

    os.makedirs("paper/tables", exist_ok=True)
    with open("paper/tables/groundtruth.tex", "w") as f:
        f.write("\\begin{tabular}{llccl}\n\\toprule\n")
        f.write("PC & CPG & SL$_{\\text{Sev}}$ & SL$_{\\text{Chry}}$ & Scholarship / check \\\\\n\\midrule\n")
        for pcid, cpg, schol, s, lo_s, hi_s, c, lo_c, hi_c in rows:
            truth = dict((g[0], g[3]) for g in GT)[pcid]
            if truth is None:
                mark = "(author absent)"
            else:
                fav = SEVERIAN if s > c else CHRYSOSTOM
                mark = "\\checkmark" if fav == truth else "$\\times$"
            f.write(f"{pcid} & {cpg} & {s:.2f}\\,[{lo_s:.2f},{hi_s:.2f}] & "
                    f"{c:.2f}\\,[{lo_c:.2f},{hi_c:.2f}] & {schol}~{mark} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("\nSL = Nagy statistical likelihood (fraction of the bootstrapped distance-"
          "difference distribution above zero); [.,.] = 95% bootstrap CI.")
    print("wrote paper/tables/groundtruth.tex")


if __name__ == "__main__":
    main()
