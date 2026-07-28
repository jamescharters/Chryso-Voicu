"""
pc_benchmark.py — quantitative verification benchmark + method triangulation (NEW file).

Successor papers report hard verification numbers; we add the same. On labelled
reference authors we build same/different-author verification problems and score them
with three feature-space methods:

  * impostors  the win-fraction verifier used in the paper (GI/BDI family)
  * delta      Burrows's Delta to the author centroid (nearest-profile)
  * cosine     cosine similarity to the author centroid

Reporting ROC-AUC, average precision, and F1 at the Youden-optimal threshold for each
shows (a) the verifier's accuracy in absolute terms and (b) that the three independent
methods agree -- the verdicts are not an artefact of one scorer.

Outputs paper/figures/roc.pdf and paper/tables/benchmark.tex.
"""
from __future__ import annotations

import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

from pc_verify import load_matrix, bdi

FIGDIR, TABDIR = "paper/figures", "paper/tables"
MIN_WORKS, N_QUERY, N_NEG = 5, 2, 2


def loo_centroid(cent, q, n):
    return (cent * n - q) / (n - 1)


def delta_score(q, cent):
    return -np.abs(q - cent).mean()


def cos_score(q, cent):
    return float(q @ cent) / (np.linalg.norm(q) * np.linalg.norm(cent) + 1e-12)


def best_f1(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = np.argmax(tpr - fpr)
    pred = (s >= thr[j]).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return p, r, f


def main():
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()
    cnt = Counter(au)
    authors = sorted(a for a in cnt if cnt[a] >= MIN_WORKS)
    cent = {a: Zref[au == a].mean(0) for a in set(au)}
    rng = np.random.RandomState(0)

    rows = []
    for a in authors:
        idx = np.where(au == a)[0]
        n = len(idx)
        for wi in rng.choice(idx, min(N_QUERY, n), replace=False):
            q = Zref[wi]
            tgt = Zref[idx[idx != wi]]
            c_self = loo_centroid(cent[a], q, n)
            rows.append((1,
                         bdi(q, tgt, Zref[au != a], rng),
                         delta_score(q, c_self), cos_score(q, c_self)))
            for b in rng.choice([x for x in authors if x != a], N_NEG, replace=False):
                rows.append((0,
                             bdi(q, Zref[au == b], Zref[au != b], rng),
                             delta_score(q, cent[b]), cos_score(q, cent[b])))

    df = pd.DataFrame(rows, columns=["label", "impostors", "delta", "cosine"])
    y = df["label"].to_numpy()
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"benchmark: {len(df)} verification problems ({n_pos} same-author, {n_neg} different-author)\n")

    methods = ["impostors", "delta", "cosine"]
    metrics = {}
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    print(f"{'method':>10} {'ROC-AUC':>8} {'AP':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    print("-" * 46)
    for m in methods:
        s = df[m].to_numpy()
        auc = roc_auc_score(y, s)
        ap = average_precision_score(y, s)
        p, r, f = best_f1(y, s)
        metrics[m] = (auc, ap, p, r, f)
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, label=f"{m} (AUC {auc:.3f})")
        print(f"{m:>10} {auc:8.3f} {ap:6.3f} {p:6.3f} {r:6.3f} {f:6.3f}")

    ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.6")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("Verification ROC on reference authors")
    ax.legend(fontsize=8, loc="lower right")
    os.makedirs(FIGDIR, exist_ok=True)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/roc.pdf"); plt.close(fig)

    # method agreement: correlation of the three scores across problems
    corr = df[methods].corr(method="spearman")
    print("\nSpearman correlation between methods (verdict agreement):")
    print(corr.round(3).to_string())

    os.makedirs(TABDIR, exist_ok=True)
    with open(f"{TABDIR}/benchmark.tex", "w") as fh:
        fh.write("\\begin{tabular}{lccccc}\n\\toprule\n")
        fh.write("Method & ROC-AUC & AP & Prec. & Rec. & F1 \\\\\n\\midrule\n")
        names = {"impostors": "Impostors (this paper)", "delta": "Burrows's Delta", "cosine": "Cosine"}
        for m in methods:
            auc, ap, p, r, f = metrics[m]
            fh.write(f"{names[m]} & {auc:.3f} & {ap:.3f} & {p:.3f} & {r:.3f} & {f:.3f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print(f"\nwrote {FIGDIR}/roc.pdf, {TABDIR}/benchmark.tex")


if __name__ == "__main__":
    main()
