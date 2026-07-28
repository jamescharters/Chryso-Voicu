"""
pc_bert.py — is a frozen neural channel worth it? (NEW file; analysis side.)

We add the SuperPeitho Ancient-Greek BERT as a fourth verification channel -- used
frozen, inference only, no training -- and ask two questions a reviewer will ask:

  1. Does the neural channel verify authorship as well as the hand-crafted ones?
  2. Does it AGREE with them, on the benchmark and on the pseudo-Chrysostom verdicts?

If a 768-d contextual embedding does no better than 1000 function words, the
transparent features are vindicated; if it agrees on the verdicts, the black box is
corroboration rather than a rival. Either way the interpretable pipeline stands.

Run order:
    venv/bin/python pc_bert.py            # dumps ocr-results/bert-input.csv, then stops
    bert-env/bin/python bert_embed.py     # writes ocr-results/bert-emb.npz (inference)
    venv/bin/python pc_bert.py            # runs the comparison
"""
from __future__ import annotations

import os
import hashlib
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_fscore_support

from pc_verify import load_matrix, bdi, name, PTA_CANDIDATES

CACHE, IN_CSV, TAB = "ocr-results/bert-emb.npz", "ocr-results/bert-input.csv", "paper/tables/bert.tex"
MIN_WORKS, N_QUERY, N_NEG = 5, 2, 2


def key_of(t):
    return hashlib.md5(str(t).encode("utf-8")).hexdigest()


def zblock(B_ref, B_pc):
    mu, sd = B_ref.mean(0), B_ref.std(0)
    sd[sd == 0] = 1.0
    return (B_ref - mu) / sd, (B_pc - mu) / sd


def cos_to_centroid(q, cent):
    return float(q @ cent) / (np.linalg.norm(q) * np.linalg.norm(cent) + 1e-12)


def best_f1(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    pred = (s >= thr[j]).astype(int)
    _, _, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return f


def main():
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()

    if not os.path.exists(CACHE):
        d = {}
        for t in list(ref["modified_text"]) + list(pc["modified_text"]):
            d[key_of(t)] = str(t)
        os.makedirs("ocr-results", exist_ok=True)
        pd.DataFrame({"key": list(d), "text": list(d.values())}).to_csv(IN_CSV, index=False)
        print(f"wrote {IN_CSV} ({len(d)} texts).\nNow run:  bert-env/bin/python bert_embed.py")
        return

    npz = np.load(CACHE, allow_pickle=True)
    emb = {str(k): v for k, v in zip(npz["keys"], npz["emb"])}
    Bref = np.array([emb[key_of(t)] for t in ref["modified_text"]], dtype=float)
    Bpc = np.array([emb[key_of(t)] for t in pc["modified_text"]], dtype=float)
    Bref, Bpc = zblock(Bref, Bpc)
    print(f"BERT channel: {Bref.shape[1]}-d; ref {Bref.shape[0]}, pc {Bpc.shape[0]}\n")

    rng = np.random.RandomState(0)
    cnt = Counter(au)
    authors = sorted(a for a in cnt if cnt[a] >= MIN_WORKS)
    centH = {a: Zref[au == a].mean(0) for a in set(au)}
    centB = {a: Bref[au == a].mean(0) for a in set(au)}

    # ---- benchmark: same/different-author problems, scored by each channel ----
    rows = []
    for a in authors:
        idx = np.where(au == a)[0]
        n = len(idx)
        for wi in rng.choice(idx, min(N_QUERY, n), replace=False):
            tgt = idx[idx != wi]
            rows.append((1,
                         bdi(Zref[wi], Zref[tgt], Zref[au != a], rng),
                         bdi(Bref[wi], Bref[tgt], Bref[au != a], rng),
                         cos_to_centroid(Bref[wi], (centB[a] * n - Bref[wi]) / (n - 1))))
            for b in rng.choice([x for x in authors if x != a], N_NEG, replace=False):
                rows.append((0,
                             bdi(Zref[wi], Zref[au == b], Zref[au != b], rng),
                             bdi(Bref[wi], Bref[au == b], Bref[au != b], rng),
                             cos_to_centroid(Bref[wi], centB[b])))

    df = pd.DataFrame(rows, columns=["label", "hand", "bert", "bert_cos"])
    y = df["label"].to_numpy()
    print(f"benchmark: {len(df)} problems ({int(y.sum())} same, {int((1-y).sum())} different)\n")

    METH = {"hand": "Hand-crafted (FW+POS+AFF) impostors", "bert": "BERT impostors",
            "bert_cos": "BERT cosine"}
    metrics = {}
    print(f"{'channel':>34} {'ROC-AUC':>8} {'AP':>6} {'F1':>6}")
    print("-" * 58)
    for m in METH:
        s = df[m].to_numpy()
        metrics[m] = (roc_auc_score(y, s), average_precision_score(y, s), best_f1(y, s))
        print(f"{METH[m]:>34} {metrics[m][0]:8.3f} {metrics[m][1]:6.3f} {metrics[m][2]:6.3f}")

    agree = df["hand"].corr(df["bert"], method="spearman")
    print(f"\nhand vs BERT impostors agreement (Spearman): {agree:.3f}")

    # ---- pseudo-Chrysostom verdicts: BERT vs hand-crafted top candidate ----
    cand = [a for a in PTA_CANDIDATES if a in centB]
    def score_group(V, docs_by, q_idx, chan_docs):
        return {a: float(np.mean([bdi(V[i], chan_docs[a], chan_docs["_imp"][a], rng)
                                  for i in q_idx])) for a in cand}
    chanH = {a: Zref[au == a] for a in cand}; chanH["_imp"] = {a: Zref[au != a] for a in cand}
    chanB = {a: Bref[au == a] for a in cand}; chanB["_imp"] = {a: Bref[au != a] for a in cand}

    same_top = 0; groups = sorted(pc["author"].unique()); pair_h, pair_b = [], []
    detail = []
    for g in groups:
        qi = pc.index[pc["author"] == g].to_numpy()
        sh = score_group(Zpc, None, qi, chanH)
        sb = score_group(Bpc, None, qi, chanB)
        th, tb = max(sh, key=sh.get), max(sb, key=sb.get)
        same_top += th == tb
        pair_h += [sh[a] for a in cand]; pair_b += [sb[a] for a in cand]
        detail.append((g, name(th), name(tb), th == tb))
    pc_agree = pd.Series(pair_h).corr(pd.Series(pair_b), method="spearman")
    print(f"\nPC top-candidate agreement: {same_top}/{len(groups)} groups; "
          f"per-candidate score Spearman {pc_agree:.3f}")
    for g, th, tb, ok in detail:
        if g in ("PC1", "PC9", "PC21", "PC16", "PC4", "PC19"):
            print(f"  {g:>5}: hand={th:<12} bert={tb:<12} {'agree' if ok else 'DIFFER'}")

    write_table(metrics, METH, agree, same_top, len(groups))
    print(f"\nwrote {TAB}")


def write_table(metrics, METH, agree, same_top, n_groups):
    lines = ["% auto-generated by pc_bert.py -- do not edit by hand",
             "\\begin{tabular}{lccc}", "\\toprule",
             "Channel & ROC-AUC & AP & $F_1$ \\\\", "\\midrule"]
    for m in METH:
        auc, ap, f = metrics[m]
        lines.append(f"{METH[m]} & {auc:.3f} & {ap:.3f} & {f:.3f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(TAB, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
