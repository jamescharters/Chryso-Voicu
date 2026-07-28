"""
pc_verify.py — calibrated authorship verification for the Pseudo-Chrysostoms via
Bootstrap Distance Impostors (BDI), plus a provenance control (NEW file; base
pipeline untouched).

pc_adjudicate.py showed every PC group leans Severian on nearest-centroid distance.
This script upgrades that leaning to a *calibrated verification score* and rules out
the obvious confound (shared PTA/Voicu digitization).

BDI (Kestemont et al.; the method Wauchier 2025 uses): to score whether text Q is by
author A, repeatedly take a random subset of features and a random pool of impostor
documents, and count how often Q's nearest A-document beats its nearest impostor. The
score in [0,1] is the fraction of iterations A wins; ~1 = confident same author, ~0 =
confident different. Target documents are sub-sampled to a fixed count each iteration
so Severian's 200+ works do not out-draw Chrysostom's 5.

Runs three things:
  1. calibration  — BDI self-verification: held-out Severian works vs Severian, and
                    other-author works vs Severian; reports the score gap + AUC.
  2. provenance   — within-PTA only: does each PC prefer Severian over the OTHER PTA
                    authors (same digitization)?  If yes, it is not a source artifact.
  3. verdicts     — BDI(PC, Severian) and BDI(PC, Chrysostom) per group, with the
                    divergent PCs highlighted.
Also checks PC/Severian title overlap to rule out circularity.
"""
from __future__ import annotations

import os
import re
import glob
import collections

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from corpus_balance import balance_corpus
from ocr_source import derive_source

CHANNELS = {"FW": "$MFW$", "POS": "$POS$", "AFF": "$TRI$"}
SEVERIAN, CHRYSOSTOM = "pta0001", "pta0002"
PTA_CANDIDATES = ["pta0001", "pta0002", "pta0003", "pta0004", "pta0005", "pta0007", "pta0022"]
# broader field (incl. First1K authors) for the divergent-case ranking
EXTRA_CANDIDATES = ["Clement of Alexandria", "Origenes", "Gregory of Nazianzus", "Eusebius"]
ITERS, FEAT_FRAC, N_IMP, TGT_SAMPLE = 300, 0.5, 25, 5


def build_pta_names():
    m = {}
    for f in glob.glob("patres/*.xml"):
        try:
            txt = open(f, encoding="utf-8").read(4000)
        except OSError:
            continue
        for key, nm in re.findall(r'<persName key="(pta\d+)">([^<]+)</persName>', txt):
            m.setdefault(key, collections.Counter())[nm.replace("\n", " ").strip()] += 1
    return {k: c.most_common(1)[0][0] for k, c in m.items()}


NAMES = build_pta_names()
def name(a): return NAMES.get(a, str(a))


def load_matrix():
    """z-scored, channel-concatenated feature matrix for reference + PC, shared cols."""
    ref = balance_corpus(pd.read_csv("tlg-features.csv"), seed=0, min_works=4, cap_works=30)
    ref = ref[derive_source(ref).to_numpy() == "clean"].reset_index(drop=True)
    pc = pd.read_csv("pc-features.csv")
    pc_cols = set(pc.columns)

    ref_blocks, pc_blocks = [], []
    for ch, pref in CHANNELS.items():
        c = [x for x in ref.columns if x.startswith(pref) and x in pc_cols]
        X = ref[c].fillna(0).to_numpy(float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        ref_blocks.append((X - mu) / sd)
        pc_blocks.append((pc[c].fillna(0).to_numpy(float) - mu) / sd)
    Zref = np.hstack(ref_blocks)
    Zpc = np.hstack(pc_blocks)
    return ref, pc, Zref, Zpc


def _cos(a, B):
    a = a / (np.linalg.norm(a) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Bn @ a


def bdi(q, target_docs, impostor_docs, rng):
    """Fraction of bootstrap iterations where q's nearest target beats nearest impostor."""
    dim = q.shape[0]
    fk = int(dim * FEAT_FRAC)
    wins = 0
    for _ in range(ITERS):
        fmask = rng.choice(dim, fk, replace=False)
        tsub = target_docs[rng.choice(len(target_docs), min(TGT_SAMPLE, len(target_docs)), replace=False)]
        isub = impostor_docs[rng.choice(len(impostor_docs), min(N_IMP, len(impostor_docs)), replace=False)]
        dt = _cos(q[fmask], tsub[:, fmask]).min()
        di = _cos(q[fmask], isub[:, fmask]).min()
        wins += dt < di
    return wins / ITERS


def score_group(q_vecs, target_docs, impostor_docs, rng):
    return float(np.mean([bdi(q, target_docs, impostor_docs, rng) for q in q_vecs]))


def main():
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()
    rng = np.random.RandomState(0)
    docs = {a: Zref[au == a] for a in set(au)}
    print(f"reference {Zref.shape[0]} works x {Zref.shape[1]} feats; PC {Zpc.shape[0]} texts\n")

    # circularity: PC vs Severian title overlap
    pc_titles = set(pc["title"].astype(str).str.lower().str.strip())
    sev_titles = set(ref[ref.author == SEVERIAN]["title"].astype(str).str.lower().str.strip())
    overlap = pc_titles & sev_titles
    print(f"[circularity] PC/Severian shared titles: {len(overlap)}  "
          f"{'(clean)' if not overlap else sorted(overlap)[:3]}")

    # ---- 1. calibration: BDI self-test on Severian ----
    sev_idx = np.where(au == SEVERIAN)[0]
    other_idx = np.where(au != SEVERIAN)[0]
    imp_all = Zref[other_idx]
    ys, ss = [], []
    for i in rng.choice(sev_idx, min(20, len(sev_idx)), replace=False):
        tgt = Zref[sev_idx[sev_idx != i]]
        ss.append(bdi(Zref[i], tgt, imp_all, rng)); ys.append(1)
    for i in rng.choice(other_idx, 20, replace=False):
        ss.append(bdi(Zref[i], Zref[sev_idx], Zref[np.setdiff1d(other_idx, [i])], rng)); ys.append(0)
    auc = roc_auc_score(ys, ss)
    pos = np.mean([s for s, y in zip(ss, ys) if y]); neg = np.mean([s for s, y in zip(ss, ys) if not y])
    print(f"[calibration] BDI Severian self={pos:.2f} vs non-Severian={neg:.2f}  AUC={auc:.3f}  "
          f"{'OK' if auc > 0.8 else 'WEAK'}\n")

    # ---- 2 + 3: per-PC verdicts + provenance (within-PTA) ----
    field = [a for a in PTA_CANDIDATES + EXTRA_CANDIDATES if a in docs]
    rows = []
    for g in sorted(pc["author"].unique()):
        qi = pc.index[pc["author"] == g].to_numpy()
        qv = Zpc[qi]
        sev = score_group(qv, docs[SEVERIAN], Zref[au != SEVERIAN], rng)
        chry = score_group(qv, docs[CHRYSOSTOM], Zref[au != CHRYSOSTOM], rng)
        # provenance: rank PTA authors only (same digitization)
        pta_scores = {a: score_group(qv, docs[a], Zref[au != a], rng)
                      for a in PTA_CANDIDATES if a in docs}
        best_pta = max(pta_scores, key=pta_scores.get)
        # broader field ranking for divergent cases
        field_scores = {a: (pta_scores[a] if a in pta_scores else
                            score_group(qv, docs[a], Zref[au != a], rng)) for a in field}
        top2 = sorted(field_scores, key=field_scores.get, reverse=True)[:2]
        rows.append(dict(PC=g, n=len(qi), BDI_Severian=round(sev, 2), BDI_Chrysostom=round(chry, 2),
                         best_PTA=name(best_pta), sev_wins_PTA=(best_pta == SEVERIAN),
                         top1=f"{name(top2[0])} {field_scores[top2[0]]:.2f}",
                         top2=f"{name(top2[1])} {field_scores[top2[1]]:.2f}"))

    out = pd.DataFrame(rows).sort_values("BDI_Severian", ascending=False)
    os.makedirs("ocr-results", exist_ok=True)
    out.to_csv("ocr-results/pc-verification.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))

    sev_win = out["sev_wins_PTA"].sum()
    print(f"\n[provenance] Severian beats other PTA authors for {sev_win}/{len(out)} PC groups "
          f"(controls for shared digitization).")
    strong = (out["BDI_Severian"] >= 0.9).sum()
    print(f"[verdict] {strong}/{len(out)} PC groups verify to Severian at BDI>=0.90.")
    div = out[out["BDI_Severian"] < 0.5]
    if len(div):
        print("\nLow-Severian-confidence PCs (nearest author in the broader field):")
        print(div[["PC", "BDI_Severian", "BDI_Chrysostom", "top1", "top2"]].to_string(index=False))
    print("\nwrote ocr-results/pc-verification.csv")


if __name__ == "__main__":
    main()
