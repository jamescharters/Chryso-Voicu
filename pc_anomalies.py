"""
pc_anomalies.py — pressure-test the divergent Pseudo-Chrysostom verdicts (NEW file).

pc_verify.py flagged four groups that do NOT point to Severian: PC1 -> John Chrysostom,
PC4 -> Cyril, PC19 -> Theodoret, PC10 -> Chrysostom/Clement. Before any of those is
worth a claim it must survive two tests:

  1. stability   — is the verdict robust across random seeds, or Monte-Carlo noise?
                   Reports BDI mean +/- std over several seeds for each relevant candidate.
  2. per-channel — do independent feature families (function words / POS / affixes)
                   agree, or does one channel carry the whole verdict? A verdict backed
                   by all three is trustworthy; one carried by a single channel is not.

PC21 and PC16 (the confident-Severian groups) are included as positive controls.
Self-contained; imports nothing from pc_verify so that file stays stable.
"""
from __future__ import annotations

import glob
import re
import collections

import numpy as np
import pandas as pd

from corpus_balance import balance_corpus
from ocr_source import derive_source

CHANNELS = {"FW": "$MFW$", "POS": "$POS$", "AFF": "$TRI$"}
FOCUS = ["PC1", "PC4", "PC10", "PC19", "PC16", "PC21"]
CANDS = {"pta0001": "Severian", "pta0002": "Chrysostom", "pta0005": "Cyril",
         "pta0004": "Theodoret", "pta0007": "Origen", "Clement of Alexandria": "Clement"}
SEEDS = [0, 1, 2, 3, 4]
ITERS, FEAT_FRAC, N_IMP, TGT = 200, 0.5, 25, 5


def _cos(a, B):
    a = a / (np.linalg.norm(a) + 1e-12)
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Bn @ a


def bdi(q, target, impostors, rng):
    dim = q.shape[0]; fk = max(1, int(dim * FEAT_FRAC)); wins = 0
    for _ in range(ITERS):
        fm = rng.choice(dim, fk, replace=False)
        ts = target[rng.choice(len(target), min(TGT, len(target)), replace=False)]
        im = impostors[rng.choice(len(impostors), min(N_IMP, len(impostors)), replace=False)]
        wins += _cos(q[fm], ts[:, fm]).min() < _cos(q[fm], im[:, fm]).min()
    return wins / ITERS


def group_bdi(qv, target, impostors, seed):
    rng = np.random.RandomState(seed)
    return float(np.mean([bdi(q, target, impostors, rng) for q in qv]))


def load_blocks():
    ref = balance_corpus(pd.read_csv("tlg-features.csv"), seed=0, min_works=4, cap_works=30)
    ref = ref[derive_source(ref).to_numpy() == "clean"].reset_index(drop=True)
    pc = pd.read_csv("pc-features.csv")
    pc_cols = set(pc.columns)
    blocks = {}
    for ch, pref in CHANNELS.items():
        c = [x for x in ref.columns if x.startswith(pref) and x in pc_cols]
        X = ref[c].fillna(0).to_numpy(float)
        mu, sd = X.mean(0), X.std(0); sd[sd == 0] = 1.0
        blocks[ch] = ((X - mu) / sd, (pc[c].fillna(0).to_numpy(float) - mu) / sd)
    Zref = np.hstack([blocks[ch][0] for ch in CHANNELS])
    Zpc = np.hstack([blocks[ch][1] for ch in CHANNELS])
    return ref, pc, blocks, Zref, Zpc


def main():
    ref, pc, blocks, Zref, Zpc = load_blocks()
    au = ref["author"].to_numpy()
    cands = [a for a in CANDS if a in set(au)]

    # ---- 1. stability across seeds (combined features) ----
    print("STABILITY  (BDI mean\u00b1std over %d seeds; combined features)\n" % len(SEEDS))
    header = "  PC   " + "".join(f"{CANDS[a]:>13}" for a in cands)
    print(header); print("-" * len(header))
    for g in FOCUS:
        qi = pc.index[pc["author"] == g].to_numpy()
        qv = Zpc[qi]
        cells = []
        for a in cands:
            tgt = Zref[au == a]; imp = Zref[au != a]
            vals = [group_bdi(qv, tgt, imp, s) for s in SEEDS]
            cells.append(f"{np.mean(vals):.2f}\u00b1{np.std(vals):.02f}")
        print(f"{g:>5}  " + "".join(f"{c:>13}" for c in cells))

    # ---- 2. per-channel evidence: leading alt vs Severian ----
    print("\nPER-CHANNEL BDI  (does each feature family agree?)\n")
    print(f"{'PC':>5} {'candidate':>11} {'FW':>6} {'POS':>6} {'AFF':>6} {'combined':>9}")
    print("-" * 46)
    for g in FOCUS:
        qi = pc.index[pc["author"] == g].to_numpy()
        # pick the two authors to contrast: Severian + the group's best non-Severian
        combined_scores = {}
        for a in cands:
            combined_scores[a] = group_bdi(Zpc[qi], Zref[au == a], Zref[au != a], 0)
        best_alt = max((a for a in cands if a != "pta0001"), key=combined_scores.get)
        for a in dict.fromkeys(["pta0001", best_alt]):
            row = []
            for ch in CHANNELS:
                Zr, Zp = blocks[ch]
                row.append(group_bdi(Zp[qi], Zr[au == a], Zr[au != a], 0))
            print(f"{g:>5} {CANDS[a]:>11} {row[0]:6.2f} {row[1]:6.2f} {row[2]:6.2f} "
                  f"{combined_scores[a]:9.2f}")
    print("\n(A verdict with all three channels pointing the same way is trustworthy; "
          "one carried by a single channel is not.)")


if __name__ == "__main__":
    main()
