"""
ocr_pos_confound.py — Phase 1 POS re-check with CONSISTENT tagging.

The model-free quick test found a fake "POS collapse" caused by the PG rows being
tagged with a foreign tagset. This script recomputes the POS-trigram confound after
ocr_retag.py has re-tagged the OCR volumes with the SAME BERT tagger, so any residual
difference between clean and OCR is a genuine OCR-through-the-tagger effect rather than
a tagset mismatch.

Sources (all now BERT/Perseus tagged):
    clean  Clement = tagged/tlg0555.*     Origen = tagged/tlg2042.*
    OCR    Clement = tagged-ocr/PG009*    Origen = tagged-ocr/PG016_3*

Each tagged text is cut into fixed-size token windows to give several samples per
(author, source); POS = first letter of each Perseus tag; features = relative
frequencies of POS-trigrams over a shared vocabulary; distance = cosine.
Reports the four populations SS_same / CS_same / SS_diff and within/cross AUC, exactly
as ocr_confound.py, so the numbers are directly comparable.
"""
from __future__ import annotations

import glob
from collections import Counter

import numpy as np
from sklearn.metrics import roc_auc_score

CLEAN = {"clement": "tagged/tlg0555.*-tagged.txt",
         "origen":  "tagged/tlg2042.*-tagged.txt"}
OCR = {"clement": "tagged-ocr/PG009_tagged_text-bert.txt",
       "origen":  "tagged-ocr/PG016_3_tagged_text-bert.txt"}
WINDOW = 3000   # tokens per chunk
TOPK = 100      # POS-trigram vocabulary size (match base pipeline's 100)


def read_pos(path: str) -> list[str]:
    """First-letter POS code for every tagged token in a file."""
    seq = []
    with open(path) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[1]:
                seq.append(parts[1][0])
    return seq


def window_trigram_counters(pos_seq, window=WINDOW):
    for i in range(0, max(1, len(pos_seq) - window + 1), window):
        w = pos_seq[i:i + window]
        if len(w) < 3:
            continue
        yield Counter("-".join(w[j:j + 3]) for j in range(len(w) - 2))


def collect(paths_glob):
    """Return list of trigram Counters (one per window) for a source glob/file."""
    counters = []
    for path in sorted(glob.glob(paths_glob)):
        counters.extend(window_trigram_counters(read_pos(path)))
    return counters


def main():
    samples = []   # (author, source, Counter)
    for who, g in CLEAN.items():
        for c in collect(g):
            samples.append((who, "clean", c))
    for who, g in OCR.items():
        for c in collect(g):
            samples.append((who, "OCR", c))

    n_ocr = sum(s[1] == "OCR" for s in samples)
    if n_ocr == 0:
        print("No OCR windows found — has ocr_retag.py finished? "
              "Expected files in tagged-ocr/.")
        return

    vocab_count = Counter()
    for _, _, c in samples:
        vocab_count.update(c)
    vocab = [t for t, _ in vocab_count.most_common(TOPK)]
    vindex = {t: i for i, t in enumerate(vocab)}

    X = np.zeros((len(samples), len(vocab)))
    for r, (_, _, c) in enumerate(samples):
        for t, n in c.items():
            if t in vindex:
                X[r, vindex[t]] = n
    X = X / (X.sum(1, keepdims=True) + 1e-12)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - Xn @ Xn.T

    authors = np.array([s[0] for s in samples])
    sources = np.array([s[1] for s in samples])
    iu, ju = np.triu_indices(len(samples), k=1)
    same_a = authors[iu] == authors[ju]
    same_s = sources[iu] == sources[ju]
    d = D[iu, ju]

    ss_same = d[same_a & same_s].mean()
    cs_same = d[same_a & ~same_s].mean()
    ss_diff = d[~same_a & same_s].mean()
    within, cross = same_s, ~same_s
    auc_within = roc_auc_score(same_a[within], -d[within]) if same_a[within].any() and (~same_a[within]).any() else float("nan")
    auc_cross = roc_auc_score(same_a[cross], -d[cross]) if same_a[cross].any() and (~same_a[cross]).any() else float("nan")

    n_clean = (sources == "clean").sum()
    print(f"POS channel, CONSISTENT BERT tagging")
    print(f"windows: {n_clean} clean, {n_ocr} OCR "
          f"(Clement {(authors=='clement').sum()}, Origen {(authors=='origen').sum()})\n")
    print(f"  SS_same (same author, same source) = {ss_same:.3f}")
    print(f"  CS_same (same author, cross source) = {cs_same:.3f}")
    print(f"  SS_diff (diff author, same source)  = {ss_diff:.3f}")
    print(f"  OCR inflation (CS_same - SS_same)   = {cs_same - ss_same:+.3f}")
    dom = "SOURCE DOMINATES" if cs_same >= ss_diff else "author still separ\u2011able"
    print(f"  CS_same vs SS_diff: {cs_same:.3f} vs {ss_diff:.3f}  \u2192 {dom}")
    print(f"  verification AUC within\u2192cross: {auc_within:.3f} \u2192 {auc_cross:.3f}")
    print("\n(Compare to the model-free quick test where POS CS_same was 0.862 under the "
          "tagset mismatch. A large drop here means that number was mostly artifact.)")


if __name__ == "__main__":
    main()
