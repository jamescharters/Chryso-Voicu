"""
ocr_confound.py — Phase 1 of the transmission-noise study (NEW file; imports the
reproduction layer read-only, edits nothing).

Question Q1: does a stylometric comparison cluster texts by their TRANSMISSION
SOURCE (clean edition vs OCR'd Patrologia Graeca) rather than by their AUTHOR?

Using the authors we hold in both forms (Clement, Origen), every pair of chunks is
labelled on two axes — same/different author, and same/different source — giving
four distance populations:

    SS_same   same author, same source     (the honest within-author baseline)
    CS_same   same author, cross source    (what OCR adds on top)
    SS_diff   different author, same source (ordinary between-author distance)
    CS_diff   different author, cross source

Two diagnostics fall out:
  * OCR inflation      mean(CS_same) - mean(SS_same)   (Q2, magnitude)
  * source dominance   mean(CS_same) vs mean(SS_diff): if same-author-across-source
                       is as far apart as different-authors-same-source, source has
                       overwhelmed authorship — a batch effect.
And a verification AUC computed WITHIN vs ACROSS source: if the cross-source AUC
collapses, OCR breaks the verifier.

This `--quick` path is model-free: per-channel relative frequencies + cosine
distance. It is a fast proxy for the trained SNR-D model (Phase 1 full run) and is
enough to tell whether the effect is real before spending training time.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from corpus_balance import repair_authors
from ocr_source import derive_source, normalize_author

CHANNELS = {"FW": "$MFW$", "POS": "$POS$", "AFF": "$TRI$"}
PAIRED = {"clement", "origen"}


def _cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def _l1(mat):
    s = mat.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return mat / s


def _cosine_dist(X):
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return 1.0 - Xn @ Xn.T


def channel_distance(df, channel):
    """Pairwise cosine-distance matrix for one channel (or the equal-weight mean
    of all three when channel == 'ALL')."""
    if channel == "ALL":
        mats = [channel_distance(df, c) for c in CHANNELS]
        return np.mean(mats, axis=0)
    X = _l1(df[_cols(df, CHANNELS[channel])].fillna(0).to_numpy(float))
    return _cosine_dist(X)


def confound_report(df):
    authors = df["author_n"].to_numpy()
    sources = df["source"].to_numpy()
    n = len(df)
    iu, ju = np.triu_indices(n, k=1)
    same_author = authors[iu] == authors[ju]
    same_source = sources[iu] == sources[ju]

    print(f"{n} paired-author chunks "
          f"({(sources=='clean').sum()} clean, {(sources=='OCR').sum()} OCR)\n")
    header = f"{'channel':6s} {'SS_same':>8s} {'CS_same':>8s} {'SS_diff':>8s} " \
             f"{'AUC_within':>11s} {'AUC_cross':>10s}"
    print(header)
    print("-" * len(header))

    results = {}
    for channel in ["FW", "POS", "AFF", "ALL"]:
        D = channel_distance(df, channel)
        d = D[iu, ju]
        ss_same = d[same_author & same_source]
        cs_same = d[same_author & ~same_source]
        ss_diff = d[~same_author & same_source]

        within = same_source
        cross = ~same_source
        auc_within = roc_auc_score(same_author[within], -d[within]) if same_author[within].any() and (~same_author[within]).any() else float("nan")
        auc_cross = roc_auc_score(same_author[cross], -d[cross]) if same_author[cross].any() and (~same_author[cross]).any() else float("nan")

        results[channel] = dict(ss_same=ss_same.mean(), cs_same=cs_same.mean(),
                                ss_diff=ss_diff.mean(), auc_within=auc_within, auc_cross=auc_cross)
        print(f"{channel:6s} {ss_same.mean():8.3f} {cs_same.mean():8.3f} "
              f"{ss_diff.mean():8.3f} {auc_within:11.3f} {auc_cross:10.3f}")

    print("\nReading the table:")
    a = results["ALL"]
    print(f"  OCR inflation (CS_same - SS_same) = {a['cs_same'] - a['ss_same']:+.3f}")
    dom = "SOURCE DOMINATES (batch effect)" if a["cs_same"] >= a["ss_diff"] else "author still separ\u2011able"
    print(f"  same-author-cross-source vs different-author-same-source: "
          f"{a['cs_same']:.3f} vs {a['ss_diff']:.3f}  \u2192 {dom}")
    print(f"  verification AUC drop clean\u2192cross-source: "
          f"{a['auc_within']:.3f} \u2192 {a['auc_cross']:.3f} "
          f"({a['auc_within'] - a['auc_cross']:+.3f})")
    # which channel is most OCR-fragile
    frag = max(("FW", "POS", "AFF"), key=lambda c: results[c]["cs_same"] - results[c]["ss_same"])
    robust = min(("FW", "POS", "AFF"), key=lambda c: results[c]["cs_same"] - results[c]["ss_same"])
    print(f"  most OCR-fragile channel: {frag}   most OCR-robust channel: {robust}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="tlg-features.csv")
    ap.add_argument("--quick", action="store_true",
                    help="model-free feature-space proxy (no training)")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    df = repair_authors(df)
    df["source"] = derive_source(df).to_numpy()
    df["author_n"] = df["author"].map(normalize_author)
    df = df[df["author_n"].isin(PAIRED)].reset_index(drop=True)

    confound_report(df)


if __name__ == "__main__":
    main()
