"""
corpus_balance.py — fair-representation controls for the training corpus.

Authorship verification with a Siamese / metric-learning objective is sensitive to
class (author) imbalance. The number of same-author pairs an author contributes
grows with the square of its work count, so a handful of prolific authors — Galen
(253 works), the Septuagint (199), Libanius (159) — would otherwise dominate the
positive-pair signal and the model would learn *their* style rather than a general
notion of authorial identity.

Three standard controls are applied, in order:

  1. repair     Recover author labels lost to an extraction bug. The PTA works
                whose <persName> failed to parse arrive with a blank author, yet
                their identity survives in the `textgroup` column (e.g. pta0001 =
                Severian of Gabala). Without this, 300+ texts by different authors
                are pooled under a single empty label and treated as one "author".

  2. min_works  Drop authors with fewer than `min_works` texts. An author with one
                text cannot form a genuine same-author pair, and the M-per-class
                sampler would pair its lone text with itself — a degenerate,
                zero-distance positive that teaches nothing and destabilises
                training.

  3. cap_works  Randomly down-sample any author with more than `cap_works` texts to
                exactly `cap_works`, flattening the long tail so no single author
                dominates the corpus or any author-disjoint train/dev/test split.

Every step is deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_MIN_WORKS = 2
DEFAULT_CAP_WORKS = 20


def repair_authors(df: pd.DataFrame) -> pd.DataFrame:
    """Fill blank/whitespace author labels from the `textgroup` id, then drop any
    row that still has no usable author."""
    if "author" not in df.columns:
        return df
    df = df.copy()
    df["author"] = df["author"].astype(str)
    blank = df["author"].str.strip().eq("") | df["author"].str.strip().eq("nan")
    if blank.any() and "textgroup" in df.columns:
        df.loc[blank, "author"] = df.loc[blank, "textgroup"].astype(str).str.strip()
        blank = df["author"].str.strip().eq("") | df["author"].str.strip().eq("nan")
    return df[~blank]


def balance_corpus(
    df: pd.DataFrame,
    seed: int,
    min_works: int = DEFAULT_MIN_WORKS,
    cap_works: int | None = DEFAULT_CAP_WORKS,
    repair: bool = True,
) -> pd.DataFrame:
    """Return a fair-representation view of `df`: repaired labels, singleton
    authors removed, and prolific authors down-sampled to `cap_works`."""
    if repair:
        df = repair_authors(df)

    if min_works and min_works > 1:
        counts = df["author"].value_counts()
        df = df[df["author"].isin(counts[counts >= min_works].index)]

    if cap_works:
        rng = np.random.RandomState(seed)
        keep = []
        for _, g in df.groupby("author", sort=False):
            idx = g.index.to_numpy()
            if len(idx) > cap_works:
                idx = rng.choice(idx, cap_works, replace=False)
            keep.extend(idx.tolist())
        df = df.loc[keep]

    return df
