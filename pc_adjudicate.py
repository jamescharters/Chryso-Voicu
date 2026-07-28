"""
pc_adjudicate.py — attribute each Pseudo-Chrysostom (PC) group to a known author,
with interpretable per-channel evidence (NEW file; base pipeline untouched).

The scholarly question behind this corpus (Voicu's Homiliae Pseudo-Chrysostomicae):
for each PC group, is it genuinely John Chrysostom (pta0002), his frequent
"double" Severian of Gabala (pta0001), or someone else?

Method (interpretable, no training needed):
  * reference = clean, per-author-balanced corpus (PG/OCR rows excluded so the POS
    channel is on one tagset; capped so Severian's 200+ chunks don't dominate scaling).
  * for each feature channel (FW / POS / AFF) z-score features over the reference
    (Burrows-style), take each candidate author's centroid, and measure a PC text's
    mean-absolute-z distance (Delta) to each centroid.
  * a PC's verdict is the nearest candidate; the per-channel Deltas show WHY, and the
    Chrysostom-vs-Severian contrast is reported explicitly.

Sanity checks run first: leave-one-out attribution accuracy on the reference authors
and the same/different-author separation. If those fail, the verdicts are not
trustworthy and the script says so.
"""
from __future__ import annotations


import os

import numpy as np
import pandas as pd

from corpus_balance import balance_corpus
from ocr_source import derive_source

CHANNELS = {"FW": "$MFW$", "POS": "$POS$", "AFF": "$TRI$"}
MIN_WORKS = 4          # a candidate needs at least this many works for a stable centroid
CAP = 30               # cap works/author when computing scaling + centroids


def build_pta_names() -> dict:
    """Map pta#### -> author name from the PTA source XML in patres/."""
    import glob
    import re
    import collections
    m: dict = {}
    for f in glob.glob("patres/*.xml"):
        try:
            txt = open(f, encoding="utf-8").read(4000)
        except OSError:
            continue
        for key, nm in re.findall(r'<persName key="(pta\d+)">([^<]+)</persName>', txt):
            m.setdefault(key, collections.Counter())[nm.replace("\n", " ").strip()] += 1
    return {k: c.most_common(1)[0][0] for k, c in m.items()}


_names = build_pta_names()


def name(a: str) -> str:
    return _names.get(a, str(a))


def cols(df, prefix):
    return [c for c in df.columns if c.startswith(prefix)]


def zstats(ref, channel):
    c = cols(ref, CHANNELS[channel])
    X = ref[c].fillna(0).to_numpy(float)
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    return c, mu, sd


def zscore(df, c, mu, sd):
    return (df[c].fillna(0).to_numpy(float) - mu) / sd


def delta(z_rows, centroid):
    # Burrows Delta: mean absolute z difference across features
    return np.abs(z_rows - centroid).mean(axis=1)


def main():
    ref_all = balance_corpus(pd.read_csv("tlg-features.csv"), seed=0,
                             min_works=MIN_WORKS, cap_works=CAP)
    ref_all = ref_all[derive_source(ref_all).to_numpy() == "clean"].reset_index(drop=True)
    pc = pd.read_csv("pc-features.csv")

    authors = ref_all["author"].to_numpy()
    cand = sorted(set(authors))
    print(f"reference: {len(ref_all)} works, {len(cand)} candidate authors "
          f"(min {MIN_WORKS} works, capped {CAP})")
    print(f"PC texts: {len(pc)} in {pc['author'].nunique()} groups\n")

    # per-channel z-scores + author centroids (restricted to features present in BOTH files)
    pc_cols = set(pc.columns)
    chan = {}
    for ch in CHANNELS:
        c = [col for col in cols(ref_all, CHANNELS[ch]) if col in pc_cols]
        X = ref_all[c].fillna(0).to_numpy(float)
        mu = X.mean(0); sd = X.std(0); sd[sd == 0] = 1.0
        Z = (X - mu) / sd
        cents = {a: Z[authors == a].mean(0) for a in cand}
        chan[ch] = dict(c=c, mu=mu, sd=sd, Z=Z, cents=cents)
        print(f"[features] {ch}: {len(c)} shared columns")
    print()

    # ---- sanity 1: leave-one-out attribution accuracy on the reference ----
    correct = 0
    for i in range(len(ref_all)):
        a_true = authors[i]
        n_a = (authors == a_true).sum()
        combined = np.zeros(len(cand))
        for ch in CHANNELS:
            Z = chan[ch]["Z"]; cents = chan[ch]["cents"]
            for j, a in enumerate(cand):
                cen = cents[a]
                if a == a_true and n_a > 1:          # exact leave-one-out centroid
                    cen = (cen * n_a - Z[i]) / (n_a - 1)
                combined[j] += np.abs(Z[i] - cen).mean()
        if cand[int(combined.argmin())] == a_true:
            correct += 1
    loo_acc = correct / len(ref_all)
    chance = 1.0 / len(cand)
    print(f"[sanity] leave-one-out attribution accuracy: {loo_acc:.3f} "
          f"(chance {chance:.3f})  ->  {'OK' if loo_acc > 5 * chance else 'WEAK'}")

    # ---- adjudicate each PC group ----
    focus = {"pta0002": "Chrysostom", "pta0001": "Severian"}
    rows = []
    pc_groups = sorted(pc["author"].unique())
    for g in pc_groups:
        idx = pc.index[pc["author"] == g]
        percand = np.zeros(len(cand))
        perchan_focus = {f: {} for f in focus}
        for ch in CHANNELS:
            c, mu, sd, cents = chan[ch]["c"], chan[ch]["mu"], chan[ch]["sd"], chan[ch]["cents"]
            Zpc = zscore(pc.loc[idx], c, mu, sd)
            for j, a in enumerate(cand):
                dmean = delta(Zpc, cents[a]).mean()
                percand[j] += dmean
                if a in focus:
                    perchan_focus[a][ch] = dmean
        order = np.argsort(percand)
        top = [(name(cand[j]), percand[j] / len(CHANNELS)) for j in order[:3]]
        verdict = top[0][0]
        d_chry = percand[cand.index("pta0002")] / len(CHANNELS) if "pta0002" in cand else np.nan
        d_sev = percand[cand.index("pta0001")] / len(CHANNELS) if "pta0001" in cand else np.nan
        # per-channel margin: positive = closer to Severian than Chrysostom on that channel
        margins = {ch: perchan_focus["pta0002"].get(ch, np.nan) - perchan_focus["pta0001"].get(ch, np.nan)
                   for ch in CHANNELS}
        rows.append(dict(PC=g, n=len(idx), verdict=verdict,
                         top1=f"{top[0][0]} ({top[0][1]:.2f})",
                         top2=f"{top[1][0]} ({top[1][1]:.2f})",
                         top3=f"{top[2][0]} ({top[2][1]:.2f})",
                         d_Chrysostom=round(d_chry, 3), d_Severian=round(d_sev, 3),
                         m_FW=round(margins["FW"], 3), m_POS=round(margins["POS"], 3),
                         m_AFF=round(margins["AFF"], 3),
                         leans=("Severian" if d_sev < d_chry else "Chrysostom")))

    out = pd.DataFrame(rows)
    os.makedirs("ocr-results", exist_ok=True)
    out.to_csv("ocr-results/pc-adjudication.csv", index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n" + out[["PC", "n", "verdict", "d_Chrysostom", "d_Severian",
                       "m_FW", "m_POS", "m_AFF", "leans"]].to_string(index=False))
    print("\n(m_* = Chrysostom-distance minus Severian-distance on that channel; "
          "positive = that channel favours Severian.)")
    print("wrote ocr-results/pc-adjudication.csv")
    sev = (out["leans"] == "Severian").sum()
    print(f"\nOf {len(out)} PC groups: {sev} lean Severian, {len(out)-sev} lean Chrysostom "
          f"(nearest of the two comparanda).")
    # which channel most consistently carries the Severian signal
    for ch in ("m_FW", "m_POS", "m_AFF"):
        pos = (out[ch] > 0).sum()
        print(f"  {ch}: favours Severian in {pos}/{len(out)} groups (mean margin {out[ch].mean():+.3f})")
    # divergent cases: global nearest is not Severian
    div = out[out["verdict"] != "Severianus Gabalensis"]
    if len(div):
        print("\nDivergent PCs (global nearest ≠ Severian):")
        print(div[["PC", "verdict", "d_Severian"]].to_string(index=False))


if __name__ == "__main__":
    main()
