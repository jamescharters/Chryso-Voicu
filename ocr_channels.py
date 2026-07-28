"""
ocr_channels.py — Phase 2, synthetic-OCR dose-response (NEW file; base untouched).

The real paired data (clean vs PG) conflates OCR noise with work selection: an
author's OCR side is a single Migne volume while its clean side spans many works.
To isolate the *pure* OCR effect we corrupt the SAME clean text at controlled
character-error rates and watch each feature channel drift. Because the reference
and the corrupted copy are the same text, any change is caused by the simulated OCR
alone.

Two tagger-independent channels are measured (POS would need re-tagging every
corrupted copy and is left for a follow-up):

    FW   1000 most-frequent function-word forms      (mfw.json)
    AFF  1000 most-frequent affix / char-trigrams     (mft.json, '_' = word edge)

Experiment A  feature drift: cosine distance between a chunk and its corrupted self,
              per channel, averaged over chunks, as a function of error rate p.
Experiment B  verification decay: with clean chunks as references and corrupted
              chunks as queries, the same/different-author AUC per channel vs p.

The corruption model is a controlled proxy for Greek print-OCR: diacritic loss
(the dominant real error), confusable-letter substitution, deletion, insertion, and
word-boundary (space) errors. It is a simulation, not a claim about any specific OCR
engine.
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

MFW = [w for w, _ in json.load(open("mfw.json"))][:1000]
MFT = [t for t, _ in json.load(open("mft.json"))][:1000]
MFW_IX = {w: i for i, w in enumerate(MFW)}
MFT_IX = {t: i for i, t in enumerate(MFT)}

TOKEN_RE = re.compile(r"[\u0300-\u036f\u0370-\u03ff\u1f00-\u1fff]+")
GREEK_BASE = "αβγδεζηθικλμνξοπρστυφχψω"
CONFUSE = {  # frequent Greek print-OCR letter confusions
    "σ": "ς", "ς": "σ", "ο": "ω", "ω": "ο", "ε": "σ", "ν": "υ", "υ": "ν",
    "γ": "τ", "τ": "γ", "λ": "χ", "η": "ν", "θ": "ϑ", "α": "ἁ", "ι": "ϊ",
}
WINDOW = 3000


def strip_diacritic(ch: str) -> str:
    d = unicodedata.normalize("NFD", ch)
    base = "".join(c for c in d if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", base) or ch


def corrupt(text: str, p: float, rng: np.random.RandomState) -> str:
    """Apply per-character OCR noise at rate p, plus coupled space errors."""
    out = []
    for ch in text:
        if rng.random() < p and ch.strip():
            e = rng.random()
            if e < 0.40:                       # diacritic loss (dominant)
                out.append(strip_diacritic(ch))
            elif e < 0.60:                     # confusable substitution
                out.append(CONFUSE.get(ch.lower(), ch))
            elif e < 0.80:                     # deletion
                continue
            else:                              # insertion
                out.append(ch)
                out.append(rng.choice(list(GREEK_BASE)))
        else:
            out.append(ch)
    s = "".join(out)
    if p > 0:                                  # word-boundary (space) errors
        s = re.sub(r" ", lambda m: "" if rng.random() < p * 0.4 else " ", s)
    return s


def fw_vector(text: str) -> np.ndarray:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    v = np.zeros(len(MFW))
    for t in toks:
        j = MFW_IX.get(t)
        if j is not None:
            v[j] += 1
    n = len(toks) or 1
    return v / n


def aff_vector(text: str) -> np.ndarray:
    toks = [t.lower() for t in TOKEN_RE.findall(text)]
    v = np.zeros(len(MFT))
    total = 0
    for t in toks:
        w = f"_{t}_"
        for k in range(len(w) - 2):
            total += 1
            j = MFT_IX.get(w[k:k + 3])
            if j is not None:
                v[j] += 1
    return v / (total or 1)


def cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float(a @ b) / (na * nb)


def load_clean_chunks():
    """Return list of (author, token_text_chunk) for clean Clement + Origen."""
    df = pd.read_csv("tlg-texts.csv", usecols=["file", "author", "full-text-raw"])
    want = {"clement": "tlg0555.", "origen": "tlg2042."}
    chunks = []
    for who, pref in want.items():
        for _, r in df[df["file"].astype(str).str.startswith(pref)].iterrows():
            toks = TOKEN_RE.findall(str(r["full-text-raw"]))
            for i in range(0, max(1, len(toks) - WINDOW + 1), WINDOW):
                w = toks[i:i + WINDOW]
                if len(w) >= WINDOW // 2:
                    chunks.append((who, " ".join(w)))
    return chunks


def main():
    rng = np.random.RandomState(0)
    chunks = load_clean_chunks()
    authors = np.array([a for a, _ in chunks])
    print(f"{len(chunks)} clean chunks "
          f"(Clement {int((authors=='clement').sum())}, Origen {int((authors=='origen').sum())})\n")

    # pre-compute clean reference vectors
    clean_fw = [fw_vector(t) for _, t in chunks]
    clean_aff = [aff_vector(t) for _, t in chunks]

    grid = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
    rows = []
    print(f"{'p':>5}  {'FW L1':>7} {'AFF L1':>7}  {'FW cos':>7} {'AFF cos':>7}   {'FW AUC':>7} {'AFF AUC':>7}")
    print("-" * 60)
    for p in grid:
        q_fw, q_aff = [], []
        drift_fw, drift_aff = [], []
        l1_fw, l1_aff = [], []
        for k, (_, t) in enumerate(chunks):
            ct = corrupt(t, p, rng)
            qf, qa = fw_vector(ct), aff_vector(ct)
            q_fw.append(qf); q_aff.append(qa)
            drift_fw.append(cos(clean_fw[k], qf))
            drift_aff.append(cos(clean_aff[k], qa))
            l1_fw.append(float(np.abs(clean_fw[k] - qf).sum()))
            l1_aff.append(float(np.abs(clean_aff[k] - qa).sum()))

        # verification AUC: corrupted query vs all clean references (excluding self)
        def auc_for(clean_vecs, query_vecs):
            ys, ds = [], []
            for qi in range(len(query_vecs)):
                for ri in range(len(clean_vecs)):
                    if qi == ri:
                        continue
                    ys.append(int(authors[qi] == authors[ri]))
                    ds.append(-cos(query_vecs[qi], clean_vecs[ri]))
            return roc_auc_score(ys, ds)

        a_fw = auc_for(clean_fw, q_fw)
        a_aff = auc_for(clean_aff, q_aff)
        rows.append(dict(p=p, fw_l1=np.mean(l1_fw), aff_l1=np.mean(l1_aff),
                         fw_cos=np.mean(drift_fw), aff_cos=np.mean(drift_aff),
                         fw_auc=a_fw, aff_auc=a_aff))
        print(f"{p:>5.2f}  {np.mean(l1_fw):>7.3f} {np.mean(l1_aff):>7.3f}  "
              f"{np.mean(drift_fw):>7.3f} {np.mean(drift_aff):>7.3f}   "
              f"{a_fw:>7.3f} {a_aff:>7.3f}")

    import os
    os.makedirs("ocr-results", exist_ok=True)
    pd.DataFrame(rows).to_csv("ocr-results/dose-response.csv", index=False)
    print("\nwrote ocr-results/dose-response.csv")
    r0, rN = rows[0], rows[-1]
    print(f"\nAt p=0.20 (L1 drift): FW {rN['fw_l1']:.3f} vs AFF {rN['aff_l1']:.3f}  "
          f"({'AFF' if rN['aff_l1']>rN['fw_l1'] else 'FW'} more OCR-fragile)")
    print(f"verification AUC decay 0\u2192{grid[-1]:.2f}: "
          f"FW {r0['fw_auc']:.3f}\u2192{rN['fw_auc']:.3f}, "
          f"AFF {r0['aff_auc']:.3f}\u2192{rN['aff_auc']:.3f}")
    print("note: cosine is direction-only and misses OCR's coverage loss; "
          "L1 is the honest drift, and the trained SNR-D model is the definitive verifier.")


if __name__ == "__main__":
    main()
