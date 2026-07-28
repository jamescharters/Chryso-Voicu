"""
pc_composite.py — intra-document composite-authorship detection (NEW file).

The corpus's fragment structure, which inflated the naive verification numbers via
within-document leakage, is here turned into a tool. A genuinely single-author homily
has internally coherent fragments; a composite one (florilegium, interpolated core,
two hands) does not. We recompute the interpretable FW+AFF features per fragment with
the SAME code used for the reference (ocr_channels), calibrate the single-author
coherence baseline, and scan each pseudo-Chrysostom homily.

Length is handled in a principled way:
  (1) fragments are a FIXED size W tokens, so their feature variance matches the
      calibration fragments (a shorter fragment is noisier and would look falsely
      incoherent);
  (2) a homily's internal coherence is scored against a null of genuine single-author
      documents, matched on fragment count k where the sample allows, pooled otherwise;
  (3) homilies with only two fragments (a single pair) are marked provisional.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pc_verify import load_matrix, PTA_CANDIDATES, name
from ocr_channels import fw_vector, aff_vector, TOKEN_RE

SEED = 0
TAB = "paper/tables/composite.tex"


def featurize(text):
    return np.concatenate([fw_vector(text), aff_vector(text)])


def dmean(V):
    """Mean pairwise mean-absolute distance among the (z-scored) fragment rows."""
    n = len(V)
    if n < 2:
        return None
    tot, c = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += np.abs(V[i] - V[j]).mean(); c += 1
    return tot / c


def main():
    ref, pc, _, _ = load_matrix()
    au = ref["author"].to_numpy()
    files = ref["file"].to_numpy()
    rng = np.random.RandomState(SEED)

    # interpretable FW+AFF for every reference segment (identical code to the PC side)
    R = np.array([featurize(t) for t in ref["modified_text"]])
    mu, sd = R.mean(0), R.std(0); sd[sd == 0] = 1.0
    Rz = (R - mu) / sd

    def zfeat(text):
        return (featurize(text) - mu) / sd

    toklen = np.array([len(TOKEN_RE.findall(str(t))) for t in ref["modified_text"]])
    W = int(np.median(toklen))
    print(f"reference {len(ref)} segments; fragment size W={W} tokens (median segment length)")

    # ---- validation: genuine vs same-author-mix vs different-author-mix ----
    doc_rows = {f: np.where(files == f)[0] for f in set(files) if (files == f).sum() >= 3}
    gen = np.array([dmean(Rz[idx]) for idx in doc_rows.values()])
    gen_sizes = [len(idx) for idx in doc_rows.values()]
    auth_docs = defaultdict(dict)
    for f in set(files):
        idx = np.where(files == f)[0]
        auth_docs[au[idx[0]]][f] = idx
    multi = [a for a in auth_docs if len(auth_docs[a]) >= 3]
    allA = list(auth_docs)

    sm, dm = [], []
    for _ in range(400):
        a = rng.choice(multi); k = min(int(rng.choice(gen_sizes)), len(auth_docs[a]))
        fs = rng.choice(list(auth_docs[a]), k, replace=False)
        sm.append(dmean(Rz[[int(rng.choice(auth_docs[a][f])) for f in fs]]))
        k2 = min(int(rng.choice(gen_sizes)), len(allA))
        aa = rng.choice(allA, k2, replace=False)
        dm.append(dmean(Rz[[int(rng.choice(np.where(au == a2)[0])) for a2 in aa]]))
    sm, dm = np.array(sm), np.array(dm)

    def auc(pos, neg):
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        return roc_auc_score(y, np.r_[-pos, -neg])

    print(f"[validate] internal dist: genuine {gen.mean():.3f} | same-author-mix {sm.mean():.3f} "
          f"| diff-author-mix {dm.mean():.3f}")
    print(f"[validate] AUC genuine-vs-composite {auc(gen, dm):.3f}; "
          f"authorship-specific (same-mix vs diff-mix) {auc(sm, dm):.3f}\n")

    # genuine single-author null indexed by fragment count
    null_by_k = defaultdict(list)
    for idx in doc_rows.values():
        null_by_k[len(idx)].append(dmean(Rz[idx]))

    def composite_p(d, k):
        pool = null_by_k.get(k, [])
        base, tag = (np.array(pool), f"k={k}") if len(pool) >= 15 else (gen, "pooled")
        return (1 + np.sum(base >= d)) / (len(base) + 1), tag

    # candidate centroids for interpretable fragment attribution
    cand = [a for a in PTA_CANDIDATES if (au == a).sum() > 0]
    cent = {a: Rz[au == a].mean(0) for a in cand}

    def attribute(v):
        return min(cand, key=lambda a: np.abs(v - cent[a]).mean())

    # ---- scan each pseudo-Chrysostom homily ----
    rows = []
    for _, r in pc.iterrows():
        toks = TOKEN_RE.findall(str(r["modified_text"]))
        frags = [toks[i:i + W] for i in range(0, len(toks), W)]
        if len(frags) >= 2 and len(frags[-1]) < W / 2:      # fold a tiny tail into the previous
            frags[-2] += frags[-1]; frags = frags[:-1]
        k = len(frags)
        if k < 2:
            rows.append(dict(group=r["author"], title=str(r["title"])[:26], k=k,
                             dist=np.nan, p=np.nan, tag="short", seq=[]))
            continue
        V = np.array([zfeat(" ".join(f)) for f in frags])
        d = dmean(V)
        p, tag = composite_p(d, k)
        seq = [name(attribute(v)) for v in V]
        rows.append(dict(group=r["author"], title=str(r["title"])[:26], k=k,
                         dist=round(d, 3), p=round(p, 3), tag=tag, seq=seq))

    out = pd.DataFrame(rows)
    os.makedirs("ocr-results", exist_ok=True)
    out.drop(columns="seq").to_csv("ocr-results/pc-composite.csv", index=False)
    assessable = out[out["k"] >= 2]
    flagged = assessable[assessable["p"] < 0.12].sort_values("p")
    print(f"assessable homilies (>=2 fragments): {len(assessable)}/{len(out)}; "
          f"flagged composite (p<0.12): {len(flagged)}")
    switches = out[out["seq"].apply(lambda s: len(set(s)) > 1)]
    print(f"homilies whose fragments do not all attribute to one author: {len(switches)}\n")
    print("Composite candidates (internal coherence worse than a genuine single-author text):")
    for _, r in flagged.iterrows():
        seam = " ".join(s[:4] for s in r["seq"])
        prov = " [provisional: single pair]" if int(r["k"]) == 2 else ""
        print(f"  {r['group']:>5} {r['title']:<26} k={int(r['k'])} p={r['p']:.3f}  {seam}{prov}")

    write_table(flagged)
    print(f"\nwrote {TAB}")


def write_table(flagged, path=TAB, topn=8):
    sel = flagged.head(topn)
    lines = ["% auto-generated by pc_composite.py -- do not edit by hand",
             "\\begin{tabular}{llccl}", "\\toprule",
             "Group & Homily & $k$ & $p$ & Fragment lean \\\\", "\\midrule"]
    for _, r in sel.iterrows():
        seam = " ".join(s[:4] for s in r["seq"])
        t = str(r["title"]).replace("&", "\\&")
        lines.append(f"{r['group']} & \\emph{{{t}}} & {int(r['k'])} & {r['p']:.2f} & {seam} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
