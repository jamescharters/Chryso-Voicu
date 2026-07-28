"""
pc_conformal.py — open-set conformal authorship verification (NEW file).

The BDI verifier of pc_verify.py returns a win-fraction in [0,1]; this script wraps
that score in split-conformal prediction so each verdict comes with a distribution-free
error guarantee and a principled *reject option*. The candidate set is the seven
patristic authors present in the reference ("the room"). For a text q and candidate A
the nonconformity score is 1 - BDI(q, A). Calibrating the same-author nonconformity
on the reference lets us convert it to a conformal p-value and emit, at level alpha,
a prediction SET of candidates:

    * singleton  -> a confident attribution,
    * empty      -> the verifier declines: the author is (probably) not in the room,
    * multiple   -> the evidence does not separate those candidates.

We validate two guarantees empirically:
  (1) coverage  — for held-out works whose author IS a candidate, the true author sits
                  in the 1-alpha prediction set at least 1-alpha of the time;
  (2) abstention — for works whose author is NOT a candidate, the set is usually empty.

Then we apply the verifier to the pseudo-Chrysostoms. The two ground-truth groups whose
true author is absent from the candidate set (PC4 Apollinaris, PC12 an Anomoean) should
abstain; PC1 should admit Chrysostom, PC9/PC21 Severian.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from pc_verify import (load_matrix, bdi, name, PTA_CANDIDATES,
                       SEVERIAN, CHRYSOSTOM)

CAND = PTA_CANDIDATES                     # the seven authors "in the room"
ALPHAS = (0.10, 0.20)                     # 90% and 80% prediction sets
SEED = 0


def pval(nonconf, null):
    """Split-conformal p-value: P(same-author nonconformity >= this)."""
    null = np.asarray(null)
    return (1 + np.sum(null >= nonconf)) / (len(null) + 1)


def main():
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()
    files = ref["file"].to_numpy()
    rng = np.random.RandomState(SEED)
    docs = {a: Zref[au == a] for a in set(au)}
    afile = {a: files[au == a] for a in set(au)}

    cand = [a for a in CAND if docs.get(a) is not None and len(docs[a]) >= 3]
    print(f"candidate 'room': {', '.join(name(a) for a in cand)}")
    print(f"reference {Zref.shape[0]} works x {Zref.shape[1]} feats; PC {Zpc.shape[0]} texts\n")

    def nonconf(q, A, drop_file=None):
        """1 - BDI(q, author A), excluding all of A's segments from document `drop_file`
        so a query is never matched to its own segmented siblings."""
        tgt = docs[A]
        if drop_file is not None:
            tgt = tgt[afile[A] != drop_file]
        if len(tgt) == 0:
            return None
        return 1.0 - bdi(q, tgt, Zref[au != A], rng)

    # ---- same-author calibration: cross-DOCUMENT nonconformity for every candidate work ----
    # A single-document author cannot be self-verified across documents and is skipped
    # from calibration (it can still be a candidate that PC texts are scored against).
    cal = []          # list of (author, global_row_index, nonconformity)
    for A in cand:
        idx = np.where(au == A)[0]
        if len(set(files[idx])) < 2:
            continue
        for gi in idx:
            nc = nonconf(Zref[gi], A, drop_file=files[gi])
            if nc is not None:
                cal.append((A, gi, nc))
    cal_nc = np.array([c[2] for c in cal])
    cal_gi = np.array([c[1] for c in cal])
    n_cal_auth = len({c[0] for c in cal})
    print(f"[calibration] {len(cal)} cross-document same-author nonconformity scores "
          f"(mean {cal_nc.mean():.2f}); pooled null across {n_cal_auth} multi-document authors.\n")

    # ---- (1) coverage: leave-one-out over candidate works (true author IS in the room) ----
    print("=" * 68)
    print("(1) COVERAGE on held-out works whose author is a candidate")
    for alpha in ALPHAS:
        covered, sizes = [], []
        for A_true, gi, nc_same in cal:
            null = cal_nc[cal_gi != gi]                      # exclude the test point
            pset = []
            for A in cand:
                nc = nc_same if A == A_true else nonconf(Zref[gi], A)
                if nc is not None and pval(nc, null) > alpha:
                    pset.append(A)
            covered.append(A_true in pset)
            sizes.append(len(pset))
        print(f"  alpha={alpha:.2f} (target coverage {1-alpha:.0%}): "
              f"empirical {np.mean(covered):.0%}, mean set size {np.mean(sizes):.2f}")

    # ---- (2) abstention: works whose author is NOT a candidate should give an empty set ----
    print("\n" + "=" * 68)
    print("(2) ABSTENTION on works whose author is NOT in the room")
    non_idx = np.where(~np.isin(au, cand))[0]
    sample = rng.choice(non_idx, min(50, len(non_idx)), replace=False)
    for alpha in ALPHAS:
        empty = []
        for gi in sample:
            pset = [A for A in cand if pval(nonconf(Zref[gi], A), cal_nc) > alpha]
            empty.append(len(pset) == 0)
        print(f"  alpha={alpha:.2f}: correctly abstains on {np.mean(empty):.0%} "
              f"of {len(sample)} non-candidate works")

    # ---- apply to the pseudo-Chrysostoms (per text, then majority over the group) ----
    print("\n" + "=" * 68)
    print("(3) PSEUDO-CHRYSOSTOM prediction sets (alpha=0.10, 90%)")
    alpha = ALPHAS[0]
    rows = []
    for g in sorted(pc["author"].unique()):
        qi = pc.index[pc["author"] == g].to_numpy()
        # per-text prediction set, then keep candidates present in >= half the texts
        votes = {A: 0 for A in cand}
        for q in Zpc[qi]:
            for A in cand:
                if pval(nonconf(q, A), cal_nc) > alpha:
                    votes[A] += 1
        need = len(qi) / 2.0
        pset = [A for A in cand if votes[A] >= need]
        # order by vote strength
        pset = sorted(pset, key=lambda a: -votes[a])
        verdict = ("abstains" if not pset else
                   name(pset[0]) if len(pset) == 1 else
                   ", ".join(name(a) for a in pset))
        rows.append(dict(PC=g, n=len(qi), kind=("empty" if not pset else
                          "single" if len(pset) == 1 else "multi"),
                         set=verdict))
    out = pd.DataFrame(rows)
    os.makedirs("ocr-results", exist_ok=True)
    out.to_csv("ocr-results/pc-conformal.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    n_abst = (out["kind"] == "empty").sum()
    n_single = (out["kind"] == "single").sum()
    print(f"\n{n_single}/{len(out)} groups get a singleton attribution, "
          f"{n_abst}/{len(out)} abstain.")

    write_table(out)
    print("wrote paper/tables/conformal.tex")


def write_table(out, path="paper/tables/conformal.tex",
                show=("PC1", "PC4", "PC9", "PC12", "PC15", "PC16", "PC21")):
    short = {"Severianus Gabalensis": "Severian", "Joannes Chrysostomus": "Chrysostom",
             "Cyrillus Alexandrinus": "Cyril", "Theodoretus Cyrensis": "Theodoret",
             "Eusebius Caesariensis": "Eusebius", "Origenes": "Origen",
             "Athanasius Alexandrinus": "Athanasius"}

    def shorten(s):
        for k, v in short.items():
            s = s.replace(k, v)
        return s

    sel = out[out["PC"].isin(show)].set_index("PC").reindex(show).reset_index()
    lines = ["% auto-generated by pc_conformal.py -- do not edit by hand",
             "\\begin{tabular}{lll}", "\\toprule",
             "Group & $n$ & 90\\% prediction set \\\\", "\\midrule"]
    for _, r in sel.iterrows():
        cell = "\\emph{abstains}" if r["kind"] == "empty" else shorten(str(r["set"]))
        lines.append(f"{r['PC']} & {int(r['n'])} & {cell} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
