"""
pc_context.py — cross-reference the BDI verification verdicts against published
scholarly attribution (CPG + current consensus). NEW file; base pipeline untouched.

Scholarly attributions were compiled from Pinakes (IRHT), the Clavis Patrum Graecorum
(CPG II), Wikidata P7988, and the editorial notes embedded in the TEI headers
(TLG/Migne + editors such as Nautin, Liebaert, Wenger, Uthemann). Only where a work
is placed in a named author's corpus is it treated as "ground truth"; the large
anonymous majority have no consensus author and our verdicts there are proposals.
"""
from __future__ import annotations

import os
import pandas as pd

# CPG number(s), current scholarly attribution, firmness, and whether that
# attribution is checkable ground truth (a NAMED author) vs anonymous.
SCHOLARSHIP = {
    "PC1":  ("4410", "John Chrysostom (genuine)", "firm", "Chrysostom"),
    "PC2":  ("4566/4567", "anonymous ps.-Chrysostom", "disputed", None),
    "PC3":  ("4579/4580/4641", "anonymous ps.-Chrysostom", "disputed", None),
    "PC4":  ("4606-4612", "Apollinaris of Laodicea (serm.1-3); anon (4-7)", "probable", "not-Severian"),
    "PC5":  ("4603", "anonymous ps.-Chrysostom", "disputed", None),
    "PC6":  ("4615-4617", "anonymous ps.-Chrysostom", "disputed", None),
    "PC7":  ("4619", "anonymous ps.-Chrysostom", "disputed", None),
    "PC8":  ("4620/4621", "anonymous ps.-Chrysostom", "disputed", None),
    "PC9":  ("4215+4657/4659/4660", "MIXED: Severian (4215, firm) + Cappadocian + anon", "mixed", "Severian-partial"),
    "PC10": ("4516", "anonymous ps.-Chrysostom", "disputed", None),
    "PC12": ("2082/2083", "Anomoean/Neo-Arian (Liebaert) - NOT Chrys/Severian", "firm", "not-Severian"),
    "PC13": ("4506/4525/epiph.", "anon; epiphany hom. 'Severian?' (Wenger)", "disputed", None),
    "PC14": ("4618", "anonymous ps.-Chrysostom", "disputed", None),
    "PC15": ("4589/4669/4969", "anonymous ps.-Chrysostom", "disputed", None),
    "PC16": ("4701/4576/4544/4545", "anonymous ps.-Chrysostom", "disputed", None),
    "PC17": ("4626/4585", "anonymous ps.-Chrysostom", "disputed", None),
    "PC18": ("4587/4577", "anonymous ps.-Chrysostom", "disputed", None),
    "PC19": ("4601", "anonymous ps.-Chrysostom", "disputed", None),
    "PC20": ("4631/4547/4757/4571/4658", "anonymous ps.-Chrysostom", "disputed", None),
    "PC20b": ("4629/4588/4699+", "MIXED: Cappadocian (3) + anon", "mixed", None),
    "PC21": ("4564", "Severian of Gabala (proposed)", "probable", "Severian"),
    "PCX":  ("4667?", "anonymous ps.-Chrysostom", "disputed", None),
}


def our_verdict(row):
    sev, chry = row["BDI_Severian"], row["BDI_Chrysostom"]
    best = row["best_PTA"]
    if chry > sev and chry >= 0.5:
        return "Chrysostom"
    if not row["sev_wins_PTA"]:
        return f"other ({best.split()[0]})"
    if sev >= 0.85:
        return "Severian (confident)"
    if sev >= 0.5:
        return "Severian (weak)"
    return "inconclusive"


def agreement(verdict, truth):
    if truth is None:
        return "-- (anonymous; proposal)"
    v = verdict.lower()
    if truth == "Chrysostom":
        return "AGREE" if "chrysostom" in v else "DISAGREE"
    if truth == "Severian":
        return "AGREE" if "severian" in v else "DISAGREE"
    if truth == "not-Severian":
        return "AGREE" if "severian" not in v else "DISAGREE"
    if truth == "Severian-partial":
        return "partial" if "severian" in v else "miss"
    return "?"


def main():
    bdi = pd.read_csv("ocr-results/pc-verification.csv").set_index("PC")
    rows = []
    for pc, (cpg, attr, firm, truth) in SCHOLARSHIP.items():
        if pc not in bdi.index:
            continue
        r = bdi.loc[pc]
        v = our_verdict(r)
        rows.append(dict(PC=pc, CPG=cpg, scholarship=attr, firm=firm,
                         BDI_Sev=r["BDI_Severian"], BDI_Chry=r["BDI_Chrysostom"],
                         our_verdict=v, check=agreement(v, truth)))
    out = pd.DataFrame(rows)
    os.makedirs("ocr-results", exist_ok=True)
    out.to_csv("ocr-results/pc-context.csv", index=False)
    pd.set_option("display.width", 220, "display.max_colwidth", 48)

    gt = out[out["check"].isin(["AGREE", "DISAGREE", "partial", "miss"])]
    print("GROUND-TRUTH cases (works with a named scholarly attribution):\n")
    print(gt[["PC", "CPG", "scholarship", "BDI_Sev", "BDI_Chry", "our_verdict", "check"]].to_string(index=False))
    agree = (gt["check"].isin(["AGREE", "partial"])).sum()
    print(f"\n  method agrees with scholarship on {agree}/{len(gt)} checkable groups.")

    print("\n\nANONYMOUS majority (our verdicts are new, calibrated proposals):\n")
    an = out[~out["check"].isin(["AGREE", "DISAGREE", "partial", "miss"])]
    print(an[["PC", "CPG", "BDI_Sev", "BDI_Chry", "our_verdict"]].to_string(index=False))
    print("\nwrote ocr-results/pc-context.csv")


if __name__ == "__main__":
    main()
