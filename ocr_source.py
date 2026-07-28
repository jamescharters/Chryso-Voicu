"""
ocr_source.py — Phase 0 of the transmission-noise study (NEW file; does not touch
the reproduction layer).

Labels each row of a feature matrix by its transmission SOURCE:

    clean   critically edited / First1KGreek / PTA text
    OCR     Patrologia Graeca, ingested from optical character recognition
            (file names look like ``PG009_tagged_text*``)

and enumerates the authors present in BOTH sources — the paired set that powers the
whole study. Author labels differ across sources (clean says "Origenes", the PG
volume says "Origen; Hippolytus"), so a light normaliser aligns them.

Import it (`from ocr_source import derive_source, normalize_author, paired_authors`)
or run it directly to print the inventory and write ocr-results/paired-manifest.csv.
"""
from __future__ import annotations

import os
import re

import pandas as pd


def derive_source(df: pd.DataFrame) -> pd.Series:
    """Return a 'clean' / 'OCR' label per row. PG (OCR) rows have file names that
    start with ``PG``; everything else (First1KGreek, PTA) is a clean edition."""
    return df["file"].astype(str).str.startswith("PG").map({True: "OCR", False: "clean"})


def normalize_author(author: str) -> str:
    """Collapse the cross-source spelling differences so the same person matches.

    Takes the first author before ';' or '(', lowercases, and folds the common
    Greek-Latin variants and see-epithets ("Origenes" -> "origen", drop
    " of Alexandria"). Deliberately conservative: only well-attested equivalences.
    """
    a = str(author).lower().strip()
    a = re.split(r"[;(]", a)[0].strip()          # first author; drop "et al."/"(15th c.)"
    a = a.replace("origenes", "origen")
    a = a.replace(" of alexandria", "").replace(" of caesarea", "")
    return a


def paired_authors(df: pd.DataFrame) -> dict[str, dict[str, list[int]]]:
    """Map each author present in both sources to its clean / OCR row indices."""
    src = derive_source(df)
    na = df["author"].map(normalize_author)
    clean = set(na[src == "clean"])
    ocr = set(na[src == "OCR"])
    out: dict[str, dict[str, list[int]]] = {}
    for name in sorted(clean & ocr):
        out[name] = {
            "clean": df.index[(na == name) & (src == "clean")].tolist(),
            "OCR":   df.index[(na == name) & (src == "OCR")].tolist(),
        }
    return out


def main(features_csv: str = "tlg-features.csv", out_dir: str = "ocr-results") -> None:
    df = pd.read_csv(features_csv)
    src = derive_source(df)
    print(f"{features_csv}: {len(df)} rows  ({(src=='clean').sum()} clean, "
          f"{(src=='OCR').sum()} OCR)")

    pairs = paired_authors(df)
    print(f"\nPaired authors (present in clean AND OCR): {len(pairs)}")
    rows = []
    for name, ix in pairs.items():
        nc, no = len(ix["clean"]), len(ix["OCR"])
        print(f"  {name:24s} clean={nc:3d}  OCR={no:3d}")
        for source, idxs in ix.items():
            for i in idxs:
                rows.append({"author": name, "source": source,
                             "file": df.at[i, "file"], "title": df.at[i, "title"]})

    os.makedirs(out_dir, exist_ok=True)
    manifest = os.path.join(out_dir, "paired-manifest.csv")
    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"\nWrote {manifest} ({len(rows)} rows).")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
