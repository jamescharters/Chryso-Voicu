"""
pc_figures.py — generate figures and full-result tables for the paper (NEW file).

Produces, from the same data and verifier as pc_verify.py:
  * paper/figures/heatmap.pdf       PC groups x candidate authors, BDI scores
  * paper/figures/calibration.pdf   BDI score distributions (Severian self vs others)
  * paper/tables/pc-full.tex        the complete 22-group results table (appendix)
  * paper/tables/internal.tex       internal-consistency of the PC groups

Internal consistency answers a question the original study raised -- whether each PC
"group" is a single authorial unit -- by measuring how tightly a group's own texts
cluster relative to the corpus.
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.decomposition import PCA

from pc_verify import load_matrix, score_group, bdi, NAMES

FIGDIR = "paper/figures"
TABDIR = "paper/tables"
CANDS = ["pta0001", "pta0002", "pta0005", "pta0004", "pta0003", "pta0022", "pta0007"]
SHORT = {"pta0001": "Severian", "pta0002": "Chrysostom", "pta0005": "Cyril",
         "pta0004": "Theodoret", "pta0003": "Eusebius", "pta0022": "Athanasius",
         "pta0007": "Origen"}


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(TABDIR, exist_ok=True)
    ref, pc, Zref, Zpc = load_matrix()
    au = ref["author"].to_numpy()
    docs = {a: Zref[au == a] for a in set(au)}
    groups = sorted(pc["author"].unique(), key=lambda g: (len(g), g))
    rng = np.random.RandomState(0)

    # ---- BDI matrix: PC groups x candidates ----
    cand = [c for c in CANDS if c in docs]
    M = np.zeros((len(groups), len(cand)))
    for i, g in enumerate(groups):
        qv = Zpc[pc.index[pc["author"] == g].to_numpy()]
        for j, a in enumerate(cand):
            M[i, j] = score_group(qv, docs[a], Zref[au != a], np.random.RandomState(0))

    # Figure 1: heatmap
    fig, ax = plt.subplots(figsize=(5.4, 6.6))
    im = ax.imshow(M, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(cand))); ax.set_xticklabels([SHORT[a] for a in cand], rotation=40, ha="right")
    ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups, fontsize=7)
    for i in range(len(groups)):
        for j in range(len(cand)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M[i, j] > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="BDI verification score")
    ax.set_title("Bootstrap Distance Impostors: PC groups vs candidates")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/heatmap.pdf"); plt.close(fig)

    # ---- Figure 2: calibration ----
    sev = np.where(au == "pta0001")[0]
    oth = np.where(au != "pta0001")[0]
    imp = Zref[oth]
    self_s = [bdi(Zref[i], Zref[sev[sev != i]], imp, rng) for i in rng.choice(sev, min(25, len(sev)), replace=False)]
    other_s = [bdi(Zref[i], Zref[sev], Zref[np.setdiff1d(oth, [i])], rng) for i in rng.choice(oth, 40, replace=False)]
    pc_s = M[:, cand.index("pta0001")]
    fig, ax = plt.subplots(figsize=(5.2, 3.1))
    bins = np.linspace(0, 1, 21)
    ax.hist(other_s, bins=bins, alpha=0.65, label="other authors vs Severian", color="0.6")
    ax.hist(self_s, bins=bins, alpha=0.65, label="genuine Severian (self)", color="0.2")
    ax.scatter(pc_s, np.full_like(pc_s, -0.6), marker="|", s=80, color="C3", label="PC groups")
    ax.set_xlabel("BDI score against Severian"); ax.set_ylabel("count")
    ax.legend(fontsize=7, loc="upper center"); ax.set_title("Calibration of the Severian verifier")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/calibration.pdf"); plt.close(fig)

    # ---- centroids of candidate authors + PC groups (for dendrogram + PCA) ----
    labels, vecs, kinds = [], [], []
    for a in cand:
        labels.append(SHORT[a]); vecs.append(Zref[au == a].mean(0)); kinds.append("author")
    for g in groups:
        labels.append(g); vecs.append(Zpc[pc.index[pc["author"] == g].to_numpy()].mean(0)); kinds.append("pc")
    V = np.vstack(vecs)
    kinds = np.array(kinds)

    # ---- Figure 3: dendrogram (Burrows-Delta = Manhattan on z-scores, average linkage) ----
    Zl = linkage(V, method="average", metric="cityblock")
    fig, ax = plt.subplots(figsize=(5.6, 7.2))
    dd = dendrogram(Zl, labels=labels, orientation="right", ax=ax,
                    color_threshold=0, above_threshold_color="0.4", leaf_font_size=7.5)
    lab_kind = {lab: k for lab, k in zip(labels, kinds)}
    for t in ax.get_yticklabels():
        if lab_kind.get(t.get_text()) == "author":
            t.set_fontweight("bold"); t.set_color("C0")
    ax.set_xlabel("Burrows's Delta"); ax.set_title("Clustering of PC groups and candidate authors")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/dendrogram.pdf"); plt.close(fig)

    # ---- Figure 4: PCA projection (axes from the reference corpus) ----
    pca = PCA(n_components=2, random_state=0).fit(Zref)
    XY = pca.transform(V)
    ev = pca.explained_variance_ratio_ * 100
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    is_a = kinds == "author"
    ax.scatter(XY[~is_a, 0], XY[~is_a, 1], s=22, c="0.55", marker="o", label="PC group")
    ax.scatter(XY[is_a, 0], XY[is_a, 1], s=60, c="C0", marker="s", label="candidate author")
    for (x, y), lab, k in zip(XY, labels, kinds):
        ax.annotate(lab, (x, y), fontsize=7, fontweight="bold" if k == "author" else "normal",
                    color="C0" if k == "author" else "0.35",
                    xytext=(3, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}%)"); ax.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
    ax.legend(fontsize=7, loc="best"); ax.set_title("Stylometric space: PC groups and candidate authors")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/pca.pdf"); plt.close(fig)

    # ---- internal consistency of PC groups ----
    def cos_pairwise(X):
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        D = 1 - Xn @ Xn.T
        iu = np.triu_indices(len(X), 1)
        return D[iu]
    # baseline: distances between random cross-group PC text pairs
    allpc = Zpc
    Xn = allpc / (np.linalg.norm(allpc, axis=1, keepdims=True) + 1e-12)
    Dall = 1 - Xn @ Xn.T
    gl = pc["author"].to_numpy()
    cross = Dall[np.triu_indices(len(allpc), 1)]
    cross_mean = float(cross.mean())
    irows = []
    for g in groups:
        idx = pc.index[pc["author"] == g].to_numpy()
        if len(idx) < 2:
            continue
        within = cos_pairwise(Zpc[idx]).mean()
        irows.append((g, len(idx), within))
    idf = pd.DataFrame(irows, columns=["PC", "n", "within"]).sort_values("within")

    # ---- write LaTeX tables ----
    ctx = pd.read_csv("ocr-results/pc-context.csv").set_index("PC")
    ver = pd.read_csv("ocr-results/pc-verification.csv").set_index("PC")
    with open(f"{TABDIR}/pc-full.tex", "w") as f:
        f.write("\\begin{tabular}{llccl}\n\\toprule\n")
        f.write("PC & CPG & BDI$_{\\text{Sev}}$ & BDI$_{\\text{Chry}}$ & Scholarship \\\\\n\\midrule\n")
        for g in sorted(ctx.index, key=lambda x: (len(x), x)):
            c = ctx.loc[g]
            sev = ver.loc[g, "BDI_Severian"] if g in ver.index else float("nan")
            chry = ver.loc[g, "BDI_Chrysostom"] if g in ver.index else float("nan")
            schol = str(c["scholarship"]).replace("&", "\\&").replace("_", " ")[:34]
            f.write(f"{g} & {c['CPG']} & {sev:.2f} & {chry:.2f} & {schol} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    with open(f"{TABDIR}/internal.tex", "w") as f:
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("PC & \\#texts & mean within-group dist. \\\\\n\\midrule\n")
        for _, r in idf.iterrows():
            f.write(f"{r['PC']} & {int(r['n'])} & {r['within']:.3f} \\\\\n")
        f.write("\\midrule\n")
        f.write(f"cross-group baseline & -- & {cross_mean:.3f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    print(f"cross-group baseline mean distance: {cross_mean:.3f}")
    print("most dispersed PC groups (candidates for composite authorship):")
    print(idf.tail(4).to_string(index=False))
    print("most cohesive PC groups:")
    print(idf.head(4).to_string(index=False))
    print(f"\nwrote {FIGDIR}/heatmap.pdf, {FIGDIR}/calibration.pdf, "
          f"{TABDIR}/pc-full.tex, {TABDIR}/internal.tex")


if __name__ == "__main__":
    main()
