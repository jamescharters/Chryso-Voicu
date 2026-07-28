"""
Reproduce the paper's authorship-VERIFICATION deliverable (Clérice & Glaise 2023,
slide 14): for each pseudo-Chrysostom group, the fraction of within-group text
pairs the model classifies as same-author, at a range of precision thresholds.

A threshold is chosen so that, on the labelled test pairs, positive predictions
(distance < threshold) achieve a target precision. That same threshold is then
applied to the within-PC pairs.

Outputs:
  pc-verification.csv   — PC × precision matrix (fraction of same-author pairs)
  pc-verification.png   — heatmap
"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import glob
import itertools
import numpy as np
import pandas as pd
import torch

# Load checkpoint with non-tensor globals (trusted, our own file)
_orig_load = torch.load
def _load(f, *a, **k):
    k['weights_only'] = False
    return _orig_load(f, *a, **k)
torch.load = _load
import lightning_fabric.utilities.cloud_io as _cio
_cio._load = _load

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader

# ── Load model ───────────────────────────────────────────────────────────────
ckpt = sorted(glob.glob("lightning_logs/version_*/checkpoints/*.ckpt"))[-1]
print(f"Model: {ckpt}")
model = SiameseFeatureModule.load_from_checkpoint(ckpt)
model.eval()
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model = model.to(device)

# ── Derive precision → distance thresholds from labelled test pairs ───────────
test = pd.read_csv(sorted(glob.glob("*test-results.csv"))[-1])
test = test.dropna(subset=["IsAPair", "Distance"]).copy()
test["IsAPair"] = test["IsAPair"].astype(bool)

def threshold_for_precision(target_precision: float) -> float:
    """Smallest distance cutoff whose positive set reaches target precision."""
    ordered = test.sort_values("Distance")  # ascending: closest first
    tp = 0
    fp = 0
    best_thr = 0.0
    for _, row in ordered.iterrows():
        if row["IsAPair"]:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        if precision >= target_precision:
            best_thr = row["Distance"]
    return best_thr

PRECISIONS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0]
thresholds = {p: threshold_for_precision(p) for p in PRECISIONS}
print("Precision → distance thresholds:")
for p, t in thresholds.items():
    print(f"  {int(p*100)}%: {t:.4f}")

# ── Embed PC samples ─────────────────────────────────────────────────────────
pc_df = pd.read_csv("pc-features.csv")
x_ignore = [c for c in pc_df.columns if not (c.startswith('$') or c == 'tokens')]
DFW = DataframeWrapper(pc_df, label=("author", "title"), target="title", x_ignore=x_ignore)
if hasattr(model.hparams, 'features') and model.hparams.features:
    DFW.update_features(model.hparams.features)
DFW.normalized._dataframe = DFW.dataframe.fillna(0)

with torch.no_grad():
    vecs = []
    for batch in FeatureDataLoader(DFW, model=model, batch_size=32):
        xs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
        out = model.forward(xs)
        if isinstance(out, (list, tuple)):
            out = out[0]
        vecs.append(out.cpu())
    pc_vecs = torch.cat(vecs)

pc_groups = pc_df["author"].tolist()  # PC1, PC20b, ...

# ── For each PC group, fraction of within-group pairs below each threshold ────
groups = sorted(set(pc_groups), key=lambda g: (len(g), g))
matrix = {}
pair_counts = {}
for g in groups:
    idx = [i for i, gg in enumerate(pc_groups) if gg == g]
    if len(idx) < 2:
        # Single sample — no within-group pairs to compare
        matrix[g] = {p: np.nan for p in PRECISIONS}
        pair_counts[g] = 0
        continue
    sub = pc_vecs[idx].to(device)
    with torch.no_grad():
        dmat = model.distance(sub, sub).cpu().numpy()
    # take upper-triangle (unique pairs)
    iu = np.triu_indices(len(idx), k=1)
    pair_dists = dmat[iu]
    pair_counts[g] = len(pair_dists)
    matrix[g] = {p: float((pair_dists < thresholds[p]).mean()) for p in PRECISIONS}

verif = pd.DataFrame(matrix).T  # rows = PC groups, cols = precision
verif.columns = [f"P{int(p*100)}" for p in PRECISIONS]
verif["n_pairs"] = pd.Series(pair_counts)
verif.index.name = "PC"
verif.to_csv("pc-verification.csv")
print("\nSaved → pc-verification.csv")

# ── Heatmap ──────────────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = verif[[c for c in verif.columns if c.startswith("P")]].astype(float)
    # order columns high→low precision like the paper
    plot_df = plot_df[[f"P{int(p*100)}" for p in PRECISIONS]]
    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
    sns.heatmap(plot_df.T, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1,
                ax=ax, cbar_kws={"label": "fraction of within-group pairs same-author"})
    ax.set_xlabel("Pseudo-Chrysostom group")
    ax.set_ylabel("Precision threshold (from test set)")
    ax.set_title("Authorship verification: within-group same-author agreement")
    plt.tight_layout()
    plt.savefig("pc-verification.png")
    print("Saved → pc-verification.png")
except Exception as e:
    print(f"(heatmap skipped: {e})")

# ── Text summary: verification verdict per group ─────────────────────────────
print("\n=== Verification verdict (fraction same-author at 90% precision) ===")
for g in groups:
    v = matrix[g][0.90]
    n = pair_counts[g]
    if n == 0:
        verdict = "single sample — n/a"
    elif v >= 0.8:
        verdict = "CONFIRMED (coherent group)"
    elif v >= 0.4:
        verdict = "partial"
    elif np.isnan(v):
        verdict = "n/a"
    else:
        verdict = "not confirmed"
    val = "n/a" if np.isnan(v) else f"{v:.2f}"
    print(f"  {g:<6} ({n:>3} pairs)  {val:>5}  {verdict}")
