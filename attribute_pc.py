"""
Compare PC texts against the training corpus using the trained Siamese model.
Produces pc-vs-corpus.csv with the closest training-corpus author for each PC text.

Usage: venv/bin/python attribute_pc.py
"""
import os
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl

# Checkpoint was saved with non-tensor objects (LabelEncoder, numpy arrays).
# Force weights_only=False — checkpoint is our own trusted source.
_orig_torch_load = torch.load
def _torch_load_unsafe(f, *args, **kwargs):
    kwargs['weights_only'] = False  # force, don't use setdefault
    return _orig_torch_load(f, *args, **kwargs)
torch.load = _torch_load_unsafe

# Patch PL's cloud_io as well since it imports torch.load directly
import lightning_fabric.utilities.cloud_io as _cloud_io
_cloud_io._load = _torch_load_unsafe

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader
from corpus_balance import balance_corpus

# ── Load model (latest checkpoint) ──────────────────────────────────────────
import glob
ckpts = sorted(glob.glob("lightning_logs/version_*/checkpoints/*.ckpt"))
ckpt = ckpts[-1]
print(f"Loading model: {ckpt}")
model = SiameseFeatureModule.load_from_checkpoint(ckpt)
model.eval()

trainer = pl.Trainer(accelerator="mps", devices=1, logger=False, enable_progress_bar=True)

# ── Load features ─────────────────────────────────────────────────────────────
print("Loading features...")
tlg_df = pd.read_csv("tlg-features.csv")
# fair representation: repair lost PTA labels + balance authors so the attribution
# corpus is not dominated by a few prolific writers (same control as training).
tlg_df = balance_corpus(tlg_df, seed=1000)
pc_df  = pd.read_csv("pc-features.csv")

# Use only features the model knows
features = model.hparams.features if hasattr(model.hparams, 'features') else None

# Build DataframeWrappers
x_ignore_tlg = [c for c in tlg_df.columns if not (c.startswith('$') or c in ('tokens',))]
x_ignore_pc  = [c for c in pc_df.columns  if not (c.startswith('$') or c in ('tokens',))]

DFW_tlg = DataframeWrapper(tlg_df, label=("author", "title"), target="title", x_ignore=x_ignore_tlg)
DFW_pc  = DataframeWrapper(pc_df,  label=("author", "title"), target="title", x_ignore=x_ignore_pc)

# Align features
if features:
    DFW_tlg.update_features(features)
    DFW_pc.update_features(features)
else:
    # Use intersection of features
    shared = list(set(DFW_tlg.features) & set(DFW_pc.features))
    DFW_tlg.update_features(shared)
    DFW_pc.update_features(shared)

# Apply normalization (same as assign_normalization in Step 07)
DFW_tlg.normalized._dataframe = DFW_tlg.dataframe.fillna(0)
DFW_pc.normalized._dataframe  = DFW_pc.dataframe.fillna(0)

print(f"Training corpus: {len(tlg_df)} samples, PC corpus: {len(pc_df)} samples")
print(f"Shared features: {len(DFW_tlg.features)}")

# ── Get embeddings directly (skip PL trainer.predict overhead) ───────────────
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader
import torch

model.eval()
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
model = model.to(device)

def get_embeddings(dfw):
    dl = FeatureDataLoader(dfw, model=model, batch_size=32)
    all_vecs = []
    with torch.no_grad():
        for batch in dl:
            xs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            vecs = model.forward(xs)
            if isinstance(vecs, (list, tuple)):
                vecs = vecs[0]
            all_vecs.append(vecs.cpu())
    embeddings = torch.cat(all_vecs)  # keep as tensor for model.distance
    labels = dfw.get_labels(as_list=True)
    return embeddings, labels

print("Computing training corpus embeddings...")
tlg_vecs, tlg_labels = get_embeddings(DFW_tlg)

print("Computing PC text embeddings...")
pc_vecs, pc_labels = get_embeddings(DFW_pc)

# ── Compute pairwise distances using the model's own SNR-D distance ──────────
# The model was trained with stn_contrastive loss; its embedding space is
# optimised for SNR-D geometry, so we must score with model.distance, not cosine.
print("Computing SNR-D distances...")
with torch.no_grad():
    # pytorch-metric-learning distances return a full (n_query, n_ref) matrix
    dist_matrix = model.distance(pc_vecs.to(device), tlg_vecs.to(device))
dists = dist_matrix.cpu().numpy()  # shape: (n_pc, n_tlg)

# SNR-D: LOWER = more similar (same-author ≈ 0). Confirm orientation via sort.

# ── Build result: best N training matches per PC text ────────────────────────
N_TOP = 5
rows = []
for pc_i, pc_label in enumerate(pc_labels):
    # Split "PC1 - InGenesimSermones1" into author and title parts
    parts = pc_label.split(" - ", 1)
    pc_author = parts[0].strip()
    pc_title  = parts[1].strip() if len(parts) > 1 else ""
    top_idx = np.argsort(dists[pc_i])[:N_TOP]
    for rank, tlg_i in enumerate(top_idx):
        tlg_label = tlg_labels[tlg_i]
        tlg_parts = tlg_label.split(" - ", 1)
        rows.append({
            "PC_label":    pc_label,
            "PC_author":   pc_author,
            "PC_title":    pc_title,
            "Rank":        rank + 1,
            "Best_label":  tlg_label,
            "Best_author": tlg_parts[0].strip(),
            "Best_title":  tlg_parts[1].strip() if len(tlg_parts) > 1 else "",
            "Distance":    float(dists[pc_i, tlg_i]),
        })

result = pd.DataFrame(rows)
result.to_csv("pc-vs-corpus.csv", index=False)
print("Saved → pc-vs-corpus.csv")

# ── Print summary: closest match per PC text ─────────────────────────────────
print("\n=== Closest training-corpus author per PC text ===")
best = result[result.Rank == 1].sort_values("PC_label")
for _, row in best.iterrows():
    print(f"  {row['PC_label']:<55} → {row['Best_author']:<40} (dist={row['Distance']:.3f})")
