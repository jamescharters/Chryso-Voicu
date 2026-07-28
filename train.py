"""
train.py — train ONE Siamese verification model with a reproducible, config-keyed
output directory. Never overwrites a model trained with a *different* configuration,
and skips retraining an identical one unless --force is given.

Each model lands in:
    models/<slug>/
        model.ckpt        the trained weights
        config.json       the exact hyperparameters + feature counts
        test-results.csv  labelled test-pair distances (for thresholding)
        metrics.json      test AUC, dev/test sizes, epochs, wall time

The <slug> encodes every parameter that affects the result, e.g.
    POS100_FW1000_TRI1000_stn_LR1e-04_S64_B64_D0.3_SMP2_EP100_seed1000
so different configs / seeds coexist and identical ones are detected.

Usage:
    venv/bin/python train.py --seed 1000
    venv/bin/python train.py --seed 1001 --epochs 40
    venv/bin/python train.py --seed 1000 --force        # retrain in place
"""
import os, sys, json, time, glob, random, argparse, shutil
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

# checkpoint (de)serialisation with non-tensor globals — our own trusted files
_orig_load = torch.load
def _load(f, *a, **k):
    k["weights_only"] = False
    return _orig_load(f, *a, **k)
torch.load = _load

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese import train_dataframewrappers, get_df_prediction

# ── Default configuration (the paper's winning row) ──────────────────────────
DEFAULTS = dict(
    features="tlg-features.csv",
    loss="stn_contrastive",
    lr=1e-4, size=64, batch=64, dropout=0.3, sample=2,
    epochs=100, patience=20,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train one Siamese verification model.")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--features", default=DEFAULTS["features"])
    p.add_argument("--loss", default=DEFAULTS["loss"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--size", type=int, default=DEFAULTS["size"])
    p.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--sample", type=int, default=DEFAULTS["sample"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    p.add_argument("--accelerator", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="retrain even if the slug exists")
    return p.parse_args()

def make_slug(a, n_pos, n_fw, n_tri):
    return (f"POS{n_pos}_FW{n_fw}_TRI{n_tri}_{a.loss.replace('_','')}"
            f"_LR{a.lr:g}_S{a.size}_B{a.batch}_D{a.dropout}_SMP{a.sample}"
            f"_EP{a.epochs}_seed{a.seed}")

def build_wrappers(features_csv, seed):
    df = pd.read_csv(features_csv).sample(frac=1, random_state=seed)
    keep = [c for c in df.columns if c.startswith(("$POS$", "$MFW$", "$TRI$"))]
    ignore = [c for c in df.columns if c not in keep]
    n_pos = sum(c.startswith("$POS$") for c in keep)
    n_fw  = sum(c.startswith("$MFW$") for c in keep)
    n_tri = sum(c.startswith("$TRI$") for c in keep)

    authors = sorted(df["author"].unique())
    r = random.Random(seed); r.shuffle(authors)
    n_test = max(1, len(authors) * 10 // 100)
    test_a, dev_a = set(authors[:n_test]), set(authors[n_test:2*n_test])
    splits = {
        "train": df[~df.author.isin(test_a | dev_a)],
        "dev":   df[df.author.isin(dev_a)],
        "test":  df[df.author.isin(test_a)],
    }
    wraps = {}
    for name, d in splits.items():
        w = DataframeWrapper(d, target="author", label=["author", "title"], x_ignore=ignore)
        wraps[name] = w
    wraps["dev"].update_features(wraps["train"].features)
    wraps["test"].update_features(wraps["train"].features)
    for w in wraps.values():
        w.normalized._dataframe = w.dataframe.fillna(0)
    return wraps, (n_pos, n_fw, n_tri)

def main():
    a = parse_args()
    wraps, (n_pos, n_fw, n_tri) = build_wrappers(a.features, a.seed)
    slug = make_slug(a, n_pos, n_fw, n_tri)
    out = os.path.join("models", slug)

    if os.path.exists(os.path.join(out, "model.ckpt")) and not a.force:
        print(f"[skip] {slug} already trained. Use --force to retrain.")
        return
    if a.force and os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out, exist_ok=True)

    pl.seed_everything(a.seed, workers=True)
    print(f"[train] {slug}  ({n_pos}+{n_fw}+{n_tri} feats, {a.epochs} epochs, {a.accelerator})")
    t0 = time.time()
    model, trainer = train_dataframewrappers(
        train=wraps["train"], dev=wraps["dev"], test=wraps["test"],
        optim="Adam", accelerator=a.accelerator,
        learning_rate=a.lr, margin=1, dimension=a.size, loss=a.loss,
        pos_strategy="easy", neg_strategy="semihard",
        sample=a.sample, batch_size=a.batch, gpus=1, dropout=a.dropout,
        min_epochs=a.epochs, max_epochs=a.epochs, miner_for_dev=True,
        patience=a.patience, split_dim=None,
    )
    wall = time.time() - t0

    # save weights + config + labelled test pairs + metrics
    trainer.save_checkpoint(os.path.join(out, "model.ckpt"))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({**vars(a), "n_pos": n_pos, "n_fw": n_fw, "n_tri": n_tri,
                   "slug": slug}, f, indent=2)

    test_pairs = get_df_prediction(trainer, model=model, compared=wraps["test"], threshold=6)
    test_pairs.to_csv(os.path.join(out, "test-results.csv"), index=False)
    tp = test_pairs.dropna(subset=["IsAPair", "Distance"])
    auc = roc_auc_score(tp["IsAPair"], -tp["Distance"]) if len(tp) else float("nan")
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump({"test_auc": auc, "wall_seconds": round(wall, 1),
                   "n_train": len(wraps["train"].dataframe),
                   "n_test_pairs": len(tp)}, f, indent=2)

    print(f"[done]  {slug}  test AUC={auc:.3f}  ({wall/60:.1f} min)")

if __name__ == "__main__":
    main()
