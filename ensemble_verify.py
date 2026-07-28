"""
ensemble_verify.py — aggregate an ensemble of trained verification models into a
per-PC same-author agreement table + Fleiss Kappa (Clérice & Glaise 2023 method).

Two modes:

  --from-models   (recommended)  load every completed model in models/*/,
                  treat each as an annotator, vote on each PC group.
                  Train the members first with train.py, e.g. via `make ensemble`.

  (default)       train N models inline then aggregate (self-contained but slow;
                  prefer train.py + --from-models so models are reused, not lost).

Outputs:
  ensemble-votes.csv     model × PC fraction of within-group same-author pairs
  ensemble-summary.csv   per PC agreement + verdict
"""
import os, sys, glob, argparse

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl

_orig_load = torch.load
def _load(f, *a, **k):
    k["weights_only"] = False
    return _orig_load(f, *a, **k)
torch.load = _load

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader

PRECISION_DEFAULT = 0.90
device = "mps" if torch.backends.mps.is_available() else "cpu"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--from-models", dest="from_models", action="store_true",
                   help="aggregate pre-trained models/*/ instead of training inline")
    p.add_argument("--precision", type=float, default=PRECISION_DEFAULT)
    p.add_argument("--n", type=int, default=10, help="models to train (inline mode)")
    return p.parse_args()


def threshold_for_precision(test_pairs, target):
    tp = test_pairs.dropna(subset=["IsAPair", "Distance"]).sort_values("Distance")
    tpn = fpn = 0
    thr = 0.0
    for _, row in tp.iterrows():
        if row["IsAPair"]:
            tpn += 1
        else:
            fpn += 1
        if tpn / (tpn + fpn) >= target:
            thr = row["Distance"]
    return thr


def pc_fractions_for_model(model, thr, pc_df, pc_groups, groups):
    """Return {group: fraction of within-group pairs below thr} for one model."""
    feats = list(model.hparams.features)
    DFW = DataframeWrapper(pc_df, target="title", label=["author", "title"],
                           x_ignore=[c for c in pc_df.columns if c not in feats])
    DFW.update_features(feats)
    DFW.normalized._dataframe = DFW.dataframe.fillna(0)
    model.eval().to(device)
    with torch.no_grad():
        vecs = []
        for batch in FeatureDataLoader(DFW, model=model, batch_size=32):
            xs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            out = model.forward(xs)
            out = out[0] if isinstance(out, (list, tuple)) else out
            vecs.append(out.cpu())
        pc_vecs = torch.cat(vecs)

    fracs = {}
    for g in groups:
        idx = [i for i, gg in enumerate(pc_groups) if gg == g]
        if len(idx) < 2:
            fracs[g] = np.nan
            continue
        sub = pc_vecs[idx].to(device)
        with torch.no_grad():
            dmat = model.distance(sub, sub).cpu().numpy()
        iu = np.triu_indices(len(idx), k=1)
        fracs[g] = float((dmat[iu] < thr).mean())
    return fracs


def fleiss_kappa(mat):
    n_items = mat.shape[0]
    n_raters = mat.sum(axis=1)[0]
    if n_raters < 2:
        return float("nan")
    p_j = mat.sum(axis=0) / (n_items * n_raters)
    P_i = (np.square(mat).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_e = np.square(p_j).sum()
    return (P_i.mean() - P_e) / (1 - P_e) if (1 - P_e) else float("nan")


def aggregate(model_fracs, precision):
    """model_fracs: list of {group: fraction}. Produce votes, summary, kappa."""
    groups = sorted({g for mf in model_fracs for g in mf}, key=lambda g: (len(g), g))
    rows = []
    votes = {g: [] for g in groups}
    for m_i, mf in enumerate(model_fracs):
        for g in groups:
            frac = mf.get(g, np.nan)
            rows.append({"model": m_i, "PC": g, "frac": frac})
            if not np.isnan(frac):
                votes[g].append(1 if frac >= 0.5 else 0)
    pd.DataFrame(rows).to_csv("ensemble-votes.csv", index=False)

    kappa_groups = [g for g in groups if votes[g]]
    mat = np.array([[sum(votes[g]), len(votes[g]) - sum(votes[g])] for g in kappa_groups])
    kappa = fleiss_kappa(mat) if len(kappa_groups) > 1 else float("nan")

    summary = []
    for g in groups:
        vs = votes[g]
        if not vs:
            summary.append({"PC": g, "models_same": 0, "models_total": 0,
                            "agreement": np.nan, "verdict": "single sample"})
            continue
        agree = sum(vs) / len(vs)
        verdict = ("CONFIRMED" if agree >= 0.8 else
                   "leaning same" if agree >= 0.5 else
                   "leaning different" if agree > 0.2 else "NOT confirmed")
        summary.append({"PC": g, "models_same": sum(vs), "models_total": len(vs),
                        "agreement": round(agree, 2), "verdict": verdict})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv("ensemble-summary.csv", index=False)

    print("\n" + "=" * 60)
    print(f"ENSEMBLE VERIFICATION — {len(model_fracs)} models, precision {precision}")
    print(f"Fleiss Kappa across models: {kappa:.3f}")
    print("=" * 60)
    print(summary_df.to_string(index=False))


def run_from_models(precision):
    pc_df = pd.read_csv("pc-features.csv")
    pc_groups = pc_df["author"].tolist()
    groups = sorted(set(pc_groups), key=lambda g: (len(g), g))

    dirs = [d for d in sorted(glob.glob("models/*/"))
            if os.path.exists(os.path.join(d, "model.ckpt"))]
    if not dirs:
        sys.exit("No trained models in models/. Train some first: make ensemble N=10")
    print(f"Aggregating {len(dirs)} trained models")

    model_fracs = []
    for d in dirs:
        model = SiameseFeatureModule.load_from_checkpoint(os.path.join(d, "model.ckpt"))
        test_pairs = pd.read_csv(os.path.join(d, "test-results.csv"))
        thr = threshold_for_precision(test_pairs, precision)
        model_fracs.append(pc_fractions_for_model(model, thr, pc_df, pc_groups, groups))
        del model
        import gc; gc.collect()
    aggregate(model_fracs, precision)


def run_inline(n, precision):
    """Self-contained fallback: train n models then aggregate (slow)."""
    import train as trainer_mod
    from freestyl.supervised.siamese import train_dataframewrappers, get_df_prediction
    pc_df = pd.read_csv("pc-features.csv")
    pc_groups = pc_df["author"].tolist()
    groups = sorted(set(pc_groups), key=lambda g: (len(g), g))
    model_fracs = []
    for m in range(n):
        seed = 1000 + m
        pl.seed_everything(seed, workers=True)
        wraps, _ = trainer_mod.build_wrappers("tlg-features.csv", seed)
        model, tr = train_dataframewrappers(
            train=wraps["train"], dev=wraps["dev"], test=wraps["test"],
            optim="Adam", accelerator=device, learning_rate=1e-4, margin=1,
            dimension=64, loss="stn_contrastive", pos_strategy="easy",
            neg_strategy="semihard", sample=2, batch_size=64, gpus=1, dropout=0.3,
            min_epochs=100, max_epochs=100, miner_for_dev=True, patience=20, split_dim=None,
        )
        test_pairs = get_df_prediction(tr, model=model, compared=wraps["test"], threshold=6)
        thr = threshold_for_precision(test_pairs, precision)
        model_fracs.append(pc_fractions_for_model(model, thr, pc_df, pc_groups, groups))
        del model, tr
        import gc; gc.collect()
    aggregate(model_fracs, precision)


if __name__ == "__main__":
    a = parse_args()
    if a.from_models:
        run_from_models(a.precision)
    else:
        run_inline(a.n, a.precision)
