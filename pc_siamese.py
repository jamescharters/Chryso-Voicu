"""
pc_siamese.py — cross-check the impostors verdicts against the reproduced Siamese +
SNR-D neural verifier (NEW file). Answers "is the neural approach still needed?": if
the interpretable impostors method agrees with the neural one on the ground-truth
cases, interpretability costs no accuracy.

Loads the model trained by train.py, embeds the PC texts and the candidate authors,
and scores each ground-truth PC by its mean SNR-D distance to Severian vs Chrysostom
(lower = more similar). Prints the Siamese verdict next to the impostors one.
"""
from __future__ import annotations

import os
import glob
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

_orig = torch.load
def _load(f, *a, **k):
    k["weights_only"] = False
    return _orig(f, *a, **k)
torch.load = _load
import lightning_fabric.utilities.cloud_io as _cio
_cio._load = _load

from freestyl.dataset.dataframe_wrapper import DataframeWrapper
from freestyl.supervised.siamese.features.model import SiameseFeatureModule
from freestyl.supervised.siamese.features.data import make_dataloader as FeatureDataLoader
from corpus_balance import balance_corpus
from ocr_source import derive_source

GT = {"PC1": "Chrysostom (genuine)", "PC21": "Severian (proposed)",
      "PC4": "Apollinaris / not Sev.", "PC9": "contains Severian work",
      "PC12": "Anomoean (not Sev.)"}
SEVERIAN, CHRYSOSTOM = "pta0001", "pta0002"


def embeddings(model, dfw, device):
    dl = FeatureDataLoader(dfw, model=model, batch_size=32)
    vs = []
    with torch.no_grad():
        for batch in dl:
            xs = batch[0] if isinstance(batch, (list, tuple)) else batch
            v = model.forward(xs.to(device))
            v = v[0] if isinstance(v, (list, tuple)) else v
            vs.append(v.cpu())
    return torch.cat(vs)


def main():
    ckpt = max(glob.glob("models/*/model.ckpt"), key=os.path.getmtime)
    print(f"model: {ckpt}")
    model = SiameseFeatureModule.load_from_checkpoint(ckpt)
    model.eval()
    feats = list(model.hparams["features"] if isinstance(model.hparams, dict) else model.hparams.features)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)

    tlg = balance_corpus(pd.read_csv("tlg-features.csv"), seed=0, min_works=4, cap_works=30)
    tlg = tlg[derive_source(tlg).to_numpy() == "clean"].reset_index(drop=True)
    pc = pd.read_csv("pc-features.csv")

    ig_t = [c for c in tlg.columns if not c.startswith("$")]
    ig_p = [c for c in pc.columns if not c.startswith("$")]
    Dt = DataframeWrapper(tlg, label=("author", "title"), target="title", x_ignore=ig_t)
    Dp = DataframeWrapper(pc, label=("author", "title"), target="title", x_ignore=ig_p)
    Dt.update_features(feats); Dp.update_features(feats)
    Dt.normalized._dataframe = Dt.dataframe.fillna(0)
    Dp.normalized._dataframe = Dp.dataframe.fillna(0)

    tlg_vecs = embeddings(model, Dt, device)
    pc_vecs = embeddings(model, Dp, device)
    with torch.no_grad():
        D = model.distance(pc_vecs.to(device), tlg_vecs.to(device)).cpu().numpy()  # (n_pc, n_tlg)

    au = tlg["author"].to_numpy()
    sev_cols = np.where(au == SEVERIAN)[0]
    chr_cols = np.where(au == CHRYSOSTOM)[0]
    pc_auth = pc["author"].to_numpy()

    print(f"\n{'PC':>5} {'SNR-D Sev':>10} {'SNR-D Chry':>11} {'Siamese':>12}   scholarship")
    print("-" * 66)
    for g, schol in GT.items():
        rows = np.where(pc_auth == g)[0]
        d_sev = D[np.ix_(rows, sev_cols)].mean()
        d_chr = D[np.ix_(rows, chr_cols)].mean()
        verdict = "Severian" if d_sev < d_chr else "Chrysostom"
        print(f"{g:>5} {d_sev:10.3f} {d_chr:11.3f} {verdict:>12}   {schol}")


if __name__ == "__main__":
    main()
