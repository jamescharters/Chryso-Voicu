"""
bert_embed.py — frozen document embeddings from the SuperPeitho Ancient-Greek BERT.

Run with the tagging environment (has flair + transformers):

    bert-env/bin/python bert_embed.py

This is *inference only*: no fine-tuning. It reuses the very transformer that
produced the paper's POS tags (SuperPeitho-FLAIR-v2 / LM SuperPeitho-v1), so the
neural channel is maximally comparable to the hand-crafted ones. It reads the text
list written by pc_bert.py (ocr-results/bert-input.csv) and writes one mean-pooled
768-d vector per text to ocr-results/bert-emb.npz, keyed by a content hash.
"""
import os
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch

# The FLAIR model embeds a relative tokenizer path ('../LM/SuperPeitho-v1');
# patch the HF repo-id validator so the local path is accepted (as in tag-in-xml.py).
import huggingface_hub.utils._validators as _hf_v
_orig_validate = _hf_v.validate_repo_id
def _patched_validate(repo_id, **kw):
    if str(repo_id).startswith(".") or os.path.isabs(str(repo_id)):
        return
    _orig_validate(repo_id, **kw)
_hf_v.validate_repo_id = _patched_validate

import flair
from flair.models import SequenceTagger

MAXLEN, MAXCHUNK, BATCH = 510, 8, 16       # <=8 windows of 510 content tokens per text
IN_CSV, OUT_NPZ = "ocr-results/bert-input.csv", "ocr-results/bert-emb.npz"


def get_hf(embeddings):
    """Pull the HuggingFace model + tokenizer out of a FLAIR embedding object."""
    cands = [embeddings]
    cands += list(getattr(embeddings, "embeddings", []) or [])
    for e in cands:
        if hasattr(e, "model") and hasattr(e, "tokenizer"):
            return e.model, e.tokenizer
    raise RuntimeError("no transformer embedding found inside the FLAIR model")


def main():
    dev = torch.device("mps" if torch.backends.mps.is_available()
                       else "cuda:0" if torch.cuda.is_available() else "cpu")
    flair.device = dev
    root = os.getcwd()
    os.chdir("SuperPeitho-FLAIR-v2")
    tagger = SequenceTagger.load("final-model.pt")
    os.chdir(root)
    model, tok = get_hf(tagger.embeddings)
    model = model.to(dev).eval()
    dim = model.config.hidden_size
    cls, sep = tok.cls_token_id, tok.sep_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    print(f"device={dev} hidden={dim}")

    inp = pd.read_csv(IN_CSV)
    keys = inp["key"].astype(str).tolist()
    texts = inp["text"].fillna("").astype(str).tolist()
    print(f"embedding {len(texts)} texts")

    # build chunk list (doc index, token ids)
    doc_chunks = []
    for i, t in enumerate(texts):
        ids = tok.encode(t, add_special_tokens=False, truncation=False) if t.strip() else []
        if not ids:
            ids = [tok.unk_token_id if tok.unk_token_id is not None else pad]
        for j in range(0, min(len(ids), MAXLEN * MAXCHUNK), MAXLEN):
            doc_chunks.append((i, ids[j:j + MAXLEN]))

    @torch.no_grad()
    def embed_batch(chunks):
        maxl = max(len(c) for c in chunks) + 2
        ids = np.full((len(chunks), maxl), pad, dtype=np.int64)
        mask = np.zeros((len(chunks), maxl), dtype=np.int64)
        for r, c in enumerate(chunks):
            seq = [cls] + c + [sep]
            ids[r, :len(seq)] = seq
            mask[r, :len(seq)] = 1
        ids_t = torch.from_numpy(ids).to(dev)
        mask_t = torch.from_numpy(mask).to(dev)
        out = model(input_ids=ids_t, attention_mask=mask_t).last_hidden_state
        m = mask_t.unsqueeze(-1).float()
        v = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return v.cpu().numpy()

    sums = np.zeros((len(texts), dim), dtype=np.float64)
    cnts = np.zeros(len(texts), dtype=np.float64)
    buf, bi = [], []
    done = 0
    for k, (di, ch) in enumerate(doc_chunks):
        buf.append(ch); bi.append(di)
        if len(buf) == BATCH or k == len(doc_chunks) - 1:
            vs = embed_batch(buf)
            for r, d in enumerate(bi):
                sums[d] += vs[r]; cnts[d] += 1
            done += len(buf); buf, bi = [], []
            if done % (BATCH * 20) < BATCH:
                print(f"  {done}/{len(doc_chunks)} chunks", flush=True)

    emb = (sums / np.clip(cnts, 1, None)[:, None]).astype(np.float32)
    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez(OUT_NPZ, keys=np.array(keys), emb=emb)
    print(f"wrote {OUT_NPZ}: {emb.shape[0]} vectors x {dim}")


if __name__ == "__main__":
    main()
