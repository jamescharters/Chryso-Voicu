"""
ocr_retag.py — re-tag specific Patrologia Graeca volumes with the SAME BERT tagger
(SuperPeitho-FLAIR-v2) used for the clean corpus, writing to a SEPARATE directory.

Why: the PG rows in tagged/ carry a foreign tagset (`n+com`, `det`, ...) that shipped
in the .vert files, so their POS-trigram features are not comparable to the clean
corpus (Perseus 9-position tags). This script produces BERT-tagged versions of the
OCR text so the POS channel can be compared on equal footing.

It NEVER touches tagged/ or any base-pipeline file: output goes to tagged-ocr/.
Run with the bert-env interpreter:

    bert-env/bin/python ocr_retag.py                 # default: PG009 + PG016_3 (paired authors)
    bert-env/bin/python ocr_retag.py PG071_tagged_text ...   # extra volumes
"""
from typing import List
import os
import re
import sys
import gc

os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import tqdm

# accept local relative repo ids (the model embeds '../LM/SuperPeitho-v1')
import huggingface_hub.utils._validators as _hf_v
_orig_validate = _hf_v.validate_repo_id
def _patched_validate(repo_id, **kw):
    if repo_id.startswith(".") or os.path.isabs(repo_id):
        return
    _orig_validate(repo_id, **kw)
_hf_v.validate_repo_id = _patched_validate

import flair
import torch
if torch.backends.mps.is_available():
    flair.device = torch.device("mps")
elif torch.cuda.is_available():
    flair.device = torch.device("cuda:0")
else:
    flair.device = torch.device("cpu")

from flair.models import SequenceTagger
from flair.data import Sentence

OUT_DIR = "tagged-ocr"
DEFAULT_FILES = ["PG009_tagged_text", "PG016_3_tagged_text"]  # Clement, Origen (paired authors)
SENTENCE_SPLITTER = re.compile(r"(?<=[!?\.])")


def load_tagger():
    root = os.getcwd()
    os.chdir(os.path.join(root, "SuperPeitho-FLAIR-v2"))
    tagger = SequenceTagger.load("final-model.pt")
    os.chdir(root)
    return tagger


def get_poses(tagger, sentence) -> List[tuple]:
    try:
        tagger.predict(sentence)
    except RuntimeError:
        return []
    return [(e.text, e.get_label("pos", zero_tag_value="-").value) for e in sentence.tokens]


def tag_text(tagger, text: str):
    for s in SENTENCE_SPLITTER.split(text):
        s = s.strip()
        if not s:
            continue
        sentence = Sentence(s)
        yield from get_poses(tagger, sentence)
        del sentence


def main():
    files = sys.argv[1:] or DEFAULT_FILES
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv("tlg-texts.csv", usecols=["file", "full-text-raw"])
    df = df[df["file"].isin(files)]
    if len(df) == 0:
        print("No matching PG files found in tlg-texts.csv:", files)
        return

    tagger = load_tagger()
    for _, row in df.iterrows():
        out = os.path.join(OUT_DIR, f"{row['file']}-bert.txt")
        if os.path.exists(out):
            print(f"[skip] {out} exists")
            continue
        print(f"[tag ] {row['file']}  ({len(str(row['full-text-raw']))} chars) -> {out}")
        pairs = list(tag_text(tagger, str(row["full-text-raw"])))
        with open(out, "w") as f:
            f.write("\n".join("\t".join(p) for p in pairs))
        print(f"[done] {row['file']}: {len(pairs)} tokens")
        del pairs
        gc.collect()


if __name__ == "__main__":
    main()
