# Documentation
from typing import List

#Export
from collections import Counter
import json
import os

# Dealing with our CSV
import pandas
import re

# The model's embedded tokenizer path '../LM/SuperPeitho-v1' is relative to
# the SuperPeitho-FLAIR-v2/ directory. We must load from there and patch the
# HuggingFace validator so it accepts local relative paths.
import sys
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import huggingface_hub.utils._validators as _hf_v
_orig_validate = _hf_v.validate_repo_id
def _patched_validate(repo_id, **kw):
    if repo_id.startswith('.') or os.path.isabs(repo_id):
        return
    _orig_validate(repo_id, **kw)
_hf_v.validate_repo_id = _patched_validate

# Dealing with lemmatization
import flair
import tqdm
import torch
# Use MPS on Apple Silicon, CUDA if available, otherwise CPU
if torch.backends.mps.is_available():
    flair.device = torch.device('mps')
elif torch.cuda.is_available():
    flair.device = torch.device('cuda:0')
else:
    flair.device = torch.device('cpu')

from flair.models import SequenceTagger
from flair.data import Sentence

# Load model from its own directory so relative path resolves correctly
_project_root = os.getcwd()
os.chdir(os.path.join(_project_root, 'SuperPeitho-FLAIR-v2'))
tagger = SequenceTagger.load('final-model.pt')
os.chdir(_project_root)

#Load CSV — read metadata only first (skip full-text-raw to save RAM)
import gc
SENTENCE_SPLITTER = re.compile(r"(?<=[!?\.])")
_csv_path = sys.argv[1]
_meta_cols = ["file", "tokens"]
# Detect column name variant
_cols = pandas.read_csv(_csv_path, nrows=0).columns.tolist()
_text_col = "text" if "text" in _cols and "full-text-raw" not in _cols else "full-text-raw"
_meta_cols.append(_text_col)
texts = pandas.read_csv(_csv_path, usecols=_meta_cols).sort_values("tokens")
if _text_col == "text":
    texts = texts.rename(columns={"text": "full-text-raw"})


def get_poses(sentence) -> List[str]:
    try:
        tagger.predict(sentence)
    except RuntimeError:
        # Tensor alignment bug in flair ≥0.13 on certain sentences — skip
        return []
    return [(e.text, e.get_label("pos", zero_tag_value="-").value) for e in sentence.tokens]


def get_text_poses(text: str):
    """Yield (word, pos) pairs one sentence at a time to minimise peak RAM."""
    for s in SENTENCE_SPLITTER.split(text):
        s = s.strip()
        if not s:
            continue
        sentence = Sentence(s)
        yield from get_poses(sentence)
        del sentence  # free immediately

total = Counter()

# Optional: limit texts per run to control memory (pass --max N as extra arg)
MAX_TEXTS = int(sys.argv[2]) if len(sys.argv) > 2 else 999999
tagged_this_run = 0

for idx, text in tqdm.tqdm(texts.iterrows()):
    if tagged_this_run >= MAX_TEXTS:
        break
    if os.path.exists(f"./tagged/{text['file']}-tagged.txt"):
        print(f"Passing {text['file']}")
        continue
    pos_text = list(get_text_poses(text["full-text-raw"]))
    with open(f"./tagged/{text['file']}-tagged.txt", "w") as f:
        f.write("\n".join([
            "\t".join(tok) for tok in pos_text
        ]))
    total.update(Counter(tok[0] for tok in pos_text if tok[1][0] != "u"))
    del pos_text
    tagged_this_run += 1
    gc.collect()