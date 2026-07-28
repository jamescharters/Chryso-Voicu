#!/bin/zsh
# =============================================================================
# setup.sh — one-shot, idempotent setup for the Chryso-Voicu pipeline.
#
# Creates both Python environments, downloads the POS-tagger model, registers
# the Jupyter kernel, and fetches the open corpora (First1KGreek + PTA + PG).
# Every step is skipped if its output already exists, so it is safe to re-run.
#
# Usage:  ./setup.sh            # full setup
#         ./setup.sh --data     # only (re)download datasets
#         ./setup.sh --env      # only (re)build environments
# =============================================================================
set -e
cd "$(dirname "$0")"

PYTHON=python3.13          # this Mac has no scipy/gensim wheels for 3.14
STEP=${1:-all}

log() { print -P "%F{cyan}==>%f $1"; }

# ── 1. Analysis environment (venv) ───────────────────────────────────────────
setup_venv() {
    if [[ -x venv/bin/python ]]; then
        log "venv already exists — skipping"
    else
        log "Creating analysis venv ($PYTHON)"
        $PYTHON -m venv venv
        venv/bin/pip install -q --upgrade pip
    fi
    log "Installing analysis requirements"
    venv/bin/pip install -q -r requirements.txt
    # make freestyl importable without packaging
    echo "$PWD" > "venv/lib/python3.13/site-packages/chryso-voicu.pth"
    # register Jupyter kernel
    venv/bin/python -m ipykernel install --user --name chryso-voicu \
        --display-name "Chryso-Voicu (Python 3.13)" >/dev/null 2>&1 || true
    log "Analysis venv ready"
}

# ── 2. Tagging environment (bert-env) ────────────────────────────────────────
setup_bertenv() {
    if [[ -x bert-env/bin/python ]]; then
        log "bert-env already exists — skipping"
    else
        log "Creating tagging bert-env ($PYTHON)"
        $PYTHON -m venv bert-env
        bert-env/bin/pip install -q --upgrade pip
    fi
    log "Installing tagging requirements"
    bert-env/bin/pip install -q -r requirements-bert.txt
    log "bert-env ready"
}

# ── 3. POS-tagger model + tokenizer ──────────────────────────────────────────
setup_model() {
    mkdir -p SuperPeitho-FLAIR-v2 LM/SuperPeitho-v1
    if [[ -f SuperPeitho-FLAIR-v2/final-model.pt ]]; then
        log "FLAIR model already present — skipping"
    else
        log "Downloading SuperPeitho FLAIR model (~450 MB)"
        curl -L --progress-bar \
            "https://media.githubusercontent.com/media/pranaydeeps/Ancient-Greek-BERT/main/SuperPeitho-FLAIR-v2/final-model.pt" \
            -o SuperPeitho-FLAIR-v2/final-model.pt
    fi
    # tokenizer/config files the embedded model path (../LM/SuperPeitho-v1) needs
    for f in config.json tokenizer_config.json vocab.txt special_tokens_map.json; do
        if [[ ! -f "LM/SuperPeitho-v1/$f" ]]; then
            log "Fetching tokenizer file $f"
            curl -sL "https://huggingface.co/pranaydeeps/Ancient-Greek-BERT/resolve/main/$f" \
                -o "LM/SuperPeitho-v1/$f"
        fi
    done
    log "Model + tokenizer ready"
}

# ── 4. Corpora ───────────────────────────────────────────────────────────────
setup_data() {
    mkdir -p patres tagged

    # First1KGreek (CC-BY-SA) — full Greek corpus via sparse blobless clone
    if [[ -z "$(ls patres/tlg*-grc*.xml 2>/dev/null)" ]]; then
        log "Fetching First1KGreek (Greek TEI)"
        rm -rf /tmp/First1KGreek
        git clone --depth 1 --filter=blob:none --no-checkout \
            https://github.com/OpenGreekAndLatin/First1KGreek /tmp/First1KGreek
        (cd /tmp/First1KGreek && git sparse-checkout init --cone \
            && git sparse-checkout set data && git checkout master >/dev/null 2>&1)
        find /tmp/First1KGreek/data -name "*-grc*.xml" ! -name "__cts__*" \
            -exec cp {} patres/ \;
        log "First1KGreek: $(ls patres/tlg*-grc*.xml | wc -l | tr -d ' ') files"
    else
        log "First1KGreek already in patres/ — skipping"
    fi

    # PTA (CC-BY) — Patristic Text Archive, public branch
    if [[ -z "$(ls patres/pta*-grc*.xml 2>/dev/null)" ]]; then
        log "Fetching PTA (Patristic Text Archive)"
        rm -rf /tmp/PTA
        git clone --depth 1 --filter=blob:none --no-checkout --branch public \
            https://github.com/PatristicTextArchive/pta_data /tmp/PTA
        (cd /tmp/PTA && git sparse-checkout init --cone \
            && git sparse-checkout set data && git checkout public >/dev/null 2>&1)
        find /tmp/PTA/data -name "*-grc*.xml" ! -name "__cts__*" \
            -exec cp {} patres/ \;
        log "PTA: $(ls patres/pta*-grc*.xml | wc -l | tr -d ' ') files"
    else
        log "PTA already in patres/ — skipping"
    fi

    # Patrologia Graeca corpus (CC-BY) — pre-tagged .vert, from Zenodo
    if [[ ! -f /tmp/PG_full.zip ]]; then
        log "Downloading PG corpus (~103 MB) from Zenodo"
        curl -L --progress-bar \
            "https://zenodo.org/api/records/19915273/files/PG.zip/content" \
            -o /tmp/PG_full.zip
    fi
    log "PG corpus ready at /tmp/PG_full.zip (Step 04b ingests it)"
    log "Corpora ready"
}

case "$STEP" in
    --env)  setup_venv; setup_bertenv ;;
    --data) setup_data ;;
    --model) setup_model ;;
    all|*)  setup_venv; setup_bertenv; setup_model; setup_data ;;
esac

log "Setup complete. Next:  make features  (or see the Makefile targets)"
