# =============================================================================
# Chryso-Voicu pipeline — single entry point.
#
#   make setup        create environments, download model + corpora
#   make tag-tlg      POS-tag the training corpus (long; uses bert-env)
#   make tag-pc       POS-tag the pseudo-Chrysostom texts
#   make features     build MFW/MFP/MFT lists + feature matrices (Steps 05-06)
#   make train        train ONE model (SEED=1000 by default; skips if exists)
#   make ensemble     train N models (N=10) then aggregate votes + Fleiss Kappa
#   make verify       per-PC verification heatmap (uses newest model)
#   make attribute    PC-vs-corpus attribution (SNR-D distance)
#   make models       list trained models and their test AUC
#   make clean-derived  remove regenerable CSV/feature artefacts (keeps tagged/)
#
# Variables:  SEED (train), N (ensemble size), EPOCHS (override 100)
#             CAP (max works/author, default 20), MIN (min works/author, default 2)
# =============================================================================

VENV      := venv/bin/python
BERT      := bert-env/bin/python
KERNEL    := chryso-voicu
NB        := jupyter nbconvert --to notebook --execute --inplace \
             --ExecutePreprocessor.kernel_name=$(KERNEL) \
             --ExecutePreprocessor.timeout=3600
SEED      ?= 1000
N         ?= 10
EPOCHS    ?= 100
CAP       ?= 20
MIN       ?= 2

.PHONY: help setup tag-tlg tag-pc features train ensemble verify attribute models clean-derived

help:
	@grep -E '^#   make ' Makefile | sed 's/^#/ /'

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	./setup.sh

# ── Tagging (bert-env; long-running, resumable via run-tagger.sh) ────────────
tag-tlg:
	./run-tagger.sh tlg-texts.csv 10

tag-pc:
	./run-tagger.sh pc-texts.csv 10

# ── Feature extraction (Steps 05-06) ─────────────────────────────────────────
features:
	$(NB) "Step 05 - Build MFW.ipynb"
	$(NB) "Step 06 - Reconcile with POS and Tokens.ipynb"
	@$(VENV) -c "import pandas as pd; [print(f, {p: sum(c.startswith(p) for c in pd.read_csv(f, nrows=1).columns) for p in ('\$$POS\$$','\$$MFW\$$','\$$TRI\$$')}) for f in ('tlg-features.csv','pc-features.csv')]"

# ── Training (param-keyed; safe to re-run) ───────────────────────────────────
train:
	$(VENV) train.py --seed $(SEED) --epochs $(EPOCHS) --cap-works $(CAP) --min-works $(MIN)

ensemble:
	@for s in $$(seq 1000 $$(( 1000 + $(N) - 1 ))); do \
		echo "=== seed $$s ==="; \
		$(VENV) train.py --seed $$s --epochs $(EPOCHS) --cap-works $(CAP) --min-works $(MIN) || exit 1; \
	done
	$(VENV) ensemble_verify.py --from-models

# ── Analysis ─────────────────────────────────────────────────────────────────
verify:
	$(VENV) verify_pcs.py

attribute:
	$(VENV) attribute_pc.py

models:
	@echo "slug                                              test_auc  minutes"
	@for d in models/*/; do \
		[ -f "$$d/metrics.json" ] && \
		$(VENV) -c "import json,sys;m=json.load(open('$$d/metrics.json'));print(f\"$$(basename $$d)  {m['test_auc']:.3f}   {m['wall_seconds']/60:.1f}\")"; \
	done

# ── Housekeeping ─────────────────────────────────────────────────────────────
clean-derived:
	rm -f tlg-features.csv pc-features.csv *pairs-last-experiment.csv \
	      *test-results.csv pc-vs-corpus.csv pc-verification.* ensemble-*.csv
	@echo "Removed regenerable artefacts (tagged/ and models/ kept)."
