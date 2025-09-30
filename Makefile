.PHONY: reproduce_phase1

GHZ_COUNTS=data/external/ghz_counts.csv
OUT=out/phase1_protocol1

reproduce_phase1:
	@mkdir -p $(OUT)
	python -m triality_qutrit.datasets.ghz_ingest --in $(GHZ_COUNTS) --outdir data/ghz_ingested
	python -m triality_qutrit.protocols.protocol1_cli --counts $(GHZ_COUNTS) --outdir $(OUT) --shots 4096 --n_boot 1000
	@echo 'Phase 1 complete -> $(OUT)'
