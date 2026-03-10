PY := $(if $(wildcard .venv/bin/python),$(CURDIR)/.venv/bin/python,python3)
PIP := $(PY) -m pip
MPLCONFIGDIR ?= $(CURDIR)/.mplconfig
export MPLCONFIGDIR

PIPE := pipeline

LAT ?= 39.55.48N
LON ?= 32.51.00E
START ?= 2026-04-01T00:00:00Z
END ?= 2026-04-03T00:00:00Z
TOPK ?= 20

GRID_TIME ?= 2026-04-01T00:00:00Z
LAT_MIN ?= 30.00.00N
LAT_MAX ?= 50.00.00N
LON_MIN ?= 20.00.00E
LON_MAX ?= 45.00.00E
STEP ?= 5
GRID_TOPK ?= 20
GRID_CSV ?= models/grid_risk_map_onecmd.csv
GRID_PNG ?= models/grid_risk_map_onecmd.png
GRID_TITLE ?= One Command Grid Risk

.PHONY: help install train train-quick predict grid app clean

help:
	@echo "Targets:"
	@echo "  make install        # install deps from requirements.txt"
	@echo "  make train          # full pipeline training"
	@echo "  make train-quick    # fast train (location model only)"
	@echo "  make predict        # location prediction"
	@echo "  make grid           # grid risk + heatmap"
	@echo "  make app            # simple web app (streamlit)"
	@echo "  make clean          # remove generated cache dirs"
	@echo ""
	@echo "Override example: make predict LAT=41.00.00N LON=29.00.00E START=2026-05-01T00:00:00Z END=2026-05-02T00:00:00Z TOPK=50"

install:
	$(PIP) install -r requirements.txt

train:
	cd $(PIPE) && \
	$(PY) fetch_earthquakes.py && \
	$(PY) generate_planet_features.py && \
	$(PY) build_model_dataset.py && \
	$(PY) train_baseline.py && \
	$(PY) train_time_split.py && \
	$(PY) train_time_rolling_cv.py && \
	$(PY) train_location_models.py

train-quick:
	cd $(PIPE) && $(PY) train_location_models.py

predict:
	cd $(PIPE) && $(PY) predict_location_cli.py \
		--lat "$(LAT)" --lon "$(LON)" --start $(START) --end $(END) --top-k $(TOPK)

grid:
	cd $(PIPE) && $(PY) run_grid_pipeline.py \
		--time $(GRID_TIME) \
		--lat-min "$(LAT_MIN)" --lat-max "$(LAT_MAX)" \
		--lon-min "$(LON_MIN)" --lon-max "$(LON_MAX)" \
		--step $(STEP) --top-k $(GRID_TOPK) \
		--out-csv ../$(GRID_CSV) --out-png ../$(GRID_PNG) \
		--title "$(GRID_TITLE)"

app:
	$(PY) -m streamlit run app.py

clean:
	rm -rf .pycache .mplconfig
