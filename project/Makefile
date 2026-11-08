# End-to-end crypto perpetuals trading project
#
# Every target is safe to run repeatedly. `make all` reproduces the whole project from
# an empty data directory.

PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help
.PHONY: help setup download panel model pipeline dashboard orders notify notebook test clean all

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtual environment and install the pinned dependencies
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

download:  ## Fetch every raw data source (Binance archive, FRED, Fear and Greed, CoinGecko)
	$(PYTHON) -m src.download_data --stage all

panel:  ## Build the unified feature panel and the design matrix
	$(PYTHON) -m src.pipeline --stages panel,dataset

model:  ## Train and tune the model zoo on the cached design matrix
	$(PYTHON) -m src.pipeline --stages train

pipeline:  ## Run the whole project from the stored raw files
	$(PYTHON) -m src.pipeline --regime cache

dashboard:  ## Rebuild the HTML dashboard from the latest run
	$(PYTHON) -m src.monitor --dashboard

orders:  ## Turn the latest signals into paper orders and write them to the ledger
	$(PYTHON) -m src.broker --signal pred_ensemble

notify:  ## Send the daily summary to Telegram, if the credentials are set
	$(PYTHON) -m src.monitor --telegram

notebook:  ## Execute the end-to-end notebook in place
	jupyter nbconvert --to notebook --execute --inplace notebooks/project_pipeline.ipynb

test:  ## Run the unit tests
	$(PYTHON) -m pytest tests -q

all: download pipeline dashboard  ## Download, build, simulate and report
	@echo "Pipeline finished. Artefacts are in output/"

clean:  ## Remove the cached panel, design matrix and predictions, keep the raw archives
	rm -rf data/cache
