.PHONY: clean data train lint format app

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Setup environment and install dependencies (Assuming uv or pip)
install:
	uv pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Download data
data:
	python src/crispdm/data/make_dataset.py

## Train the model and log to MLflow
train:
	python src/crispdm/models/train_model.py

## Run drift monitoring using Evidently AI
monitor:
	python scripts/monitor_drift.py

## Lint using ruff
lint:
	ruff check src
	mypy src

## Format using ruff
format:
	ruff format src

## Run Streamlit App
app:
	streamlit run src/crispdm/app/finapp.py

## Podman Compose up
compose-up:
	podman compose -f podman-compose.yml up -d --build

## Podman Compose down
compose-down:
	podman compose -f podman-compose.yml down

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make install - Install project dependencies"
	@echo "  make clean   - Remove python cache files"
	@echo "  make data    - Download raw data"
	@echo "  make train   - Train model and register to MLflow"
	@echo "  make lint    - Run ruff and mypy"
	@echo "  make format  - Format code using ruff"
	@echo "  make app     - Run Streamlit application"
