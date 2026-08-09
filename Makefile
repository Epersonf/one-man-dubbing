PYTHON ?= python
SRC_DIR := src
HOST ?= 127.0.0.1
PORT ?= 7860

.DEFAULT_GOAL := help

.PHONY: help install setup run dev clean

help:
	@echo "OneManDubbing"
	@echo ""
	@echo "  make install   install Python dependencies (fastapi, uvicorn, etc.)"
	@echo "  make setup     full one-time setup: GPU/CUDA detect, torch, engines, model weights"
	@echo "  make run       start the app (runs first-time setup automatically if needed)"
	@echo "  make dev       start the web UI with auto-reload (no setup, no browser)"
	@echo "  make clean     remove __pycache__ and .pyc files"

install:
	$(PYTHON) -m pip install -r requirements.txt

setup: install
	cd $(SRC_DIR) && $(PYTHON) -m installer.run_installer

run:
	cd $(SRC_DIR) && $(PYTHON) main.py

dev:
	cd $(SRC_DIR) && $(PYTHON) -m uvicorn webui.app:app --reload --host $(HOST) --port $(PORT)

clean:
	find $(SRC_DIR) -type d -name "__pycache__" -exec rm -rf {} +
	find $(SRC_DIR) -type f -name "*.pyc" -delete
