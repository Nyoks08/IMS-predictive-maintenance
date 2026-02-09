# Makefile for Realtime Predictive Maintenance (FastAPI + Redis + Worker)
# Usage examples:
#   make setup
#   make api
#   make worker
#   make docker-up
#   make train

PYTHON ?= python
PIP ?= pip

BACKEND_DIR := backend
VENV_DIR := .venv

.DEFAULT_GOAL := help

help:
	@echo "Available targets:"
	@echo "  setup        Create venv and install requirements"
	@echo "  install      Install Python dependencies"
	@echo "  api          Run FastAPI locally (from backend/)"
	@echo "  worker       Run worker locally (from backend/)"
	@echo "  train        Run IMS training script"
	@echo "  test         Run pytest"
	@echo "  redis        Run Redis in Docker (local dev)"
	@echo "  docker-up    Start full stack with docker-compose"
	@echo "  docker-down  Stop docker-compose stack"
	@echo "  docker-logs  Tail docker-compose logs"
	@echo "  clean        Remove caches and local artifacts (safe)"

setup:
	$(PYTHON) -m venv $(VENV_DIR)
	@echo ""
	@echo "Venv created at $(VENV_DIR). Activate it then run: make install"
	@echo "Windows PowerShell: .\\$(VENV_DIR)\\Scripts\\Activate.ps1"
	@echo "macOS/Linux: source $(VENV_DIR)/bin/activate"

install:
	$(PIP) install -r requirements.txt

api:
	cd $(BACKEND_DIR) && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

worker:
	cd $(BACKEND_DIR) && $(PYTHON) -m workers.worker

train:
	$(PYTHON) $(BACKEND_DIR)/src/training/training_from_ims.py --repo_root .

test:
	cd $(BACKEND_DIR) && pytest -q

redis:
	docker run --rm -p 6379:6379 --name redis redis:7

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	@echo "Cleaning python caches..."
	rm -rf **/__pycache__ **/*.pyc .pytest_cache
	@echo "Done."