PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate
IMAGE ?= home-llm:latest
PLATFORM ?= linux/amd64
PORT ?= 8000

.PHONY: install install-dev update update-dev lock format lint typecheck test coverage security check build docker-build docker-run api clean

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && python -m pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install .

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && python -m pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install ".[dev]"

update:
	$(ACTIVATE) && python -m pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install --upgrade .

update-dev:
	$(ACTIVATE) && python -m pip install --upgrade pip setuptools wheel
	$(ACTIVATE) && pip install --upgrade ".[dev]"
	$(ACTIVATE) && pre-commit autoupdate

lock:
	$(ACTIVATE) && python -m pip freeze > requirements.lock.txt

format:
	$(ACTIVATE) && ruff format src tests

lint:
	$(ACTIVATE) && ruff check src tests

typecheck:
	$(ACTIVATE) && mypy src

test:
	$(ACTIVATE) && pytest -q

coverage:
	$(ACTIVATE) && pytest --cov=home_llm --cov-report=term-missing --cov-report=xml

security:
	$(ACTIVATE) && bandit -c pyproject.toml -r src
	$(ACTIVATE) && pip-audit

check: lint typecheck test coverage security

api:
	$(ACTIVATE) && uvicorn home_llm.api:app --host 0.0.0.0 --port $(PORT)

build: docker-build

docker-build:
	docker build --platform $(PLATFORM) -t $(IMAGE) .

docker-run:
	docker run --rm \
		-p $(PORT):8000 \
		-e HOME_LLM_CONFIG=/app/config.toml \
		-v "$(PWD)/config.toml:/app/config.toml:ro" \
		--platform $(PLATFORM) \
		$(IMAGE)

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist htmlcov coverage.xml .coverage
