.PHONY: setup lint typecheck test smoke prepare train track export benchmark evaluate gif all clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src/

test:
	pytest -m "not slow" --cov=mot_pipeline --cov-report=term-missing

prepare:
	$(PYTHON) scripts/prepare_dataset.py --config configs/default.yaml

train:
	$(PYTHON) scripts/train.py --config configs/default.yaml

track:
	$(PYTHON) scripts/track.py --config configs/default.yaml

export:
	$(PYTHON) scripts/export_onnx.py --config configs/default.yaml

benchmark:
	$(PYTHON) scripts/benchmark.py --config configs/default.yaml

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml

gif:
	$(PYTHON) scripts/make_gif.py --config configs/default.yaml

# Full pipeline in smoke mode: tiny data, 1 epoch, CPU. Must run end-to-end
# on a clean clone with no manual steps beyond `make setup` + MOT17 present.
smoke:
	$(PYTHON) scripts/prepare_dataset.py --config configs/default.yaml --override configs/smoke.yaml
	$(PYTHON) scripts/train.py --config configs/default.yaml --override configs/smoke.yaml
	$(PYTHON) scripts/track.py --config configs/default.yaml --override configs/smoke.yaml
	$(PYTHON) scripts/export_onnx.py --config configs/default.yaml --override configs/smoke.yaml
	$(PYTHON) scripts/benchmark.py --config configs/default.yaml --override configs/smoke.yaml
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --override configs/smoke.yaml

all: prepare train track export benchmark evaluate

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
