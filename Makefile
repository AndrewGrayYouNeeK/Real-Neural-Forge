PYTHON ?= .venv/bin/python

.PHONY: test train eval serve

test:
	$(PYTHON) -m pytest tests -q

train:
	$(PYTHON) -m src.train --config config/config.yaml

eval:
	$(PYTHON) -m src.eval --config config/config.yaml

serve:
	$(PYTHON) -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
