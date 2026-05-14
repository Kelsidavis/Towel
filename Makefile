# Convenience targets for Towel development.
#
# Everything routes through the project's own virtualenv at .venv (created by
# ``pip install -e ".[all,dev]"``) so contributors don't have to think about
# which Python interpreter to call. Override PYTHON= on the make command line
# if you've installed Towel system-wide.

PYTHON ?= .venv/bin/python
PYTEST ?= $(PYTHON) -m pytest
RUFF   ?= $(PYTHON) -m ruff

.PHONY: help test test-fast lint fmt fix doctor clean

help:
	@echo "Towel — common dev targets"
	@echo ""
	@echo "  make test       Run the full pytest suite (~30s on a warm cache)"
	@echo "  make test-fast  Run tests with -x (stop at first failure)"
	@echo "  make lint       ruff check src/ tests/"
	@echo "  make fmt        ruff format src/ tests/"
	@echo "  make fix        ruff check --fix src/ tests/ (auto-fix lints)"
	@echo "  make doctor     Run towel doctor against the current env"
	@echo "  make clean      Remove caches and build artefacts"

test:
	$(PYTEST) tests/ -q

test-fast:
	$(PYTEST) tests/ -q -x

lint:
	$(RUFF) check src/towel/ tests/

fmt:
	$(RUFF) format src/towel/ tests/

fix:
	$(RUFF) check --fix src/towel/ tests/

doctor:
	$(PYTHON) -m towel.cli.main doctor

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .ruff_cache  -prune -exec rm -rf {} +
	rm -rf build/ dist/ *.egg-info/
