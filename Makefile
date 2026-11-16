# Single-command execution. `make help` lists every target.

RUN := uv run

.PHONY: help install hooks lint typecheck test ci data push-data pull-data clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-11s %s\n", $$1, $$2}'

install:  ## Install dependencies from uv.lock and the git pre-commit hooks
	uv sync --locked
	$(RUN) pre-commit install

hooks:  ## Install the git pre-commit hooks (ruff, mypy, file hygiene)
	$(RUN) pre-commit install

lint:  ## All pre-commit checks: ruff lint and format, mypy, file hygiene
	$(RUN) pre-commit run --all-files

typecheck:  ## mypy only
	$(RUN) mypy src

test:  ## All tests
	$(RUN) pytest

ci: lint test  ## Everything CI runs

data:  ## Fetch, stage and validate data (re-runs only changed stages)
	$(RUN) dvc repro

push-data:  ## Store data versions in the DVC remote
	$(RUN) dvc push

pull-data:  ## Restore data versions from the DVC remote
	$(RUN) dvc pull

clean:  ## Remove caches and CI outputs
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache
