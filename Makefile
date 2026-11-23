# Single-command execution. `make help` lists every target.

RUN := uv run
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help install hooks lint typecheck test ci data features train train-tabular train-nlp \
        text-baseline benchmark mlflow-ui gate bundle serve up down simulate drift push-data pull-data infra-up infra-plan infra-check infra-down clean

help:  ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

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

features:  ## Split, fit the preprocessor on train only, save the drift reference
	$(RUN) dvc repro features

train-tabular:  ## Tune LightGBM with Optuna, export ONNX, register in MLflow
	$(RUN) python -m src.training.tabular

train-nlp:  ## Fine-tune DistilBERT with LoRA, export ONNX FP32 + INT8, register
	$(RUN) python -m src.training.nlp

text-baseline:  ## TF-IDF + logistic regression baseline for the notes classifier
	$(RUN) python -m src.training.text_baseline

benchmark:  ## Single-request latency of every ONNX model (FP32 vs INT8)
	$(RUN) python -m src.training.benchmark

train: train-tabular train-nlp benchmark  ## Train both models and benchmark them

mlflow-ui:  ## MLflow server on the local store: http://localhost:5000
	$(RUN) mlflow server --backend-store-uri sqlite:///mlflow.db \
		--default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

gate:  ## Champion vs challenger gate for both models (promotes only if all checks pass)
	$(RUN) python -m src.evaluation.gate --model tabular; \
	$(RUN) python -m src.evaluation.gate --model notes

bundle:  ## Export the @champion models from MLflow into serving_bundle/
	$(RUN) python -m src.serving.model_store

serve:  ## Run the API locally from the MLflow registry: http://localhost:8000/docs
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) uvicorn src.serving.app:app --port 8000

up:  ## Build and start API + MLflow + Prometheus + Grafana (docker compose)
	mkdir -p data/inference
	UID=$$(id -u) GID=$$(id -g) docker compose up --build -d

down:  ## Stop the docker compose stack
	docker compose down

simulate:  ## Send 500 test claims through the API (SHIFT=1 for a drifted population)
	$(RUN) python -m monitoring.simulate_traffic --n 500 $(if $(SHIFT),--shift,)

drift:  ## Drift report (Evidently, own pinned env): logged requests vs training reference
	uv run --script monitoring/drift_detector.py

push-data:  ## Store data versions in the DVC remote
	$(RUN) dvc push

pull-data:  ## Restore data versions from the DVC remote
	$(RUN) dvc pull

TF := terraform -chdir=infra/terraform
INFRA := docker compose -f infra/docker-compose.yml

infra-up:  ## Start the AWS emulator (Moto) and apply infra/terraform; a second plan must show no changes
	$(INFRA) up -d --wait moto
	$(TF) init -input=false
	$(TF) apply -auto-approve -input=false
	# Moto ignores tags sent with CreateBucket; a second apply sets them the normal way
	$(TF) apply -auto-approve -input=false
	$(TF) plan -detailed-exitcode -input=false
	# Point the DVC remote at the emulator, in .dvc/config.local (not committed):
	# the committed config stays what a real AWS account would use
	$(RUN) dvc remote modify --local s3 endpointurl http://localhost:15100
	$(RUN) dvc remote modify --local s3 access_key_id test
	$(RUN) dvc remote modify --local s3 secret_access_key test

infra-plan:  ## Show what Terraform would change (exit 2 means drift)
	$(TF) plan -detailed-exitcode -input=false

infra-check:  ## Check the buckets and IAM roles are safe (versioned, encrypted, private, least privilege)
	AWS_ENDPOINT_URL=http://localhost:15100 $(RUN) python -m src.infra.storage_check

infra-down:  ## Stop the emulator and MLflow storage (their data is wiped); forget the local Terraform state
	# Not `terraform destroy`: it rightly refuses to delete buckets that hold data
	$(INFRA) down -v
	rm -f infra/terraform/terraform.tfstate infra/terraform/terraform.tfstate.backup

clean:  ## Remove caches and CI outputs
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache
