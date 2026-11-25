# Single-command execution. `make help` lists every target.

RUN := uv run
export MLFLOW_DISABLE_AGENT_HINT := 1

.PHONY: help install hooks lint typecheck test ci data feature-store materialize features train train-tabular train-nlp \
        text-baseline benchmark mlflow-ui gate bundle serve up down simulate drift push-data pull-data fairness model-card batch-score batch-backfill orchestrate retrain nightly infra-up mlflow-up mlflow-check infra-plan infra-check infra-down k8s-tools k8s-up k8s-image k8s-deploy k8s-smoke deploy image-local canary-bundle canary-start canary-weight canary-rollback canary-promote loadtest envs promote rollback-model k8s-down clean

help:  ## List targets
	@grep -E '^[a-z0-9-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

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

feature-store:  ## Rebuild the offline feature store: every version of each policy's features
	$(RUN) dvc repro feature_store

materialize:  ## Load the online store as of now (AS_OF=2025-02-01 for a past date), then check skew
	$(RUN) python -m src.feature_store.materialize $(if $(AS_OF),--as-of $(AS_OF))

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

fairness:  ## Fairness audit of the tabular champion by driver age and region: reports/fairness/
	$(RUN) python -m src.evaluation.fairness

model-card:  ## Regenerate docs/model_card_tabular.md from the served tabular champion
	$(RUN) python -m src.evaluation.model_card

batch-score:  ## Score the claims reported on one day: make batch-score DATE=2025-01-06
	$(RUN) python -m src.batch.score --date $(DATE)

batch-backfill:  ## Replay every night: score, outcomes, weekly performance (SCENARIO=inflation)
	$(RUN) python -m src.batch.backfill --scenario $(or $(SCENARIO),none)

DAGSTER := DAGSTER_HOME=$(CURDIR)/.dagster $(RUN) dagster
DAGSTER_DEFS := -m src.orchestration.definitions
MEMORY_CAP ?= 2500M

.dagster/dagster.yaml: src/orchestration/dagster.yaml
	mkdir -p .dagster
	cp $< $@

orchestrate: .dagster/dagster.yaml  ## Dagster UI (assets, jobs, schedule, sensors): http://localhost:3001
	$(DAGSTER) dev $(DAGSTER_DEFS) -p 3001

retrain: .dagster/dagster.yaml  ## Retrain the triage model with Dagster: data, train, gate, bundle (~6 min)
	CLAIM_TRIAGE_MEMORY_CAP=$(MEMORY_CAP) $(DAGSTER) job execute $(DAGSTER_DEFS) -j retrain_tabular

nightly: .dagster/dagster.yaml  ## One night of the batch job with Dagster: make nightly DATE=2025-06-15
	$(DAGSTER) job execute $(DAGSTER_DEFS) -j nightly_batch --tags '{"dagster/partition": "$(DATE)"}'

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

mlflow-up:  ## MLflow server on Postgres + S3 (run make infra-up first): http://localhost:5001
	$(INFRA) --profile mlflow up -d --build --wait
	@echo "Train against it: MLFLOW_TRACKING_URI=http://localhost:5001 make train-tabular"

mlflow-check:  ## Smoke-check the MLflow server: run to Postgres, artifact to S3, registry, download
	$(RUN) python -m src.infra.mlflow_check

infra-plan:  ## Show what Terraform would change (exit 2 means drift)
	$(TF) plan -detailed-exitcode -input=false

infra-check:  ## Check the buckets and IAM roles are safe (versioned, encrypted, private, least privilege)
	AWS_ENDPOINT_URL=http://localhost:15100 $(RUN) python -m src.infra.storage_check

infra-down:  ## Stop the emulator and MLflow storage (their data is wiped); forget the local Terraform state
	# Not `terraform destroy`: it rightly refuses to delete buckets that hold data
	$(INFRA) --profile mlflow down -v
	rm -f infra/terraform/terraform.tfstate infra/terraform/terraform.tfstate.backup

clean:  ## Remove caches and CI outputs
	rm -rf reports .pytest_cache .mypy_cache .ruff_cache

# Kubernetes (step 6): a one-node kind cluster, the API's Helm chart, a canary
# release and a k6 load test. Stop other stacks first (make down, make
# infra-down): the cluster needs about 2 GB of RAM.
KIND_CLUSTER := claim-triage
K8S_CONFIG := .kind/kind-config.yaml
METRICS_SERVER_CHART := 3.12.2
# Gateway API (the standard channel the Traefik chart's README names) and
# Traefik as its implementation, for weighted canary routing
GATEWAY_API_CRDS := https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.6.3/standard-install.yaml
TRAEFIK_CHART := 42.3.2

k8s-tools:  ## Install kind, kubectl, helm and k6 (pinned, checksum-verified) into ~/.local/bin
	infra/k8s/install_tools.sh

k8s-up:  ## Start the one-node kind cluster: metrics-server (autoscaling), Gateway API + Traefik (canary)
	mkdir -p .kind
	sed 's|__REPO__|$(CURDIR)|g' infra/k8s/kind-config.yaml.tmpl > $(K8S_CONFIG)
	kind get clusters 2>/dev/null | grep -qx $(KIND_CLUSTER) || kind create cluster --config $(K8S_CONFIG) --wait 120s
	helm upgrade --install metrics-server metrics-server \
		--repo https://kubernetes-sigs.github.io/metrics-server/ --version $(METRICS_SERVER_CHART) \
		--namespace kube-system --set 'args={--kubelet-insecure-tls}' \
		--set resources.requests.memory=40Mi --wait --timeout 3m
	kubectl apply --server-side -f $(GATEWAY_API_CRDS) >/dev/null
	helm upgrade --install traefik traefik --repo https://traefik.github.io/charts \
		--version $(TRAEFIK_CHART) --namespace traefik --create-namespace \
		--values infra/k8s/traefik-values.yaml --wait --timeout 3m
	kubectl get nodes

# The git commit; with uncommitted changes to the image's sources, plus
# -dirty-<hash of the diff>, so every different build gets a different tag
# (an unchanged tag means no rollout: Kubernetes sees nothing to update).
IMAGE_SOURCES := src configs Dockerfile pyproject.toml uv.lock
IMAGE_DIFF_HASH = $(shell git diff HEAD -- $(IMAGE_SOURCES) | sha256sum | cut -c1-8)
IMAGE_TAG ?= $(shell git rev-parse --short HEAD)$(shell git diff --quiet HEAD -- $(IMAGE_SOURCES) || echo -dirty-$(IMAGE_DIFF_HASH))
CHART := infra/k8s/chart

k8s-image:  ## Build the working tree's image and load it into kind (tag: the git commit)
	docker build --quiet --tag claim-triage-api:$(IMAGE_TAG) .
	kind load docker-image claim-triage-api:$(IMAGE_TAG) --name $(KIND_CLUSTER)

# Environments (infra/k8s/environments/), one namespace each:
#   dev      the working tree (make k8s-deploy), the @champion bundle
#   staging  a released image, the @staging models
#   prod     a released image, the @prod models; autoscaled, canaries
# Through the gateway (localhost:8090): dev.localhost, staging.localhost, and
# prod for any other host; localhost:8080 goes straight to prod's service.
ENV ?= dev
RELEASE_REPO := ghcr.io/armandpl1/claim-triage-api
env_host = $(if $(filter prod,$(1)),localhost,$(1).localhost)

k8s-deploy: k8s-image  ## Deploy the working tree to dev (dev.localhost through the gateway)
	rev=$$(sha256sum serving_bundle/metadata.json | cut -c1-12); \
	helm upgrade --install claim-triage-api $(CHART) --namespace dev --create-namespace \
		--values infra/k8s/environments/dev.yaml --set image.tag=$(IMAGE_TAG) \
		--set models.revision=$$rev --wait --timeout 3m
	kubectl get pods -n dev -o wide

deploy:  ## Deploy an environment's pinned image and its models: make deploy ENV=staging|prod
	@case "$(ENV)" in staging|prod) ;; *) echo "ENV must be staging or prod (dev: make k8s-deploy)"; exit 1;; esac
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) python -m src.serving.model_store \
		--alias $(ENV) --notes-alias $(ENV) --out serving_bundle_$(ENV)
	image=$$($(RUN) python -c "import yaml; i = yaml.safe_load(open('infra/k8s/environments/$(ENV).yaml'))['image']; print(i['repository'] + ':' + i['tag'])"); \
	docker image inspect $$image >/dev/null 2>&1 || docker pull $$image || \
		{ echo "$$image is not on GHCR (or not public) yet; to test before the release: make image-local VERSION=<tag>"; exit 1; }; \
	[ -n "$$(docker image inspect -f '{{index .Config.Labels "org.opencontainers.image.revision"}}' $$image)" ] || \
		echo "WARNING: $$image is a local stand-in (make image-local), not the released image"; \
	kind load docker-image $$image --name $(KIND_CLUSTER)
	rev=$$(sha256sum serving_bundle_$(ENV)/metadata.json | cut -c1-12); \
	helm upgrade --install claim-triage-api $(CHART) --namespace $(ENV) --create-namespace \
		--values infra/k8s/environments/$(ENV).yaml --set models.revision=$$rev --wait --timeout 3m
	kubectl get pods -n $(ENV) -o wide

image-local:  ## Stand-in for a release not yet on GHCR, built from the working tree: make image-local VERSION=1.0.0
	@test -n "$(VERSION)" || { echo "VERSION is required"; exit 1; }
	docker build --quiet --build-arg APP_VERSION=$(VERSION)+local --tag $(RELEASE_REPO):$(VERSION) .

k8s-smoke:  ## Smoke-test an environment through the gateway: make k8s-smoke ENV=dev|staging|prod
	curl -fsS -H 'Host: $(call env_host,$(ENV))' localhost:8090/health && echo
	curl -fsS -H 'Host: $(call env_host,$(ENV))' localhost:8090/predict -H 'content-type: application/json' -d @infra/k8s/smoke_claim.json && echo
	curl -fsS -H 'Host: $(call env_host,$(ENV))' localhost:8090/predict/policy -H 'content-type: application/json' -d '{"policy_id": 1}' && echo

# Canary release in prod: the @staging models (the version about to go to
# prod) on one pod with prod's image, WEIGHT percent of prod's requests.
# Promoting it is `promote ENV=prod` then a prod deploy; it is refused if the
# canary no longer serves @staging (src/serving/canary.py).
WEIGHT ?= 10
CANARY_BUNDLE := serving_bundle_canary

canary-bundle:  ## Export the @staging models into serving_bundle_canary/
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) python -m src.serving.model_store \
		--alias staging --notes-alias staging --out $(CANARY_BUNDLE)

canary-start: canary-bundle  ## Canary the @staging models in prod with WEIGHT% of prod's requests (default 10)
	rev=$$(sha256sum $(CANARY_BUNDLE)/metadata.json | cut -c1-12); \
	helm upgrade --install claim-triage-api-canary $(CHART) --namespace prod \
		--values infra/k8s/environments/prod.yaml --values infra/k8s/canary-values.yaml \
		--set models.revision=$$rev --wait --timeout 3m
	$(MAKE) canary-weight WEIGHT=$(WEIGHT)

canary-weight:  ## Send WEIGHT% of prod's requests to the canary: make canary-weight WEIGHT=50
	helm upgrade claim-triage-api $(CHART) --namespace prod --reuse-values --set route.canaryWeight=$(WEIGHT) --wait
	kubectl get httproute claim-triage-api -n prod -o jsonpath='{range .spec.rules[0].backendRefs[*]}{.name}={.weight} {end}' && echo

canary-rollback:  ## All of prod's traffic back to stable, then remove the canary
	$(MAKE) canary-weight WEIGHT=0
	helm uninstall claim-triage-api-canary --namespace prod --wait

canary-promote:  ## Promote the canary's models to prod (must still be @staging), redeploy prod, remove the canary
	MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) python -m src.serving.canary --bundle $(CANARY_BUNDLE)
	$(MAKE) promote ENV=prod
	$(MAKE) deploy ENV=prod
	helm uninstall claim-triage-api-canary --namespace prod --wait

LOADTEST_DIR := reports/loadtest

loadtest:  ## k6 load test through the gateway, 10 -> 30 req/s (~5.5 min); results in reports/loadtest/
	mkdir -p $(LOADTEST_DIR) .kind
	$(RUN) python -c "import json, pandas as pd; ids = sorted(set(pd.read_parquet('data/processed/test.parquet').policy_id.astype(int))); json.dump(ids, open('.kind/policies.json', 'w'))"
	infra/k8s/watch_pods.sh $(LOADTEST_DIR)/pods.csv & echo $$! > .kind/watch.pid; \
	cd infra/k8s && k6 run --quiet --summary-export ../../$(LOADTEST_DIR)/summary.json loadtest.js; \
	status=$$?; kill $$(cat ../../.kind/watch.pid); exit $$status

# Model promotion between environments (src/serving/environments.py):
# gate -> @champion -> @staging -> @prod, each move reversible.
ENVIRONMENTS := MLFLOW_TRACKING_URI=sqlite:///mlflow.db $(RUN) python -m src.serving.environments

envs:  ## Which model version each environment runs (registry aliases)
	$(ENVIRONMENTS) status

promote:  ## Promote models: make promote ENV=staging (from @champion) or ENV=prod (from @staging)
	$(ENVIRONMENTS) promote $(ENV)

rollback-model:  ## Put an environment's models back on their previous version: make rollback-model ENV=prod
	$(ENVIRONMENTS) rollback $(ENV)

k8s-down:  ## Delete the kind cluster (frees its RAM)
	kind delete cluster --name $(KIND_CLUSTER)
