#!/usr/bin/env bash
# Install the Kubernetes tools step 6 uses, pinned, into ~/.local/bin (no sudo).
# Each download is checked against the checksum its project publishes; a
# mismatch stops the script before anything is installed.
#
#   kind     a Kubernetes cluster in one Docker container
#   kubectl  the Kubernetes CLI (same minor version as kind's node image)
#   helm     installs the API's chart
#   k6       load tests
#
# Usage: infra/k8s/install_tools.sh    (or make k8s-tools)
set -euo pipefail

KIND_VERSION=v0.29.0
KUBECTL_VERSION=v1.33.1
HELM_VERSION=v3.18.3
K6_VERSION=v1.1.0

BIN="${BIN_DIR:-$HOME/.local/bin}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$BIN"
cd "$WORK"

fetch() { curl -fsSL --retry 3 -o "$2" "$1"; }

check() {  # file, expected sha256
  echo "$2  $1" | sha256sum --check --quiet - || { echo "checksum mismatch: $1" >&2; exit 1; }
}

installed() {  # name, version string to look for in its version output
  command -v "$1" >/dev/null && "$1" version 2>/dev/null | grep -q -- "$2"
}

if installed kind "$KIND_VERSION"; then echo "kind $KIND_VERSION already installed"; else
  fetch "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-linux-amd64" kind
  fetch "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-linux-amd64.sha256sum" kind.sha
  check kind "$(cut -d' ' -f1 kind.sha)"
  install -m 0755 kind "$BIN/kind"
fi

if installed kubectl "$KUBECTL_VERSION"; then echo "kubectl $KUBECTL_VERSION already installed"; else
  fetch "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl" kubectl
  fetch "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl.sha256" kubectl.sha
  check kubectl "$(cat kubectl.sha)"
  install -m 0755 kubectl "$BIN/kubectl"
fi

if installed helm "$HELM_VERSION"; then echo "helm $HELM_VERSION already installed"; else
  fetch "https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz" helm.tgz
  fetch "https://get.helm.sh/helm-$HELM_VERSION-linux-amd64.tar.gz.sha256sum" helm.sha
  check helm.tgz "$(cut -d' ' -f1 helm.sha)"
  tar -xzf helm.tgz linux-amd64/helm
  install -m 0755 linux-amd64/helm "$BIN/helm"
fi

if installed k6 "${K6_VERSION#v}"; then echo "k6 $K6_VERSION already installed"; else
  name="k6-$K6_VERSION-linux-amd64"
  fetch "https://github.com/grafana/k6/releases/download/$K6_VERSION/$name.tar.gz" k6.tgz
  fetch "https://github.com/grafana/k6/releases/download/$K6_VERSION/k6-$K6_VERSION-checksums.txt" k6.sha
  check k6.tgz "$(grep " $name.tar.gz\$" k6.sha | cut -d' ' -f1)"
  tar -xzf k6.tgz "$name/k6"
  install -m 0755 "$name/k6" "$BIN/k6"
fi

for tool in kind kubectl helm k6; do
  command -v "$tool" >/dev/null || { echo "$tool is not on PATH (add $BIN)" >&2; exit 1; }
done
kind version
kubectl version --client | head -1
helm version --short
k6 version | head -1
