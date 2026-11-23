"""Load the @champion models for serving.

Two sources, tried in order:
    MLflow registry   when MLFLOW_TRACKING_URI is set: the ONNX artifacts of
                      the @champion version of each registered model, plus the
                      version's tags (decision threshold, labels, max length)
    bundle folder     SERVING_BUNDLE_DIR (default serving_bundle/), written by
                      `make bundle` from the same registry, for running
                      without MLflow

Only onnxruntime runs the models, so serving needs no training libraries.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import onnxruntime as ort

from src.common.config import ROOT
from src.serving.tokenizer import NoteTokenizer

TABULAR_MODEL = os.environ.get("TABULAR_MODEL_NAME", "claim-triage-tabular")
NOTES_MODEL = os.environ.get("NOTES_MODEL_NAME", "claim-triage-notes")
BUNDLE_DIR = Path(os.environ.get("SERVING_BUNDLE_DIR", ROOT / "serving_bundle"))
TOKENIZER_DIR = Path(os.environ.get("TOKENIZER_DIR", ROOT / "artifacts" / "tokenizer"))


@dataclass
class LoadedModels:
    tabular: ort.InferenceSession
    threshold: float
    notes: ort.InferenceSession | None
    labels: list[str]
    tokenizer: NoteTokenizer | None
    versions: dict[str, str]
    source: str


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # Same thread count the gate measured latency with (export.inference_threads).
    options.intra_op_num_threads = int(os.environ.get("ORT_THREADS", "2"))
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _onnx_file(model_dir: Path) -> Path:
    return next(model_dir.rglob("*.onnx"))


def _download_champion(name: str, target: Path) -> dict[str, Any]:
    """Download @champion's ONNX artifact into target; return its version info."""
    from mlflow import MlflowClient
    from mlflow.artifacts import download_artifacts

    client = MlflowClient()
    version = client.get_model_version_by_alias(name, "champion")
    target.mkdir(parents=True, exist_ok=True)
    local = Path(download_artifacts(artifact_uri=f"models:/{name}@champion", dst_path=str(target)))
    return {
        "version": str(version.version),
        "tags": dict(version.tags),
        "onnx": str(_onnx_file(local)),
    }


def single_file_onnx(source: Path, target: Path) -> None:
    """Copy an ONNX model as one self-contained file.

    MLflow may store a model's weights as external data (a separate .data
    file next to the .onnx graph). Copying only the .onnx breaks the model,
    so load it with its weights and save it with the weights inline.
    """
    import onnx

    model = onnx.load(str(source), load_external_data=True)
    onnx.save_model(model, str(target), save_as_external_data=False)


def export_bundle(bundle_dir: Path = BUNDLE_DIR) -> dict[str, Any]:
    """Copy the @champion models and tokenizer from MLflow into a bundle folder.

    The new bundle is built next to the old one and swapped in only when
    complete, so a failed export (registry down, no champion) leaves the
    working bundle in place. The notes model is optional: without a notes
    champion the bundle is tabular-only, which the API serves as "degraded".
    """
    staging = bundle_dir.with_name(f"{bundle_dir.name}.new")
    if staging.exists():
        shutil.rmtree(staging)
    downloads = staging / "_download"
    meta: dict[str, Any] = {}
    for key, name in (("tabular", TABULAR_MODEL), ("notes", NOTES_MODEL)):
        try:
            info = _download_champion(name, downloads / key)
        except Exception:
            if key == "tabular":
                shutil.rmtree(staging, ignore_errors=True)
                raise
            continue  # no notes champion yet: tabular-only bundle
        single_file_onnx(Path(info["onnx"]), staging / f"{key}.onnx")
        meta[key] = {"name": name, "version": info["version"], "tags": info["tags"]}
    shutil.rmtree(downloads)
    shutil.copy(TOKENIZER_DIR / "tokenizer.json", staging / "tokenizer.json")
    (staging / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")

    old = bundle_dir.with_name(f"{bundle_dir.name}.old")
    if bundle_dir.exists():
        bundle_dir.rename(old)
    staging.rename(bundle_dir)
    shutil.rmtree(old, ignore_errors=True)
    return meta


def _from_files(
    tabular: Path, notes: Path | None, tokenizer: Path, meta: dict[str, Any], source: str
) -> LoadedModels:
    notes_meta = meta.get("notes")
    labels = json.loads(notes_meta["tags"]["labels"]) if notes_meta else []
    max_len = int(notes_meta["tags"].get("max_length", 48)) if notes_meta else 48
    return LoadedModels(
        tabular=_session(tabular),
        threshold=float(meta["tabular"]["tags"]["decision_threshold"]),
        notes=_session(notes) if notes else None,
        labels=labels,
        tokenizer=NoteTokenizer(tokenizer, max_len) if notes else None,
        versions={k: v["version"] for k, v in meta.items()},
        source=source,
    )


def load_models() -> LoadedModels:
    if os.environ.get("MLFLOW_TRACKING_URI"):
        cache = Path(os.environ.get("MODEL_CACHE_DIR", "/tmp/claim-triage-models"))
        meta: dict[str, Any] = {}
        files: dict[str, Path] = {}
        for key, name in (("tabular", TABULAR_MODEL), ("notes", NOTES_MODEL)):
            try:
                info = _download_champion(name, cache / key)
            except Exception:
                if key == "tabular":
                    raise
                continue  # serve tabular-only if no notes champion exists yet
            meta[key] = {"name": name, "version": info["version"], "tags": info["tags"]}
            files[key] = Path(info["onnx"])
        return _from_files(
            files["tabular"],
            files.get("notes"),
            TOKENIZER_DIR / "tokenizer.json",
            meta,
            "mlflow-registry",
        )

    meta = json.loads((BUNDLE_DIR / "metadata.json").read_text())
    notes = BUNDLE_DIR / "notes.onnx"
    return _from_files(
        BUNDLE_DIR / "tabular.onnx",
        notes if notes.exists() else None,
        BUNDLE_DIR / "tokenizer.json",
        meta,
        f"bundle:{BUNDLE_DIR}",
    )


def main() -> None:
    meta = export_bundle()
    print(json.dumps({k: {"name": v["name"], "version": v["version"]} for k, v in meta.items()}))


if __name__ == "__main__":
    main()
