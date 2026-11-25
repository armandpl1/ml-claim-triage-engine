"""Serving pieces: request validation, feature parity, tokenizer parity."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.common.config import ROOT
from src.features.row import log_density
from src.features.transform import add_row_features
from src.serving import model_store
from src.serving.app import softmax, tabular_feeds
from src.serving.schemas import ClaimRequest
from src.training.onnx_export import onnx_inputs

VALID = {
    "exposure": 0.5,
    "vehicle_power": 6,
    "vehicle_age": 3,
    "driver_age": 40,
    "bonus_malus": 50,
    "density": 1200,
    "area": "C",
    "vehicle_brand": "B12",
    "fuel_type": "Diesel",
    "region": "R82",
}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("driver_age", 15),
        ("exposure", 0),
        ("area", "Z"),
        ("vehicle_brand", "Toyota"),
        ("fuel_type", "Electric"),
        ("density", -5),
        ("region", "82"),
    ],
)
def test_out_of_contract_requests_are_rejected(field, value):
    with pytest.raises(ValidationError):
        ClaimRequest(**{**VALID, field: value})


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ClaimRequest(**VALID, claim_amount=1000)  # the target must never be an input


def test_serving_features_match_training_features(cfg):
    claim = ClaimRequest(**VALID)
    training = onnx_inputs(add_row_features(pd.DataFrame([VALID])), cfg)
    serving = tabular_feeds(claim)
    assert training.keys() == serving.keys()
    for name in training:
        np.testing.assert_array_equal(training[name], serving[name])


@pytest.mark.parametrize("density", [1200, 0, -3, None, float("nan")])
def test_log_density_matches_training(density):
    frame = add_row_features(pd.DataFrame({"density": [density]}))
    assert frame["log_density"].iloc[0] == pytest.approx(log_density(density))


def test_softmax_sums_to_one():
    probs = softmax(np.array([2.0, 1.0, 0.1]))
    assert probs.sum() == pytest.approx(1.0) and probs.argmax() == 0


@pytest.mark.skipif(
    not (ROOT / "artifacts/tokenizer/tokenizer.json").exists(),
    reason="tokenizer snapshot not built yet",
)
def test_serving_tokenizer_matches_training_tokenizer(cfg):
    from src.features.text import load_tokenizer, tokenize
    from src.serving.tokenizer import NoteTokenizer

    notes = [
        "Ins'd veh rear-ended at a roundabout. Severe dmg to bonnet.",
        "Windscreen cracked during a storm. Replacement requested.",
        "x " * 100,
    ]
    training = tokenize(load_tokenizer(cfg), notes, 48)
    serving = NoteTokenizer(ROOT / "artifacts/tokenizer/tokenizer.json", 48)(notes)
    np.testing.assert_array_equal(training["input_ids"], serving["input_ids"])
    np.testing.assert_array_equal(training["attention_mask"], serving["attention_mask"])


def test_bundled_model_keeps_external_weights(tmp_path):
    """A model whose weights MLflow stored as a separate .data file must still
    load from the bundle (regression: copying only the .onnx graph broke it)."""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper, numpy_helper

    from src.serving.model_store import single_file_onnx

    weight = numpy_helper.from_array(np.arange(2048, dtype=np.float32), name="w")
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "w"], ["y"])],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2048])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2048])],
        [weight],
    )
    source_dir = tmp_path / "mlflow_download"
    source_dir.mkdir()
    onnx.save_model(
        helper.make_model(graph),
        str(source_dir / "model.onnx"),
        save_as_external_data=True,
        location="weights.data",
        size_threshold=0,
    )
    assert (source_dir / "weights.data").exists()

    target = tmp_path / "bundle" / "notes.onnx"
    target.parent.mkdir()
    single_file_onnx(source_dir / "model.onnx", target)

    assert sorted(p.name for p in target.parent.iterdir()) == ["notes.onnx"]
    session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
    out = session.run(None, {"x": np.ones(2048, dtype=np.float32)})[0]
    assert out[5] == 6.0


def _fake_champions(monkeypatch, tmp_path, available: set[str]):
    def download(name, target, alias="champion"):
        if name not in available:
            raise RuntimeError(f"no @champion for {name}")
        target.mkdir(parents=True, exist_ok=True)
        (target / "model.onnx").write_text(f"onnx of {name}")
        return {
            "version": "3" if alias == "champion" else "4",
            "tags": {"decision_threshold": "0.08"},
            "onnx": str(target / "model.onnx"),
        }

    monkeypatch.setattr(model_store, "_download_champion", download)
    monkeypatch.setattr(model_store, "single_file_onnx", lambda src, dst: shutil.copy(src, dst))
    tokenizer = tmp_path / "tok"
    tokenizer.mkdir(exist_ok=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(model_store, "TOKENIZER_DIR", tokenizer)


def test_bundle_without_a_notes_champion_is_tabular_only(monkeypatch, tmp_path):
    _fake_champions(monkeypatch, tmp_path, {model_store.TABULAR_MODEL})
    meta = model_store.export_bundle(tmp_path / "bundle")
    assert set(meta) == {"tabular"}
    assert sorted(p.name for p in (tmp_path / "bundle").iterdir()) == [
        "metadata.json",
        "tabular.onnx",
        "tokenizer.json",
    ]


def test_a_failed_export_keeps_the_working_bundle(monkeypatch, tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "tabular.onnx").write_text("the working model")
    _fake_champions(monkeypatch, tmp_path, set())  # registry has no champion at all
    with pytest.raises(RuntimeError, match="no @champion"):
        model_store.export_bundle(bundle)
    assert (bundle / "tabular.onnx").read_text() == "the working model"
    assert not (tmp_path / "bundle.new").exists()


def test_canary_bundle_exports_the_challenger_triage_model(monkeypatch, tmp_path):
    _fake_champions(monkeypatch, tmp_path, {model_store.TABULAR_MODEL, model_store.NOTES_MODEL})
    meta = model_store.export_bundle(tmp_path / "canary", tabular_alias="challenger")
    assert meta["tabular"]["version"] == "4" and meta["tabular"]["alias"] == "challenger"
    assert meta["notes"]["version"] == "3"  # the notes model stays @champion


def test_responses_name_the_track_and_release(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.serving import app as app_module

    monkeypatch.setattr(app_module, "TRACK", "canary")
    monkeypatch.setattr(app_module, "APP_VERSION", "1.2.3")
    monkeypatch.setattr(app_module, "ENVIRONMENT", "staging")
    monkeypatch.setattr(app_module, "INFERENCE_LOG", tmp_path / "log.db")
    monkeypatch.setattr(app_module, "ONLINE_STORE", tmp_path / "none.db")
    monkeypatch.setattr(app_module, "load_models", _tiny_models)
    with TestClient(app_module.app) as client:
        health = client.get("/health")
        assert health.headers["x-serving-track"] == "canary"
        assert health.json()["version"] == "1.2.3"
        assert health.json()["environment"] == "staging"


def _tiny_models():
    import numpy as np

    from src.serving.model_store import LoadedModels

    class Session:
        def run(self, _outputs, feeds):
            p = np.full(len(feeds["driver_age"]), 0.1)
            return [p > 0.5, np.column_stack([1 - p, p])]

    return LoadedModels(Session(), 0.5, None, [], None, {"tabular": "1"}, "test")
