"""The pinned tokenizer: saved with its version, fixed-length output, typo tolerant."""

from __future__ import annotations

import json

import pytest

from src.features.text import TOKENIZER_DIR, load_tokenizer, tokenize

pytestmark = pytest.mark.skipif(
    not (TOKENIZER_DIR / "version.json").exists(),
    reason="tokenizer snapshot not built yet (make features)",
)


def test_snapshot_records_the_pinned_revision(cfg):
    version = json.loads((TOKENIZER_DIR / "version.json").read_text())
    assert version == {"tokenizer": cfg["text"]["tokenizer"], "revision": cfg["text"]["revision"]}


def test_output_is_padded_and_truncated_to_max_length(cfg):
    tokenizer = load_tokenizer(cfg)
    max_length = cfg["text"]["max_length"]
    out = tokenize(tokenizer, ["Short note.", "word " * 200], max_length)
    assert out["input_ids"].shape == (2, max_length)
    assert out["attention_mask"][0].sum() < max_length  # padded
    assert out["attention_mask"][1].sum() == max_length  # truncated


def test_typos_and_shorthand_still_tokenize(cfg):
    tokenizer = load_tokenizer(cfg)
    out = tokenize(tokenizer, ["Ins'd veh reprted stolen, polcie report filed."], 48)
    assert out["attention_mask"].sum() > 5
