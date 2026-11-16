"""Versioned tokenizer for the adjuster notes.

The tokenizer is pinned to an exact Hugging Face commit (configs/data_config.yaml)
and saved to artifacts/tokenizer/ with that commit recorded, so training and
serving tokenize identically even if the upstream model changes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.common.config import ROOT

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"


def load_tokenizer(cfg: dict[str, Any], directory: Path = TOKENIZER_DIR) -> PreTrainedTokenizerBase:
    """The pinned tokenizer: from the local snapshot, or downloaded and saved once."""
    text_cfg = cfg["text"]
    version_file = directory / "version.json"
    expected = {"tokenizer": text_cfg["tokenizer"], "revision": text_cfg["revision"]}
    if version_file.exists() and json.loads(version_file.read_text()) == expected:
        return AutoTokenizer.from_pretrained(directory)

    tokenizer = AutoTokenizer.from_pretrained(text_cfg["tokenizer"], revision=text_cfg["revision"])
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(directory)
    version_file.write_text(json.dumps(expected, indent=2) + "\n")
    return tokenizer


def tokenize(
    tokenizer: PreTrainedTokenizerBase, texts: list[str], max_length: int
) -> dict[str, Any]:
    """Token IDs and attention masks, padded and truncated to max_length."""
    return dict(
        tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
    )
