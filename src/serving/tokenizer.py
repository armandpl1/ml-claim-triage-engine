"""Note tokenization for serving, without the Transformers library.

Loads the same pinned tokenizer.json that training saved (artifacts/tokenizer/)
with the small `tokenizers` library, and pads and truncates exactly like
training (src/features/text.py). A test checks both give identical token IDs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from tokenizers import Tokenizer


class NoteTokenizer:
    def __init__(self, tokenizer_json: Path, max_length: int):
        self.max_length = max_length
        self._tok = Tokenizer.from_file(str(tokenizer_json))
        pad_id = self._tok.token_to_id("[PAD]")
        self._tok.enable_truncation(max_length=max_length)
        self._tok.enable_padding(length=max_length, pad_id=pad_id, pad_token="[PAD]")

    def __call__(self, texts: list[str]) -> dict[str, np.ndarray]:
        encodings = self._tok.encode_batch(texts)
        return {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
