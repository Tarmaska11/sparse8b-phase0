"""Corpus utilities for the sparse8b Phase-0 harness.

Tokenizes wikitext-2 and (streamed) ultrachat into fixed sliding windows using the
MLC model's own tokenizer.json. No transformers; only the `tokenizers` package.
"""

import itertools

import torch
from tokenizers import Tokenizer

import common

BOS_ID = 128000  # <|begin_of_text|>


def get_tokenizer(model_dir: str) -> Tokenizer:
    import os
    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def _bos_id(tok: Tokenizer) -> int:
    tid = tok.token_to_id("<|begin_of_text|>")
    return tid if tid is not None else BOS_ID


def _windows_from_ids(ids, seq_len, n_windows, bos):
    """Slice a flat id list into [n_windows, seq_len]; each window[0] set to `bos`."""
    rows = []
    for w in range(n_windows):
        s = w * seq_len
        chunk = ids[s:s + seq_len]
        if len(chunk) < seq_len:
            break
        chunk = list(chunk)
        chunk[0] = bos
        rows.append(chunk)
    if not rows:
        raise RuntimeError("not enough tokens to form a single window")
    return torch.tensor(rows, dtype=torch.long)


def wikitext_tokens(tokenizer: Tokenizer, n_tokens: int, seq_len: int,
                    split: str = "test", seed: int = 0) -> torch.Tensor:
    """WikiText-2 raw sliding windows [n_windows, seq_len]."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    lines = [ln for ln in ds["text"] if ln and ln.strip()]
    text = "\n\n".join(lines)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    n_windows = max(1, n_tokens // seq_len)
    return _windows_from_ids(ids, seq_len, n_windows, _bos_id(tokenizer))


def chat_tokens(tokenizer: Tokenizer, n_tokens: int, seq_len: int,
                split: str, seed: int = 0) -> torch.Tensor:
    """Streamed UltraChat sliding windows [n_windows, seq_len].

    split: "train_sft" (calib) or "test_sft" (eval). Streams to avoid a full download.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split, streaming=True)
    # Cap the number of examples we pull; ~seq_len tokens per doc is a loose lower bound.
    max_docs = max(64, (n_tokens // max(1, seq_len)) * 4 + 64)
    parts = []
    total_est = 0
    for ex in itertools.islice(ds, max_docs):
        msgs = ex.get("messages")
        if not msgs:
            continue
        norm = []
        for m in msgs:
            role = m.get("role") if isinstance(m, dict) else None
            content = m.get("content") if isinstance(m, dict) else None
            if role and content is not None:
                norm.append({"role": role, "content": content})
        if not norm:
            continue
        parts.append(common.llama3_chat_format(norm))
        total_est += len(parts[-1])
        # rough char->token ratio ~4; stop once we clearly have enough
        if total_est // 4 >= n_tokens + seq_len:
            break
    text = "".join(parts)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    n_windows = max(1, n_tokens // seq_len)
    # chat docs already embed <|begin_of_text|>; keep natural window starts.
    rows = []
    for w in range(n_windows):
        s = w * seq_len
        chunk = ids[s:s + seq_len]
        if len(chunk) < seq_len:
            break
        rows.append(list(chunk))
    if not rows:
        raise RuntimeError("not enough chat tokens to form a single window")
    return torch.tensor(rows, dtype=torch.long)


def batch_windows(windows: torch.Tensor, batch_size: int) -> list:
    """[N, T] -> list of [B, T] chunks (last chunk may be smaller)."""
    n = windows.shape[0]
    return [windows[s:s + batch_size] for s in range(0, n, batch_size)]
