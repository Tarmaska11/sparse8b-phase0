"""Bit-exact loader/dequantizer for MLC group-quant shards (q3f16_0/1, q4f16_0/1).

Reads ndarray-cache.json + params_shard_*.bin and reproduces mlc-llm's dequant exactly:
    w = (float(q) - max_int) * scale        # max_int: q3=3, q4=7  (see PROGRESS.md session 1)
Group along K (axis of reduction), one fp16 scale per group (q3: 40, q4: 32).
Layouts: *_1 = NK (q_weight [N, K_words]); *_0 = KN (q_weight [K_words, N]).
Output is always a torch.float32 tensor of shape [N, K] (or [rows, cols] for embeddings).
"""

import json
import os

import numpy as np
import torch

from common import QUANT


class ShardStore:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        # newer MLC writes tensor-cache.json; prebuilt HF dirs ship ndarray-cache.json
        cache_path = next(
            p for n in ("tensor-cache.json", "ndarray-cache.json")
            if os.path.isfile(p := os.path.join(model_dir, n))
        )
        with open(cache_path, encoding="utf-8") as f:
            cache = json.load(f)
        self.index = {}
        for shard in cache["records"]:
            path = shard["dataPath"]
            for rec in shard["records"]:
                self.index[rec["name"]] = (path, rec["byteOffset"], rec["nbytes"],
                                           tuple(rec["shape"]), rec["dtype"])
        self._open = {}

    def names(self):
        return self.index.keys()

    def raw(self, name: str) -> np.ndarray:
        path, off, nbytes, shape, dtype = self.index[name]
        f = self._open.get(path)
        if f is None:
            if len(self._open) > 8:  # keep few fds
                for h in self._open.values():
                    h.close()
                self._open.clear()
            f = open(os.path.join(self.model_dir, path), "rb")
            self._open[path] = f
        f.seek(off)
        buf = f.read(nbytes)
        np_dtype = {"uint32": np.uint32, "float16": np.float16, "float32": np.float32,
                    "bfloat16": np.uint16}[dtype]
        arr = np.frombuffer(buf, dtype=np_dtype).reshape(shape)
        if dtype == "bfloat16":  # raw bf16 -> fp32
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        return arr


def _unpack(qw: np.ndarray, bits: int, elem_per_storage: int) -> np.ndarray:
    """[.., W] uint32 -> [.., W*elem] float32 of raw codes."""
    mask = (1 << bits) - 1
    shifts = np.arange(elem_per_storage, dtype=np.uint32) * bits
    # broadcast: [.., W, 1] >> [elem] -> [.., W, elem]
    vals = (qw[..., None] >> shifts) & mask
    return vals.reshape(*qw.shape[:-1], qw.shape[-1] * elem_per_storage).astype(np.float32)


def dequant(qw: np.ndarray, scale: np.ndarray, quant: str, k_logical: int) -> torch.Tensor:
    """Dequantize one linear/embedding weight to [N, K] fp32 torch tensor.

    qw/scale exactly as stored. K padding (to a group multiple) is trimmed.
    """
    q = QUANT[quant]
    if q["layout"] == "NK":
        # qw [N, W], scale [N, G]; K runs along the last axis
        codes = _unpack(qw, q["bits"], q["elem_per_storage"])  # [N, W*elem]
        s = np.repeat(scale.astype(np.float32), q["group_size"], axis=-1)  # [N, G*gs]
        k_padded = min(codes.shape[-1], s.shape[-1])
        w = (codes[..., :k_padded] - q["max_int"]) * s[..., :k_padded]
        return torch.from_numpy(w[..., :k_logical].copy())
    else:  # KN: qw [W, N], scale [G, N]; K runs along axis 0
        codes = _unpack(qw.T.copy(), q["bits"], q["elem_per_storage"])  # [N, W*elem] via transpose
        s = np.repeat(scale.astype(np.float32).T, q["group_size"], axis=-1)  # [N, G*gs]
        k_padded = min(codes.shape[-1], s.shape[-1])
        w = (codes[..., :k_padded] - q["max_int"]) * s[..., :k_padded]
        return torch.from_numpy(w[..., :k_logical].copy())


class WeightSource:
    """Named access to dequantized model tensors, one layer at a time."""

    def __init__(self, model_dir: str, quant: str):
        self.store = ShardStore(model_dir)
        self.quant = quant

    def linear(self, name: str, k_logical: int) -> torch.Tensor:
        qw = self.store.raw(f"{name}.q_weight")
        sc = self.store.raw(f"{name}.q_scale")
        return dequant(qw, sc, self.quant, k_logical)

    def norm(self, name: str) -> torch.Tensor:
        return torch.from_numpy(self.store.raw(f"{name}.weight").astype(np.float32).copy())

    def embedding_rows(self, row_ids: np.ndarray) -> torch.Tensor:
        """Dequantize ONLY the given vocab rows of the embedding table -> [len(rows), >=hidden].

        Embedding storage is always [vocab, hidden_words] regardless of linear layout.
        Full-table dequant is a multi-GB blow-up (128256x4120 fp32) - never do it.
        """
        qw = self.store.raw("model.embed_tokens.q_weight")[row_ids]
        sc = self.store.raw("model.embed_tokens.q_scale")[row_ids]
        q = QUANT[self.quant]
        codes = _unpack(qw, q["bits"], q["elem_per_storage"])
        s = np.repeat(sc.astype(np.float32), q["group_size"], axis=-1)
        k_padded = min(codes.shape[-1], s.shape[-1])
        w = (codes[..., :k_padded] - q["max_int"]) * s[..., :k_padded]
        return torch.from_numpy(w.copy())

    def linear_rows(self, name: str, k_logical: int, n_start: int, n_end: int) -> torch.Tensor:
        """Dequantize output rows [n_start:n_end] of a linear -> [n_end-n_start, K] fp32.

        Used to stream lm_head in vocab blocks instead of materializing [128256, 4096] fp32.
        """
        qw = self.store.raw(f"{name}.q_weight")
        sc = self.store.raw(f"{name}.q_scale")
        if QUANT[self.quant]["layout"] == "NK":   # [N, W] / [N, G]
            qw, sc = qw[n_start:n_end], sc[n_start:n_end]
        else:                                     # KN: [W, N] / [G, N]
            qw, sc = qw[:, n_start:n_end], sc[:, n_start:n_end]
        return dequant(qw, sc, self.quant, k_logical)

    def has(self, name: str) -> bool:
        return name in self.store.index
