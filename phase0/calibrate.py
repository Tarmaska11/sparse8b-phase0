"""Calibrate per-(layer,class) magnitude thresholds for a target sparsity grid.

One streamed forward over a 50/50 wikitext(train)+ultrachat(train_sft) mix, collecting
|x| histograms per (layer, class), then inverts the CDF at each grid sparsity.
"""

import argparse
import json
import os

import numpy as np

import common
import data
from model_sim import LlamaSim, Sparsity

CLASSES = ("qkv", "o", "gateup", "down", "gate_select")
GRIDS = (0.25, 0.40, 0.50, 0.60)
NBINS = 4096
EDGES = np.geomspace(1e-7, 1e2, NBINS)  # 4096 edges -> 4095 bins


class HistCollector:
    """Element |x| histograms + block-max |x| histograms (block = quant packing width).

    Block-granular thresholds are needed because q3/q4 pack 10/8 consecutive K values per
    uint32 word on device - only whole-word skips save bytes (PROGRESS.md amendment 1).
    A quantile s over the block-max distribution zeroes s of blocks = s of elements.
    """

    def __init__(self, num_layers: int, block_k: int):
        self.block_k = block_k
        self.h = {(L, c): np.zeros(NBINS - 1, dtype=np.int64)
                  for L in range(num_layers) for c in CLASSES}
        # block-max hists for matmul-input classes only (gate_select is neuron-granular)
        self.hb = {(L, c): np.zeros(NBINS - 1, dtype=np.int64)
                   for L in range(num_layers) for c in CLASSES if c != "gate_select"}

    @staticmethod
    def _hist(vals: np.ndarray) -> np.ndarray:
        vals = np.clip(vals, EDGES[0], EDGES[-1])
        counts, _ = np.histogram(vals, bins=EDGES)
        return counts.astype(np.int64)

    def __call__(self, layer: int, cls: str, x):
        ax = x.detach().abs()
        vals = ax.reshape(-1).numpy()[::4]
        if vals.size == 0:
            return
        self.h[(layer, cls)] += self._hist(vals)
        if cls != "gate_select" and self.block_k > 1:
            bk = self.block_k
            K = ax.shape[-1]
            nb = K // bk
            bmax = ax[..., : nb * bk].reshape(-1, nb, bk).amax(-1).reshape(-1).numpy()[::4]
            if bmax.size:
                self.hb[(layer, cls)] += self._hist(bmax)

    def _quantile(self, counts: np.ndarray, s: float) -> float:
        total = int(counts.sum())
        if total == 0:
            return 0.0
        cum = np.cumsum(counts)
        j = int(np.searchsorted(cum, s * total, side="left"))
        # smallest bin edge whose cumulative fraction >= s is the right edge of bin j
        return float(EDGES[min(j + 1, NBINS - 1)])

    def threshold(self, layer: int, cls: str, s: float) -> float:
        return self._quantile(self.h[(layer, cls)], s)

    def threshold_block(self, layer: int, cls: str, s: float) -> float:
        if cls == "gate_select":  # neuron-granular either way
            return self._quantile(self.h[(layer, cls)], s)
        return self._quantile(self.hb[(layer, cls)], s)


def build_windows(tok, n_tokens, seq_len):
    half = max(seq_len, n_tokens // 2)
    wiki = data.wikitext_tokens(tok, half, seq_len, split="train")
    chat = data.chat_tokens(tok, half, seq_len, split="train_sft")
    import torch
    return torch.cat([wiki, chat], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)

    windows = build_windows(tok, args.tokens, args.seq_len)
    chunks = data.batch_windows(windows, args.batch)
    calib_tokens = int(windows.shape[0] * windows.shape[1])

    block_k = common.QUANT[cfg.quantization]["elem_per_storage"]  # 10 (q3) / 8 (q4)
    sim = LlamaSim(model_dir, cfg, threads=args.threads)
    hc = HistCollector(cfg.num_hidden_layers, block_k)
    sp = Sparsity(None, "off")
    sim.stream_forward(chunks, sp, calib=hc,
                       progress=lambda li: print(f"  layer {li} done", flush=True))

    grids, grids_block = {}, {}
    for s in GRIDS:
        key = f"{s:.2f}"
        grids[key] = {str(L): {c: hc.threshold(L, c, s) for c in CLASSES}
                      for L in range(cfg.num_hidden_layers)}
        grids_block[key] = {str(L): {c: hc.threshold_block(L, c, s) for c in CLASSES}
                            for L in range(cfg.num_hidden_layers)}

    out = args.out or f"thresholds-{args.model}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "quant": cfg.quantization,
                   "calib_tokens": calib_tokens, "block_k": block_k,
                   "grids": grids, "grids_block": grids_block}, f, indent=1)
    print(f"wrote {out}  (calib_tokens={calib_tokens}, block_k={block_k})")


if __name__ == "__main__":
    main()
