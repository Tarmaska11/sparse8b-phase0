"""Collect FFN gate-select firing traces (per-neuron counts + sampled packed masks).

Runs the chat corpus in gate mode and records which intermediate neurons fire, for the
offline cache-miss simulation (missim.py).
"""

import argparse
import json

import numpy as np
import torch

import common
import data
from model_sim import LlamaSim, Sparsity

RAM_LIMIT = 2.5e9  # bytes budget for the packed masks


class TraceHook:
    def __init__(self, num_layers, inter, sample_every):
        self.L = num_layers
        self.I = inter
        self.step = sample_every
        self.counts = np.zeros((num_layers, inter), dtype=np.int64)
        self.masks = [[] for _ in range(num_layers)]  # per-layer list of packed uint8 arrays

    def __call__(self, layer, sel):
        s = sel.detach()
        B, T, I = s.shape
        self.counts[layer] += s.sum(dim=(0, 1)).numpy().astype(np.int64)
        flat = s.reshape(B * T, I).numpy()
        sampled = flat[::self.step]
        self.masks[layer].append(np.packbits(sampled, axis=-1))


def parse_shard(spec):
    i, n = spec.split("/")
    return int(i), int(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b-q3", choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--thresholds", required=True)
    ap.add_argument("--sparsity", type=int, default=50, choices=[25, 40, 50, 60])
    ap.add_argument("--select", default="gate", choices=["gate"])
    ap.add_argument("--tokens", type=int, default=10240)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sample-every", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)
    si, sn = parse_shard(args.shard)

    with open(args.thresholds, encoding="utf-8") as f:
        th = json.load(f)
    # Mirror the device config: attention/gateup inputs are block-granular (stock packing),
    # the FFN gate_select mask is neuron-granular either way.
    key = f"{args.sparsity / 100:.2f}"
    grid = th.get("grids_block", th["grids"])[key]
    block_k = th.get("block_k", 1)

    windows = data.chat_tokens(tok, args.tokens, args.seq_len, split="test_sft")
    if sn > 1:
        windows = windows[si::sn]
    chunks = data.batch_windows(windows, args.batch)
    total_tokens = int(windows.shape[0] * windows.shape[1])

    # RAM guard: grow sample_every until packed masks fit the budget.
    step = args.sample_every
    ipacked = (cfg.intermediate_size + 7) // 8
    halved = 0
    while (cfg.num_hidden_layers * (total_tokens / step) * ipacked) > RAM_LIMIT:
        step *= 2
        halved += 1

    sim = LlamaSim(model_dir, cfg, threads=args.threads)
    hook = TraceHook(cfg.num_hidden_layers, cfg.intermediate_size, step)
    sp = Sparsity(grid, "gate", block_k=block_k)
    sim.stream_forward(chunks, sp, ffn_mask_hook=hook,
                       progress=lambda li: print(f"  layer {li} done", flush=True))

    masks = np.stack([np.concatenate(hook.masks[L], axis=0) if hook.masks[L]
                      else np.zeros((0, ipacked), dtype=np.uint8)
                      for L in range(cfg.num_hidden_layers)])
    meta = json.dumps({
        "model": args.model, "tokens": total_tokens, "sparsity": args.sparsity,
        "select": "gate", "layers": cfg.num_hidden_layers,
        "intermediate": cfg.intermediate_size, "sample_every": step,
        "halved": halved, "n_sampled": int(masks.shape[1]),
    })
    out = args.out or f"traces_{si}.npz"
    np.savez_compressed(out, counts=hook.counts, masks=masks, meta=meta)
    print(f"wrote {out}  sampled={masks.shape[1]} tokens/layer  step={step}  halved={halved}")


if __name__ == "__main__":
    main()
