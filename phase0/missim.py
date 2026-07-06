"""Offline cache-miss simulation over FFN firing traces.

Builds a global hot-neuron set at several cache budgets and replays sampled firing masks
to estimate per-token miss bytes and hit rate. Vectorized per layer in token blocks.
"""

import argparse
import csv
import glob
import json
import math

import numpy as np

import common

BLOCK = 2048


def bundle_bytes_q3(h):
    """One neuron bundle: up-row(H) + down-col(H) = 2H q3 vals + fp16 group scales."""
    return math.ceil(2 * h / 10) * 4 + (2 * h / 40) * 2


def load_traces(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no trace files match {pattern}")
    counts = None
    mask_parts = []
    tokens_total = 0
    for fp in files:
        z = np.load(fp, allow_pickle=False)
        c = z["counts"].astype(np.int64)
        counts = c if counts is None else counts + c
        mask_parts.append(z["masks"])
        try:
            tokens_total += int(json.loads(str(z["meta"]))["tokens"])
        except Exception:
            pass
    masks = np.concatenate(mask_parts, axis=1)  # [L, N, ipacked]
    return counts, masks, tokens_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="traces_*.npz")
    ap.add_argument("--model", default="8b-q3", choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--hot-fracs", default="0.4,0.5,0.6,0.68,0.75,0.85")
    ap.add_argument("--out-csv", default="missim.csv")
    ap.add_argument("--out-rank", default="hot_rank.json")
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    H, I, L = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers

    counts, masks, tokens_total = load_traces(args.traces)
    N = masks.shape[1]
    bb = bundle_bytes_q3(H)
    total_bundle_bytes = L * I * bb

    fracs = [float(x) for x in args.hot_fracs.split(",")]
    order = np.argsort(counts.ravel())[::-1]
    hotmask, cache_gb = {}, {}
    for fr in fracs:
        k = int(fr * L * I)
        flat = np.zeros(L * I, dtype=bool)
        flat[order[:k]] = True
        hotmask[fr] = flat.reshape(L, I)
        cache_gb[fr] = (k * bb) / 1e9

    miss_counts = {fr: np.zeros(N, dtype=np.int64) for fr in fracs}
    fired_hit = {fr: 0 for fr in fracs}
    fired_tot = 0

    for layer in range(L):
        packed = masks[layer]  # [N, ipacked]
        for s in range(0, N, BLOCK):
            blk = packed[s:s + BLOCK]
            fired = np.unpackbits(blk, axis=-1)[:, :I].astype(bool)  # [nt, I]
            fired_tot += int(fired.sum())
            for fr in fracs:
                hot = hotmask[fr][layer]
                miss_counts[fr][s:s + blk.shape[0]] += (fired & ~hot).sum(axis=1)
                fired_hit[fr] += int((fired & hot).sum())

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hot_frac", "cache_gb", "hit_rate", "mean_mb",
                    "p50_mb", "p95_mb", "p99_mb"])
        for fr in fracs:
            mb = miss_counts[fr] * bb / 1e6
            hit = fired_hit[fr] / fired_tot if fired_tot else 0.0
            if mb.size:
                mean, p50, p95, p99 = (float(mb.mean()),
                                       float(np.percentile(mb, 50)),
                                       float(np.percentile(mb, 95)),
                                       float(np.percentile(mb, 99)))
            else:
                mean = p50 = p95 = p99 = 0.0
            w.writerow([f"{fr:.2f}", f"{cache_gb[fr]:.3f}", f"{hit:.4f}",
                        f"{mean:.3f}", f"{p50:.3f}", f"{p95:.3f}", f"{p99:.3f}"])

    rank = {f"layer_{i}": [int(x) for x in np.argsort(counts[i])[::-1]]
            for i in range(L)}
    rank["bundle_bytes"] = int(round(bb))
    rank["counts_tokens"] = int(tokens_total)
    rank["n_sampled"] = int(N)
    with open(args.out_rank, "w", encoding="utf-8") as f:
        json.dump(rank, f)
    print(f"wrote {args.out_csv} and {args.out_rank}  (N={N}, bundle={bb:.0f}B)")


if __name__ == "__main__":
    main()
