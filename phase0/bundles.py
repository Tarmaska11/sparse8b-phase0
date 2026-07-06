"""Offline converter: MLC q4f16_0 model dir -> clarity-ffn-bundles.bin (the M3 residency FFN store).

ARCH-AGNOSTIC, CPU/CI only, numpy repack (NO torch training). Emits, per FFN layer, one contiguous
BUNDLE per intermediate neuron j:
    bundle_j = [ up_col_j (H q4 vals, group32 along H) | down_row_j (H q4 vals, group32 along H)
                 | up_scales (NG fp16) | down_scales (NG fp16) | pad-to-64B ]
Neurons are laid out in a co-activation / frequency PERM order and grouped into CLUSTERS of
`--cluster` bundles, so one cold cluster is a single ~64 KiB UFS read (microbench/RESULTS.md: 3.7 KB
bundle reads = 18 MB/s, 64 KiB = 143 MB/s). The runtime (cpp/serve/clarity_sparse/residency.cc)
mmaps this file, pins the hot clusters, and streams the rest.

WHAT IS LOSSLESS vs REQUANTIZED
-------------------------------
* up_col_j is copied DIRECTLY from the source q4 codes+scales: the fused gate_up_proj is quantized
  q4f16_0 KN = group32 ALONG H already, so up half column (I+j) IS a native q4-group32-along-H
  vector. No dequant, bit-exact.
* down_row_j MUST be requantized. down_proj is quantized along its reduction axis (=I, the neuron
  axis), so a single neuron's H down weights are one lane scattered across all H columns with a
  per-H scale -> they cannot be re-expressed as q4-group32-along-H without a fresh quantization.
  We dequantize down_proj (via q3loader) and RE-QUANTIZE each down_row_j to q4 group32 along H
  (MLC rule: max_int=7, scale=max|w|/7, q=clip(round(w/scale+7),0,14)). Small extra error on down
  only; keeps the bundle q4-sized and the kernel uniform. (The Phase-0 element-TEAL quality numbers
  are on the source quant; this adds a down-requant delta — validate on device, see M3-TODO.)

The header carries H, I, n_layers, cluster_size, strides + a per-layer {n_clusters, perm, offsets},
all derived from the model config/manifest (NOT hardcoded) so any Llama-3.1-8B fine-tune drops in,
and other archs work by pointing --gate-up-name / --down-name at their fused-gate_up / down tensors.

CLI:
    python bundles.py --model <mlc q4f16_0 dir> --out clarity-ffn-bundles.bin \
        [--firing-counts hot_rank.json] [--cluster 16]

DO NOT run on big weights locally (thermal). This is a CI / dad's-PC job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys

import numpy as np

from common import load_model_config
from q3loader import ShardStore, dequant

MAGIC = b"CLRB"
VERSION = 1
QUANT = "q4f16_0"
P = 8            # q4: 8 vals / uint32
GROUP = 32       # q4f16_0 group size (along the packed axis)
MAX_INT = 7      # q4 dequant/requant offset (w = (q-7)*scale)


def _ceil(a: int, b: int) -> int:
    return (a + b - 1) // b


def _f32_to_f16_bits(x: np.ndarray) -> np.ndarray:
    """float32 array -> uint16 IEEE half bits (numpy native)."""
    return x.astype(np.float16).view(np.uint16)


def requant_q4_along_last(vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Requantize a [N, H] float array to q4 group32 along H.

    Returns (words[N, ceil(H/8)] uint32, scales[N, ceil(H/32)] uint16-halfbits).
    Matches MLC q4f16_0: per group scale = max|w|/7 ; q = clip(round(w/scale + 7), 0, 14).
    """
    N, H = vals.shape
    ng = _ceil(H, GROUP)
    hpad = ng * GROUP
    padded = np.zeros((N, hpad), dtype=np.float32)
    padded[:, :H] = vals
    g = padded.reshape(N, ng, GROUP)
    maxabs = np.max(np.abs(g), axis=2)                 # [N, ng]
    scale = maxabs / MAX_INT
    scale[scale == 0.0] = 1.0                          # dead group -> unit scale, codes become 7 (=0)
    codes = np.rint(g / scale[:, :, None] + MAX_INT)
    codes = np.clip(codes, 0, 2 * MAX_INT).astype(np.uint32).reshape(N, hpad)
    # pack 8 codes/uint32 along H
    nw = _ceil(H, P)
    wpad = nw * P
    if wpad != hpad:  # ng*32 is already a multiple of 8, so hpad==wpad; guard anyway
        cc = np.zeros((N, wpad), dtype=np.uint32)
        cc[:, :hpad] = codes
        codes = cc
    codes = codes.reshape(N, nw, P)
    words = np.zeros((N, nw), dtype=np.uint32)
    for j in range(P):
        words |= (codes[:, :, j] & 0xF) << (4 * j)
    return words, _f32_to_f16_bits(scale)


def _load_perm(I: int, firing_counts_path: str | None, layer: int) -> np.ndarray:
    """perm[slot] = ORIGINAL neuron id, ordered hottest-first when firing counts are given.

    firing-counts JSON: {"layers": {"<L>": [count per neuron ...]}} OR a flat {"<L>": [...]}. If
    absent for this layer -> identity (v1 fallback; the runtime still works, hot cache just holds the
    low-index neurons)."""
    if firing_counts_path:
        with open(firing_counts_path, encoding="utf-8") as f:
            fc = json.load(f)
        table = fc.get("layers", fc)
        arr = table.get(str(layer)) if isinstance(table, dict) else None
        if arr is not None and len(arr) == I:
            counts = np.asarray(arr, dtype=np.float64)
            return np.argsort(-counts, kind="stable").astype(np.uint32)
    return np.arange(I, dtype=np.uint32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="MLC q4f16_0 model dir (ndarray-cache.json + shards)")
    ap.add_argument("--out", default="clarity-ffn-bundles.bin")
    ap.add_argument("--firing-counts", default=None, help="optional hot_rank.json (per-layer neuron counts)")
    ap.add_argument("--cluster", type=int, default=16, help="neurons per cluster (power of two; ~16 -> 64KiB)")
    ap.add_argument("--gate-up-name", default="model.layers.{L}.mlp.gate_up_proj",
                    help="fused gate_up tensor name template ({L}=layer). up = second half [I:2I].")
    ap.add_argument("--down-name", default="model.layers.{L}.mlp.down_proj",
                    help="down_proj tensor name template ({L}=layer)")
    args = ap.parse_args()

    cs = args.cluster
    if cs & (cs - 1):
        sys.exit(f"--cluster must be a power of two (got {cs}); the runtime uses shift/mask on it.")

    cfg = load_model_config(args.model)
    if cfg.quantization != QUANT:
        sys.exit(f"model quantization is {cfg.quantization!r}; residency bundles require {QUANT!r} "
                 f"(P=8 pow2). Convert the model with mlc convert_weight --quantization q4f16_0.")
    H = cfg.hidden_size
    I = cfg.intermediate_size
    L = cfg.num_hidden_layers
    store = ShardStore(args.model)

    upw = _ceil(H, P)
    dnw = _ceil(H, P)
    ng = _ceil(H, GROUP)
    bundle_stride = ((upw + dnw) * 4 + (2 * ng) * 2 + 63) // 64 * 64  # bytes, 64B aligned
    cluster_stride = cs * bundle_stride
    n_clusters = _ceil(I, cs)
    upsc_h = (upw + dnw) * 2       # half-index of up scales within a bundle
    dnsc_h = upsc_h + ng
    bundle_stride_words = bundle_stride // 4

    # deterministic layout: header(128) + table(L*32) + per layer [ perm(I u32) | data(n_clusters*cluster_stride) ]
    header_bytes = 128
    table_bytes = L * 32
    perm_bytes = I * 4
    layer_data_bytes = n_clusters * cluster_stride
    layer_block = perm_bytes + layer_data_bytes

    meta = {
        "magic": "CLRB", "version": VERSION, "quant": QUANT, "n_layers": L, "H": H, "I": I,
        "P": P, "group": GROUP, "cluster_size": cs, "bundle_stride": bundle_stride,
        "cluster_stride": cluster_stride, "n_clusters_per_layer": n_clusters,
        "upw": upw, "dnw": dnw, "ng": ng, "upsc_h": upsc_h, "dnsc_h": dnsc_h,
        "total_bytes": header_bytes + table_bytes + L * layer_block,
        "firing_counts": args.firing_counts or "identity",
        "note": "up=direct-copy q4; down=requantized q4-group32-along-H",
    }
    print(f"[bundles] H={H} I={I} layers={L} cluster={cs} bundle={bundle_stride}B "
          f"cluster={cluster_stride}B n_clusters={n_clusters} total={meta['total_bytes']/1e9:.2f} GB")

    with open(args.out, "wb") as fout:
        # --- header (128 bytes) ---
        hdr = struct.pack(
            "<4s IIII 8s IIIIIIIII",
            MAGIC, VERSION, L, H, I,
            QUANT.encode("ascii"),
            GROUP, P, cs, bundle_stride, cluster_stride, upw, dnw, ng, 0,  # flags=0
        )
        hdr = hdr + b"\x00" * (header_bytes - len(hdr))
        assert len(hdr) == header_bytes, len(hdr)
        fout.write(hdr)

        # --- layer table ---
        for li in range(L):
            perm_offset = header_bytes + table_bytes + li * layer_block
            data_offset = perm_offset + perm_bytes
            fout.write(struct.pack("<II QQQ", n_clusters, 0, perm_offset, data_offset, layer_data_bytes))

        # --- per-layer perm + cluster data ---
        for li in range(L):
            perm = _load_perm(I, args.firing_counts, li)   # [I] slot->orig
            fout.write(perm.astype("<u4").tobytes())

            gate_up = args.gate_up_name.format(L=li)
            down = args.down_name.format(L=li)
            # up: RAW q4 codes+scales, take the up half (columns [I, 2I)) of the fused gate_up (KN)
            up_qw = store.raw(f"{gate_up}.q_weight")     # [ceil(H/8), 2I] uint32 (KN)
            up_qs = store.raw(f"{gate_up}.q_scale")      # [ceil(H/32), 2I] fp16
            if up_qw.shape[1] != 2 * I:
                sys.exit(f"{gate_up}.q_weight cols {up_qw.shape[1]} != 2I {2*I}; is gate_up fused?")
            up_words_all = up_qw[:, I:2 * I].T.copy()                    # [I, upw] uint32
            up_scales_all = up_qs[:, I:2 * I].T.copy().view(np.uint16)   # [I, ng] halfbits
            # down: dequant [H, I] then requant each column j (down_row_j = column j) to q4-along-H
            down_f = dequant(store.raw(f"{down}.q_weight"), store.raw(f"{down}.q_scale"),
                             QUANT, I).numpy().astype(np.float32)        # [H, I]
            down_words_all, down_scales_all = requant_q4_along_last(down_f.T.copy())  # [I, dnw],[I, ng]

            # assemble cluster data in perm order
            data = np.zeros((n_clusters * cs, bundle_stride), dtype=np.uint8)
            for slot in range(I):
                orig = int(perm[slot])
                b = data[slot]
                # up words | down words (as uint32 little-endian bytes)
                uw = up_words_all[orig].astype("<u4").tobytes()
                dw = down_words_all[orig].astype("<u4").tobytes()
                b[0:len(uw)] = np.frombuffer(uw, dtype=np.uint8)
                b[len(uw):len(uw) + len(dw)] = np.frombuffer(dw, dtype=np.uint8)
                # scales (fp16 halfbits) at upsc_h/dnsc_h (in HALF units -> *2 for byte offset)
                us = up_scales_all[orig].astype("<u2").tobytes()
                ds = down_scales_all[orig].astype("<u2").tobytes()
                b[upsc_h * 2:upsc_h * 2 + len(us)] = np.frombuffer(us, dtype=np.uint8)
                b[dnsc_h * 2:dnsc_h * 2 + len(ds)] = np.frombuffer(ds, dtype=np.uint8)
            fout.write(data.tobytes())
            print(f"[bundles] layer {li+1}/{L} written", flush=True)

    with open(args.out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[bundles] wrote {args.out} + {args.out}.meta.json")


if __name__ == "__main__":
    main()
