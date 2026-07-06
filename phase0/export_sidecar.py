"""Export a phase-0 calibrated threshold file into the flat clarity-sparse.json sidecar
that the on-device Clarity sparse-decode runtime (cpp/serve/clarity_sparse/sparse_runtime.cc)
loads at engine reload via "clarity.sparse.configure".

INPUT  (produced by phase0/calibrate.py), thresholds-<model>.json:
    {
      "block_k": 10,                      # 10 (q3) / 8 (q4) = elem_per_storage
      "grids_block": {
        "0.50": { "0": {"qkv": t, "o": t, "gateup": t, "down": t, "gate_select": tau}, ... },
        "0.60": { ... }, ...
      },
      ...
    }

OUTPUT clarity-sparse.json (drop next to the MLC model dir):
    {
      "mode": "teal",
      "block_k": 10,
      "thresholds": {
        "model.layers.0.self_attn.qkv_proj": t,
        "model.layers.0.self_attn.o_proj":   t,
        "model.layers.0.mlp.gate_up_proj":   t,
        "model.layers.0.mlp.down_proj":      t,
        ...
      }
    }

Class -> full-tag mapping (matches the tags the compiled graph passes to clarity.sparse_gemv):
    qkv    -> self_attn.qkv_proj
    o      -> self_attn.o_proj
    gateup -> mlp.gate_up_proj
    down   -> mlp.down_proj
    gate_select -> NOT exported in teal mode (neuron-granular selection rule, not a GEMV input).

Usage:
    python export_sidecar.py --thresholds thresholds-llama3.json --sparsity 50 \
        --mode teal --out clarity-sparse.json
"""

from __future__ import annotations

import argparse
import json
import sys

# class key -> "<proj-suffix under model.layers.<i>>"
CLASS_TO_SUFFIX = {
    "qkv": "self_attn.qkv_proj",
    "o": "self_attn.o_proj",
    "gateup": "mlp.gate_up_proj",
    "down": "mlp.down_proj",
}
# gate_select is deliberately excluded (see module docstring).
EXPORT_CLASSES = ("qkv", "o", "gateup", "down")


def _pick_grid_key(grids_block: dict, sparsity_pct: int) -> str:
    """Find the grids_block key for the requested sparsity. calibrate.py writes keys as
    f"{s:.2f}" (fraction), e.g. 50 -> "0.50". Fall back to a few tolerant spellings."""
    frac = sparsity_pct / 100.0
    candidates = [f"{frac:.2f}", f"{frac:.1f}", str(frac), str(sparsity_pct), f"{sparsity_pct}"]
    for k in candidates:
        if k in grids_block:
            return k
    # last resort: numeric match against the available keys
    for k in grids_block:
        try:
            if abs(float(k) - frac) < 1e-6:
                return k
        except ValueError:
            continue
    raise SystemExit(
        f"sparsity {sparsity_pct}% (key ~{frac:.2f}) not found in grids_block; "
        f"available keys: {sorted(grids_block)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thresholds", required=True, help="phase0 thresholds-<model>.json")
    ap.add_argument("--sparsity", type=int, required=True, help="target sparsity percent, e.g. 50")
    ap.add_argument("--mode", default="teal", help="sparse mode string (default: teal)")
    ap.add_argument("--out", default="clarity-sparse.json", help="output sidecar path")
    args = ap.parse_args()

    with open(args.thresholds, encoding="utf-8") as f:
        th = json.load(f)

    if "grids_block" not in th:
        raise SystemExit("input has no 'grids_block' (need a block-granular calibration file)")
    block_k = th.get("block_k", 1)
    if block_k <= 1:
        print(f"WARNING: block_k={block_k} (expected 10 for q3 / 8 for q4); "
              "the runtime validates block_k == quant P.", file=sys.stderr)

    grid_key = _pick_grid_key(th["grids_block"], args.sparsity)
    grid = th["grids_block"][grid_key]

    thresholds: dict[str, float] = {}
    n_layers = 0
    for layer_key, classes in grid.items():
        n_layers += 1
        for cls in EXPORT_CLASSES:
            if cls not in classes:
                continue
            tag = f"model.layers.{layer_key}.{CLASS_TO_SUFFIX[cls]}"
            thresholds[tag] = float(classes[cls])

    out = {"mode": args.mode, "block_k": int(block_k), "thresholds": thresholds}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)

    print(f"wrote {args.out}  (mode={args.mode}, block_k={block_k}, sparsity_key={grid_key}, "
          f"layers={n_layers}, tags={len(thresholds)})")


if __name__ == "__main__":
    main()
