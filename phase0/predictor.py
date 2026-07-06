"""FFN activation-firing PREDICTOR feasibility study (CPU-only).

Scientific question
--------------------
For each FFN layer, given the layer input h (the post_attention_layernorm output,
dim = hidden_size = 4096), can a SMALL predictor rank the intermediate neurons so
that reading only the top-K predicted-firing bundles from storage captures most of
the down-proj signal?  The true down-proj input is  mid = silu(gate) * up  (dim =
intermediate_size = 14336); neuron j "fires" if |mid_j| >= tau_layer.

KEY QUALITY METRIC = MAGNITUDE RECALL:  of the total sum of |mid| over a token's
neurons, what fraction lands in the predictor's top-K (K = density * I).  A high
magnitude-recall at low density is what lets the residency engine skip reading the
cold up+down weights.

Subcommands
-----------
  collect  one streamed forward over wikitext(train)+ultrachat(train_sft); dump
           per-FFN-layer (h_in fp16, |mid| fp16) shards via the existing calib hook.
  train    per layer, fit low-rank predictors; eval magnitude/firing recall at a
           density sweep vs a static hot-set baseline and a random baseline; write
           predictor_report_<model>.json + REPORT.md.
  eval     re-run eval on already-trained... (folded into `train`; kept as alias).

Hook approach: model_sim.layer_forward calls calib(layer,'gateup', x) with x = the
FFN input h BEFORE gate_up, then calib(layer,'down', mid) with mid = silu(gate)*up
(the true down input) in the non-gate branch.  Running with Sparsity(None,'off')
makes 'down' fire with the dense mid.  We pair the two per (layer, chunk).

CPU-only feasibility harness; deterministic seeds.  Heavy runs belong on CI.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

import common
import data
from model_sim import LlamaSim, Sparsity

EPS = 1e-8
DENSITIES = (0.25, 0.40, 0.50, 0.60)
LOWRANK_RS = (64, 128, 256)


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #

def parse_layers(spec: str, num_layers: int) -> list:
    if not spec or spec == "all":
        return list(range(num_layers))
    out = sorted({int(x) for x in spec.split(",") if x.strip() != ""})
    for L in out:
        if not 0 <= L < num_layers:
            raise ValueError(f"layer {L} out of range [0,{num_layers})")
    return out


def build_windows(tok, n_tokens: int, seq_len: int):
    """50/50 wikitext(train) + ultrachat(train_sft) sliding windows (like calibrate.py)."""
    half = max(seq_len, n_tokens // 2)
    wiki = data.wikitext_tokens(tok, half, seq_len, split="train")
    chat = data.chat_tokens(tok, half, seq_len, split="train_sft")
    return torch.cat([wiki, chat], dim=0)


def shard_path(outdir: str, layer: int) -> str:
    return os.path.join(outdir, f"ffnpairs_L{layer}.npz")


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #

def cmd_collect(args):
    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)
    H, I = cfg.hidden_size, cfg.intermediate_size
    layers = parse_layers(args.layers, cfg.num_hidden_layers)
    layerset = set(layers)

    windows = build_windows(tok, args.tokens, args.seq_len)
    chunks = data.batch_windows(windows, args.batch)
    n_tokens = int(windows.shape[0] * windows.shape[1])
    print(f"collect: model={args.model} layers={layers} tokens~{n_tokens} "
          f"seq_len={args.seq_len} H={H} I={I}", flush=True)

    os.makedirs(args.out, exist_ok=True)
    sim = LlamaSim(model_dir, cfg, threads=args.threads)

    # per-target-layer accumulators (layer-major stream_forward -> flush at progress)
    cur_h = {L: [] for L in layers}
    cur_mid = {L: [] for L in layers}
    pending = {}   # layer -> h_in for the chunk currently in flight

    def calib(layer, cls, x):
        if layer not in layerset:
            return
        if cls == "gateup":                       # x = FFN input h  [B,T,H]
            pending[layer] = x.detach().reshape(-1, H).to(torch.float16).numpy()
        elif cls == "down":                       # x = mid = silu(gate)*up  [B,T,I]
            h = pending.pop(layer, None)
            if h is None:
                return
            m = x.detach().abs().reshape(-1, I).to(torch.float16).numpy()
            cur_h[layer].append(h)
            cur_mid[layer].append(m)

    def progress(li):
        # layer li fully processed across all chunks -> flush + free (bounds RAM to 1 layer)
        if li in layerset and cur_h[li]:
            h = np.concatenate(cur_h[li], axis=0)
            m = np.concatenate(cur_mid[li], axis=0)
            np.savez_compressed(shard_path(args.out, li), h_in=h, abs_mid=m)
            mb = (h.nbytes + m.nbytes) / 1e6
            print(f"  layer {li}: wrote {h.shape[0]} tokens ({mb:.0f} MB raw)", flush=True)
            cur_h[li] = []
            cur_mid[li] = []
        else:
            print(f"  layer {li} done", flush=True)

    sp = Sparsity(None, "off")                    # 'off' -> 'down' calib fires with dense mid
    sim.stream_forward(chunks, sp, calib=calib, progress=progress)

    meta = {
        "model": args.model, "quant": cfg.quantization,
        "hidden_size": H, "intermediate_size": I,
        "num_hidden_layers": cfg.num_hidden_layers,
        "layers": layers, "tokens": n_tokens, "seq_len": args.seq_len,
    }
    with open(os.path.join(args.out, "predictor_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"collect done -> {args.out}/ (predictor_meta.json + {len(layers)} shards)", flush=True)


# --------------------------------------------------------------------------- #
# predictors
# --------------------------------------------------------------------------- #

def make_predictor(kind: str, H: int, I: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    if kind == "linear":
        return nn.Linear(H, I)                    # 58.7M params/layer -- upper-bound reference
    if kind.startswith("lowrank"):
        r = int(kind[len("lowrank"):])
        return nn.Sequential(nn.Linear(H, r, bias=False), nn.GELU(), nn.Linear(r, I))
    raise ValueError(f"unknown predictor kind {kind!r}")


def predictor_bytes(model: nn.Module) -> int:
    return 2 * sum(p.numel() for p in model.parameters())   # fp16 storage


def train_predictor(model, Xtr, Ytr, epochs, bs, lr, wd, seed):
    """MSE on globally-standardized log|mid| (monotonic -> ranking == ranking by |mid|)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    N = Xtr.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(N, generator=g)
        for i in range(0, N, bs):
            idx = perm[i:i + bs]
            xb = Xtr[idx].float()
            yb = Ytr[idx].float()
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()


# --------------------------------------------------------------------------- #
# eval metrics (vectorized over val tokens)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def scores_of(model, X, bs=256):
    outs = []
    for i in range(0, X.shape[0], bs):
        outs.append(model(X[i:i + bs].float()))
    return torch.cat(outs, 0)


def magnitude_recall(scores, absmid, K):
    topk = scores.topk(K, dim=1).indices
    captured = absmid.gather(1, topk).sum(1)
    total = absmid.sum(1).clamp_min(EPS)
    return float((captured / total).mean())


def firing_recall(scores, fire, K):
    nfire = fire.sum(1).clamp_min(1)
    topk = scores.topk(K, dim=1).indices
    hit = fire.gather(1, topk).float().sum(1)
    return float((hit / nfire.float()).mean())


def eval_layer(scores, absmid, fire, K):
    return {"firing_recall": round(firing_recall(scores, fire, K), 4),
            "magnitude_recall": round(magnitude_recall(scores, absmid, K), 4)}


# --------------------------------------------------------------------------- #
# train + eval driver
# --------------------------------------------------------------------------- #

def load_shard(outdir, layer):
    d = np.load(shard_path(outdir, layer))
    return d["h_in"], d["abs_mid"]      # fp16 [N,H], fp16 [N,I]


def tau_for_layer(thresholds, layer, absmid_tr, firing_sparsity):
    """down-class threshold at firing_sparsity; fall back to an empirical quantile."""
    if thresholds is not None:
        try:
            key = f"{firing_sparsity:.2f}"
            return float(thresholds["grids"][key][str(layer)]["down"])
        except (KeyError, TypeError):
            pass
    sub = absmid_tr.reshape(-1)
    if sub.shape[0] > 2_000_000:
        sub = sub[:: sub.shape[0] // 2_000_000 + 1]
    return float(np.quantile(sub.astype(np.float32), firing_sparsity))


def cmd_train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(os.path.join(args.data, "predictor_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    H, I = meta["hidden_size"], meta["intermediate_size"]
    avail = set(meta["layers"])
    want = parse_layers(args.layers, meta["num_hidden_layers"])
    layers = [L for L in want if L in avail]
    if not layers:
        raise SystemExit(f"none of --layers {want} present in shards {sorted(avail)}")

    thresholds = None
    if args.thresholds and os.path.isfile(args.thresholds):
        with open(args.thresholds, encoding="utf-8") as f:
            thresholds = json.load(f)

    kinds = [k.strip() for k in args.predictors.split(",") if k.strip()]
    report = {"model": meta["model"], "firing_sparsity": args.firing_sparsity,
              "densities": list(DENSITIES), "hidden_size": H, "intermediate_size": I,
              "layers": {}, "predictor_bytes": {}}

    for L in layers:
        h_np, m_np = load_shard(args.data, L)
        N = h_np.shape[0]
        n_val = max(1, int(round(N * args.val_frac)))
        n_tr = N - n_val
        if n_tr < 1:
            raise SystemExit(f"layer {L}: too few tokens ({N}) to split")
        X = torch.from_numpy(h_np)                 # fp16 [N,H]
        A = torch.from_numpy(m_np)                 # fp16 [N,I]  (|mid|)
        Xtr, Xval = X[:n_tr], X[n_tr:]
        Atr, Aval = A[:n_tr], A[n_tr:]

        # target = globally standardized log|mid| (affine-monotonic in |mid|)
        logm = torch.log(Atr.float() + EPS)
        mu, sigma = float(logm.mean()), float(logm.std()) + EPS
        Ytr = ((torch.log(Atr.float() + EPS) - mu) / sigma).to(torch.float16)

        tau = tau_for_layer(thresholds, L, m_np[:n_tr], args.firing_sparsity)
        Aval_f = Aval.float()
        fire_val = Aval_f >= tau

        lay = {"tokens": N, "n_train": n_tr, "n_val": n_val, "tau": tau,
               "mean_fire_frac": round(float(fire_val.float().mean()), 4),
               "densities": {}}

        # static hot-set baseline: rank by mean |mid| over TRAIN (input-independent)
        static_score = Atr.float().mean(0, keepdim=True).expand(n_val, -1)
        # random baseline: per-token random scores (seeded)
        rg = torch.Generator().manual_seed(args.seed + L)
        rand_score = torch.rand(n_val, I, generator=rg)

        # train each predictor once; eval across densities
        trained = {}
        for kind in kinds:
            epochs = args.linear_epochs if kind == "linear" else args.epochs
            model = make_predictor(kind, H, I, args.seed)
            report["predictor_bytes"][kind] = predictor_bytes(model)
            train_predictor(model, Xtr, Ytr, epochs, args.batch_size,
                            args.lr, args.weight_decay, args.seed)
            trained[kind] = scores_of(model, Xval)
            print(f"  layer {L} {kind}: trained ({epochs} ep, "
                  f"{report['predictor_bytes'][kind]/1e6:.2f} MB)", flush=True)

        for d in DENSITIES:
            K = int(round(d * I))
            dd = {"K": K,
                  "static_baseline": eval_layer(static_score, Aval_f, fire_val, K),
                  "random_baseline": eval_layer(rand_score, Aval_f, fire_val, K),
                  "predictors": {}}
            for kind in kinds:
                dd["predictors"][kind] = eval_layer(trained[kind], Aval_f, fire_val, K)
            lay["densities"][f"{d:.2f}"] = dd

        report["layers"][str(L)] = lay
        print(f"layer {L} done (tau={tau:.4g}, fire={lay['mean_fire_frac']})", flush=True)

    report["avg"] = _average(report, kinds)
    out = args.out or f"predictor_report_{meta['model']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    _write_markdown(report, kinds, os.path.join(os.path.dirname(out) or ".", "REPORT.md"))
    _print_summary(report, kinds)
    print(f"wrote {out} + REPORT.md", flush=True)


def _average(report, kinds):
    """Average each metric across layers, per density."""
    layers = report["layers"]
    avg = {}
    for d in DENSITIES:
        dk = f"{d:.2f}"
        acc = {"static_baseline": {"firing_recall": [], "magnitude_recall": []},
               "random_baseline": {"firing_recall": [], "magnitude_recall": []},
               "predictors": {k: {"firing_recall": [], "magnitude_recall": []} for k in kinds}}
        for L in layers.values():
            cell = L["densities"][dk]
            for base in ("static_baseline", "random_baseline"):
                for m in ("firing_recall", "magnitude_recall"):
                    acc[base][m].append(cell[base][m])
            for k in kinds:
                for m in ("firing_recall", "magnitude_recall"):
                    acc["predictors"][k][m].append(cell["predictors"][k][m])

        def mean(lst):
            return round(sum(lst) / len(lst), 4) if lst else None
        avg[dk] = {
            "static_baseline": {m: mean(acc["static_baseline"][m])
                                for m in ("firing_recall", "magnitude_recall")},
            "random_baseline": {m: mean(acc["random_baseline"][m])
                                for m in ("firing_recall", "magnitude_recall")},
            "predictors": {k: {m: mean(acc["predictors"][k][m])
                               for m in ("firing_recall", "magnitude_recall")} for k in kinds},
        }
    return avg


def _write_markdown(report, kinds, path):
    L = [f"# FFN firing-predictor feasibility — {report['model']}", "",
         f"Layers: {sorted(int(x) for x in report['layers'])} | "
         f"firing_sparsity(tau)={report['firing_sparsity']} | "
         f"hidden={report['hidden_size']} inter={report['intermediate_size']}", "",
         "Predictor sizes (per layer, fp16):", ""]
    for k in kinds:
        b = report["predictor_bytes"].get(k)
        if b is not None:
            L.append(f"- `{k}`: {b/1e6:.2f} MB ({b//2:,} params)")
    L += ["", "## Averaged magnitude-recall (the key metric) vs density", "",
          "| density | K | " + " | ".join(kinds) + " | static | random |",
          "|---|---|" + "|".join(["---"] * (len(kinds) + 2)) + "|"]
    for d in DENSITIES:
        dk = f"{d:.2f}"
        a = report["avg"][dk]
        K = int(round(d * report["intermediate_size"]))
        row = [dk, str(K)]
        row += [f"{a['predictors'][k]['magnitude_recall']}" for k in kinds]
        row += [f"{a['static_baseline']['magnitude_recall']}",
                f"{a['random_baseline']['magnitude_recall']}"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## Averaged firing-recall vs density", "",
          "| density | " + " | ".join(kinds) + " | static | random |",
          "|---|" + "|".join(["---"] * (len(kinds) + 2)) + "|"]
    for d in DENSITIES:
        dk = f"{d:.2f}"
        a = report["avg"][dk]
        row = [dk]
        row += [f"{a['predictors'][k]['firing_recall']}" for k in kinds]
        row += [f"{a['static_baseline']['firing_recall']}",
                f"{a['random_baseline']['firing_recall']}"]
        L.append("| " + " | ".join(row) + " |")
    L += ["", "Interpretation: an INPUT-CONDITIONAL predictor is worth building iff its "
          "magnitude-recall clearly beats the static hot-set baseline at low density — "
          "that gap is the signal that lets the residency engine skip cold up+down reads "
          "(see M3_SPEED_REALITY.md).", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _print_summary(report, kinds):
    print("\n=== magnitude-recall (avg over layers) ===", flush=True)
    hdr = "dens  " + "  ".join(f"{k:>10}" for k in kinds) + "     static     random"
    print(hdr, flush=True)
    for d in DENSITIES:
        a = report["avg"][f"{d:.2f}"]
        cells = "  ".join(f"{a['predictors'][k]['magnitude_recall']:>10}" for k in kinds)
        print(f"{d:.2f}  {cells}  {a['static_baseline']['magnitude_recall']:>10}  "
              f"{a['random_baseline']['magnitude_recall']:>10}", flush=True)


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("collect", help="stream a forward and dump per-layer (h_in,|mid|) shards")
    c.add_argument("--model", required=True, choices=list(common.MODELS))
    c.add_argument("--models-root", default="models")
    c.add_argument("--layers", default="0,8,16,24,31", help="comma list or 'all'")
    c.add_argument("--tokens", type=int, default=8192)
    c.add_argument("--seq-len", type=int, default=512)
    c.add_argument("--batch", type=int, default=4)
    c.add_argument("--threads", type=int, default=4)
    c.add_argument("--out", default="ffn_pairs")
    c.set_defaults(func=cmd_collect)

    t = sub.add_parser("train", help="fit predictors per layer and eval recall")
    t.add_argument("--data", default="ffn_pairs", help="collect output dir")
    t.add_argument("--layers", default="0,8,16,24,31")
    t.add_argument("--predictors", default="lowrank64,lowrank128,lowrank256,linear")
    t.add_argument("--thresholds", default=None, help="thresholds-<model>.json for tau")
    t.add_argument("--firing-sparsity", type=float, default=0.50,
                   help="down-class sparsity whose threshold defines 'firing'")
    t.add_argument("--val-frac", type=float, default=0.20)
    t.add_argument("--epochs", type=int, default=8)
    t.add_argument("--linear-epochs", type=int, default=2)
    t.add_argument("--batch-size", type=int, default=128)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None)
    t.set_defaults(func=cmd_train)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
