"""Perplexity evaluation under a given sparsity config (wikitext test + ultrachat test).

Scores next-token CE separately per corpus, reports ppl and achieved model-wide sparsity.
Also stores raw nll/count per corpus so report.py can merge shards exactly.
"""

import argparse
import json
import math

import common
import data
from model_sim import LlamaSim, Sparsity

BYTE_CLASSES = ("qkv", "o", "gateup", "down")


def parse_shard(spec: str):
    i, n = spec.split("/")
    return int(i), int(n)


def eval_corpus(sim, windows, sp, batch, shard):
    i, n = shard
    if n > 1:
        windows = windows[i::n]
    chunks = data.batch_windows(windows, batch)
    if not chunks:
        return 0.0, 0
    states = sim.stream_forward(chunks, sp)
    return sim.score_ce(states, chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--sparsity", type=int, default=0, choices=[0, 25, 40, 50, 60])
    ap.add_argument("--select", default="teal", choices=["teal", "gate"])
    ap.add_argument("--granularity", default="element", choices=["element", "block"],
                    help="block = packing-width block sparsity (what stock device weights "
                         "support; see PROGRESS.md amendment 1)")
    ap.add_argument("--tokens", type=int, default=20480)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)
    shard = parse_shard(args.shard)

    if args.sparsity == 0:
        sp = Sparsity(None, "off")
        select = "off"
    else:
        with open(args.thresholds, encoding="utf-8") as f:
            th = json.load(f)
        key = f"{args.sparsity / 100:.2f}"
        mode = "gate" if args.select == "gate" else "teal"
        if args.granularity == "block":
            sp = Sparsity(th["grids_block"][key], mode, block_k=th.get("block_k", 1))
            select = f"{args.select}-b{th.get('block_k', 1)}"
        else:
            sp = Sparsity(th["grids"][key], mode)
            select = args.select

    half = max(args.seq_len, args.tokens // 2)
    wiki = data.wikitext_tokens(tok, half, args.seq_len, split="test")
    chat = data.chat_tokens(tok, half, args.seq_len, split="test_sft")

    sim = LlamaSim(model_dir, cfg, threads=args.threads)
    nll_w, cnt_w = eval_corpus(sim, wiki, sp, args.batch, shard)
    nll_c, cnt_c = eval_corpus(sim, chat, sp, args.batch, shard)

    ppl_w = math.exp(nll_w / cnt_w) if cnt_w else float("nan")
    ppl_c = math.exp(nll_c / cnt_c) if cnt_c else float("nan")
    tot_n, tot_c = nll_w + nll_c, cnt_w + cnt_c
    ppl_all = math.exp(tot_n / tot_c) if tot_c else float("nan")

    achieved = sp.achieved()
    bw = common.sparsifiable_byte_weights(cfg)
    achieved_mw = sum(achieved[c] * bw[c] for c in BYTE_CLASSES)

    out = args.out or f"ppl_{args.model}_s{args.sparsity}_{select}.json"
    result = {
        "model": args.model, "sparsity": args.sparsity, "select": select,
        "shard": args.shard,
        "ppl_wikitext": ppl_w, "ppl_chat": ppl_c, "ppl_overall": ppl_all,
        "tokens_scored": tot_c,
        "nll_wikitext": nll_w, "count_wikitext": cnt_w,
        "nll_chat": nll_c, "count_chat": cnt_c,
        "achieved": achieved, "achieved_model_wide": achieved_mw,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}  ppl_overall={ppl_all:.3f}  achieved_mw={achieved_mw:.3f}")


if __name__ == "__main__":
    main()
