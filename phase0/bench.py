"""Zero-shot multiple-choice accuracy (MMLU / ARC-Challenge) under a sparsity config.

Scores the letter/label token immediately after "Answer:" via candidate-restricted logits.
"""

import argparse
import json
import random

import torch

import common
import data
from model_sim import LlamaSim, Sparsity

BYTE_CLASSES = ("qkv", "o", "gateup", "down")
BOS_ID = 128000


def parse_shard(spec):
    i, n = spec.split("/")
    return int(i), int(n)


def last_id(tok, s):
    ids = tok.encode(s, add_special_tokens=False).ids
    return ids[-1]


def mmlu_prompt(subject, question, choices):
    letters = "ABCD"
    body = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices[:4]))
    return (f"The following is a multiple choice question (with answer) about "
            f"{subject.replace('_', ' ')}.\n\n{question}\n{body}\nAnswer:")


def arc_prompt(question, labels, texts):
    body = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts))
    return f"{question}\n{body}\nAnswer:"


def sample_mmlu(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    by_subj = {}
    for ex in ds:
        by_subj.setdefault(ex["subject"], []).append(ex)
    rng = random.Random(seed)
    for s in sorted(by_subj):
        rng.shuffle(by_subj[s])
    subjects = sorted(by_subj)
    picked, k = [], 0
    while len(picked) < n:
        progressed = False
        for s in subjects:
            if k < len(by_subj[s]):
                picked.append(by_subj[s][k])
                progressed = True
                if len(picked) >= n:
                    break
        if not progressed:
            break
        k += 1
    return picked


def sample_arc(n, seed=0):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    items = []
    for ex in ds:
        labels = ex["choices"]["label"]
        if not (3 <= len(labels) <= 5):
            continue
        if ex["answerKey"] not in labels:
            continue
        items.append(ex)
    rng = random.Random(seed)
    rng.shuffle(items)
    return items[:n]


def make_sparsity(args, th):
    if args.sparsity == 0:
        return Sparsity(None, "off"), "off"
    mode = "gate" if args.select == "gate" else "teal"
    key = f"{args.sparsity / 100:.2f}"
    if getattr(args, "granularity", "element") == "block":
        bk = th.get("block_k", 1)
        return Sparsity(th["grids_block"][key], mode, block_k=bk), f"{args.select}-b{bk}"
    return Sparsity(th["grids"][key], mode), args.select


def score_questions(sim, tok, questions, seq_len, batch, sp):
    """questions: list of (prompt_str, cand_strs, gold_idx). Returns (correct, total)."""
    correct = total = 0
    for s in range(0, len(questions), batch):
        group = questions[s:s + batch]
        chunk = torch.zeros((len(group), seq_len), dtype=torch.long)
        rows = []
        for b, (prompt, cand_strs, gold) in enumerate(group):
            toks = [BOS_ID] + tok.encode(prompt, add_special_tokens=False).ids
            if len(toks) > seq_len - 1:
                toks = toks[-(seq_len - 1):]
            chunk[b, :len(toks)] = torch.tensor(toks, dtype=torch.long)
            cands = [last_id(tok, cs) for cs in cand_strs]
            rows.append((b, len(toks) - 1, cands, gold))
        states = sim.stream_forward([chunk], sp)
        for b, t, cands, gold in rows:
            logits = sim.logits_at(states, [[(b, t)]], [cands])[0]  # [1, C]
            pred = int(logits[0].argmax())
            correct += int(pred == gold)
            total += 1
    return correct, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--sparsity", type=int, default=0, choices=[0, 25, 40, 50, 60])
    ap.add_argument("--select", default="teal", choices=["teal", "gate"])
    ap.add_argument("--granularity", default="element", choices=["element", "block"])
    ap.add_argument("--suite", default="both", choices=["mmlu", "arc", "both"])
    ap.add_argument("--mmlu-n", type=int, default=200)
    ap.add_argument("--arc-n", type=int, default=100)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)
    si, sn = parse_shard(args.shard)

    th = None
    if args.sparsity != 0:
        with open(args.thresholds, encoding="utf-8") as f:
            th = json.load(f)
    sp, select = make_sparsity(args, th)

    sim = LlamaSim(model_dir, cfg, threads=args.threads)

    mmlu_acc = arc_acc = None
    mmlu_c = mmlu_t = arc_c = arc_t = 0
    if args.suite in ("mmlu", "both"):
        exs = sample_mmlu(args.mmlu_n)[si::sn] if sn > 1 else sample_mmlu(args.mmlu_n)
        qs = [(mmlu_prompt(e["subject"], e["question"], e["choices"]),
               [" A", " B", " C", " D"], int(e["answer"])) for e in exs]
        mmlu_c, mmlu_t = score_questions(sim, tok, qs, args.seq_len, args.batch, sp)
        mmlu_acc = mmlu_c / mmlu_t if mmlu_t else None
    if args.suite in ("arc", "both"):
        exs = sample_arc(args.arc_n)
        exs = exs[si::sn] if sn > 1 else exs
        qs = []
        for e in exs:
            labels, texts = e["choices"]["label"], e["choices"]["text"]
            gold = labels.index(e["answerKey"])
            qs.append((arc_prompt(e["question"], labels, texts),
                       [f" {lab}" for lab in labels], gold))
        arc_c, arc_t = score_questions(sim, tok, qs, args.seq_len, args.batch, sp)
        arc_acc = arc_c / arc_t if arc_t else None

    bw = common.sparsifiable_byte_weights(cfg)
    achieved = sp.achieved()
    achieved_mw = sum(achieved[c] * bw[c] for c in BYTE_CLASSES)

    out = args.out or f"bench_{args.model}_s{args.sparsity}_{select}.json"
    result = {
        "model": args.model, "sparsity": args.sparsity, "select": select,
        "shard": args.shard,
        "mmlu_acc": mmlu_acc, "mmlu_n": mmlu_t, "mmlu_correct": mmlu_c,
        "arc_acc": arc_acc, "arc_n": arc_t, "arc_correct": arc_c,
        "achieved_model_wide": achieved_mw,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}  mmlu={mmlu_acc}  arc={arc_acc}")


if __name__ == "__main__":
    main()
