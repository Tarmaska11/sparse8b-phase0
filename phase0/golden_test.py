"""Greedy decode sanity check (no kv-cache reuse) for validating the sim against device.

Recomputes a full forward each step over the growing sequence; short + slow but exact.
"""

import argparse

import torch

import common
import data
from model_sim import LlamaSim, Sparsity

EOT_IDS = {128009, 128001}  # <|eot_id|>, <|end_of_text|>


def read_device_tokens(path, tok):
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()
    parts = raw.split()
    if parts and all(p.lstrip("-").isdigit() for p in parts):
        return [int(p) for p in parts]
    return tok.encode(raw, add_special_tokens=False).ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="8b-q3", choices=list(common.MODELS))
    ap.add_argument("--models-root", default="models")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device-tokens", default=None)
    args = ap.parse_args()

    model_dir = common.resolve_model_dir(args.model, args.models_root)
    cfg = common.load_model_config(model_dir)
    tok = data.get_tokenizer(model_dir)

    chat = common.llama3_chat_format([{"role": "user", "content": args.prompt}])
    chat += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt_ids = tok.encode(chat, add_special_tokens=False).ids

    sim = LlamaSim(model_dir, cfg, threads=args.threads)
    sp = Sparsity(None, "off")

    def full_logits(h):  # stream the head in vocab blocks; h: [H]
        parts = []
        V = cfg.vocab_size
        for vs in range(0, V, 16384):
            blk = sim._head_rows(vs, min(vs + 16384, V))
            parts.append(h @ blk.T)
        return torch.cat(parts, 0)

    ids = list(prompt_ids)
    generated = []
    for step in range(args.steps):
        chunk = torch.tensor([ids], dtype=torch.long)
        states = sim.stream_forward([chunk], sp)
        logits = full_logits(states[0][0, -1])  # [V]
        nxt = int(logits.argmax())
        if step < 4:
            top = torch.topk(logits, 5)
            pairs = [(int(i), round(float(v), 3))
                     for v, i in zip(top.values, top.indices)]
            print(f"step {step} top5: {pairs}", flush=True)
        generated.append(nxt)
        ids.append(nxt)
        if nxt in EOT_IDS:
            break

    print("\ngenerated ids:", generated)
    print("decoded:", tok.decode(generated))

    if args.device_tokens:
        dev = read_device_tokens(args.device_tokens, tok)
        div = None
        for i in range(min(len(dev), len(generated))):
            if dev[i] != generated[i]:
                div = i
                break
        if div is None and len(dev) != len(generated):
            div = min(len(dev), len(generated))
        print("device ids:", dev)
        print("first divergence index:", "none" if div is None else div)


if __name__ == "__main__":
    main()
