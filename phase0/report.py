"""Merge Phase-0 artifacts (ppl/bench shards + missim.csv) into REPORT.md + GATE-S1 verdict.

Combines ppl shards by summing raw nll/count, bench shards by summing correct/total.
"""

import argparse
import csv
import glob
import json
import math
import os


def load_jsons(d, prefix):
    out = []
    for fp in glob.glob(os.path.join(d, f"{prefix}*.json")):
        with open(fp, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def agg_ppl(rows):
    """key (model,sparsity,select) -> merged ppl metrics."""
    acc = {}
    for r in rows:
        k = (r["model"], r["sparsity"], r["select"])
        a = acc.setdefault(k, dict(nw=0.0, cw=0, nc=0.0, cc=0, amw=0.0, tok=0))
        a["nw"] += r.get("nll_wikitext", 0.0); a["cw"] += r.get("count_wikitext", 0)
        a["nc"] += r.get("nll_chat", 0.0); a["cc"] += r.get("count_chat", 0)
        t = r.get("tokens_scored", 0)
        a["amw"] += r.get("achieved_model_wide", 0.0) * t; a["tok"] += t
    res = {}
    for k, a in acc.items():
        pw = math.exp(a["nw"] / a["cw"]) if a["cw"] else float("nan")
        pc = math.exp(a["nc"] / a["cc"]) if a["cc"] else float("nan")
        tot_c = a["cw"] + a["cc"]
        po = math.exp((a["nw"] + a["nc"]) / tot_c) if tot_c else float("nan")
        amw = a["amw"] / a["tok"] if a["tok"] else 0.0
        res[k] = dict(ppl_w=pw, ppl_c=pc, ppl_o=po, amw=amw)
    return res


def agg_bench(rows):
    acc = {}
    for r in rows:
        k = (r["model"], r["sparsity"], r["select"])
        a = acc.setdefault(k, dict(mc=0, mn=0, ac=0, an=0))
        a["mc"] += r.get("mmlu_correct", 0); a["mn"] += r.get("mmlu_n", 0)
        a["ac"] += r.get("arc_correct", 0); a["an"] += r.get("arc_n", 0)
    res = {}
    for k, a in acc.items():
        res[k] = dict(mmlu=a["mc"] / a["mn"] if a["mn"] else None,
                      arc=a["ac"] / a["an"] if a["an"] else None,
                      mn=a["mn"], an=a["an"])
    return res


def load_missim(d):
    fp = os.path.join(d, "missim.csv")
    if not os.path.isfile(fp):
        return []
    with open(fp, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default="REPORT.md")
    args = ap.parse_args()

    ppl = agg_ppl(load_jsons(args.dir, "ppl_"))
    bench = agg_bench(load_jsons(args.dir, "bench_"))
    miss = load_missim(args.dir)

    lines = ["# Sparse8B Phase-0 Report", ""]

    # Table 1: perplexity
    lines += ["## Perplexity", "",
              "| model | sparsity | select | wikitext | chat | overall | achieved_mw |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted(ppl):
        m, s, sel = k
        p = ppl[k]
        lines.append(f"| {m} | {s} | {sel} | {fmt(p['ppl_w'])} | {fmt(p['ppl_c'])} "
                     f"| {fmt(p['ppl_o'])} | {fmt(p['amw'])} |")
    lines.append("")

    # Table 2: bench
    lines += ["## Zero-shot accuracy", "",
              "| model | sparsity | select | mmlu_acc | mmlu_n | arc_acc | arc_n |",
              "|---|---|---|---|---|---|---|"]
    for k in sorted(bench):
        m, s, sel = k
        b = bench[k]
        lines.append(f"| {m} | {s} | {sel} | {fmt(b['mmlu'])} | {b['mn']} "
                     f"| {fmt(b['arc'])} | {b['an']} |")
    lines.append("")

    # Table 3: missim
    lines += ["## Cache-miss simulation", "",
              "| hot_frac | cache_gb | hit_rate | mean_mb | p50_mb | p95_mb | p99_mb |",
              "|---|---|---|---|---|---|---|"]
    for r in miss:
        lines.append(f"| {r['hot_frac']} | {r['cache_gb']} | {r['hit_rate']} "
                     f"| {r['mean_mb']} | {r['p50_mb']} | {r['p95_mb']} | {r['p99_mb']} |")
    lines.append("")

    # GATE-S1 verdict
    base = ppl.get(("3b-q4", 0, "off"))
    base_b = bench.get(("3b-q4", 0, "off"))
    p95_ok = None
    if miss:
        closest = min(miss, key=lambda r: abs(float(r["hot_frac"]) - 0.68))
        p95_ok = float(closest["p95_mb"]) <= 25.0
        p95_val = float(closest["p95_mb"])
    lines += ["## GATE-S1 verdict", ""]
    reasons = []
    winner = None
    if base is None or base_b is None:
        reasons.append("missing 3b-q4 dense baseline (ppl and/or bench)")
    else:
        cands = []
        for k in ppl:
            m, s, sel = k
            if s == 0:
                continue
            b = bench.get(k)
            p = ppl[k]
            if b is None or b["mmlu"] is None or b["arc"] is None:
                continue
            ok = (p["amw"] >= 0.40 and p["ppl_o"] < base["ppl_o"]
                  and b["mmlu"] >= base_b["mmlu"] and b["arc"] >= base_b["arc"]
                  and (p95_ok is True))
            if ok:
                cands.append((p["ppl_o"], k))
        if cands:
            cands.sort()
            winner = cands[0][1]
    verdict = "PASS" if winner else "FAIL"
    lines.append(f"**{verdict}**")
    lines.append("")
    if winner:
        m, s, sel = winner
        p = ppl[winner]; b = bench[winner]
        lines += [f"- Winning config: **{m} s{s} {sel}**",
                  f"- ppl_overall {fmt(p['ppl_o'])} (3b dense {fmt(base['ppl_o'])})",
                  f"- mmlu {fmt(b['mmlu'])} (3b {fmt(base_b['mmlu'])}), "
                  f"arc {fmt(b['arc'])} (3b {fmt(base_b['arc'])})",
                  f"- achieved_mw {fmt(p['amw'])}"]
    else:
        if base is not None and p95_ok is False:
            reasons.append(f"p95 miss {p95_val:.1f}MB > 25MB at hot_frac~0.68")
        if base is not None and p95_ok is None:
            reasons.append("missing missim.csv (cache-miss criterion unevaluated)")
        if not reasons:
            reasons.append("no sparse config beat all 3b-dense baselines")
        lines += ["No config satisfied GATE-S1. Reasons:"] + [f"- {r}" for r in reasons]
    lines.append("")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"wrote {args.out}  verdict={verdict}")


if __name__ == "__main__":
    main()
