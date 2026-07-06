"""Shared config/utilities for the sparse8b Phase-0 harness.

Ground rules (verified against mlc-llm group_quantization.py, pinned 2008fe83):
  dequant: w = (float(q) - max_int) * scale,  max_int = 2^(bits-1)-1
    q3 (int3): max_int = 3, 10 vals/uint32, group_size 40   (NOT 3.5 - bench.c comment is wrong)
    q4 (int4): max_int = 7,  8 vals/uint32, group_size 32
  layouts: *_0 = KN (q_weight [K_words, N]), *_1 = NK (q_weight [N, K_words])
"""

import json
import os
from dataclasses import dataclass, field

# HF repos with the exact device weights (canonical values for the sim)
MODELS = {
    "8b-q3": {
        "hf": "mlc-ai/Llama-3.1-8B-Instruct-q3f16_0-MLC",
        "local_fallback": "Llama-3.1-8B-Instruct-q3f16_1-MLC",  # same values, NK layout
    },
    "8b-q4": {
        # Quant-control point: is q3 the villain behind 8b-q3 < 3b-q4 (Phase-0 run 1)?
        "hf": "mlc-ai/Llama-3.1-8B-Instruct-q4f16_1-MLC",
        "local_fallback": None,
    },
    "3b-q4": {
        "hf": "mlc-ai/Llama-3.2-3B-Instruct-q4f16_0-MLC",
        "local_fallback": None,
    },
}

QUANT = {
    "q3f16_0": dict(bits=3, elem_per_storage=10, group_size=40, max_int=3, layout="KN"),
    "q3f16_1": dict(bits=3, elem_per_storage=10, group_size=40, max_int=3, layout="NK"),
    "q4f16_0": dict(bits=4, elem_per_storage=8, group_size=32, max_int=7, layout="KN"),
    "q4f16_1": dict(bits=4, elem_per_storage=8, group_size=32, max_int=7, layout="NK"),
}

# Sparsifiable matmul input classes (mirrors the planned device kernels).
# 'gate_select' is the residency selection rule |silu(gate)| > tau (CATS-style), not a matmul input.
CLASSES = ("qkv", "o", "gateup", "down")

# Byte weights per class for "model-wide sparsity" accounting (fraction of per-token
# sparsifiable weight reads; computed per model at runtime from actual dims).


@dataclass
class ModelConfig:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    tie_word_embeddings: bool
    rope_theta: float
    rope_scaling: dict | None
    quantization: str
    conv_template: str = ""
    extras: dict = field(default_factory=dict)


def load_model_config(model_dir: str) -> ModelConfig:
    with open(os.path.join(model_dir, "mlc-chat-config.json"), encoding="utf-8") as f:
        cfg = json.load(f)
    mc = cfg["model_config"]
    return ModelConfig(
        hidden_size=mc["hidden_size"],
        intermediate_size=mc["intermediate_size"],
        num_hidden_layers=mc["num_hidden_layers"],
        num_attention_heads=mc["num_attention_heads"],
        num_key_value_heads=mc["num_key_value_heads"],
        head_dim=mc.get("head_dim", mc["hidden_size"] // mc["num_attention_heads"]),
        rms_norm_eps=mc["rms_norm_eps"],
        vocab_size=mc["vocab_size"],
        tie_word_embeddings=mc.get("tie_word_embeddings", False),
        rope_theta=mc.get("position_embedding_base", 10000.0),
        rope_scaling=mc.get("rope_scaling"),
        quantization=cfg["quantization"],
        conv_template=(cfg.get("conv_template") or {}).get("name", "")
        if isinstance(cfg.get("conv_template"), dict)
        else str(cfg.get("conv_template", "")),
        extras=mc,
    )


def resolve_model_dir(model_key: str, models_root: str) -> str:
    """Find the model directory: <models_root>/<basename(hf)> or the repo-root local fallback."""
    entry = MODELS[model_key]
    cand = os.path.join(models_root, entry["hf"].split("/")[-1])
    if os.path.isfile(os.path.join(cand, "ndarray-cache.json")):
        return cand
    if entry["local_fallback"]:
        # repo root (…/research/sparse8b/phase0 -> repo root is 3 up)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        cand2 = os.path.join(root, entry["local_fallback"])
        if os.path.isfile(os.path.join(cand2, "ndarray-cache.json")):
            return cand2
    raise FileNotFoundError(
        f"Model '{model_key}' not found under {models_root}. "
        f"Download with: python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('{entry['hf']}', local_dir=r'{cand}')\""
    )


def sparsifiable_byte_weights(cfg: ModelConfig) -> dict:
    """Per-class share of sparsifiable weight bytes (per layer; uniform across layers)."""
    h, i = cfg.hidden_size, cfg.intermediate_size
    kv = cfg.num_key_value_heads * cfg.head_dim
    q = cfg.num_attention_heads * cfg.head_dim
    sizes = {
        "qkv": h * (q + 2 * kv),
        "o": q * h,
        "gateup": h * 2 * i,
        "down": i * h,
    }
    tot = sum(sizes.values())
    return {k: v / tot for k, v in sizes.items()}


LLAMA3_CHAT_TEMPLATE_HEADER = "<|begin_of_text|>"


def llama3_chat_format(messages: list) -> str:
    """Llama-3.x instruct chat template (text form, tokenized with the model tokenizer)."""
    out = [LLAMA3_CHAT_TEMPLATE_HEADER]
    for m in messages:
        out.append(f"<|start_header_id|>{m['role']}<|end_header_id|>\n\n{m['content']}<|eot_id|>")
    return "".join(out)
