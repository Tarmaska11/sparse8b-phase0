"""Layer-streamed Llama-3.x forward pass with TEAL / gate-select sparsity hooks.

Design (fits 16 GB CPU runners):
  - hidden states for ALL chunks stay in RAM (fp32), weights loaded ONE LAYER at a time,
    dequantized on the fly from the MLC shards (bit-exact device values).
  - a "chunk" is a [B, T] batch of independent sequences (causal attention per row).
  - sparsity is applied to matmul INPUTS (TEAL) exactly where the device kernels will apply it:
        qkv   <- input_layernorm output
        o     <- attention output (pre-o_proj)
        gateup<- post_attention_layernorm output
        down  <- silu(gate) * up            (TEAL mode)
        down  <- gate-select: rows j with |silu(gate_j)| <= tau contribute 0 (residency mode;
                 'up' for unselected j is also skippable on device - same math)
  - hooks: calib(layer, cls, x_abs_flat_np) for histogram collection;
           ffn_mask(layer, mask_bool[B,T,I]) for firing traces (gate-select mask).

Everything computes in fp32. RoPE llama3-scaling params come from mlc-chat-config.json.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from common import ModelConfig
from q3loader import WeightSource


def _rope_inv_freq(cfg: ModelConfig) -> torch.Tensor:
    d = cfg.head_dim
    inv = 1.0 / (cfg.rope_theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
    rs = cfg.rope_scaling
    if rs and rs.get("rope_type") == "llama3":
        factor = rs["factor"]
        lo_f, hi_f = rs["low_freq_factor"], rs["high_freq_factor"]
        orig = rs["original_max_position_embeddings"]
        wavelen = 2 * math.pi / inv
        lo_wl, hi_wl = orig / lo_f, orig / hi_f
        smooth = (orig / wavelen - lo_f) / (hi_f - lo_f)
        scaled = torch.where(
            wavelen > lo_wl,
            inv / factor,
            torch.where(wavelen < hi_wl, inv, (1 - smooth) * inv / factor + smooth * inv),
        )
        return scaled
    return inv


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]; split-half (NeoX/llama) convention
    d2 = x.shape[-1] // 2
    x1, x2 = x[..., :d2], x[..., d2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


def _rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    v = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return v * w


def _sparsify(x: torch.Tensor, thr: float) -> torch.Tensor:
    if thr <= 0:
        return x
    return x * (x.abs() >= thr)


class Sparsity:
    """thresholds[layer_idx] = {'qkv': t, 'o': t, 'gateup': t, 'down': t, 'gate_select': tau}

    block_k > 1 = packing-granular sparsity: zero CONTIGUOUS blocks of block_k input dims
    by their block-max magnitude. This matches the on-device reality that q3 packs 10
    (q4: 8) consecutive K values per uint32 word - element-level skips save no bytes on
    stock MLC weights; only whole-word skips do. Thresholds for block mode must be
    calibrated on the block-max distribution (calibrate.py --block-k).
    """

    def __init__(self, thresholds: dict | None, mode: str = "off", block_k: int = 1):
        assert mode in ("off", "teal", "gate")
        self.mode = mode if thresholds else "off"
        self.t = thresholds or {}
        self.block_k = max(1, int(block_k))
        # achieved-sparsity accounting: {cls: [zeroed, total]}
        self.zeroed = {c: 0 for c in ("qkv", "o", "gateup", "down")}
        self.total = {c: 0 for c in ("qkv", "o", "gateup", "down")}

    def thr(self, layer: int, cls: str) -> float:
        return float(self.t.get(str(layer), self.t.get(layer, {})).get(cls, 0.0)) \
            if self.mode != "off" else 0.0

    def apply(self, x: torch.Tensor, layer: int, cls: str) -> torch.Tensor:
        if self.mode == "off":
            return x
        thr = self.thr(layer, cls)
        if thr <= 0:
            return x
        if self.block_k > 1:
            bk = self.block_k
            K = x.shape[-1]
            nb = K // bk
            head = x[..., : nb * bk].reshape(*x.shape[:-1], nb, bk)
            bmask = head.abs().amax(-1, keepdim=True) >= thr        # [.., nb, 1]
            mask = bmask.expand_as(head).reshape(*x.shape[:-1], nb * bk)
            if K > nb * bk:  # ragged tail always kept (device reads it anyway)
                mask = torch.cat([mask, torch.ones_like(x[..., nb * bk:], dtype=torch.bool)], -1)
        else:
            mask = x.abs() >= thr
        self.zeroed[cls] += int((~mask).sum())
        self.total[cls] += mask.numel()
        return x * mask

    def achieved(self) -> dict:
        out = {}
        for c in self.zeroed:
            out[c] = self.zeroed[c] / self.total[c] if self.total[c] else 0.0
        return out


def _q4_roundtrip_along_rows(w: torch.Tensor, group: int = 32) -> torch.Tensor:
    """Simulate the M3 bundle's down_proj representation: down is stored per-neuron as a column
    down_proj[:, j] (H values) requantized to q4 group-`group` ALONG H (bundles.py can't keep the
    native along-neuron q4 losslessly). w is [H, I]; requantize each column along axis 0 (H) at q4
    (max_int 7, symmetric group scale) and dequantize back. Measures the added quant error."""
    H, I = w.shape
    pad = (-H) % group
    if pad:
        w = torch.cat([w, torch.zeros(pad, I, dtype=w.dtype)], 0)
    Hp = w.shape[0]
    g = w.reshape(Hp // group, group, I)                       # [G, group, I]
    scale = g.abs().amax(1, keepdim=True) / 7.0                # [G,1,I]
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.clamp(torch.round(g / scale) + 7, 0, 14)
    deq = (q - 7) * scale
    return deq.reshape(Hp, I)[:H, :]


class LlamaSim:
    def __init__(self, model_dir: str, cfg: ModelConfig, threads: int = 0,
                 down_requant: bool = False):
        self.dir = model_dir
        self.cfg = cfg
        self.ws = WeightSource(model_dir, cfg.quantization)
        self.down_requant = down_requant  # M3: simulate bundle down-proj requant-along-H
        if threads:
            torch.set_num_threads(threads)
        self.inv_freq = _rope_inv_freq(cfg)
        self._cos_sin_cache = {}

    # ---------- pieces ----------

    def _cos_sin(self, t_len: int):
        cs = self._cos_sin_cache.get(t_len)
        if cs is None:
            pos = torch.arange(t_len, dtype=torch.float32)
            freqs = torch.outer(pos, self.inv_freq)  # [T, D/2]
            cs = (freqs.cos()[None, None], freqs.sin()[None, None])  # [1,1,T,D/2]
            self._cos_sin_cache = {t_len: cs}
        return cs

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        flat = tokens.reshape(-1)
        uniq, inv = torch.unique(flat, return_inverse=True)
        rows = self.ws.embedding_rows(uniq.numpy())[:, : self.cfg.hidden_size]  # [U, H]
        return rows[inv].reshape(*tokens.shape, self.cfg.hidden_size)

    def _load_layer(self, i: int) -> dict:
        cfg, p = self.cfg, f"model.layers.{i}"
        h = cfg.hidden_size
        return {
            "ln1": self.ws.norm(f"{p}.input_layernorm"),
            "ln2": self.ws.norm(f"{p}.post_attention_layernorm"),
            "qkv": self.ws.linear(f"{p}.self_attn.qkv_proj", h),        # [(Q+2KV), H]
            "o": self.ws.linear(f"{p}.self_attn.o_proj",
                                cfg.num_attention_heads * cfg.head_dim),  # [H, Q]
            "gate_up": self.ws.linear(f"{p}.mlp.gate_up_proj", h),      # [2I, H]
            "down": self._maybe_requant_down(
                self.ws.linear(f"{p}.mlp.down_proj", cfg.intermediate_size)),  # [H, I]
        }

    def _maybe_requant_down(self, w: torch.Tensor) -> torch.Tensor:
        return _q4_roundtrip_along_rows(w) if self.down_requant else w

    def layer_forward(self, hs: torch.Tensor, w: dict, layer: int,
                      sp: Sparsity, calib=None, ffn_mask_hook=None) -> torch.Tensor:
        cfg = self.cfg
        B, T, H = hs.shape
        nh, nkv, d = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        q_dim, kv_dim = nh * d, nkv * d

        # ---- attention ----
        x = _rms_norm(hs, w["ln1"], cfg.rms_norm_eps)
        if calib:
            calib(layer, "qkv", x)
        x = sp.apply(x, layer, "qkv")
        qkv = x @ w["qkv"].T                                     # [B,T,Q+2KV]
        q = qkv[..., :q_dim].view(B, T, nh, d).transpose(1, 2)   # [B,nh,T,d]
        k = qkv[..., q_dim:q_dim + kv_dim].view(B, T, nkv, d).transpose(1, 2)
        v = qkv[..., q_dim + kv_dim:].view(B, T, nkv, d).transpose(1, 2)
        cos, sin = self._cos_sin(T)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        if nkv != nh:
            rep = nh // nkv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B,nh,T,d]
        attn = attn.transpose(1, 2).reshape(B, T, q_dim)
        if calib:
            calib(layer, "o", attn)
        attn = sp.apply(attn, layer, "o")
        hs = hs + attn @ w["o"].T

        # ---- mlp ----
        x = _rms_norm(hs, w["ln2"], cfg.rms_norm_eps)
        if calib:
            calib(layer, "gateup", x)
        x = sp.apply(x, layer, "gateup")
        gu = x @ w["gate_up"].T                                  # [B,T,2I]
        I = cfg.intermediate_size
        g, u = gu[..., :I], gu[..., I:]
        act = F.silu(g)
        if calib:
            calib(layer, "gate_select", act)
        if sp.mode == "gate":
            tau = sp.thr(layer, "gate_select")
            sel = act.abs() > tau
            if ffn_mask_hook is not None:
                ffn_mask_hook(layer, sel)
            sp.zeroed["down"] += int((~sel).sum())
            sp.total["down"] += sel.numel()
            mid = torch.where(sel, act * u, torch.zeros((), dtype=act.dtype))
        else:
            mid = act * u
            if calib:
                calib(layer, "down", mid)
            if ffn_mask_hook is not None and sp.mode != "off":
                thr = sp.thr(layer, "down")
                ffn_mask_hook(layer, mid.abs() >= thr)
            mid = sp.apply(mid, layer, "down")
        hs = hs + mid @ w["down"].T
        return hs

    # ---------- drivers ----------

    def stream_forward(self, chunks: list, sp: Sparsity, calib=None,
                       ffn_mask_hook=None, progress=None) -> list:
        """chunks: list of [B,T] LongTensors -> list of FINAL-NORMED [B,T,H] fp32."""
        states = [self.embed(c) for c in chunks]
        for li in range(self.cfg.num_hidden_layers):
            w = self._load_layer(li)
            for ci, hs in enumerate(states):
                states[ci] = self.layer_forward(hs, w, li, sp, calib, ffn_mask_hook)
            del w
            if progress:
                progress(li)
        fw = self.ws.norm("model.norm")
        return [_rms_norm(s, fw, self.cfg.rms_norm_eps) for s in states]

    def _head_rows(self, n_start: int, n_end: int) -> torch.Tensor:
        """lm_head rows [n_start:n_end] -> [rows, H] fp32 (tied models use the embedding)."""
        if self.cfg.tie_word_embeddings:
            ids = np.arange(n_start, n_end)
            return self.ws.embedding_rows(ids)[:, : self.cfg.hidden_size]
        return self.ws.linear_rows("lm_head", self.cfg.hidden_size, n_start, n_end)

    def score_ce(self, final_states: list, chunks: list,
                 vocab_block: int = 16384, row_block: int = 4096):
        """Sum next-token cross-entropy (teacher forcing), streaming the lm_head in vocab
        blocks (never materializes the [V, H] table). Returns (total_nll, total_tokens)."""
        H, V = self.cfg.hidden_size, self.cfg.vocab_size
        flat = torch.cat([hs[:, :-1].reshape(-1, H) for hs in final_states], 0)
        tgt = torch.cat([toks[:, 1:].reshape(-1) for toks in chunks], 0)
        R = flat.shape[0]
        lse = torch.full((R,), float("-inf"))
        tgt_logit = torch.empty(R)
        for vs in range(0, V, vocab_block):
            ve = min(vs + vocab_block, V)
            blk = self._head_rows(vs, ve)              # [B, H]
            for rs in range(0, R, row_block):
                re_ = min(rs + row_block, R)
                logits = flat[rs:re_] @ blk.T          # [r, B]
                lse[rs:re_] = torch.logaddexp(lse[rs:re_], torch.logsumexp(logits, -1))
                t = tgt[rs:re_]
                sel = (t >= vs) & (t < ve)
                if sel.any():
                    tgt_logit[rs:re_][sel] = logits[sel, t[sel] - vs]
            del blk
        return float((lse - tgt_logit).sum()), R

    def logits_at(self, final_states: list, positions: list, token_sets: list):
        """For bench: positions[i] = list of (b, t) per chunk; token_sets = candidate token ids.
        Returns per-chunk list of [n_pos, n_candidates] logits. Dequantizes ONLY candidate rows."""
        out = []
        for hs, pos, cands in zip(final_states, positions, token_sets):
            if not pos:
                out.append(None)
                continue
            idx_b = torch.tensor([p[0] for p in pos])
            idx_t = torch.tensor([p[1] for p in pos])
            h = hs[idx_b, idx_t]                       # [n, H]
            if self.cfg.tie_word_embeddings:
                sub = self.ws.embedding_rows(np.array(cands))[:, : self.cfg.hidden_size]
            else:
                rows = [self.ws.linear_rows("lm_head", self.cfg.hidden_size, c, c + 1)
                        for c in cands]
                sub = torch.cat(rows, 0)
            out.append(h @ sub.T)
        return out
