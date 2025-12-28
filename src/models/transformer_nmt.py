# src/models/transformer_nmt.py
# -*- coding: utf-8 -*-
"""
Transformer-based NMT (CN -> EN) from scratch.

Features (for your homework requirements):
- Encoder-Decoder Transformer
- Position embedding schemes:
    * absolute: sinusoidal positional encoding (default)
    * relative: T5-style relative position bias (added to attention scores)
- Normalization methods:
    * LayerNorm
    * RMSNorm
- Decoding:
    * greedy_decode
    * beam_search_decode (optional but useful for the decoding comparison)

Expected batch format (matches your data.py):
    src: LongTensor [B, S]
    src_len: LongTensor [B]
    tgt: LongTensor [B, T]  (contains <bos> ... <eos>)

Forward returns logits for next-token prediction:
    logits: FloatTensor [B, T-1, Vtgt]
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# Normalization: LayerNorm vs RMSNorm
# =========================================================
class RMSNorm(nn.Module):
    """
    RMSNorm (no mean subtraction), used in some modern Transformers.
    """
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., d_model]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x_norm = x / rms
        return x_norm * self.weight


def make_norm(norm_type: str, d_model: int, eps: float = 1e-5) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type in ("layernorm", "ln"):
        return nn.LayerNorm(d_model, eps=eps)
    if norm_type in ("rmsnorm", "rms"):
        return RMSNorm(d_model, eps=eps)
    raise ValueError(f"Unknown norm_type={norm_type}. Use layernorm or rmsnorm.")


# =========================================================
# Absolute position encoding (sinusoidal)
# =========================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic sinusoidal positional encoding:
        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        """
        L = x.size(1)
        x = x + self.pe[:, :L, :]
        return self.dropout(x)


# =========================================================
# Relative position bias (T5-style)
# =========================================================
class RelativePositionBias(nn.Module):
    """
    T5-style relative position bias added to attention logits.

    We create a bucket for relative distances (j - i), then lookup an embedding:
        bias: [num_heads, Q, K]
    and add it to attention scores:
        scores = (QK^T)/sqrt(d) + bias

    This is a light, homework-friendly "relative" scheme.
    """
    def __init__(
        self,
        num_heads: int,
        num_buckets: int = 32,
        max_distance: int = 128,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)

    @staticmethod
    def _relative_position_bucket(relative_position: torch.Tensor, num_buckets: int, max_distance: int) -> torch.Tensor:
        """
        relative_position: [Q, K] where values are (k_idx - q_idx)
        Return: bucket indices in [0, num_buckets)
        """
        # half buckets for negative, half for positive
        n = -relative_position
        # n < 0 => relative_position > 0 (key after query)
        # We follow T5: separate buckets for negative and positive distances
        ret = torch.zeros_like(n)
        num_buckets //= 2
        is_neg = n < 0
        n = torch.abs(n)

        # now n is absolute distance
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # small: use exact buckets
        val_if_small = n

        # large: log buckets
        # avoid log(0)
        n_clipped = torch.clamp(n, min=1)
        val_if_large = max_exact + (
            (torch.log(n_clipped.float() / max_exact) / math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        bucket = torch.where(is_small, val_if_small, val_if_large)
        bucket = torch.where(is_neg, bucket + num_buckets, bucket)
        return bucket

    def forward(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        """
        Return bias: [1, num_heads, q_len, k_len]
        """
        q_pos = torch.arange(q_len, device=device)[:, None]
        k_pos = torch.arange(k_len, device=device)[None, :]
        rel = k_pos - q_pos  # [q_len, k_len]
        buckets = self._relative_position_bucket(rel, self.num_buckets, self.max_distance)  # [q_len, k_len]
        bias = self.relative_attention_bias(buckets)  # [q_len, k_len, num_heads]
        bias = bias.permute(2, 0, 1).unsqueeze(0)     # [1, num_heads, q_len, k_len]
        return bias


# =========================================================
# Masks
# =========================================================
def make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    lengths: [B]
    Return mask: [B, 1, 1, L] with True for positions to mask (PAD)
    """
    # positions >= length are padding
    B = lengths.size(0)
    range_row = torch.arange(max_len, device=lengths.device).unsqueeze(0).expand(B, max_len)
    pad = range_row >= lengths.unsqueeze(1)  # [B, L], True where PAD
    return pad.unsqueeze(1).unsqueeze(2)     # [B,1,1,L]


def make_causal_mask(L: int, device: torch.device) -> torch.Tensor:
    """
    Causal mask for decoder self-attn.
    Return [1, 1, L, L] True where future positions are masked.
    """
    m = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


# =========================================================
# Multi-Head Attention
# =========================================================
class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_relative_bias: bool = False,
        rel_num_buckets: int = 32,
        rel_max_distance: int = 128,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.use_relative_bias = use_relative_bias
        self.rel_bias = None
        if use_relative_bias:
            self.rel_bias = RelativePositionBias(
                num_heads=num_heads,
                num_buckets=rel_num_buckets,
                max_distance=rel_max_distance,
            )

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model] -> [B, heads, L, head_dim]
        B, L, _ = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,  # True = mask out
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q,k,v: [B, L, d_model]
        attn_mask: broadcastable to [B, heads, Q, K] with True positions masked.
        Return:
          out: [B, Q, d_model]
          attn: [B, heads, Q, K]
        """
        B, QL, _ = q.size()
        _, KL, _ = k.size()

        qh = self._shape(self.q_proj(q))
        kh = self._shape(self.k_proj(k))
        vh = self._shape(self.v_proj(v))

        # scores: [B, heads, Q, K]
        scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale

        # relative bias (added to scores)
        if self.use_relative_bias and self.rel_bias is not None:
            bias = self.rel_bias(q_len=QL, k_len=KL, device=q.device)  # [1, heads, Q, K]
            scores = scores + bias

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, vh)  # [B, heads, Q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, QL, self.d_model)
        out = self.o_proj(out)
        return out, attn


# =========================================================
# Feed Forward
# =========================================================
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "relu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation.lower()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            x = F.gelu(self.fc1(x))
        else:
            x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =========================================================
# Encoder / Decoder Layers
# =========================================================
class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str,
        use_relative_bias: bool,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_bias=use_relative_bias,
        )
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        # Pre-norm
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=src_mask)
        x = x + self.drop(attn_out)

        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.drop(ffn_out)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str,
        use_relative_bias: bool,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_bias=use_relative_bias,
        )
        self.cross_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_relative_bias=False,  # cross-attn usually doesn't use relative bias in simple setups
        )
        self.ffn = FeedForward(d_model, d_ff, dropout=dropout)

        self.norm1 = make_norm(norm_type, d_model)
        self.norm2 = make_norm(norm_type, d_model)
        self.norm3 = make_norm(norm_type, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        mem_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # masked self-attn
        h = self.norm1(x)
        attn_out, _ = self.self_attn(h, h, h, attn_mask=tgt_mask)
        x = x + self.drop(attn_out)

        # cross-attn
        h = self.norm2(x)
        attn_out, _ = self.cross_attn(h, memory, memory, attn_mask=mem_mask)
        x = x + self.drop(attn_out)

        # ffn
        h = self.norm3(x)
        ffn_out = self.ffn(h)
        x = x + self.drop(ffn_out)
        return x


# =========================================================
# Encoder / Decoder stacks
# =========================================================
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str,
        use_relative_bias: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_relative_bias)
            for _ in range(n_layers)
        ])
        self.final_norm = make_norm(norm_type, d_model)

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor]) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, src_mask)
        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        norm_type: str,
        use_relative_bias: bool,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout, norm_type, use_relative_bias)
            for _ in range(n_layers)
        ])
        self.final_norm = make_norm(norm_type, d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor],
        mem_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        return self.final_norm(x)


# =========================================================
# Full Transformer NMT
# =========================================================
@dataclass
class TransformerConfig:
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1
    activation: str = "relu"

    pos_type: str = "absolute"   # absolute | relative
    norm_type: str = "layernorm" # layernorm | rmsnorm
    max_len: int = 2048


class TransformerNMT(nn.Module):
    """
    CN -> EN Transformer Encoder-Decoder.
    """
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        bos_id: int,
        eos_id: int,
        cfg: TransformerConfig,
    ):
        super().__init__()
        self.src_pad_id = int(src_pad_id)
        self.tgt_pad_id = int(tgt_pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        self.cfg = cfg
        self.d_model = cfg.d_model

        self.src_emb = nn.Embedding(src_vocab_size, cfg.d_model, padding_idx=self.src_pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, cfg.d_model, padding_idx=self.tgt_pad_id)

        # Absolute position encoding only (if pos_type == "absolute")
        self.pos_enc = None
        if cfg.pos_type.lower() == "absolute":
            self.pos_enc = SinusoidalPositionalEncoding(cfg.d_model, max_len=cfg.max_len, dropout=cfg.dropout)

        use_relative = cfg.pos_type.lower() == "relative"

        self.encoder = TransformerEncoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            num_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            use_relative_bias=use_relative,
        )
        self.decoder = TransformerDecoder(
            n_layers=cfg.n_layers,
            d_model=cfg.d_model,
            num_heads=cfg.n_heads,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            norm_type=cfg.norm_type,
            use_relative_bias=use_relative,
        )

        self.out_proj = nn.Linear(cfg.d_model, tgt_vocab_size)
        self.drop = nn.Dropout(cfg.dropout)

    def encode(self, src: torch.Tensor, src_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        src: [B,S], src_len: [B]
        returns:
          memory: [B,S,d]
          src_pad_mask: [B,1,1,S] True for PAD
        """
        B, S = src.size()
        src_pad_mask = make_padding_mask(src_len, max_len=S)  # [B,1,1,S]

        x = self.src_emb(src) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)

        memory = self.encoder(x, src_pad_mask)
        return memory, src_pad_mask

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        tgt_in: [B,Tin] (usually tgt[:, :-1])
        memory: [B,S,d]
        src_pad_mask: [B,1,1,S]
        returns:
          dec_out: [B,Tin,d]
        """
        B, T = tgt_in.size()

        # padding mask for tgt (for self-attn)
        # Here we build from tokens: pad positions are True
        tgt_pad = (tgt_in == self.tgt_pad_id)  # [B,T]
        tgt_pad_mask = tgt_pad.unsqueeze(1).unsqueeze(2)  # [B,1,1,T]
        causal = make_causal_mask(T, device=tgt_in.device)  # [1,1,T,T]
        tgt_mask = tgt_pad_mask | causal  # broadcast to [B,1,T,T] via masking in attention

        x = self.tgt_emb(tgt_in) * math.sqrt(self.d_model)
        if self.pos_enc is not None:
            x = self.pos_enc(x)
        x = self.drop(x)

        # mem_mask: when attending to encoder memory, mask PAD in src
        mem_mask = src_pad_mask  # [B,1,1,S]
        dec_out = self.decoder(x, memory, tgt_mask=tgt_mask, mem_mask=mem_mask)
        return dec_out

    def forward(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        tgt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Train-time forward (teacher forcing by construction):
          input tgt_in  = tgt[:, :-1]
          predict tgt_out = tgt[:, 1:]
        """
        memory, src_pad_mask = self.encode(src, src_len)

        tgt_in = tgt[:, :-1].contiguous()
        dec_out = self.decode(tgt_in, memory, src_pad_mask)
        logits = self.out_proj(dec_out)  # [B, T-1, V]
        return {"logits": logits}

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        max_len: int = 60,
    ) -> torch.Tensor:
        """
        Greedy decoding. Returns token ids: [B, L] (includes BOS ... EOS maybe)
        """
        device = src.device
        memory, src_pad_mask = self.encode(src, src_len)

        B = src.size(0)
        ys = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)

        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len):
            dec_out = self.decode(ys, memory, src_pad_mask)  # [B, t, d]
            logits = self.out_proj(dec_out[:, -1:, :])       # [B,1,V]
            next_id = torch.argmax(logits.squeeze(1), dim=-1) # [B]

            ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)

            finished = finished | (next_id == self.eos_id)
            if bool(finished.all()):
                break
        return ys

    @torch.no_grad()
    def beam_search_decode(
        self,
        src: torch.Tensor,
        src_len: torch.Tensor,
        beam_size: int = 4,
        max_len: int = 60,
        length_penalty: float = 0.6,
    ) -> torch.Tensor:
        """
        Simple beam search per batch item (loop over batch for clarity).
        Returns: [B, L] best sequence per item.
        """
        device = src.device
        B = src.size(0)
        results: List[torch.Tensor] = []

        memory, src_pad_mask = self.encode(src, src_len)

        for b in range(B):
            mem_b = memory[b:b+1]                 # [1,S,d]
            spm_b = src_pad_mask[b:b+1]           # [1,1,1,S]

            # beams: list of (seq, logprob, finished)
            beams = [(torch.tensor([[self.bos_id]], device=device, dtype=torch.long), 0.0, False)]

            for _ in range(max_len):
                new_beams = []
                for seq, score, fin in beams:
                    if fin:
                        new_beams.append((seq, score, True))
                        continue

                    dec_out = self.decode(seq, mem_b, spm_b)
                    logits = self.out_proj(dec_out[:, -1, :])  # [1,V]
                    logp = F.log_softmax(logits, dim=-1).squeeze(0)  # [V]

                    topk_logp, topk_ids = torch.topk(logp, k=beam_size)
                    for k in range(beam_size):
                        nid = int(topk_ids[k].item())
                        nscore = score + float(topk_logp[k].item())
                        nseq = torch.cat([seq, torch.tensor([[nid]], device=device)], dim=1)
                        nfin = (nid == self.eos_id)
                        new_beams.append((nseq, nscore, nfin))

                # keep best beams (with length penalty)
                def lp(length: int) -> float:
                    # GNMT style length penalty
                    return ((5.0 + length) / 6.0) ** length_penalty

                new_beams.sort(key=lambda x: x[1] / lp(x[0].size(1)), reverse=True)
                beams = new_beams[:beam_size]

                if all(fin for _, _, fin in beams):
                    break

            best = beams[0][0].squeeze(0)  # [L]
            results.append(best)

        # pad to same length
        maxL = max(r.numel() for r in results)
        out = torch.full((B, maxL), self.tgt_pad_id, dtype=torch.long, device=device)
        for i, r in enumerate(results):
            out[i, : r.numel()] = r
        return out
