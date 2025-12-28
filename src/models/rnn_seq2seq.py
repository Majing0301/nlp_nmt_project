# src/models/rnn_seq2seq.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Attention module
# -----------------------------
class Attention(nn.Module):
    """
    Attention alignment functions:
      - "dot":       score(h_t, h_s) = h_t · h_s
      - "general":   score(h_t, h_s) = h_t^T W h_s   (multiplicative / Luong general)
      - "additive":  score(h_t, h_s) = v^T tanh(Ws h_s + Wt h_t) (Bahdanau)
    Shapes:
      decoder_query: [B, H]          (decoder hidden at current step, top layer)
      encoder_out:   [B, L, H]       (encoder outputs for all timesteps)
      src_mask:      [B, L] bool     (True for valid tokens, False for pad)
    Returns:
      context: [B, H]
      attn_w:  [B, L]
    """

    def __init__(self, hidden_size: int, attn_type: str = "general"):
        super().__init__()
        assert attn_type in ("dot", "general", "additive")
        self.hidden_size = hidden_size
        self.attn_type = attn_type

        if attn_type == "general":
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attn_type == "additive":
            self.Ws = nn.Linear(hidden_size, hidden_size, bias=False)
            self.Wt = nn.Linear(hidden_size, hidden_size, bias=False)
            self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self,
        decoder_query: torch.Tensor,      # [B, H]
        encoder_out: torch.Tensor,        # [B, L, H]
        src_mask: Optional[torch.Tensor]  # [B, L] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B, L, H = encoder_out.shape
        assert decoder_query.shape == (B, H)

        if self.attn_type == "dot":
            # scores: [B, L] = sum over H of (enc * query)
            scores = torch.sum(encoder_out * decoder_query.unsqueeze(1), dim=-1)

        elif self.attn_type == "general":
            # scores = (W enc) · query
            enc_proj = self.W(encoder_out)  # [B, L, H]
            scores = torch.sum(enc_proj * decoder_query.unsqueeze(1), dim=-1)  # [B, L]

        else:  # additive (Bahdanau)
            # score = v^T tanh(Ws enc + Wt query)
            enc_part = self.Ws(encoder_out)                         # [B, L, H]
            dec_part = self.Wt(decoder_query).unsqueeze(1)          # [B, 1, H]
            e = torch.tanh(enc_part + dec_part)                     # [B, L, H]
            scores = self.v(e).squeeze(-1)                          # [B, L]

        # mask pad positions: set very negative so softmax ~ 0 there
        if src_mask is not None:
            scores = scores.masked_fill(~src_mask, -1e9)

        attn_w = F.softmax(scores, dim=-1)                          # [B, L]
        context = torch.bmm(attn_w.unsqueeze(1), encoder_out).squeeze(1)  # [B, H]
        return context, attn_w


# -----------------------------
# Encoder
# -----------------------------
class RNNEncoder(nn.Module):
    """
    2-layer unidirectional GRU encoder.
    Inputs:
      src_ids: [B, L]
      src_len: [B]
    Outputs:
      enc_out: [B, L, H]
      h_n:     [num_layers, B, H]
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, src_ids: torch.Tensor, src_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # src_ids: [B, L]
        emb = self.emb(src_ids)  # [B, L, E]

        # pack for speed & correctness with padding
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, src_len.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, h_n = self.gru(packed)  # packed_out: packed, h_n: [layers, B, H]
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)  # [B, L, H]
        return enc_out, h_n


# -----------------------------
# Decoder
# -----------------------------
class RNNDecoder(nn.Module):
    """
    2-layer unidirectional GRU decoder with attention.
    At each step:
      input token -> embedding
      GRU -> query (top hidden)
      attention(query, enc_out) -> context
      concat([query, context]) -> vocab logits
    """

    def __init__(
        self,
        vocab_size: int,
        emb_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_id: int = 0,
        attn_type: str = "general",
    ):
        super().__init__()
        self.pad_id = pad_id
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_id)
        self.gru = nn.GRU(
            input_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = Attention(hidden_size, attn_type=attn_type)

        # combine query + context -> logits
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size),
        )

    def forward_step(
        self,
        input_ids: torch.Tensor,          # [B] current input token ids
        prev_h: torch.Tensor,             # [layers, B, H]
        enc_out: torch.Tensor,            # [B, L, H]
        src_mask: Optional[torch.Tensor], # [B, L]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        One decoding step.
        Returns:
          logits: [B, V]
          h:      [layers, B, H]
          attn_w: [B, L]
        """
        emb = self.emb(input_ids).unsqueeze(1)  # [B, 1, E]
        out, h = self.gru(emb, prev_h)          # out: [B,1,H]
        query = out.squeeze(1)                  # [B,H] (top-layer output at this step)

        context, attn_w = self.attn(query, enc_out, src_mask)  # [B,H], [B,L]
        logits = self.out(torch.cat([query, context], dim=-1)) # [B,V]
        return logits, h, attn_w


# -----------------------------
# Seq2Seq wrapper (train + decode)
# -----------------------------
@dataclass
class BeamHyp:
    tokens: List[int]
    logprob: float
    h: torch.Tensor  # [layers, 1, H]
    ended: bool


class Seq2SeqNMT(nn.Module):
    """
    Full seq2seq model for CN->EN NMT:
      encoder: src (zh)
      decoder: tgt (en)
    Works with your batch dict from data.py:
      batch["src"]     [B,Ls]
      batch["src_len"] [B]
      batch["tgt"]     [B,Lt]
      batch["tgt_len"] [B]
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        src_pad_id: int,
        tgt_pad_id: int,
        bos_id: int,
        eos_id: int,
        emb_size: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        attn_type: str = "general",
    ):
        super().__init__()
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.attn_type = attn_type   # ✅ 必须加这一行

        self.encoder = RNNEncoder(
            vocab_size=src_vocab_size,
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=src_pad_id,
        )
        self.decoder = RNNDecoder(
            vocab_size=tgt_vocab_size,
            emb_size=emb_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            pad_id=tgt_pad_id,
            attn_type=attn_type,
        )

    def make_src_mask(self, src_ids: torch.Tensor) -> torch.Tensor:
        # True for non-pad
        return src_ids.ne(self.src_pad_id)

    def forward(
        self,
        src_ids: torch.Tensor,
        src_len: torch.Tensor,
        tgt_ids: torch.Tensor,
        teacher_forcing_ratio: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward.
        Inputs:
          src_ids: [B,Ls]
          src_len: [B]
          tgt_ids: [B,Lt]  (contains BOS ... EOS)
          teacher_forcing_ratio:
              1.0 -> always teacher forcing
              0.0 -> always free running
        Returns:
          logits: [B, Lt-1, V]  (predict next token for each step)
        """
        B, Lt = tgt_ids.shape
        enc_out, h_enc = self.encoder(src_ids, src_len)  # enc_out [B,Ls,H], h_enc [layers,B,H]
        src_mask = self.make_src_mask(src_ids)           # [B,Ls] bool

        # decoder initial hidden = encoder final hidden (common baseline)
        h = h_enc

        # decoder input starts with BOS
        input_ids = tgt_ids[:, 0]  # [B] should be BOS

        logits_steps: List[torch.Tensor] = []

        # we will predict tgt at positions 1..Lt-1
        for t in range(1, Lt):
            step_logits, h, _ = self.decoder.forward_step(input_ids, h, enc_out, src_mask)
            logits_steps.append(step_logits.unsqueeze(1))  # [B,1,V]

            # choose next input
            use_tf = (torch.rand(1, device=src_ids.device).item() < teacher_forcing_ratio)
            if use_tf:
                input_ids = tgt_ids[:, t]  # teacher forcing: ground-truth
            else:
                input_ids = torch.argmax(step_logits, dim=-1)  # free running: model prediction

        logits = torch.cat(logits_steps, dim=1)  # [B, Lt-1, V]
        return {"logits": logits}

    @torch.no_grad()
    def greedy_decode(
        self,
        src_ids: torch.Tensor,
        src_len: torch.Tensor,
        max_len: int = 80,
    ) -> torch.Tensor:
        """
        Greedy decoding.
        Returns:
          out_ids: [B, <=max_len] including BOS ... EOS (padded to max_len)
        """
        device = src_ids.device
        B = src_ids.size(0)

        enc_out, h = self.encoder(src_ids, src_len)
        src_mask = self.make_src_mask(src_ids)

        out = torch.full((B, max_len), fill_value=self.tgt_pad_id, dtype=torch.long, device=device)
        out[:, 0] = self.bos_id

        input_ids = out[:, 0]
        ended = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(1, max_len):
            step_logits, h, _ = self.decoder.forward_step(input_ids, h, enc_out, src_mask)
            next_ids = torch.argmax(step_logits, dim=-1)  # [B]
            out[:, t] = next_ids
            input_ids = next_ids

            ended = ended | next_ids.eq(self.eos_id)
            if ended.all():
                break

        return out

    @torch.no_grad()
    def beam_search_decode(
        self,
        src_ids: torch.Tensor,
        src_len: torch.Tensor,
        beam_size: int = 5,
        max_len: int = 80,
        length_penalty: float = 0.7,
    ) -> torch.Tensor:
        """
        Simple beam search (batch-wise by looping over samples).
        Returns:
          out_ids: [B, max_len] padded
        Note: This is a readable baseline implementation for homework.
        """
        device = src_ids.device
        B = src_ids.size(0)

        enc_out, h0 = self.encoder(src_ids, src_len)     # [B,L,H], [layers,B,H]
        src_mask = self.make_src_mask(src_ids)           # [B,L]

        results = torch.full((B, max_len), self.tgt_pad_id, dtype=torch.long, device=device)

        for b in range(B):
            enc_b = enc_out[b:b+1]                       # [1,L,H]
            mask_b = src_mask[b:b+1]                     # [1,L]
            h_b = h0[:, b:b+1, :].contiguous()           # [layers,1,H]

            beams: List[BeamHyp] = [BeamHyp(tokens=[self.bos_id], logprob=0.0, h=h_b, ended=False)]

            for t in range(1, max_len):
                all_cand: List[BeamHyp] = []

                for hyp in beams:
                    if hyp.ended:
                        all_cand.append(hyp)
                        continue

                    inp = torch.tensor([hyp.tokens[-1]], device=device, dtype=torch.long)
                    step_logits, h_new, _ = self.decoder.forward_step(inp, hyp.h, enc_b, mask_b)
                    logp = F.log_softmax(step_logits, dim=-1).squeeze(0)  # [V]

                    topk_logp, topk_ids = torch.topk(logp, k=beam_size)

                    for k in range(beam_size):
                        tok = int(topk_ids[k].item())
                        new_tokens = hyp.tokens + [tok]
                        new_logprob = hyp.logprob + float(topk_logp[k].item())
                        ended = (tok == self.eos_id)
                        all_cand.append(BeamHyp(tokens=new_tokens, logprob=new_logprob, h=h_new, ended=ended))

                # length penalty + keep top beams
                def score(h: BeamHyp) -> float:
                    # exclude BOS from length
                    length = max(1, len(h.tokens) - 1)
                    return h.logprob / (length ** length_penalty)

                all_cand.sort(key=score, reverse=True)
                beams = all_cand[:beam_size]

                if all(h.ended for h in beams):
                    break

            best = beams[0].tokens[:max_len]
            # pad to max_len
            for i, tok in enumerate(best):
                results[b, i] = tok

        return results
