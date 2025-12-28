# inference.py
# -*- coding: utf-8 -*-
"""
One-click inference script for CN->EN NMT.

Supports:
  - RNN Seq2Seq checkpoint (.pt) saved by train_rnn.py
  - Transformer checkpoint (.pt) saved by train_transformer.py
  - Fine-tuned T5 model folder (HuggingFace format)

Usage examples:
  # 1) Single sentence
  python inference.py --model_type transformer --model_path checkpoints/transformer/best.pt --sentence "但是 即使 是 官方 活动 也 带有 政治 色彩 。"

  # 2) Batch translate jsonl + compute BLEU if refs exist
  python inference.py --model_type rnn --model_path checkpoints/rnn/best.pt \
      --input data/processed/test.jsonl --output outputs/preds_rnn.jsonl --decode beam --beam_size 4 --bleu

  # 3) T5 (offline/local)
  python inference.py --model_type t5 --model_path checkpoints/t5 \
      --sentence "但是 即使 是 官方 活动 也 带有 政治 色彩 。"
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

# ----------------------------
# Helpers: jsonl
# ----------------------------
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def get_ref_text(obj: Dict[str, Any]) -> str:
    """Support ref from both raw and tokenized formats."""
    if "en" in obj and isinstance(obj["en"], str) and obj["en"].strip():
        return obj["en"].strip()
    if "en_tok" in obj and isinstance(obj["en_tok"], list):
        return " ".join([str(t) for t in obj["en_tok"]]).strip()
    return ""


# ----------------------------
# BLEU (optional)
# ----------------------------
def compute_bleu(preds: List[str], refs: List[str]) -> float:
    """sacrebleu corpus BLEU; returns NaN if no refs."""
    try:
        import sacrebleu
    except Exception:
        return float("nan")

    pairs = [(p, r) for p, r in zip(preds, refs) if isinstance(r, str) and r.strip()]
    if not pairs:
        return float("nan")
    preds_f = [p for p, _ in pairs]
    refs_f = [r for _, r in pairs]
    return float(sacrebleu.corpus_bleu(preds_f, [refs_f]).score)


# ============================================================
# RNN / Transformer (your from-scratch models)
# ============================================================
def load_vocab_from_ckpt(ckpt: Dict[str, Any]):
    # Reuse your Vocab class
    from src.data import Vocab

    zh_itos = ckpt["zh_itos"]
    en_itos = ckpt["en_itos"]
    zh_stoi = {t: i for i, t in enumerate(zh_itos)}
    en_stoi = {t: i for i, t in enumerate(en_itos)}

    meta = ckpt.get("meta", {})
    src_pad_id = int(meta.get("src_pad_id", zh_stoi.get("<pad>", 0)))
    tgt_pad_id = int(meta.get("tgt_pad_id", en_stoi.get("<pad>", 0)))
    bos_id = int(meta.get("bos_id", en_stoi.get("<bos>", 2)))
    eos_id = int(meta.get("eos_id", en_stoi.get("<eos>", 3)))

    zh_vocab = Vocab(
        stoi=zh_stoi, itos=zh_itos,
        pad_id=src_pad_id, unk_id=zh_stoi.get("<unk>", 1),
        bos_id=zh_stoi.get("<bos>", 2), eos_id=zh_stoi.get("<eos>", 3),
    )
    en_vocab = Vocab(
        stoi=en_stoi, itos=en_itos,
        pad_id=tgt_pad_id, unk_id=en_stoi.get("<unk>", 1),
        bos_id=bos_id, eos_id=eos_id,
    )
    return zh_vocab, en_vocab


def build_src_tensor(obj: Dict[str, Any], zh_vocab, max_len: int = 60):
    """Build src tensor from either zh_tok or raw zh."""
    from src.data import tokenize_zh

    if "zh_tok" in obj:
        zh_toks = obj["zh_tok"][:max_len]
        zh_text = obj.get("zh", " ".join(zh_toks))
    else:
        zh_text = obj.get("zh", "")
        zh_toks = tokenize_zh(zh_text)[:max_len]

    ids = [zh_vocab.bos_id] + [zh_vocab.stoi.get(t, zh_vocab.unk_id) for t in zh_toks] + [zh_vocab.eos_id]
    src = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    src_len = torch.tensor([len(ids)], dtype=torch.long)
    return src, src_len, zh_text


def decode_ids_to_text(en_vocab, ids: List[int]) -> str:
    toks = en_vocab.decode(ids, stop_at_eos=True)
    if toks and toks[0] == "<bos>":
        toks = toks[1:]
    return " ".join(toks).strip()


def load_rnn_or_transformer(model_type: str, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    zh_vocab, en_vocab = load_vocab_from_ckpt(ckpt)

    if model_type == "rnn":
        from src.models.rnn_seq2seq import Seq2SeqNMT
        attn_type = ckpt["meta"]["attn_type"]

        model = Seq2SeqNMT(
            src_vocab_size=len(zh_vocab),
            tgt_vocab_size=len(en_vocab),
            src_pad_id=zh_vocab.pad_id,
            tgt_pad_id=en_vocab.pad_id,
            bos_id=en_vocab.bos_id,
            eos_id=en_vocab.eos_id,
            emb_size=256,
            hidden_size=256,
            num_layers=2,
            dropout=0.1,
            attn_type=attn_type,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        return model, zh_vocab, en_vocab

    if model_type == "transformer":
        from src.models.transformer_nmt import TransformerNMT, TransformerConfig

        cfg_dict = ckpt.get("cfg", None)
        cfg = TransformerConfig(**cfg_dict) if cfg_dict is not None else TransformerConfig()

        model = TransformerNMT(
            src_vocab_size=len(zh_vocab),
            tgt_vocab_size=len(en_vocab),
            src_pad_id=zh_vocab.pad_id,
            tgt_pad_id=en_vocab.pad_id,
            bos_id=en_vocab.bos_id,
            eos_id=en_vocab.eos_id,
            cfg=cfg,
        ).to(device)
        model.load_state_dict(ckpt["model_state"], strict=True)
        model.eval()
        return model, zh_vocab, en_vocab

    raise ValueError("model_type must be rnn or transformer")


@torch.no_grad()
def translate_one_rnn_or_transformer(
    model,
    model_type: str,
    src: torch.Tensor,
    src_len: torch.Tensor,
    en_vocab,
    decode: str = "greedy",
    max_len: int = 60,
    beam_size: int = 4,
) -> str:
    decode = decode.lower()
    if decode == "beam":
        out_ids = model.beam_search_decode(src, src_len, beam_size=beam_size, max_len=max_len)
    else:
        out_ids = model.greedy_decode(src, src_len, max_len=max_len)
    return decode_ids_to_text(en_vocab, out_ids[0].tolist())


# ============================================================
# T5 (fine-tuned)
# ============================================================
def load_t5(model_dir: str, device: torch.device):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # For offline environments, user can export:
    # HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tok


@torch.no_grad()
def translate_one_t5(
    model,
    tok,
    zh_text: str,
    device: torch.device,
    decode: str = "greedy",
    max_len: int = 60,
    beam_size: int = 4,
) -> str:
    # T5 expects a task prefix
    inp = f"translate Chinese to English: {zh_text}"
    enc = tok(inp, return_tensors="pt", truncation=True, max_length=max_len).to(device)

    if decode == "beam":
        out = model.generate(
            **enc,
            max_length=max_len,
            num_beams=beam_size,
            early_stopping=True,
        )
    else:
        out = model.generate(
            **enc,
            max_length=max_len,
            num_beams=1,
            do_sample=False,
        )
    return tok.decode(out[0], skip_special_tokens=True).strip()


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_type", type=str, required=True, default="rnn", choices=["rnn", "transformer", "t5"])
    ap.add_argument("--model_path", type=str, required=True, default="checkpoints/rnn/rnn_attn-additive_tf-1.0_bs-128_100k.pt",
                    help="RNN/Transformer: checkpoint .pt | T5: model directory")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    ap.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    ap.add_argument("--beam_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=60)

    ap.add_argument("--sentence", type=str, default="", help="Translate one sentence")
    ap.add_argument("--input", type=str, default="", help="Input jsonl (batch translate)")
    ap.add_argument("--output", type=str, default="outputs/predictions.jsonl", help="Output jsonl")
    ap.add_argument("--bleu", action="store_true", help="Compute corpus BLEU if refs exist")
    ap.add_argument("--show_n", type=int, default=5, help="Print N examples (batch mode)")

    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print("Device    :", device)
    print("ModelType :", args.model_type)
    print("ModelPath :", args.model_path)
    print("Decode    :", args.decode, f"(beam={args.beam_size})" if args.decode == "beam" else "")
    print("MaxLen    :", args.max_len)

    # ------------------------
    # Single sentence mode
    # ------------------------
    if args.sentence.strip():
        zh_text = args.sentence.strip()

        if args.model_type in ("rnn", "transformer"):
            model, zh_vocab, en_vocab = load_rnn_or_transformer(args.model_type, args.model_path, device)
            obj = {"zh": zh_text, "index": -1}
            src, src_len, zh_text = build_src_tensor(obj, zh_vocab, max_len=args.max_len)
            src, src_len = src.to(device), src_len.to(device)
            pred = translate_one_rnn_or_transformer(
                model, args.model_type, src, src_len, en_vocab,
                decode=args.decode, max_len=args.max_len, beam_size=args.beam_size
            )
        else:
            model, tok = load_t5(args.model_path, device)
            pred = translate_one_t5(
                model, tok, zh_text, device,
                decode=args.decode, max_len=args.max_len, beam_size=args.beam_size
            )

        print("\n[SRC ]", zh_text)
        print("[PRED]", pred)
        return

    # ------------------------
    # Batch mode
    # ------------------------
    if not args.input:
        raise SystemExit("Please provide --sentence or --input")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    preds: List[str] = []
    refs: List[str] = []
    show_buf: List[Dict[str, str]] = []

    if args.model_type in ("rnn", "transformer"):
        model, zh_vocab, en_vocab = load_rnn_or_transformer(args.model_type, args.model_path, device)

        n = 0
        with open(args.output, "w", encoding="utf-8") as fo:
            for obj in iter_jsonl(args.input):
                idx = obj.get("index", n)
                ref = get_ref_text(obj)

                src, src_len, zh_text = build_src_tensor(obj, zh_vocab, max_len=args.max_len)
                src, src_len = src.to(device), src_len.to(device)

                pred = translate_one_rnn_or_transformer(
                    model, args.model_type, src, src_len, en_vocab,
                    decode=args.decode, max_len=args.max_len, beam_size=args.beam_size
                )

                out_obj = {"index": idx, "zh": zh_text, "pred": pred, "ref": ref}
                fo.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                preds.append(pred)
                refs.append(ref)
                if len(show_buf) < args.show_n:
                    show_buf.append({"zh": zh_text, "ref": ref, "pred": pred})

                n += 1
                if n % 50 == 0:
                    print(f"Translated {n} lines...")

    else:
        model, tok = load_t5(args.model_path, device)

        n = 0
        with open(args.output, "w", encoding="utf-8") as fo:
            for obj in iter_jsonl(args.input):
                idx = obj.get("index", n)
                zh_text = obj.get("zh", "")
                if not zh_text and "zh_tok" in obj:
                    zh_text = " ".join(obj["zh_tok"])

                ref = get_ref_text(obj)

                pred = translate_one_t5(
                    model, tok, zh_text, device,
                    decode=args.decode, max_len=args.max_len, beam_size=args.beam_size
                )

                out_obj = {"index": idx, "zh": zh_text, "pred": pred, "ref": ref}
                fo.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

                preds.append(pred)
                refs.append(ref)
                if len(show_buf) < args.show_n:
                    show_buf.append({"zh": zh_text, "ref": ref, "pred": pred})

                n += 1
                if n % 50 == 0:
                    print(f"Translated {n} lines...")

    print(f"\nDone. Wrote {len(preds)} lines to {args.output}")

    if args.bleu:
        bleu = compute_bleu(preds, refs)
        if bleu == bleu:
            print(f"Corpus BLEU: {bleu:.2f}")
        else:
            print("Corpus BLEU: N/A (no references found?)")

    if args.show_n > 0:
        print("\nQualitative examples:")
        for i, ex in enumerate(show_buf, 1):
            print(f"\n[Example {i}]")
            print("  SRC :", ex["zh"])
            print("  REF :", ex["ref"])
            print("  PRED:", ex["pred"])


if __name__ == "__main__":
    main()
