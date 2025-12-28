# test_translate.py
# -*- coding: utf-8 -*-
"""
Evaluate / translate a CN->EN NMT checkpoint on a jsonl file (valid/test/custom).

Adds:
- handle ref from both "en" and "en_tok"
- compute BLEU (sacrebleu preferred; fallback to evaluate)
- print a few qualitative examples (SRC/REF/PRED)

Output jsonl:
  {"index":..., "zh":..., "pred":..., "ref":...}
"""

from __future__ import annotations
import argparse
import json
import os
from typing import Dict, Any, Iterable, List, Tuple, Optional

import torch

from src.data import Vocab, tokenize_zh


def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_vocab_from_ckpt(ckpt: Dict[str, Any]) -> Tuple[Vocab, Vocab]:
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


def build_src_tensor(
    obj: Dict[str, Any],
    zh_vocab: Vocab,
    max_len: int = 60,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
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


def get_ref_text(obj: Dict[str, Any]) -> str:
    """
    Prefer raw 'en' if exists; else use 'en_tok' joined by space.
    """
    if "en" in obj and isinstance(obj["en"], str) and obj["en"].strip():
        return obj["en"].strip()
    if "en_tok" in obj and isinstance(obj["en_tok"], list):
        return " ".join([str(t) for t in obj["en_tok"]]).strip()
    return ""


def decode_ids_to_text(en_vocab: Vocab, ids: List[int]) -> str:
    toks = en_vocab.decode(ids, stop_at_eos=True)
    if toks and toks[0] == "<bos>":
        toks = toks[1:]
    return " ".join(toks).strip()


def load_model(
    arch: str,
    ckpt: Dict[str, Any],
    zh_vocab: Vocab,
    en_vocab: Vocab,
    device: torch.device,
    tf_cfg_overrides: Optional[Dict[str, Any]] = None,
):
    arch = arch.lower()

    if arch == "rnn":
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
        return model

    if arch == "transformer":
        from src.models.transformer_nmt import TransformerNMT, TransformerConfig

        cfg_dict = ckpt.get("cfg", None)
        if cfg_dict is not None:
            cfg = TransformerConfig(**cfg_dict)
        else:
            cfg = TransformerConfig()

        if tf_cfg_overrides:
            for k, v in tf_cfg_overrides.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)

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
        return model

    raise ValueError(f"--arch must be rnn or transformer, got {arch}")


@torch.no_grad()
def translate_one(
    model,
    arch: str,
    src: torch.Tensor,
    src_len: torch.Tensor,
    en_vocab: Vocab,
    decode: str = "greedy",
    max_len: int = 60,
    beam_size: int = 4,
) -> str:
    arch = arch.lower()
    decode = decode.lower()

    if decode == "beam":
        out_ids = model.beam_search_decode(src, src_len, beam_size=beam_size, max_len=max_len)
    else:
        out_ids = model.greedy_decode(src, src_len, max_len=max_len)

    return decode_ids_to_text(en_vocab, out_ids[0].tolist())


def compute_bleu(preds, refs):
    import sacrebleu
    pairs = [(p, r) for p, r in zip(preds, refs) if isinstance(r, str) and r.strip()]
    if not pairs:
        return float("nan")
    preds_f = [p for p, _ in pairs]
    refs_f = [r for _, r in pairs]
    return float(sacrebleu.corpus_bleu(preds_f, [refs_f]).score)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", type=str, default="rnn", choices=["rnn", "transformer"])
    p.add_argument("--ckpt", type=str, required=True, help="path to checkpoint .pt/.pth")
    p.add_argument("--data", type=str, default="data/processed/test.jsonl", help="jsonl to translate")
    p.add_argument("--out", type=str, default="predictions.jsonl", help="output jsonl")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    p.add_argument("--max_len", type=int, default=60)
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=4)

    p.add_argument("--sentence", type=str, default="", help="translate one sentence instead of file (optional)")
    p.add_argument("--show_n", type=int, default=5, help="print N qualitative examples from the file")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    zh_vocab, en_vocab = load_vocab_from_ckpt(ckpt)

    model = load_model(
        arch=args.arch,
        ckpt=ckpt,
        zh_vocab=zh_vocab,
        en_vocab=en_vocab,
        device=device,
    )

    print("Device:", device)
    print("Arch  :", args.arch)
    print("Vocab :", len(zh_vocab), len(en_vocab))
    print("Decode:", args.decode, f"(beam_size={args.beam_size})" if args.decode == "beam" else "")
    print("MaxLen:", args.max_len)

    # translate one sentence
    if args.sentence.strip():
        obj = {"zh": args.sentence.strip(), "index": -1}
        src, src_len, zh_text = build_src_tensor(obj, zh_vocab, max_len=args.max_len)
        src, src_len = src.to(device), src_len.to(device)
        pred = translate_one(model, args.arch, src, src_len, en_vocab, decode=args.decode, max_len=args.max_len, beam_size=args.beam_size)
        print("\n[INPUT ]", zh_text)
        print("[PRED  ]", pred)
        return

    # translate a jsonl file
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    preds: List[str] = []
    refs: List[str] = []
    show_buf: List[Dict[str, str]] = []

    n = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        for obj in iter_jsonl(args.data):
            idx = obj.get("index", n)
            ref = get_ref_text(obj)

            src, src_len, zh_text = build_src_tensor(obj, zh_vocab, max_len=args.max_len)
            src, src_len = src.to(device), src_len.to(device)

            pred = translate_one(model, args.arch, src, src_len, en_vocab, decode=args.decode, max_len=args.max_len, beam_size=args.beam_size)

            out_obj = {"index": idx, "zh": zh_text, "pred": pred, "ref": ref}
            fo.write(json.dumps(out_obj, ensure_ascii=False) + "\n")

            preds.append(pred)
            refs.append(ref)

            if len(show_buf) < args.show_n:
                show_buf.append({"zh": zh_text, "ref": ref, "pred": pred})

            n += 1
            if n % 50 == 0:
                print(f"Translated {n} lines...")

    bleu = compute_bleu(preds, refs)

    print(f"\nDone. Wrote {n} lines to {args.out}")
    print(f"Corpus BLEU: {bleu:.2f}" if bleu == bleu else "Corpus BLEU: N/A (no references found?)")

    if args.show_n > 0:
        print("\nQualitative examples:")
        for i, ex in enumerate(show_buf, 1):
            print(f"\n[Example {i}]")
            print("  SRC :", ex["zh"])
            print("  REF :", ex["ref"])
            print("  PRED:", ex["pred"])


if __name__ == "__main__":
    main()
