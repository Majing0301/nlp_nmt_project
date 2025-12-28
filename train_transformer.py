# train_transformer.py
# -*- coding: utf-8 -*-
"""
Train Transformer-based NMT (CN -> EN) from scratch.

You said you put train scripts in the project root (not under src/), so this file
assumes you run it like:

  python train_transformer.py --data_dir data/processed --train_file train_10k.jsonl --valid_file valid.jsonl

It will still import your existing pipeline:
  from src.data import build_vocab_from_jsonl, create_dataloaders
  from src.models.transformer_nmt import TransformerNMT, TransformerConfig

What this trainer supports (per homework):
- From scratch training: encoder-decoder Transformer
- Ablations:
    --pos_type absolute / relative
    --norm_type layernorm / rmsnorm
- Hyperparameter sensitivity:
    --batch_size, --lr, --d_model, --n_layers, --n_heads, --d_ff, etc.
- Decode sanity check each epoch:
    --decode greedy / beam
- Logging to csv:
    logs/transformer_*.csv

Notes:
- Use valid loss for model selection; do NOT tune on test set.
- You can train on train_10k for debugging, then train_100k for final results.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from src.data import build_vocab_from_jsonl, create_dataloaders, Vocab
from src.models.transformer_nmt import TransformerNMT, TransformerConfig


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


# -----------------------------
# Decode helpers
# -----------------------------
@torch.no_grad()
def decode_examples(
    model: TransformerNMT,
    batch: Dict[str, torch.Tensor],
    zh_vocab: Vocab,
    en_vocab: Vocab,
    n: int = 2,
    max_len: int = 60,
    decode: str = "greedy",
    beam_size: int = 4,
):
    """
    Print:
      SRC (zh tokens)
      PRED (en tokens)
      REF  (en tokens)
    """
    model.eval()

    if decode == "beam":
        out_ids = model.beam_search_decode(batch["src"], batch["src_len"], beam_size=beam_size, max_len=max_len)
    else:
        out_ids = model.greedy_decode(batch["src"], batch["src_len"], max_len=max_len)

    B = out_ids.size(0)
    n = min(n, B)

    for i in range(n):
        # show source too, to avoid "which sentence is this?" confusion
        src_zh = " ".join(zh_vocab.decode(batch["src"][i].tolist(), stop_at_eos=True))
        pred = " ".join(en_vocab.decode(out_ids[i].tolist(), stop_at_eos=True))
        ref = " ".join(en_vocab.decode(batch["tgt"][i].tolist(), stop_at_eos=True))

        print(f"[Example {i}]")
        print("  SRC :", src_zh)
        print("  PRED:", pred)
        print("  REF :", ref)


# -----------------------------
# Loss / Eval
# -----------------------------
def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits: [B, T-1, V]
    tgt:    [B, T]     (contains BOS ... EOS)
    Compare with tgt_out = tgt[:, 1:].
    """
    tgt_out = tgt[:, 1:].contiguous()  # [B, T-1]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=pad_id,
    )


@torch.no_grad()
def evaluate_loss(model: TransformerNMT, loader, device: torch.device, pad_id: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch["src"], batch["src_len"], batch["tgt"])
        loss = compute_loss(out["logits"], batch["tgt"], pad_id)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(
    model: TransformerNMT,
    train_loader,
    optimizer,
    device: torch.device,
    pad_id: int,
    grad_clip: float = 1.0,
    log_every: int = 50,
    use_amp: bool = True,
):
    model.train()
    total_loss = 0.0
    steps = 0

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    for step, batch in enumerate(train_loader, start=1):
        batch = to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        if scaler.is_enabled():
            with torch.cuda.amp.autocast():
                out = model(batch["src"], batch["src_len"], batch["tgt"])
                loss = compute_loss(out["logits"], batch["tgt"], pad_id)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(batch["src"], batch["src_len"], batch["tgt"])
            loss = compute_loss(out["logits"], batch["tgt"], pad_id)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += float(loss.detach().item())
        steps += 1

        if step % log_every == 0:
            avg = total_loss / steps
            print(f"  step {step:5d} | loss {float(loss.detach().item()):.4f} | avg {avg:.4f}")

    return total_loss / max(1, steps)


# -----------------------------
# Checkpoint & Logging
# -----------------------------
def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_checkpoint(model: TransformerNMT, vocab: Dict[str, Vocab], cfg: TransformerConfig, path: str):
    ensure_dir(os.path.dirname(path))
    obj = {
        "model_state": model.state_dict(),
        "zh_itos": vocab["zh"].itos,
        "en_itos": vocab["en"].itos,
        "cfg": cfg.__dict__,
        "meta": {
            "src_pad_id": model.src_pad_id,
            "tgt_pad_id": model.tgt_pad_id,
            "bos_id": model.bos_id,
            "eos_id": model.eos_id,
        },
    }
    torch.save(obj, path)


def append_csv(log_path: str, row: Dict[str, object], header: Optional[list] = None):
    ensure_dir(os.path.dirname(log_path))
    file_exists = os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header or list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# -----------------------------
# Main
# -----------------------------
def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--train_file", type=str, default="train_100k.jsonl")
    p.add_argument("--valid_file", type=str, default="valid.jsonl")
    p.add_argument("--test_file", type=str, default="test.jsonl")
    p.add_argument("--max_len", type=int, default=60)

    # vocab
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_vocab", type=int, default=30000)

    # model ablations
    p.add_argument("--pos_type", type=str, default="absolute", choices=["absolute", "relative"])
    p.add_argument("--norm_type", type=str, default="layernorm", choices=["layernorm", "rmsnorm"])

    # model scale / hyperparams
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=50)

    # decoding sanity check
    p.add_argument("--decode", type=str, default="greedy", choices=["greedy", "beam"])
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--decode_n", type=int, default=2)

    # misc
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--no_amp", action="store_true")

    # save / log
    p.add_argument("--save_dir", type=str, default="checkpoints/transformer")
    p.add_argument("--save_name", type=str, default="")
    p.add_argument("--no_save", action="store_true")
    p.add_argument("--log_csv", type=str, default="logs/transformer_runs.csv")

    args = p.parse_args()

    set_seed(args.seed)

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    print(f"Data: {args.data_dir} | Train: {args.train_file} | Valid: {args.valid_file}")
    print(f"Ablation: pos={args.pos_type} | norm={args.norm_type}")
    print(f"Scale: d_model={args.d_model} heads={args.n_heads} layers={args.n_layers} d_ff={args.d_ff}")
    print(f"Train: epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.weight_decay}")

    # build vocab from train
    train_path = os.path.join(args.data_dir, args.train_file)
    vocab = build_vocab_from_jsonl(
        train_path=train_path,
        min_freq=args.min_freq,
        max_size=args.max_vocab,
        max_len=args.max_len,
    )
    print("Vocab sizes:", len(vocab["zh"]), len(vocab["en"]))

    # dataloaders
    train_loader, valid_loader, _ = create_dataloaders(
        data_dir=args.data_dir,
        vocab=vocab,
        train_file=args.train_file,
        valid_file=args.valid_file,
        test_file=args.test_file,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=0,
    )

    # model config
    cfg = TransformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pos_type=args.pos_type,
        norm_type=args.norm_type,
        max_len=max(256, args.max_len + 10),
    )

    model = TransformerNMT(
        src_vocab_size=len(vocab["zh"]),
        tgt_vocab_size=len(vocab["en"]),
        src_pad_id=vocab["zh"].pad_id,
        tgt_pad_id=vocab["en"].pad_id,
        bos_id=vocab["en"].bos_id,
        eos_id=vocab["en"].eos_id,
        cfg=cfg,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ckpt name
    if not args.save_name:
        args.save_name = (
            f"tfm_pos-{args.pos_type}_norm-{args.norm_type}_d{args.d_model}_h{args.n_heads}_L{args.n_layers}"
            f"_bs{args.batch_size}_lr{args.lr}_len{args.max_len}.pt"
        )
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    # sanity batch
    first_batch = next(iter(valid_loader))
    first_batch = to_device(first_batch, device)

    # train loop
    best_valid = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_id=vocab["en"].pad_id,
            grad_clip=args.grad_clip,
            log_every=args.log_every,
            use_amp=(not args.no_amp),
        )

        valid_loss = evaluate_loss(model, valid_loader, device, vocab["en"].pad_id)
        dt = time.time() - t0

        print(f"Epoch {epoch} done | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | time={dt:.1f}s")

        print("Validation examples:")
        decode_examples(
            model=model,
            batch=first_batch,
            zh_vocab=vocab["zh"],
            en_vocab=vocab["en"],
            n=args.decode_n,
            max_len=args.max_len,
            decode=args.decode,
            beam_size=args.beam_size,
        )

        # log csv
        append_csv(
            args.log_csv,
            row={
                "epoch": epoch,
                "train_file": args.train_file,
                "pos_type": args.pos_type,
                "norm_type": args.norm_type,
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "d_ff": args.d_ff,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "max_len": args.max_len,
                "train_loss": round(train_loss, 6),
                "valid_loss": round(valid_loss, 6),
                "time_sec": round(dt, 2),
                "device": device.type,
            },
            header=[
                "epoch", "train_file",
                "pos_type", "norm_type",
                "d_model", "n_heads", "n_layers", "d_ff", "dropout",
                "batch_size", "lr", "weight_decay", "max_len",
                "train_loss", "valid_loss", "time_sec", "device"
            ],
        )

        # save best
        if (not args.no_save) and (valid_loss < best_valid):
            best_valid = valid_loss
            save_checkpoint(model, vocab, cfg, ckpt_path)
            print(f"✅ Saved best: {ckpt_path} (best_valid={best_valid:.4f})")

    print("\nTraining finished.")
    print("Best valid loss:", best_valid)
    if not args.no_save:
        print("Best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
