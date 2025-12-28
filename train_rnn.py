# src/train_rnn.py
# -*- coding: utf-8 -*-
"""
Train RNN-based Seq2Seq NMT (CN -> EN).

Example:
  python -m src.train_rnn --data_dir data/processed --train_file train_10k.jsonl --valid_file valid.jsonl \
      --attn general --teacher_forcing 1.0 --epochs 5 --batch_size 64

Adds (aligned with train_transformer.py):
- epoch time + total time
- save best checkpoint (by valid_loss)
- csv logging (logs/rnn_runs.csv)

Compare:
  - Attention: --attn dot / general / additive
  - Training policy: --teacher_forcing 1.0 (teacher forcing) vs 0.0 (free running)
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
from torch.optim import Adam

from src.data import build_vocab_from_jsonl, create_dataloaders, Vocab
from src.models.rnn_seq2seq import Seq2SeqNMT


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
    model: Seq2SeqNMT,
    batch: Dict[str, torch.Tensor],
    en_vocab: Vocab,
    n: int = 2,
    max_len: int = 60,
):
    """
    Print a few greedy decoded samples from a batch.
    """
    model.eval()
    out_ids = model.greedy_decode(batch["src"], batch["src_len"], max_len=max_len)

    B = out_ids.size(0)
    n = min(n, B)
    for i in range(n):
        pred = " ".join(en_vocab.decode(out_ids[i].tolist(), stop_at_eos=True))
        ref = " ".join(en_vocab.decode(batch["tgt"][i].tolist(), stop_at_eos=True))
        print(f"[Example {i}]")
        print("  PRED:", pred)
        print("  REF :", ref)


# -----------------------------
# Loss / Eval
# -----------------------------
def compute_loss(logits: torch.Tensor, tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """
    logits: [B, T-1, V]
    tgt:    [B, T] (BOS...EOS)
    """
    tgt_out = tgt[:, 1:].contiguous()  # [B, T-1]
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        tgt_out.reshape(-1),
        ignore_index=pad_id,
    )


@torch.no_grad()
def evaluate_loss(model: Seq2SeqNMT, valid_loader, device: torch.device, en_pad_id: int) -> float:
    model.eval()
    total = 0.0
    n = 0
    for batch in valid_loader:
        batch = to_device(batch, device)
        out = model(batch["src"], batch["src_len"], batch["tgt"], teacher_forcing_ratio=1.0)  # eval with TF
        loss = compute_loss(out["logits"], batch["tgt"], en_pad_id)
        total += float(loss.item())
        n += 1
    return total / max(1, n)


# -----------------------------
# Training
# -----------------------------
def train_one_epoch(
    model: Seq2SeqNMT,
    train_loader,
    optimizer,
    device: torch.device,
    en_pad_id: int,
    teacher_forcing_ratio: float,
    log_every: int = 50,
    grad_clip: float = 1.0,
):
    model.train()
    total_loss = 0.0
    steps = 0

    for step, batch in enumerate(train_loader, start=1):
        batch = to_device(batch, device)

        out = model(
            batch["src"],
            batch["src_len"],
            batch["tgt"],
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        loss = compute_loss(out["logits"], batch["tgt"], en_pad_id)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
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


def save_checkpoint(model: Seq2SeqNMT, vocab: Dict[str, Vocab], path: str):
    ensure_dir(os.path.dirname(path))
    obj = {
        "model_state": model.state_dict(),
        "zh_itos": vocab["zh"].itos,
        "en_itos": vocab["en"].itos,
        "meta": {
            "src_pad_id": model.src_pad_id,
            "tgt_pad_id": model.tgt_pad_id,
            "bos_id": model.bos_id,
            "eos_id": model.eos_id,
            "attn_type": model.attn_type,  # ✅ keep for inference
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--train_file", type=str, default="train_100k.jsonl")
    parser.add_argument("--valid_file", type=str, default="valid.jsonl")
    parser.add_argument("--test_file", type=str, default="test.jsonl")

    parser.add_argument("--attn", type=str, default="general", choices=["dot", "general", "additive"])
    parser.add_argument("--teacher_forcing", type=float, default=1.0, help="1.0=TF, 0.0=Free Running")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=60)

    parser.add_argument("--emb_size", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # save / log (align with transformer)
    parser.add_argument("--save_dir", type=str, default="checkpoints/rnn")
    parser.add_argument("--save_name", type=str, default="")
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--log_csv", type=str, default="logs/rnn_runs.csv")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

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
    print(f"RNN:  attn={args.attn} | TF={args.teacher_forcing}")
    print(f"Train: epochs={args.epochs} bs={args.batch_size} lr={args.lr} max_len={args.max_len}")

    # build vocab from train
    train_path = os.path.join(args.data_dir, args.train_file)
    vocab = build_vocab_from_jsonl(
        train_path=train_path,
        min_freq=2,
        max_size=30000,
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

    # model
    model = Seq2SeqNMT(
        src_vocab_size=len(vocab["zh"]),
        tgt_vocab_size=len(vocab["en"]),
        src_pad_id=vocab["zh"].pad_id,
        tgt_pad_id=vocab["en"].pad_id,
        bos_id=vocab["en"].bos_id,
        eos_id=vocab["en"].eos_id,
        emb_size=args.emb_size,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=args.dropout,
        attn_type=args.attn,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    # checkpoint name
    if not args.save_name:
        args.save_name = (
            f"rnn_attn-{args.attn}_tf-{args.teacher_forcing}_emb{args.emb_size}_hid{args.hidden_size}"
            f"_bs{args.batch_size}_lr{args.lr}_len{args.max_len}.pt"
        )
    ckpt_path = os.path.join(args.save_dir, args.save_name)

    # sanity batch for qualitative samples
    first_batch = next(iter(valid_loader))
    first_batch = to_device(first_batch, device)

    # (optional) quick unk ratio debug
    b = next(iter(valid_loader))
    tgt_out = b["tgt"][:, 1:]
    unk_ratio = (tgt_out == vocab["en"].unk_id).float().mean().item()
    print(f"[DEBUG] valid unk_ratio={unk_ratio:.4f}")

    # train loop with time + save best
    best_valid = float("inf")
    total_t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            en_pad_id=vocab["en"].pad_id,
            teacher_forcing_ratio=args.teacher_forcing,
            log_every=args.log_every,
            grad_clip=args.grad_clip,
        )

        valid_loss = evaluate_loss(model, valid_loader, device, vocab["en"].pad_id)
        dt = time.time() - t0

        print(f"Epoch {epoch} done | train_loss={train_loss:.4f} | valid_loss={valid_loss:.4f} | time={dt:.1f}s")

        print("Validation examples (greedy):")
        decode_examples(model, first_batch, vocab["en"], n=2, max_len=args.max_len)

        # csv log (aligned with transformer style)
        append_csv(
            args.log_csv,
            row={
                "epoch": epoch,
                "train_file": args.train_file,
                "attn": args.attn,
                "teacher_forcing": args.teacher_forcing,
                "emb_size": args.emb_size,
                "hidden_size": args.hidden_size,
                "dropout": args.dropout,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "max_len": args.max_len,
                "train_loss": round(train_loss, 6),
                "valid_loss": round(valid_loss, 6),
                "time_sec": round(dt, 2),
                "device": device.type,
            },
            header=[
                "epoch", "train_file",
                "attn", "teacher_forcing",
                "emb_size", "hidden_size", "dropout",
                "batch_size", "lr", "max_len",
                "train_loss", "valid_loss",
                "time_sec", "device",
            ],
        )

        # save best (by valid loss)
        if (not args.no_save) and (valid_loss < best_valid):
            best_valid = valid_loss
            save_checkpoint(model, vocab, ckpt_path)
            print(f"✅ Saved best: {ckpt_path} (best_valid={best_valid:.4f})")

    total_dt = time.time() - total_t0
    print("\nTraining finished.")
    print("Best valid loss:", best_valid)
    print(f"Total time: {total_dt:.1f}s")
    if not args.no_save:
        print("Best checkpoint:", ckpt_path)


if __name__ == "__main__":
    main()
