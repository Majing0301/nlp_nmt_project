#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_t5_and_plot.py

One-click:
1) Fine-tune T5/MT5 on CN->EN jsonl
2) Log train_loss + eval_loss per epoch to CSV
3) Plot loss curves to plots/
4) Compute BLEU (greedy and beam) on test/valid set
5) Save predictions jsonl for qualitative inspection

Data format (jsonl):
- raw: {"zh":"...", "en":"...", "index":...}
- tokenized: {"zh_tok":[...], "en_tok":[...], "index":...}
"""

import os
# Use HF mirror if needed (optional)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_OFFLINE'] = '0'  # 确保在线下载

import json
import argparse
from typing import Dict, Any, List, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def load_jsonl_pairs(path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    data = []
    for i, obj in enumerate(iter_jsonl(path), start=1):
        if max_samples is not None and len(data) >= max_samples:
            break

        # zh
        if "zh" in obj and isinstance(obj["zh"], str) and obj["zh"].strip():
            zh = obj["zh"].strip()
        elif "zh_tok" in obj and isinstance(obj["zh_tok"], list) and len(obj["zh_tok"]) > 0:
            zh = " ".join([str(t) for t in obj["zh_tok"]]).strip()
        else:
            continue

        # en
        if "en" in obj and isinstance(obj["en"], str) and obj["en"].strip():
            en = obj["en"].strip()
        elif "en_tok" in obj and isinstance(obj["en_tok"], list) and len(obj["en_tok"]) > 0:
            en = " ".join([str(t) for t in obj["en_tok"]]).strip()
        else:
            continue

        data.append({"index": obj.get("index", i), "zh": zh, "en": en})

    return data

def compute_corpus_bleu_sacrebleu(preds: List[str], refs: List[str]) -> float:
    import sacrebleu
    # filter empty refs
    pairs = [(p, r) for p, r in zip(preds, refs) if isinstance(r, str) and r.strip()]
    if not pairs:
        return float("nan")
    preds_f = [p for p, _ in pairs]
    refs_f = [r for _, r in pairs]
    return float(sacrebleu.corpus_bleu(preds_f, [refs_f]).score)

def save_predictions_jsonl(out_path: str, items: List[Dict[str, Any]]):
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def plot_loss_curve(csv_path: str, out_png: str, title: str = "T5 Training / Validation Loss"):
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    # expected columns: epoch, train_loss, eval_loss
    plt.figure(figsize=(8, 6))

    if "train_loss" in df.columns and df["train_loss"].notna().any():
        plt.plot(df["epoch"], df["train_loss"], linestyle="--", linewidth=2, label="train_loss")
    if "eval_loss" in df.columns and df["eval_loss"].notna().any():
        plt.plot(df["epoch"], df["eval_loss"], linestyle="-", linewidth=2, label="valid_loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    ensure_dir(os.path.dirname(out_png))
    plt.savefig(out_png, dpi=300)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--data_dir", type=str, default="data/processed")
    p.add_argument("--train_file", type=str, default="train_100k.jsonl")
    p.add_argument("--valid_file", type=str, default="valid.jsonl")
    p.add_argument("--test_file", type=str, default="test.jsonl")
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_valid_samples", type=int, default=None)
    p.add_argument("--max_test_samples", type=int, default=None)

    # model
    p.add_argument("--model_name", type=str, default="google-t5/t5-small",
                   help="e.g., google-t5/t5-small, google-t5/t5-base, google/mt5-small")
    p.add_argument("--max_source_len", type=int, default=64)
    p.add_argument("--max_target_len", type=int, default=64)

    # training
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")

    # decoding / evaluation
    p.add_argument("--beam_size", type=int, default=4)
    p.add_argument("--eval_split", type=str, default="test", choices=["valid", "test"],
                   help="compute BLEU on which split")
    p.add_argument("--save_best", action="store_true",
                   help="save best checkpoint by eval_loss (needs evaluation each epoch)")

    # outputs
    p.add_argument("--out_dir", type=str, default="checkpoints/t5")
    p.add_argument("--logs_csv", type=str, default="logs/t5_train_log.csv")
    p.add_argument("--plots_dir", type=str, default="plots")
    p.add_argument("--pred_dir", type=str, default="outputs")

    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- load data --------
    train_path = os.path.join(args.data_dir, args.train_file)
    valid_path = os.path.join(args.data_dir, args.valid_file)
    test_path  = os.path.join(args.data_dir, args.test_file)

    train_data = load_jsonl_pairs(train_path, max_samples=args.max_train_samples)
    valid_data = load_jsonl_pairs(valid_path, max_samples=args.max_valid_samples) if os.path.exists(valid_path) else []
    test_data  = load_jsonl_pairs(test_path,  max_samples=args.max_test_samples)  if os.path.exists(test_path) else []

    if len(train_data) == 0:
        raise RuntimeError(f"Empty train set: {train_path}")

    # if valid too small, split from train
    if len(valid_data) < 50:
        ds_tmp = Dataset.from_list(train_data)
        split = ds_tmp.train_test_split(test_size=0.1, seed=args.seed)
        train_ds = split["train"]
        valid_ds = split["test"]
    else:
        train_ds = Dataset.from_list(train_data)
        valid_ds = Dataset.from_list(valid_data)

    # evaluation target split
    eval_items = valid_data if args.eval_split == "valid" else test_data
    if len(eval_items) == 0:
        # fallback to valid if test missing
        eval_items = valid_data

    # -------- tokenizer/model --------
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(device)

    # -------- preprocess --------
    prefix = "translate Chinese to English: "

    def preprocess_fn(examples):
        inputs = [prefix + z for z in examples["zh"]]
        targets = examples["en"]

        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_len,
            truncation=True,
            padding="max_length",
        )
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets,
                max_length=args.max_target_len,
                truncation=True,
                padding="max_length",
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_ds.map(preprocess_fn, batched=True, remove_columns=train_ds.column_names)
    tokenized_valid = valid_ds.map(preprocess_fn, batched=True, remove_columns=valid_ds.column_names)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # -------- training args --------
    ensure_dir(args.out_dir)
    ensure_dir(os.path.dirname(args.logs_csv))
    ensure_dir(args.plots_dir)
    ensure_dir(args.pred_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.out_dir,
        overwrite_output_dir=True,

        # evaluate per epoch so we can plot valid loss & save best
        eval_strategy="epoch",
        save_strategy="epoch",

        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,

        predict_with_generate=False,  # 我们自己在最后单独 generate + BLEU（更可控）
        fp16=(args.fp16 and torch.cuda.is_available()),

        logging_strategy="steps",
        logging_steps=50,

        load_best_model_at_end=bool(args.save_best),
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=[],
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_valid,
        tokenizer=tokenizer,  # HF 旧版本用 tokenizer；新版本也能用
        data_collator=data_collator,
    )

    # -------- train --------
    print(f"Device: {device}")
    print(f"Train size: {len(tokenized_train)} | Valid size: {len(tokenized_valid)}")
    print(f"Model: {args.model_name} | epochs={args.epochs} bs={args.batch_size} lr={args.lr}")

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    # -------- extract logs to CSV (per epoch) --------
    # trainer.state.log_history includes step-level and epoch-level logs
    rows = []
    for log in trainer.state.log_history:
        # we care about epoch-level train loss and eval loss
        if "epoch" in log:
            rows.append({
                "epoch": float(log.get("epoch")),
                "train_loss": log.get("loss", np.nan),
                "eval_loss": log.get("eval_loss", np.nan),
            })
    df = pd.DataFrame(rows)

    # aggregate by epoch (some epochs may have multiple entries)
    if not df.empty:
        df = df.sort_values("epoch")
        df = df.groupby("epoch", as_index=False).agg({
            "train_loss": "mean",
            "eval_loss": "mean",
        })
        df.to_csv(args.logs_csv, index=False, encoding="utf-8")
        print(f"[OK] Saved training log: {args.logs_csv}")

        plot_path = os.path.join(args.plots_dir, "t5_loss_curve.png")
        plot_loss_curve(args.logs_csv, plot_path, title="T5 Training vs Validation Loss")
        print(f"[OK] Saved loss plot: {plot_path}")
    else:
        print("[WARN] No log_history found. Loss plot/CSV skipped.")

    # -------- compute BLEU (greedy + beam) --------
    # Use raw eval_items to generate predictions
    model.eval()
    model.to(device)

    def generate_preds(decode: str) -> Tuple[List[str], List[Dict[str, Any]]]:
        preds = []
        out_jsonl = []
        for obj in eval_items:
            src = obj["zh"]
            ref = obj["en"]
            idx = obj.get("index", None)

            inp = prefix + src
            enc = tokenizer(
                inp,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_source_len,
            ).to(device)

            if decode == "beam":
                gen_ids = model.generate(
                    **enc,
                    max_length=args.max_target_len,
                    num_beams=args.beam_size,
                    early_stopping=True,
                )
            else:
                gen_ids = model.generate(
                    **enc,
                    max_length=args.max_target_len,
                    num_beams=1,
                    do_sample=False,
                )

            pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
            preds.append(pred)
            out_jsonl.append({"index": idx, "zh": src, "ref": ref, "pred": pred})
        return preds, out_jsonl

    refs = [x["en"] for x in eval_items]

    # greedy
    greedy_preds, greedy_jsonl = generate_preds("greedy")
    greedy_bleu = compute_corpus_bleu_sacrebleu(greedy_preds, refs)

    greedy_out = os.path.join(args.pred_dir, f"t5_preds_{args.eval_split}_greedy.jsonl")
    save_predictions_jsonl(greedy_out, greedy_jsonl)

    # beam
    beam_preds, beam_jsonl = generate_preds("beam")
    beam_bleu = compute_corpus_bleu_sacrebleu(beam_preds, refs)

    beam_out = os.path.join(args.pred_dir, f"t5_preds_{args.eval_split}_beam{args.beam_size}.jsonl")
    save_predictions_jsonl(beam_out, beam_jsonl)

    print("\n===== BLEU (sacrebleu) =====")
    print(f"Eval split: {args.eval_split} (size={len(eval_items)})")
    print(f"Greedy BLEU: {greedy_bleu:.2f}   | preds: {greedy_out}")
    print(f"Beam   BLEU: {beam_bleu:.2f} (beam={args.beam_size}) | preds: {beam_out}")

    # show a few examples
    print("\n===== Examples (first 3) =====")
    for i in range(min(3, len(eval_items))):
        print(f"\n[Example {i+1}]")
        print("SRC :", eval_items[i]["zh"])
        print("REF :", eval_items[i]["en"])
        print("GRE :", greedy_preds[i])
        print("BEAM:", beam_preds[i])

    print("\nDone.")
    print(f"- Model dir : {args.out_dir}")
    print(f"- Plot dir  : {args.plots_dir}")
    print(f"- Pred dir  : {args.pred_dir}")
    print(f"- Log CSV   : {args.logs_csv}")

if __name__ == "__main__":
    main()
