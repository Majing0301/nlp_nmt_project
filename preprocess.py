# src/preprocess.py
# -*- coding: utf-8 -*-
"""
One-time preprocessing for CN-EN NMT:
- Read raw *.jsonl (each line: {"zh": "...", "en": "...", "index": ...})
- Clean (light)
- Tokenize (zh via jieba, en via regex)
- Truncate by max_len (tokens) and add special tokens later in training (recommended)
- Save to data/processed/*.jsonl as:
    {"zh_tok": [...], "en_tok": [...], "index": i}

Why:
- Avoid re-tokenizing every training run (big speedup on CPU)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Optional


# -----------------------------
# Tokenizers (keep consistent with training)
# -----------------------------
def tokenize_zh(text: str) -> List[str]:
    import jieba
    # reduce noisy logs
    try:
        jieba.setLogLevel(20)
    except Exception:
        pass
    return [t.strip() for t in jieba.cut(text) if t.strip()]


_en_word_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def tokenize_en(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return _en_word_re.findall(text)


# -----------------------------
# Cleaning (lightweight, safe)
# -----------------------------
def clean_text(s: str) -> str:
    # normalize whitespace
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def iter_jsonl(path: str) -> Iterable[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def truncate(tokens: List[str], max_len: int) -> List[str]:
    # max_len counts tokens only (without BOS/EOS). We'll add BOS/EOS in Dataset later.
    if max_len is None or max_len <= 0:
        return tokens
    return tokens[:max_len]


def preprocess_file(
    in_path: str,
    out_path: str,
    max_len: int,
    lowercase_en: bool,
    drop_empty: bool = True,
):
    n_in, n_out = 0, 0

    def gen():
        nonlocal n_in, n_out
        for obj in iter_jsonl(in_path):
            n_in += 1
            zh = clean_text(obj.get("zh", ""))
            en = clean_text(obj.get("en", ""))
            idx = obj.get("index", n_in - 1)

            zh_tok = truncate(tokenize_zh(zh), max_len)
            en_tok = truncate(tokenize_en(en, lowercase=lowercase_en), max_len)

            if drop_empty and (len(zh_tok) == 0 or len(en_tok) == 0):
                continue

            n_out += 1
            yield {"zh_tok": zh_tok, "en_tok": en_tok, "index": idx}

    write_jsonl(out_path, gen())
    return n_in, n_out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", type=str, default="data/raw")
    p.add_argument("--out_dir", type=str, default="data/processed")
    p.add_argument("--max_len", type=int, default=60, help="truncate token length (without BOS/EOS)")
    p.add_argument("--lowercase_en", action="store_true", help="lowercase English")
    p.add_argument("--files", nargs="*", default=["train_10k.jsonl", "train_100k.jsonl", "valid.jsonl", "test.jsonl"])
    args = p.parse_args()

    print("Raw dir :", args.raw_dir)
    print("Out dir :", args.out_dir)
    print("Files   :", args.files)
    print("max_len :", args.max_len)
    print("lowercase_en:", args.lowercase_en)

    for fn in args.files:
        in_path = os.path.join(args.raw_dir, fn)
        out_path = os.path.join(args.out_dir, fn)

        if not os.path.exists(in_path):
            print(f"[SKIP] not found: {in_path}")
            continue

        n_in, n_out = preprocess_file(
            in_path=in_path,
            out_path=out_path,
            max_len=args.max_len,
            lowercase_en=args.lowercase_en,
        )
        print(f"[OK] {fn}: {n_in} -> {n_out} lines | saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
