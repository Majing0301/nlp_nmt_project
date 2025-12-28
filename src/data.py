# src/data.py
# -*- coding: utf-8 -*-
"""
Data pipeline for CN-EN NMT (RNN/Transformer from scratch).
- Read JSONL with keys: {"en": "...", "zh": "...", "index": ...}
- Basic cleaning
- Tokenization: zh -> jieba, en -> simple regex tokens
- Build vocabulary from training data
- Numericalize + PyTorch Dataset/DataLoader with padding

Usage (example):
    from src.data import build_vocab_from_jsonl, create_dataloaders

    data_dir = "data/raw"
    vocab = build_vocab_from_jsonl(
        train_path=f"{data_dir}/train_10k.jsonl",
        min_freq=2,
        max_size=50000,
        max_len=60,
    )

    loaders = create_dataloaders(
        data_dir=data_dir,
        vocab=vocab,
        train_file="train_10k.jsonl",
        valid_file="valid.jsonl",
        test_file="test.jsonl",
        batch_size=64,
        max_len=60,
        num_workers=0
    )
    train_loader, valid_loader, test_loader = loaders
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# Chinese tokenization
try:
    import jieba  # type: ignore
except Exception:
    jieba = None


# ----------------------------
# Special tokens
# ----------------------------
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


# ----------------------------
# Cleaning + Tokenization
# ----------------------------
_illegal_ctrl = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")  # keep \t,\n,\r out; jsonl line already split
_multi_space = re.compile(r"\s+")
_en_tok = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def clean_text(s: str) -> str:
    """Minimal cleaning: remove control chars, normalize whitespace."""
    s = _illegal_ctrl.sub("", s)
    s = s.strip()
    s = _multi_space.sub(" ", s)
    return s


def tokenize_zh(s: str) -> List[str]:
    """Jieba tokenization (word-level). Falls back to char-level if jieba not available."""
    s = clean_text(s)
    if not s:
        return []
    if jieba is None:
        # fallback: character-level (still works, but vocab larger)
        return list(s.replace(" ", ""))
    # jieba returns generator; HMM default is ok for homework
    return [t for t in jieba.cut(s, cut_all=False) if t.strip()]


def tokenize_en(s: str) -> List[str]:
    """Simple English tokenization using regex (keeps punctuation as tokens)."""
    s = clean_text(s)
    s = s.lower()        # ✅ 加这一行
    if not s:
        return []
    return _en_tok.findall(s)


# ----------------------------
# Vocabulary
# ----------------------------
@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]
    pad_id: int
    unk_id: int
    bos_id: int
    eos_id: int

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], add_bos_eos: bool = True) -> List[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokens]
        if add_bos_eos:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> List[str]:
        out = []
        for i in ids:
            if stop_at_eos and i == self.eos_id:
                break
            out.append(self.itos[i] if 0 <= i < len(self.itos) else UNK_TOKEN)
        return out

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"itos": self.itos}, f, ensure_ascii=False)

    @staticmethod
    def load(path: str) -> "Vocab":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        itos = obj["itos"]
        stoi = {t: i for i, t in enumerate(itos)}
        return Vocab(
            stoi=stoi,
            itos=itos,
            pad_id=stoi[PAD_TOKEN],
            unk_id=stoi[UNK_TOKEN],
            bos_id=stoi[BOS_TOKEN],
            eos_id=stoi[EOS_TOKEN],
        )


def build_vocab(
    token_seqs: Iterable[Sequence[str]],
    min_freq: int = 2,
    max_size: int = 50000,
) -> Vocab:
    """
    Build vocab from token sequences.
    Keeps special tokens at the beginning.
    """
    counter = Counter()
    for seq in token_seqs:
        counter.update(seq)

    # Special tokens first
    itos: List[str] = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    # Most common with frequency filter
    for tok, freq in counter.most_common():
        if freq < min_freq:
            continue
        if tok in (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN):
            continue
        itos.append(tok)
        if len(itos) >= max_size:
            break

    stoi = {t: i for i, t in enumerate(itos)}
    return Vocab(
        stoi=stoi,
        itos=itos,
        pad_id=stoi[PAD_TOKEN],
        unk_id=stoi[UNK_TOKEN],
        bos_id=stoi[BOS_TOKEN],
        eos_id=stoi[EOS_TOKEN],
    )


def iter_jsonl_pairs(
    path: str,
    max_len: Optional[int] = None,
) -> Iterable[Tuple[List[str], List[str]]]:
    """
    Yield (zh_tokens, en_tokens) pairs from jsonl.
    Supports both:
      - raw jsonl: {"zh": "...", "en": "..."}
      - processed jsonl: {"zh_tok": [...], "en_tok": [...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # ✅ 支持 processed
            if "zh_tok" in obj and "en_tok" in obj:
                zh_toks = obj["zh_tok"]
                en_toks = [t.lower() for t in obj["en_tok"]]   # ✅ 统一在训练阶段 lowercase
            else:
                zh = obj.get("zh", "")
                en = obj.get("en", "")
                zh_toks = tokenize_zh(zh)
                en_toks = tokenize_en(en)

            if not zh_toks or not en_toks:
                continue
            if max_len is not None and (len(zh_toks) > max_len or len(en_toks) > max_len):
                continue
            yield zh_toks, en_toks


def build_vocab_from_jsonl(
    train_path: str,
    min_freq: int = 2,
    max_size: int = 50000,
    max_len: int = 60,
) -> Dict[str, Vocab]:
    """
    Build separate vocabs for zh and en from train jsonl.
    Returns {"zh": Vocab, "en": Vocab}
    """
    zh_seqs = []
    en_seqs = []
    cnt = 0  # ✅ 加在这里：循环前初始化计数器
    for zh_toks, en_toks in iter_jsonl_pairs(train_path, max_len=max_len):
        cnt += 1  # ✅ 每读取到一对句子就+1
        zh_seqs.append(zh_toks)
        en_seqs.append(en_toks)

    print("Pairs used to build vocab:", cnt)  # ✅ 循环结束后打印

    zh_vocab = build_vocab(zh_seqs, min_freq=min_freq, max_size=max_size)
    en_vocab = build_vocab(en_seqs, min_freq=min_freq, max_size=max_size)
    return {"zh": zh_vocab, "en": en_vocab}


# ----------------------------
# Dataset + Collate
# ----------------------------
class JsonlNMTDataset(Dataset):
    """
    Returns:
        src_ids: LongTensor [src_len]
        tgt_ids: LongTensor [tgt_len]
        index: int
    """
    def __init__(
        self,
        path: str,
        zh_vocab: Vocab,
        en_vocab: Vocab,
        max_len: int = 60,
    ):
        self.path = path
        self.zh_vocab = zh_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len

        # Load into memory (10k/100k is fine for homework; if too big, we can stream later)
        self.samples: List[Tuple[List[int], List[int], int]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                idx = int(obj.get("index", len(self.samples)))

                # ✅ 优先使用预处理后的 token（data/processed）
                if "zh_tok" in obj and "en_tok" in obj:
                    zh_toks = obj["zh_tok"]
                    en_toks = [t.lower() for t in obj["en_tok"]]   # ✅ 统一在训练阶段 lowercase
                else:
                    # ✅ 兼容原始数据（data/raw）
                    zh_toks = tokenize_zh(obj.get("zh", ""))
                    en_toks = tokenize_en(obj.get("en", ""))

                if not zh_toks or not en_toks:
                    continue
                if len(zh_toks) > max_len or len(en_toks) > max_len:
                    continue
                src_ids = zh_vocab.encode(zh_toks, add_bos_eos=True)
                tgt_ids = en_vocab.encode(en_toks, add_bos_eos=True)
                self.samples.append((src_ids, tgt_ids, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        src_ids, tgt_ids, idx = self.samples[i]
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            idx,
        )


def pad_1d(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad list of [L] -> [B, Lmax], also return lengths [B]
    """
    lengths = torch.tensor([s.numel() for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) else 0
    out = torch.full((len(seqs), max_len), fill_value=pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : s.numel()] = s
    return out, lengths


def collate_nmt(batch, src_pad_id: int, tgt_pad_id: int):
    src_seqs, tgt_seqs, idxs = zip(*batch)
    src_padded, src_lens = pad_1d(list(src_seqs), src_pad_id)
    tgt_padded, tgt_lens = pad_1d(list(tgt_seqs), tgt_pad_id)
    idxs = torch.tensor(list(idxs), dtype=torch.long)
    return {
        "src": src_padded,
        "src_len": src_lens,
        "tgt": tgt_padded,
        "tgt_len": tgt_lens,
        "index": idxs,
    }


def create_dataloaders(
    data_dir: str,
    vocab: Dict[str, Vocab],
    train_file: str = "train_10k.jsonl",
    valid_file: str = "valid.jsonl",
    test_file: str = "test.jsonl",
    batch_size: int = 64,
    max_len: int = 60,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    zh_vocab = vocab["zh"]
    en_vocab = vocab["en"]

    train_path = os.path.join(data_dir, train_file)
    valid_path = os.path.join(data_dir, valid_file)
    test_path = os.path.join(data_dir, test_file)

    train_ds = JsonlNMTDataset(train_path, zh_vocab, en_vocab, max_len=max_len)
    valid_ds = JsonlNMTDataset(valid_path, zh_vocab, en_vocab, max_len=max_len)
    test_ds = JsonlNMTDataset(test_path, zh_vocab, en_vocab, max_len=max_len)

    collate_fn = lambda b: collate_nmt(b, src_pad_id=zh_vocab.pad_id, tgt_pad_id=en_vocab.pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    return train_loader, valid_loader, test_loader
