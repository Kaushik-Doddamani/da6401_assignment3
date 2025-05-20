#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solution_5a.py: Tune beam size for your pretrained attention‐based Hindi transliteration model.

Usage example:
    python solution_5a.py \
        --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
        --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
        --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
        --gpu_ids   0 1

This script:
  1. Loads the Dakshina Hindi train/dev/test splits.
  2. Trains a Seq2SeqAttention model with your best attention hyperparameters.
  3. Runs beam‐search decoding on the dev set for various beam sizes.
  4. Reports exact‐match accuracy for each beam size.
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# import training utilities and dataset from Q1
from solution_1 import DakshinaLexicon, collate_batch, train_epoch, eval_epoch
# import attention model definitions from Q5
from solution_5_model import Seq2SeqAttentionConfig, Seq2SeqAttention

# ──────────────────────────── best hyperparameters ────────────────────────────
best = {
    "batch_size":        128,
    "cell_type":       "GRU",
    "encoder_layers":     1,
    "decoder_layers":     2,
    "hidden_size":      512,
    "embedding_method":"svd_ppmi",
    "embedding_size":   64,
    "dropout":          0.2,
    "learning_rate":   0.0006899910999897612,
    "teacher_forcing":  0.5,
    "use_attestations": True,
    "epochs":           10,
}
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Tune beam size for your best attention‐based model"
    )
    p.add_argument(
        "--train_tsv",  type=str, required=True,
        help="Path to Hindi training lexicon TSV"
    )
    p.add_argument(
        "--dev_tsv",    type=str, required=True,
        help="Path to Hindi development lexicon TSV"
    )
    p.add_argument(
        "--test_tsv",   type=str, required=True,
        help="Path to Hindi test lexicon TSV"
    )
    p.add_argument(
        "--gpu_ids", type=int, nargs="+", default=[0],
        help="CUDA device IDs to use (e.g. 0 1)."
    )
    return p.parse_args()

def build_data_loaders(
    train_path: str,
    dev_path:   str,
    test_path:  str,
    batch_size: int,
    use_attest: bool
) -> Tuple[DataLoader, DataLoader, DataLoader, DakshinaLexicon, DakshinaLexicon]:
    """
    Builds train/dev/test datasets and loaders, re-using vocab built on train.
    If use_attest=True, uses a WeightedRandomSampler on train counts.
    """
    # build train ds + vocabs
    train_ds = DakshinaLexicon(
        train_path,
        build_vocabs=True,
        use_attestations=use_attest
    )
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab

    # dev/test reuse same vocabs
    dev_ds  = DakshinaLexicon(dev_path,  src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(test_path, src_vocab, tgt_vocab)

    # collate_fn for padding
    pad_id = src_vocab.stoi["<pad>"]
    collate_fn = lambda batch: collate_batch(batch, pad_id=pad_id)

    # train loader: optionally weighted by attestations
    if use_attest:
        sampler = WeightedRandomSampler(
            weights=train_ds.example_counts,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            sampler=sampler, collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size,
            shuffle=True, collate_fn=collate_fn
        )

    # dev/test loaders
    dev_loader  = DataLoader(
        dev_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, collate_fn=collate_fn
    )

    return train_loader, dev_loader, test_loader, train_ds, dev_ds

def train_model(
    train_loader: DataLoader,
    dev_loader:   DataLoader,
    cfg:          Seq2SeqAttentionConfig,
    device:       torch.device,
    epochs:       int,
    lr:           float,
    teacher_forcing: float
) -> Seq2SeqAttention:
    """
    Trains the Seq2SeqAttention model for `epochs` epochs.
    Returns the trained model.
    """
    model = Seq2SeqAttention(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model, train_loader,
            optimizer, loss_fn,
            device, teacher_forcing
        )
        dev_loss   = eval_epoch(
            model, dev_loader,
            loss_fn, device
        )
        print(f"Epoch {epoch:02d} | train ppl={math.exp(train_loss):.2f} | dev ppl={math.exp(dev_loss):.2f}")
    return model

def evaluate_beam_exact_match(
    model:       Seq2SeqAttention,
    dataset:     DakshinaLexicon,
    beam_size:   int,
    device:      torch.device
) -> float:
    """
    Runs beam_search_decode on each example in `dataset` (batch_size=1),
    computes exact‐match rate vs. gold target.
    """
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for src_ids, tgt_ids in tqdm(dataset, desc=f"beam={beam_size}", leave=False):
            total += 1
            # prepare tensors
            src_len = len(src_ids)
            src_tensor = torch.tensor([src_ids], device=device)
            len_tensor = torch.tensor([src_len], device=device)

            # beam search decode
            pred_ids = model.beam_search_decode(
                src_tensor, len_tensor,
                beam_size=beam_size,
                max_len=max(src_len * 2, 50)
            )[0].tolist()

            # drop leading <sos> if present
            if pred_ids and pred_ids[0] == dataset.tgt_vocab.stoi["<sos>"]:
                pred_ids = pred_ids[1:]

            # decode strings
            pred_str = dataset.tgt_vocab.decode(pred_ids)
            gold_str = dataset.tgt_vocab.decode(tgt_ids[1:])

            if pred_str == gold_str:
                correct += 1

    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()

    # pin GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build data loaders
    train_loader, dev_loader, _, train_ds, dev_ds = build_data_loaders(
        train_path=args.train_tsv,
        dev_path=args.dev_tsv,
        test_path=args.test_tsv,    # unused here
        batch_size=best["batch_size"],
        use_attest=best["use_attestations"]
    )

    # construct model config
    extra_cfg = {}
    if best["embedding_method"] == "svd_ppmi":
        extra_cfg["svd_sources"] = train_ds.encoded_sources

    cfg = Seq2SeqAttentionConfig(
        source_vocab_size=train_ds.src_vocab.size,
        target_vocab_size=train_ds.tgt_vocab.size,
        embedding_dim=best["embedding_size"],
        hidden_dim=best["hidden_size"],
        encoder_layers=best["encoder_layers"],
        decoder_layers=best["decoder_layers"],
        cell_type=best["cell_type"],
        dropout=best["dropout"],
        pad_index=train_ds.src_vocab.stoi["<pad>"],
        sos_index=train_ds.tgt_vocab.stoi["<sos>"],
        eos_index=train_ds.tgt_vocab.stoi["<eos>"],
        embedding_method=best["embedding_method"],
        **extra_cfg,
    )

    # train the model
    model = train_model(
        train_loader, dev_loader,
        cfg=cfg,
        device=device,
        epochs=best["epochs"],
        lr=best["learning_rate"],
        teacher_forcing=best["teacher_forcing"]
    )

    # tune beam size on dev set
    print("\nTuning beam size on dev set (exact‐match rate):")
    for beam in [1, 2, 3, 5, 8, 10]:
        acc = evaluate_beam_exact_match(
            model, dev_ds,
            beam_size=beam,
            device=device
        )
        print(f"  beam_size={beam:2d} → dev accuracy = {acc * 100:5.2f}%")

if __name__ == "__main__":
    main()
