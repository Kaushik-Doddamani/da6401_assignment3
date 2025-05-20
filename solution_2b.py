#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solution_2b.py: Tune beam size for your pretrained Hindi transliteration model.

This script:
  1. Loads the Hindi Dakshina train/dev/test splits.
  2. Trains a Seq2Seq model with your best hyper-parameters.
  3. Runs beam-search decoding on the dev set for various beam sizes.
  4. Reports exact‐match accuracy for each beam size.

Usage example:
    python solution_2b.py \
        --train_tsv path/to/hi.translit.sampled.train.tsv \
        --dev_tsv   path/to/hi.translit.sampled.dev.tsv \
        --test_tsv  path/to/hi.translit.sampled.test.tsv \
        --gpu_ids   3

Arguments:
  --train_tsv   Path to the Hindi training lexicon TSV
  --dev_tsv     Path to the Hindi development lexicon TSV
  --test_tsv    Path to the Hindi test lexicon TSV
  --gpu_ids     CUDA device IDs to use (e.g. --gpu_ids 0 1)
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Tuple
import os

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# import everything from your Q1 solution
from solution_1 import (
    DakshinaLexicon,
    CharVocabulary,
    Seq2SeqConfig,
    Seq2Seq,
    collate_batch,
    train_epoch,
    eval_epoch,
)

# ─── best hyper-parameters found ─────────────────────────────
best = {
    "batch_size":        64,
    "cell":             "LSTM",
    "decoder_layers":     1,
    "dropout":          0.3,
    "embedding_method":"onehot",
    "embedding_size":    16,
    "encoder_layers":     1,
    "epochs":            15,
    "hidden_size":       64,
    "learning_rate":   1.1367425385944718e-3,
    "teacher_forcing":   0.7,
    "use_attestations":  True,
}

def parse_args():
    p = argparse.ArgumentParser(description="Train with fixed hyperparams and tune beam size")
    p.add_argument("--train_tsv",  type=str, required=True, help="Path to Hindi train lexicon TSV")
    p.add_argument("--dev_tsv",    type=str, required=True, help="Path to Hindi dev lexicon TSV")
    p.add_argument("--test_tsv",   type=str, required=True, help="Path to Hindi test lexicon TSV")
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
) -> Tuple[DataLoader, DataLoader, DataLoader, CharVocabulary, CharVocabulary]:
    """
    Builds train/dev/test datasets and loaders, re-using vocab built on train.
    If use_attest=True, uses a WeightedRandomSampler on train counts.
    """
    # build train ds + vocabs
    train_ds = DakshinaLexicon(train_path, build_vocabs=True, use_attestations=use_attest)
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
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  shuffle=True, collate_fn=collate_fn)

    # dev/test
    dev_loader  = DataLoader(dev_ds,  batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn)

    return train_loader, dev_loader, test_loader, src_vocab, tgt_vocab

def train_model(
    train_loader: DataLoader,
    dev_loader:   DataLoader,
    cfg:          Seq2SeqConfig,
    device:       torch.device,
    epochs:       int,
    lr:           float,
    teacher_force:    float
) -> Seq2Seq:
    """
    Trains the Seq2Seq model for `epochs` epochs.
    Returns the trained model.
    """
    model = Seq2Seq(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader,
                                 optimizer, loss_fn, device, teacher_force)
        dev_loss   = eval_epoch(model, dev_loader, loss_fn, device)
        print(f"Epoch {epoch:02d} | "
              f"train ppl={math.exp(train_loss):.2f} | "
              f"dev   ppl={math.exp(dev_loss):.2f}")
    return model

def evaluate_beam_exact_match(
    model:       Seq2Seq,
    dataset:     DakshinaLexicon,
    src_vocab:   CharVocabulary,
    tgt_vocab:   CharVocabulary,
    beam_size:   int,
    device:      torch.device
) -> float:
    """
    Runs beam_search_decode on *each* example in `dataset` (batch_size=1),
    computes exact‐match rate vs. gold target.
    """
    model.eval()
    # unwrap DataParallel if needed
    inference_model = getattr(model, "module", model)

    total, correct = 0, 0
    with torch.no_grad():
        for src_ids, tgt_ids in tqdm(dataset, desc=f"beam={beam_size}", leave=False):
            # prepare tensors
            src_len = len(src_ids)
            src_tensor = torch.tensor([src_ids], device=device)
            len_tensor = torch.tensor([src_len], device=device)

            # beam search decode
            pred_ids = inference_model.beam_search_decode(
                src_tensor, len_tensor,
                beam_size=beam_size,
                max_len= max(len(src_ids)*2, 30)
            )[0].tolist()

            # decode strings
            pred_str = tgt_vocab.decode(pred_ids)
            gold_str = tgt_vocab.decode(tgt_ids[1:])  # skip <sos>

            if pred_str == gold_str:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0

def main():
    args = parse_args()
    # pin visible GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── build data loaders ────────────────────────────────────────────
    train_loader, dev_loader, test_loader, src_vocab, tgt_vocab = build_data_loaders(
        train_path= args.train_tsv,
        dev_path=   args.dev_tsv,
        test_path=  args.test_tsv,
        batch_size= best["batch_size"],
        use_attest= best["use_attestations"],
    )

    # ─── construct model config ────────────────────────────────────────
    model_cfg = Seq2SeqConfig(
        source_vocab_size= src_vocab.size,
        target_vocab_size= tgt_vocab.size,
        embedding_dim=     best["embedding_size"],
        hidden_dim=        best["hidden_size"],
        encoder_layers=    best["encoder_layers"],
        decoder_layers=    best["decoder_layers"],
        cell_type=         best["cell"],
        dropout=           best["dropout"],
        pad_index=         src_vocab.stoi["<pad>"],
        sos_index=         tgt_vocab.stoi["<sos>"],
        eos_index=         tgt_vocab.stoi["<eos>"],
        embedding_method=  best["embedding_method"],
        # no SVD here since embedding_method="onehot"
    )

    # ─── train the model ────────────────────────────────────────────────
    model = train_model(
        train_loader, dev_loader,
        cfg=model_cfg,
        device=device,
        epochs=best["epochs"],
        lr=best["learning_rate"],
        teacher_force=best["teacher_forcing"],
    )

    # ─── final test perplexity ───────────────────────────────────────────
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model_cfg.pad_index)
    test_loss = eval_epoch(model, test_loader, loss_fn, device)
    print(f"\nTest perplexity: {math.exp(test_loss):.2f}\n")

    # ─── beam‐size sweep on dev set ──────────────────────────────────────
    print("Tuning beam size on dev set (exact‐match rate):")
    for beam in [1, 2, 3, 5, 8, 10]:
        acc = evaluate_beam_exact_match(
            model, dev_loader.dataset, src_vocab, tgt_vocab,
            beam_size=beam, device=device
        )
        print(f"  beam_size={beam:2d} → dev accuracy = {acc * 100:5.2f}%")

if __name__ == "__main__":
    main()
