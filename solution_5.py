#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5: W&B sweep driver for attention‐augmented Seq2Seq
────────────────────────────────────────────────────────────────
Reuses the attention Seq2Seq from solution_5_model.py.
A YAML under ./configs/ specifies the sweep space.

Usage:
    # Sweep:
    python solution_5.py \
      --mode sweep \
      --sweep_config sweep_attention.yaml \
      --wandb_project transliteration \
      --wandb_run_tag attention \
      --gpu_ids 0 1 \
      --train_tsv ... \
      --dev_tsv   ... \
      --test_tsv  ... \
      --sweep_count 30

    # Single debug run:
    python solution_5.py \
      --mode single \
      --wandb_project transliteration \
      --wandb_run_tag attention_debug \
      --train_tsv ... \
      --dev_tsv   ... \
      --test_tsv  ...
"""
from __future__ import annotations
import argparse
import math
import os
import yaml
from pathlib import Path
from typing import Any, Dict

import torch
import wandb
from torch.utils.data import DataLoader

# Q1 utilities
from solution_1 import (
    DakshinaLexicon,
    CharVocabulary,
    collate_batch,
    train_epoch,
    eval_epoch,
)

# Our attention model
from solution_5_model import Seq2SeqAttentionConfig, Seq2SeqAttention


def get_configs(project_root: str | Path, config_filename: str) -> Dict[str, Any]:
    """Load a YAML sweep configuration from ./configs/."""
    cfg_path = Path(project_root) / "configs" / config_filename
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def compute_sequence_accuracy(
    model: Seq2SeqAttention,
    dataset: DakshinaLexicon,
    device: str,
    beam_size: int = 1
) -> float:
    """
    Exact‐match accuracy over the dataset using beam-search.
    Strips leading <sos> from predictions before decoding.
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src_ids, tgt_ids in dataset:
            src_tensor = torch.tensor([src_ids], device=device)
            src_len    = torch.tensor([len(src_ids)], device=device)

            pred = model.beam_search_decode(
                src_tensor, src_len,
                beam_size=beam_size,
                max_len=len(tgt_ids)
            )[0]  # (L,)

            pred_list = pred.tolist()
            # drop leading <sos>
            sos_idx = dataset.tgt_vocab.stoi["<sos>"]
            if pred_list and pred_list[0] == sos_idx:
                pred_list = pred_list[1:]

            pred_str = dataset.tgt_vocab.decode(pred_list)
            gold_str = dataset.tgt_vocab.decode(tgt_ids[1:])

            correct += int(pred_str == gold_str)
            total   += 1

    return correct / total if total > 0 else 0.0


def run_single_training(sweep_config: Dict[str, Any], static_args: argparse.Namespace) -> None:
    """
    Train + evaluate once using hyperparams in sweep_config
    and fixed filepaths and tags from CLI.
    """
    # pin GPUs
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in static_args.gpu_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_ds = DakshinaLexicon(
        static_args.train_tsv,
        build_vocabs=True,
        use_attestations=sweep_config.get("use_attestations", False)
    )
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
    dev_ds  = DakshinaLexicon(static_args.dev_tsv,  src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(static_args.test_tsv, src_vocab, tgt_vocab)

    collate_fn = lambda batch: collate_batch(batch,
                                             pad_id=src_vocab.stoi["<pad>"])
    train_loader = DataLoader(train_ds,
                              batch_size=sweep_config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,
                              batch_size=sweep_config["batch_size"],
                              shuffle=False,
                              collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,
                              batch_size=sweep_config["batch_size"],
                              shuffle=False,
                              collate_fn=collate_fn)

    # Model + optimizer + loss
    extra = {}
    if sweep_config["embedding_method"] == "svd_ppmi":
        extra["svd_sources"] = train_ds.encoded_sources

    cfg = Seq2SeqAttentionConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=sweep_config["embedding_size"],
        hidden_dim=sweep_config["hidden_size"],
        encoder_layers=sweep_config["encoder_layers"],
        decoder_layers=sweep_config["decoder_layers"],
        cell_type=sweep_config["cell"],
        dropout=sweep_config["dropout"],
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=sweep_config["embedding_method"],
        **extra
    )
    model     = Seq2SeqAttention(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sweep_config["learning_rate"])
    loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    # build a run name from hyperparams
    run_name = (
        f"emb:{sweep_config['embedding_method']}|es:{sweep_config['embedding_size']}|"
        f"cell:{sweep_config['cell']}|hs:{sweep_config['hidden_size']}|"
        f"enc:{sweep_config['encoder_layers']}|dec:{sweep_config['decoder_layers']}|"
        f"dr:{sweep_config['dropout']}|lr:{sweep_config['learning_rate']:.1e}|"
        f"bsz:{sweep_config['batch_size']}|tf:{sweep_config['teacher_forcing']}|"
        f"ep:{sweep_config['epochs']}|beam:{sweep_config.get('beam_size',1)}|"
        f"att:{sweep_config.get('use_attestations',False)}"
    )
    wandb.run.name = run_name
    wandb.run.tags = [static_args.wandb_run_tag]

    # Train / validate
    for epoch in range(1, sweep_config["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn,
            device, sweep_config["teacher_forcing"]
        )
        dev_loss = eval_epoch(model, dev_loader, loss_fn, device)

        wandb.log({
            "Q5_epoch":       epoch,
            "Q5_train_loss":  train_loss,
            "Q5_train_ppl":   math.exp(train_loss),
            "Q5_dev_loss":    dev_loss,
            "Q5_dev_ppl":     math.exp(dev_loss),
        })

    # final dev accuracy via beam-search
    beam = sweep_config.get("beam_size", 1)
    dev_acc = compute_sequence_accuracy(model, dev_ds,
                                        device, beam_size=beam)
    wandb.log({"Q5_dev_accuracy": dev_acc})


def main():
    parser = argparse.ArgumentParser(
        description="W&B sweep driver for Q5 attention model."
    )
    parser.add_argument("--mode",         choices=["sweep","single"], required=True)
    parser.add_argument("--sweep_config", type=str, default="sweep_attention.yaml")
    parser.add_argument("--wandb_project",type=str, required=True)
    parser.add_argument("--wandb_run_tag",type=str, default="attention")
    parser.add_argument("--gpu_ids",      nargs="+", type=int, default=[0])
    parser.add_argument("--train_tsv",    type=str, required=True)
    parser.add_argument("--dev_tsv",      type=str, required=True)
    parser.add_argument("--test_tsv",     type=str, required=True)
    parser.add_argument("--sweep_count",  type=int, default=30)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    sweep_yaml   = get_configs(project_root, args.sweep_config)

    if args.mode == "sweep":
        # ensure metric, method, etc.
        sweep_yaml.setdefault("method", "bayes")
        sweep_yaml.setdefault("metric", {"name":"dev_perplexity","goal":"minimize"})
        sweep_yaml.setdefault("parameters", sweep_yaml.get("parameters",{}))
        sweep_yaml["program"] = Path(__file__).name
        sweep_yaml["run_cap"]  = args.sweep_count

        sweep_id = wandb.sweep(sweep=sweep_yaml,
                               project=args.wandb_project)
        print(f"Registered sweep: {sweep_id}")

        def _agent():
            with wandb.init(project=args.wandb_project) as run:
                run_single_training(dict(run.config), args)

        wandb.agent(sweep_id, function=_agent,
                    count=args.sweep_count)

    else:
        # single debug run
        with wandb.init(
            project=args.wandb_project,
            config=sweep_yaml.get("parameters", {})
        ) as run:
            run.config.update({
                "epochs":            3,
                "batch_size":       64,
                "embedding_method":"learned",
                "embedding_size":   64,
                "hidden_size":     128,
                "encoder_layers":    1,
                "decoder_layers":    1,
                "cell":            "LSTM",
                "dropout":          0.1,
                "learning_rate":   1e-3,
                "teacher_forcing":  0.5,
                "use_attestations":False,
                "beam_size":        1,
            }, allow_val_change=True)
            run_single_training(dict(run.config), args)

if __name__ == "__main__":
    main()
