#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q2: W&B sweep driver for character-level Hindi transliteration
────────────────────────────────────────────────────────────────
Re-uses the full Seq2Seq implementation from solution_1.py (Q1).
A YAML file under ./configs/ specifies the sweep search space.

Example usage
~~~~~~~~~~~~~
# 1) Create the sweep and directly launch the agent:
python solution_2.py \
    --mode sweep \
    --sweep_config sweep_config.yaml \
    --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
    --wandb_run_tag solution_2 \
    --gpu_ids 0 1 \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
    --sweep_count 50

# 2) For a single debug run:
python solution_2.py \
    --mode single \
    --wandb_project transliteration \
    --wandb_run_tag baseline \
    --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
    --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
    --test_tsv  ./lexicons/hi.translit.sampled.test.tsv
"""

from __future__ import annotations

# ───────────────────────── Imports ─────────────────────────
import argparse
import math
import os
import warnings
import yaml
from pathlib import Path
from typing import Any, Dict, List

import torch
import wandb

# Import Q1 implementation (updated solution_1.py)
from solution_1 import (
    DakshinaLexicon,
    CharVocabulary,
    Seq2SeqConfig,
    Seq2Seq,
    collate_batch,
    train_epoch,
    eval_epoch,
)

# ────────────── Helpers ──────────────

def get_configs(project_root: str | Path, config_filename: str) -> Dict[str, Any]:
    """
    Load a YAML sweep configuration from ./configs/ relative to project_root.
    """
    cfg_path = Path(project_root) / "configs" / config_filename
    with open(cfg_path, "r", encoding="utf-8") as handle:
        config: Dict[str, Any] = yaml.safe_load(handle)
    return config


def compute_sequence_accuracy(
    model: Seq2Seq,
    dataset: DakshinaLexicon,
    device: str,
    beam_size: int = 1
) -> float:
    """
    Compute exact-match accuracy over a dataset using greedy or beam decoding.

    Args:
        model      – trained Seq2Seq module (unwrapped if DataParallel)
        dataset    – DakshinaLexicon providing (src_ids, tgt_ids)
        device     – 'cpu' or 'cuda'
        beam_size  – if >1, use beam_search_decode; else greedy_decode
    Returns:
        accuracy   – fraction of examples where pred == gold
    """
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src_ids, tgt_ids in dataset:
            # prepare input tensors
            src_tensor = torch.tensor([src_ids], device=device)
            src_len_tensor = torch.tensor([len(src_ids)], device=device)


            pred_ids = model.beam_search_decode(
                src_tensor, src_len_tensor,
                beam_size=beam_size, max_len=len(tgt_ids)
            )[0]

            # convert to strings (skip leading <sos>)
            gold_str = dataset.tgt_vocab.decode(tgt_ids[1:])

            # convert to strings, *dropping* the leading <sos> token
            pred_ids_list = pred_ids.tolist()
            if pred_ids_list and pred_ids_list[0] == dataset.tgt_vocab.stoi["<sos>"]:
                pred_ids_list = pred_ids_list[1:]
            pred_str = dataset.tgt_vocab.decode(pred_ids_list)
            correct += int(pred_str == gold_str)
            total += 1
    return correct / total if total > 0 else 0.0


# ────────────── 2. A single training run (used by sweep) ──────────────
def run_single_training(sweep_config: Dict[str, Any], static_args: argparse.Namespace) -> None:
    """
    Train + evaluate once using the hyper-parameters in sweep_config and
    the static file paths / tag supplied via CLI.
    """
    # ─── Honour multi-GPU pinning via CUDA_VISIBLE_DEVICES ─────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in static_args.gpu_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Warn if encoder/decoder depths differ ─────────────────────────────
    if sweep_config["encoder_layers"] != sweep_config["decoder_layers"]:
        warnings.warn(
            f"Encoder layers ({sweep_config['encoder_layers']}) != "
            f"decoder layers ({sweep_config['decoder_layers']}), "
            "hidden states will be aligned automatically.",
            UserWarning
        )

    # ─── Data loading ────────────────────────────────────────────────
    train_dataset = DakshinaLexicon(
        static_args.train_tsv,
        build_vocabs=True,
        use_attestations=sweep_config.get("use_attestations", False),
    )
    src_vocab: CharVocabulary = train_dataset.src_vocab
    tgt_vocab: CharVocabulary = train_dataset.tgt_vocab

    dev_dataset  = DakshinaLexicon(static_args.dev_tsv,  src_vocab, tgt_vocab)
    test_dataset = DakshinaLexicon(static_args.test_tsv, src_vocab, tgt_vocab)

    collate_fn = lambda batch: collate_batch(batch, pad_id=src_vocab.stoi["<pad>"])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=sweep_config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader  = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=sweep_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=sweep_config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn
    )

    # ─── Build the Seq2Seq model ─────────────────────────────────────────
    extra_cfg: Dict[str, Any] = {}
    if sweep_config["embedding_method"] == "svd_ppmi":
        extra_cfg["svd_sources"] = train_dataset.encoded_sources

    model_cfg = Seq2SeqConfig(
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
        **extra_cfg,
    )
    model = Seq2Seq(model_cfg)

    # ─── Wrap in DataParallel if multiple GPUs specified ─────────────────
    if device == "cuda" and len(static_args.gpu_ids) > 1:
        model = torch.nn.DataParallel(model)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=sweep_config["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=model_cfg.pad_index)

    # ─── Unique run name ─────────────────────────────
    run_name = (
        f"emb_method:{sweep_config['embedding_method']}|emb_size:{sweep_config['embedding_size']}|"
        f"cell:{sweep_config['cell']}|hid_size:{sweep_config['hidden_size']}|"
        f"enc:{sweep_config['encoder_layers']}|dec:{sweep_config['decoder_layers']}|"
        f"dr:{sweep_config['dropout']}|lr:{sweep_config['learning_rate']}|"
        f"bs:{sweep_config['batch_size']}|tf:{sweep_config['teacher_forcing']}|"
        f"ep:{sweep_config['epochs']}|beam:{sweep_config['beam_size']}|"
        f"attest:{sweep_config['use_attestations']}"
    )
    wandb.run.name = run_name
    wandb.run.tags = [static_args.wandb_run_tag]

    # ─── Training loop ───────────────────────────────────────────────
    for epoch in range(1, sweep_config["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, sweep_config["teacher_forcing"]
        )
        dev_loss = eval_epoch(model, dev_loader, loss_fn, device)

        # Log both raw loss and perplexity
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_perplexity": math.exp(train_loss),
            "dev_loss":   dev_loss,
            "dev_perplexity": math.exp(dev_loss),
        })


    # ─── Compute and log sequence-level accuracies ─────────────────────
    # Unwrap from DataParallel if needed for decoding
    inference_model = model.module if hasattr(model, "module") else model
    beam = sweep_config.get("beam_size", 1)

    # training_accuracy = compute_sequence_accuracy(inference_model, train_dataset, device, beam_size=1)
    dev_accuracy      = compute_sequence_accuracy(inference_model, dev_dataset,   device, beam_size=beam)
    # test_accuracy     = compute_sequence_accuracy(inference_model, test_dataset,  device, beam_size=beam)

    wandb.log({
        "dev_accuracy":   dev_accuracy,
    })

    # ─── Qualitative beam-search samples ─────────────────────────────
    print(f"\nSample dev-set translations (beam_size={beam}):")
    model.eval()

    with torch.no_grad():
        for i in range(5):
            src_ids, tgt_ids = dev_dataset[i]  # dev_dataset returns (src, tgt)
            src_len = len(src_ids)
            src_tensor = torch.tensor([src_ids], device=device)
            len_tensor = torch.tensor([src_len], device=device)

            # always use beam-search (beam_size may be 1)
            pred_ids = inference_model.beam_search_decode(
                src_tensor,
                len_tensor,
                beam_size=beam,
                max_len=30
            )[0].tolist()

            romanized = src_vocab.decode(src_ids)
            gold       = tgt_vocab.decode(tgt_ids[1:])  # skip <sos>
            prediction = tgt_vocab.decode(pred_ids[1:])  # skip <sos>
            print(f"{romanized:15} → {prediction:15} (gold: {gold})")


# ────────────── 3. Sweep & CLI plumbing ──────────────
def main():
    parser = argparse.ArgumentParser(
        description="Launch or run a W&B sweep for Q2."
    )
    parser.add_argument(
        "--mode", choices=["sweep", "single"], required=True,
        help="'sweep' to create & launch the sweep; 'single' for debug run"
    )
    parser.add_argument(
        "--sweep_config", type=str, default="sweep_config.yaml",
        help="YAML filename under ./configs/ defining the sweep space"
    )
    parser.add_argument(
        "--wandb_project", type=str, required=True, help="W&B project name"
    )
    parser.add_argument(
        "--wandb_run_tag", type=str, default="baseline",
        help="Static tag added to every W&B run"
    )
    parser.add_argument(
        "--gpu_ids", type=int, nargs="+", default=[0],
        help="CUDA device IDs to use (e.g. --gpu_ids 0 1 for two GPUs)"
    )
    parser.add_argument(
        "--train_tsv", type=str, required=True, help="Path to train TSV"
    )
    parser.add_argument(
        "--dev_tsv",   type=str, required=True, help="Path to dev TSV"
    )
    parser.add_argument(
        "--test_tsv",  type=str, required=True, help="Path to test TSV"
    )
    parser.add_argument(
        "--sweep_count", type=int, default=30,
        help="Number of sweep runs to launch if mode==sweep"
    )
    args = parser.parse_args()

    # Load sweep_config YAML
    project_root = Path(__file__).resolve().parent
    sweep_yaml   = get_configs(project_root, args.sweep_config)

    if args.mode == "sweep":
        # ─── Prepare sweep definition ─────────────────────────────────
        sweep_yaml.setdefault("method", "bayes")
        sweep_yaml["program"]    = Path(__file__).name
        sweep_yaml.setdefault(
            "metric", {"name": "dev_perplexity", "goal": "minimize"}
        )
        sweep_yaml.setdefault("parameters", {})
        sweep_yaml["run_cap"]    = args.sweep_count

        # ─── Register the sweep with W&B ───────────────────────────────
        sweep_id = wandb.sweep(
            sweep=sweep_yaml,
            project=args.wandb_project
        )
        print(f"Registered sweep with id: {sweep_id}")

        # ─── Define the function the agent will call for each trial ────
        def sweep_train():
            # start a new W&B run for this trial
            run = wandb.init(
                project=args.wandb_project
            )
            # now wandb.config is initialized, so we can read it
            run_single_training(dict(run.config), args)
            run.finish()

        # ─── Launch W&B agent in-process ───────────────────────────────
        print(f"Launching W&B agent for sweep {sweep_id}, count={args.sweep_count}")
        wandb.agent(
            sweep_id,
            function=sweep_train,
            count=args.sweep_count
        )

    else:
        # ─── Single run for debugging ───────────────────────────────────
        with wandb.init(
            project=args.wandb_project,
            config=sweep_yaml.get("parameters", {}),
        ) as run:
            run.config.update(
                {
                    "epochs": 3,
                    "batch_size": 64,
                    "embedding_size": 128,
                    "hidden_size": 256,
                    "encoder_layers": 2,
                    "decoder_layers": 2,
                    "cell": "LSTM",
                    "dropout": 0.1,
                    "learning_rate": 1e-3,
                    "teacher_forcing": 0.5,
                    "embedding_method": "learned",
                    "use_attestations": False,
                    "beam_size": 1,
                },
                allow_val_change=True,
            )
            run_single_training(dict(run.config), args)


if __name__ == "__main__":
    main()
