# -*- coding: utf-8 -*-
"""Q2 – W&B sweep launcher for Hindi transliteration (Dakshina).

Major changes vs. the first draft
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Sweep configuration is stored in an external YAML file** (e.g. *hi_sweep.yaml*)
   and loaded via the helper `get_configs`.
2. CLI now accepts (and passes through to W&B) **`wandb_project`, `wandb_entity`,
   `wandb_run_tag`, `gpu_id`, `sweep_count`, and custom TSV paths**.
3. Every run gets a **unique, human‑readable name** summarising the active
   hyper‑parameters, and the requested tag is attached to `wandb.run.tags`.

The rest of the logic (imports from *solution_1.py*, training loop, metrics) is
unchanged for clarity.
"""

from __future__ import annotations

# ───────────────────── Imports ─────────────────────
import argparse
import os
import math
from pathlib import Path
from typing import Dict, Any

import yaml  # for reading the sweep YAML
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from solution_1 import (
    CharVocabulary,
    DakshinaLexicon,
    collate_batch,
    Seq2SeqConfig,
    Seq2Seq,
    train_epoch,
    eval_epoch,
)

# ───────────────────── Utility helpers ─────────────────────

def get_configs(project_root: str | Path, config_filename: str) -> Dict[str, Any]:
    """Read YAML file from ``<project_root>/config/<config_filename>``."""
    path = Path(project_root) / "config" / config_filename
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg

# Character‑level exact‑match accuracy --------------------------------------

def character_accuracy(model: Seq2Seq, loader: DataLoader, device: str) -> float:
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for src, src_len, tgt in loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            preds = model.greedy_decode(src, src_len, max_len=tgt.size(1))
            gold  = tgt[:, 1:preds.size(1) + 1]
            mask  = gold != model.cfg.pad_index
            match = (preds == gold) | (~mask)
            correct += match.all(dim=1).sum().item()
            total   += src.size(0)
    return correct / total if total else 0.0

# Run‑name helper -----------------------------------------------------------

def make_run_name(cfg: Dict[str, Any]) -> str:
    """Readable unique run name built from key hyper‑parameters."""
    parts = [
        f"Emb:{cfg['embedding_method']}-{cfg['embedding_size']}",
        f"Hid:{cfg['hidden_size']}",
        f"Cell:{cfg['cell']}",
        f"Depth:{cfg['encoder_layers']}enc/{cfg['decoder_layers']}dec",
        f"DO:{cfg['dropout']}",
        f"TF:{cfg['teacher_forcing']}",
        f"LR:{cfg['lr']}",
    ]
    return " | ".join(parts)

# ───────────────────── Single run function ─────────────────────

def run_single_experiment(config: Dict[str, Any]):
    """One W&B run executed by an agent or in "single" mode."""

    run = wandb.init(
        config=config,
        project=config.get("wandb_project"),
        entity=config.get("wandb_entity"),
        reinit=True,
    )
    cfg = wandb.config  # convenience

    # Assign human‑readable run name + tag
    wandb.run.name = make_run_name(cfg)
    if cfg.get("wandb_run_tag"):
        wandb.run.tags = [cfg["wandb_run_tag"]]

    device = cfg.device

    # ------------ Data -------------
    train_ds = DakshinaLexicon(cfg.train_tsv, build_vocabs=True, use_attestations=cfg.use_attestations)
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab
    dev_ds  = DakshinaLexicon(cfg.dev_tsv,  src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(cfg.test_tsv, src_vocab, tgt_vocab)

    collate_fn = lambda b: collate_batch(b, pad_id=src_vocab.stoi["<pad>"])

    if cfg.use_attestations:
        sampler = WeightedRandomSampler(train_ds.example_counts, len(train_ds), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)

    dev_loader  = DataLoader(dev_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    # ------------ Model -------------
    extra_cfg = {}
    if cfg.embedding_method == "svd_ppmi":
        extra_cfg["svd_sources"] = train_ds.encoded_sources

    model_cfg = Seq2SeqConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=cfg.embedding_size,
        hidden_dim=cfg.hidden_size,
        encoder_layers=cfg.encoder_layers,
        decoder_layers=cfg.decoder_layers,
        cell_type=cfg.cell,
        dropout=cfg.dropout,
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=cfg.embedding_method,
        **extra_cfg
    )
    model = Seq2Seq(model_cfg).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=model_cfg.pad_index)

    # ------------ Training -----------
    for epoch in range(1, cfg.epochs + 1):
        train_ce = train_epoch(model, train_loader, optimizer, loss_fn, device, cfg.teacher_forcing)
        dev_ce   = eval_epoch(model, dev_loader, loss_fn, device)
        wandb.log({"epoch": epoch, "train_ppl": math.exp(train_ce), "dev_ppl": math.exp(dev_ce)})

    # ------------ Final metrics ------
    test_ce  = eval_epoch(model, test_loader, loss_fn, device)
    dev_acc  = character_accuracy(model, dev_loader, device)
    test_acc = character_accuracy(model, test_loader, device)

    wandb.log({
        "dev_ppl_final": math.exp(dev_ce),
        "test_ppl_final": math.exp(test_ce),
        "dev_acc": dev_acc,
        "test_acc": test_acc,
    })

    run.finish()

# ───────────────────── CLI / sweep launcher ─────────────────────

def main():
    parser = argparse.ArgumentParser(description="Q2 sweep runner / launcher")
    parser.add_argument("--mode", choices=["sweep", "single"], default="sweep")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--config_file", default="hi_sweep.yaml")
    parser.add_argument("--wandb_project", default="transliteration")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_run_tag", default="Q2")
    parser.add_argument("--gpu_id", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--sweep_count", type=int, default=None, help="Max runs in the sweep")
    # Allow overriding TSV paths
    parser.add_argument("--train_tsv", default=str(Path(DEFAULT_TRAIN)))
    parser.add_argument("--dev_tsv",   default=str(Path(DEFAULT_DEV)))
    parser.add_argument("--test_tsv",  default=str(Path(DEFAULT_TEST)))
    args = parser.parse_args()

    # GPU selection ---------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Load sweep YAML -------------------------------------------------------
    sweep_cfg = get_configs(args.project_root, args.config_file)

    # Inject CLI overrides / fixed params
    sweep_cfg.setdefault("parameters", {})
    sweep_cfg["parameters"].update({
        "train_tsv":       {"value": args.train_tsv},
        "dev_tsv":         {"value": args.dev_tsv},
        "test_tsv":        {"value": args.test_tsv},
        "device":          {"value": "cuda" if torch.cuda.is_available() else "cpu"},
        "wandb_project":   {"value": args.wandb_project},
        "wandb_entity":    {"value": args.wandb_entity},
        "wandb_run_tag":   {"value": args.wandb_run_tag},
    })

    if args.mode == "single":
        # Build default config dict (pick first value from each search space)
        default_cfg = {k: v.get("value", v.get("values")[0]) for k, v in sweep_cfg["parameters"].items()}
        run_single_experiment(default_cfg)
    else:
        sweep_id = wandb.sweep(sweep_cfg, project=args.wandb_project, entity=args.wandb_entity)
        print("Created sweep:", sweep_id)
        cmd = f"wandb agent {args.wandb_project}/{sweep_id}"
        if args.sweep_count:
            cmd += f" --count {args.sweep_count}"
        print("Run agents with:", cmd)


if __name__ == "__main__":
    main()
