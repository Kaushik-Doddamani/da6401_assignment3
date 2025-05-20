#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q4: Train (if needed) & evaluate best model on Dakshina transliteration test set

Usage example:
    # 1) Train+evaluate (if no checkpoint exists) or just evaluate:
    python solution_4.py \
      --train_tsv ./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv \
      --dev_tsv   ./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv \
      --test_tsv  ./dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv \
      --checkpoint ./checkpoints/best_seq2seq.pt \
      --output_dir predictions_vanilla \
      --gpu_ids 0 1 \
      --wandb_project transliteration \
      --wandb_run_name solution_4_run \
      --wandb_run_tag solution_4

    # 2) If you don't want WandB logging, simply omit --wandb_project / --wandb_run_tag
Outputs:
  - Trains with early stopping (patience=3) if checkpoint not found.
  - Saves checkpoint to `--checkpoint`.
  - Prints exact‐match accuracy on test set.
  - Saves all predictions in both TSV and CSV under `--output_dir`.
  - Builds a colored table of 10 samples, logs it to WandB, and prints a markdown table.
"""

from __future__ import annotations
import argparse
import os
import math
import warnings
from pathlib import Path
import urllib.request

import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import wandb
from wcwidth import wcswidth

# Re-use Q1 code unchanged:
from solution_1 import (
    DakshinaLexicon,
    collate_batch,
    Seq2SeqConfig,
    Seq2Seq,
    train_epoch,
    eval_epoch,
)

# ────────────────────────────────
# Download & register Devanagari font
# ────────────────────────────────
font_dir = Path.cwd() / "fonts"
font_dir.mkdir(exist_ok=True)
font_path = font_dir / "Hind-Regular.ttf"

if not font_path.exists():
    print("Font file not found. Please ensure 'Hind-Regular.ttf' is in the 'fonts' directory which"
    "can be downloaded from Google Fonts. https://www.cufonfonts.com/font/noto-sans-devanagari")
else:
    fm.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Hind"
    plt.style.use("seaborn-v0_8-pastel")
    print("Font loaded and matplotlib configured.")

# ────────────────────────────────
# GLOBAL: best hyperparameters
# ────────────────────────────────
best = {
    "batch_size":       128,
    "beam_size":        5,
    "cell":             "LSTM",
    "decoder_layers":   3,
    "dropout":          0.1,
    "embedding_method": "learned",
    "embedding_size":   64,
    "encoder_layers":   3,
    "epochs":           5,
    "hidden_size":      512,
    "learning_rate":    0.0017737018583641314,
    "teacher_forcing":  0.3,
    "use_attestations": False,
    # early stopping patience
    "patience":         3,
}

def main():
    parser = argparse.ArgumentParser(
        description="Q4: Train (if needed) and apply best Seq2Seq model to test set"
    )
    parser.add_argument("--train_tsv",   type=str, required=True,
                        help="Path to training lexicon TSV (for vocabulary)")
    parser.add_argument("--dev_tsv",     type=str, required=True,
                        help="Path to development lexicon TSV (for early stopping)")
    parser.add_argument("--test_tsv",    type=str, required=True,
                        help="Path to test lexicon TSV")
    parser.add_argument("--checkpoint",  type=str, required=True,
                        help="Path to save/load model checkpoint (.pt file)")
    parser.add_argument("--output_dir",  type=str, default="predictions_vanilla",
                        help="Directory to write predictions.tsv and predictions.csv")
    parser.add_argument("--gpu_ids",     type=int, nargs="+", default=[0],
                        help="CUDA device IDs to use (e.g. 0 1).")
    parser.add_argument("--wandb_project",  type=str, default=None,
                        help="W&B project name (omit to disable WandB)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name (omit to disable WandB)")
    parser.add_argument("--wandb_run_tag",  type=str, default="baseline",
                        help="Static tag added to every W&B run")
    args = parser.parse_args()

    # ─── 0. Pin GPUs ───────────────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpu_ids)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── Initialize WandB if requested ─────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=[args.wandb_run_tag],
            config=best
        )

    # ensure output dirs exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)

    # ─── 1. Build vocabulary from train set ───────────────────────────
    train_ds = DakshinaLexicon(
        args.train_tsv,
        build_vocabs=True,
        use_attestations=best["use_attestations"]
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ─── 2. Prepare dev/test loaders ─────────────────────────────────
    dev_ds  = DakshinaLexicon(args.dev_tsv,  src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(args.test_tsv, src_vocab, tgt_vocab)
    collate_fn = lambda batch: collate_batch(batch, pad_id=src_vocab.stoi["<pad>"])

    dev_loader = DataLoader(dev_ds,  batch_size=best["batch_size"],
                            shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=best["batch_size"],
                             shuffle=False, collate_fn=collate_fn)

    # ─── 3. Build model & optimizer ───────────────────────────────────
    cfg = Seq2SeqConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=best["embedding_size"],
        hidden_dim=best["hidden_size"],
        encoder_layers=best["encoder_layers"],
        decoder_layers=best["decoder_layers"],
        cell_type=best["cell"],
        dropout=best["dropout"],
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=best["embedding_method"],
    )
    model     = Seq2Seq(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["learning_rate"])
    loss_fn   = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    # ─── 4. Train w/ early stopping if no checkpoint ─────────────────
    if not os.path.exists(args.checkpoint):
        print("No checkpoint found; starting training with early stopping...")
        best_dev, no_improve = float("inf"), 0
        train_loader = DataLoader(
            train_ds,
            batch_size=best["batch_size"],
            shuffle=True,
            collate_fn=collate_fn
        )
        for epoch in range(1, best["epochs"] + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn,
                device, best["teacher_forcing"]
            )
            dev_loss = eval_epoch(model, dev_loader, loss_fn, device)
            train_ppl = math.exp(train_loss)
            dev_ppl   = math.exp(dev_loss)
            print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} ppl={train_ppl:.2f} "
                  f"| dev_loss={dev_loss:.4f} ppl={dev_ppl:.2f}")
            if use_wandb:
                wandb.log({
                    "Q4_epoch": epoch,
                    "Q4_train_loss": train_loss,
                    "Q4_train_ppl": train_ppl,
                    "Q4_dev_loss": dev_loss,
                    "Q4_dev_ppl": dev_ppl,
                })
            # early stopping
            if dev_loss < best_dev:
                best_dev, no_improve = dev_loss, 0
                torch.save({"model_state_dict": model.state_dict()}, args.checkpoint)
                print("  ↳ dev improved; checkpoint saved.")
            else:
                no_improve += 1
                print(f"  ↳ no improvement for {no_improve} epoch(s)")
                if no_improve >= best["patience"]:
                    print("Early stopping.")
                    break
        print("Training finished.\n")
    else:
        print(f"Found checkpoint at {args.checkpoint}; skipping training.\n")

    # ─── 5. Load checkpoint & evaluate on test set ───────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    total, correct = 0, 0
    all_preds = []  # list of (source, gold, prediction)
    with torch.no_grad():
        for src_batch, src_lens, tgt_batch in test_loader:
            src_batch = src_batch.to(device)
            src_lens  = src_lens.to(device)
            for i in range(src_batch.size(0)):
                total += 1
                single_src   = src_batch[i].unsqueeze(0)
                single_len   = src_lens[i].unsqueeze(0)
                gold_ids     = tgt_batch[i].tolist()[1:]  # skip <sos>
                pred_ids     = model.beam_search_decode(
                                    single_src, single_len,
                                    beam_size=best["beam_size"], max_len=50
                               )[0].tolist()
                src_str      = src_vocab.decode(single_src[0].tolist())
                gold_str     = tgt_vocab.decode(gold_ids)
                pred_str     = tgt_vocab.decode(pred_ids)
                if pred_str == gold_str:
                    correct += 1
                all_preds.append((src_str, gold_str, pred_str))

    test_loss = eval_epoch(model, test_loader, loss_fn, device)
    test_ppl  = math.exp(test_loss)
    accuracy  = correct / total * 100
    print(f"\nTest loss={test_loss:.4f} ppl={test_ppl:.2f} "
          f"accuracy={accuracy:.2f}% ({correct}/{total})\n")
    if use_wandb:
        wandb.log({
            "Q4_test_loss": test_loss,
            "Q4_test_ppl": test_ppl,
            "Q4_test_accuracy": accuracy,
        })

    # ─── 6. Save predictions as TSV & CSV ────────────────────────────
    preds_df = pd.DataFrame(all_preds, columns=["source", "target", "prediction"])
    tsv_path = Path(args.output_dir) / "predictions.tsv"
    csv_path = Path(args.output_dir) / "predictions.csv"
    preds_df.to_csv(tsv_path, sep="\t", index=False)
    preds_df.to_csv(csv_path,            index=False)
    print(f"Saved all predictions → {tsv_path}, {csv_path}\n")

    # ─── 7. Creative colored table of 20 examples ─────────────────────
    sample_df = preds_df.sample(20, random_state=42)
    colors = [["#c8e6c9" if row["target"]==row["prediction"] else "#ffcdd2"
               for _ in sample_df.columns] for _, row in sample_df.iterrows()]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    tbl = ax.table(cellText=sample_df.values.tolist(),
                   colLabels=sample_df.columns.tolist(),
                   cellColours=colors,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 2)
    if use_wandb:
        wandb.log({"Q4_sample_table": wandb.Image(fig)})
    figure_path = Path(args.output_dir) / "sample_predictions.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample predictions figure to {figure_path}\n")

    # ─── 8. Print markdown table ───────────────────────────────────────
    # Print a properly aligned markdown table:
    cols = ["source", "target", "prediction"]
    # compute the max display width of each column
    col_widths = {
        col: max(
            wcswidth(str(val)) for val in ([col] + sample_df[col].astype(str).tolist())
        )
        for col in cols
    }

    # header line
    header = "| " + " | ".join(
        col.center(col_widths[col]) for col in cols
    ) + " |"
    sep = "|-" + "-|-".join("-" * col_widths[col] for col in cols) + "-|"
    print(header)
    print(sep)

    for _, row in sample_df.iterrows():
        mark = "✅" if row["target"] == row["prediction"] else "❌"
        line = "| " + " | ".join(
            # pad each cell on the right so it fills its column’s display width
            row[col].ljust(col_widths[col] + (len(row[col]) - wcswidth(row[col])))
            for col in cols
        ) + f" | {mark}"
        print(line)

    # print("| source        | target         | prediction     |")
    # print("|---------------|----------------|----------------|")
    # for src, target, pred in sample_df.values.tolist():
    #     mark = "✅" if target==pred else "❌"
    #     print(f"| {src:13} | {target:14} | {pred:14} | {mark}")

    if use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()