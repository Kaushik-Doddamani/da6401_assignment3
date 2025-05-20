#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q5.b: Train (with early stopping) & evaluate best attention‐augmented Seq2Seq model on the Dakshina Hindi test set.

This script:
  1. Loads train/dev/test lexicons and builds/uses the same vocabularies.
  2. Constructs the Seq2SeqAttention model with your best hyperparameters.
  3. If no checkpoint exists:
       - Trains with early stopping on dev cross‐entropy loss (patience=3).
       - Saves the best‐so‐far model to --checkpoint.
     Otherwise loads the existing checkpoint.
  4. Runs beam-search decoding on the test set, computes exact‐match accuracy.
  5. Saves all (source, gold, prediction) triples under `predictions_attention/` as TSV and CSV.
  6. Samples 20 predictions, builds a colored table figure, saves it, and logs it to W&B.
  7. Selects 10 random test examples, computes their greedy attention heatmaps, and
     plots them in a 3×4 grid, saving & logging the figure to W&B.

Usage example:

    python solution_5b.py \
      --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
      --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv \
      --test_tsv  ./lexicons/hi.translit.sampled.test.tsv \
      --checkpoint ./checkpoints/best_attention.pt \
      --output_dir predictions_attention \
      --gpu_ids 0 \
      --wandb_project transliteration \
      --wandb_run_name solution_5b_run \
      --wandb_run_tag solution_5b
"""
from __future__ import annotations
import argparse
import os
import math
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader
import pandas as pd
import wandb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Q1 data & training utilities
from solution_1 import DakshinaLexicon, collate_batch, train_epoch, eval_epoch
# Q5 attention model
from solution_5_model import Seq2SeqAttentionConfig, Seq2SeqAttention, _align_hidden_state

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

# ─────────────────────────────────────────────────────────────────────
# Replace these with best attention‐model hyperparameters:
best = {
    "batch_size":        128,
    "beam_size":         5,
    "cell_type":       "GRU",
    "decoder_layers":     2,
    "dropout":          0.2,
    "embedding_method":"svd_ppmi",
    "embedding_size":   64,
    "encoder_layers":     1,
    "hidden_size":      512,
    "learning_rate":   0.0006899910999897612,
    "teacher_forcing":  0.5,
    "use_attestations": True,
    # early stopping patience
    "patience":          3,
    # number of training epochs to try
    "epochs":           25,
}
# ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Q5.b: Train (with early stopping) & evaluate best attention model"
    )
    p.add_argument("--train_tsv",     type=str, required=True, help="Path to train lexicon TSV")
    p.add_argument("--dev_tsv",       type=str, required=True, help="Path to dev lexicon TSV")
    p.add_argument("--test_tsv",      type=str, required=True, help="Path to test lexicon TSV")
    p.add_argument("--checkpoint",    type=str, required=True, help="Path to save/load model checkpoint")
    p.add_argument("--output_dir",    type=str, default="predictions_attention",
                   help="Directory to write predictions.tsv/csv")
    p.add_argument("--gpu_ids",       type=int, nargs="+", default=[0], help="CUDA device IDs")
    p.add_argument("--wandb_project", type=str, default=None, help="W&B project name")
    p.add_argument("--wandb_run_name",type=str, default=None, help="W&B run name")
    p.add_argument("--wandb_run_tag", type=str, default="solution_5b", help="W&B run tag")
    return p.parse_args()

def get_attention_heatmap(
    model: Seq2SeqAttention,
    src_ids: list[int],
    src_lens: torch.Tensor,
    src_vocab: DakshinaLexicon.src_vocab.__class__,
    tgt_vocab: DakshinaLexicon.tgt_vocab.__class__,
    device: torch.device,
    max_len: int = 50
) -> tuple[list[int], list[list[float]]]:
    """
    Run greedy decode step-by-step, collecting the attention weights at each step.
    Returns (predicted_ids, attention_weights_matrix) where matrix[t][i] is
    the attention weight at decoder time t for encoder position i.
    """
    model.eval()
    # prepare tensors
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    enc_outputs, enc_hidden = model.encoder(src_tensor, src_lens)
    enc_mask = (src_tensor != model.cfg.pad_index).to(device)
    dec_hidden = _align_hidden_state(enc_hidden, model.cfg.decoder_layers)
    # start with <sos>
    dec_input = torch.full((1,1), model.cfg.sos_index, dtype=torch.long, device=device)

    predicted_ids: list[int] = []
    attention_weights: list[list[float]] = []

    for _ in range(max_len):
        logits, dec_hidden, align = model.decoder(
            dec_input, dec_hidden, enc_outputs, enc_mask
        )
        # align: (1, T_src) → record
        alignment = align.squeeze(0).tolist()
        attention_weights.append(alignment)
        # pick argmax
        dec_input = logits.argmax(-1)  # (1,1)
        next_id = dec_input.item()
        if next_id == model.cfg.eos_index:
            break
        predicted_ids.append(next_id)

    return predicted_ids, attention_weights


def main():
    args = parse_args()

    # ─── Pin GPUs & select device ────────────────────────────────────
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu_ids))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Init WandB if requested ─────────────────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=[args.wandb_run_tag],
            config=best
        )

    # ─── Ensure output directories exist ──────────────────────────────
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Build vocab from train set ──────────────────────────────────
    train_ds = DakshinaLexicon(
        args.train_tsv,
        build_vocabs=True,
        use_attestations=best["use_attestations"]
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # ─── Prepare dev/test datasets ───────────────────────────────────
    collate_fn = lambda batch: collate_batch(batch, pad_id=src_vocab.stoi["<pad>"])
    train_loader = DataLoader(train_ds, batch_size=best["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    dev_ds = DakshinaLexicon(args.dev_tsv, src_vocab, tgt_vocab)
    dev_loader = DataLoader(dev_ds, batch_size=best["batch_size"],
                            shuffle=False, collate_fn=collate_fn)
    test_ds = DakshinaLexicon(args.test_tsv, src_vocab, tgt_vocab)
    test_loader = DataLoader(test_ds, batch_size=best["batch_size"],
                             shuffle=False, collate_fn=collate_fn)

    # ─── Build the Seq2SeqAttention model ────────────────────────────
    extra = {}
    if best["embedding_method"] == "svd_ppmi":
        extra["svd_sources"] = train_ds.encoded_sources

    cfg = Seq2SeqAttentionConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=best["embedding_size"],
        hidden_dim=best["hidden_size"],
        encoder_layers=best["encoder_layers"],
        decoder_layers=best["decoder_layers"],
        cell_type=best["cell_type"],
        dropout=best["dropout"],
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=best["embedding_method"],
        **extra
    )
    model = Seq2SeqAttention(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=best["learning_rate"])
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    # ─── Train with early stopping if checkpoint missing ─────────────
    if not ckpt_path.exists():
        print("No checkpoint found; starting training with early stopping...")
        best_dev_loss = float("inf")
        no_improve = 0

        for epoch in range(1, best["epochs"] + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, loss_fn,
                device, teacher_forcing=best["teacher_forcing"]
            )
            dev_loss = eval_epoch(model, dev_loader, loss_fn, device)
            train_ppl = math.exp(train_loss)
            dev_ppl   = math.exp(dev_loss)

            print(f"Epoch {epoch:02d} | "
                  f"train_loss={train_loss:.4f} ppl={train_ppl:.2f} | "
                  f"dev_loss={dev_loss:.4f} ppl={dev_ppl:.2f}")

            if use_wandb:
                wandb.log({
                    "Q5b_epoch":        epoch,
                    "Q5b_train_loss":   train_loss,
                    "Q5b_train_ppl":    train_ppl,
                    "Q5b_dev_loss":     dev_loss,
                    "Q5b_dev_ppl":      dev_ppl,
                })

            # early stopping on dev_loss
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict()}, str(ckpt_path))
                print("  ↳ dev improved; checkpoint saved.")
            else:
                no_improve += 1
                print(f"  ↳ no improvement for {no_improve} epoch(s)")
                if no_improve >= best["patience"]:
                    print("Early stopping.")
                    break

        print("Training complete.\n")
    else:
        print(f"Found existing checkpoint at {ckpt_path}; skipping training.\n")

    # ─── Load the best checkpoint ─────────────────────────────────────
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ─── Decode test set & compute exact‐match accuracy ──────────────
    total, correct = 0, 0
    predictions = []

    with torch.no_grad():
        for src_batch, src_lens, tgt_batch in test_loader:
            for i in range(src_batch.size(0)):
                total += 1
                # Trim to true length to fix mask/score mismatch
                length = src_lens[i].item()
                s = src_batch[i, :length].unsqueeze(0).to(device)
                l = torch.tensor([length], device=device)

                pred_ids = model.beam_search_decode(
                    s, l,
                    beam_size=best["beam_size"],
                    max_len=50
                )[0].tolist()

                # drop leading <sos> if present
                if pred_ids and pred_ids[0] == tgt_vocab.stoi["<sos>"]:
                    pred_ids = pred_ids[1:]

                src_str  = src_vocab.decode(s[0].tolist())
                gold_str = tgt_vocab.decode(tgt_batch[i].tolist()[1:])
                pred_str = tgt_vocab.decode(pred_ids)

                if pred_str == gold_str:
                    correct += 1
                predictions.append((src_str, gold_str, pred_str))

    accuracy = correct / total * 100
    print(f"\nTest exact‐match accuracy: {accuracy:.2f}% ({correct}/{total})\n")
    if use_wandb:
        wandb.log({"Q5b_test_accuracy": accuracy})

    # ─── Save all predictions ─────────────────────────────────────────
    df = pd.DataFrame(predictions, columns=["source", "target", "prediction"])
    df.to_csv(output_path / "predictions.tsv", sep="\t", index=False)
    df.to_csv(output_path / "predictions.csv", index=False)
    print(f"Saved predictions → {output_path/'predictions.tsv'}, {output_path/'predictions.csv'}\n")

    # ─── Sample 20 and build colored table figure ─────────────────────
    sample_df = df.sample(20, random_state=42)
    # green for correct, red for wrong
    colors = [
        ["#c8e6c9" if row.target == row.prediction else "#ffcdd2"
         for _ in sample_df.columns]
        for _, row in sample_df.iterrows()
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    tbl = ax.table(
        cellText=sample_df.values.tolist(),
        colLabels=sample_df.columns.tolist(),
        cellColours=colors,
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2)
    figure_path = output_path / "sample_predictions.png"
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved sample predictions figure to {figure_path}")
    if use_wandb:
        wandb.log({"Q5b_sample_table": wandb.Image(str(figure_path))})
    
    # ─── Attention heatmaps for 10 random test examples ──────────────
    random.seed(42)
    indices = random.sample(range(len(test_ds)), 10)
    heatmaps = []
    src_lists, pred_lists = [], []
    for idx in indices:
        src_ids, tgt_ids = test_ds[idx]
        _, attn = get_attention_heatmap(
            model, src_ids, torch.tensor([len(src_ids)], device=device),
            src_vocab, tgt_vocab, device
        )
        heatmaps.append(attn)
        src_lists.append(src_ids)
        # we drop <sos> from predictions
        pred_ids = [i for i in attn and []]  # placeholder

    # actually regenerate preds & char lists
    preds_and_attn = []
    for src_ids in src_lists:
        pred_ids, attn = get_attention_heatmap(
            model, src_ids, torch.tensor([len(src_ids)], device=device),
            src_vocab, tgt_vocab, device
        )
        # drop any leading <sos>
        if pred_ids and pred_ids[0] == tgt_vocab.stoi["<sos>"]:
            pred_ids = pred_ids[1:]
            attn = attn[1:]
        preds_and_attn.append((src_ids, pred_ids, attn))

    # plot in a 3×4 grid
    n = len(preds_and_attn)
    cols = 3
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3))
    axes = axes.flatten()
    for i, (src_ids, pred_ids, attn) in enumerate(preds_and_attn):
        ax = axes[i]
        im = ax.imshow(attn, aspect="auto", origin="lower")
        # x-axis: source chars
        sx = [src_vocab.itos[id] for id in src_ids]
        ax.set_xticks(range(len(sx)))
        ax.set_xticklabels(sx, rotation=90, fontsize=8)
        # y-axis: predicted chars
        py = [tgt_vocab.itos[id] for id in pred_ids]
        ax.set_yticks(range(len(py)))
        ax.set_yticklabels(py, fontsize=8)
        ax.set_xlabel("Source")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Example {i+1}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # turn off any extra axes
    for ax in axes[n:]:
        ax.axis("off")
    plt.tight_layout()
    heatmap_path = output_path/"attention_heatmaps.png"
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved attention heatmaps to {heatmap_path}")
    if use_wandb:
        wandb.log({"Q5b_attention_heatmaps": wandb.Image(str(heatmap_path))})
        wandb.finish()

if __name__ == "__main__":
    main()
