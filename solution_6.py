#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q6: Interactive “Connectivity” visualization for your attention-augmented Seq2Seq model

This script lets you:
  1. Load your trained Seq2SeqAttention checkpoint.
  2. Sample N examples from the test set.
  3. For each example, run a greedy decode while recording the attention weights
     at each decoder timestep.
  4. Save a standalone HTML that displays, for each example:
       - The source characters along the x-axis.
       - The predicted output characters along the y-axis.
       - A Plotly heatmap of attention weights.
     You can hover over any cell to see the exact weight—and thus see “connectivity.”
  5. (New!) Log the final HTML to WandB so you can browse it in your project.

Usage (as a script):

    python solution_6.py \
      --checkpoint ./checkpoints/best_attention.pt \
      --train_tsv   ./lexicons/hi.translit.sampled.train.tsv \
      --test_tsv    ./lexicons/hi.translit.sampled.test.tsv \
      --n_examples  5 \
      --output_html connectivity.html \
      --wandb_project DA6401_Intro_to_DeepLearning_Assignment_3 \
      --wandb_run_name solution_6_run \
      --wandb_run_tag  solution_6
"""

from __future__ import annotations
import argparse
import os
import random
from pathlib import Path

import torch
import pandas as pd
import plotly.graph_objects as go
import wandb

# your seq2seq imports:
from solution_1 import DakshinaLexicon
from solution_5_model import Seq2SeqAttentionConfig, Seq2SeqAttention, _align_hidden_state

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters for your best attention model (must match how you trained it)
best = {
    "batch_size":        128,
    "beam_size":         3,
    "cell_type":         "GRU",
    "decoder_layers":    3,
    "dropout":           0.2,
    "embedding_method":  "svd_ppmi",
    "embedding_size":    64,
    "encoder_layers":    1,
    "hidden_size":       512,
    "learning_rate":     0.0006899910999897612,
    "teacher_forcing":   0.5,
    "use_attestations":  True,
    # early stopping patience
    "patience":         3,
    # number of training epochs to try
    "epochs":          20,
}
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Q6: Visualize character-level attention connectivity interactively"
    )
    p.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to your trained Seq2SeqAttention .pt file"
    )
    p.add_argument(
        "--train_tsv", type=str, required=True,
        help="TSV of the TRAIN split (native\tr omanized\tcount) for building vocab"
    )
    p.add_argument(
        "--test_tsv", type=str, required=True,
        help="TSV of the TEST split (native\tr omanized\tcount)"
    )
    p.add_argument(
        "--n_examples", type=int, default=3,
        help="Number of random examples to visualize"
    )
    p.add_argument(
        "--output_html", type=str, default="connectivity.html",
        help="Path to save the standalone HTML with Plotly plots"
    )
    # WandB logging args
    p.add_argument(
        "--wandb_project", type=str, default=None,
        help="(optional) WandB project name to log this HTML"
    )
    p.add_argument(
        "--wandb_run_name", type=str, default=None,
        help="(optional) WandB run name"
    )
    p.add_argument(
        "--wandb_run_tag", type=str, default=None,
        help="(optional) WandB run tag"
    )
    return p.parse_args()

def load_model_and_vocab(
    ckpt_path: str,
    train_tsv: str,
    device: torch.device
) -> tuple[Seq2SeqAttention, DakshinaLexicon, DakshinaLexicon]:
    """
    Loads the checkpoint into a Seq2SeqAttention model,
    and builds the vocab from the TRAIN split so that src/tgt vocabs align.
    """
    # 1) Build the vocabulary from the true training split
    train_ds = DakshinaLexicon(
        train_tsv,
        build_vocabs=True,
        use_attestations=best["use_attestations"]
    )
    src_vocab = train_ds.src_vocab
    tgt_vocab = train_ds.tgt_vocab

    # 2) Peek at the checkpoint to detect how many decoder layers it actually has
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"]

    # Collect all weight keys of the form "decoder.rnn.weight_ih_l{n}"
    layer_indices = []
    for key in state_dict:
        if key.startswith("decoder.rnn.weight_ih_l"):
            # e.g. "decoder.rnn.weight_ih_l0", "decoder.rnn.weight_ih_l1", ...
            idx_str = key.split("decoder.rnn.weight_ih_l", 1)[1]
            try:
                layer_indices.append(int(idx_str))
            except ValueError:
                continue

    if layer_indices:
        # max index + 1 = number of layers
        actual_decoder_layers = max(layer_indices) + 1
        print(f"  ↳ Checkpoint has {actual_decoder_layers} decoder layers (detected)")
    else:
        actual_decoder_layers = best["decoder_layers"]
        print(f"  ↳ No decoder.rnn.weight_ih_l* keys found; defaulting to {actual_decoder_layers}")

    # 3) Construct exactly the same config you used for training,
    #    except use the detected number of decoder layers:
    cfg = Seq2SeqAttentionConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=best["embedding_size"],
        hidden_dim=best["hidden_size"],
        encoder_layers=best["encoder_layers"],
        decoder_layers=actual_decoder_layers,
        cell_type=best["cell_type"],
        dropout=best["dropout"],
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=best["embedding_method"],
        **({"svd_sources": train_ds.encoded_sources}
           if best["embedding_method"] == "svd_ppmi" else {})
    )

    # 4) Instantiate and load
    model = Seq2SeqAttention(cfg).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, src_vocab, tgt_vocab

def record_attention(
    model: Seq2SeqAttention,
    src_ids: list[int],
    src_vocab,
    tgt_vocab,
    device: torch.device,
    max_len: int = 50
) -> tuple[list[str], list[str], list[list[float]]]:
    """
    Greedy-decode `src_ids` with the model, capturing attention weights at each step.
    Returns:
      (source_chars, predicted_chars, attention_matrix)
    where
      attention_matrix[t][i] = score attending to src position i when predicting
                              output char at position t
    """
    # Prepare tensors
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_len    = torch.tensor([len(src_ids)], device=device)

    # Encoder: get outputs and initial hidden
    enc_outputs, enc_hidden = model.encoder(src_tensor, src_len)
    enc_mask = (src_tensor != model.cfg.pad_index).to(device)
    dec_hidden = _align_hidden_state(enc_hidden, model.cfg.decoder_layers)

    # Initialize decoder input (<sos>)
    dec_input = torch.full((1,1), model.cfg.sos_index, dtype=torch.long, device=device)
    predicted_ids: list[int] = []
    attentions: list[list[float]] = []

    # Step through decoder, capture alignments
    for _ in range(max_len):
        logits, dec_hidden, alignments = model.decoder(
            dec_input, dec_hidden, enc_outputs, enc_mask
        )
        # record attention over source
        attn = alignments.squeeze(0).tolist()
        attentions.append(attn)

        # next token
        dec_input = logits.argmax(-1)  # shape (1,1)
        next_id = dec_input.item()
        if next_id == model.cfg.eos_index:
            break
        predicted_ids.append(next_id)

    # Convert indices back to characters
    source_chars    = [src_vocab.itos[i] for i in src_ids]
    predicted_chars = [tgt_vocab.itos[i] for i in predicted_ids]
    return source_chars, predicted_chars, attentions

def make_plotly_figure(
    source_chars:    list[str],
    predicted_chars: list[str],
    attentions:      list[list[float]],
    title:           str
) -> go.Figure:
    """
    Builds a Plotly heatmap figure with x=source_chars, y=predicted_chars,
    z=attentions matrix.  Hover text will show exact weight.
    """
    heatmap = go.Heatmap(
        z=attentions,
        x=source_chars,
        y=predicted_chars,
        colorscale="Blues",
        zmin=0, zmax=1,
        colorbar=dict(
            title=dict(
                text="Attention",
                side="right",
                font=dict(size=12)
            ),
            lenmode="fraction",
            len=0.6,
            tickfont=dict(size=10)
        ),
        hovertemplate=(
            "input: %{x}<br>"
            "output: %{y}<br>"
            "weight: %{z:.3f}"
            "<extra></extra>"
        )
    )

    fig = go.Figure(data=[heatmap])
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,           # center
            xanchor="center",
            yanchor="top",
            font=dict(size=16)
        ),
        margin=dict(l=80, r=50, t=100, b=80),
        width=600,
        height=450,
        xaxis=dict(
            title=dict(
                text="Source characters",
                font=dict(size=14)
            ),
            tickangle=-45,
            tickfont=dict(size=12),
            side="top",
            automargin=True
        ),
        yaxis=dict(
            title=dict(
                text="Predicted characters",
                font=dict(size=14)
            ),
            tickfont=dict(size=12),
            automargin=True
        )
    )
    return fig


if __name__ == "__main__":
    args = parse_args()

    # ─── WandB initialization (optional) ─────────────────────────
    use_wandb = args.wandb_project is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            tags=[args.wandb_run_tag] if args.wandb_run_tag else None
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load model and vocab built from the true training split
    model, src_vocab, tgt_vocab = load_model_and_vocab(
        args.checkpoint, args.train_tsv, device
    )

    # 2) Load test set (for sampling)
    test_ds = DakshinaLexicon(args.test_tsv, src_vocab, tgt_vocab)

    # 3) Randomly pick N examples
    random.seed(42)
    selected_indices = random.sample(range(len(test_ds)), k=args.n_examples)

    # 4) For each example, record attention and make a Plotly figure
    figures: list[go.Figure] = []
    for idx_rank, idx in enumerate(selected_indices, start=1):
        src_ids, _ = test_ds[idx]
        source_chars, predicted_chars, attn_matrix = record_attention(
            model, src_ids, src_vocab, tgt_vocab, device
        )
        title = (
            f'Example {idx_rank}: “{"".join(source_chars)}” → '
            f'“{"".join(predicted_chars)}”'
        )
        fig = make_plotly_figure(source_chars, predicted_chars, attn_matrix, title)
        figures.append(fig)

    # 5) Assemble all figures into one standalone HTML
    html_snippets = [
        fig.to_html(full_html=False, include_plotlyjs="cdn")
        for fig in figures
    ]
    html_body = "\n<hr>\n".join(html_snippets)
    full_html = f"""\
<html>
  <head><meta charset="utf-8"/><title>Attention Connectivity</title></head>
  <body>
    <h1>Q6: Attention Connectivity Visualizations</h1>
    {html_body}
  </body>
</html>
"""

    # 6) Write out the HTML
    out_path = Path(args.output_html)
    out_path.write_text(full_html, encoding="utf-8")
    print(f"Wrote interactive connectivity HTML to {out_path}")

    # 7) Log to WandB (if requested)
    if use_wandb:
        # wandb.Html will render your HTML in the WandB UI
        wandb.log({
            "connectivity": wandb.Html(str(out_path))
        })
        wandb.finish()
