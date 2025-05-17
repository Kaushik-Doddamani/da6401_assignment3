# -*- coding: utf-8 -*-
"""
End-to-end training script for Q1 on the Dakshina Hindi lexicon.

This script:
  1. Loads and preprocesses the Hindi transliteration lexicon into PyTorch datasets.
  2. Builds a flexible Seq2Seq model with configurable embedding size, hidden size,
     cell type (RNN/LSTM/GRU), and number of layers.
  3. Trains the model with teacher forcing and evaluates on dev set.
  4. Prints training/dev perplexities and runs a few qualitative translations.

Usage example:

    python train_dakshina_seq2seq.py \
        --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
        --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv   \
        --test_tsv  ./lexicons/hi.translit.sampled.test.tsv  \
        --embedding_size 256 \
        --hidden_size    512 \
        --encoder_layers 2   \
        --decoder_layers 2   \
        --rnn_cell_type LSTM \
        --epochs 15

All hyper-parameters can be overridden via CLI. The companion `seq2seq_model.py`
defines the flexible `Seq2Seq` architecture.
"""

from __future__ import annotations

# ───────────────────────────── Imports ──────────────────────────────
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Use GPU if available
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────── 1. Vocabulary helpers ──────────────────────
SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2}


class CharVocabulary:
    """Character‑level vocabulary that handles <pad>, <sos>, and <eos>."""

    def __init__(self, characters: List[str]):
        unique_characters = sorted(set(characters))
        # `stoi`: string‑to‑index, `itos`: index‑to‑string
        self.stoi: Dict[str, int] = {**SPECIAL_TOKENS,
                                     **{char: idx + len(SPECIAL_TOKENS)
                                        for idx, char in enumerate(unique_characters)}}
        self.itos: Dict[int, str] = {idx: char for char, idx in self.stoi.items()}

    # --- Encoding / decoding -------------------------------------------------
    def encode(self, text: str, *, add_sos: bool = False, add_eos: bool = True) -> List[int]:
        """Convert a string to a list of integer ids."""
        ids = [self.stoi[char] for char in text]
        if add_eos:
            ids.append(self.stoi["<eos>"])
        if add_sos:
            ids.insert(0, self.stoi["<sos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert ids back to a string (stop at <eos>)."""
        characters: List[str] = []
        for idx in ids:
            if idx == self.stoi["<eos>"]:
                break
            characters.append(self.itos.get(idx, ""))
        return "".join(characters)

    # --- Convenience property ----------------------------------------------
    @property
    def size(self) -> int:  # number of tokens in the vocabulary
        return len(self.stoi)


# ───────────────────────── 2. Dataset class ─────────────────────────
class DakshinaLexicon(Dataset):
    """Loads a Dakshina *lexicon* TSV file and encodes each word pair.

    Each line of the TSV is:  <native_word> <romanized_word> <count>
    We treat `romanized_word` as *source* and `native_word` as *target*.
    """

    def __init__(self, tsv_path: str | Path,
                 source_vocab: CharVocabulary | None = None,
                 target_vocab: CharVocabulary | None = None,
                 *, build_vocabs: bool = False):
        # Read TSV – no header, three columns
        dataframe = pd.read_csv(tsv_path, sep="\t", header=None,
                                names=["target_native", "source_roman", "count"],
                                dtype=str)
        
        # Drop rows with NaN values
        # dataframe = dataframe.fillna("")
        dataframe = dataframe.dropna()

        # Keep (src, tgt) pairs only
        self.word_pairs: List[Tuple[str, str]] = list(zip(dataframe.source_roman,
                                                          dataframe.target_native))

        # Build or reuse vocabularies
        if build_vocabs:
            assert source_vocab is None and target_vocab is None, (
                "Cannot pass existing vocabs when build_vocabs=True")
            source_vocab = CharVocabulary([ch for src, _ in self.word_pairs for ch in src])
            target_vocab = CharVocabulary([ch for _, tgt in self.word_pairs for ch in tgt])
        assert source_vocab is not None and target_vocab is not None, (
            "Vocabularies must be provided or set build_vocabs=True")
        self.src_vocab, self.tgt_vocab = source_vocab, target_vocab

        # Encode all pairs once → speed‑ups during batching
        self.encoded_pairs: List[Tuple[List[int], List[int]]] = [
            (self.src_vocab.encode(src),
             self.tgt_vocab.encode(tgt, add_sos=True))
            for src, tgt in self.word_pairs
        ]
        self.pad_id: int = self.src_vocab.stoi["<pad>"]

    # --- Dataset protocol ----------------------------------------------------
    def __len__(self):
        return len(self.encoded_pairs)

    def __getitem__(self, index):
        return self.encoded_pairs[index]


# Collate function → pads a batch to equal length
def collate_batch(batch: List[Tuple[List[int], List[int]]], pad_id: int):
    """Pads source and target sequences in a batch and returns tensors."""
    src_seqs, tgt_seqs = zip(*batch)
    src_lengths = torch.tensor([len(seq) for seq in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(seq) for seq in tgt_seqs], dtype=torch.long)

    max_src_len = src_lengths.max().item()
    max_tgt_len = tgt_lengths.max().item()

    padded_sources = torch.full((len(batch), max_src_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_tgt_len), pad_id, dtype=torch.long)

    for idx, (src, tgt) in enumerate(zip(src_seqs, tgt_seqs)):
        padded_sources[idx, : len(src)] = torch.tensor(src)
        padded_targets[idx, : len(tgt)] = torch.tensor(tgt)

    return padded_sources, src_lengths, padded_targets


# ────────────────────── 3. Seq2Seq architecture ─────────────────────
@dataclass
class Seq2SeqConfig:
    """Holds hyper‑parameters for the Seq2Seq model.

    Modify these values (or pass overrides to the constructor) to change the
    embedding size, hidden size, number of layers, cell type, etc.
    """
    # Vocab sizes – must be supplied
    source_vocab_size: int
    target_vocab_size: int
    # Embedding / hidden sizes
    embedding_dim: int = 256
    hidden_dim: int = 512
    # Depth and cell type
    encoder_layers: int = 2
    decoder_layers: int = 2
    cell_type: str = "LSTM"  # choices: "RNN", "LSTM", "GRU"
    # Misc.
    dropout: float = 0.1
    pad_index: int = SPECIAL_TOKENS["<pad>"]
    sos_index: int = SPECIAL_TOKENS["<sos>"]
    eos_index: int = SPECIAL_TOKENS["<eos>"]

    # Validate at construction time
    def __post_init__(self):
        assert self.cell_type in {"RNN", "LSTM", "GRU"}, "cell_type must be one of RNN | LSTM | GRU"


# Helper: map string to the corresponding PyTorch RNN module
_RNN_MAP: Dict[str, nn.Module] = {
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}


class Encoder(nn.Module):
    """Encoder that converts a source character sequence → hidden state(s)."""

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.source_vocab_size, cfg.embedding_dim, padding_idx=cfg.pad_index)
        rnn_cls = _RNN_MAP[cfg.cell_type]
        self.rnn = rnn_cls(
            cfg.embedding_dim,
            cfg.hidden_dim,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0,
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:
        src          – LongTensor (B, T_src)
        src_lengths  – LongTensor (B,) actual sequence lengths before padding
        Returns → hidden (and cell for LSTM)
        """
        # Embed and pack for efficient processing of variable‑length batches
        embedded = self.embedding(src)  # (B, T_src, D)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden_state = self.rnn(packed)  # last hidden / (hidden, cell)
        return hidden_state  # LSTM → Tuple[h, c]; GRU/RNN → Tensor


class Decoder(nn.Module):
    """Decoder that generates the target (native script) sequence one token at a time."""

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.target_vocab_size, cfg.embedding_dim, padding_idx=cfg.pad_index)
        rnn_cls = _RNN_MAP[cfg.cell_type]
        self.rnn = rnn_cls(
            cfg.embedding_dim,
            cfg.hidden_dim,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0,
        )
        self.to_vocab_logits = nn.Linear(cfg.hidden_dim, cfg.target_vocab_size)

    def forward(self, current_input: torch.Tensor, hidden_state):
        """Perform a single decoding step.

        Args:
            current_input    – LongTensor (B, 1) current input character ids
            hidden           – previous hidden state(s) (from encoder or previous step)
        Returns:
            logits – (B, 1, vocab_size)
            hidden – updated hidden state(s)
        """
        embedding = self.embedding(current_input)  # (B, 1, D)
        rnn_output, hidden_state = self.rnn(embedding, hidden_state)  # output: (B, 1, H)
        logits = self.to_vocab_logits(rnn_output)            # (B, 1, V)
        return logits, hidden_state

class Seq2Seq(nn.Module):
    """Flexible encoder‑decoder wrapper combining Encoder and Decoder."""

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # ─────────────────────────────────────────────────────────────────────
    # Training‑time forward (teacher forcing)
    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        tgt: torch.Tensor, *,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Compute logits for each timestep in the target sequence.

        Args:
            src                   – (B, T_src) source token ids
            src_lengths           – (B,) true lengths of src sequences
            tgt                   – (B, T_tgt) target token ids (with <sos> at position 0)
            teacher_forcing_ratio – probability of using ground truth at each step
        Returns:
            logits – (B, T_tgt, vocab_size) over all timesteps *including* t=0
        """
        batch_size, tgt_len  = tgt.size()
        logits_all = torch.zeros(batch_size, tgt_len, self.cfg.target_vocab_size,
                             device=tgt.device)

        # 1. Encode source
        hidden_state = self.encoder(src, src_lengths)

        # 2. First decoder input is <sos>
        decoder_input = tgt[:, 0].unsqueeze(1)  # (B,1)

        for timestep in range(1, tgt_len):
            step_logits, hidden_state = self.decoder(decoder_input, hidden_state)
            logits_all[:, timestep] = step_logits.squeeze(1)  # store

            # Decide next input (teacher vs. model prediction)
            use_teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = step_logits.argmax(dim=-1)  # (B,1)
            next_input = tgt[:, timestep].unsqueeze(1) if use_teacher_force else top1
            decoder_input = next_input  # (B,1)

        return logits_all

    # ─────────────────────────────────────────────────────────────────────
    # Inference (greedy decoding)
    # ─────────────────────────────────────────────────────────────────────
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        *,
        max_len: int = 50,
    ) -> torch.Tensor:
        """Greedy decoding for inference.

        Returns LongTensor (B, <=max_len) of generated token ids.
        """
        B = src.size(0)
        hidden_state = self.encoder(src, src_lengths)
        decoder_input = torch.full((B, 1), self.cfg.sos_index, dtype=torch.long, device=src.device)
        decoded_ids = []
        for _ in range(max_len):
            step_logits, hidden_state = self.decoder(decoder_input, hidden_state)
            next_ids = step_logits.argmax(-1)  # (batch, 1)
            decoded_ids.append(next_ids)
            decoder_input = next_ids
        return torch.cat(decoded_ids, dim=1)  # (batch, seq_len)

# ────────────────────────────────────────────────────────────────────────────
# Training & evaluation utils
# ────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, loss_fn, device, teacher_forcing):
    model.train()
    total_loss, n_tokens = 0.0, 0
    for src, src_len, tgt in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, src_len, tgt, teacher_forcing_ratio=teacher_forcing)
        # Shift targets so we predict t when truth is at t
        tgt_gold = tgt[:, 1:].contiguous()
        logits = logits[:, 1:].contiguous()  # same shift
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_gold.view(-1))
        loss.backward()
        optimizer.step()

        n_valid = (tgt_gold != loss_fn.ignore_index).sum().item()
        total_loss += loss.item() * n_valid
        n_tokens += n_valid
    return total_loss / n_tokens


def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss, n_tokens = 0.0, 0
    with torch.no_grad():
        for src, src_len, tgt in loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            logits = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
            tgt_gold = tgt[:, 1:].contiguous()
            logits = logits[:, 1:].contiguous()
            loss = loss_fn(logits.view(-1, logits.size(-1)), tgt_gold.view(-1))
            n_valid = (tgt_gold != loss_fn.ignore_index).sum().item()
            total_loss += loss.item() * n_valid
            n_tokens += n_valid
    return total_loss / n_tokens


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Seq2Seq on Dakshina Hindi lexicon (Q1)")
    p.add_argument("--train_tsv", type=str, required=True)
    p.add_argument("--dev_tsv", type=str, required=True)
    p.add_argument("--test_tsv", type=str, required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--embedding_size", type=int, default=256)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--encoder_layers", type=int, default=2)
    p.add_argument("--decoder_layers", type=int, default=2)
    p.add_argument("--cell", choices=["RNN", "LSTM", "GRU"], default="LSTM")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--teacher_forcing", type=float, default=0.5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # ---------- Sanity check ----------
    if args.encoder_layers != args.decoder_layers:
        raise ValueError(
            "For this implementation the encoder and decoder must have the same "
            f"number of layers (got enc={args.encoder_layers}, dec={args.decoder_layers})."
        )

    # ─── Data ────────────────────────────────────────────────────────────────
    train_ds = DakshinaLexicon(args.train_tsv, build_vocabs=True)
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab

    dev_ds = DakshinaLexicon(args.dev_tsv, src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(args.test_tsv, src_vocab, tgt_vocab)

    collate_fn = lambda batch: collate_batch(batch, pad_id=src_vocab.stoi["<pad>"])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)


    # ─── Model ────────────────────────────────────────────────────────────────
    cfg = Seq2SeqConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=args.embedding_size,
        hidden_dim=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        cell_type=args.cell,
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"]
    )
    model = Seq2Seq(cfg).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # CrossEntropy expects (N, C) logits vs (N,) gold
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    # ─── Training loop ───────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, args.teacher_forcing)
        dev_loss   = eval_epoch(model, dev_loader,   loss_fn, args.device)
        print(f"Epoch {epoch:02d} | train ppl={math.exp(train_loss):.2f} | dev ppl={math.exp(dev_loss):.2f}")

    # ---------- Final evaluation on test ------------------------------------
    test_ppl = eval_epoch(model, test_loader, loss_fn, args.device)
    print(f"Test perplexity: {test_ppl:.2f}")


    # ─── Quick qualitative check on a few dev examples ───────────────────────
    model.eval()
    print("Sample dev-set predictions:")
    with torch.no_grad():
        for i in range(5):
            src_ids, tgt_ids = dev_ds[i]          # dev_ds returns (src, tgt)
            src_len = len(src_ids)

            src_tensor = torch.tensor([src_ids], device=args.device)
            len_tensor = torch.tensor([src_len],  device=args.device)

            pred_ids = model.greedy_decode(src_tensor, len_tensor, max_len=30)[0].tolist()

            romanized = src_vocab.decode(src_ids)
            gold       = tgt_vocab.decode(tgt_ids[1:])   # skip <sos>
            prediction = tgt_vocab.decode(pred_ids)

            print(f"{romanized:15} → {prediction:15} (gold: {gold})")


if __name__ == "__main__":
    main()
