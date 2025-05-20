# -*- coding: utf-8 -*-
"""
End-to-end training script for Q1 on the Dakshina Hindi lexicon.

This script:
  1. Loads and preprocesses the Hindi transliteration lexicon into PyTorch datasets.
  2. Builds a flexible Seq2Seq model with configurable embedding size, hidden size,
     cell type (RNN/LSTM/GRU), number of layers, and choice of character‐vector methods.
  3. Trains the model with teacher forcing (optionally sampling by attestation counts).
  4. Evaluates on dev/test sets and prints perplexities.
  5. Runs a few qualitative transliteration examples.

Usage example:

    python train_dakshina_seq2seq.py \
      --train_tsv ./lexicons/hi.translit.sampled.train.tsv \
      --dev_tsv   ./lexicons/hi.translit.sampled.dev.tsv   \
      --test_tsv  ./lexicons/hi.translit.sampled.test.tsv  \
      --embedding_size 256 \
      --hidden_size    512 \
      --encoder_layers 2 \
      --decoder_layers 2 \
      --cell LSTM \
      --epochs 15 \
      --embedding_method svd_ppmi \
      --use_attestations
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Use GPU if available
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────── 1. Vocabulary helpers ───────────────────────
SPECIAL_TOKENS = {"<pad>": 0, "<sos>": 1, "<eos>": 2}


class CharVocabulary:
    """Character-level vocabulary that handles <pad>, <sos>, and <eos>."""
    def __init__(self, characters: List[str]):
        unique_chars = sorted(set(characters))
        # string→index and index→string maps
        self.stoi: Dict[str, int] = {
            **SPECIAL_TOKENS,
            **{ch: idx + len(SPECIAL_TOKENS) for idx, ch in enumerate(unique_chars)}
        }
        self.itos: Dict[int, str] = {idx: ch for ch, idx in self.stoi.items()}

    def encode(self, text: str, *, add_sos: bool = False, add_eos: bool = True) -> List[int]:
        """Convert a string to list of token ids (with optional <sos>/<eos>)."""
        ids = [self.stoi[ch] for ch in text]
        if add_eos:
            ids.append(self.stoi["<eos>"])
        if add_sos:
            ids.insert(0, self.stoi["<sos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Convert ids back to string (stop at <eos>)."""
        chars: List[str] = []
        for idx in ids:
            if idx == self.stoi["<eos>"]:
                break
            chars.append(self.itos.get(idx, ""))
        return "".join(chars)

    @property
    def size(self) -> int:
        """Total number of tokens in vocabulary."""
        return len(self.stoi)


# ───────────────────────── 2. Dataset class ─────────────────────────
class DakshinaLexicon(Dataset):
    """Loads a Dakshina *lexicon* TSV and encodes (source, target) pairs.

    TSV columns: native_word, romanized_word, count
    We treat romanized_word as source and native_word as target.
    """
    def __init__(
        self,
        tsv_path: str | Path,
        source_vocab: Optional[CharVocabulary] = None,
        target_vocab: Optional[CharVocabulary] = None,
        *,
        build_vocabs: bool = False,
        use_attestations: bool = False
    ):
        # Read TSV – three columns, ensure correct dtypes
        dataframe = pd.read_csv(
            tsv_path, sep="\t", header=None,
            names=["target_native", "source_roman", "count"],
            dtype={"target_native": str, "source_roman": str, "count": int}
        ).dropna()

        # Optionally keep annotator counts for sampling or weighting
        self.example_counts: Optional[List[int]] = (
            dataframe["count"].tolist() if use_attestations else None
        )

        # Keep only the (src, tgt) pairs
        self.word_pairs: List[Tuple[str, str]] = list(zip(
            dataframe["source_roman"], dataframe["target_native"]
        ))

        # Build new or reuse provided vocabularies
        if build_vocabs:
            assert source_vocab is None and target_vocab is None, (
                "Cannot pass existing vocabs when build_vocabs=True"
            )
            # collect all chars
            all_src_chars = [ch for src, _ in self.word_pairs for ch in src]
            all_tgt_chars = [ch for _, tgt in self.word_pairs for ch in tgt]
            source_vocab = CharVocabulary(all_src_chars)
            target_vocab = CharVocabulary(all_tgt_chars)

        assert source_vocab is not None and target_vocab is not None, (
            "Must provide or build vocabularies"
        )
        self.src_vocab, self.tgt_vocab = source_vocab, target_vocab

        # Encode all pairs once for efficiency
        self.encoded_pairs: List[Tuple[List[int], List[int]]] = [
            (self.src_vocab.encode(src),
             self.tgt_vocab.encode(tgt, add_sos=True))
            for src, tgt in self.word_pairs
        ]
        # Also keep just the source sequences for SVD/PPMI embedding
        self.encoded_sources: List[List[int]] = [src for src, _ in self.encoded_pairs]

        # Padding token id
        self.pad_id: int = self.src_vocab.stoi["<pad>"]

    def __len__(self) -> int:
        return len(self.encoded_pairs)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        return self.encoded_pairs[index]


def collate_batch(
    batch: List[Tuple[List[int], List[int]]],
    pad_id: int
) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """Pad source and target sequences to uniform length within a batch."""
    src_seqs, tgt_seqs = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    tgt_lengths = torch.tensor([len(t) for t in tgt_seqs], dtype=torch.long)

    max_src_len = src_lengths.max().item()
    max_tgt_len = tgt_lengths.max().item()

    padded_sources = torch.full((len(batch), max_src_len), pad_id, dtype=torch.long)
    padded_targets = torch.full((len(batch), max_tgt_len), pad_id, dtype=torch.long)

    for i, (s, t) in enumerate(zip(src_seqs, tgt_seqs)):
        padded_sources[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        padded_targets[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    return padded_sources, src_lengths, padded_targets


# ──────────────────── 2.5. Embedding-method modules ────────────────────
class OneHotEmbedding(nn.Module):
    """Convert token ids → explicit one-hot → linear projection to embedding_dim."""
    def __init__(self, vocab_size: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        # Project a one-hot vector of length vocab_size → embedding_dim
        self.projection = nn.Linear(vocab_size, embedding_dim, bias=False)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # token_ids: (B, T)
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()  # (B,T,V)
        # zero out pad positions if desired
        one_hot[token_ids == self.padding_idx] = 0.0
        # project → (B, T, D)
        return self.projection(one_hot)


class CharCNNEmbedding(nn.Module):
    """Convert token ids → one-hot → conv1d over time → (B,T,embedding_dim)."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        kernel_size: int = 3,
        num_filters: Optional[int] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.padding_idx = padding_idx
        self.num_filters = num_filters or embedding_dim
        # Convolution: in_channels=vocab_size, out_channels=num_filters
        self.conv1d = nn.Conv1d(
            in_channels=vocab_size,
            out_channels=self.num_filters,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        # Optionally project filters → embedding_dim
        if self.num_filters != embedding_dim:
            self.projection = nn.Linear(self.num_filters, embedding_dim, bias=False)
        else:
            self.projection = None

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # token_ids: (B, T)
        one_hot = F.one_hot(token_ids, num_classes=self.vocab_size).float()  # (B,T,V)
        x = one_hot.permute(0, 2, 1)  # (B, V, T)
        x = self.conv1d(x)           # (B, F, T)
        x = x.permute(0, 2, 1)       # (B, T, F)
        if self.projection:
            x = self.projection(x)   # (B, T, D)
        return x                     # (B, T, embedding_dim)


class SVDPPMIEmbedding(nn.Module):
    """Build PPMI→SVD char embeddings, then (if needed) project to embedding_dim."""
    def __init__(
        self,
        token_seqs: List[List[int]],
        vocab_size: int,
        embedding_dim: int,
        padding_idx: int,
        window: int = 2
    ):
        super().__init__()
        # 1) Build co-occurrence counts
        cooc = np.zeros((vocab_size, vocab_size), dtype=np.float64)
        for seq in token_seqs:
            for i, u in enumerate(seq):
                if u == padding_idx: continue
                for j in range(max(0, i - window), min(len(seq), i + window + 1)):
                    if i == j: continue
                    v = seq[j]
                    if v == padding_idx: continue
                    cooc[u, v] += 1

        # 2) Compute PPMI matrix
        total = cooc.sum()
        row_sums = cooc.sum(axis=1, keepdims=True)
        col_sums = cooc.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            pmi = np.log((cooc * total) / (row_sums * col_sums))
        pmi[np.isnan(pmi)] = 0.0
        pmi[pmi < 0] = 0.0

        # 3) Truncated SVD
        U, S, _ = np.linalg.svd(pmi, full_matrices=False)
        # D0 is the actual SVD dimension we get (≤ vocab_size)
        D0 = min(embedding_dim, U.shape[1])
        U = U[:, :D0]            # (vocab_size, D0)
        S = S[:D0]               # (D0,)
        emb_matrix = U * np.sqrt(S)  # (vocab_size, D0)

        # Register static SVD weights
        self.register_buffer("weight", torch.from_numpy(emb_matrix).float())

        # 4) If the SVD rank D0 is smaller than requested embedding_dim, add a projection
        if D0 < embedding_dim:
            # Warn the user clearly
            warnings.warn(
                f"SVD/PPMI yielded only {D0} dimensions (≤ vocab size), "
                f"but embedding_size={embedding_dim} was requested. "
                "Adding a learnable linear projection to expand from "
                f"{D0} → {embedding_dim} dimensions.",
                UserWarning
            )
            self.expander = nn.Linear(D0, embedding_dim, bias=False)
        else:
            self.expander = None

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        # Lookup static SVD embeddings → (B, T, D0)
        x = F.embedding(token_ids, self.weight, padding_idx=self.weight.new_zeros(1).long())
        # If we have an expander, project to the full embedding_dim
        if self.expander:
            x = self.expander(x)
        return x  # (B, T, embedding_dim)


# ───────────────────────── 3. Seq2SeqConfig ─────────────────────────
@dataclass
class Seq2SeqConfig:
    """Holds hyper-parameters and options for the Seq2Seq model."""
    # mandatory vocab sizes
    source_vocab_size: int
    target_vocab_size: int

    # embedding / hidden dims
    embedding_dim: int = 256
    hidden_dim: int = 512

    # encoder/decoder depth & type
    encoder_layers: int = 2
    decoder_layers: int = 2
    cell_type: str = "LSTM"  # choices: RNN | LSTM | GRU

    # dropout for multi-layer RNNs
    dropout: float = 0.1

    # special token indices
    pad_index: int = SPECIAL_TOKENS["<pad>"]
    sos_index: int = SPECIAL_TOKENS["<sos>"]
    eos_index: int = SPECIAL_TOKENS["<eos>"]

    # which character embedding method to use
    embedding_method: str = "learned"  # learned | onehot | char_cnn | svd_ppmi

    # only used if embedding_method == "svd_ppmi"
    svd_sources: Optional[List[List[int]]] = None

    def __post_init__(self):
        assert self.cell_type in {"RNN", "LSTM", "GRU"}, "cell_type must be RNN, LSTM or GRU"
        assert self.embedding_method in {"learned", "onehot", "char_cnn", "svd_ppmi"}, (
            "embedding_method must be one of learned | onehot | char_cnn | svd_ppmi"
        )
        if self.embedding_method == "svd_ppmi":
            assert self.svd_sources is not None, "svd_ppmi requires svd_sources"


# map a string to the corresponding nn.RNN module
_RNN_MAP: Dict[str, nn.Module] = {
    "RNN": nn.RNN,
    "LSTM": nn.LSTM,
    "GRU": nn.GRU,
}

# ─────────────────────────── Helper: Align Hidden State ──────────────────────────
def _align_hidden_state(hidden_state, target_num_layers: int):
    """
    Adjust the encoder's final hidden state to match the decoder's expected
    number of layers. Works for both LSTM (tuple of (h,c)) and GRU/RNN (single tensor).
    
    Strategies:
    - If encoder_layers == decoder_layers: return hidden_state unchanged.
    - If encoder_layers  > decoder_layers: take the **last** `target_num_layers` layers.
    - If encoder_layers  < decoder_layers: **repeat** the final layer's state
      so that the total number of layers equals `target_num_layers`.
    """
    def _repeat_last_layer(tensor, repeat_count: int):
        # tensor shape: (enc_layers, batch_size, hidden_dim)
        last_layer = tensor[-1:]                             # shape: (1, B, H)
        repeated   = last_layer.expand(repeat_count, -1, -1) # shape: (repeat_count, B, H)
        return torch.cat([tensor, repeated], dim=0)          # new shape: (enc_layers+repeat_count, B, H)

    if isinstance(hidden_state, tuple):
        h, c = hidden_state
        enc_layers, batch_size, hid_dim = h.shape

        if enc_layers == target_num_layers:
            return h, c
        
        # Warn whenever we need to truncate or repeat
        warnings.warn(
            f"Encoder has {enc_layers} layers but decoder expects {target_num_layers}. "
            f"{'Truncating' if enc_layers>target_num_layers else 'Repeating last layer'} hidden state.",
            UserWarning
        )

        if enc_layers > target_num_layers:
            # keep only the last `target_num_layers` layers
            return h[-target_num_layers:], c[-target_num_layers:]
        else:
            # repeat the final layer's state to pad up to target_num_layers
            to_repeat = target_num_layers - enc_layers
            return _repeat_last_layer(h, to_repeat), _repeat_last_layer(c, to_repeat)
    else:
        h = hidden_state
        enc_layers, batch_size, hid_dim = h.shape

        if enc_layers == target_num_layers:
            return h
        
        # Same warning for single‐tensor hidden states
        warnings.warn(
            f"Encoder has {enc_layers} layers but decoder expects {target_num_layers}. "
            f"{'Truncating' if enc_layers>target_num_layers else 'Repeating last layer'} hidden state.",
            UserWarning
        )

        if enc_layers > target_num_layers:
            return h[-target_num_layers:]
        else:
            to_repeat = target_num_layers - enc_layers
            return _repeat_last_layer(h, to_repeat)


# ───────────────────────── 4. Encoder ─────────────────────────
class Encoder(nn.Module):
    """Encoder: character embeddings → RNN → hidden state(s)."""
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg

        # choose between various embedding modules
        if cfg.embedding_method == "learned":
            self.embedding = nn.Embedding(
                cfg.source_vocab_size,
                cfg.embedding_dim,
                padding_idx=cfg.pad_index
            )
        elif cfg.embedding_method == "onehot":
            self.embedding = OneHotEmbedding(
                vocab_size=cfg.source_vocab_size,
                embedding_dim=cfg.embedding_dim,
                padding_idx=cfg.pad_index
            )
        elif cfg.embedding_method == "char_cnn":
            self.embedding = CharCNNEmbedding(
                vocab_size=cfg.source_vocab_size,
                embedding_dim=cfg.embedding_dim,
                padding_idx=cfg.pad_index,
                kernel_size=3
            )
        elif cfg.embedding_method == "svd_ppmi":
            self.embedding = SVDPPMIEmbedding(
                token_seqs=cfg.svd_sources,
                vocab_size=cfg.source_vocab_size,
                embedding_dim=cfg.embedding_dim,
                padding_idx=cfg.pad_index,
                window=2
            )
        else:
            raise ValueError(f"Unknown embedding_method {cfg.embedding_method}")

        # set up the RNN cell
        rnn_cls = _RNN_MAP[cfg.cell_type]
        self.rnn = rnn_cls(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0
        )

    def forward(self, src: torch.Tensor, src_lengths: torch.Tensor):
        """
        Args:
            src         – LongTensor (B, T_src)
            src_lengths – LongTensor (B,) true lengths before padding
        Returns:
            hidden_state – (h, c) for LSTM or h for GRU/RNN
        """
        embedded = self.embedding(src)  # (B, T_src, D)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden_state = self.rnn(packed)
        return hidden_state  # LSTM → (h,c), GRU/RNN → h


# ───────────────────────── 5. Decoder ─────────────────────────
class Decoder(nn.Module):
    """Decoder: one step of embedding → RNN → vocab logits."""
    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(
            cfg.target_vocab_size,
            cfg.embedding_dim,
            padding_idx=cfg.pad_index
        )
        rnn_cls = _RNN_MAP[cfg.cell_type]
        self.rnn = rnn_cls(
            cfg.embedding_dim,
            cfg.hidden_dim,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0
        )
        self.to_vocab_logits = nn.Linear(cfg.hidden_dim, cfg.target_vocab_size)

    def forward(
        self,
        current_input: torch.LongTensor,
        hidden_state
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """
        Args:
          current_input – LongTensor (B, 1)
          hidden_state  – previous hidden state(s)
        Returns:
          logits        – (B, 1, V)
          hidden_state  – updated hidden state(s)
        """
        emb = self.embedding(current_input)            # (B,1,D)
        output, new_hidden = self.rnn(emb, hidden_state)  # (B,1,H)
        logits = self.to_vocab_logits(output)          # (B,1,V)
        return logits, new_hidden


# ───────────────────────── 6. Seq2Seq wrapper ─────────────────────────
class Seq2Seq(nn.Module):
    """Flexible encoder-decoder wrapper combining Encoder and Decoder."""

    def __init__(self, cfg: Seq2SeqConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    # ─────────────────────────────────────────────────────────────────────
    # Training-time forward (teacher forcing)
    # ─────────────────────────────────────────────────────────────────────
    def forward(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
        tgt: torch.LongTensor, *,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Compute logits for each timestep in the target sequence using teacher forcing.
        Returns logits_all of shape (B, T_tgt, vocab_size).
        """
        batch_size, tgt_len = tgt.size()
        logits_all = torch.zeros(batch_size, tgt_len, self.cfg.target_vocab_size, device=tgt.device)

        # 1) Encode source
        hidden_state = self.encoder(src, src_lengths)

        # 2) Align hidden_state to decoder depth
        hidden_state = _align_hidden_state(hidden_state, self.cfg.decoder_layers)

        # 3) First decoder input is <sos>
        decoder_input = tgt[:, 0].unsqueeze(1)  # (B,1)

        # 4) Unroll for each timestep
        for t in range(1, tgt_len):
            step_logits, hidden_state = self.decoder(decoder_input, hidden_state)
            logits_all[:, t] = step_logits.squeeze(1)
            # decide next input
            if torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = tgt[:, t].unsqueeze(1)
            else:
                decoder_input = step_logits.argmax(-1)

        return logits_all

    # ─────────────────────────────────────────────────────────────────────
    # Greedy decoding
    # ─────────────────────────────────────────────────────────────────────
    def greedy_decode(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor, *,
        max_len: int = 50,
    ) -> torch.LongTensor:
        """
        Greedy decoding for inference: always take argmax.
        Returns tensor of shape (B, <=max_len).
        """
        B = src.size(0)
        hidden_state = self.encoder(src, src_lengths)
        hidden_state = _align_hidden_state(hidden_state, self.cfg.decoder_layers)

        decoder_input = torch.full(
            (B, 1),
            self.cfg.sos_index,
            dtype=torch.long,
            device=src.device
        )
        generated_ids: List[torch.LongTensor] = []

        for _ in range(max_len):
            step_logits, hidden_state = self.decoder(decoder_input, hidden_state)
            next_ids = step_logits.argmax(-1)  # (B,1)
            generated_ids.append(next_ids)
            decoder_input = next_ids

        return torch.cat(generated_ids, dim=1)  # (B, seq_len)

    # ─────────────────────────────────────────────────────────────────────
    # Beam-search decoding
    # ─────────────────────────────────────────────────────────────────────
    def beam_search_decode(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
        *,
        beam_size: int = 5,
        max_len: int = 50,
    ) -> torch.LongTensor:
        """
        Beam search decoding (only batch_size=1 supported).
        Returns the best sequence of token ids as a tensor of shape (1, <=max_len).
        """
        B = src.size(0)
        assert B == 1, "beam_search_decode only supports batch_size=1 for now"

        # 1) Encode & align hidden state
        hidden_state = self.encoder(src, src_lengths)
        hidden_state = _align_hidden_state(hidden_state, self.cfg.decoder_layers)

        # 2) Initialize beams: list of (sequence_ids, cumulative_log_prob, hidden_state)
        beams: List[Tuple[List[int], float, Any]] = [
            ([self.cfg.sos_index], 0.0, hidden_state)
        ]

        for _ in range(max_len):
            all_candidates: List[Tuple[List[int], float, Any]] = []
            # Expand each current beam
            for seq_ids, cum_logprob, h_state in beams:
                last_token = seq_ids[-1]
                # if EOS already, just carry it forward
                if last_token == self.cfg.eos_index:
                    all_candidates.append((seq_ids, cum_logprob, h_state))
                    continue

                # run one step
                inp = torch.tensor([[last_token]], device=src.device)
                logits, new_h_state = self.decoder(inp, h_state)       # (1,1,V)
                log_probs = torch.log_softmax(logits.squeeze(1), dim=-1)  # (1,V) → (V,)

                # pick top-k continuations
                # collapse the batch dimension so we get 1D lists
                vals, idxs = log_probs.topk(beam_size, dim=-1)      # shape: (1, beam_size)
                vals = vals[0].detach().cpu().tolist()               # → [v₁, v₂, …]
                idxs = idxs[0].detach().cpu().tolist()               # → [i₁, i₂, …]
                for log_p, idx in zip(vals, idxs):
                    new_seq   = seq_ids + [idx]
                    new_score = cum_logprob + log_p                 # log_p is now a float
                    all_candidates.append((new_seq, new_score, new_h_state))

            # keep best beam_size beams
            beams = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

        # pick the single best final sequence
        best_seq, best_score, _ = max(beams, key=lambda x: x[1])
        # convert to tensor
        return torch.tensor(best_seq, dtype=torch.long, device=src.device).unsqueeze(0)



# ───────────────────────── 7. Training & evaluation ─────────────────────────
def train_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.CrossEntropyLoss,
    device: str,
    teacher_forcing: float
) -> float:
    """Run one epoch of training; return average token-level cross-entropy loss."""
    model.train()
    total_loss, total_tokens = 0.0, 0
    for src, src_len, tgt in loader:
        src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
        optimizer.zero_grad()
        logits = model(src, src_len, tgt, teacher_forcing_ratio=teacher_forcing)
        # shift tgt so we predict t when truth is at t
        gold = tgt[:, 1:].contiguous()
        preds = logits[:, 1:].contiguous()
        loss = loss_fn(preds.view(-1, preds.size(-1)), gold.view(-1))
        loss.backward()
        optimizer.step()

        n_valid = (gold != loss_fn.ignore_index).sum().item()
        total_loss += loss.item() * n_valid
        total_tokens += n_valid

    return total_loss / total_tokens


def eval_epoch(
    model: Seq2Seq,
    loader: DataLoader,
    loss_fn: nn.CrossEntropyLoss,
    device: str
) -> float:
    """Run one epoch of evaluation; return average token-level cross-entropy loss."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for src, src_len, tgt in loader:
            src, src_len, tgt = src.to(device), src_len.to(device), tgt.to(device)
            logits = model(src, src_len, tgt, teacher_forcing_ratio=0.0)
            gold = tgt[:, 1:].contiguous()
            preds = logits[:, 1:].contiguous()
            loss = loss_fn(preds.view(-1, preds.size(-1)), gold.view(-1))
            n_valid = (gold != loss_fn.ignore_index).sum().item()
            total_loss += loss.item() * n_valid
            total_tokens += n_valid

    return total_loss / total_tokens


# ───────────────────────── 8. Argument parsing ─────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train Seq2Seq on the Dakshina Hindi lexicon (character-level transliteration)"
    )
    p.add_argument(
        "--train_tsv",
        type=str,
        required=True,
        help="Path to the training lexicon TSV (columns: native_word, romanized_word, count)"
    )
    p.add_argument(
        "--dev_tsv",
        type=str,
        required=True,
        help="Path to the development (validation) lexicon TSV"
    )
    p.add_argument(
        "--test_tsv",
        type=str,
        required=True,
        help="Path to the test lexicon TSV"
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of full passes over the training data"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Number of examples per mini-batch"
    )
    p.add_argument(
        "--embedding_size",
        type=int,
        default=256,
        help="Dimensionality of the character embedding vectors"
    )
    p.add_argument(
        "--hidden_size",
        type=int,
        default=512,
        help="Size of the hidden states in the encoder and decoder RNNs"
    )
    p.add_argument(
        "--encoder_layers",
        type=int,
        default=2,
        help="Number of stacked RNN layers in the encoder"
    )
    p.add_argument(
        "--decoder_layers",
        type=int,
        default=2,
        help="Number of stacked RNN layers in the decoder"
    )
    p.add_argument(
        "--cell",
        choices=["RNN", "LSTM", "GRU"],
        default="LSTM",
        help="Type of RNN cell to use for both encoder and decoder"
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate for the Adam optimizer"
    )
    p.add_argument(
        "--teacher_forcing",
        type=float,
        default=0.5,
        help="Probability of using teacher forcing at each decoder time step (0.0–1.0)"
    )
    p.add_argument(
        "--device",
        type=str,
        default=DEFAULT_DEVICE,
        help="Computation device: 'cpu' or 'cuda' (default automatically detects GPU if available)"
    )
    p.add_argument(
        "--embedding_method",
        choices=["learned", "onehot", "char_cnn", "svd_ppmi"],
        default="learned",
        help=(
            "How to convert characters to vectors: "
            "'learned' lookup (default), explicit 'onehot'+linear, "
            "'char_cnn' for a 1D CNN over one-hots, or 'svd_ppmi' for "
            "static SVD over PPMI co-occurrences"
        )
    )
    p.add_argument(
        "--use_attestations",
        action="store_true",
        help="If set, sample training examples proportional to their annotation count"
    )
    return p.parse_args()


# ─────────────────────────── 9. Main driver ───────────────────────────
def main():
    args = parse_args()

    # ─── Prepare datasets ───────────────────────────────────────────────
    # Train set: build vocabularies and optionally keep counts
    train_ds = DakshinaLexicon(
        args.train_tsv,
        build_vocabs=True,
        use_attestations=args.use_attestations
    )
    src_vocab, tgt_vocab = train_ds.src_vocab, train_ds.tgt_vocab

    # Dev/test: reuse the same vocabs (counts not needed)
    dev_ds  = DakshinaLexicon(args.dev_tsv,  src_vocab, tgt_vocab)
    test_ds = DakshinaLexicon(args.test_tsv, src_vocab, tgt_vocab)

    # Collate function for padding
    collate_fn = lambda batch: collate_batch(batch, pad_id=src_vocab.stoi["<pad>"])

    # Training loader: either shuffle or sample by counts
    if args.use_attestations:
        assert train_ds.example_counts is not None, "Attestations requested but counts missing"
        sampler = WeightedRandomSampler(
            weights=train_ds.example_counts,
            num_samples=len(train_ds),
            replacement=True
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            collate_fn=collate_fn
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn
        )

    dev_loader  = DataLoader(dev_ds,  batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # ─── Build model ────────────────────────────────────────────────────
    extra_cfg = {}
    if args.embedding_method == "svd_ppmi":
        # supply the token sequences for PPMI/SVD
        extra_cfg["svd_sources"] = train_ds.encoded_sources

    cfg = Seq2SeqConfig(
        source_vocab_size=src_vocab.size,
        target_vocab_size=tgt_vocab.size,
        embedding_dim=args.embedding_size,
        hidden_dim=args.hidden_size,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        cell_type=args.cell,
        dropout=0.1,
        pad_index=src_vocab.stoi["<pad>"],
        sos_index=tgt_vocab.stoi["<sos>"],
        eos_index=tgt_vocab.stoi["<eos>"],
        embedding_method=args.embedding_method,
        **extra_cfg
    )
    model = Seq2Seq(cfg).to(args.device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=cfg.pad_index)

    # ─── Training loop ───────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, args.device, args.teacher_forcing)
        dev_loss   = eval_epoch(model, dev_loader, loss_fn, args.device)
        print(f"Epoch {epoch:02d} | train ppl={math.exp(train_loss):.2f} | dev ppl={math.exp(dev_loss):.2f}")

    # ─── Final evaluation on test ────────────────────────────────────────
    test_loss = eval_epoch(model, test_loader, loss_fn, args.device)
    print(f"Test perplexity: {math.exp(test_loss):.2f}")

    # ─── Qualitative samples ─────────────────────────────────────────────
    model.eval()
    print("Sample dev-set transliterations:")
    with torch.no_grad():
        for i in range(5):
            src_ids, tgt_ids = dev_ds[i]
            src_len = len(src_ids)
            src_tensor = torch.tensor([src_ids], device=args.device)
            len_tensor = torch.tensor([src_len], device=args.device)

            pred_ids = model.greedy_decode(src_tensor, len_tensor, max_len=30)[0].tolist()
            romanized = src_vocab.decode(src_ids)
            gold       = tgt_vocab.decode(tgt_ids[1:])  # skip <sos>
            pred_str   = tgt_vocab.decode(pred_ids[1:])  # skip <sos>
            print(f"{romanized:15} → {pred_str:15} (gold: {gold})")


if __name__ == "__main__":
    main()
