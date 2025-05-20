# solution_5_model.py
# -*- coding: utf-8 -*-
"""
Attention‐augmented Seq2Seq for Dakshina Hindi transliteration (Q5).

This module defines:
  - Seq2SeqAttentionConfig: hyper‐parameters & model options
  - DotProductAttention: simple global dot‐product attention
  - EncoderWithOutputs: returns per‐time‐step features + final hidden
  - DecoderWithAttention: attends + decodes one token at a time
  - Seq2SeqAttention: end‐to‐end training / inference wrapper
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# reuse your embedding & RNN machinery from solution_1
from solution_1 import (
    OneHotEmbedding,
    CharCNNEmbedding,
    SVDPPMIEmbedding,
    _RNN_MAP,
    _align_hidden_state,
)


# ───────────────────────── 1. Configuration ─────────────────────────
@dataclass
class Seq2SeqAttentionConfig:
    source_vocab_size: int
    target_vocab_size: int

    embedding_dim: int = 256
    hidden_dim:    int = 512

    encoder_layers: int = 1
    decoder_layers: int = 1

    cell_type: str = "LSTM"       # "RNN" | "GRU" | "LSTM"
    dropout:   float = 0.1

    pad_index: int = 0
    sos_index: int = 1
    eos_index: int = 2

    embedding_method: str = "learned"  # "learned" | "onehot" | "char_cnn" | "svd_ppmi"
    svd_sources: Optional[List[List[int]]] = None

    attention_dim: Optional[int] = None

    def __post_init__(self):
        assert self.cell_type in {"RNN", "GRU", "LSTM"}, "cell_type must be RNN, GRU or LSTM"
        assert self.embedding_method in {"learned", "onehot", "char_cnn", "svd_ppmi"}
        if self.embedding_method == "svd_ppmi":
            assert self.svd_sources is not None, "svd_ppmi requires svd_sources"
        if self.attention_dim is None:
            self.attention_dim = self.hidden_dim


# ───────────────────────── 2. Encoder ───────────────────────────────
class EncoderWithOutputs(nn.Module):
    """
    Encoder that returns both:
      - outputs      : (B, T_src, hidden_dim)
      - hidden_state : final hidden state(s) for decoder init
    """
    def __init__(self, cfg: Seq2SeqAttentionConfig):
        super().__init__()
        self.cfg = cfg

        # character embedding
        if cfg.embedding_method == "learned":
            self.embedding = nn.Embedding(cfg.source_vocab_size,
                                          cfg.embedding_dim,
                                          padding_idx=cfg.pad_index)
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
                padding_idx=cfg.pad_index
            )
        else:  # "svd_ppmi"
            self.embedding = SVDPPMIEmbedding(
                token_seqs=cfg.svd_sources,
                vocab_size=cfg.source_vocab_size,
                embedding_dim=cfg.embedding_dim,
                padding_idx=cfg.pad_index
            )

        # RNN stack
        RNNClass = _RNN_MAP[cfg.cell_type]
        self.rnn = RNNClass(
            input_size=cfg.embedding_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.encoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.encoder_layers > 1 else 0.0
        )

    def forward(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor
    ) -> Tuple[torch.Tensor, Any]:
        # embed: (B, T_src, D_emb)
        embedded = self.embedding(src)

        # pack for RNN
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden_state = self.rnn(packed)

        # unpack to (B, T_src, hidden_dim)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            padding_value=self.cfg.pad_index
        )
        return outputs, hidden_state


# ───────────────────────── 3. Attention ─────────────────────────────
class DotProductAttention(nn.Module):
    """
    Global dot-product attention:
      score_{t,i} = h_dec(t) · h_enc(i)
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (B, hidden_dim)
        encoder_outputs: torch.Tensor,  # (B, T_src, hidden_dim)
        mask: torch.Tensor              # (B, T_src)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # scores: (B, T_src)
        scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        alignments = F.softmax(scores, dim=1)                 # (B, T_src)
        context    = torch.bmm(alignments.unsqueeze(1),      # (B,1,T_src)
                               encoder_outputs
                              ).squeeze(1)                   # (B, hidden_dim)
        return context, alignments


# ───────────────────────── 4. Decoder ───────────────────────────────
class DecoderWithAttention(nn.Module):
    """
    Decoder RNN that at each step:
      1) embeds input
      2) attends over encoder_outputs
      3) feeds [embed; context] to RNN
      4) projects to vocab logits
    """
    def __init__(self, cfg: Seq2SeqAttentionConfig):
        super().__init__()
        self.cfg = cfg

        self.embedding = nn.Embedding(cfg.target_vocab_size,
                                      cfg.embedding_dim,
                                      padding_idx=cfg.pad_index)
        self.attention = DotProductAttention()

        rnn_input_size = cfg.embedding_dim + cfg.hidden_dim
        RNNClass = _RNN_MAP[cfg.cell_type]
        self.rnn = RNNClass(
            input_size=rnn_input_size,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.decoder_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.decoder_layers > 1 else 0.0
        )
        self.output_projection = nn.Linear(cfg.hidden_dim,
                                           cfg.target_vocab_size)

    def forward(
        self,
        input_token: torch.LongTensor,    # (B,1)
        last_hidden: Any,
        encoder_outputs: torch.Tensor,    # (B, T_src, hidden_dim)
        encoder_mask: torch.Tensor        # (B, T_src)
    ) -> Tuple[torch.Tensor, Any, torch.Tensor]:
        # embed: (B,1,D_emb)
        emb = self.embedding(input_token)

        # extract last layer of hidden
        if isinstance(last_hidden, tuple):
            dec_h = last_hidden[0][-1]  # (B, hidden_dim)
        else:
            dec_h = last_hidden[-1]     # (B, hidden_dim)

        # attention
        context, alignments = self.attention(dec_h,
                                             encoder_outputs,
                                             encoder_mask)
        # concat: (B,1, D_emb+hidden_dim)
        rnn_in = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, new_hidden = self.rnn(rnn_in, last_hidden)  # (B,1,hidden_dim)
        logits = self.output_projection(output)              # (B,1,V)
        return logits, new_hidden, alignments


# ──────────────────── 5. Seq2SeqAttention ─────────────────────────
class Seq2SeqAttention(nn.Module):
    """
    Full encoder-decoder with attention.
    Supports forward, greedy_decode, and beam_search_decode.
    """
    def __init__(self, cfg: Seq2SeqAttentionConfig):
        super().__init__()
        self.cfg     = cfg
        self.encoder = EncoderWithOutputs(cfg)
        self.decoder = DecoderWithAttention(cfg)

    def forward(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
        tgt: torch.LongTensor,
        *,
        teacher_forcing_ratio: float = 0.5
    ) -> torch.Tensor:
        B, T_tgt = tgt.size()
        device   = src.device
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)
        enc_mask  = (src != self.cfg.pad_index).to(device)
        dec_hidden = _align_hidden_state(enc_hidden,
                                        self.cfg.decoder_layers)

        logits_all = torch.zeros(B, T_tgt,
                                 self.cfg.target_vocab_size,
                                 device=device)
        dec_input  = tgt[:, 0].unsqueeze(1)  # (B,1)

        for t in range(1, T_tgt):
            step_logits, dec_hidden, _ = self.decoder(
                dec_input, dec_hidden,
                enc_outputs, enc_mask
            )
            logits_all[:, t] = step_logits.squeeze(1)
            if torch.rand(1).item() < teacher_forcing_ratio:
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                dec_input = step_logits.argmax(-1)

        return logits_all

    def greedy_decode(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
        *,
        max_len: int = 50
    ) -> torch.LongTensor:
        B = src.size(0)
        device = src.device
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)
        enc_mask = (src != self.cfg.pad_index).to(device)
        dec_hidden = _align_hidden_state(enc_hidden,
                                        self.cfg.decoder_layers)

        dec_input = torch.full((B,1),
                               self.cfg.sos_index,
                               device=device,
                               dtype=torch.long)
        generated = []
        for _ in range(max_len):
            logits, dec_hidden, _ = self.decoder(
                dec_input, dec_hidden,
                enc_outputs, enc_mask
            )
            dec_input = logits.argmax(-1)  # (B,1)
            generated.append(dec_input)
        return torch.cat(generated, dim=1)  # (B, max_len)

    def beam_search_decode(
        self,
        src: torch.LongTensor,
        src_lengths: torch.LongTensor,
        *,
        beam_size: int = 5,
        max_len:   int = 50
    ) -> torch.LongTensor:
        """
        Beam-search decoding (batch_size=1 only).
        Returns best seq (1, L) without leading <sos>.
        """
        B = src.size(0)
        assert B == 1, "beam_search_decode only supports batch_size=1"

        device = src.device
        enc_outputs, enc_hidden = self.encoder(src, src_lengths)
        enc_mask  = (src != self.cfg.pad_index).to(device)
        dec_hidden = _align_hidden_state(enc_hidden,
                                        self.cfg.decoder_layers)

        # beams: list of (token_list, score, hidden_state)
        beams = [([self.cfg.sos_index], 0.0, dec_hidden)]
        completed = []

        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden in beams:
                last = seq[-1]
                if last == self.cfg.eos_index:
                    completed.append((seq, score))
                    continue

                inp = torch.tensor([[last]],
                                   device=device)
                logits, new_hidden, _ = self.decoder(
                    inp, hidden,
                    enc_outputs, enc_mask
                )
                # logits: (1,1,V) → (V,)
                log_probs = F.log_softmax(logits.squeeze(1), dim=-1)[0]

                topk_vals, topk_idx = log_probs.topk(beam_size)
                for lp, idx in zip(topk_vals.tolist(),
                                   topk_idx.tolist()):
                    # detach hidden state for this candidate
                    if isinstance(new_hidden, tuple):
                        h, c = new_hidden
                        nh = (h.detach().clone(),
                              c.detach().clone())
                    else:
                        nh = new_hidden.detach().clone()
                    all_candidates.append((
                        seq + [idx],
                        score + lp,
                        nh
                    ))

            if not all_candidates:
                break

            # keep top beam_size
            beams = sorted(all_candidates,
                           key=lambda x: x[1],
                           reverse=True)[:beam_size]
            # if all beams ended, stop early
            if all(b[-1] == self.cfg.eos_index for b, _, _ in beams):
                completed.extend((b, s) for b, s, _ in beams)
                break

        if not completed:
            completed = [(b, s) for b, s, _ in beams]

        # pick best
        best_seq, _ = max(completed, key=lambda x: x[1])
        # drop leading <sos>
        if best_seq and best_seq[0] == self.cfg.sos_index:
            best_seq = best_seq[1:]
        return torch.tensor(best_seq,
                            dtype=torch.long,
                            device=device).unsqueeze(0)
