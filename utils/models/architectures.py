"""
PyTorch text-classification architectures shared across task notebooks.

Exported classes
----------------
EmbeddingMLP       — mean-pooled embeddings + two-layer MLP
TextCNN            — parallel Conv1d filters with global max-pooling
AdditiveAttention  — learned weighted average over LSTM hidden states
BiLSTMAttention    — bidirectional LSTM with additive attention
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_embedding_layer(
    embedding_matrix: np.ndarray | None,
    vocab_size: int | None,
    embed_dim: int | None,
    freeze_emb: bool = False,
):
    if embedding_matrix is not None:
        return nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_emb, padding_idx=0
        )

    if vocab_size is None or embed_dim is None:
        raise ValueError("vocab_size and embed_dim are required when embedding_matrix is omitted.")

    return nn.Embedding(vocab_size, embed_dim, padding_idx=0)

class EmbeddingMLP(nn.Module):
    """
    Mean-pools pre-trained token embeddings, then applies a two-layer MLP.
    Serves as the lightweight "GloVe + MLP" step in the pipeline.

    Parameters
    ----------
    embedding_matrix : np.ndarray | None, shape (vocab_size, embed_dim)
                       If None, vocab_size and embed_dim must be provided.
    num_classes      : number of output classes
    hidden_dim       : hidden layer width
    dropout          : dropout probability
    freeze_emb       : if True, embedding weights are not updated during training
    """
    def __init__(self, embedding_matrix: np.ndarray | None, num_classes: int,
                 hidden_dim: int = 256, dropout: float = 0.3,
                 freeze_emb: bool = False,
                 vocab_size: int | None = None, embed_dim: int | None = None):
        super().__init__()
        if embedding_matrix is not None:
            vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = build_embedding_layer(
            embedding_matrix, vocab_size=vocab_size, embed_dim=embed_dim, freeze_emb=freeze_emb
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)       # (B, L, 1) — zero out PAD
        emb  = self.embedding(x) * mask             # (B, L, D)
        doc  = emb.sum(1) / mask.sum(1).clamp(min=1)  # masked mean → (B, D)
        return self.fc(doc)

class TextCNN(nn.Module):
    """
    Parallel Conv1d filters over token embeddings with global max-pooling.

    Parameters
    ----------
    embedding_matrix : np.ndarray | None, shape (vocab_size, embed_dim)
                       If None, vocab_size and embed_dim must be provided.
    num_classes      : number of output classes
    num_filters      : number of filters per kernel size
    kernel_sizes     : list of filter heights (e.g. [2, 3, 4])
    dropout          : dropout probability
    freeze_emb       : if True, embedding weights are frozen
    """
    def __init__(self, embedding_matrix: np.ndarray | None, num_classes: int,
                 num_filters: int = 128, kernel_sizes: list = None,
                 dropout: float = 0.3, freeze_emb: bool = False,
                 vocab_size: int | None = None, embed_dim: int | None = None):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]

        if embedding_matrix is not None:
            vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = build_embedding_layer(
            embedding_matrix, vocab_size=vocab_size, embed_dim=embed_dim, freeze_emb=freeze_emb
        )
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)    # (B, D, L)
        pooled = [
            F.relu(conv(emb)).max(dim=2).values      # (B, F)
            for conv in self.convs
        ]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))

class AdditiveAttention(nn.Module):
    """
    Computes a weighted average over LSTM hidden states.
    Weights are learned via a single linear layer + softmax.
    Padding positions are masked out before softmax.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask=None):
        scores = self.attn(h).squeeze(-1)                       # (B, L)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)                  # (B, L)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)        # (B, H)
        return context, weights

class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with additive attention for text classification.

    Parameters
    ----------
    embedding_matrix : np.ndarray | None, shape (vocab_size, embed_dim)
                       If None, vocab_size and embed_dim must be provided.
    num_classes      : number of output classes
    hidden_dim       : LSTM hidden state size per direction
    num_layers       : number of stacked LSTM layers
    dropout          : dropout probability
    freeze_emb       : freeze pre-trained embeddings during training
    """
    def __init__(self, embedding_matrix: np.ndarray | None, num_classes: int,
                 hidden_dim: int = 128, num_layers: int = 2,
                 dropout: float = 0.3, freeze_emb: bool = False,
                 vocab_size: int | None = None, embed_dim: int | None = None):
        super().__init__()
        if embedding_matrix is not None:
            vocab_size, embed_dim = embedding_matrix.shape
        self.embedding = build_embedding_layer(
            embedding_matrix, vocab_size=vocab_size, embed_dim=embed_dim, freeze_emb=freeze_emb
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.attention = AdditiveAttention(hidden_dim * 2)   # *2 for bidirectional
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = (x != 0)                              # (B, L) True = real token
        emb  = self.dropout(self.embedding(x))       # (B, L, D)
        h, _ = self.lstm(emb)                        # (B, L, 2*H)
        ctx, _ = self.attention(h, mask)             # (B, 2*H)
        return self.fc(self.dropout(ctx))
