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


class EmbeddingMLP(nn.Module):
    """
    Mean-pools pre-trained token embeddings, then applies a two-layer MLP.

    Parameters
    ----------
    embedding_matrix : np.ndarray, shape (vocab_size, embed_dim)
    num_classes      : number of output classes
    hidden_dim       : hidden layer width
    dropout          : dropout probability
    freeze_emb       : if True, embedding weights are not updated during training
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
        freeze_emb: bool = False,
    ):
        super().__init__()
        _, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_emb, padding_idx=0
        )
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)
        emb = self.embedding(x) * mask
        doc = emb.sum(1) / mask.sum(1).clamp(min=1)
        return self.fc(doc)


class TextCNN(nn.Module):
    """
    Parallel Conv1d filters over token embeddings with global max-pooling.

    Parameters
    ----------
    embedding_matrix : np.ndarray, shape (vocab_size, embed_dim)
    num_classes      : number of output classes
    num_filters      : number of filters per kernel size
    kernel_sizes     : list of filter heights (e.g. [2, 3, 4])
    dropout          : dropout probability
    freeze_emb       : if True, embedding weights are frozen
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        num_classes: int,
        num_filters: int = 128,
        kernel_sizes: list = None,
        dropout: float = 0.3,
        freeze_emb: bool = False,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [2, 3, 4]
        _, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_emb, padding_idx=0
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(embed_dim, num_filters, kernel_size=k, padding=k // 2)
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x):
        emb = self.embedding(x).permute(0, 2, 1)
        pooled = [F.relu(conv(emb)).max(dim=2).values for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


class AdditiveAttention(nn.Module):
    """
    Weighted average over LSTM hidden states via a learned linear layer + softmax.
    Padding positions are masked out before softmax.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask=None):
        scores = self.attn(h).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=1)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)
        return context, weights


class BiLSTMAttention(nn.Module):
    """
    Bidirectional LSTM with additive attention for text classification.

    Parameters
    ----------
    embedding_matrix : np.ndarray, shape (vocab_size, embed_dim)
    num_classes      : number of output classes
    hidden_dim       : LSTM hidden state size per direction
    num_layers       : number of stacked LSTM layers
    dropout          : dropout probability
    freeze_emb       : freeze pre-trained embeddings during training
    """

    def __init__(
        self,
        embedding_matrix: np.ndarray,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        freeze_emb: bool = False,
    ):
        super().__init__()
        _, embed_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            torch.FloatTensor(embedding_matrix), freeze=freeze_emb, padding_idx=0
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        mask = x != 0
        emb = self.dropout(self.embedding(x))
        h, _ = self.lstm(emb)
        ctx, _ = self.attention(h, mask)
        return self.fc(self.dropout(ctx))
