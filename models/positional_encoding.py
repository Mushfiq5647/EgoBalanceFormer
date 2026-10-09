"""
Sinusoidal positional encoding for Transformer sequences.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Adds fixed sinusoidal positional information to a sequence.

    Formula:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    Args:
        d_model:  feature dimension (must match transformer d_model)
        max_len:  maximum sequence length (default 512, 11 is more than enough here)
        dropout:  applied after adding encoding
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                          # (L, D)
        position = torch.arange(max_len).unsqueeze(1).float()       # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))                 # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D)
        Returns:
            x + positional encoding, same shape
        """
        return self.dropout(x + self.pe[:, : x.size(1)])
