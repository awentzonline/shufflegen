import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

    def decode(self, y):
        return y - self.pe[:y.size(0), :]

    @property
    def additional_dims(self):
        return 0


class PositionalEncodingCat(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pe = self.pe[:x.shape[0], :].repeat(1, x.shape[1], 1)
        return torch.cat([x, pe], -1)

    def decode(self, y):
        return y[..., :-self.d_model]

    @property
    def additional_dims(self):
        return self.d_model


class LearnablePositionalEncodingCat(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('embedding_ids', torch.arange(max_len))

    def forward(self, x):
        embedding_ids = self.embedding_ids[:len(x)][..., None]
        embeddings = self.embedding(embedding_ids)
        seq, batch = x.shape[:2]
        embeddings = embeddings.repeat(1, batch, 1)
        return torch.cat([x, embeddings], -1)

    def decode(self, y):
        return y[..., :-self.d_model]

    @property
    def additional_dims(self):
        return self.d_model
