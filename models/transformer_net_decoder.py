import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.nn import TransformerEncoderLayer, TransformerDecoderLayer

if torch.cuda.is_available():
    device = torch.device('cuda')


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        attention = self.attention(value, key, query, mask=None)

        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, n_layers=10, in_dim=100, embed_size=128, heads=4, forward_expansion=4, dropout=0.2):
        super(Encoder, self).__init__()

        self.in_dim = in_dim
        self.embed_size = embed_size
        self.fc1 = nn.Linear(in_dim, embed_size)
        self.position_embedding = PositionalEncoding(d_model=100, dropout=0.2, max_len=1000)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    in_dim,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N = len(x)

        # x = self.position_embedding(x)
        # x = self.fc1(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, x, x, mask=None)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(embed_size=hid_dim, heads=n_heads)
        self.ea = SelfAttention(embed_size=hid_dim, heads=n_heads)

        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        # trg_mask = [batch size, compound sent len]
        # src_mask = [batch size, protein len]

        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        # trg = self.ln(trg + self.do(self.pf(trg)))

        return trg


class Decoder(nn.Module):
    """ compound feature extraction."""
    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, dropout):
        super().__init__()
        self.ln = nn.LayerNorm(hid_dim)
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.attention = SelfAttention(embed_size=hid_dim, heads=n_heads)
        self.dropout = dropout
        self.device = device
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, dropout)
             for _ in range(n_layers)])
        self.ft = nn.Linear(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)


    def forward(self, trg, src, trg_mask=None,src_mask=None):
        # trg = [batch_size, compound len, atom_dim]
        # src = [batch_size, protein len, hid_dim] # encoder output
        trg = self.ft(trg)

        # trg = [batch size, compound len, hid dim]

        for layer in self.layers:
            trg = layer(trg, src)

        # trg = [batch size, compound len, hid dim]
        """Use norm to determine which atom is significant. """
        norm = torch.norm(trg,dim=2)
        # norm = [batch size,compound len]
        norm = F.softmax(norm,dim=1)
        # norm = [batch size,compound len]
        trg = torch.squeeze(trg,dim=0)
        norm = torch.squeeze(norm,dim=0)
        sum = torch.zeros((self.hid_dim)).to(self.device)
        for i in range(norm.shape[0]):
            v = trg[i,]
            v = v * norm[i]
            sum += v
        sum = sum.unsqueeze(dim=0)

        # trg = [batch size,hid_dim]

        return sum



