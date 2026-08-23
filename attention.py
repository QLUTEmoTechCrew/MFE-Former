import math

import torch
from torch import nn


def _expand_attention_mask(attn_mask, batch_size, num_heads):
    if attn_mask is None:
        return None
    if attn_mask.dim() == 3:
        attn_mask = attn_mask.unsqueeze(1)
    if attn_mask.dim() != 4:
        raise ValueError("attn_mask must have shape [B, Lq, Lk] or [B, H, Lq, Lk].")
    if attn_mask.size(0) != batch_size:
        raise ValueError("attn_mask batch size does not match the attention input.")
    if attn_mask.size(1) == 1:
        attn_mask = attn_mask.expand(-1, num_heads, -1, -1)
    return attn_mask.bool()


class FullAttention(nn.Module):
    def __init__(self, d_k, d_v, d_model, n_heads, dropout, mix=False):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.mix = mix
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual = input_Q
        batch_size = input_Q.size(0)
        query = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        mask = _expand_attention_mask(attn_mask, batch_size, self.n_heads)
        if mask is not None:
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention, value).transpose(1, 2).contiguous()
        context = context.view(batch_size, input_Q.size(1), self.n_heads * self.d_v)
        return self.norm(residual + self.dropout(self.fc(context)))


class ProbAttention(nn.Module):
    """Deterministic ProbSparse attention for emotional-word sequences."""

    def __init__(self, d_k, d_v, d_model, n_heads, c, dropout, mix=False):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.c = c
        self.mix = mix
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_Q, input_K, input_V, attn_mask=None):
        residual = input_Q
        batch_size, query_length, _ = input_Q.shape
        key_length = input_K.size(1)
        query = self.W_Q(input_Q).view(batch_size, query_length, self.n_heads, self.d_k).transpose(1, 2)
        key = self.W_K(input_K).view(batch_size, key_length, self.n_heads, self.d_k).transpose(1, 2)
        value = self.W_V(input_V).view(batch_size, key_length, self.n_heads, self.d_v).transpose(1, 2)

        all_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.d_k)
        sparsity = all_scores.max(dim=-1).values - all_scores.mean(dim=-1)
        top_count = min(query_length, max(1, int(self.c * math.log(max(query_length, 2)))))
        top_index = sparsity.topk(top_count, dim=-1, sorted=False).indices
        selected_scores = all_scores.gather(
            2,
            top_index.unsqueeze(-1).expand(-1, -1, -1, key_length),
        )

        mask = _expand_attention_mask(attn_mask, batch_size, self.n_heads)
        if mask is not None:
            selected_mask = mask.gather(
                2,
                top_index.unsqueeze(-1).expand(-1, -1, -1, key_length),
            )
            selected_scores = selected_scores.masked_fill(
                selected_mask,
                torch.finfo(selected_scores.dtype).min,
            )

        selected_attention = torch.softmax(selected_scores, dim=-1)
        selected_context = torch.matmul(selected_attention, value)
        context = value.mean(dim=-2, keepdim=True).expand(
            batch_size,
            self.n_heads,
            query_length,
            self.d_v,
        ).clone()
        context.scatter_(
            2,
            top_index.unsqueeze(-1).expand(-1, -1, -1, self.d_v),
            selected_context,
        )
        context = context.transpose(1, 2).contiguous().view(
            batch_size,
            query_length,
            self.n_heads * self.d_v,
        )
        return self.norm(residual + self.dropout(self.fc(context)))
