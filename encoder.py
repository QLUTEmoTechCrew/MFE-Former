import torch
import torch.nn.functional as F
from torch import nn

from attention import ProbAttention


class ConvLayer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ELU()
        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.down_conv(x.transpose(1, 2))
        x = self.activation(self.norm(x))
        return self.max_pool(x).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, d_k, d_v, d_model, d_ff, n_heads, c, dropout):
        super().__init__()
        self.attention = ProbAttention(d_k, d_v, d_model, n_heads, c, dropout)
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        x = self.attention(x, x, x, attn_mask=attn_mask)
        residual = x
        hidden = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        hidden = self.dropout(self.conv2(hidden).transpose(1, 2))
        return self.norm(residual + hidden)


class Encoder(nn.Module):
    """Three stacked granularities with a common final length of ``N / 4``."""

    def __init__(
        self,
        d_k,
        d_v,
        d_model,
        d_ff,
        n_heads,
        n_layer=3,
        n_stack=3,
        d_feature=None,
        d_mark=None,
        dropout=0.1,
        c=5,
    ):
        super().__init__()
        if n_layer != 3 or n_stack != 3:
            raise ValueError("The paper-aligned encoder requires three layers and three stacks.")
        self.stacks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for stack_index in range(n_stack):
            depth = n_layer - stack_index
            modules = nn.ModuleList()
            for layer_index in range(depth):
                modules.append(EncoderLayer(d_k, d_v, d_model, d_ff, n_heads, c, dropout))
                if layer_index < depth - 1:
                    modules.append(ConvLayer(d_model))
            self.stacks.append(modules)
            self.norms.append(nn.LayerNorm(d_model))

    def forward(self, x, mask=None):
        if x.size(1) % 4 != 0:
            raise ValueError(f"The emotional-word sequence length must be divisible by 4, got {x.size(1)}.")

        target_length = x.size(1) // 4
        scale_features = []
        for stack_index, (stack, norm) in enumerate(zip(self.stacks, self.norms)):
            input_length = x.size(1) // (2 ** stack_index)
            branch = x[:, -input_length:, :]
            for module in stack:
                branch = module(branch)
            branch = norm(branch)
            if branch.size(1) != target_length:
                raise RuntimeError(
                    f"Scale {stack_index} produced {branch.size(1)} tokens; expected {target_length}."
                )
            scale_features.append(branch)
        return torch.cat(scale_features, dim=1), scale_features
