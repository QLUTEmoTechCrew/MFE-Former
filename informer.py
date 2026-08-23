from torch import nn

from embed import PositionalEmbedding
from encoder import Encoder


class Informer(nn.Module):
    def __init__(
        self,
        d_k=32,
        d_v=32,
        d_model=256,
        d_ff=1024,
        n_heads=8,
        e_layer=3,
        e_stack=3,
        dropout=0.1,
        c=5,
    ):
        super().__init__()
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(
            d_k=d_k,
            d_v=d_v,
            d_model=d_model,
            d_ff=d_ff,
            n_heads=n_heads,
            n_layer=e_layer,
            n_stack=e_stack,
            dropout=dropout,
            c=c,
        )

    def forward(self, x, mask=None):
        embedded = self.input_dropout(x + self.position_embedding(x))
        return self.encoder(embedded, mask)
