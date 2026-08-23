import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityGuidedSpeechReconstructor(nn.Module):
    """Use InfoID queries and learned ``F_id`` memory to reconstruct word-level Mel vectors."""

    def __init__(
        self,
        model_dim=256,
        output_dim=80,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        max_words=256,
    ):
        super().__init__()
        layer = nn.TransformerDecoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.position = nn.Parameter(torch.zeros(1, max_words, model_dim))
        nn.init.normal_(self.position, std=0.02)
        self.output = nn.Linear(model_dim, output_dim)

    def forward(self, identity_embedding, identity_features, target_length):
        if target_length > self.position.size(1):
            raise ValueError(
                f"target_length {target_length} exceeds max_words {self.position.size(1)}."
            )
        query = identity_embedding.unsqueeze(1) + self.position[:, :target_length]
        decoded = self.decoder(tgt=query, memory=identity_features)
        return self.output(decoded)


def masked_reconstruction_loss(reconstruction, target, mask=None):
    squared_error = (reconstruction - target) ** 2
    if mask is None:
        return squared_error.mean()
    if mask.shape != reconstruction.shape[:-1]:
        raise ValueError("Reconstruction mask must have shape [B, N].")
    expanded_mask = mask.bool().unsqueeze(-1).expand_as(reconstruction)
    if not torch.any(expanded_mask):
        return squared_error.sum() * 0.0
    return squared_error.masked_select(expanded_mask).mean()
