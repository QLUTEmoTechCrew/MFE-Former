import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_mlp(model_dim, dropout):
    return nn.Sequential(
        nn.Linear(model_dim, model_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(model_dim, model_dim),
        nn.LayerNorm(model_dim),
    )


class EmotionIdentityDisentangler(nn.Module):
    """Learn ``F_emo`` and ``F_id`` and apply equations (12)-(13)."""

    def __init__(self, model_dim=256, dropout=0.1):
        super().__init__()
        self.emotion_mlp = _make_mlp(model_dim, dropout)
        self.identity_mlp = _make_mlp(model_dim, dropout)

    def forward(self, x_ms, identity_embedding, compute_loss=True):
        emotion_features = self.emotion_mlp(x_ms)
        identity_features = self.identity_mlp(x_ms)
        identity_summary = identity_features.mean(dim=1)

        identity_alignment_loss = None
        orthogonal_loss = None
        regularization_loss = None
        if compute_loss:
            identity_alignment_loss = 1.0 - F.cosine_similarity(
                identity_summary,
                identity_embedding,
                dim=-1,
            ).mean()
            orthogonal_loss = torch.abs(
                F.cosine_similarity(emotion_features, identity_features, dim=-1)
            ).mean()
            regularization_loss = identity_alignment_loss + orthogonal_loss

        aux = {
            "identity_features": identity_features,
            "identity_summary": identity_summary,
            "identity_alignment_loss": identity_alignment_loss,
            "orthogonal_loss": orthogonal_loss,
        }
        return emotion_features, identity_features, regularization_loss, aux
