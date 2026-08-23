from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveProjectionHead(nn.Module):
    def __init__(self, model_dim=256, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(model_dim, projection_dim, kernel_size=1),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def _remove_diagonal(similarity):
    lower = torch.tril(similarity, diagonal=-1)[..., :, :-1]
    upper = torch.triu(similarity, diagonal=1)[..., :, 1:]
    return lower + upper


def instance_contrastive_loss(first, second, temperature=0.2):
    batch_size = first.size(0)
    if batch_size == 1:
        return first.new_tensor(0.0)
    features = torch.cat([first, second], dim=0).transpose(0, 1)
    similarity = torch.matmul(features, features.transpose(1, 2)) / temperature
    negative_log_probability = -F.log_softmax(_remove_diagonal(similarity), dim=-1)
    index = torch.arange(batch_size, device=first.device)
    return (
        negative_log_probability[:, index, batch_size + index - 1].mean()
        + negative_log_probability[:, batch_size + index, index].mean()
    ) / 2


def temporal_contrastive_loss(first, second, temperature=0.2):
    steps = min(first.size(1), second.size(1))
    if steps == 1:
        return first.new_tensor(0.0)
    first = first[:, :steps]
    second = second[:, :steps]
    features = torch.cat([first, second], dim=1)
    similarity = torch.matmul(features, features.transpose(1, 2)) / temperature
    negative_log_probability = -F.log_softmax(_remove_diagonal(similarity), dim=-1)
    index = torch.arange(steps, device=first.device)
    return (
        negative_log_probability[:, index, steps + index - 1].mean()
        + negative_log_probability[:, steps + index, index].mean()
    ) / 2


class MultiScaleContrastiveModule(nn.Module):
    """Contrast aligned representations from distinct multi-scale branches."""

    def __init__(self, model_dim=256, projection_dim=128, temperature=0.2):
        super().__init__()
        self.projector = ContrastiveProjectionHead(model_dim, projection_dim)
        self.temperature = temperature

    def forward(self, scale_features):
        if len(scale_features) < 2:
            raise ValueError("At least two scale representations are required for contrastive learning.")
        projected = [F.normalize(self.projector(feature), dim=-1) for feature in scale_features]
        pair_losses = []
        for first_index, second_index in combinations(range(len(projected)), 2):
            first = projected[first_index]
            second = projected[second_index]
            pair_losses.append(
                0.5 * instance_contrastive_loss(first, second, self.temperature)
                + 0.5 * temporal_contrastive_loss(first, second, self.temperature)
            )
        return torch.stack(pair_losses).mean(), projected
