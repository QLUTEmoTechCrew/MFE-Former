import torch
import torch.nn as nn


def build_identity_prompt(age="unknown", sex="person", education="unknown"):
    """Build the identity prompt described in the paper."""

    return f"a spectrogram of a {age} years old {sex} who has a {education} education"


def build_identity_prompts(demographics):
    """Convert demographic dicts into CLIP text prompts.

    Args:
        demographics: a list of dicts with optional keys: age, sex, education.
    """

    prompts = []
    for item in demographics:
        prompts.append(
            build_identity_prompt(
                age=item.get("age", "unknown"),
                sex=item.get("sex", "person"),
                education=item.get("education", "unknown"),
            )
        )
    return prompts


class IdentityEncoder(nn.Module):
    """Encodes demographic or speaker identity descriptors."""

    def __init__(
        self,
        identity_dim: int = 16,
        model_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(identity_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def forward(self, identity: torch.Tensor) -> torch.Tensor:
        if identity.dim() != 2:
            raise ValueError(f"Expected identity descriptors [B, D_id], got {tuple(identity.shape)}.")
        return self.net(identity)


class ResidualTextAdapter(nn.Module):
    """Trainable residual adapter for frozen or precomputed CLIP text features."""

    def __init__(self, clip_feature_dim: int = 512, hidden_dim: int | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or clip_feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(clip_feature_dim),
            nn.Linear(clip_feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, clip_feature_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, clip_features: torch.Tensor) -> torch.Tensor:
        return clip_features + self.net(clip_features)


class CLIPTextIdentityEncoder(nn.Module):
    """CLIP-text identity encoder with trainable residual text adapter.

    The paper builds an identity prompt such as:
    "a spectrogram of a {age} years old {sex} who has a {education} education",
    extracts InfoID with CLIP, then uses it to guide identity-aware learning.

    This class supports two practical modes:
    1. pass precomputed CLIP text features through `clip_text_features`;
    2. inject a local CLIP model and processor/tokenizer to encode text prompts.
    """

    def __init__(
        self,
        clip_feature_dim: int = 512,
        model_dim: int = 256,
        adapter_hidden_dim: int | None = None,
        dropout: float = 0.1,
        clip_model=None,
        clip_processor=None,
        clip_tokenizer=None,
        freeze_clip: bool = True,
    ):
        super().__init__()
        self.clip_feature_dim = clip_feature_dim
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.clip_tokenizer = clip_tokenizer
        self.freeze_clip = freeze_clip

        if self.clip_model is not None and self.freeze_clip:
            self.clip_model.eval()
            for param in self.clip_model.parameters():
                param.requires_grad_(False)

        self.text_adapter = ResidualTextAdapter(
            clip_feature_dim=clip_feature_dim,
            hidden_dim=adapter_hidden_dim,
            dropout=dropout,
        )
        self.to_model_dim = nn.Sequential(
            nn.LayerNorm(clip_feature_dim),
            nn.Linear(clip_feature_dim, model_dim),
            nn.LayerNorm(model_dim),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        if self.clip_model is not None and self.freeze_clip:
            self.clip_model.eval()
        return self

    def _move_batch_to_device(self, batch, device):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}

    def encode_texts(self, identity_texts, device):
        if self.clip_model is None:
            raise ValueError(
                "identity_texts require a local CLIP model. Pass clip_text_features "
                "instead, or construct CLIPTextIdentityEncoder with clip_model and processor/tokenizer."
            )

        if self.clip_processor is not None:
            inputs = self.clip_processor(
                text=identity_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        elif self.clip_tokenizer is not None:
            inputs = self.clip_tokenizer(
                identity_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            raise ValueError("A CLIP processor or tokenizer is required to encode identity_texts.")

        inputs = self._move_batch_to_device(inputs, device)
        context = torch.no_grad() if self.freeze_clip else torch.enable_grad()
        with context:
            if hasattr(self.clip_model, "get_text_features"):
                clip_features = self.clip_model.get_text_features(**inputs)
            else:
                outputs = self.clip_model(**inputs)
                clip_features = outputs.text_embeds if hasattr(outputs, "text_embeds") else outputs[0]
        return clip_features

    def forward(self, identity_texts=None, clip_text_features: torch.Tensor | None = None, device=None):
        if clip_text_features is None:
            if identity_texts is None:
                raise ValueError("Either identity_texts or clip_text_features must be provided.")
            device = device or next(self.parameters()).device
            clip_text_features = self.encode_texts(identity_texts, device)

        if device is not None:
            clip_text_features = clip_text_features.to(
                device=device,
                dtype=self.to_model_dim[1].weight.dtype,
            )

        if clip_text_features.dim() != 2:
            raise ValueError(
                f"Expected CLIP text features [B, D_clip], got {tuple(clip_text_features.shape)}."
            )
        if clip_text_features.size(-1) != self.clip_feature_dim:
            raise ValueError(
                f"Expected CLIP feature dim {self.clip_feature_dim}, got {clip_text_features.size(-1)}."
            )

        adapted_clip_features = self.text_adapter(clip_text_features)
        identity_embedding = self.to_model_dim(adapted_clip_features)
        aux = {
            "clip_text_features": clip_text_features,
            "adapted_clip_text_features": adapted_clip_features,
        }
        return identity_embedding, aux
