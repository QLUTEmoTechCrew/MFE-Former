import torch
import torch.nn as nn

from contrastive_learning import MultiScaleContrastiveModule
from disentanglement import EmotionIdentityDisentangler
from identity_encoder import CLIPTextIdentityEncoder, IdentityEncoder, build_identity_prompts
from reconstruction import IdentityGuidedSpeechReconstructor, masked_reconstruction_loss
from speech_encoder import MultiScaleEmotionPerception, SpeechEncoder


class BernoulliJointDecision(nn.Module):
    def __init__(self, model_dim=256, num_scales=3, dropout=0.1):
        super().__init__()
        self.scale_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(model_dim),
                    nn.Linear(model_dim, model_dim // 2),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(model_dim // 2, 1),
                )
                for _ in range(num_scales)
            ]
        )
        self.weight_logits = nn.Parameter(torch.zeros(num_scales))

    def forward(self, scale_features):
        probabilities = []
        for features, head in zip(scale_features, self.scale_heads):
            probabilities.append(torch.sigmoid(head(features.mean(dim=1))))
        scale_probabilities = torch.cat(probabilities, dim=-1)
        scale_weights = torch.softmax(self.weight_logits, dim=0)
        depression_probability = torch.sum(
            scale_probabilities * scale_weights.unsqueeze(0),
            dim=-1,
        ).clamp(1e-6, 1.0 - 1e-6)
        logits = torch.stack(
            [torch.log1p(-depression_probability), torch.log(depression_probability)],
            dim=-1,
        )
        return logits, scale_probabilities, scale_weights


class MFEFormer(nn.Module):
    """Paper-aligned MFE-Former operating on a sequence of emotional words."""

    def __init__(
        self,
        input_dim=80,
        identity_dim=7,
        clip_feature_dim=512,
        model_dim=256,
        projection_dim=128,
        num_classes=2,
        num_heads=8,
        dropout=0.1,
        clip_model=None,
        clip_processor=None,
        clip_tokenizer=None,
        freeze_clip=True,
        max_words=256,
    ):
        super().__init__()
        if num_classes != 2:
            raise ValueError("BernoulliJointDecision supports binary classification only.")
        self.speech_encoder = SpeechEncoder(input_dim=input_dim, model_dim=model_dim, dropout=dropout)
        self.multi_scale_encoder = MultiScaleEmotionPerception(
            model_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.text_identity_encoder = CLIPTextIdentityEncoder(
            clip_feature_dim=clip_feature_dim,
            model_dim=model_dim,
            dropout=dropout,
            clip_model=clip_model,
            clip_processor=clip_processor,
            clip_tokenizer=clip_tokenizer,
            freeze_clip=freeze_clip,
        )
        self.numeric_identity_encoder = IdentityEncoder(
            identity_dim=identity_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.contrastive = MultiScaleContrastiveModule(model_dim, projection_dim)
        self.disentangler = EmotionIdentityDisentangler(model_dim=model_dim, dropout=dropout)
        self.reconstructor = IdentityGuidedSpeechReconstructor(
            model_dim=model_dim,
            output_dim=input_dim,
            num_layers=3,
            num_heads=num_heads,
            dropout=dropout,
            max_words=max_words,
        )
        self.classifier = BernoulliJointDecision(model_dim, num_scales=3, dropout=dropout)

    def _encode_identity(
        self,
        speech,
        identity_texts=None,
        demographics=None,
        clip_text_features=None,
        identity=None,
    ):
        if demographics is not None:
            identity_texts = build_identity_prompts(demographics)
        if clip_text_features is not None or identity_texts is not None:
            return self.text_identity_encoder(
                identity_texts=identity_texts,
                clip_text_features=clip_text_features,
                device=speech.device,
            )
        if identity is not None:
            identity = identity.to(device=speech.device, dtype=speech.dtype)
            return self.numeric_identity_encoder(identity), {"numeric_identity": identity}
        raise ValueError(
            "Explicit identity information is required: provide clip_text_features, "
            "identity_texts/demographics with a configured CLIP model, or identity."
        )

    @staticmethod
    def _word_mel_target(speech, frame_mask):
        if frame_mask is None:
            return speech.mean(dim=2), torch.ones(
                speech.shape[:2], dtype=torch.bool, device=speech.device
            )
        weights = frame_mask.to(dtype=speech.dtype).unsqueeze(-1)
        target = (speech * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        return target, frame_mask.any(dim=2)

    def forward(
        self,
        speech,
        frame_mask=None,
        identity_texts=None,
        demographics=None,
        clip_text_features=None,
        identity=None,
        reconstruction_mask=None,
        return_aux=False,
        compute_losses=True,
    ):
        identity_embedding, identity_aux = self._encode_identity(
            speech,
            identity_texts=identity_texts,
            demographics=demographics,
            clip_text_features=clip_text_features,
            identity=identity,
        )
        word_features = self.speech_encoder(speech, frame_mask)
        x_ms, scale_features = self.multi_scale_encoder(word_features)
        need_losses = return_aux and compute_losses
        emotion_features, identity_features, regularization_loss, disentangle_aux = self.disentangler(
            x_ms,
            identity_embedding,
            compute_loss=need_losses,
        )
        split_sizes = [feature.size(1) for feature in scale_features]
        emotion_scales = list(torch.split(emotion_features, split_sizes, dim=1))
        logits, scale_probabilities, scale_weights = self.classifier(emotion_scales)
        if not return_aux:
            return logits

        reconstruction = None
        reconstruction_target = None
        projected_scales = None
        contrastive_loss = None
        reconstruction_loss = None
        if need_losses:
            contrastive_loss, projected_scales = self.contrastive(scale_features)
            reconstruction = self.reconstructor(
                identity_embedding,
                identity_features,
                target_length=speech.size(1),
            )
            reconstruction_target, valid_word_mask = self._word_mel_target(speech, frame_mask)
            if reconstruction_mask is None:
                reconstruction_mask = valid_word_mask
            reconstruction_loss = masked_reconstruction_loss(
                reconstruction,
                reconstruction_target,
                reconstruction_mask,
            )

        aux = {
            "word_features": word_features,
            "identity_embedding": identity_embedding,
            "x_ms": x_ms,
            "scale_features": scale_features,
            "projected_scales": projected_scales,
            "emotion_features": emotion_features,
            "identity_features": identity_features,
            "reconstruction": reconstruction,
            "reconstruction_target": reconstruction_target,
            "scale_probs": scale_probabilities,
            "scale_weights": scale_weights,
            **identity_aux,
            **disentangle_aux,
        }
        if need_losses:
            aux.update(
                {
                    "contrastive_loss": contrastive_loss,
                    "regularization_loss": regularization_loss,
                    "reconstruction_loss": reconstruction_loss,
                }
            )
        return logits, aux


if __name__ == "__main__":
    torch.manual_seed(42)
    model = MFEFormer(model_dim=64, projection_dim=32, num_heads=4, dropout=0.0).eval()
    dummy_speech = torch.randn(2, 72, 24, 80)
    dummy_mask = torch.ones(2, 72, 24, dtype=torch.bool)
    dummy_clip_features = torch.randn(2, 512)
    with torch.no_grad():
        logits, aux = model(
            dummy_speech,
            frame_mask=dummy_mask,
            clip_text_features=dummy_clip_features,
            return_aux=True,
        )
    print("Input:", tuple(dummy_speech.shape))
    print("Word features:", tuple(aux["word_features"].shape))
    print("X_MS:", tuple(aux["x_ms"].shape))
    print("Scale features:", [tuple(item.shape) for item in aux["scale_features"]])
    print("Reconstruction:", tuple(aux["reconstruction"].shape))
    print("Logits:", tuple(logits.shape))
    print("Contrastive loss:", float(aux["contrastive_loss"]))
    print("Regularization loss:", float(aux["regularization_loss"]))
    print("Reconstruction loss:", float(aux["reconstruction_loss"]))
