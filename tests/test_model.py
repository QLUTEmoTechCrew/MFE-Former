import torch

from MFEFormer import MFEFormer
from reconstruction import masked_reconstruction_loss


def _small_model():
    return MFEFormer(
        input_dim=80,
        identity_dim=7,
        clip_feature_dim=512,
        model_dim=32,
        projection_dim=16,
        num_heads=4,
        dropout=0.0,
    )


def test_word_and_multiscale_shapes():
    torch.manual_seed(0)
    speech = torch.randn(2, 72, 12, 80)
    frame_mask = torch.ones(2, 72, 12, dtype=torch.bool)
    model = _small_model().eval()

    logits, aux = model(
        speech,
        frame_mask=frame_mask,
        clip_text_features=torch.randn(2, 512),
        return_aux=True,
    )

    assert logits.shape == (2, 2)
    assert aux["word_features"].shape == (2, 72, 32)
    assert aux["x_ms"].shape == (2, 54, 32)
    assert [feature.shape for feature in aux["scale_features"]] == [
        (2, 18, 32),
        (2, 18, 32),
        (2, 18, 32),
    ]
    assert aux["reconstruction"].shape == (2, 72, 80)
    assert torch.isfinite(aux["contrastive_loss"])
    assert torch.isfinite(aux["regularization_loss"])
    assert torch.isfinite(aux["reconstruction_loss"])


def test_masked_reconstruction_is_elementwise_mean():
    reconstruction = torch.zeros(1, 2, 80)
    target = torch.ones_like(reconstruction)
    mask = torch.ones(1, 2, dtype=torch.bool)

    assert masked_reconstruction_loss(reconstruction, target, mask).item() == 1.0


def test_identity_is_required():
    model = _small_model().eval()
    speech = torch.randn(1, 72, 8, 80)

    try:
        model(speech)
    except ValueError as error:
        assert "identity" in str(error).lower()
    else:
        raise AssertionError("MFEFormer must not silently replace missing identity information.")


def test_inference_skips_auxiliary_branches():
    model = _small_model().eval()
    speech = torch.randn(1, 72, 8, 80)
    identity = torch.randn(1, 7)

    logits = model(speech, identity=identity)

    assert logits.shape == (1, 2)
