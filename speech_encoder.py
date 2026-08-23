import torch
import torch.nn as nn

from informer import Informer


class SpeechEncoder(nn.Module):
    """Encode every emotional-word spectrogram into one CNN-BiLSTM vector."""

    def __init__(self, input_dim=80, model_dim=256, lstm_layers=1, dropout=0.1):
        super().__init__()
        self.local_cnn = nn.Sequential(
            nn.Conv1d(input_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(model_dim, model_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(model_dim),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=model_dim,
            hidden_size=model_dim // 2,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, speech, frame_mask=None):
        if speech.dim() != 4:
            raise ValueError(f"Expected word-level speech [B, N, T, F], got {tuple(speech.shape)}.")
        batch_size, num_words, num_frames, feature_dim = speech.shape
        flat_speech = speech.reshape(batch_size * num_words, num_frames, feature_dim)
        if frame_mask is None:
            flat_mask = torch.ones(
                batch_size * num_words,
                num_frames,
                dtype=torch.bool,
                device=speech.device,
            )
        else:
            if frame_mask.shape != speech.shape[:3]:
                raise ValueError("frame_mask must have shape [B, N, T].")
            flat_mask = frame_mask.reshape(batch_size * num_words, num_frames).bool()

        lengths = flat_mask.sum(dim=1)
        if torch.any(lengths == 0):
            raise ValueError("Every emotional word must contain at least one valid Mel frame.")

        local = self.local_cnn(flat_speech.transpose(1, 2)).transpose(1, 2)
        local = local * flat_mask.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            local,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        word_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        word_features = self.norm(word_features)
        return word_features.view(batch_size, num_words, -1)


class MultiScaleEmotionPerception(nn.Module):
    """Three-stack ProbSparse encoder producing ``X_MS`` with length ``3N/4``."""

    def __init__(self, model_dim=256, num_heads=8, dropout=0.1, sparsity_factor=5):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads.")
        head_dim = model_dim // num_heads
        self.encoder = Informer(
            d_k=head_dim,
            d_v=head_dim,
            d_model=model_dim,
            d_ff=model_dim * 4,
            n_heads=num_heads,
            e_layer=3,
            e_stack=3,
            dropout=dropout,
            c=sparsity_factor,
        )

    def forward(self, x):
        return self.encoder(x)
