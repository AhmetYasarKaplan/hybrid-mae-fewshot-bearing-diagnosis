from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

__all__ = [
    "SpatialEncoder",
    "SpatialDecoder",
    "HybridMAE",
    "MultiEncoderEmbedder",
    "FusionHead",
    "build_embedder",
    "CNN1D",
    "create_svm_model",
    "create_xgboost_model",
]


class SpatialEncoder(nn.Module):
    """1D convolutional encoder. Returns the latent map and the pooling indices."""

    def __init__(self, in_channels: int = 2, latent_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels

        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        self.enc2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        self.enc3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        self.enc4 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

        self.enc5 = nn.Sequential(
            nn.Conv1d(192, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(latent_channels),
            nn.ReLU(inplace=True),
        )
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        pool_indices, output_sizes = [], []

        for enc, pool in ((self.enc1, self.pool1), (self.enc2, self.pool2),
                          (self.enc3, self.pool3), (self.enc4, self.pool4),
                          (self.enc5, self.pool5)):
            x = enc(x)
            output_sizes.append(x.size())
            x, idx = pool(x)
            pool_indices.append(idx)

        return x, (pool_indices, output_sizes)


class SpatialDecoder(nn.Module):
    """Mirror decoder for SpatialEncoder, used only during pretraining."""

    def __init__(self, out_channels: int = 2, latent_channels: int = 128):
        super().__init__()
        self.latent_channels = latent_channels

        self.unpool5 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(latent_channels, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
        )

        self.unpool4 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(192, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.unpool3 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.unpool2 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=11, stride=1, padding=5),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.unpool1 = nn.MaxUnpool1d(kernel_size=2, stride=2)
        self.dec1 = nn.ConvTranspose1d(32, out_channels, kernel_size=15, stride=1, padding=7)

    def forward(self, z, pool_info):
        pool_indices, output_sizes = pool_info
        x = z

        for unpool, dec, i in ((self.unpool5, self.dec5, 4), (self.unpool4, self.dec4, 3),
                               (self.unpool3, self.dec3, 2), (self.unpool2, self.dec2, 1),
                               (self.unpool1, self.dec1, 0)):
            x = unpool(x, pool_indices[i], output_size=output_sizes[i])
            x = dec(x)

        return x


class HybridMAE(nn.Module):
    """Autoencoder pretrained with masked reconstruction and a contrastive term.

    ``forward`` returns (reconstruction, latent map); ``embed`` returns the
    global-average-pooled embedding consumed by every downstream head.
    """

    def __init__(self, in_channels: int = 2, latent_channels: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.encoder = SpatialEncoder(in_channels, latent_channels)
        self.decoder = SpatialDecoder(in_channels, latent_channels)

    def forward(self, x: torch.Tensor):
        z, pool_info = self.encoder(x)
        return self.decoder(z, pool_info), z

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.encoder(x)
        return z.flatten(2).mean(-1)


class MultiEncoderEmbedder(nn.Module):
    """Stacks the embeddings of one or more frozen expert autoencoders.

    ``channel_slices`` routes each expert its own input channel, which is how the
    per-sensor (DE / FE) experts are wired.
    """

    def __init__(self, autoencoders: Sequence[HybridMAE], freeze: bool = True,
                 channel_slices: Sequence[Sequence[int]] | None = None):
        super().__init__()
        self.experts = nn.ModuleList(autoencoders)
        self.num_experts = len(autoencoders)
        self.latent_channels = int(autoencoders[0].latent_channels)
        self.embedding_dim = self.latent_channels * self.num_experts
        self.channel_slices = ([list(s) for s in channel_slices]
                               if channel_slices is not None else None)
        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()

    def _expert_input(self, x: torch.Tensor, k: int) -> torch.Tensor:
        if self.channel_slices is None:
            return x
        return x[:, self.channel_slices[k], ...]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([ae.embed(self._expert_input(x, k))
                            for k, ae in enumerate(self.experts)], dim=1)


def build_embedder(autoencoders: Sequence[HybridMAE], freeze: bool = True,
                   channel_slices: Sequence[Sequence[int]] | None = None
                   ) -> MultiEncoderEmbedder:
    if isinstance(autoencoders, HybridMAE):
        autoencoders = [autoencoders]
    return MultiEncoderEmbedder(list(autoencoders), freeze=freeze,
                                channel_slices=channel_slices)


class FusionHead(nn.Module):
    """MLP classifier over the concatenated expert embeddings."""

    def __init__(self, num_experts: int, latent_channels: int, num_classes: int,
                 hidden: int = 128, dropout: float = 0.3):
        super().__init__()
        self.num_experts = num_experts
        self.latent_channels = latent_channels

        fused_dim = num_experts * latent_channels
        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        return self.classifier(emb.flatten(1))


class CNN1D(nn.Module):
    """Supervised convolutional baseline."""

    def __init__(self, num_classes: int = 10, in_channels: int = 2, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=15, stride=1, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

        self.gap = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.gap(self.features(x)))


def create_svm_model():
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale",
                    decision_function_shape="ovr", random_state=42)),
    ])


def create_xgboost_model(num_classes: int = 10):
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
