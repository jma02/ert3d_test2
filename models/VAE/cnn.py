import torch
import torch.nn as nn


class CNNVAE(nn.Module):
    """Simple 3D CNN VAE for (B, 1, 64, 32, 64) -> (B, 16, 16, 16)."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 16,
        base_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.base_channels = base_channels

        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels * 2, kernel_size=3, stride=(2, 1, 2), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 4, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
            nn.Conv3d(base_channels * 4, base_channels * 4, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.SiLU(),
        )

        self.mu = nn.Conv3d(base_channels * 4, latent_channels, kernel_size=1)
        self.logvar = nn.Conv3d(base_channels * 4, latent_channels, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(
                latent_channels,
                base_channels * 4,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels * 4,
                base_channels * 4,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels * 4,
                base_channels * 2,
                kernel_size=3,
                stride=(2, 1, 1),
                padding=1,
                output_padding=(1, 0, 0),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels * 2,
                base_channels,
                kernel_size=3,
                stride=(2, 1, 2),
                padding=1,
                output_padding=(1, 0, 1),
            ),
            nn.SiLU(),
            nn.ConvTranspose3d(
                base_channels,
                base_channels // 2,
                kernel_size=3,
                stride=(2, 2, 2),
                padding=1,
                output_padding=(1, 1, 1),
            ),
            nn.SiLU(),
            nn.Conv3d(base_channels // 2, in_channels, kernel_size=3, padding=1),
        )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, (nn.Conv3d, nn.Linear, nn.ConvTranspose3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h).squeeze(2)
        logvar = self.logvar(h).squeeze(2)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = z.unsqueeze(2)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
