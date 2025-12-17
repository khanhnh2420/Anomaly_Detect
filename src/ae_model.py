import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(64, input_dim//2)),
            nn.BatchNorm1d(max(64, input_dim//2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(max(64, input_dim//2), latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(64, input_dim//2)),
            nn.BatchNorm1d(max(64, input_dim//2)),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(max(64, input_dim//2), input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
