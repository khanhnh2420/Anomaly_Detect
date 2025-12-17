import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        # Define hidden layer sizes
        h1_dim = max(128, input_dim // 2)
        h2_dim = max(64, h1_dim // 2)

        # Encoder (3 layers)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h1_dim, h2_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h2_dim, latent_dim)
        )
        # Decoder (3 layers)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h2_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h2_dim, h1_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(h1_dim, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z
