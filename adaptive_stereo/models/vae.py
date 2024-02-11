# Reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
  def forward(self, input):
    return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
  def __init__(self, channels, height, width):
    super(UnFlatten, self).__init__()
    self.channels = channels
    self.height = height
    self.width = width

  def forward(self, input):
    return input.view(input.shape[0], self.channels, self.height, self.width)


class VAE(nn.Module):
  def __init__(self, image_channels=3, z_dim=32, input_height=64, input_width=64):
    super(VAE, self).__init__()
    self.encoder = nn.Sequential(
      nn.Conv2d(image_channels, 32, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
      nn.ReLU(),
      Flatten()
    )

    h_dim = 256 * (input_height // 16) * (input_width // 16)

    self.fc1 = nn.Linear(h_dim, z_dim)
    self.fc2 = nn.Linear(h_dim, z_dim)
    self.fc3 = nn.Linear(z_dim, h_dim)

    self.decoder = nn.Sequential(
      UnFlatten(256, input_height // 16, input_width // 16),
      nn.ConvTranspose2d(256, 128, kernel_size=6, stride=2, padding=2),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=6, stride=2, padding=2),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2, padding=2),
      nn.ReLU(),
      nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2, padding=2),
      nn.Sigmoid(),
    )

  def reparameterize(self, mu, logvar):
    std = logvar.mul(0.5).exp_()
    # return torch.normal(mu, std)
    esp = torch.randn(*mu.size()).cuda()
    z = mu + std * esp
    return z

  def bottleneck(self, h):
    mu, logvar = self.fc1(h), self.fc2(h)
    z = self.reparameterize(mu, logvar)
    return z, mu, logvar

  def representation(self, x):
    return self.bottleneck(self.encoder(x))[0]

  def forward(self, x):
    h = self.encoder(x)
    z, mu, logvar = self.bottleneck(h)
    z = self.fc3(z)
    return self.decoder(z), mu, logvar


def vae_loss_function(x_true, x_pred, mu, logvar, beta_kl=0.001):
  """
  Loss for training a variational autoencoder. Combines reconstruction loss (L1) with KL divergence,
  which regularizes the latent manifold to be close to a standard multivariate Gaussian.

  beta_kl (float) : Coefficient to weight the importance of the KL divergence term.
  """
  L_re = F.l1_loss(x_pred, x_true)

  # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014.
  L_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

  return L_re + beta_kl*L_kl, L_re, beta_kl*L_kl
