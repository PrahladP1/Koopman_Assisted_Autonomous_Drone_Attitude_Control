import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, out_dim)
    )
  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 128), nn.ReLU(),
        nn.Linear(128, 256), nn.ReLU(),
        nn.Linear(256, out_dim)
    )
  def forward(self, x):
    return self.net(x)

class LambdaNet(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, out_dim)
    )
  def forward(self, y):
    return self.net(y)

class ControlNet(nn.Module):
  def __init__(self, p, d_u):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(p, 32), nn.ReLU(),
        nn.Linear(32, d_u)
    )
  def forward(self, u):
    return self.net(u)

class KoopmanModel(nn.Module):
    def __init__(self, x_dim, z_dim, u_dim):
        super().__init__()

        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(z_dim, x_dim)
        self.ctrl_net = ControlNet(u_dim, u_dim)
        self.lambda_net = LambdaNet(z_dim//2, z_dim)
        self.B = nn.Parameter(torch.randn(z_dim, u_dim))
        self.bz = nn.Parameter(torch.zeros(z_dim))
        self.z_dim = z_dim

    def forward(self, x, u):
        z = self.encoder(x)
        z_next = self.step_latent(z, u, self.dt)
        x_rec = self.decoder(z)

        return z, z_next, x_rec

    def step(self, y, u, dt):
        latent_dim = y.shape[1]
        m_dim = latent_dim//2
        y_pair = y.view(y.shape[0], m_dim, 2)
        radius = torch.sum(y_pair**2, dim=2)
        mu_omega = self.lambda_net(radius)
        mu_raw = mu_omega[:, :m_dim]
        omega_raw = mu_omega[:, m_dim:]

        mu = -0.5*torch.nn.functional.softplus(mu_raw)
        omega = (2*np.pi*100) * torch.tanh(omega_raw)

        a = torch.exp(mu*dt)
        c = torch.cos(omega*dt)
        s = torch.sin(omega*dt)

        y0, y1 = y_pair[:, :, 0], y_pair[:, :, 1]

        y0n = (a*c*y0) - (a*s*y1)
        y1n = (a*s*y0) + (a*c*y1)

        y_dyn = torch.stack([y0n, y1n], dim=-1).reshape(y.shape[0], latent_dim)

        phi_u = self.ctrl_net(u)

        return y_dyn+(phi_u@self.B.T)+self.bz