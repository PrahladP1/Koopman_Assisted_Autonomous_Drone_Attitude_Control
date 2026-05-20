import torch
import torch.nn as nn
import numpy as np

class Encoder(nn.Module):
  def __init__(self, in_dim, out_dim, dropout=0.1):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 128), nn.ReLU(), nn.Dropout(p=dropout),
        nn.Linear(128, 64), nn.ReLU(), nn.Dropout(p=dropout),
        nn.Linear(64, out_dim), nn.ReLU()
    )
  def forward(self, x):
    return self.net(x)

class Decoder(nn.Module):
  def __init__(self, in_dim, out_dim):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(in_dim, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, out_dim)
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

class BNet(nn.Module):
    def __init__(self, z_dim, u_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, u_dim*z_dim*z_dim))
        self.z_dim = z_dim
        self.u_dim = u_dim

    def forward(self, z):
        B_flat = self.net(z)
        return B_flat.view(z.shape[0], self.u_dim, self.z_dim, self.z_dim)

#class ControlNet(nn.Module):
#  def __init__(self, p, d_u):
#    super().__init__()
#    self.net = nn.Sequential(
#        nn.Linear(p, 32), nn.ReLU(),
#        nn.Linear(32, d_u)
#    )
#  def forward(self, u):
#    return self.net(u)

class KoopmanModel(nn.Module):
    def __init__(self, x_dim, z_dim, u_dim, dt):
        super().__init__()

        self.encoder = Encoder(x_dim, z_dim)
        self.decoder = Decoder(z_dim, x_dim)
        #self.ctrl_net = ControlNet(u_dim, u_dim)
        self.lambda_net = LambdaNet(z_dim//2, z_dim)
        self.B0 = nn.Parameter(0.01*torch.randn(z_dim, u_dim))
        self.B_net = BNet(z_dim, u_dim)
        self.bz = nn.Parameter(torch.zeros(z_dim))
        self.z_dim = z_dim
        self.dt = dt

    def forward(self, x, u):
        z = self.encoder(x)
        z_next = self.step(z, u, self.dt)
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

        B_ops = self.B_net(y)
        u_expand = u.unsqueeze(-1).unsqueeze(-1)
        B_weight = u_expand*B_ops
        B_tot = torch.sum(B_weight, dim=1)
        B_tot = 0.001*torch.tanh(B_tot)

        bilinear_term = torch.bmm(B_tot, y.unsqueeze(-1)).squeeze(-1)

        return y_dyn+bilinear_term+self.bz