import torch
import numpy as np
from load_model import load_trained_model
from models.koopman_arch import KoopmanModel


class KoopmanPredict:
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.device = device

        self.mu_x = torch.tensor(config["mu_x"], dtype=torch.float32, device=device)
        self.std_x = torch.tensor(config["std_x"], dtype=torch.float32, device=device)
        self.mu_u = torch.tensor(config["mu_u"], dtype=torch.float32, device=device)
        self.std_u = torch.tensor(config["std_u"], dtype=torch.float32, device=device)
        self.dt = config["dt"]

    def normalize_x(self, x_phys):
        x = torch.tensor(x_phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        return (x-self.mu_x)/self.std_x

    def normalize_u(self, u_phys):
        u = torch.tensor(u_phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        return (u-self.mu_u)/self.std_u

    def denormalize_x(self, x_norm):
        return (x_norm*self.std_x + self.mu_x).squeeze(0).cpu().numpy()

    def encode(self, x_phys):
        x_norm = self.normalize_x(x_phys)
        with torch.no_grad():
            z = self.model.encoder(x_norm)
        return z.squeeze(0).cpu().numpy()

    def predict_step(self, x_phys, u_phys):
        x_norm = self.normalize_x(x_phys)
        u_norm = self.normalize_u(u_phys)
        with torch.no_grad():
            z = self.model.encoder(x_norm)
            z_next = self.model.step(z, u_norm, self.dt)
            x_next_norm = self.denormalize_x(z_next)
        x_next_phys = self.denormalize_x(x_next_norm)
        return x_next_phys

    def ms_rollout(self, x0_phys, u_seq_phys):
        x_traj = [x0_phys]
        z_traj = []
        x = x0_phys.copy()

        for u in u_seq_phys:
            x_norm = self.normalize_x(x)
            u_norm = self.normalize_u(u)
            with torch.no_grad():
                z = self.model.encoder(x_norm)
                z_next = self.model.step(z, u_norm, self.dt)
                x_next_norm = self.model.decoder(z_next)

            x_next = self.denormalize_x(x_next_norm)
            x_traj.append(x_next)
            z_traj.append(z_next.squeeze(0).cpu().numpy())
            x = x_next

        return np.array(x_traj), np.array(z_traj)