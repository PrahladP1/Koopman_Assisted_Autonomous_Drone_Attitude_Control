import numpy as np
import torch

class KoopmanRuntime:
    def __init__(self, model, config, device):
        self.model = model
        self.device = device

        self.mu_x = config["mu_x"]
        self.std_x = config["std_x"]
        self.mu_u = config["mu_u"]
        self.std_u = config["std_u"]
        self.dt = config["dt"]

    def norm_state(self, x):

        return(x-self.mu_x.ravel())/(self.std_x.ravel())

    def denorm_state(self, x_n):

        return(x_n*self.std_x.ravel() + self.mu_x.ravel())

    def encode(self, x_phys):
        x_n = self.norm_state(x_phys)
        xt = torch.from_numpy(x_n.astype(np.float32)).to(self.device)
        with torch.no_grad():
            z = self.model.encoder(xt.unsqueeze(0)).squeeze(0)

        return z.astype(np.float32).cpu().numpy()

    def decode(self, z):
        zt = torch.from_numpy(z.astype(np.float32)).to(self.device)
        with torch.no_grad():
            x_n = self.model.decoder(zt.unsqueeze(0)).squeeze(0)

        return self.denorm_state(x_n.cpu().numpy())

    def latent_step(self, z, u):
        zt = torch.tensor(z, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
        ut = torch.tensor(u, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
        with torch.no_grad():
            z_next = self.model.step(zt, ut, self.dt)

        return z_next.squeeze(0).cpu().numpy()

    def step(self, x_phys, u_phys):
        z = self.encode(x_phys)
        u_n = ((u_phys - self.mu_u.ravel())/self.std_u.ravel())
        z_next = self.latent_step(z, u_n)
        x_next = self.decode(z_next)

        return x_next

    def linearize_dynamics(self, z, u):
        zt = torch.tensor(z, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
        ut = torch.tensor(u, dtype=torch.float32, device=self.device, requires_grad=True).unsqueeze(0)
        f = self.model.step(zt, ut, self.dt)
        z_dim = z.shape[0]
        u_dim = u.shape[0]

        A = torch.zeros(z_dim, z_dim, device=self.device)
        B = torch.zeros(z_dim, u_dim, device=self.device)

        for i in range(z_dim):
            grad_z = torch.autograd.grad(f[0, i], zt, retain_graph=True)[0]
            grad_u = torch.autograd.grad(f[0, i], ut, retain_graph=True)[0]
            A[i] = grad_z.squeeze(0)
            B[i] = grad_u.squeeze(0)

        return A.detach().cpu().numpy(), B.detach().cpu().numpy()