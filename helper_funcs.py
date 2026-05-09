import torch
import numpy as np
import torch.nn.functional as F

def compute_state_dependent_K(y, lambda_net, dt, device, latent_dim):
  # Get raw mu and omega parameters from lambda_net
  m_dim = latent_dim//2
  y_pair = y.view(y.shape[0], m_dim, 2)
  radius = torch.sum(y_pair**2, dim=2)

  mu_omega_output = lambda_net(radius) # (batch_size, 2 * latent_dim)

  # Split the output into mu and omega raw parameters
  mu_params_from_net = mu_omega_output[:, :m_dim] # Full latent_dim for mu
  omega_params_from_net = mu_omega_output[:, m_dim:] # Full latent_dim for omega

  mu_vals = -0.5 * F.softplus(mu_params_from_net)
  omega_max = 2*np.pi*100.0 # dt=0.01s (100Hz)
  omega_vals = omega_max*torch.tanh(omega_params_from_net)

  exp_mu_dt = torch.exp(mu_vals*dt)
  cos_omega_dt = torch.cos(omega_vals*dt)
  sin_omega_dt = torch.sin(omega_vals*dt)

  K_dynamic_batch = torch.zeros(y.shape[0], latent_dim, latent_dim, device=device)

  for i in range(latent_dim//2):
    K_dynamic_batch[:, 2*i, 2*i] = exp_mu_dt[:, i]*cos_omega_dt[:, i]
    K_dynamic_batch[:, 2*i, 2*i+1] = -exp_mu_dt[:, i]*sin_omega_dt[:, i]
    K_dynamic_batch[:, 2*i+1, 2*i] = exp_mu_dt[:, i]*sin_omega_dt[:, i]
    K_dynamic_batch[:, 2*i+1, 2*i+1] = exp_mu_dt[:, i]*cos_omega_dt[:, i]

  return K_dynamic_batch

def spectral_projection(K_tensor, s_max=0.97):
  with torch.no_grad():
    K_cpu = K_tensor.detach().cpu()
    try:
      U_K, S_K, V_K = torch.linalg.svd(K_cpu, full_matrices=False)
    except Exception:
      U_K, S_K, V_K = torch.linalg.svd(K_cpu)
    S_clamp = torch.clamp(S_K, max=s_max)
    K_proj = (U_K @ torch.diag_embed(S_clamp) @ V_K)
    K_tensor.data.copy_(K_proj.to(K_tensor.device))