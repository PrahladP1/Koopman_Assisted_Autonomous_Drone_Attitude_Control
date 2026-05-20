from torch.distributions import gamma
import torch.nn as nn
import torch
from torch.nn import MSELoss
import numpy as np

def rollout_multistep(x0, u_seq, x_seq_true, model, dt, gamma=gamma, p_free=1.0):
  Bsz, T, _ = u_seq.shape
  y = model.encoder(x0)
  loss = 0.0
  mse = nn.MSELoss()

  for t in range(T):
    ut = u_seq[:, t, :]

    # Scheduled sampling
    if t == 0 or torch.rand(1).item() < p_free:
      y = model.step(y, ut, dt)
    else:
      # teacher forcing
      y = model.encoder(x_seq_true[:, t - 1, :])
    xhat = model.decoder(y)

    x_true_t = x_seq_true[:, t, :]
    weight = gamma**t
    loss += weight*mse(xhat, x_true_t)

    y_reenc = model.encoder(xhat)
    latent_consistency = mse(y_reenc, y.detach())
    loss += 0.1*latent_consistency

  return loss/T


def p_free_schedule(ep, config):
  if not config.get("scheduled_sampling", True):
    return 1.0
  start = config["scheduled_start_epoch"]
  end = config["scheduled_end_epoch"]
  if ep <= start:
    return 0.0
  elif ep >= end:
    return 1.0
  else:
    return (ep-start)/(end-start)

def rollout_x0(model, x0_phys, U_phys, config, device):
  model.eval()

  x0_phys = np.asarray(x0_phys).reshape(-1)
  U_phys = np.asarray(U_phys)
  # Load normalization stats
  mu_x = torch.from_numpy(config["mu_x"]).float().to(device)
  std_x = torch.from_numpy(config["std_x"]).float().to(device)
  mu_u = torch.from_numpy(config["mu_u"]).float().to(device)
  std_u = torch.from_numpy(config["std_u"]).float().to(device)
  dt = config["dt"]

  # Normalize initial state
  x = (torch.from_numpy(x0_phys).float().to(device)-mu_x)/std_x
  x = x.unsqueeze(0)

  with torch.no_grad():
    y = model.encoder(x)
    traj = []
    for k in range(U_phys.shape[0]):
      u = (torch.from_numpy(U_phys[k]).float().to(device)-mu_u)/std_u
      u = u.unsqueeze(0)
      y = model.step(y, u, dt)
      xhat = model.decoder(y)
      xhat_phys = xhat * std_x + mu_x
      traj.append(xhat_phys.squeeze(0).cpu().numpy())
  return np.vstack(traj)