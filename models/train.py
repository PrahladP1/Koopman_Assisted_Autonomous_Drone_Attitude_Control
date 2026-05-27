from distutils.command.config import config

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from train_utils import rollout_multistep, p_free_schedule

def eval_one_step(loader, model, config, device):
    model.eval()
    mse = nn.MSELoss()
    dt = config["dt"]
    latent_dim = config["latent_dim"]
    metrics = {"recon": 0.0, "lin": 0.0, "pred": 0.0}
    n_samples = 0

    with torch.no_grad():
        for xb, xpb, ub in loader:
            xb = xb.to(device)
            xpb = xpb.to(device)
            ub = ub.to(device)
            Bsz = xb.shape[0]

            yb = model.encoder(xb)
            # Reconstruction
            x_rec = model.decoder(yb)
            L_rec = mse(x_rec, xb)
            y1_true = model.encoder(xpb) # True future latent state
            y1_pred = model.step(yb, ub, config["dt"]) # Predicted future latent state
            L_lin = mse(y1_pred, y1_true) # Linearity Loss
            x1_pred = model.decoder(y1_pred)
            L_pred = mse(x1_pred, xpb)


            metrics["recon"] += L_rec.item()*Bsz
            metrics["lin"] += L_lin.item()*Bsz
            metrics["pred"] += L_pred.item()*Bsz
            n_samples += Bsz
    for i in metrics:
        metrics[i] /= n_samples
    return metrics

def eval_multistep_rollout(loader, model, config, device, ep):
    model.eval()
    mse = nn.MSELoss()
    dt = config["dt"]
    latent_dim = config["latent_dim"]
    gamma = config["gamma"]
    std_x = torch.from_numpy(config["std_x"]).float().to(device)
    mu_x = torch.from_numpy(config["mu_x"]).float().to(device)
    metrics = {
        "recon": 0.0,
        "lin": 0.0,
        "pred": 0.0,
        "kin": 0.0,
        "psi": 0.0,
        "ms": 0.0}

    n_samples = 0
    with torch.no_grad():
        for x0b, u_seqb, x_seqb in loader:
            x0b = x0b.to(device)
            x_seqb = x_seqb.to(device)
            u_seqb = u_seqb.to(device)
            Bsz = x0b.shape[0]

            y0 = model.encoder(x0b)
            x_rec = model.decoder(y0) # Reconstruction
            L_rec = mse(x_rec, x0b)

            u0 = u_seqb[:, 0, :]
            y1_true = model.encoder(x_seqb[:, 0, :])
            y1_pred = model.step(y0, u0, config["dt"])
            L_lin = mse(y1_pred, y1_true)
            x1_pred = model.decoder(y1_pred)
            L_pred = mse(x1_pred, x_seqb[:, 0, :])

            # Kinematic Loss
            p_pred = x1_pred[:, 0:2]
            p_k = x0b[:, 0:2]
            v_k = x0b[:, 3:5]
            L_kin = mse(p_pred, dt*v_k*(std_x[:, 3:5]/std_x[0, 0:2]))

            # Yaw Kinematic Loss
            sinpsi_k = x0b[:, 10]
            cospsi_k = x0b[:, 11]
            r_k = x0b[:, 14]
            psi_k = torch.atan2(
                sinpsi_k*std_x[0, 10]+mu_x[0, 10],
                cospsi_k*std_x[0, 11]+mu_x[0, 11])
            r_k = r_k*std_x[0, 14]+mu_x[0, 14]
            psi_next = psi_k+dt*r_k

            spsi_tgt = (torch.sin(psi_next)-mu_x[0, 10])/std_x[0, 10]
            cpsi_tgt = (torch.cos(psi_next)-mu_x[0, 11])/std_x[0, 11]

            L_psi = mse(x1_pred[:, 10], spsi_tgt) + mse(x1_pred[:, 11], cpsi_tgt)

            p_free = p_free_schedule(ep, config)
            L_ms = rollout_multistep(x0b, u_seqb, x_seqb, model, dt=dt, gamma=gamma, p_free=p_free)

            metrics["recon"] += L_rec.item()*Bsz
            metrics["lin"] += L_lin.item()*Bsz
            metrics["pred"] += L_pred.item()*Bsz
            metrics["kin"] += L_kin.item()*Bsz
            metrics["psi"] += L_psi.item()*Bsz
            metrics["ms"] += L_ms.item()*Bsz
            n_samples += Bsz

    for i in metrics:
        metrics[i] /= n_samples
    return metrics

# Stage 1: Autoencoder
def train_stage1(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config["lr_stage1"])
    mse = nn.MSELoss()

    for ep in range(1, config["N_stage1"] + 1):
        n_samples = 0
        epoch_loss = 0.0
        model.train()
        for xb, xpb, ub in train_loader:
            xb, xpb, ub = xb.to(device), xpb.to(device), ub.to(device)
            Bsz = xb.shape[0]

            z = model.encoder(xb)
            x_rec = model.decoder(z)
            L_rec = mse(x_rec, xb)

            z_next = model.step(z, ub, config["dt"])
            x_next = model.decoder(z_next)
            L_pred = mse(x_next, xpb)

            loss = L_rec + 0.3*L_pred

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_samples += Bsz
            epoch_loss += loss.item()*Bsz
        epoch_loss /= n_samples

        if ep % 20 == 0 or ep == 1:
            metrics = eval_one_step(val_loader, model, config, device)
            val_loss = metrics["recon"]
            print(f"[Stage1] Ep {ep}",
                  f"train_rec={epoch_loss:.4e}",
                  f"val_rec={val_loss:.4e}",
                  f"val_pred={metrics['pred']:.4e}")


# Stage 2: Training to learn Koopman Operator
def train_stage2(model, train_loader_roll, val_loader_roll, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config["lr_stage2"])
    mse = nn.MSELoss()

    best_val = np.inf
    best_ms = np.inf
    patience = config["patience"]
    no_improve = 0
    n_samples = 0
    epoch_loss = 0.0

    assert config["std_x"].shape[1] == 15, \
        f"std_x has wrong dim {config['std_x'].shape}, expected (1,15)"

    for ep in range(1, config["N_stage2"] + 1):
        std_x = torch.from_numpy(config["std_x"]).float().to(device)
        mu_x = torch.from_numpy(config["mu_x"]).float().to(device)
        model.train()
        for x0b, u_seqb, x_seqb in train_loader_roll:
            x0b = x0b.to(device)
            u_seqb = u_seqb.to(device)
            x_seqb = x_seqb.to(device)
            Bsz = x0b.shape[0]

            y0 = model.encoder(x0b)
            x_rec = model.decoder(y0)

            L_rec = mse(x_rec, x0b)

            u0 = u_seqb[:, 0, :]
            y1_true = model.encoder(x_seqb[:, 0, :])
            y1_pred = model.step(y0, u0, config["dt"])
            x1_pred = model.decoder(y1_pred)
            L_lin = mse(y1_pred, y1_true)
            L_pred = mse(x1_pred, x_seqb[:, 0, :])

            p_free = p_free_schedule(ep, config)
            L_ms = rollout_multistep(x0b, u_seqb, x_seqb, model, dt=config["dt"], gamma=config["gamma"], p_free=p_free)

            # Kinematic Loss
            p_pred = x1_pred[:, 0:2]
            p_k = x0b[:, 0:2]
            v_k = x0b[:, 3:5]
            L_kin = mse(p_pred, config["dt"]*v_k*(std_x[:, 3:5]/std_x[0, 0:2]))

            # Yaw Kinematic Loss
            sinpsi_k = x0b[:, 10]
            cospsi_k = x0b[:, 11]
            r_k = x0b[:, 14]
            psi_k = torch.atan2(
                sinpsi_k*std_x[0, 10]+mu_x[0, 10],
                cospsi_k*std_x[0, 11]+mu_x[0, 11])
            r_k = r_k*std_x[0, 14]+mu_x[0, 14]
            psi_next = psi_k+config["dt"]*r_k

            spsi_tgt = (torch.sin(psi_next)-mu_x[0, 10])/std_x[0, 10]
            cpsi_tgt = (torch.cos(psi_next)-mu_x[0, 11])/std_x[0, 11]

            L_psi = mse(x1_pred[:, 10], spsi_tgt) + mse(x1_pred[:, 11], cpsi_tgt)


            loss = (
                    config["w_recon"]*L_rec +
                    config["w_pred"]*L_pred +
                    config["w_lin"]*L_lin +
                    config["w_ms"]*L_ms +
                    config["w_kin"]*L_kin +
                    config["w_psi"]*L_psi)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            n_samples += Bsz
            epoch_loss += loss.item()*Bsz
        epoch_loss /= n_samples

        if ep % 20 == 0 or ep == 1:
            metrics = eval_multistep_rollout(val_loader_roll, model, config, device, ep)

            val_loss = (
                config["w_recon"]*metrics["recon"] +
                config["w_pred"]*metrics["pred"] +
                config["w_lin"]*metrics["lin"] +
                config["w_ms"]*metrics["ms"] +
                config["w_kin"]*metrics["kin"] +
                config["w_psi"]*metrics["psi"])

            print(f"[Stage2] Ep {ep}",
                  f"train_loss={epoch_loss:.4e}",
                  f"val={val_loss:.4e}",
                  f"rec={metrics['recon']:.2e}",
                  f"lin={metrics['lin']:.2e} ",
                  f"pred={metrics['pred']:.2e}",
                  f"ms={metrics['ms']:.2e}",
                  f"kin={metrics['kin']:.2e}",
                  f"psi={metrics['psi']:.2e}")

            val_ms = metrics["ms"]
            if val_ms < best_ms:
                best_ms = val_ms
                best_val = val_loss
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict(), "config": config}, "best_model.pth")
                print("Saved best model.")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break
    return best_ms

def train_model(model, train_loader, train_loader_roll, val_loader, val_loader_roll, config, device):
    print("Stage 1 Training...")
    train_stage1(model, train_loader, val_loader, config, device)

    print("Stage 2 Training...")
    best_val = train_stage2(model, train_loader_roll, val_loader_roll, config, device)

    return float(best_val)