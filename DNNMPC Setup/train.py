import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from helper_funcs import compute_state_dependent_K
from train_utils import rollout_multistep, p_free_schedule

def eval_one_step(loader, model, config, device):
    model.eval()
    mse = nn.MSELoss()
    dt = config["dt"]
    latent_dim = config["latent_dim"]
    metrics = {
        "recon": 0.0,
        "lin": 0.0,
        "pred": 0.0,
        "ctrl": 0.0,
        "B": 0.0,
        "residual": 0.0,
        "energy": 0.0,
        "pred_ctrl": 0.0}

    n_samples = 0
    with torch.no_grad():
        for xb, xpb, ub in loader:
            xb = xb.to(device)
            xpb = xpb.to(device)
            ub = ub.to(device)
            Bsz = xb.shape[0]

            yb = model.encoder(xb)
            x_rec = model.decoder(yb)
            L_rec = mse(x_rec, xb)
            y1_true = model.encoder(xpb)
            K_dynamic = compute_state_dependent_K(yb, model.lambda_net, dt, device, latent_dim)

            y_linear = torch.bmm(K_dynamic, yb.unsqueeze(-1)).squeeze(-1)
            phi_u = model.ctrl_net(ub)
            y1_lin = y_linear+(phi_u@model.B.T)+model.bz
            y1_no_u = y_linear + model.bz

            L_lin = mse(y1_true, y1_lin)
            L_ctrl = mse(y1_true-y1_no_u, phi_u@model.B.T)
            L_B = torch.norm(model.B, p='fro')
            L_residual = mse(y1_true-y_linear, phi_u@model.B.T)
            L_ctrl_energy = torch.mean((phi_u@model.B.T)**2)

            x1_pred = model.decoder(y1_lin)
            L_pred = mse(x1_pred, xpb)
            L_pred_ctrl = mse(model.decoder(y1_no_u), xpb)

            metrics["recon"] += L_rec.item()*Bsz
            metrics["lin"] += L_lin.item()*Bsz
            metrics["pred"] += L_pred.item()*Bsz
            metrics["ctrl"] += L_ctrl.item()*Bsz
            metrics["B"] += L_B.item()*Bsz
            metrics["residual"] += L_residual.item()*Bsz
            metrics["energy"] += L_ctrl_energy.item()*Bsz
            metrics["pred_ctrl"] += L_pred_ctrl.item()*Bsz
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
        "ms": 0.0,
        "ctrl": 0.0,
        "B": 0.0,
        "residual": 0.0,
        "energy": 0.0,
        "pred_ctrl": 0.0}

    n_samples = 0
    with torch.no_grad():
        for x0b, u_seqb, x_seqb in loader:
            x0b = x0b.to(device)
            x_seqb = x_seqb.to(device)
            u_seqb = u_seqb.to(device)
            Bsz = x0b.shape[0]

            yb = model.encoder(x0b)
            x_rec = model.decoder(yb)
            L_rec = mse(x_rec, x0b)

            y1_true = model.encoder(x_seqb[:, 0, :])
            u0 = u_seqb[:, 0, :]
            K_dynamic = compute_state_dependent_K(yb, model.lambda_net, dt, device, latent_dim)
            y_linear = torch.bmm(K_dynamic, yb.unsqueeze(-1)).squeeze(-1)

            phi_u = model.ctrl_net(u0)
            y1_lin = y_linear+(phi_u@model.B.T) + model.bz
            y1_no_u = y_linear + model.bz

            L_lin = mse(y1_true, y1_lin)
            L_ctrl = mse(y1_lin-y1_no_u, phi_u@model.B.T)
            L_B = torch.norm(model.B, p='fro')

            L_residual = mse(y1_true-y_linear, phi_u@model.B.T)
            L_ctrl_energy = torch.mean((phi_u@model.B.T)**2)

            x1_pred = model.decoder(y1_lin)
            L_pred = mse(x1_pred, x_seqb[:, 0, :])
            L_pred_ctrl = mse(model.decoder(y1_no_u), x_seqb[:, 0, :])

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
            metrics["ctrl"] += L_ctrl.item()*Bsz
            metrics["B"] += L_B.item()*Bsz
            metrics["residual"] += L_residual.item()*Bsz
            metrics["energy"] += L_ctrl_energy.item()*Bsz
            metrics["pred_ctrl"] += L_pred_ctrl.item()*Bsz
            n_samples += Bsz

    for i in metrics:
        metrics[i] /= n_samples
    return metrics

# Stage 1: Autoencoder
def train_stage1(model, train_loader, val_loader, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config["lr_stage1"])
    mse = nn.MSELoss()
    n_samples = 0
    epoch_loss = 0.0

    for ep in range(1, config["N_stage1"] + 1):
        model.train()
        for xb, _, _ in train_loader:
            xb = xb.to(device)
            Bsz = xb.shape[0]

            z = model.encoder(xb)
            x_rec = model.decoder(z)

            loss = mse(x_rec, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            n_samples += Bsz
            epoch_loss += loss.item()*Bsz
        epoch_loss /= n_samples

        if ep % 20 == 0 or ep == 1:
            metrics = eval_one_step(val_loader, model, config, device)
            val_loss = metrics["recon"]
            print(f"[Stage1] Ep {ep} "
                  f"train_rec={epoch_loss:.4e} "
                  f"val_rec={val_loss:.4e}")


# Stage 2: Training to learn Koopman Operator
def train_stage2(model, train_loader_roll, val_loader_roll, config, device):
    optimizer = optim.Adam(model.parameters(), lr=config["lr_stage2"])
    mse = nn.MSELoss()

    best_val = float("inf")
    patience = config["patience"]
    no_improve = 0
    n_samples = 0
    epoch_loss = 0.0

    for ep in range(1, config["N_stage2"] + 1):
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

            K_dynamic = compute_state_dependent_K(y0, model.lambda_net, config["dt"], device, config["latent_dim"])

            y_linear = torch.bmm(K_dynamic, y0.unsqueeze(-1)).squeeze(-1)
            phi_u = model.ctrl_net(u0)
            y1_pred = y_linear + (phi_u @ model.B.T) + model.bz

            x1_pred = model.decoder(y1_pred)

            L_pred = mse(x1_pred, x_seqb[:, 0, :])
            L_lin = mse(y1_pred, y1_true)

            p_free = p_free_schedule(ep, config)
            L_ms = rollout_multistep(x0b, u_seqb, x_seqb, model, dt=config["dt"], gamma=config["gamma"], p_free=p_free)

            loss = (config["w_recon"]*L_rec +
                    config["w_pred"]*L_pred +
                    config["w_lin"]*L_lin +
                    config["w_ms"]*L_ms)

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
                config["w_ms"]*metrics["ms"])

            print(f"[Stage2] Ep {ep}",
                  f"train_loss={epoch_loss:.4e}",
                  f"val={val_loss:.4e}",
                  f"rec={metrics['recon']:.2e}",
                  f"lin={metrics['lin']:.2e} ",
                  f"pred={metrics['pred']:.2e}",
                  f"ms={metrics['ms']:.2e}")

            if metrics["ms"] < best_val:
                best_val = metrics["ms"]
                no_improve = 0
                torch.save({"model_state_dict": model.state_dict(), "config": config}, "best_model.pth")
                print("Saved best model.")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print("Early stopping.")
                    break
    return best_val

def train_model(model, train_loader, train_loader_roll, val_loader, val_loader_roll, config, device):
    print("Stage 1 Training...")
    train_stage1(model, train_loader, val_loader, config, device)

    print("Stage 2 Training...")
    best_val = train_stage2(model, train_loader_roll, val_loader_roll, config, device)

    return float(best_val)