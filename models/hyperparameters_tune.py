import torch
import optuna
import numpy as np
from koopman_arch import KoopmanModel
from train import train_model
from build_datasets import load_state_group, load_control_group, build_mats_with_traj_ids
from train_loaders import build_dataloaders
from pathlib import Path
import json

BASE_DIR = Path.home() / "PycharmProjects" / "MyProjects"
DATA_DIR = BASE_DIR / "Data"
SINE_DIR = DATA_DIR / "State-Control History Sine"
CHIRP_DIR = DATA_DIR / "State-Control History Chirp"
PRBS_DIR = DATA_DIR / "State-Control History PRBS"
CONFIG_DIR = BASE_DIR/"Config"
def load_state_ctrl_hist():
    X_trajs = load_state_group(SINE_DIR, "state_history_x0", suffix="sin")
    X_trajs_c = load_state_group(CHIRP_DIR, "state_history_x0", suffix="chirp")
    X_trajs_p = load_state_group(PRBS_DIR, "state_history_x0", suffix="prbs")
    U_trajs = load_control_group(SINE_DIR/"control_inputs_sin.csv")
    U_trajs_c = load_control_group(CHIRP_DIR/"control_inputs_chirp.csv")
    U_trajs_p = load_control_group(PRBS_DIR/"control_inputs_prbs.csv")

    X_all = X_trajs + X_trajs_c + X_trajs_p
    U_all = U_trajs + U_trajs_c + U_trajs_p

    return X_all, U_all

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = trial.suggest_int("latent_dim", 8, 16, step=2)
    lr_stage1 = trial.suggest_float("lr_stage1", 1e-4, 5e-4, log=True)
    lr_stage2 = trial.suggest_float("lr_stage2", 1e-5, 2e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.96, 0.995)
    T_roll = trial.suggest_int("T_roll", 15, 35, step=5)
    w_recon = trial.suggest_float("w_recon", 1e-3, 5e-2, log=True)
    w_lin = trial.suggest_float("w_lin", 0.1, 10.0, log=True)
    w_pred = trial.suggest_float("w_pred", 0.1, 10.0, log=True)
    w_ms = trial.suggest_float("w_ms", 1.0, 20.0, log=True)
    w_kin = trial.suggest_float("w_kin", 1e-2, 1e-1, log=True)
    w_psi = trial.suggest_float("w_psi", 1e-3, 5e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])

    X_all, U_all = load_state_ctrl_hist()
    X1, X2, Ups, Omega, traj_ids = build_mats_with_traj_ids(X_all, U_all)
    Xk = X1.T.astype(np.float32)
    Xkp1 = X2.T.astype(np.float32)
    Uk = Ups.T.astype(np.float32)

    loaders = build_dataloaders(Xk, Xkp1, Uk, traj_ids, batch_size=batch_size, T_roll=T_roll)
    stats = loaders["stats"]
    n = Xk.shape[1]
    p = Uk.shape[1]

    config = {"dt": 0.01,
              "latent_dim": latent_dim,
              "gamma": gamma,
              "lr_stage1": lr_stage1,
              "lr_stage2": lr_stage2,
              "N_stage1": 60,
              "N_stage2": 360,
              "w_recon": w_recon,
              "w_pred": w_pred,
              "w_lin": w_lin,
              "w_ms": w_ms,
              "w_kin": w_kin,
              "w_psi": w_psi,
              "patience": 50,
              "std_x": stats["std_x"],
              "mu_x": stats["mu_x"],
              "std_u": stats["std_u"],
              "mu_u": stats["mu_u"],
              "scheduled_sampling": True,
              "scheduled_start_epoch": 10,
              "scheduled_end_epoch": 30}
    model = KoopmanModel(n, latent_dim, p, dt=config["dt"]).to(device)

    best_val = train_model(model, loaders["train_loader"], loaders["train_loader_roll"], loaders["val_loader"],
                           loaders["val_loader_roll"], config=config, device=device)
    return best_val

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", study_name="Best Params")
    study.optimize(objective, n_trials=50)
    print("\nBest trial: ", study.best_trial)
    print("\nBest Params: ", study.best_params)
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_DIR/"best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)