import optuna
import torch
import numpy as np
from build_datasets import load_state_group, load_control_group, get_initial_condition_seq, build_mats_with_traj_ids
from train_loaders import build_dataloaders
from koopman_arch import KoopmanModel
from train import train_model
from pathlib import Path
import json
import random

BASE_DIR = Path.home()/"PycharmProjects"/"MyProjects"
DATA_DIR = BASE_DIR/"Data"
SINE_DIR = DATA_DIR/"State-Control History Sine"
CHIRP_DIR = DATA_DIR/"State-Control History Chirp"
PRBS_DIR = DATA_DIR/"State-Control History PRBS"
CONFIG_DIR = BASE_DIR/"Config"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_trajs = load_state_group(SINE_DIR, "state_history_x0", suffix="sin")
    X_trajs_c = load_state_group(CHIRP_DIR, "state_history_x0", suffix="chirp")
    X_trajs_p = load_state_group(PRBS_DIR, "state_history_x0", suffix="prbs")

    U_trajs = load_control_group(SINE_DIR/"control_inputs_sin.csv")
    U_trajs_c = load_control_group(CHIRP_DIR/"control_inputs_chirp.csv")
    U_trajs_p = load_control_group(PRBS_DIR/"control_inputs_prbs.csv")

    X_all = X_trajs + X_trajs_c + X_trajs_p
    U_all = U_trajs + U_trajs_c + U_trajs_p

    X1, X2, Ups, Omega, traj_ids = build_mats_with_traj_ids(X_all, U_all)
    Xk = X1.T.astype(np.float32)
    Xkp1 = X2.T.astype(np.float32)
    Uk = Ups.T.astype(np.float32)

    loaders = build_dataloaders(Xk, Xkp1, Uk, traj_ids)

    train_loader = loaders["train_loader"]
    val_loader = loaders["val_loader"]
    train_loader_roll = loaders["train_loader_roll"]
    val_loader_roll = loaders["val_loader_roll"]

    stats = loaders["stats"]

    n = Xk.shape[1]
    p = Uk.shape[1]
    latent_dim = 8
    model = KoopmanModel(n, latent_dim, p).to(device)

    config = {
        "latent_dim": latent_dim,
        "dt": 0.01,
        "gamma": 0.99,
        "lr_stage1": 1e-3,
        "lr_stage2": 1e-4,
        "N_stage1": 200,
        "N_stage2": 500,
        "patience": 20,
        "w_recon": 0.01,
        "w_pred": 0.5,
        "w_lin": 0.01,
        "w_ms": 1e-2,
        "w_kin": 1e-2,
        "w_psi": 1e-3,
        "w_B": 0.5,
        "w_res": 10.0,
        "w_ctrl_energy": 2.0,
        "w_pred_ctrl": 2.0,
        "std_x": stats["std_x"],
        "mu_x": stats["mu_x"],
        "std_u": stats["std_u"],
        "mu_u": stats["mu_u"],
        "scheduled_sampling": True,
        "scheduled_start_epoch": 200,
        "scheduled_end_epoch": 400}

    best_config_path = CONFIG_DIR/"best_params.json"
    if best_config_path.exists():
        print("Loading best params...")
        with open(best_config_path, "r") as f:
            tuned_params = json.load(f)
        config.update(tuned_params)
    else:
        print("No best params file found.")
    train_model(model, train_loader, train_loader_roll, val_loader, val_loader_roll, config, device)


if __name__ == "__main__":
    main()