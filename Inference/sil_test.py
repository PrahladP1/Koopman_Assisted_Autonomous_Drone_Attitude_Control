import numpy as np
import matplotlib.pyplot as plt
import torch
from Inference.load_model import load_trained_model
from Inference.predict import KoopmanPredict
from models.build_datasets import load_state_group, load_control_group, encode_euler_angles
from pathlib import Path

def main():
    BASE_DIR = Path.home()/"PycharmProjects"/"MyProjects"
    DATA_DIR = BASE_DIR/"Data"
    SINE_DIR = DATA_DIR/"State-Control History Sine"
    CHIRP_DIR = DATA_DIR/"State-Control History Chirp"
    PRBS_DIR = DATA_DIR/"State-Control History PRBS"
    CONFIG_DIR = BASE_DIR/"Config"
    MODEL_PATH = BASE_DIR/"models"/"best_model.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_trained_model(MODEL_PATH, device)
    predictor = KoopmanPredict(model, config, device)

    X_trajs = load_state_group(SINE_DIR, "state_history_x0", suffix="sin")
    X_trajs_c = load_state_group(CHIRP_DIR, "state_history_x0", suffix="chirp")
    X_trajs_p = load_state_group(PRBS_DIR, "state_history_x0", suffix="prbs")

    U_trajs = load_control_group(SINE_DIR / "control_inputs_sin.csv")
    U_trajs_c = load_control_group(CHIRP_DIR / "control_inputs_chirp.csv")
    U_trajs_p = load_control_group(PRBS_DIR / "control_inputs_prbs.csv")

    X_all = X_trajs + X_trajs_c + X_trajs_p
    U_all = U_trajs + U_trajs_c + U_trajs_p

    traj_idx = 0
    X_true = X_all[traj_idx]
    U_true = U_all[traj_idx]

    X_enc = encode_euler_angles(X_true)
    x0 = X_enc[0]

    x_pred, z_pred = predictor.ms_rollout(x0, U_true)
    X_t = X_enc[:x_pred.shape[0]]

    N = min(len(x_pred)-1, len(X_t))
    x_pred_eval = x_pred[1:N+1]
    x_true_eval = X_t[:N]

    e_rmse = np.sqrt(np.mean((x_pred_eval-x_true_eval)**2))
    print(f"\nMS Rollout RMSE: {e_rmse:.6f}")

    tspan = np.arange(N)
    N_plot = 200
    plt.figure()
    plt.plot(tspan[:N_plot], x_true_eval[:, 0][:N_plot], label="True x")
    plt.plot(tspan[:N_plot], x_pred_eval[:, 0][:N_plot], label="Predicted x")

    plt.plot(tspan[:N_plot], x_true_eval[:, 1][:N_plot], label="True y")
    plt.plot(tspan[:N_plot], x_pred_eval[:, 1][:N_plot], label="Predicted y")

    plt.plot(tspan[:N_plot], x_true_eval[:, 2][:N_plot], label="True z")
    plt.plot(tspan[:N_plot], x_pred_eval[:, 2][:N_plot], label="Predicted z")

    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.legend()
    plt.title("Koopman Predictions")
    plt.show()

if __name__ == "__main__":
    main()