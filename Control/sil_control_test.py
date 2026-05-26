import numpy as np
import matplotlib.pyplot as plt
from koopman_mpc import KoopmanMPC
from Inference.koopman_runtime import KoopmanRuntime

def exec_sil(model, config, x0_phys, x_ref_phys, device, sim_steps=500, horizon=10):
    runtime = KoopmanRuntime(model, config, device)
    latent_dim = config["latent_dim"]
    ctrl_dim = model.B.shape[1]

    B = model.B.detach().cpu().numpy()
    bz = model.bz.detach().cpu().numpy()
    Qz = 10*np.eye(latent_dim)
    Ru = 0.01*np.eye(B.shape[1])
    umin = -1.0*np.ones((B.shape[1],))
    umax = 1.0*np.ones((B.shape[1],))

    mpc = KoopmanMPC(latent_dim=latent_dim, ctrl_dim=B.shape[1], horizon=10, Qz=Qz, Ru=Ru, umin=umin, umax=umax)
    x_phys = x0_phys.copy()
    u_prev = np.zeros((ctrl_dim,))
    x_hist = []
    u_hist = []
    z_hist = []

    print("Starting SIL Testing...")

    for i in range(sim_steps):
        z0 = runtime.encode(x_phys)
        A, B = runtime.linearize_dynamics(z0, u_prev)
        z_nom = runtime.latent_step(z0, u_prev)
        b = z_nom - (A@z0 + B@u_prev)

        z_ref_seq = []
        for j in range(horizon):
            idx = min(i+j, sim_steps-1)
            z_ref = runtime.encode(x_ref_phys[idx])
            z_ref_seq.append(z_ref)
        z_ref_seq = np.array(z_ref_seq)

        mpc.setup_form(A, B, b)
        u_cmd = mpc.solve(z0, z_ref_seq)
        x_next = runtime.step(x_phys, u_cmd)
        x_hist.append(x_phys.copy())
        u_hist.append(u_cmd.copy())
        z_hist.append(z0.copy())

        x_phys = x_next.copy()
        u_prev = u_cmd.copy()

        if i % 25 == 0:
            pos_err = np.linalg.norm(x_phys[0:3] - x_ref_phys[i][0:3])
            print(f"[{i:04d}] " f"|Pos Err| = {pos_err:.4f}")

    print("SIL Testing Complete.")
    history = {"x": np.array(x_hist), "u": np.array(u_hist), "z": np.array(z_hist)}
    return history

def plot_history(history, x_ref_phys):
    x_pred = history["x"]
    fig = plt.figure()
    plt.plot(x_ref_phys[:, 0], x_ref_phys[:, 1], label="Reference")
    plt.plot(x_pred[:, 0], x_pred[:, 1], label="Prediction")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory Tracking")
    plt.legend()
    plt.grid(True)
    plt.show()