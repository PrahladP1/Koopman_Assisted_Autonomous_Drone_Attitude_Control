import numpy as np
import pandas as pd
#from main import DATA_DIR


def load_state_group(data_dir, prefix, suffix="", n_traj=12):
    trajs = []
    for i in range(n_traj):
        file_path = data_dir/f"{prefix}_{i}{suffix}.csv"
        #print("Trying:", file_path, "| Exists:", file_path.exists())
        trajs.append(np.genfromtxt(file_path, delimiter=','))
    return trajs

def load_control_group(file_path):
    df = pd.read_csv(file_path)
    df = df.sort_values(["traj_id", "t"]).reset_index(drop=True)
    u_cols = ["u_thrust", "u_roll", "u_pitch", "u_yaw"]
    U_dict = {
        traj_id: g[u_cols].to_numpy()
        for traj_id, g in df.groupby("traj_id", sort=False)}

    return [U_dict[k] for k in sorted(U_dict.keys())]

def encode_euler_angles(Xtraj):
    """
    Xtraj: (T,12) with
      [px,py,pz,vx,vy,vz,phi,theta,psi,p,q,r]

    Returns (T,15):
      [px,py,pz,vx,vy,vz,
       sin(phi),cos(phi),
       sin(theta),cos(theta),
       sin(psi),cos(psi),
       p,q,r]
    """

    Xtraj = np.asarray(Xtraj, dtype=float)

    if Xtraj.shape[1] != 12:
        raise ValueError(f"Expected 12 states, got {Xtraj.shape[1]}.")

    phi   = Xtraj[:, 6]
    theta = Xtraj[:, 7]
    psi   = Xtraj[:, 8]

    X_enc = np.column_stack([
        Xtraj[:, 0:6],            # position + velocity
        np.sin(phi),
        np.cos(phi),
        np.sin(theta),
        np.cos(theta),
        np.sin(psi),
        np.cos(psi),
        Xtraj[:, 9:12]            # angular rates
    ])

    return X_enc

def build_mats_with_traj_ids(X_trajs, U_trajs):
    """
    Builds stacked DMDc matrices.

    Inputs
    -------
    X_trajs : list
        list of state trajectories (T_i, 12)

    U_trajs : list
        list of control trajectories (T_i, 4)

    Returns
    -------
    X1 : (n, N)
    X2 : (n, N)
    Ups : (p, N)
    Omega : (n+p, N)
    traj_ids : (N,)
    """

    X_col = []
    Xp_col = []
    U_col = []
    traj_ids = []

    for tid, (Xtraj, Utraj) in enumerate(zip(X_trajs, U_trajs)):
        Xtraj_enc = encode_euler_angles(Xtraj)

        Utraj = np.asarray(Utraj, dtype=float)

        Tx = Xtraj_enc.shape[0]
        Tu = Utraj.shape[0]

        T = min(Tx, Tu)

        if T < 2:
            raise ValueError(f"Trajectory {tid} too short.")

        X = Xtraj_enc[:T].T     # (n, T)
        U = Utraj[:T].T         # (p, T)

        Xk  = X[:, :-1]
        Xkp = X[:, 1:]
        Uk  = U[:, :-1]

        X_col.append(Xk)
        Xp_col.append(Xkp)
        U_col.append(Uk)

        traj_ids.append(np.full(Xk.shape[1], tid, dtype=int))

    X1 = np.hstack(X_col)
    X2 = np.hstack(Xp_col)
    Ups = np.hstack(U_col)
    Omega = np.vstack([X1, Ups])

    traj_ids = np.concatenate(traj_ids)

    return X1, X2, Ups, Omega, traj_ids

def get_initial_condition_seq(deg=np.pi/180):
    # Sinusoid Set
    x0_0 = np.array([[0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 * deg, 0.0 * deg, 0.0, 0.0, 0.0, 0.0]]).T
    x0_1 = np.array([[1.5, -0.5, 3.0, 1.0, 0.2, 0.0, 5.0 * deg, -3.0 * deg, 20.0 * deg, 0.2, -0.1, 0.1]]).T
    x0_2 = np.array([[-2.0, 1.0, 1.5, -1.5, 0.5, 0.3, -10.0 * deg, 8.0 * deg, -45.0 * deg, -0.3, 0.2, -0.2]]).T
    x0_3 = np.array([[3.0, 2.0, 5.0, 2.5, -0.8, -0.5, 12.0 * deg, -10.0 * deg, 90.0 * deg, 0.5, -0.4, 0.3]]).T
    x0_4 = np.array([[-4.0, -3.0, 4.0, -2.0, -1.0, 0.2, -15.0 * deg, 15.0 * deg, 170.0 * deg, -0.6, 0.6, -0.1]]).T
    x0_5 = np.array([[5.0, -2.0, 6.0, 0.5, 2.0, 0.0, 8.0 * deg, 5.0 * deg, -120.0 * deg, 0.1, 0.3, 0.8]]).T
    x0_6 = np.array([[-1.0, 4.0, 2.5, 0.0, -2.5, -1.0, -6.0 * deg, 12.0 * deg, 60.0 * deg, -0.2, 0.4, -1.0]]).T
    x0_7 = np.array([[2.0, 6.0, 8.0, 3.0, 1.0, 0.5, 20.0 * deg, -20.0 * deg, -10.0 * deg, 1.0, -1.0, 0.0]]).T
    x0_8 = np.array([[-6.0, 0.5, 1.0, -3.5, 0.0, 1.5, -25.0 * deg, 0.0 * deg, 30.0 * deg, -1.0, 0.0, 1.5]]).T
    x0_9 = np.array([[0.5, -7.0, 7.0, 0.0, 3.5, -2.0, 0.0 * deg, 25.0 * deg, -170.0 * deg, 0.0, 1.2, -2.0]]).T
    x0_10 = np.array([[7.0, 7.0, 9.5, 4.0, -4.0, 0.0, 18.0 * deg, -12.0 * deg, 140.0 * deg, 1.5, -0.8, 2.5]]).T
    x0_11 = np.array([[-7.5, -5.0, 0.8, -1.0, 1.0, 0.8, -12.0 * deg, 10.0 * deg, -80.0 * deg, -0.7, 0.5, -2.5]]).T
    x0_s = (np.vstack((x0_0, x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9, x0_10, x0_11)))

    # Chirp Set
    x0_c0 = np.array([[0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 2.0 * deg, -2.0 * deg, 5.0 * deg, 0.05, -0.05, 0.02]]).T
    x0_c1 = np.array([[0.5, -0.5, 3.0, 0.3, -0.2, 0.0, 10.0 * deg, -3.0 * deg, 15.0 * deg, 0.3, -0.1, 0.2]]).T
    x0_c2 = np.array([[-1.0, 0.5, 2.0, -0.4, 0.6, -0.2, -4.0 * deg, 14.0 * deg, -25.0 * deg, -0.2, 0.6, -0.1]]).T
    x0_c3 = np.array([[1.0, 2.0, 4.0, 0.5, -0.5, 0.2, 3.0 * deg, 2.0 * deg, 110.0 * deg, 0.1, 0.1, 0.9]]).T
    x0_c4 = np.array([[-2.0, -1.5, 3.5, -0.7, -0.4, 0.0, -15.0 * deg, 10.0 * deg, 45.0 * deg, -0.5, 0.4, 0.2]]).T
    x0_c5 = np.array([[2.5, -2.0, 5.0, 0.8, 0.2, -0.3, 6.0 * deg, -5.0 * deg, -90.0 * deg, 0.4, -0.2, 0.7]]).T
    x0_c6 = np.array([[-3.0, 1.0, 2.2, -1.2, 0.8, 0.6, -8.0 * deg, 18.0 * deg, 10.0 * deg, -0.3, 0.8, 0.0]]).T
    x0_c7 = np.array([[0.2, 3.0, 4.5, 0.0, 0.5, 0.0, 0.0 * deg, 4.0 * deg, 160.0 * deg, 0.0, 0.2, -0.6]]).T
    x0_c8 = np.array([[1.5, -3.5, 1.2, 0.0, -1.0, -0.8, 20.0 * deg, -8.0 * deg, 20.0 * deg, 1.0, -0.4, 0.1]]).T
    x0_c9 = np.array([[-2.5, 2.5, 6.0, -0.5, 1.2, 0.3, -10.0 * deg, 12.0 * deg, -140.0 * deg, -0.6, 0.5, -0.7]]).T
    x0_c10 = np.array([[3.0, 0.0, 4.0, 1.0, -0.2, 0.0, 4.0 * deg, -2.0 * deg, 60.0 * deg, 0.2, -0.1, 1.5]]).T
    x0_c11 = np.array([[-1.5, -2.5, 0.9, -0.3, 0.4, 1.0, -6.0 * deg, 8.0 * deg, -70.0 * deg, -0.3, 0.3, -0.5]]).T
    x0_c = (np.vstack((x0_c0, x0_c1, x0_c2, x0_c3, x0_c4, x0_c5, x0_c6, x0_c7, x0_c8, x0_c9, x0_c10, x0_c11)))

    # PRBS Set
    x0_p0 = np.array([[0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 1.0 * deg, -1.0 * deg, 0.0 * deg, 0.02, -0.02, 0.02]]).T
    x0_p1 = np.array([[0.5, -0.3, 3.0, 0.1, 0.0, 0.0, 2.0 * deg, -3.0 * deg, 30.0 * deg, 0.05, -0.05, 0.10]]).T
    x0_p2 = np.array([[-1.5, 0.5, 2.8, -0.8, 0.6, 0.0, -4.0 * deg, 5.0 * deg, -20.0 * deg, -0.2, 0.2, -0.1]]).T
    x0_p3 = np.array([[1.0, 1.5, 4.0, 0.4, -0.6, 0.1, 3.0 * deg, 15.0 * deg, 10.0 * deg, 0.1, 0.7, 0.0]]).T
    x0_p4 = np.array([[-2.0, -1.5, 3.5, -0.3, -0.5, 0.0, 18.0 * deg, -5.0 * deg, 40.0 * deg, 0.8, -0.2, 0.1]]).T
    x0_p5 = np.array([[1.5, -2.0, 5.0, 0.6, 0.2, -0.2, 4.0 * deg, 3.0 * deg, 120.0 * deg, 0.2, 0.2, 0.9]]).T
    x0_p6 = np.array([[-3.0, 1.0, 3.0, -0.5, 0.9, 0.3, -12.0 * deg, 14.0 * deg, 60.0 * deg, -0.4, 0.6, 0.2]]).T
    x0_p7 = np.array([[0.2, 3.0, 4.2, 0.0, 0.3, 0.0, 2.0 * deg, 2.0 * deg, 170.0 * deg, 0.0, 0.1, -0.5]]).T
    x0_p8 = np.array([[1.5, -3.5, 1.1, 0.0, -0.8, -0.7, 10.0 * deg, -6.0 * deg, 25.0 * deg, 0.4, -0.3, 0.1]]).T
    x0_p9 = np.array([[-2.5, 2.5, 6.0, -0.4, 1.1, 0.2, -8.0 * deg, 10.0 * deg, -130.0 * deg, -0.4, 0.4, -1.2]]).T
    x0_p10 = np.array([[3.0, 0.0, 4.0, 0.9, -0.1, 0.0, 5.0 * deg, -4.0 * deg, 70.0 * deg, 1.1, -0.9, 0.3]]).T
    x0_p11 = np.array([[-1.5, -2.5, 0.9, -0.2, 0.3, 1.2, -6.0 * deg, 9.0 * deg, -60.0 * deg, -0.3, 0.4, -0.5]]).T
    x0_p = (np.vstack((x0_p0, x0_p1, x0_p2, x0_p3, x0_p4, x0_p5, x0_p6, x0_p7, x0_p8, x0_p9, x0_p10, x0_p11)))

    x0 = np.vstack((x0_s, x0_c, x0_p))
    return x0.astype(np.float32)
