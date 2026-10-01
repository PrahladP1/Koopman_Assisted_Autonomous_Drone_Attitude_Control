from pathlib import Path
import numpy as np
import pandas as pd
from ODE import constraints

DT = 0.01
T_SIM = 10.0
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"Data"
SIGNAL_DIRS = {
    "sin": DATA_DIR/"State-Control History Sine",
    "chirp": DATA_DIR/"State-Control History Chirp",
    "prbs": DATA_DIR/"State-Control History PRBS",}

def build_control_history(U_list, dt=DT, T_sim=T_SIM):
    """
    Convert a collection of control input sequences into a DataFrame.
    U_list : list of ndarray
        Each array has shape (N, 4) as [u_thrust u_roll u_pitch u_yaw]. Control values are taken
        directly from U_list. All centering and compensation is applied upstream in gen_seq.py.
    dt : float
        Sampling time [s].
    T_sim : float
        Simulation duration [s].
    Returns :
    df : pandas.DataFrame
        Control history for all trajectories.
    """
    t_grid = np.arange(0.0, T_sim + dt, dt)
    rows = []

    for traj_id, U in enumerate(U_list):
        U = np.asarray(U, dtype=float)

        if U.ndim != 2 or U.shape[1] != 4:
            raise ValueError(
                f"Trajectory {traj_id}: expected U shape (N, 4), "
                f"got {U.shape}.")

        if len(U) < len(t_grid):
            raise ValueError(
                f"Trajectory {traj_id}: control sequence has "
                f"{len(U)} samples but {len(t_grid)} are required.")

        for k, t in enumerate(t_grid):
            u_sat = constraints.saturate_control(U[k])
            rows.append([
                traj_id,
                f"x0_{traj_id}",
                t,
                u_sat[0],
                u_sat[1],
                u_sat[2],
                u_sat[3],])

    return pd.DataFrame(rows, columns=["traj_id", "ic_label", "t", "u_thrust", "u_roll", "u_pitch", "u_yaw"],)

def print_control_ranges(df):
    """Print control ranges from the DataFrame."""
    for i, col in enumerate(["u_thrust", "u_roll", "u_pitch", "u_yaw"]):
        v = df[col].to_numpy()
        print(f"u{i}: min={v.min():.3f}, max={v.max():.3f}, "
              f"mean={v.mean():.3f}, std={v.std():.3f}")

def main():
    from ODE import gen_seq
    for signal, U_list, fname in [
        ("sin",  gen_seq.U_sin_list,   "control_inputs_sin.csv"),
        ("chirp",gen_seq.U_chirp_list, "control_inputs_chirp.csv"),
        ("prbs", gen_seq.U_prbs_list,  "control_inputs_prbs.csv"),]:

        df = build_control_history(U_list)
        print(f"\n{signal.upper()} control ranges:")
        print_control_ranges(df)

        out_dir = SIGNAL_DIRS[signal]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir/fname

        df.to_csv(out_path, index=False)
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()