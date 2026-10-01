from pathlib import Path
import numpy as np
from ODE import gen_seq
from ODE import init_conditions as ic
from ODE import state_space as ss

# Simulation configuration
T_SIM = 10.0
DT = 0.01

# Output locations sorted by signal type
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"Data"
SIGNAL_DIRS = {
    "sin": DATA_DIR/"State-Control History Sine",
    "chirp": DATA_DIR/"State-Control History Chirp",
    "prbs": DATA_DIR/"State-Control History PRBS",}

# Plant parameters
PARAMS = {
    "m": 0.65, # mass [kg]
    "g": 9.81, # acceleration due to gravity [m.s^-2]
    "kt": np.diag([0.1, 0.1, 0.1]), # Aerodynamic thrust drag coeff. [N.s/m]
    "kr": np.diag([0.1, 0.1, 0.1]), # Aerodynamic moment drag coeff. [N.m*s]
    "I": np.diag([7.5e-3, 7.5e-3, 1.3e-2]), # Mass moment of inertia [kg.m^2]
    "l": 0.23, # Arm length
    "Ir": 6e-5, # Motor mass moment of inertia [kg.m^2]
    "w_r": 848, # Maximum motor angular velocity
    "km": 7.5e-7, # Moment coeff. [N.m.s^2]
    "kf": 3.13e-5} # Thrust coeff. [N.s^2]

def simulate_trajectory(x0, U, T_sim=T_SIM, dt=DT, params=PARAMS):
    """
    Simulate one nonlinear quadrotor trajectory.
    x0 : array-like, shape (12,)
        Initial state.
    U : ndarray, shape (N, 4)
        Control history.
    T_sim : float
        Simulation duration [s].
    dt : float
        Simulation timestep [s].
    params : dict
        Nonlinear plant parameters.
    Returns
    time_history : ndarray
        Simulation timestamps.
    state_history : ndarray
        Simulated state trajectory.
    """

    x0 = np.asarray(x0, dtype=float).reshape(-1)
    U = np.asarray(U, dtype=float)

    if x0.size != 12:
        raise ValueError(f"Expected x0 with 12 states, got {x0.size}.")
    if U.ndim != 2 or U.shape[1] != 4:
        raise ValueError(f"Expected U with shape (N, 4), got {U.shape}.")
    t_grid = np.arange(0.0, T_sim + dt, dt)
    if len(U) < len(t_grid):
        raise ValueError(f"Control sequence is too short. "
                         f"Need at least {len(t_grid)} samples, got {len(U)}.")

    # The solver stores the state at each t_k and advances
    # through the first len(t_grid)-1 intervals.
    num_steps = len(t_grid) - 1
    x_current = ss.project_state(x0)
    time_history = np.empty(num_steps)
    state_history = np.empty((num_steps, 12))

    for k in range(num_steps):
        tk = t_grid[k]
        time_history[k] = tk
        state_history[k] = x_current
        x_current = ss.step_dynamics(tk, x_current, U, k, dt, params)

    return time_history, state_history

def simulate_dataset(X0_list, U_list, T_sim=T_SIM, dt=DT,
                     params=PARAMS):
    """
    Simulate a collection of trajectories.
    Returns
    results : list of dict
        Each entry contains:
            traj_id
            time
            state history
            control history
    """

    if len(X0_list) != len(U_list):
        raise ValueError("X0_list and U_list must contain the same number of trajectories.")

    results = []
    for traj_id, (x0, U) in enumerate(zip(X0_list, U_list)):
        time_history, state_history = simulate_trajectory(x0, U, T_sim=T_sim, dt=dt, params=params)
        results.append({
            "traj_id": traj_id,
            "time": time_history,
            "states": state_history,
            "controls": np.asarray(U)[:len(time_history)]})

    return results

def save_trajectory(result, signal_name, out_dir=None):
    """
    Save one simulated trajectory to CSV, in the Data subfolder
    matching its signal type ("sin" -> Sine, "chirp" -> Chirp, "prbs" -> PRBS).
    """
    traj_id = result["traj_id"]
    states = result["states"]
    filename = f"state_history_x0_{traj_id}{signal_name}.csv"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir/filename
    else:
        filepath = Path(filename)
    np.savetxt(filepath, states, delimiter=",")

    return str(filepath)

def main():
    """
    Run the sine, chirp, and PRBS simulation sets, saving each
    trajectory into its matching Data subfolder.
    """
    signal_sets = [
        ("sin", ic.sine_ic(), gen_seq.U_sin_list),
        ("chirp", ic.chirp_ic(), gen_seq.U_chirp_list),
        ("prbs", ic.prbs_ic(), gen_seq.U_prbs_list),]

    for signal_name, X0_list, U_list in signal_sets:
        results = simulate_dataset(X0_list, U_list, T_sim=T_SIM, dt=DT, params=PARAMS)
        for result in results:
            filename = save_trajectory(
                result, signal_name, out_dir=SIGNAL_DIRS[signal_name])
            print(
                f"Simulation completed for "
                f"x0_{result['traj_id']} ({signal_name}) over {T_SIM} seconds "
                f"with dt={DT}.")
            print(f"Time shape: {result['time'].shape}")
            print(f"State shape: {result['states'].shape}")
            print(f"Saved: {filename}")

    # Check with dummy trajectory to see if physics and data processing pipelines are consistent
    x_hover = np.zeros(12)
    x_hover[2] = 2.0
    u_hover = np.array([PARAMS['m']*PARAMS['g'], 0.0, 0.0, 0.0])

    U_test = np.tile(u_hover, (101, 1))
    print("U shape:", U_test.shape)  # must be (101, 4)
    print("U[0]:", U_test[0])  # must be [6.377, 0, 0, 0]

    time_history, state_history_hover = simulate_trajectory(x_hover, U_test, T_sim=1.0)

    print("t=0:   ", state_history_hover[0])
    print("t=0.01:", state_history_hover[1])
    print("t=1.0: ", state_history_hover[-1])

    assert abs(state_history_hover[-1, 0]) < 0.01, f"x drift: {state_history_hover[-1, 0]}"
    assert abs(state_history_hover[-1, 1]) < 0.01, f"y drift: {state_history_hover[-1, 1]}"
    assert abs(state_history_hover[-1, 2] - 2.0) < 0.01, f"alt drift: {state_history_hover[-1, 2]}"
    print("PASS: hover trajectory stays put")

    print("First 5 thrust values:", gen_seq.U_sin_list[0][:5, 0])
    print("Hover thrust:", PARAMS['m']*PARAMS['g'])
    print("Mean thrust:", gen_seq.U_sin_list[0][:, 0].mean())
    print("Min thrust:", gen_seq.U_sin_list[0][:, 0].min())
    print("Max thrust:", gen_seq.U_sin_list[0][:, 0].max())

    x0 = ic.sine_ic()[0].reshape(-1)
    u0 = gen_seq.U_sin_list[0][0]
    print("x0:", x0)
    print("u0:", u0)

    dx = ss.quad_ode(0, x0, u0, PARAMS)
    print("dx[5]:", dx[5])
    print("vz:", x0[5])
    print("phi:", x0[6], "theta:", x0[7])
    print("cphi*ctheta:", np.cos(x0[6]) * np.cos(x0[7]))
    print("kt_z*vz:", PARAMS["kt"][2, 2] * x0[5])
    print("mg:", PARAMS["m"] * 9.81)
    print("u1*cphi*ctheta:", u0[0] * np.cos(x0[6]) * np.cos(x0[7]))
    print("bracket:", PARAMS["kt"][2, 2] * x0[5] - PARAMS["m"] * 9.81 + u0[0] * np.cos(x0[6]) * np.cos(x0[7]))

    time_hist, state_hist = simulate_trajectory(x0, gen_seq.U_sin_list[0], T_sim=10.0)
    print("vz at t=0:   ", state_hist[0, 5])
    print("vz at t=0.1: ", state_hist[10, 5])
    print("vz at t=0.5: ", state_hist[50, 5])
    print("vz at t=1.0: ", state_hist[100, 5])
    print("pz at t=1.0: ", state_hist[100, 2])
    print("pz at t=10.0:", state_hist[-1, 2])

    T_hover = PARAMS['m']*PARAMS['g']
    x_test = np.zeros(12)
    x_test[2] = 5.0
    x_test[9] = 4.0  # p = 4 rad/s
    dx = ss.quad_ode(0, x_test, np.array([T_hover, 0, 0, 0]), params=PARAMS)
    print("dx[6] with p=4:", dx[6])

if __name__ == "__main__":
    main()