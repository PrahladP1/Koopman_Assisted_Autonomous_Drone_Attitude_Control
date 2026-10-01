from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from ODE import init_conditions as ic
from ODE import gen_seq
from ODE import ivp_solve

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR/"Data"
SINE_DIR = DATA_DIR/"State-Control History Sine"
CHIRP_DIR = DATA_DIR/"State-Control History Chirp"
PRBS_DIR = DATA_DIR/"State-Control History PRBS"

STATE_LABELS = [
    "x position [m]",
    "y position [m]",
    "z position [m]",
    "x velocity [m/s]",
    "y velocity [m/s]",
    "z velocity [m/s]",
    "roll [rad]",
    "pitch [rad]",
    "yaw [rad]",
    "roll rate [rad/s]",
    "pitch rate [rad/s]",
    "yaw rate [rad/s]"]

def plot_positions_and_velocities(time_history, state_history, traj_id=None):
    """
    Plot position and velocity states.
    """
    time_history = np.asarray(time_history)
    state_history = np.asarray(state_history)

    label = (f"x0_{traj_id}" if traj_id is not None else "trajectory")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Position
    axes[0].plot(time_history, state_history[:, 0], label="x position [m]")
    axes[0].plot(time_history, state_history[:, 1], label="y position [m]")
    axes[0].plot(time_history, state_history[:, 2], label="z position [m]")
    axes[0].set_ylabel("Position [m]")
    axes[0].set_title(f"Quadrotor Position ({label})")
    axes[0].legend()
    axes[0].grid(True)

    # Velocity
    axes[1].plot(time_history, state_history[:, 3], label="x velocity [m/s]")
    axes[1].plot(time_history, state_history[:, 4], label="y velocity [m/s]")
    axes[1].plot(time_history, state_history[:, 5], label="z velocity [m/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[1].set_title(f"Quadrotor Velocity ({label})")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()

    return fig, axes

def plot_attitude_and_rates(time_history, state_history, traj_id=None):
    """
    Plot attitude and body-rate states.
    """
    time_history = np.asarray(time_history)
    state_history = np.asarray(state_history)
    label = (f"x0_{traj_id}" if traj_id is not None else "trajectory")
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Attitude
    axes[0].plot(time_history, state_history[:, 6], label="Roll [rad]")
    axes[0].plot(time_history, state_history[:, 7], label="Pitch [rad]")
    axes[0].plot(time_history, state_history[:, 8], label="Yaw [rad]")
    axes[0].set_ylabel("Angle [rad]")
    axes[0].set_title(f"Quadrotor Attitude ({label})")
    axes[0].legend()
    axes[0].grid(True)

    # Body rates
    axes[1].plot(time_history, state_history[:, 9], label="Roll rate [rad/s]")
    axes[1].plot(time_history, state_history[:, 10], label="Pitch rate [rad/s]")
    axes[1].plot(time_history, state_history[:, 11], label="Yaw rate [rad/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Body rate [rad/s]")
    axes[1].set_title(f"Quadrotor Body Rates ({label})")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()

    return fig, axes

def load_state_history(filename):
    """
    Load a saved state-history CSV.
    """
    state_history = np.loadtxt(filename, delimiter=",")

    return state_history

def plot_saved_trajectory(filename, dt=0.01, traj_id=None):
    """
    Load and plot a saved trajectory.
    The time vector is reconstructed from dt because the current
    state-history CSV contains only the state columns.
    """
    state_history = load_state_history(filename)
    time_history = np.arange(len(state_history))*dt
    plot_positions_and_velocities(time_history, state_history, traj_id=traj_id)
    plot_attitude_and_rates(time_history, state_history, traj_id=traj_id)
    plt.show()

    print("Roll:  min={:.3f} max={:.3f} rad".format(
        state_history[:, 6].min(), state_history[:, 6].max()))
    print("Pitch: min={:.3f} max={:.3f} rad".format(
        state_history[:, 7].min(), state_history[:, 7].max()))
    print("Yaw:   min={:.3f} max={:.3f} rad".format(
        state_history[:, 8].min(), state_history[:, 8].max()))
    print("p:     min={:.3f} max={:.3f} rad/s".format(
        state_history[:, 9].min(), state_history[:, 9].max()))
    print("q:     min={:.3f} max={:.3f} rad/s".format(
        state_history[:, 10].min(), state_history[:, 10].max()))
    print("r:     min={:.3f} max={:.3f} rad/s".format(
        state_history[:, 11].min(), state_history[:, 11].max()))

    '''
    Check for consistency in physics and data processing below. Some trajectories had problems with actuator saturation, motor
    mixing, and physically unreasonable time evolution, which helped me diagnose why certain states and inputs were strange.
    For instance, the diagnostics below identified major issues with the way the state space was represented with inconsistent
    coordinate system definitions, inconsistent signage (some ODEs were defined with direction of thrust as positive, while other
    ODEs were defined with direction of acceleration due to gravity as positive).
    '''
    # Check for saturation
    p_sat = np.mean(np.abs(state_history[:, 9]) > 3.9)
    q_sat = np.mean(np.abs(state_history[:, 10]) > 3.9)
    r_sat = np.mean(np.abs(state_history[:, 11]) > 3.9)
    print(f"\nFraction of time at body rate clip limits:")
    print(f"p: {p_sat:.1%}, q: {q_sat:.1%}, r: {r_sat:.1%}")
    print("(should all be <5% for healthy dynamics)")

    # x0_7 uses control_seq[7] which has scale=1.0
    x0 = ic.sine_ic()[7].reshape(-1)
    time_hist, state_hist = ivp_solve.simulate_trajectory(x0, gen_seq.U_sin_list[7])
    print("Roll:  min={:.3f} max={:.3f} rad".format(
        state_hist[:, 6].min(), state_hist[:, 6].max()))
    print("Pitch: min={:.3f} max={:.3f} rad".format(
        state_hist[:, 7].min(), state_hist[:, 7].max()))
    print("Yaw:   min={:.3f} max={:.3f} rad".format(
        state_hist[:, 8].min(), state_hist[:, 8].max()))
    print("p:     min={:.3f} max={:.3f} rad/s".format(
        state_hist[:, 9].min(), state_hist[:, 9].max()))
    print("q:     min={:.3f} max={:.3f} rad/s".format(
        state_hist[:, 10].min(), state_hist[:, 10].max()))
    print("r:     min={:.3f} max={:.3f} rad/s".format(
        state_hist[:, 11].min(), state_hist[:, 11].max()))

    p_sat = np.mean(np.abs(state_hist[:, 9]) > 3.9)
    q_sat = np.mean(np.abs(state_hist[:, 10]) > 3.9)
    r_sat = np.mean(np.abs(state_hist[:, 11]) > 3.9)
    print(f"\np: {p_sat:.1%}, q: {q_sat:.1%}, r: {r_sat:.1%}")

    print("Checking all 12 chirp trajectories...")
    any_failed = False
    for i, x0 in enumerate(ic.chirp_ic()):
        _, states = ivp_solve.simulate_trajectory(
            x0.reshape(-1), gen_seq.U_chirp_list[i])
        p_sat = np.mean(np.abs(states[:, 9]) > 3.9)
        q_sat = np.mean(np.abs(states[:, 10]) > 3.9)
        r_sat = np.mean(np.abs(states[:, 11]) > 3.9)
        phi_max = np.rad2deg(np.abs(states[:, 6]).max())
        theta_max = np.rad2deg(np.abs(states[:, 7]).max())
        alt_min = states[:, 2].min()
        ok = (p_sat < 0.05 and q_sat < 0.05 and r_sat < 0.05
              and phi_max < 50.0
              and theta_max < 50.0
              and alt_min > 0.5)
        status = "PASS" if ok else "FAIL"
        print(f"  x0_c{i}: {status} | "
              f"phi={phi_max:.1f}° theta={theta_max:.1f}° "
              f"alt_min={alt_min:.2f}m "
              f"sat=({p_sat:.1%},{q_sat:.1%},{r_sat:.1%})")
        if not ok:
            any_failed = True
    if not any_failed:
        print("All chirp trajectories passed.")
    else:
        print("Some chirp trajectories failed.")

    print("\nChecking all 12 PRBS trajectories...")
    any_failed = False
    for i, x0 in enumerate(ic.prbs_ic()):
        _, states = ivp_solve.simulate_trajectory(
            x0.reshape(-1), gen_seq.U_prbs_list[i])
        p_sat = np.mean(np.abs(states[:, 9]) > 3.9)
        q_sat = np.mean(np.abs(states[:, 10]) > 3.9)
        r_sat = np.mean(np.abs(states[:, 11]) > 3.9)
        phi_max = np.rad2deg(np.abs(states[:, 6]).max())
        theta_max = np.rad2deg(np.abs(states[:, 7]).max())
        alt_min = states[:, 2].min()
        ok = (p_sat < 0.05 and q_sat < 0.05 and r_sat < 0.05
              and phi_max < 50.0
              and theta_max < 50.0
              and alt_min > 0.5)
        status = "PASS" if ok else "FAIL"
        print(f"  x0_p{i}: {status} | "
              f"phi={phi_max:.1f}° theta={theta_max:.1f}° "
              f"alt_min={alt_min:.2f}m "
              f"sat=({p_sat:.1%},{q_sat:.1%},{r_sat:.1%})")
        if not ok:
            any_failed = True
    if not any_failed:
        print("All PRBS trajectories passed.")
        print("\nAll 36 trajectories validated. Safe to regenerate full dataset.")
    else:
        print("Some PRBS trajectories failed.")

if __name__ == "__main__":
    filename = SINE_DIR/"state_history_x0_0sin.csv"
    plot_saved_trajectory(filename, dt=0.01, traj_id=0)