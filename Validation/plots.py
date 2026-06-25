import numpy as np
import matplotlib.pyplot as plt
from sympy.printing.pretty.pretty_symbology import line_width


def plot_xy_traj(trajectory, reference=None, title="XY Trajectory",):
    plt.figure(figsize=(8, 6))
    if reference is not None:
        plt.plot(trajectory[:, 0], reference[:, 1], '--', linewidth=2, label="Reference",)
    plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, label="Trajectory",)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_xyz_states(trajectory, reference=None,):
    labels = ["x", "y", "z"]
    fig, axs = plt.subplots(3, 1, fig_size=(10, 8))
    for i in range(3):
        axs[i].plot(trajectory[:, i], label="Measured")
        if reference is not None:
            axs[i].plot(reference[:, i], '--', label="Reference")
            axs[i].set_ylabel(labels[i])
            axs[i].grid(True)
    axs[-1].set_xlabel("Time [s")
    axs[0].legend()
    plt.tight_layout()

def plot_err_histogram(errs, bins=25,):
    plt.figure(figsize=(8, 5))
    plt.hist(errs, bins=bins, edgecolor="black",)
    plt.xlabel("Final Position error [m]")
    plt.ylabel("Frequency")
    plt.title("Monte Carlo Error Distribution")
    plt.grid(True)
    plt.tight_layout()

def plot_region_of_attract(init_err, final_err, success_flags,):
    plt.figure(figsize=(8, 5))
    plt.scatter(init_err[success_flags], final_err[success_flags], label="Success", alpha=0.5)
    plt.scatter(init_err[~success_flags], final_err[~success_flags], label="Failure", alpha=0.5)
    plt.xlabel("Initial Position Error [m]")
    plt.ylabel("Final Position Error [m]")
    plt.title("Emperical Region of Attraction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

def plot_success_probability(init_err, success_flags, bins=20,):
    edges = np.linspace(0, np.max(init_err), bins,)
    centers = 0.5*(edges[:-1]+edges[1:])
    probs = []
    for k in range(len(edges)-1):
        mask = ((init_err >= edges[k]) & (init_err < edges[k+1]))
        if np.sum(mask) == 0:
            probs.append(np.nan)
        else:
            probs.append(np.mean(success_flags[mask]))
    plt.figure(figsize=(8, 5))
    plt.plot(centers, probs, marker="o",)
    plt.xlabel("Initial Position Error [m]")
    plt.ylabel("Probability of Stabilization")
    plt.grid(True)
    plt.tight_layout()

def plot_controls(controls,):
    labels = ["u1", "u2", "u3", "u4"]
    fig, axs = plt.subplots(controls.shape[1], 1, fig_size=(10, 8), sharex=True,)
    for i in range(controls.shape[1]):
        axs[i].plot(controls[:, i],)
        axs[i].set_ylabel(labels[i])
        axs[i].grid(True)
    axs[-1].set_xlabel("Time [s]")
    plt.tight_layout()

def plot_latent_norm(latent,):
    nrm = np.linalg.norm(latent, axis=1,)
    plt.figure(figsize=(8, 5))
    plt.plot(nrm)
    plt.xlabel("Time [s]")
    plt.ylabel(r"$||z||_2$")
    plt.title("Latent State Norm")
    plt.grid(True)
    plt.tight_layout()

def plot_val_summary(history, reference=None,):
    plot_xy_traj(history["x"], reference,)
    plot_xyz_states(history["x"], reference,)
    plot_controls(history["u"])
    if "z" in history:
        plot_latent_norm(history["z"])
    plt.show()