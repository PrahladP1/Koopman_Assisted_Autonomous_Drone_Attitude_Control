import numpy as np
from ODE import signal_configs as sc
from ODE import constraints

import sys
# I had issues with importing module object directly from signal_configs.py,
# so I hardcoded the module object import here

sc_in_gen_seq = sys.modules['ODE.signal_configs']
sc_in_gen_seq.chirp_control_seq[2] = dict(
    scale=0.35,
    chirp_ranges=[(1.0,4),(1.0,5),(1.0,8),(1.0,6)],
    sigmas=[0.15,0.15,0.15,0.15])

dt = 0.01
t = np.arange(0, 10 + dt, dt)
g = 9.81
m = 0.65
T_hover = m*g
cos_max_tilt = np.cos(np.deg2rad(35))*np.cos(np.deg2rad(35))
T_center = T_hover/cos_max_tilt

def generate_control_sequence(t, bounds, freqs=None, sigmas=None, chirp_ranges=None, mode=None, seed=None):
    rng = np.random.default_rng(seed)
    U = []
    for i in range(4):
        if mode == "sin":
            u = sc.noisy_sinusoid(t, bounds[i][0], bounds[i][1], freqs[i], sigmas[i], rng)
        elif mode == "chirp":
            f0, f1 = chirp_ranges[i]
            u = sc.noisy_chirp(t, bounds[i][0], bounds[i][1], f0, f1, sigmas[i], rng)
        U.append(u)

    return np.column_stack(U)

def _torque_bounds(scale):
    """Return (roll, pitch, yaw) bounds scaled by scale."""
    return [(constraints.PHYSICAL_BOUNDS[1][0] * scale,
             constraints.PHYSICAL_BOUNDS[1][1] * scale),
            (constraints.PHYSICAL_BOUNDS[2][0] * scale,
             constraints.PHYSICAL_BOUNDS[2][1] * scale),
            (constraints.PHYSICAL_BOUNDS[3][0] * scale,
             constraints.PHYSICAL_BOUNDS[3][1] * scale),]

def _thrust_bounds(scale):
    """Return thrust bounds centered on compensated hover thrust."""
    T_amp = scale * T_hover * 0.5
    return (max(0.0, T_center - T_amp), T_center + T_amp)

# Sine
U_sin_list = []
for i, cfg in enumerate(sc.control_seq):
    scale = cfg["scale"]
    bounds = [_thrust_bounds(scale)] + _torque_bounds(scale)
    U_raw = generate_control_sequence(t, bounds=bounds, freqs=cfg["freqs"], sigmas=cfg["sigmas"], mode="sin", seed=200+i)
    U_sat = np.array([constraints.saturate_control(u) for u in U_raw])
    U_sin_list.append(U_sat)

# Chirp
U_chirp_list = []
for i, cfg in enumerate(sc.chirp_control_seq):
    scale = cfg["scale"]
    bounds = [_thrust_bounds(scale)] + _torque_bounds(scale)
    U_raw = generate_control_sequence(t, bounds=bounds, chirp_ranges=cfg["chirp_ranges"], sigmas=cfg["sigmas"], mode="chirp", seed=200+i)
    U_sat = np.array([constraints.saturate_control(u) for u in U_raw])
    U_chirp_list.append(U_sat)

# PRBS
def generate_prbs_control_sequence(t, bounds, dwell_steps, seed=None):
    rng = np.random.default_rng(seed)
    U = np.column_stack([sc.generate_prbs(t, bounds[i][0], bounds[i][1], dwell_steps[i], rng) for i in range(4)])
    return U

U_prbs_list = []
for i, cfg in enumerate(sc.prbs_control_seq):
    scale = cfg["scale"]
    bounds = [_thrust_bounds(scale)] + _torque_bounds(scale)
    U_raw = generate_prbs_control_sequence(t, bounds=bounds, dwell_steps=cfg["dwell"], seed=300+i)
    U_sat = np.array([constraints.saturate_control(u) for u in U_raw])
    U_prbs_list.append(U_sat)