import numpy as np

m = 0.65
g = 9.81
T_hover = m*g
T_max = 3.0*T_hover
u_roll_max  = 30.0 * 7.5e-3 / 0.23  # ≈ 0.978 Nm
u_pitch_max = 30.0 * 7.5e-3 / 0.23  # ≈ 0.978 Nm
u_yaw_max   = 10.0 * 1.3e-2         # ≈ 0.130 Nm
PHYSICAL_BOUNDS = [(0.0, T_max),       # thrust
                   (-u_roll_max, u_roll_max),   # roll
                   (-u_pitch_max, u_pitch_max),   # pitch
                   (-u_yaw_max, u_yaw_max)]      # yaw

def saturate_control(u):
    u = np.asarray(u, dtype=float).copy()
    u[0] = np.clip(u[0], *PHYSICAL_BOUNDS[0])
    u[1] = np.clip(u[1], *PHYSICAL_BOUNDS[1])
    u[2] = np.clip(u[2], *PHYSICAL_BOUNDS[2])
    u[3] = np.clip(u[3], *PHYSICAL_BOUNDS[3])
    return u

def _safe_cos(angle, eps=1e-3):
    c = np.cos(angle)
    if abs(c) < eps:
        return eps if c >= 0 else -eps
    return c