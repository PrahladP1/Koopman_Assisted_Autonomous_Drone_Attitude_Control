import numpy as np
from ODE import constraints

STATE_DIM = 12
CONTROL_DIM = 4

def _safe_cos(angle, eps=1e-3):
    """
    Numerically safe cosine for divisions.
    Prevents division by values extremely close to zero while
    preserving the sign of cos(angle).
    """
    c = np.cos(angle)
    if abs(c) < eps:
        return eps if c >= 0.0 else -eps

    return c

def quad_ode(t, x, u, params):
    """
    Nonlinear 12-state quadrotor dynamics.
    State :
        x = [px, py, pz, vx, vy, vz, phi, theta, psi, p, q, r]
    Control :
        u = [u_thrust, u_roll, u_pitch, u_yaw]
    Parameters are supplied through the params dictionary.
    Returns :
        dx/dt : ndarray, shape (12,)
    """
    del t
    x = np.asarray(x, dtype=float).reshape(-1)
    u = np.asarray(u, dtype=float).reshape(-1)

    if x.size != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {x.size}.")
    if u.size != CONTROL_DIM:
        raise ValueError(f"Expected control dimension {CONTROL_DIM}, got {u.size}.")

    # Parameters
    m = params["m"]
    kt = params["kt"]
    kr = params["kr"]
    I = params["I"]
    l = params["l"]
    Ir = params["Ir"]
    w_r = params["w_r"]
    g = params['g']

    # State unpacking
    phi = x[6]
    theta = x[7]
    psi = x[8]
    p = x[9]
    q = x[10]
    r = x[11]

    # Trigonometric quantities
    cphi = np.cos(phi)
    sphi = np.sin(phi)
    cpsi = np.cos(psi)
    spsi = np.sin(psi)
    stheta = np.sin(theta)
    ctheta = _safe_cos(theta)
    theta_for_tan = np.clip(theta, -np.deg2rad(85.0), np.deg2rad(85.0))
    ttheta = np.tan(theta_for_tan)

    # State derivative
    dx = np.zeros(STATE_DIM, dtype=float)
    # position
    dx[0] = x[3]
    dx[1] = x[4]
    dx[2] = x[5]

    # velocity
    dx[3] = -(kt[0, 0]/m)*x[3] + (u[0]/m)*(sphi*spsi + cphi*cpsi*stheta)
    dx[4] = -(kt[1, 1]/m)*x[4] + (u[0]/m)*(sphi*cpsi - cphi*spsi*stheta)
    dx[5] = -(kt[2, 2]/m)*x[5] + (u[0]/m)*cphi*ctheta - g

    # attitude kinematics
    dx[6] = p + (q*sphi + r*cphi)*ttheta
    dx[7] = q*cphi - r*sphi
    dx[8] = (r*cphi + q*sphi)/ctheta

    # angular rates
    dx[9] = -(1.0/I[0, 0])*(kr[0, 0]*p - l*u[1] - I[1, 1]*q*r + I[2, 2]*q*r + Ir*q*w_r)
    dx[10] = -(1.0/I[1, 1])*(kr[1, 1]*q + l*u[2] - I[0, 0]*p*r + I[2, 2]*p*r + Ir*p*w_r)
    dx[11] = -(1.0/I[2, 2])*(kr[2, 2]*r + u[3] + I[0, 0]*p*q - I[1, 1]*p*q)

    return dx

def project_state(x):
    """
    Apply the state constraints used by the simulation.
    Returns a new array; the input is not modified.
    """
    x = np.asarray(x, dtype=float).reshape(-1).copy()
    if x.size != STATE_DIM:
        raise ValueError(f"Expected state dimension {STATE_DIM}, got {x.size}.")
    # Raise altitude floor
    x[2] = max(x[2], 1.0)
    # Minimum altitude
    x[2] = max(x[2], 0.1)

    # Velocity limits
    x[3] = np.clip(x[3], -15.0, 15.0)
    x[4] = np.clip(x[4], -15.0, 15.0)
    x[5] = np.clip(x[5], -5.0, 5.0)

    # Attitude limits
    phi_max = np.deg2rad(50.0)
    theta_max = np.deg2rad(50.0)
    x[6] = np.clip(x[6], -phi_max, phi_max)
    x[7] = np.clip(x[7], -theta_max, theta_max)
    # Yaw remains unconstrained as we would like the drone to be able to rotate about it's yaw axis, if necessary.

    # Body-rate limits
    x[9] = np.clip(x[9], -5.0, 5.0)
    x[10] = np.clip(x[10], -5.0, 5.0)
    x[11] = np.clip(x[11], -4.0, 4.0)

    return x

def step_dynamics(t, x, U, k, dt, params):
    """
    Advance the nonlinear plant by one timestep using RK4.
    - k1 uses U[k]
    - k2, k3, k4 use U[k+1]
    """
    U = np.asarray(U, dtype=float)
    if U.ndim != 2 or U.shape[1] != CONTROL_DIM:
        raise ValueError(f"Expected U with shape (N, {CONTROL_DIM}), got {U.shape}.")
    if not 0 <= k < len(U):
        raise IndexError(f"Control index k={k} is outside U with length {len(U)}.")

    x = np.asarray(x, dtype=float).reshape(-1)
    # Current control
    u1 = U[k]
    # Next control
    u2 = U[min(k+1, len(U)-1)]
    # RK4
    k1 = quad_ode(t, x, u1, params)
    k2 = quad_ode(t+0.5*dt, x+0.5*dt*k1, u2, params)
    k3 = quad_ode(t+0.5*dt, x+0.5*dt*k2, u2, params)
    k4 = quad_ode(t+dt, x+dt*k3, u2, params)
    x_next = x+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)

    return project_state(x_next)