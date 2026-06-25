from dataclasses import dataclass
import numpy as np

@dataclass
class RefConfig:
    dt: float = 0.01
    sim_step: int = 500

def hover_ref(x0: np.ndarray, sim_step: int):
    x_ref = np.tile(x0, (sim_step, 1))
    x_ref[:, 3:6] = 0.0
    x_ref[:, 6:9] = 0.0
    x_ref[:, 9:12] = 0.0
    return x_ref

def step_ref(x0, step_xyz, sim_step, step_time=100):
    x_ref = hover_ref(x0, sim_step)
    x_ref[step_time:, 0] += step_xyz[0]
    x_ref[step_time:, 1] += step_xyz[1]
    x_ref[step_time:, 2] += step_xyz[2]
    return x_ref

def circle_ref(center, radius, altitude, dt, sim_step, period=20.0):
    t = np.arange(sim_step)*dt
    ref = np.zeros((sim_step, 12))
    omega = (2*np.pi)/(period)

    ref[:, 0] = center[0]+radius*np.cos(omega*t)
    ref[:, 1] = center[1]+radius*np.sin(omega*t)
    ref[:, 2] = altitude
    ref[:, 3] = -radius*omega*np.sin(omega*t)
    ref[:, 4] = radius*omega*np.cos(omega*t)
    return ref

def fig8_ref(center, radius, altitude, dt, sim_step, period=20.0):
    t = np.arange(sim_step)*dt
    ref = np.zeros((sim_step, 12))
    omega = (2*np.pi)/(period)

    ref[:, 0] = center[0]+radius*np.sin(omega*t)
    ref[:, 1] = center[0]+radius*np.cos(omega*t)
    ref[:, 2] = altitude
    return ref

def helical_ref(center, radius, z_start, z_stop, dt, sim_step, revolutions=2):
    t = np.linspace(0, 1, sim_step)
    ref = np.zeros((sim_step, 12))
    theta = 2*np.pi*revolutions*t

    ref[:, 0] = center[0]+radius*np.cos(theta)
    ref[:, 1] = center[1]+radius*np.sin(theta)
    ref[:, 2] = z_start+(z_stop-z_start)*t
    return ref

def waypoint_ref(waypts, samples_per_seg):
    ref = []
    for k in range(len(waypts)-1):
        p0 = waypts[k]
        p1 = waypts[k+1]
        alpha = np.linspace(0, 1, samples_per_seg, endpoint=False)
        segment = np.outer(1-alpha, p0)+np.outer(alpha, p1)

        states = np.zeros((samples_per_seg, 12))
        states[:, 0:3] = segment
        ref.append(states)
    ref.append(np.zeros((1, 12)))
    ref[-1][0, 0:3] = waypts[-1]
    return np.vstack(ref)

def build_ref(ref_type, **kwargs):
    ref_type = ref_type.lower()
    if ref_type == "hover":
        return hover_ref(**kwargs)
    if ref_type == "step":
        return step_ref(**kwargs)
    if ref_type == "circle":
        return circle_ref(**kwargs)
    if ref_type == "fig8":
        return fig8_ref(**kwargs)
    if ref_type == "helix":
        return helical_ref(**kwargs)
    if ref_type == "waypoint":
        return waypoint_ref(**kwargs)