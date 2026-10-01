import numpy as np

def noisy_sinusoid(t, umin, umax, freq, sigma, rng=None):
    """
    Generate a clipped noisy sinusoid over time t.
    Returns shape (len(t),)
    """
    if rng is None:
        rng = np.random
    base = 0.5*(umax - umin)*np.sin(2*np.pi*freq*t) + 0.5*(umax + umin)
    u = base + sigma*rng.standard_normal(size=len(t))

    return np.clip(u, umin, umax)

def noisy_chirp(t, umin, umax, f0, f1, sigma, rng=None):
    """
    Generate clipped noisy chirp signal
    f0 -> starting frequency
    f1 -> ending frequency
    """
    if rng is None:
        rng = np.random
    T = t[-1]
    k = (f1 - f0)/T # sweep rate
    phase = 2*np.pi*(f0*t + 0.5*k*t**2)
    base_c = 0.5*(umax - umin)*np.sin(phase) + 0.5*(umax + umin)
    u = base_c + sigma * rng.standard_normal(len(t))

    return np.clip(u, umin, umax)

def generate_prbs(t, umin, umax, dwell_steps=5, rng=None):
    """
    Generate a bounded PRBS signal.
    dwell_steps : number of samples each value is held
    """
    if rng is None:
        rng = np.random
    N = len(t)

    # number of switching events
    n_blocks = int(np.ceil(N/dwell_steps))
    # random binary sequence
    bits = rng.integers(0, 2, size=n_blocks)
    # map {0,1} -> {umin, umax}
    levels = np.where(bits == 0, umin, umax)
    # hold values
    u = np.repeat(levels, dwell_steps)[:N]

    return u

# Sine Sequence
control_seq = [
    dict(scale=0.25,
         freqs=[1.0,2.0,3.0,4.0],
         sigmas=[0.05,0.05,0.05,0.05]),
    dict(scale=0.35,
         freqs=[12,15,10,7],
         sigmas=[0.10,0.10,0.10,0.10]),
    dict(scale=0.40,
         freqs=[2.2,3.8,5.7,6.7],
         sigmas=[0.15,0.15,0.15,0.15]),
    dict(scale=0.50,
         freqs=[2.5,4.2,9.4,13.5],
         sigmas=[0.20,0.20,0.20,0.20]),
    dict(scale=0.50,
         freqs=[1.5, 2.0, 1.8, 2.5],
         sigmas=[0.25,0.25,0.25,0.25]),
    dict(scale=0.50,
         freqs=[3.9,5.9,1.7,2.4],
         sigmas=[0.30,0.30,0.30,0.30]),
    dict(scale=0.50,
         freqs=[1.5,3.9,7.9,13.4],
         sigmas=[0.35,0.35,0.35,0.35]),
    dict(scale=0.38,
         freqs=[1.6,9.8,1.6,7.8],
         sigmas=[0.40,0.40,0.40,0.40]),
    dict(scale=0.46,
         freqs=[5.0,6.0,7.0,8.0],
         sigmas=[0.50,0.50,0.50,0.50]),
    dict(scale=0.45,
         freqs=[0.7,2.5,6.0,12.0],
         sigmas=[0.60,0.60,0.60,0.60]),
    dict(scale=0.20,
         freqs=[2.7,7.6,4.8,3.1],
         sigmas=[0.75,0.75,0.75,0.75]),
    dict(scale=0.19,
         freqs=[0.7,0.3,0.4,0.8],
         sigmas=[0.90,0.90,0.90,0.90]),]
# Chirp Sequence
chirp_control_seq = [
    dict(scale=0.25,
         chirp_ranges=[(0.2,3),(0.5,4),(0.3,5),(0.4,6)],
         sigmas=[0.05,0.05,0.05,0.05]),
    dict(scale=0.35,
         chirp_ranges=[(1.0, 4), (1.0, 5), (1.0, 8), (1.0, 6)],
         sigmas=[0.15, 0.15, 0.15, 0.15]),
    dict(scale=0.50,
         chirp_ranges=[(0.1,2),(0.2,4),(0.6,10),(0.4,1)],
         sigmas=[0.15,0.15,0.15,0.15]),
    dict(scale=0.45,
         chirp_ranges=[(0.5,10),(0.7,3),(0.6,2),(0.8,2)],
         sigmas=[0.20,0.20,0.20,0.20]),
    dict(scale=0.30,
         chirp_ranges=[(0.1,1),(0.2,2),(0.3,3),(0.4,4)],
         sigmas=[0.25,0.25,0.25,0.25]),
    dict(scale=0.42,
         chirp_ranges=[(0.5,5),(0.8,8),(0.3,2),(0.4,5)],
         sigmas=[0.30,0.30,0.30,0.30]),
    dict(scale=0.23,
         chirp_ranges=[(0.3,8),(0.1,9),(0.2,6),(0.4,10)],
         sigmas=[0.35,0.35,0.35,0.35]),
    dict(scale=0.12,
         chirp_ranges=[(0.1,9),(0.2,4),(0.4,6),(0.7,2)],
         sigmas=[0.40,0.40,0.40,0.40]),
    dict(scale=0.1,
         chirp_ranges=[(0.1,10),(0.3,6),(0.2,8),(0.5,10)],
         sigmas=[0.50,0.50,0.50,0.50]),
    dict(scale=0.4,
         chirp_ranges=[(1.0, 5), (1.0, 8), (1.0, 7), (1.0, 8)],
         sigmas=[0.60,0.60,0.60,0.60]),
    dict(scale=0.5,
         chirp_ranges=[(0.5,3),(0.2,7),(0.4,9),(0.1,10)],
         sigmas=[0.75,0.75,0.75,0.75]),
    dict(scale=0.34,
         chirp_ranges=[(0.2,10),(0.2,5),(0.2,4),(0.3,9)],
         sigmas=[0.90,0.90,0.90,0.90])]

# PRBS Sequence
prbs_control_seq = [
                    dict(scale=0.15, dwell=[5, 6, 6, 7]),
                    dict(scale=0.18, dwell=[5, 6, 6, 7]),
                    dict(scale=0.20, dwell=[5, 6, 6, 7]),
                    dict(scale=0.22, dwell=[5, 6, 6, 7]),
                    dict(scale=0.22, dwell=[5, 6, 6, 7]),
                    dict(scale=0.18, dwell=[5, 6, 6, 7]),  # was 0.25: pinned phi/theta at the 50 deg clamp
                    dict(scale=0.20, dwell=[5, 6, 6, 7]),  # was 0.25: pinned phi at the 50 deg clamp
                    dict(scale=0.12, dwell=[5, 6, 6, 7]),  # was 0.25: pinned phi/theta at the 50 deg clamp
                    dict(scale=0.25, dwell=[5, 6, 6, 7]),
                    dict(scale=0.25, dwell=[5, 6, 6, 7]),
                    dict(scale=0.25, dwell=[5, 6, 6, 7]),
                    dict(scale=0.18, dwell=[5, 6, 6, 7])]  # was 0.25: pinned phi at the 50 deg clamp
