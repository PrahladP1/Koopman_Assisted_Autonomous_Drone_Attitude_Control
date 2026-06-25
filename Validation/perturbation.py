from dataclasses import dataclass
import numpy as np

@dataclass
class PerturbedConfigs:
    pos_std: float = 0.2
    vel_std: float = 0.0
    ang_std: float = 0.0
    body_std: float = 0.0

def perturb_init_cond(x_nom: np.ndarray, config: PerturbedConfigs, rng: np.random.Generator | None = None,):
    if rng is None:
        rng = np.random.default_rng()
    x = np.asarray(x_nom).copy()
    if x.shape != (12,):
        raise ValueError(f"Expected shape (12,), received {x.shape}")
    x[0:3] += rng.normal(0.0, config.pos_std, size=3)
    x[3:6] += rng.normal(0.0, config.vel_std, size=3)
    x[6:9] += rng.normal(0.0, config.ang_std, size=3)
    x[9:12] += rng.normal(0.0, config.body_std, size=3)

    init_pos_err = np.linalg.norm(x[0:3] - x_nom[0:3])
    init_vel_err = np.linalg.norm(x[3:6] - x_nom[3:6])
    init_ang_err = np.linalg.norm(x[6:9] - x_nom[6:9])
    init_body_err = np.linalg.norm(x[9:12] - x_nom[9:12])
    return x, init_pos_err

def sample_init_cond(init_cond: np.ndarray, rng: np.random.Generator | None = None,):
    if rng is None:
        rng = np.random.default_rng()
    idx = rng.integers(0, init_cond.shape[0])
    return idx, init_cond[idx].copy()

def gen_perturbed_init_cond(init_cond: np.ndarray, config: PerturbedConfigs, rng: np.random.Generator | None = None,):
    if rng is None:
        rng = np.random.default_rng()
    idx, nom = sample_init_cond(init_cond, rng,)
    perturbed, init_err = perturb_init_cond(nom, config, rng,)

    return idx, nom, perturbed, init_err
