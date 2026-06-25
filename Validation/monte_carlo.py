from dataclasses import dataclass
import numpy as np
from perturbation import PerturbedConfigs, gen_perturbed_init_cond
from ref_gen import hover_ref
from metrics import compute_pos_err, compute_att_err, compute_sum_err
from statistics import summarize_stats
from Control.sil_control_test import exec_sil

@dataclass
class MonteCarloConfig:
    num_runs: int = 10000
    success_threshold: float = 5.0
    sim_step: int = 500
    random_seed: int = 42

def run_monte_carlo(model, config, x0_pool, device, mc_config: MonteCarloConfig, perturbation_config: PerturbedConfigs,):
    rng = np.random.default_rng(mc_config.random_seed)
    results = []
    init_err = []
    final_err = []
    success_flags = []
    print(f"Running Monte-Carlo ({mc_config.num_runs} runs)...")

    for run in range(mc_config.num_runs):
        ic_idx, nom_x0, perturbed_x0, init_err = gen_perturbed_init_cond(x0_pool, perturbation_config, rng)
        x_ref = hover_ref(nom_x0, mc_config.sim_step,)
        history = exec_sil(model=model, config=config, x0_phys=perturbed_x0, x_ref_phys=x_ref, device=device, sim_steps=mc_config.sim_step,)
        final_state = history["x"][-1]

        pos_err = compute_pos_err(final_state, x_ref[-1])
        att_err = compute_att_err(final_state, x_ref[-1])
        sum_err = compute_sum_err(final_state, x_ref[-1])
        success = (pos_err < mc_config.success_threshold)
        success_flags.append(success)

        results.append({"run": run,
                        "ic_index": ic_idx,
                        "success": success,
                        "initial_error": init_err,
                        "position_error": pos_err,
                        "attitude_error": att_err,
                        "total_error": sum_err,
                        "trajectory": history["x"],
                        "controls": history["u"],
                        "latent": history["z"]})

        if run % 100 == 0:
            print(f"Run {run:5d}"
                  f" | error = {pos_err:8.3f}")

    summary = summarize_stats(np.asarray(init_err), np.asarray(final_err), np.asarray(success_flags), mc_config.success_threshold)
    return results, summary