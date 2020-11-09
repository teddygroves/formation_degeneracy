from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
from datetime import datetime
from functools import reduce
import numpy as np
import os
from sympy import Matrix
from matplotlib import pyplot as plt
from utils import generate_reparameterisation_matrix

STAN_FILE_NAIVE = "model_naive.stan"
STAN_FILE_SMART = "model_smart.stan"
RESULTS_DIR = "results/toy_case_study"
SAMPLE_DIR_NAIVE = reduce(os.path.join, [RESULTS_DIR, "samples", "naive"])
SAMPLE_DIR_SMART = reduce(os.path.join, [RESULTS_DIR, "samples", "smart"])
LAST_UPDATED_FILE = reduce(
    os.path.join, [RESULTS_DIR, "samples", "last_updated.txt"]
)
STAN_INPUT = os.path.join(RESULTS_DIR, "stan_input.json")

# Hardcoded information
PRIOR_LOC_THETA = [-900, -700, -600, -2000, -1400, -2700]
PRIOR_LOC_GAMMA = [-1000, -700, -600, -2000]
PRIOR_SCALE_THETA = 400
PRIOR_SCALE_GAMMA = 400
PRIOR_TAU = [50, 20]

# Simulation input
STOICHIOMETRY = [
    [0, 0,  0, -1],
    [0, 0,  0, -1],
    [0, 0,  0,  2],
    [0, 0,  -1, 0],
    [0, -1, -1, 0],
    [1, 1,  2,  0]
]
GROUP_INCIDENCE = [
    [1, 0,  0, 0],
    [0, 1,  0, 0],
    [0, 0,  1, 0],
    [0, 0,  0, 1],
    [1, 1,  0, 0],
    [0, 0,  1, 1],
]
TRUE_GAMMA = [-900, -700, -700, -1910]
TRUE_TAU = 40
ERROR_SCALE_NON_FORMATION = 1
ERROR_SCALE_FORMATION = 20

# MCMC config
ITER_WARMUP = 2000
ITER_SAMPLING = 2000
CHAINS = 4
PARALLEL_CHAINS = 2
MAX_TREEDEPTH=13
SHOW_PROGRESS = True


def generate_data():
    # generate observations
    G = np.array(GROUP_INCIDENCE)
    S = Matrix(STOICHIOMETRY)
    is_formation = (np.abs(S).sum(axis=0) == 1).astype(int)
    error_scale = np.where(
        is_formation, ERROR_SCALE_FORMATION, ERROR_SCALE_NON_FORMATION
    )
    true_gamma = np.array(TRUE_GAMMA)
    true_theta = np.random.normal(G @ true_gamma, TRUE_TAU)
    true_y = np.array(S.T).astype(np.float64) @ true_theta
    y_obs = np.random.normal(true_y, error_scale)

    # extra information
    n_cpd, n_rxn = S.shape
    _, n_grp = G.shape
    prior_theta = [PRIOR_LOC_THETA, [PRIOR_SCALE_THETA] * n_cpd]
    prior_gamma = [PRIOR_LOC_GAMMA, [PRIOR_SCALE_GAMMA] * n_grp]

    # Construct reparameterisation matrices
    R = generate_reparameterisation_matrix(S)
    RG = generate_reparameterisation_matrix((S.T @ G).T)
    R_inv = np.linalg.inv(R)
    RG_inv = np.linalg.inv(RG)
    return {
        "N_rxn": n_rxn,
        "N_cpd": n_cpd,
        "N_grp": n_grp,
        "S": np.array(S).astype(np.float64).tolist(),
        "G": np.array(G).astype(np.float64).tolist(),
        "true_theta": true_theta.tolist(),
        "true_gamma": true_gamma.tolist(),
        "true_y": true_y.tolist(),
        "R": R.tolist(),
        "RG": RG.tolist(),
        "R_inv": R_inv.tolist(),
        "RG_inv": RG_inv.tolist(),
        "y": y_obs.tolist(),
        "sigma": error_scale.tolist(),
        "prior_theta": prior_theta,
        "prior_gamma": prior_gamma,
        "prior_tau": PRIOR_TAU,
    }


def main():
    data = generate_data()
    jsondump(STAN_INPUT, data)
    for stan_file, sample_dir in zip(
       [STAN_FILE_NAIVE, STAN_FILE_SMART], [SAMPLE_DIR_NAIVE, SAMPLE_DIR_SMART]
    ):
        model = CmdStanModel(stan_file=stan_file)
        fit = model.sample(
            data=STAN_INPUT,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            chains=CHAINS,
            max_treedepth=MAX_TREEDEPTH,
            parallel_chains=PARALLEL_CHAINS,
            output_dir=sample_dir,
            show_progress=SHOW_PROGRESS
        )
        fit.diagnose()
        print(fit.summary())
    with open(LAST_UPDATED_FILE, "w") as f:
        f.write(str(datetime.now()))

if __name__ == "__main__":
    main()
