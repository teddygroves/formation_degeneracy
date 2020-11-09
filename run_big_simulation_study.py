from cmdstanpy import CmdStanModel
from datetime import datetime
from functools import reduce
import json
import numpy as np
import os
import pandas as pd
from quilt.data.equilibrator import component_contribution as pkg
from utils import generate_reparameterisation_matrix

RESULTS_DIR = "results/big_case_study"
SAMPLE_DIR_NAIVE = reduce(os.path.join, [RESULTS_DIR, "samples", "naive"])
SAMPLE_DIR_SMART = reduce(os.path.join, [RESULTS_DIR, "samples", "smart"])
LAST_UPDATED_FILE = reduce(
    os.path.join, [RESULTS_DIR, "samples", "last_updated.txt"]
)
STAN_INPUT_FILE = os.path.join(RESULTS_DIR, "stan_input.json")
R_MATRIX_FILE = os.path.join(RESULTS_DIR, "equilibrator_reparam.json")
STAN_PROGRAM_SMART = "model_smart.stan"
STAN_PROGRAM_NAIVE = "model_naive.stan"

PRIOR_BIAS_GAMMA = 0
PRIOR_BIAS_THETA = 0

# Priors
PRIOR_BIAS_TAU = 0
PRIOR_SCALE_GAMMA = 300
PRIOR_SCALE_THETA = 500
PRIOR_SCALE_TAU = 20

# MCMC config
ITER_WARMUP = 1000
ITER_SAMPLING = 1000
CHAINS = 4
PARALLEL_CHAINS = 4
SHOW_PROGRESS = True

TRUE_TAU = 50
TRUE_SIGMA = {"formation": 200, "non_formation": 20}

RANDOM_SEED = 12345

def simulate_data(tau, sigma):
    """Simulate some measurements based on data from equilibrator.

    Specifically, start with the stoichiometric matrix, group incidence matrix
    and estimated group formation energies from equilibrator. Next, assuming
    that the equilibrator estimates are exactly correct and that group
    additivity is approximately true with standard deviation tau, generate true
    compound formation energies and use these to generate reaction delta
    Gs. Finally, assuming measurement error sigma["formation"] for formation
    reactions and sigma["non_formation"] for others, generate simulated
    measurements from the reaction delta Gs.

    """

    S_raw = pkg.parameters.train_S()
    G_raw = pkg.parameters.train_G()
    rxn_ix, rxn_ix_orig = pd.factorize(
        S_raw.apply(lambda col: "-".join(col.map(str)))
    )
    S = S_raw.copy()
    S.columns = rxn_ix
    S = (
        S.loc[lambda df: ~df.duplicated()]
        .T.loc[lambda df: ~df.duplicated()]
        .T
        # .iloc[:100]
        .copy()
    )
    G = (
        G_raw.loc[S.index]
        .T.reset_index(drop=True)
        .loc[lambda df: ~df.duplicated()]
        .T.copy()
    )
    gamma = pkg.parameters.dG0_gc()[G.columns]
    S = S.values
    G = G.values
    is_formation = np.abs(S).sum(axis=0) == 1
    theta = np.random.normal(G @ gamma, tau)
    dgr = S.T @ theta
    sigma = np.where(is_formation, sigma["formation"], sigma["non_formation"])
    y = np.random.normal(dgr, sigma)
    return {
        "N_rxn": S.shape[1],
        "N_cpd": S.shape[0],
        "N_grp": G.shape[1],
        "y": y.tolist(),
        "theta": theta.tolist(),
        "gamma": gamma.tolist(),
        "dgr": dgr.tolist(),
        "S": S.tolist(),
        "G": G.tolist(),
        "is_formation": is_formation.astype(np.int64).tolist(),
        "tau": tau,
        "sigma": sigma.tolist(),
    }


def proportionise(arr: np.array, axis):
    sums = np.abs(arr).sum(axis=axis)
    if axis == 0:
        return arr / sums
    elif axis == 1:
        return arr / sums[:, None]
    else:
        raise ValueError("axis must be 0 or 1")


def main():
    np.random.seed(seed=RANDOM_SEED)
    sim = simulate_data(TRUE_TAU, TRUE_SIGMA)
    if not os.path.exists(R_MATRIX_FILE):
        S = np.array(sim["S"])
        G = np.array(sim["G"])
        R = generate_reparameterisation_matrix(S)
        RG = generate_reparameterisation_matrix((S.T @ G).T)
        with open(R_MATRIX_FILE, "w") as f:
            json.dump({"R": R.tolist(), "RG": RG.tolist()}, f)
    else:
        with open(R_MATRIX_FILE, "r") as f:
            rs = json.load(f)
            R = np.array(rs["R"])
            RG = np.array(rs["RG"])
    R_inv = np.linalg.inv(R).tolist()
    RG_inv = np.linalg.inv(RG).tolist()
    prior_mean_gamma = np.random.normal(sim["gamma"], PRIOR_BIAS_GAMMA).tolist()
    prior_mean_theta = np.random.normal(sim["theta"], PRIOR_BIAS_THETA).tolist()
    prior_mean_tau = np.random.normal(sim["tau"], PRIOR_BIAS_TAU)
    prior_scale_gamma = [PRIOR_SCALE_GAMMA] * len(prior_mean_gamma)
    prior_scale_theta = [PRIOR_SCALE_THETA] * len(prior_mean_theta)
    stan_input = {
        **sim,
        **{
            "R": R.tolist(),
            "RG": RG.tolist(),
            "R_inv": R_inv,
            "RG_inv": RG_inv,
            "prior_theta": [prior_mean_theta, prior_scale_theta],
            "prior_gamma": [prior_mean_gamma, prior_scale_gamma],
            "prior_tau": [prior_mean_tau, PRIOR_SCALE_TAU]
        }
    }
    with open(STAN_INPUT_FILE, "w") as outfile:
        json.dump(stan_input, outfile)
    model_smart = CmdStanModel(stan_file=STAN_PROGRAM_SMART)
    model_naive = CmdStanModel(stan_file=STAN_PROGRAM_NAIVE)
    for model, sample_dir in zip(
        [model_smart, model_naive], [SAMPLE_DIR_SMART, SAMPLE_DIR_NAIVE]
    ):
        fit = model.sample(
            data=STAN_INPUT_FILE,
            output_dir=sample_dir,
            iter_warmup=ITER_WARMUP,
            iter_sampling=ITER_SAMPLING,
            step_size=0.005,
            chains=CHAINS,
            max_treedepth=13,
            parallel_chains=PARALLEL_CHAINS,
            save_warmup=True,
            show_progress=SHOW_PROGRESS,
            adapt_init_phase=0,
            inits={
                "tau": stan_input["prior_tau"][0],
                "eta_cpd": R @ np.array(sim["theta"]),
                "eta_grp": RG @ np.array(sim["gamma"]),
                "theta": sim["theta"],
                "gamma": sim["gamma"],
            }
        )
        fit.diagnose()
        summary = fit.summary()
        printvars = filter(
            lambda i: any([j in i for j in ["lp__", "tau", "theta", "gamma"]]),
            summary.index
        )
        print(summary.loc[printvars])
    with open(LAST_UPDATED_FILE, "w") as f:
        f.write(str(datetime.now()))
    

    
if __name__ == "__main__":
    main()
