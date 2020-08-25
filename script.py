import arviz as az
from cmdstanpy import CmdStanModel
from cmdstanpy.utils import jsondump
import numpy as np
from sympy import Matrix
from matplotlib import pyplot as plt

RELATIVE_PATHS = {
    "naive_model": "model_naive.stan",
    "rref_model": "model_rref.stan",
}

# Specify some data
S = Matrix([
    [0, 0,  0, -1],
    [0, 0,  0, -1],
    [0, 0,  0,  2],
    [0, 0,  -1, 0],
    [0, -1, -1, 0],
    [1, 1,  2,  0]
])
error_scale = np.array([20, 1, 1, 1])
true_theta = np.array([-1000, -900, -1000, -2010, -2500, -2200])
prior_loc_theta = np.array([-1000, -1000, -900, -2000, -3000, -2000])
prior_scale_theta = np.array([200, 100, 200, 600, 300, 200])

# Find the reduced row echelon form of the transpose of the stoichiometric
# matrix
n_cpd, n_rxn = S.shape
S_T_rref, leading_one_cols = S.T.rref()
S_T_rref = np.array(S_T_rref).astype(np.float64)

# Find a matrix defining the reparameterisation
R = S_T_rref.copy()
for i in range(n_cpd):
    if i not in leading_one_cols:
        to_insert = np.zeros(n_cpd)
        to_insert[i] = 1
        R = np.insert(R, i, to_insert, 0)

# Simulate some observations
y_true = np.array(S.T).astype(np.float64) @ true_theta
y_obs = np.random.normal(y_true, error_scale)

# Define input data for the Stan models
input_data = {
    "N_rxn": n_rxn,
    "N_cpd": n_cpd,
    "S": np.array(S).astype(np.float64).tolist(),
    "pos_leading_ones": (np.array(leading_one_cols) + 1).tolist(),
    "R": R.tolist(),
    "y": y_obs.tolist(),
    "error_scale": error_scale.tolist(),
    "prior_loc_theta": prior_loc_theta.tolist(),
    "prior_scale_theta": prior_scale_theta.tolist(),
}
jsondump("input_data.json", input_data)

# Compile and fit both models
model_naive = CmdStanModel(stan_file=RELATIVE_PATHS["naive_model"])
draws_naive = model_naive.sample(data="input_data.json")
infd_naive = az.from_cmdstanpy(draws_naive)
draws_naive.diagnose()

model_rref = CmdStanModel(stan_file=RELATIVE_PATHS["rref_model"])
draws_rref = model_rref.sample(data="input_data.json")
infd_rref = az.from_cmdstanpy(draws_rref)
draws_rref.diagnose()


# Analyse results...

# compare marginal posterior distributions
f, axes = plt.subplots(1, 3, figsize=[15, 5])
for var, ax in zip(["theta", "yrep", "log_lik"], axes):
    gs = az.plot_forest(
        [infd_naive, infd_rref],
        model_names=["naive", "rref"],
        var_names=[var],
        combined=True,
        ax=ax
    )
    ax.set_title(var)
plt.tight_layout()
f.savefig("marginals.png", bbox_inches="tight")

# compare number of leapfrog steps
bins = np.linspace(0, 1100, 50)
f, ax = plt.subplots(figsize=[8, 5])
for infd, label in zip(
    [infd_naive, infd_rref],
    ["naive parameterisation", "improved parameterisation"]
):
    ax.hist(
        infd.sample_stats.n_leapfrog.to_series(),
        label=label,
        alpha=0.6,
        bins=bins
    )
ax.set(
    xlabel="Leapfrog steps",
    ylabel="Frequency",
    title="Does changing the parameterisation reduce the number of leapfrog steps?"
)
ax.legend(frameon=False)
f.savefig("leapfrog_comparison.png", bbox_inches="tight")
