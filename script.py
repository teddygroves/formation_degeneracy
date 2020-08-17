import arviz as az
from cmdstanpy import CmdStanModel
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
prior_scale_theta = np.array([200, 400, 200, 600, 300, 200])

# Find the reduced row echelon form of the transpose of the stoichiometric
# matrix
S_T_rref = np.array(S.T.rref()[0]).astype(np.float64)

# Simulate some observations
y_true = np.array(S.T).astype(np.float64) @ true_theta
y_obs = np.random.normal(y_true, error_scale)

# Define input data for the Stan models
input_data = {
    "N_rxn": S.shape[1],
    "N_cpd": S.shape[0],
    "S": np.array(S).astype(np.float64).tolist(),
    "S_T_rref": np.array(S_T_rref).astype(np.float64).tolist(),
    "y": y_obs.tolist(),
    "error_scale": error_scale.tolist(),
    "prior_loc_theta": prior_loc_theta.tolist(),
    "prior_scale_theta": prior_scale_theta.tolist(),
}

# Compile and fit both models
model_naive, model_rref = (
    CmdStanModel(stan_file=RELATIVE_PATHS[k])
    for k in ["naive_model", "rref_model"]
)
draws_naive, draws_rref = (
    model.sample(data=input_data) for model in [model_naive, model_rref]
)
infd_naive, infd_rref = (
    az.from_cmdstanpy(draws) for draws in [draws_naive, draws_rref]
)
print(draws_rref.summary())
print(draws_naive.summary())

# Analyse results
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
f.savefig("fig.png", bbox_inches="tight")

