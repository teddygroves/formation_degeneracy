import arviz as az
from matplotlib import pyplot as plt
from run_toy_simulation_study import (
    RESULTS_DIR,
    SAMPLE_DIR_NAIVE,
    SAMPLE_DIR_SMART,
    STAN_FILE_NAIVE,
    STAN_FILE_SMART,
    DATA_FILE,
    CHAINS
)
import json
import numpy as np
import os
import pandas as pd

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")


def plot_leapfrogs(infd_naive, infd_smart):
    bins = np.linspace(0, 4500, 50)
    f, ax = plt.subplots(figsize=[8, 5])
    for infd, label in zip(
        [infd_naive, infd_smart],
        ["naive parameterisation", "new parameterisation"]
    ):
        ax.hist(
            infd.sample_stats.n_leapfrog.to_series(),
            label=label,
            alpha=0.6,
            bins=bins,
        )
    ax.set(xlabel="Leapfrog steps", ylabel="Frequency")
    ax.legend(frameon=False)
    return f, ax


def plot_param_recovery(infd_naive, infd_smart, stan_input):
    f, axes = plt.subplots(1, 2, figsize=[15, 5])
    qs = [0.025, 0.25, 0.5, 0.75, 0.975]
    ax_titles = ["naive parameterisation", "new parameterisation"]
    true_dgfs = pd.Series(stan_input["true_theta"])
    y = true_dgfs.rank(method="dense")
    for ax, infd, title in zip(axes, [infd_naive, infd_smart], ax_titles):
        theta_qs = infd.posterior["theta"].to_series().unstack().quantile(qs).T
        true_dgf_sct = ax.scatter(true_dgfs, y, marker="|", color="black")
        wide_pct_hlines = ax.hlines(
            y,
            theta_qs[0.025],
            theta_qs[0.975],
            color="tab:orange",
            zorder=0,
            alpha=0.5
        )
        thin_pct_hlines = ax.hlines(
            y,
            theta_qs[0.25],
            theta_qs[0.75],
            color="tab:orange",
            zorder=0,
            alpha=1
        )
        ax.set_title(title)
    f.legend(
        [true_dgf_sct, wide_pct_hlines, thin_pct_hlines],
        ["True dgf", "95% posterior_interval", "50% posterior interval"],
        ncol=3,
        frameon=False
    )
    return f, axes


def get_directory_csvs(directory):
    return list(filter(lambda f: f.enswith(".csv"), directory))


def main():
    stan_input = json.load(open(DATA_FILE, "r"))
    infd_smart = az.from_cmdstan(get_directory_csvs(SAMPLE_DIR_SMART))
    infd_naive = az.from_cmdstan(get_directory_csvs(SAMPLE_DIR_NAIVE))
    f, axes = plot_leapfrogs(infd_naive, infd_smart)
    f.savefig(os.path.join(PLOT_DIR, "leapfrogs.png"), bbox_inches="tight")
    f, axes = plot_param_recovery(infd_naive, infd_smart, stan_input)
    f.savefig(os.path.join(PLOT_DIR, "recovery.png"), bbox_inches="tight")
    plt.close("all")


if __name__ == "__main__":
    main()
