import arviz as az
import json
import os
import pandas as pd
from plotting import plot_leapfrog

STAN_INPUT_FILE = "results/big_case_study/stan_input.json"
SAMPLE_DIR_SMART = "results/big_case_study/samples/smart"
SAMPLE_DIR_NAIVE = "results/big_case_study/samples/naive"
PLOT_DIR = "results/big_case_study/plots"

def load_data():
    with open(STAN_INPUT_FILE, "r") as f:
        stan_input = json.load(f)
    true_theta = stan_input["theta"]
    true_gamma = stan_input["gamma"]
    csvs_smart, csvs_naive = (
        [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".csv") ]
        for d in [SAMPLE_DIR_SMART, SAMPLE_DIR_NAIVE]
    )
    infd_smart = az.from_cmdstan(csvs_smart, save_warmup=True)
    infd_naive = az.from_cmdstan(csvs_naive, save_warmup=True)
    return stan_input, infd_naive, infd_smart


def main():
    stan_input, infd_naive, infd_smart = load_data()
    leapfrog_plot = plot_leapfrog(infd_naive, infd_smart)
    leapfrog_plot.save(os.path.join(PLOT_DIR, "leapfrog_cumsum.png"))


if __name__ == "__main__":
    main()
