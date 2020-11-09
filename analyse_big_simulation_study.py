import arviz as az
import json
import pandas as pd
import plotnine as p9

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
    infd_smart = az.from_cmdstan(csvs_smart)
    infd_naive = az.from_cmdstan(csvs_naive)
    return stan_input, infd_naive, infd_smart


def plot_leapfrog(infd_naive, infd_smart):
    leapfrogs = (
        pd.DataFrame({
            "smart": infd_smart.sample_stats.n_leapfrog.to_series(),
            "naive": infd_naive.sample_stats.n_leapfrog.to_series()
        })
        .rename_axis("Parameterisation", axis="columns")
        .unstack(level="chain")
        .cumsum()
        .stack()
        .stack()
        .rename("steps")
        .reset_index()
    )
    aes_leapfrog = p9.aes(x="draw", color="Parameterisation", y="steps")
    return (
        p9.ggplot(leapfrogs, aes_leapfrog) +
        p9.geom_line() +
        p9.labs(y="Cumulative leapfrog steps")
    )

def main():
    stan_input, infd_naive, infd_smart = load_data()
    leapfrog_plot = plot_leapfrog(infd_naive, infd_smart)
    leapfrog_plot.save(os.path.join(PLOT_DIR, "leapfrog_cumsum.png"))
