import plotnine as p9
import pandas as pd

def get_leapfrogs(infd):
    warmup_leapfrogs = infd.warmup_sample_stats.n_leapfrog.to_series()
    sample_leapfrogs = infd.sample_stats.n_leapfrog.to_series()
    n_warmup = infd.sample_stats.num_warmup[0]
    sample_leapfrogs.index = sample_leapfrogs.index.set_levels(
        sample_leapfrogs.index.levels[1] + n_warmup, level=1
    )
    return pd.concat([warmup_leapfrogs, sample_leapfrogs])




def plot_leapfrog(infd_naive, infd_smart):
    leapfrogs = (
        pd.DataFrame({
            "new": get_leapfrogs(infd_smart),
            "naive": get_leapfrogs(infd_naive)
        })
        .rename_axis("Parameterisation", axis="columns")
        .unstack(level="chain")
        .cumsum()
        .stack()
        .stack()
        .rename("steps")
        .reset_index()
        .assign(group=lambda df: df["Parameterisation"].str.cat(df["chain"].astype(str)))
    )
    aes_leapfrog = p9.aes(x="draw", y="steps", color="Parameterisation", group="group")
    return (
        p9.ggplot(leapfrogs, aes_leapfrog) +
        p9.theme_minimal() +
        p9.geom_line() +
        p9.labs(y="Cumulative leapfrog steps")
    )

