"""Run a scenario over many random seeds and summarise / plot the spread of outcomes."""

from pathlib import Path

import pandas as pd

from .params import SimulationParams
from .simulation import run_simulation


def run_many(params: SimulationParams, runs: int = 30, first_seed: int = 0):
    """Run `runs` simulations with seeds first_seed..first_seed+runs-1.

    Returns (daily, summaries):
      daily     - long DataFrame with one row per (seed, day)
      summaries - one row per seed with that run's summary statistics
    """
    daily, summaries = [], []
    for seed in range(first_seed, first_seed + runs):
        result = run_simulation(params, seed=seed)
        daily.append(result.log.reset_index().assign(seed=seed))
        summaries.append({"seed": seed, **result.summary})
    return pd.concat(daily, ignore_index=True), pd.DataFrame(summaries).set_index("seed")


def describe(summaries: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and 5th-95th percentile of each summary statistic across runs."""
    cols = ["duration_days", "peak_infected", "peak_day", "total_infected", "attack_rate",
            "ever_aware", "ever_quarantined"]
    s = summaries[cols]
    return pd.DataFrame({"mean": s.mean(), "std": s.std(), "p5": s.quantile(0.05), "p95": s.quantile(0.95)})


def plot_runs(daily: pd.DataFrame, title: str, path: str | Path) -> Path:
    """Plot the median daily curve with a 5-95% band across runs for each compartment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Runs end on different days; after a run ends its counts stay constant, so forward-fill.
    max_day = int(daily["Day"].max())
    columns = ["Susceptible", "Infected", "Recovered", "Aware"]
    colours = {"Susceptible": "#3b6fb6", "Infected": "#c8423b", "Recovered": "#4a9b5f", "Aware": "#d39b2a"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for col in columns:
        wide = daily.pivot(index="Day", columns="seed", values=col).reindex(range(1, max_day + 1)).ffill()
        ax.plot(wide.index, wide.median(axis=1), color=colours[col], label=col, linewidth=2)
        ax.fill_between(wide.index, wide.quantile(0.05, axis=1), wide.quantile(0.95, axis=1),
                        color=colours[col], alpha=0.18, linewidth=0)
    runs = daily["seed"].nunique()
    ax.set(title=f"{title} — median and 5–95% range over {runs} runs", xlabel="Day", ylabel="People")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
