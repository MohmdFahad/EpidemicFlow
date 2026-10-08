"""Main simulation loop. Returns results instead of only printing them."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .awareness import awareness_cycle_length, global_awareness, spread_awareness
from .grid_utils import (
    AWARE_STATES,
    INFECTED_AWARE,
    INFECTED_QUARANTINED,
    INFECTED_STATES,
    RECOVERED,
    SPREADER_STATES,
    SUSCEPTIBLE_AWARE,
    SUSCEPTIBLE_QUARANTINED,
    SUSCEPTIBLE_STATES,
    Population,
    count,
    get_neighbours,
    get_pos,
    get_score,
    make_grid,
)
from .infection import infect, recover, start_quarantine
from .params import SimulationParams


@dataclass
class SimulationResult:
    params: SimulationParams
    seed: int | None
    log: pd.DataFrame  # one row per simulated day
    summary: dict = field(default_factory=dict)
    final_grid: np.ndarray | None = None

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.log.to_csv(path)
        return path


def advance_quarantine(pop: Population) -> None:
    """Count every quarantine down by one day (once per day) and release finished ones as aware."""
    grid, days_left = pop.grid, pop.quarantine_duration_grid
    in_quarantine = np.isin(grid, (SUSCEPTIBLE_QUARANTINED, INFECTED_QUARANTINED))
    finished = in_quarantine & (days_left <= 0)

    grid[finished & (grid == SUSCEPTIBLE_QUARANTINED)] = SUSCEPTIBLE_AWARE
    grid[finished & (grid == INFECTED_QUARANTINED)] = INFECTED_AWARE
    pop.was_ever_aware[finished] = True
    days_left[in_quarantine & ~finished] -= 1


def quarantine_surrounded_infected(pop: Population, params: SimulationParams,
                                   rng: np.random.Generator) -> None:
    """Infected people with at least half their neighbours infected may quarantine."""
    for pos in get_pos(pop.grid, *SPREADER_STATES):
        share = get_score(pop.grid, get_neighbours(pos, pop.size), *SPREADER_STATES)
        if share >= 0.5 and rng.random() < params.quarantine_chance:
            p = (pos[0], pos[1])
            pop.grid[p] = INFECTED_QUARANTINED
            start_quarantine(pop, p, rng)


def awareness_step(pop: Population, params: SimulationParams, day: int,
                   rng: np.random.Generator) -> None:
    n_total = pop.grid.size
    infection_percent = count(pop.grid, *INFECTED_STATES) / n_total
    awareness_percent = count(pop.grid, *AWARE_STATES) / n_total
    if day % awareness_cycle_length(infection_percent, awareness_percent, params) != 0:
        return

    global_awareness(pop, rng)  # once per event, not once per neighbour
    for pos in get_pos(pop.grid, *AWARE_STATES):
        for neighbour in get_neighbours(pos, pop.size):
            spread_awareness(pop, neighbour, params.awareness_rate, rng)


def run_simulation(params: SimulationParams, seed: int | None = None,
                   max_days: int = 5000, verbose: bool = False) -> SimulationResult:
    """Run one simulation until nobody is infected. Same params + same seed = same result."""
    rng = np.random.default_rng(seed)
    pop = make_grid(params.grid_size, params.init_infections, rng)
    n_total = pop.grid.size

    rows = []
    recovery_durations = []
    day = 0
    while count(pop.grid, *INFECTED_STATES) > 0:  # includes aware and quarantined infected
        day += 1
        if day > max_days:
            raise RuntimeError(f"Simulation did not finish within {max_days} days")

        if params.behavioural:
            advance_quarantine(pop)
            quarantine_surrounded_infected(pop, params, rng)
            awareness_step(pop, params, day, rng)
        new_infections = infect(pop, params, day, rng)
        recovery_durations.append(recover(pop, params, day, rng))

        g = pop.grid
        rows.append({
            "Day": day,
            "Susceptible": count(g, *SUSCEPTIBLE_STATES),
            "Infected": count(g, *INFECTED_STATES),
            "Recovered": count(g, RECOVERED),
            "Aware": count(g, *AWARE_STATES),
            "Quarantined": count(g, SUSCEPTIBLE_QUARANTINED, INFECTED_QUARANTINED),
            "New_Infections": new_infections,
        })
        if verbose:
            print(f"Day {day}\n{g}\n")

    log = pd.DataFrame(rows).set_index("Day")
    durations = np.concatenate(recovery_durations) if recovery_durations else np.array([])
    ever_infected = int((pop.infection_day_grid != -1).sum())

    summary = {
        "duration_days": len(log),
        "peak_infected": int(log["Infected"].max()),
        "peak_day": int(log["Infected"].idxmax()),
        "total_infected": ever_infected,
        "attack_rate": ever_infected / n_total,
        "never_infected": n_total - ever_infected,
        "ever_aware": int(pop.was_ever_aware.sum()),
        "ever_quarantined": int(pop.was_ever_quarantined.sum()),
        "recovery_mean": float(durations.mean()),
        "recovery_std": float(durations.std()),
        "recovery_min": int(durations.min()),
        "recovery_max": int(durations.max()),
    }
    return SimulationResult(params=params, seed=seed, log=log, summary=summary, final_grid=pop.grid.copy())


def format_summary(result: SimulationResult) -> str:
    s = result.summary
    return "\n".join([
        f"Epidemic duration:      {s['duration_days']} days",
        f"Peak active infections: {s['peak_infected']} (day {s['peak_day']})",
        f"Total infected:         {s['total_infected']} ({s['attack_rate']:.1%} of population)",
        f"Never infected:         {s['never_infected']}",
        f"Became aware:           {s['ever_aware']}",
        f"Quarantined:            {s['ever_quarantined']}",
        f"Recovery time:          mean {s['recovery_mean']:.2f}, sd {s['recovery_std']:.2f}, "
        f"range {s['recovery_min']}-{s['recovery_max']} days",
    ])
