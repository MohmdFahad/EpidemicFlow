"""Infection spread and recovery."""

import numpy as np

from .grid_utils import (
    INFECTED,
    INFECTED_AWARE,
    INFECTED_QUARANTINED,
    RECOVERED,
    SPREADER_STATES,
    SUSCEPTIBLE,
    SUSCEPTIBLE_AWARE,
    SUSCEPTIBLE_QUARANTINED,
    INFECTED_STATES,
    Population,
    count,
    get_neighbours,
    get_pos,
    get_score,
)
from .params import SimulationParams

AWARE_SPREADER_FACTOR = 0.3  # an aware infected person transmits at 30% of the normal rate
QUARANTINE_PROTECTION = 0.9  # quarantine removes 90% of infection risk
QUARANTINE_DAYS = (7, 14)  # quarantine lasts between 7 and 14 days (inclusive)


def generate_infection_day_cycle_parameters(infection_prob: float, awareness_rate: float):
    """Hand-tuned parameters for how often (in days) infection spread events happen."""
    if infection_prob >= 0.8 and awareness_rate <= 0.3:
        n_min, n_max, p0, a = 2, 5, 0.65, 3.0
    elif infection_prob >= 0.7 and awareness_rate <= 0.5:
        n_min, n_max, p0, a = 1, 6, 0.4, 5.0
    elif infection_prob <= 0.4 and awareness_rate >= 0.6:
        n_min, n_max, p0, a = 4, 8, 0.3, 9.0
    elif infection_prob >= 0.7 and awareness_rate >= 0.6:
        n_min, n_max, p0, a = 3, 7, 0.5, 10.5
    else:
        n_min, n_max, p0, a = 1, 5, 0.7, 2.0
    return n_min, n_max, p0, a


def infection_cycle_length(infection_percent: float, params: SimulationParams) -> int:
    """Days between infection spread events: n = n_min + (n_max - n_min) / (1 + exp(a (I - p0)))."""
    n_min, n_max, p0, a = generate_infection_day_cycle_parameters(
        params.infection_prob, params.awareness_rate
    )
    n_float = n_min + (n_max - n_min) / (1.0 + np.exp(a * (infection_percent - p0)))
    return max(1, int(round(n_float)))


def start_quarantine(pop: Population, pos, rng: np.random.Generator) -> None:
    pop.quarantine_duration_grid[pos] = rng.integers(QUARANTINE_DAYS[0], QUARANTINE_DAYS[1] + 1)
    pop.was_ever_quarantined[pos] = True


def awareness_probability(pop: Population, y: np.ndarray, infection_percent: float,
                          awareness_rate: float) -> float:
    """Chance that an exposed, unaware individual adopts protective behaviour instead of being infected."""
    neighbours = get_neighbours(y, pop.size)
    aware_share = get_score(pop.grid, neighbours, SUSCEPTIBLE_AWARE, INFECTED_AWARE)
    infected_share = get_score(pop.grid, neighbours, *SPREADER_STATES)
    x = np.clip(0.6 * aware_share + 0.4 * infected_share, 0.0, 1.0)
    k = 6  # steepness
    b = 0.45 - 0.15 * infection_percent  # threshold
    sigmoid = 1 / (1 + np.exp(-k * (x - b)))
    baseline_spont = 0.01  # baseline spontaneous adoption
    raw = baseline_spont + (1 - baseline_spont) * sigmoid
    return float(np.clip(awareness_rate * raw, 0.0, 1.0))


def _infect(pop: Population, pos, day: int, quarantined: bool = False) -> None:
    current = pop.grid[pos]
    if quarantined:
        pop.grid[pos] = INFECTED_QUARANTINED
    elif current == SUSCEPTIBLE_AWARE:
        pop.grid[pos] = INFECTED_AWARE
    else:
        pop.grid[pos] = INFECTED
    pop.infection_day_grid[pos] = day


def infect(pop: Population, params: SimulationParams, day: int, rng: np.random.Generator) -> int:
    """Spread infection from every spreader to its susceptible neighbours. Returns the number of new infections."""
    grid = pop.grid
    n_total = grid.size

    infection_percent = count(grid, *INFECTED_STATES) / n_total
    if day % infection_cycle_length(infection_percent, params) != 0:
        return 0  # no spread event today

    new_infections = 0
    for x in get_pos(grid, *SPREADER_STATES):
        spreader_aware = grid[x[0], x[1]] == INFECTED_AWARE
        neighbours = get_neighbours(x, pop.size)
        states = grid[neighbours[:, 0], neighbours[:, 1]]  # read the grid *now*, so nobody is infected twice

        prob = params.infection_prob * (AWARE_SPREADER_FACTOR if spreader_aware else 1.0)
        prob_aware = prob * (1 - params.awareness_efficacy)

        exposed = []
        for nb, state in zip(neighbours, states):
            if state in (SUSCEPTIBLE, SUSCEPTIBLE_QUARANTINED):
                p = prob
            elif state == SUSCEPTIBLE_AWARE:
                p = prob_aware
            else:
                continue  # already infected or recovered
            if rng.random() < p:
                exposed.append(nb)

        for y in exposed:
            pos = (y[0], y[1])
            state = grid[pos]

            if not params.behavioural:
                _infect(pop, pos, day)
                new_infections += 1
                continue

            if state == SUSCEPTIBLE_QUARANTINED:
                # quarantine protects against most exposures
                if rng.random() < 1 - QUARANTINE_PROTECTION:
                    _infect(pop, pos, day, quarantined=True)
                    new_infections += 1
                continue

            infected_share = get_score(grid, get_neighbours(y, pop.size), *SPREADER_STATES)
            if infected_share >= 0.5:
                # surrounded by infections: may quarantine instead of being infected
                if rng.random() < params.quarantine_chance:
                    grid[pos] = SUSCEPTIBLE_QUARANTINED
                    start_quarantine(pop, pos, rng)
                else:
                    _infect(pop, pos, day)
                    new_infections += 1
            elif state == SUSCEPTIBLE_AWARE:
                _infect(pop, pos, day)
                new_infections += 1
            else:
                infection_percent = count(grid, *SPREADER_STATES) / n_total
                if rng.random() < awareness_probability(pop, y, infection_percent, params.awareness_rate):
                    grid[pos] = SUSCEPTIBLE_AWARE  # adopts protective behaviour in time
                    pop.was_ever_aware[pos] = True
                else:
                    _infect(pop, pos, day)
                    new_infections += 1

    return new_infections


def recover(pop: Population, params: SimulationParams, day: int, rng: np.random.Generator) -> np.ndarray:
    """Assign Gamma-distributed recovery days to newly infected individuals and recover those due today.

    Returns the recovery durations assigned today.
    """
    k = params.recovery_mean**2 / params.recovery_var  # shape
    theta = params.recovery_var / params.recovery_mean  # scale

    needs_day = np.isin(pop.grid, INFECTED_STATES) & (pop.recovery_grid == -1)
    durations = np.maximum(1, np.rint(rng.gamma(k, theta, size=int(needs_day.sum())))).astype(np.int64)
    pop.recovery_grid[needs_day] = day + durations

    recovering = pop.recovery_grid == day
    pop.grid[recovering] = RECOVERED
    pop.recovery_grid[recovering] = -1
    pop.quarantine_duration_grid[recovering] = 0
    return durations
