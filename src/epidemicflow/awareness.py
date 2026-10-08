"""Awareness (protective behaviour) spread, both local (neighbours) and global (campaigns)."""

import numpy as np

from .grid_utils import (
    AWARE_STATES,
    INFECTED,
    INFECTED_AWARE,
    SPREADER_STATES,
    SUSCEPTIBLE,
    SUSCEPTIBLE_AWARE,
    Population,
    count,
    get_neighbours,
    get_score,
)
from .params import SimulationParams


def generate_awareness_day_cycle_parameters(infection_prob: float, awareness_rate: float):
    """Hand-tuned parameters for how often (in days) awareness spread events happen."""
    if infection_prob >= 0.8 and awareness_rate <= 0.3:
        alpha, beta, min_cycle, max_cycle = 4, 5, 3, 6
    elif infection_prob >= 0.7 and awareness_rate <= 0.5:
        alpha, beta, min_cycle, max_cycle = 3, 8, 2, 7
    elif infection_prob <= 0.4 and awareness_rate >= 0.6:
        alpha, beta, min_cycle, max_cycle = 3, 7, 4, 8
    elif infection_prob >= 0.7 and awareness_rate >= 0.6:
        alpha, beta, min_cycle, max_cycle = 4.5, 6, 3, 7
    else:
        alpha, beta, min_cycle, max_cycle = 3, 1, 1, 4
    return alpha, beta, min_cycle, max_cycle


def awareness_cycle_length(infection_percent: float, awareness_percent: float,
                           params: SimulationParams) -> int:
    """Days between awareness spread events: n = n_max * exp(-alpha I + beta A), clipped to [n_min, n_max]."""
    alpha, beta, min_cycle, max_cycle = generate_awareness_day_cycle_parameters(
        params.infection_prob, params.awareness_rate
    )
    n_float = max_cycle * np.exp(-alpha * infection_percent + beta * awareness_percent)
    return int(np.clip(np.round(n_float), min_cycle, max_cycle))


def global_awareness(pop: Population, rng: np.random.Generator) -> None:
    """Population-wide awareness campaigns. Runs ONCE per awareness event.

    Each unaware susceptible person becomes aware with a small probability that rises
    with infection levels and falls as awareness saturates.
    """
    grid = pop.grid
    n_total = grid.size
    infection_percent = count(grid, *SPREADER_STATES) / n_total
    awareness_percent = count(grid, *AWARE_STATES) / n_total

    base_rate, max_rate = 0.002, 0.01
    infection_term = 1 / (1 + np.exp(-6.0 * (infection_percent - 0.5)))
    awareness_term = 1 - 1 / (1 + np.exp(-13.0 * (awareness_percent - 0.1)))
    daily_prob = base_rate + (max_rate - base_rate) * infection_term * awareness_term

    newly_aware = (grid == SUSCEPTIBLE) & (rng.random(grid.shape) < daily_prob)
    grid[newly_aware] = SUSCEPTIBLE_AWARE
    pop.was_ever_aware[newly_aware] = True


def spread_awareness(pop: Population, pos: np.ndarray, awareness_rate: float,
                     rng: np.random.Generator) -> None:
    """Local awareness: an unaware individual at pos may become aware from aware neighbours."""
    grid = pop.grid
    r, c = pos[0], pos[1]
    if grid[r, c] not in (SUSCEPTIBLE, INFECTED):  # was `== 0 | == 1`, which never matched 0
        return

    infection_percent = count(grid, *SPREADER_STATES) / grid.size
    score = get_score(grid, get_neighbours(pos, pop.size), *AWARE_STATES)

    k = 1 + (33 - 1) * score**2  # steepness grows with aware neighbours
    b = 0.74 + (0.24 - 0.74) * np.exp(-score)  # threshold
    x = score + 1.5 * infection_percent
    sigmoid = 1 / (1 + np.exp(-k * (x - b)))

    if rng.random() < awareness_rate * sigmoid:
        grid[r, c] = INFECTED_AWARE if grid[r, c] == INFECTED else SUSCEPTIBLE_AWARE
        pop.was_ever_aware[r, c] = True
