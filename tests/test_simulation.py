import dataclasses

import numpy as np
import pandas as pd
import pytest

from epidemicflow import SCENARIOS, SimulationParams, run_simulation
from epidemicflow.awareness import spread_awareness
from epidemicflow.grid_utils import (
    INFECTED_QUARANTINED,
    SUSCEPTIBLE,
    SUSCEPTIBLE_AWARE,
    SUSCEPTIBLE_QUARANTINED,
    make_grid,
)
from epidemicflow.simulation import advance_quarantine

SMALL = dataclasses.replace(SCENARIOS["influenza"], grid_size=15, init_infections=3)


def test_same_seed_gives_identical_results():
    a, b = run_simulation(SMALL, seed=42), run_simulation(SMALL, seed=42)
    pd.testing.assert_frame_equal(a.log, b.log)
    assert a.summary == b.summary


def test_different_seeds_give_different_results():
    logs = [run_simulation(SMALL, seed=s).log for s in range(5)]
    assert any(not logs[0].equals(other) for other in logs[1:])


@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("seed", range(3))
def test_simulation_ends_with_no_infections(name, seed):
    params = dataclasses.replace(SCENARIOS[name], grid_size=15, init_infections=3)
    result = run_simulation(params, seed=seed)
    assert result.log["Infected"].iloc[-1] == 0


@pytest.mark.parametrize("seed", range(3))
def test_population_is_conserved_every_day(seed):
    log = run_simulation(SMALL, seed=seed).log
    totals = log["Susceptible"] + log["Infected"] + log["Recovered"]
    assert (totals == SMALL.grid_size**2).all()


@pytest.mark.parametrize("seed", range(3))
def test_summary_is_consistent(seed):
    result = run_simulation(SMALL, seed=seed)
    s, n = result.summary, SMALL.grid_size**2
    assert s["total_infected"] + s["never_infected"] == n
    assert s["total_infected"] == result.log["Recovered"].iloc[-1]  # everyone infected has recovered
    assert s["duration_days"] == len(result.log)


def test_no_behaviour_model_means_no_awareness_or_quarantine():
    params = dataclasses.replace(SMALL, awareness_rate=0.0)
    result = run_simulation(params, seed=0)
    assert result.summary["ever_aware"] == 0
    assert result.summary["ever_quarantined"] == 0


def test_susceptible_can_become_aware_from_neighbours():
    pop = make_grid(3, 1, np.random.default_rng(0))
    pop.grid[:] = SUSCEPTIBLE_AWARE
    pop.grid[1, 1] = SUSCEPTIBLE  # one unaware person surrounded by aware neighbours
    spread_awareness(pop, np.array([1, 1]), awareness_rate=1.0, rng=np.random.default_rng(0))
    assert pop.grid[1, 1] == SUSCEPTIBLE_AWARE


def test_quarantine_counts_down_once_per_day():
    pop = make_grid(3, 1, np.random.default_rng(0))
    pop.grid[:] = SUSCEPTIBLE
    pop.grid[0, 0], pop.grid[2, 2] = SUSCEPTIBLE_QUARANTINED, INFECTED_QUARANTINED
    pop.quarantine_duration_grid[0, 0] = pop.quarantine_duration_grid[2, 2] = 3
    advance_quarantine(pop)
    assert pop.quarantine_duration_grid[0, 0] == 2
    assert pop.quarantine_duration_grid[2, 2] == 2


@pytest.mark.parametrize("field, value", [
    ("infection_prob", 1.5), ("awareness_rate", -0.1), ("recovery_var", 0), ("init_infections", 0),
])
def test_invalid_params_are_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(SMALL, **{field: value})


def test_params_validate_on_creation():
    with pytest.raises(ValueError):
        SimulationParams(2.0, 8, 4, 0.5, 0.5, 0.5)
