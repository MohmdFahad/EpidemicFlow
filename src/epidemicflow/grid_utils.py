"""Grid set-up and neighbourhood helpers."""

from dataclasses import dataclass

import numpy as np

# Individual states stored in the grid
SUSCEPTIBLE = 0
INFECTED = 1
RECOVERED = 2
SUSCEPTIBLE_QUARANTINED = 3
SUSCEPTIBLE_AWARE = 4
INFECTED_AWARE = 5
INFECTED_QUARANTINED = 6

INFECTED_STATES = (INFECTED, INFECTED_AWARE, INFECTED_QUARANTINED)
SPREADER_STATES = (INFECTED, INFECTED_AWARE)  # quarantined people do not spread
SUSCEPTIBLE_STATES = (SUSCEPTIBLE, SUSCEPTIBLE_QUARANTINED, SUSCEPTIBLE_AWARE)
AWARE_STATES = (SUSCEPTIBLE_AWARE, INFECTED_AWARE)

NEIGHBOUR_OFFSETS = np.array(
    [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
)


@dataclass
class Population:
    """All per-individual state for one simulation, kept together."""

    grid: np.ndarray  # current state of each individual (see constants above)
    recovery_grid: np.ndarray  # day each infected individual recovers (-1 = not set)
    was_ever_aware: np.ndarray  # True if the individual was ever aware
    was_ever_quarantined: np.ndarray  # True if the individual ever quarantined
    quarantine_duration_grid: np.ndarray  # days of quarantine left
    infection_day_grid: np.ndarray  # day the individual was infected (-1 = never)

    @property
    def size(self) -> int:
        return self.grid.shape[0]


def make_grid(grid_size: int, init_infections: int, rng: np.random.Generator) -> Population:
    """Create a grid_size x grid_size population with init_infections random infected individuals."""
    n_total = grid_size * grid_size
    if not 0 < init_infections <= n_total:
        raise ValueError(f"init_infections must be between 1 and {n_total}")

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    infected_idx = rng.choice(n_total, size=init_infections, replace=False)
    grid.flat[infected_idx] = INFECTED

    infection_day_grid = np.full(grid.shape, -1, dtype=np.int64)
    infection_day_grid[grid == INFECTED] = 0  # initial cases were infected on day 0

    return Population(
        grid=grid,
        recovery_grid=np.full(grid.shape, -1, dtype=np.int64),
        was_ever_aware=np.zeros(grid.shape, dtype=bool),
        was_ever_quarantined=np.zeros(grid.shape, dtype=bool),
        quarantine_duration_grid=np.zeros(grid.shape, dtype=np.int64),
        infection_day_grid=infection_day_grid,
    )


def get_neighbours(pos: np.ndarray, gridsize: int) -> np.ndarray:
    """Return the (row, col) positions of the up-to-8 neighbours of pos that lie inside the grid."""
    neighbours = np.asarray(pos) + NEIGHBOUR_OFFSETS
    inside = np.all((neighbours >= 0) & (neighbours < gridsize), axis=1)
    return neighbours[inside]


def get_pos(grid: np.ndarray, *labels: int) -> np.ndarray:
    """Return an (n, 2) array of positions whose state is any of labels."""
    return np.argwhere(np.isin(grid, labels))


def get_score(grid: np.ndarray, neighbours: np.ndarray, *labels: int) -> float:
    """Return the fraction (0 to 1) of the given neighbours whose state is any of labels."""
    if len(neighbours) == 0:
        return 0.0
    tags = grid[neighbours[:, 0], neighbours[:, 1]]
    return float(np.isin(tags, labels).mean())


def count(grid: np.ndarray, *labels: int) -> int:
    """Number of individuals in any of the given states."""
    return int(np.isin(grid, labels).sum())
