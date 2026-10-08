import numpy as np
from dataclasses import dataclass

SUSCEPTIBLE, INFECTED, RECOVERED = 0, 1, 2
SUSCEPTIBLE_QUARANTINED, SUSCEPTIBLE_AWARE = 3, 4
INFECTED_AWARE, INFECTED_QUARANTINED = 5, 6

INFECTED_STATES    = (INFECTED, INFECTED_AWARE, INFECTED_QUARANTINED)
SPREADER_STATES    = (INFECTED, INFECTED_AWARE)          # quarantined people don't spread
SUSCEPTIBLE_STATES = (SUSCEPTIBLE, SUSCEPTIBLE_QUARANTINED, SUSCEPTIBLE_AWARE)
AWARE_STATES       = (SUSCEPTIBLE_AWARE, INFECTED_AWARE)

@dataclass
class Population:
    grid: np.ndarray
    recovery_grid: np.ndarray
    was_ever_aware: np.ndarray
    was_ever_quarantined: np.ndarray
    quarantine_duration_grid: np.ndarray
    infection_day_grid: np.ndarray

# Generates the grids required for the simulation
# Input -
#   grid_size - grid size
#   init_infections - the initial amount of infections the grid should consist of
# Output -
#   grid - the main grid used for labeling the state of each individual
#   recovery_grid - a grid to track when each individual will recover
#   was_ever_aware - a grid to track every individual of whether they were ever aware
#   was_ever_quarantined - a grid to track every individual of whether they ever quarantined
#   quarantine_duration_grid - a grid to track how many days are left for each quarantined individual
#   infection_day_grid - a grid to track the day each individual got infected
def make_grid(
        grid_size: int, init_infections: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Used for plotting infections on the main grid
    for i in range(init_infections):
        mask = grid == 0  # identify susceptible individuals
        view = grid[mask]
        new = np.zeros(view.shape)
        randindex = rng.integers(0, new.size)  # chooses a random individual
        new[randindex] = 1  # update the individual to infected
        grid[mask] = new  # apply the change

    recovery_grid = np.zeros_like(grid)
    recovery_grid[...] = (
        -1
    )  # -1 used to indicate recovered / never infected individuals
    recovery_grid = recovery_grid.astype(np.int64)

    was_ever_aware = np.full(
        shape=grid.shape, fill_value=False, dtype=np.bool
    )  # Generates a grid of False to start every individual of with the status of not being aware
    was_ever_quarantined = np.full(
        shape=grid.shape, fill_value=False, dtype=np.bool
    )  # Generates a grid of False to start every individual of with the status of not being quarantined

    quarantine_duration_grid = np.zeros(
        (grid_size, grid_size), dtype=np.int_
    )  # to track how many days are left for each quarantined individual

    infection_day_grid = np.zeros_like(grid, dtype=np.int64)
    infection_day_grid[...] = (
        -1
    )  # -1 used to indicate recovered / never infected individuals

    return (
        grid,
        recovery_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    )

# Used for generating neighbours for an individual
# Input -
#   x - the position of an individual
#   gridsize - square root of the size of the grid (15 x 15 - 15)
# Output -
#   neighbours - a two dimensional array of positions of the neighbours
def get_neighbours(x: np.ndarray, gridsize: int) -> np.ndarray:

    if x.shape == (0,):  # prevents generating neighbours invalid individual position
        return x

    possible_neighbours = np.array(
        [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
    )  # all possible neighbours

    neighbours = (
        x + possible_neighbours
    )  # generates all possible neighbours for the individual

    mask1 = (neighbours[:, 0] < gridsize) & (
        neighbours[:, 1] < gridsize
    )  # Ensures the neighbours generated are valid

    mask2 = (neighbours[:, 0] >= 0) & (
        neighbours[:, 1] >= 0
    )  # Ensures the neighbours generated are valid

    row_mask = mask1 & mask2  # filters out invalid neighbours
    neighbours = neighbours[row_mask]  # applies the filter to the neighbours

    return neighbours


# Used for getting the positions of a certain label
# Input -
#   grid - a N x N grid to check for labels
#   label - the label number [0,1,2,3,4,5,6]
# Output -
#   positions - a two-dimensional array of the positions
def get_pos(grid: np.ndarray, label: int) -> np.ndarray:
    it = np.nditer(
        grid, order="C", flags=["multi_index"]
    )  # used to iterate through the grid entry by entry
    positions = np.zeros((1, 2))  # Used instead of an empty array
    first = True
    for y in it:
        if first and label == y:  # for the first position identified
            positions = np.array(
                it.multi_index
            )  # redefine positons to include a valid one
            first = False
        elif label == y:
            positions = np.vstack(
                [positions, it.multi_index]
            )  # add the new position to the array

    if (
        positions.ndim == 1 and not first
    ):  # Makes positons a two-dimensional array in case there was only one position identified
        positions = positions[np.newaxis, ...]
    elif positions.dtype == np.float64:  # For cases where no positons were found
        positions = np.array([])
    return positions

# Used to obtain a percentage of a label among neighbours
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   neighbours - a 2 dim array of positions of neighbours
#   x - the label to obtain a percentage of
# Output -
#   score - the percentage of that label among the passed neighbours
def get_score(grid: np.ndarray, neighbours: np.ndarray, x: int) -> float:
    # in case of there not existing any neighbours
    if neighbours.shape == (0,):
        return -1
    # to get the neighbours tag
    neighbour_tag = grid[
        neighbours.T.astype(np.int_)[0], neighbours.T.astype(np.int_)[1]
    ]
    # in case of 8 neighbours
    if neighbour_tag.size == 8:
        score = 0.125 * neighbour_tag[neighbour_tag == x].size
        return score

    # in case of 5 neighbours
    elif neighbour_tag.size == 3:
        score = 0.2 * neighbour_tag[neighbour_tag == x].size
        return score

    # in case of 3 neighbours
    else:
        score = 0.3333333333 * neighbour_tag[neighbour_tag == x].size
        return score