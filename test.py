import numpy as np
import pandas as pd


# Generates the grids required for the simulation
# Input -
#   grid_size - grid size
#   init_infections - the inital amount of infections the grid should consist of
# Output -
#   grid - the main grid used for labeling the state of each individual
#   recovery_grid - a grid to track when each individual will recover
#   was_ever_aware - a grid to track every individual of whether they were ever aware
#   was_ever_quarantined - a grid to track every individual of whether they ever quarantined
def make_grid(
        grid_size: int, init_infections: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Used for ploting infections on the main grid
    for i in range(init_infections):
        mask = grid == 0  # identify susceptible individuals
        view = grid[mask]
        new = np.zeros(view.shape)
        randindex = np.random.randint(0, new.size)  # chooses a random individual
        new[randindex] = 1  # update the individual to be infected
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

    quarantine_duration_grid = np.zeros((grid_size, grid_size), dtype=np.int_) # Tracks how long each individual has been in quarantine

    infection_day_grid = np.zeros_like(grid, dtype=np.int64)
    infection_day_grid[...] = -1 # -1 used to indicate recovered / never infected individuals

    return (
        grid,
        recovery_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    )


if __name__ == "__main__":
    grid_size = 35
    init_infections = 6
    (
        grid,
        recovery_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    ) = make_grid(grid_size, init_infections)

    awarness_rate = 0.45
    gridsize = grid.shape[0]


    print("Grid:")
    for row in grid:
        print(" " * 21, end="")
        print(row)

    infection_percent = (grid[grid == 1].adsize + grid[grid == 5].size) / gridsize ** 2

    global_awareness_sensitivity = 2.5
    base_awarness_rate = 0.002

    global_awareness_factor = awarness_rate + \
                              (1 - base_awarness_rate) * (
                                      1 / (1 + np.exp(-global_awareness_sensitivity * (infection_percent - 0.2))))

    tmp_grid = grid[grid == 0].copy()
    reached = np.random.random(size=grid[grid == 0].shape) < 0.1

    global_reached_chance = np.random.random(size=tmp_grid.shape)

    print(grid[grid == 0][(global_reached_chance < global_awareness_factor) & reached])

    tmp_grid[(global_reached_chance < global_awareness_factor) & reached] = 4
    grid[grid == 0] = tmp_grid



    print("Grid:")
    for row in grid:
        print(" " * 21, end="")
        print(row)


    print("\nRecovery Grid:")
    print(recovery_grid)
    print("\nWas Ever Aware Grid:")
    print(was_ever_aware)
    print("\nWas Ever Quarantined Grid:")
    print(was_ever_quarantined)
    print("\nQuarantine Duration Grid:")
    print(quarantine_duration_grid)
    print("\nInfection Day Grid:")
    print(infection_day_grid)
