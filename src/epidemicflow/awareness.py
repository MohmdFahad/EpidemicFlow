import numpy as np
from .grid_utils import get_neighbours, get_score


def generate_awareness_day_cycle_parameters(
    infection_prob: float, awareness_rate: float
):

    if infection_prob >= 0.8 and awareness_rate <= 0.3:
        alpha = 4  # infection sensitivity
        beta = 5  # awareness sensitivity
        min_cycle = 3
        max_cycle = 6
    elif infection_prob >= 0.7 and awareness_rate <= 0.5:  # current
        alpha = 3  # 4
        beta = 8 # 7 !9 >8 7
        min_cycle = 2  # !5 4 >3
        max_cycle = 7
    elif infection_prob <= 0.4 and awareness_rate >= 0.6:
        alpha = 3
        beta = 7
        min_cycle = 4
        max_cycle = 8

    elif infection_prob >= 0.7 and awareness_rate >= 0.6:
        alpha = 4.5
        beta = 6
        min_cycle = 3
        max_cycle = 7
    else:
        alpha =  3# >9 >()8 >7 >6 5 4
        beta = 1  # 4 >3 2 ()5
        min_cycle = 1  # 2 >3 4 ()1
        max_cycle = 4  # 8 7 6 ()7

    return alpha, beta, min_cycle, max_cycle


# Used to run population-wide awareness campaigns: each unaware susceptible individual becomes aware with a small
# probability that rises with the infection level and falls as awareness saturates. Called ONCE per awareness event.
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a boolean two dimensional array capturing the state of every individual over the simulation
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   rng - the random number generator for this run
# Output -
#   None
def global_awareness(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    gridsize: int,
    *,
    rng: np.random.Generator,
):
    infection_percent = (
        grid[grid == 1].size + grid[grid == 5].size
    ) / gridsize**2  # percentage of infected individuals in the grid
    awareness_percent = (
        grid[grid == 4].size + grid[grid == 5].size
    ) / gridsize**2  # percentage of aware individuals in the grid

    base_awareness_rate = 0.002  # minimum awareness rate
    max_awareness_rate = 0.01  # maximum awareness rate
    infection_sensitivity = 6.0  # steepness of the infection response curve
    infection_inflection = 0.5  # where infection triggers strongest awareness response
    awareness_sensitivity = 13.0  # how strongly high awareness slows new adoption
    awareness_inflection = 0.1  # where slowdown starts

    # Infection driven term: higher infection percent -> higher awareness
    infection_term = 1 / (
        1 + np.exp(-infection_sensitivity * (infection_percent - infection_inflection))
    )
    # Awareness-driven term: higher awareness percent -> lower new awareness
    awareness_term = 1 - (
        1 / (1 + np.exp(-awareness_sensitivity * (awareness_percent - awareness_inflection)))
    )

    global_awareness_factor = (
        base_awareness_rate
        + (max_awareness_rate - base_awareness_rate) * infection_term * awareness_term
    )  # chance that each unaware susceptible individual becomes aware in this event

    newly_aware = (grid == 0) & (
        rng.random(size=grid.shape) < global_awareness_factor
    )  # unaware susceptible individuals reached by the campaign
    grid[newly_aware] = 4  # make them aware and susceptible
    was_ever_aware_grid[newly_aware] = True  # update the individuals to having been aware


# Used to spread awareness to a position based on function parameters and individual spatiality
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a boolean two dimensional array capturing the state of every individual over the simulation
#   pos - a 1-dimensional array that contains the position of the individual to spread awareness to
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   awareness_rate - the chance a susceptible individual can gain awareness from aware individuals | None - in case the model chosen in non behavioral
#   rng - the random number generator for this run
# Output -
#   None
def spread_awareness(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    pos: np.ndarray,
    gridsize: int,
    awareness_rate: float = None,
    *,
    rng: np.random.Generator,
):
    # Local awareness

    if grid[pos[0], pos[1]] in (0, 1):  # if the individual is susceptible or infected, and not yet aware
        infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize**2
        chance = rng.random()  # produces a random chance
        score = get_score(grid, get_neighbours(pos, gridsize), 4) + get_score(
            grid, get_neighbours(pos, gridsize), 5
        )  # gets the percentage of aware neighbours
        # Applying a sigmoid function for awareness probability

        # Steepness parameter
        k_min = 1  # min steepness
        k_max = 33  # max steepness
        lda = 2  # steepness growth rate
        k = k_min + (k_max - k_min) * score**lda  # steepness
        # More aware neighbours -> higher steepness -> higher awareness

        # threshold parameter (awareness threshold)
        b_max = 0.24  # min threshold
        b_min = 0.74  # max threshold
        lda = 1  # threshold decay rate
        b = b_min + (b_max - b_min) * np.exp(-lda * score)
        # More aware neighbours -> lower threshold -> higher awareness

        alpha = 1.5  # weight of infection percent (that is global awareness) in the final score
        x = (
            score + alpha * infection_percent
        )  # final score combining local and global awareness factors

        # Sigmoid function
        sigmoid = 1 / (1 + np.exp(-k * (x - b)))  # the probability

        if (
            awareness_rate and chance < awareness_rate * sigmoid
        ):  # If the individual falls in the chance of being aware
            if grid[pos[0], pos[1]] == 1:  # if the individual is infected
                grid[pos[0], pos[1]] = 5  # make the individual aware and infected
                was_ever_aware_grid[pos[0], pos[1]] = (
                    True  # update the individual to having been aware
                )
            else:
                grid[pos[0], pos[1]] = 4  # make the individual aware and susceptible
                was_ever_aware_grid[pos[0], pos[1]] = (
                    True  # update the individual to having been aware
                )
