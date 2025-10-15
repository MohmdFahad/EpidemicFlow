#  Imports
import numpy as np
import pandas as pd


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
    grid_size: int, init_infections: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    # Used for plotting infections on the main grid
    for i in range(init_infections):
        mask = grid == 0  # identify susceptible individuals
        view = grid[mask]
        new = np.zeros(view.shape)
        randindex = np.random.randint(0, new.size)  # chooses a random individual
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


# Used for updating the grid, in particular to update status of individuals who are in quarantine status and spread awareness among the individuals who are aware
# Input -
#   grid - an N x N grid to update and check for neighbours and labels
#   was_ever_aware_grid - a boolean two-dimensional array capturing the awareness state of every individual over the simulation
#   was_ever_quarantined - a boolean two-dimensional array capturing the quarantine state of every individual over the simulation
#   quarantine_duration_grid - a grid to track how many days are left for each quarantined individual
#   day - the current day of the simulation
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   awareness_rate - the chance a susceptible individual can gain awareness from aware individuals | None - in case the model chosen in non-behavioral
#   quarantine_chance - the chance an infected individual surrounded by infected individuals can quarantine | None - in case the model chosen in non-behavioral
# Output -
#   None
def grid_status_update(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    was_ever_quarantined: np.ndarray,
    quarantine_duration_grid: np.ndarray,
    day: int,
    gridsize: int,
    awareness_rate: float = None,
    quarantine_chance: float = None,
):
    # Quarantine updates

    quarantine_pos = get_pos(
        grid, 3
    )  # get positions of quarantined susceptible individuals

    # to update quarantine duration and change status if quarantine is over
    for pos in quarantine_pos:
        if pos.size == 0:
            continue
        if (
            quarantine_duration_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)]
            == 0  # if quarantine is over
        ):
            grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                4  # make the individual aware and susceptible
            )
            was_ever_aware_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                True  # update the individual to having been aware
            )
        else:
            quarantine_duration_grid[
                pos[0].astype(np.int_), pos[1].astype(np.int_)
            ] -= 1  # decrease the quarantine duration by 1 day

    infected_pos = get_pos(grid, 1)  # get positions of infected unaware individuals
    infected_aware_pos = get_pos(grid, 5)  # get positions of infected aware individuals

    if (
        infected_aware_pos.size != 0 and infected_pos.size != 0
    ):  # combine the two arrays if both have values
        infected_pos = np.vstack([infected_pos, infected_aware_pos])

    # to make infected individuals quarantine if surrounded by infected individuals
    for pos in infected_pos:

        score = get_score(
            grid, get_neighbours(pos, gridsize), 1
        )  # get the percentage of infected neighbours
        if score >= 0.5:  # if over half the neighbours are infected
            chance = np.random.random()
            if (
                chance < quarantine_chance
            ):  # if the individual falls in the chance of quarantining
                grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                    6  # make the individual quarantined and infected
                )
                min_days = 7  # minimum quarantine duration
                max_days = 14  # maximum quarantine duration
                days = np.random.randint(
                    min_days, max_days + 1
                )  # randomly choose a quarantine duration
                quarantine_duration_grid[
                    pos[0].astype(np.int_), pos[1].astype(np.int_)
                ] = days  # set the quarantine duration
                was_ever_quarantined[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                    True  # update the individual to having been quarantined
                )

        infected_quarantined_pos = get_pos(
            grid, 6
        )  # get positions of infected quarantined individuals
        for pos in infected_quarantined_pos:  # for each infected quarantined individual
            if pos.size == 0:
                continue
            if (
                quarantine_duration_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)]
                == 0
            ):  # if quarantine is over
                grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                    5  # make the individual aware and infected
                )
                was_ever_aware_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = (
                    True  # update the individual to having been aware
                )
            else:
                quarantine_duration_grid[
                    pos[0].astype(np.int_), pos[1].astype(np.int_)
                ] -= 1  # decrease the quarantine duration by 1 day

    # Awareness updates

    positions = get_pos(grid, 4)  # get positions of aware susceptible individuals
    positions2 = get_pos(grid, 5)  # get positions of aware infected individuals
    if (
        positions.size != 0 and positions2.size != 0
    ):  # combine the two arrays if both have values
        positions = np.vstack([positions, get_pos(grid, 5)])

    N_total = gridsize * gridsize  # total number of individuals in the grid

    awareness_percent = (
        grid[grid == 4].size + grid[grid == 5].size
    ) / N_total  # percentage of aware individuals in the grid
    infection_percent = (
        grid[grid == 1].size + grid[grid == 5].size + grid[grid == 6].size
    ) / N_total  # percentage of infected individuals in the grid

    alpha = 2  # Infection sensitivity | Higher -> lower the cycle |  6 maybe 5
    beta = 9.0  # Awareness sensitivity | Higher -> higher the cycle | 2
    min_cycle = 5  # Minimum number of days | 3
    max_cycle = 6  # Maximum number of days | 6

    n_float = max_cycle * np.exp(
        -alpha * infection_percent + beta * awareness_percent
    )  # calculate the cycle length based on infection and awareness percentages

    n = int(
        np.clip(np.round(n_float, 1), min_cycle, max_cycle)
    )  # ensure the cycle length is within the defined bounds

    # spread the awareness to susceptible neighbours of aware individuals
    if day % n == 0:
        for position in positions:
            for neighbour in get_neighbours(position, gridsize):
                spread_awareness(
                    grid, was_ever_aware_grid, neighbour, gridsize, awareness_rate
                )
    # TODO think about this
    """positions = get_pos(grid, 5)
    for position in positions:
        for neighbour in get_neighbours(position, gridsize):
            spread_awareness(grid, was_ever_aware_grid, neighbour, gridsize, awareness_rate)"""


# Used to spread awareness to a position based on function parameters and individual spatiality
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a boolean two dimensional array capturing the state of every individual over the simulation
#   pos - a 1-dimensional array that contains the position of the individual to spread awareness to
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   awareness_rate - the chance a susceptible individual can gain awareness from aware individuals | None - in case the model chosen in non behavioral
# Output -
#   None
def spread_awareness(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    pos: np.ndarray,
    gridsize: int,
    awareness_rate: float = None,
):
    # Global awareness
    infection_percent = (
        grid[grid == 1].size + grid[grid == 5].size
    ) / gridsize**2  # percentage of infected individuals in the grid
    awareness_percent = (
        grid[grid == 4].size + grid[grid == 5].size
    ) / gridsize**2  # percentage of aware individuals in the grid

    base_awareness_rate = 0.002  # minimum awareness rate
    max_awareness_rate = 0.01  # maximum awareness rate
    infection_sensitivity = 4.0  # steepness of the infection response curve
    infection_inflection = 0.7  # where infection triggers strongest awareness response
    awareness_sensitivity = 13.0  # how strongly high awareness slow new adoption
    awareness_inflection = 0.1  # where slowdown starts

    # Infection driven term
    infection_term = 1 / (
        1 + np.exp(-infection_sensitivity * (infection_percent - infection_inflection))
    )  # sigmoid function for infection impact on awareness
    # Higher infection percent -> higher infection term -> higher awareness

    # Awareness-driven term
    awareness_term = 1 - (
        1
        / (
            1
            + np.exp(
                -awareness_sensitivity * (awareness_percent - awareness_inflection)
            )
        )
    )  # sigmoid function for awareness impact on awareness
    # Higher awareness percent -> lower awareness term -> lower awareness

    global_awareness_factor = (
        base_awareness_rate
        + (max_awareness_rate - base_awareness_rate) * infection_term * awareness_term
    )  # final global awareness factor

    tmp_grid = grid[
        grid == 0
    ].copy()  # temporary grid to apply global awareness changes
    reached = (
        np.random.random(size=tmp_grid.shape) < 0.025
    )  # 2.5% chance of being reached by global awareness campaigns

    global_chance = np.random.random(
        size=tmp_grid.shape
    )  # random chance for each susceptible individual

    tmp_grid[(global_chance < global_awareness_factor) & reached] = (
        4  # make susceptible individuals aware based on global awareness factor and reach chance
    )

    grid[grid == 0] = tmp_grid  # apply changes to the main grid
    was_ever_aware_grid[np.logical_and(grid == 4, was_ever_aware_grid == False)] = (
        True  # update the individuals to having been aware
    )

    # Local awareness
    # if the individual is susceptible
    if grid[pos[0], pos[1]] == 0 | grid[pos[0], pos[1]] == 1:  #
        infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize**2
        chance = np.random.random()  # produces a random chance
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


# Used to infect new individuals around the grid who are amongst infected individuals with a probability of infection, and precautionary
# parameters such as probability of awareness,quarantine and awareness efficacy
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a grid to track every individual of whether they were ever aware
#   was_ever_quarantined - a grid to track every individual of whether they ever quarantined
#   quarantine_duration_grid - a grid to track how many days are left for each quarantined individual
#   infection_day_grid - a grid to track the day each individual got infected
#   infection_prob - the probability of a susceptible individual acquiring the disease around infected neighbours
#   awareness_rate - the probability that an susceptible individual adopts protective measures
#   quarantine_chance - the probability that an individual quarantines when surrounded by infected individuals
#   awareness_efficacy - the extent to which awareness reduces infection risk
# Output -
#   sum - the number of new infections that occurred that day
def infect(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    was_ever_quarantined: np.ndarray,
    quarantine_duration_grid: np.ndarray,
    infection_day_grid: np.ndarray,
    day: int,
    infection_prob: float,
    awareness_rate: float = None,
    quarantine_chance: float = None,
    awareness_efficacy: float = None,
) -> int:

    # Dynamic infection cycle based on current infection percentage

    gridsize = grid.shape[
        0
    ]  # size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)

    infection_percent = (
        grid[grid == 1].size + grid[grid == 5].size
    ) / gridsize**2  # percentage of infected individuals in the grid

    n_min, n_max = 2, 5 # min and max cycle length
    p0, a = 0.25, 8.0  # pivot at 25% infected; a = steepness
    n_float = n_min + (n_max - n_min) / (1.0 + np.exp(a * (infection_percent - p0)))
    n = max(1, int(round(n_float)))

    if not (day % n == 0): # to control infection cycle
        return 0 # no new infections this day

    infected_pos = get_pos(grid, 1) # get positions of infected unaware individuals
    infected_aware_pos = get_pos(grid, 5) # get positions of infected aware individuals
    if infected_aware_pos.size != 0 and infected_pos.size != 0: # combine the two arrays if both have values
        infected_pos = np.concat([infected_pos, infected_aware_pos], axis=0)

    recovered_pos = get_pos(grid, 2) # get positions of recovered individuals

    sum = 0 # to count new infections

    for x in infected_pos:  # for each infected individual

        neighbours = get_neighbours(x, gridsize) # get the neighbours of the infected individual

        aware_diseased = np.array([]) # to store aware neighbours who get infected

        aware_neighbours = np.array([]) # to store aware neighbours

        for neighbour in neighbours: # to get aware neighbours
            if grid[neighbour[0], neighbour[1]] == 4: # aware and susceptible
                if aware_neighbours.size == 0:
                    aware_neighbours = np.array([neighbour])
                else:
                    aware_neighbours = np.vstack([aware_neighbours, neighbour])

        # Filter out neighbours who are not susceptible
        for y in infected_pos: # filter out infected neighbours
            mask3 = neighbours[:, 0] != y[0]
            mask4 = neighbours[:, 1] != y[1]
            neighbours = neighbours[np.logical_or(mask3, mask4)]
        if recovered_pos.size != 0: # filter out recovered neighbours
            for y in recovered_pos:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]

        # TODO think about this one
        """recovery_time = (recovery_grid[x[0], x[1]] - infection_day_grid[x[0], x[1]])
        sigma = recovery_time / 2.5
        center = recovery_time * 0.2
        day_since_infection = day - infection_day_grid[x[0], x[1]]
        decay_factor = np.exp(-(((day_since_infection - center) / 2) ** 2) / (2 * sigma ** 2))
        decay_factor = np.clip(decay_factor, 0.3, 1.0)
        infection_prob = infection_prob * decay_factor"""


        if aware_neighbours.size != 0: # if there are aware neighbours
            # filter out aware neighbours from the main neighbours array
            for y in aware_neighbours:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]

            chance_of_disease = np.random.random((aware_neighbours.size // 2,)) # random chance for each aware neighbour

            infection_prob_aware = infection_prob * (1 - awareness_efficacy) # reduced infection probability for aware individuals

            if grid[x[0], x[1]] == 5: # if the infected individual is aware
                infection_prob_aware = infection_prob_aware * 0.3 # further reduce infection probability

            diseased = chance_of_disease < infection_prob_aware # determine which aware neighbours get infected
            aware_diseased = aware_neighbours[diseased] # store the newly infected aware neighbours

        chance_of_disease = np.random.random((neighbours.size // 2,)) # random chance for each unaware susceptible neighbour
        if grid[x[0], x[1]] == 5: # if the infected individual is aware
            reduced_prob = infection_prob * 0.3 # further reduce infection probability
            diseased = chance_of_disease < reduced_prob # determine which susceptible neighbours get infected
        else:
            diseased = chance_of_disease < infection_prob

        if aware_diseased.size != 0: # if there are newly infected aware neighbours
            diseased = np.vstack([aware_diseased, neighbours[diseased]]) # combine with newly infected unaware neighbours
        else:
            diseased = neighbours[diseased]

        # for spreading the infection in grid
        for y in diseased:
            score = get_score(grid, get_neighbours(y, gridsize), 1) # get the percentage of infected neighbours
            infection_percent = (
                grid[grid == 1].size + grid[grid == 5].size
            ) / gridsize**2 # percentage of infected individuals in the grid

            if score >= 0.5: # if over half the neighbours are infected

                if awareness_rate:
                    if not grid[y[0], y[1]] == 3: # if the individual is not already quarantined
                        chance = np.random.random()
                        if chance > quarantine_chance:  # does not quarantine
                            if grid[y[0], y[1]] == 4: # if the individual is aware
                                grid[y[0], y[1]] = 5 # make the individual aware and infected
                                sum += 1 # increment new infections count
                                infection_day_grid[y[0], y[1]] = day # set the infection day
                            else:
                                grid[y[0], y[1]] = 1 # infect the individual
                                sum += 1 # increment new infections count
                                infection_day_grid[y[0], y[1]] = day # set the infection day
                        else: # quarantines

                            grid[y[0], y[1]] = 3 # make the individual quarantined and susceptible
                            was_ever_quarantined[y[0], y[1]] = True # update the individual to having been quarantined

                            min_days = 7 # minimum quarantine duration
                            max_days = 14 # maximum quarantine duration

                            days = np.random.randint(min_days, max_days + 1) # randomly choose a quarantine duration
                            quarantine_duration_grid[y[0], y[1]] = days # set the quarantine duration
                    else:
                        quarantine_infection_reduction = 0.9 # 90% reduction in infection risk due to quarantine
                        chance = np.random.random()
                        new_chance_infection = infection_prob * (
                            1 - quarantine_infection_reduction
                        ) # reduced infection probability due to quarantine
                        if new_chance_infection > chance: # infects despite quarantine
                            grid[y[0], y[1]] = 1 # infect the individual
                            sum += 1 # increment new infections count
                            infection_day_grid[y[0], y[1]] = day # set the infection day
                        else: # does not infect due to quarantine
                            continue
                else: # no awareness model
                    grid[y[0], y[1]] = 1 # infect the individual
                    sum += 1 # increment new infections count
                    infection_day_grid[y[0], y[1]] = day # set the infection day
            else: # if less than half the neighbours are infected

                if grid[y[0], y[1]] != 4: # if the individual is not already aware

                    if awareness_rate: # if awareness model is active

                        # Applying a sigmoid function for awareness probability

                        scoreA = get_score(
                            grid, get_neighbours(y, gridsize), 4
                        ) + get_score(grid, get_neighbours(y, gridsize), 5)  # get the percentage of aware neighbours
                        chance = np.random.random()

                        x = 0.6 * scoreA + 0.4 * score # final score combining local awareness and infection factors (weights are adjustable)
                        x = np.clip(x, 0.0, 1.0) # ensure x is within [0, 1]

                        k = 6 # steepness parameter

                        b = 0.45 - 0.15 * infection_percent # threshold parameter (awareness threshold)
                        sigmoid = 1 / (1 + np.exp(-k * (x - b))) # sigmoid function
                        baseline_spont = 0.01 # baseline spontaneous awareness adoption rate
                        raw = baseline_spont + (1 - baseline_spont) * sigmoid # raw awareness probability before scaling
                        P_aware = np.clip(awareness_rate * raw, 0.0, 1.0) # final awareness probability scaled by awareness_rate

                        if chance > P_aware: # does not become aware

                            grid[y[0], y[1]] = 1 # infect the individual
                            sum += 1 # increment new infections count
                            infection_day_grid[y[0], y[1]] = day # set the infection day

                        else: # becomes aware
                            was_ever_aware_grid[y[0], y[1]] = True # update the individual to having been aware
                            grid[y[0], y[1]] = 4 # make the individual aware and susceptible
                    else: # no awareness model

                        grid[y[0], y[1]] = 1 # infect the individual
                        sum += 1 # increment new infections count
                        infection_day_grid[y[0], y[1]] = day # set the infection day
                else: # if the individual is already aware
                    if awareness_rate: # if awareness model is active
                        grid[y[0], y[1]] = 1 # infect the individual
                        sum += 1 # increment new infections count
                        infection_day_grid[y[0], y[1]] = day # set the infection day
    return sum # return the number of new infections


def recover(
    day: int,
    grid: np.ndarray,
    recovery_grid: np.ndarray,
    recovery_mean: int,
    recovery_var: int,
    recovery_times: np.ndarray,
) -> np.ndarray:
    k = recovery_mean**2 / recovery_var
    theta = recovery_var / recovery_mean
    infected = (grid == 1) & (recovery_grid == -1) | (grid == 5) & (recovery_grid == -1)
    random_recovery_days = np.round(
        np.random.gamma(shape=k, scale=theta, size=recovery_grid[infected].shape), 2
    ).astype(np.int64)
    if recovery_times.size == 0:
        recovery_times = random_recovery_days
    else:
        recovery_times = np.hstack([recovery_times, random_recovery_days])
    recovery_grid[infected] = (random_recovery_days + day).astype(np.int64)
    grid[day == recovery_grid] = 2
    recovery_grid[day == recovery_grid] = -1
    return recovery_times


def simulation(
    grid: np.ndarray,
    recovery_grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    was_ever_quarantined: np.ndarray,
    quarantine_duration_grid: np.ndarray,
    infection_day_grid: np.ndarray,
    infection_prob: float,
    recovery_mean: int,
    recovery_var: int,
    awareness_rate: float = None,
    quarantine_chance: float = None,
    awareness_efficacy: float = None,
    print_: bool = True,
):
    total_infected = grid[grid == 1].size
    total_recovered = 0
    prev_peak_infection = total_infected
    peak_infection = total_infected
    peak_infection_day = 0
    peak_awareness = grid[grid == 4].size
    gridsize = grid.shape[0]
    quarantined = 0

    if print_:
        print(
            "-" * 23,
            ">",
            " E P I D E M I C  S I M U L A T I O N ",
            "<",
            "-" * 23,
            "\n\n",
            sep="",
        )
    if print_:
        print(
            """Legend:
    0 - Susceptible (no infection, unaware)
    1 - Infected 
    2 - Recovered
    3 - quarantined (immune)
    4 - Aware and susceptible
    5 - Aware and Infected
    6 - Quarantined and Infected\n"""
        )
        print(f"Grid Size: {gridsize} x {gridsize}\n")

        print(
            "-" * 30,
            ">",
            " I N I T I A L  G R I D ",
            "<",
            "-" * 30,
            "\n",
            sep="",
        )

        for row in grid:
            print(" " * 21, end="")
            print(row)
        print()
        print("-" * 86 + "\n")

    day = 0

    sum_recovery_duration = 0
    recovery_times = np.array([])

    recovery_times = recover(
        day, grid, recovery_grid, recovery_mean, recovery_var, recovery_times
    )  # --

    day += 1
    i = 0

    max_grid = None

    date_now = np.datetime64("today")
    simulation_log = pd.DataFrame(
        columns=[
            "Day",
            "Susceptible",
            "Infected",
            "Recovered",
            "Aware",
            "New_Infections",
        ],
        dtype=np.int64,
    )

    while grid[grid == 1].size > 0:
        before_quarantined = quarantined
        grid_status_update(
            grid,
            was_ever_aware_grid,
            was_ever_quarantined,
            quarantine_duration_grid,
            day,
            gridsize,
            awareness_rate,
            quarantine_chance,
        )
        new_infections = infect(
            grid,
            was_ever_aware_grid,
            was_ever_quarantined,
            quarantine_duration_grid,
            infection_day_grid,
            day,
            infection_prob,
            awareness_rate,
            quarantine_chance,
            awareness_efficacy,
        )
        recovery_times = recover(
            day,
            grid,
            recovery_grid,
            recovery_mean,
            recovery_var,
            recovery_times,
        )
        after_quarantined = grid[grid == 3].size
        quarantined = max(after_quarantined - before_quarantined, quarantined)
        total_recovered = grid[grid == 2].size
        total_infected = grid[grid == 1].size + grid[grid == 5].size
        total_susceptible = (
            grid[grid == 0].size + grid[grid == 3].size + grid[grid == 4].size
        )
        total_aware = grid[grid == 4].size + grid[grid == 5].size
        simulation_log.loc[day] = [
            day,
            total_susceptible,
            total_infected,
            total_recovered,
            total_aware,
            new_infections,
        ]
        i += 1
        day += 1
        for row in grid:
            print(" " * 21, end="")
            print(row)
        print()

    total_recovered = grid[grid == 2].size
    date_range = pd.date_range(start=date_now, freq="D", periods=day - 1, name="Date")
    simulation_log.index = date_range
    peak_infection = simulation_log["Infected"].max()
    peak_infection_day = simulation_log.query("Infected == Infected.max()")["Day"].iloc[
        0
    ]
    peak_awareness = simulation_log["Aware"].max()

    never_infected = grid[grid == 0].size + grid[grid == 3].size + grid[grid == 4].size
    sum_recovery_duration = np.sum(recovery_times)
    if awareness_efficacy:
        total_ever_aware = np.sum(was_ever_aware_grid)
        total_grid = gridsize**2
        average_awareness_efficacy = (
            total_ever_aware / total_grid
        ) * awareness_efficacy
        total_ever_quarantined = np.sum(was_ever_quarantined)

    if print_:
        print(
            "-" * 32,
            ">",
            " F I N A L  G R I D ",
            "<",
            "-" * 32,
            "\n",
            sep="",
        )

        for row in grid:
            print(" " * 21, end="")
            print(row)
        print()
        print("-" * 86 + "\n")

        print(
            "-" * 25,
            ">",
            " E N D  O F  S I M U L A T I O N ",
            "<",
            "-" * 26,
            "\n\n",
            sep="",
        )

        print(
            "-" * 16,
            ">",
            " E P I D E M I C  S I M U L A T I O N  R E S U L T S ",
            "<",
            "-" * 15,
            "\n\n",
            sep="",
        )

        simulation_no_awareness = None

        print("> Epidemic Duration:", day)
        print(
            f"> Peak Active Infections: {peak_infection} (on Day {peak_infection_day})"
        )
        print("> Total Infected:", total_recovered)
        print("> Total Recovered:", total_recovered)
        print("> Never Infected:", never_infected)
        print()
        print("--- Behavior and awareness ---")
        print("> Individuals who became aware:", total_ever_aware)
        print(
            "> Average awareness Efficacy:",
            np.round(average_awareness_efficacy * 100, 2),
        )
        print("> Individuals who quarantined:", total_ever_quarantined)
        print()
        print("--- Recovery Time Statistics ---")
        print(
            "> Mean Recovery Duration",
            np.round(sum_recovery_duration / recovery_times.size, 2),
        )
        print("> Recovery Std Dev:", np.round(np.std(recovery_times), 2))
        print("> Min Recovery Duration:", np.min(recovery_times))
        print("> Max Recovery Duration:", np.max(recovery_times))
        print()
        print(simulation_log)
        time_now = np.datetime64("now").astype(str).replace(":", "-").replace(" ", "_")
        simulation_log.to_csv(f"./Results/Sim_results-{time_now}.csv")

    if not print_:
        return total_infected


def main():
    print("-" * 29, ">", " E P I D E M I C  F L O W ", "<", "-" * 29, "\n\n", sep="")
    print(" " * 25, "Welcome to the Epidemic Flow Program!\n")
    print(
        """AIM: The Aim of this project is to simulate the spread of a infectious disease throught 
     a 2D grid of individuals over time in order to conlude the total infected,total 
     recovered, peak infection count and Duration of the epidemic, all over time.\n"""
    )
    print(">> E P I D E M I C   S I M U L A T I O N   P A R A M E T E R S <<\n")
    behavioral = input(
        "Would you like the model to incorporate behavioral dynamics? (y/n): "
    ).lower()
    print()
    infection_prob = (
        float(
            input(
                "\nEnter the *infection probability* (%):\n"
                "   The likelihood that a susceptible individual acquires the infection.\n"
                "   Example: 65\n> "
            )
        )
        / 100
    )
    print()
    recovery_mean = int(
        input(
            "\nEnter the *average recovery time* (days):\n"
            "   The typical number of days an individual takes to recover.\n"
            "   Example: 21\n> "
        )
    )
    print()
    recovery_var = int(
        input(
            "\nEnter the *variance in recovery time*:\n"
            "   Higher values create greater variability in recovery durations.\n"
            "   Example: 3\n> "
        )
    )
    print()
    if behavioral == "y":
        awareness_rate = (
            float(
                input(
                    "\nEnter the *protective behavior response rate* (%):\n"
                    "   The probability that an individual adopts protective measures.\n"
                    "   Example: 65\n> "
                )
            )
            / 100
        )
        print()
        quarantine_chance = (
            float(
                input(
                    "\nEnter the *quarantine probability* (%):\n"
                    "   The likelihood that an individual quarantines when surrounded by infections.\n"
                    "   Example: 90\n> "
                )
            )
            / 100
        )
        print()
        awareness_efficacy = (
            float(
                input(
                    "\nEnter the *awareness efficacy* (%):\n"
                    "   The extent to which awareness reduces infection risk.\n"
                    "   Example: 50\n> "
                )
            )
            / 100
        )
        print()
    else:
        awareness_rate = None
        quarantine_chance = None
        awareness_efficacy = None

    grid_size = int(
        input(
            "\nEnter the *grid size*:\n"
            "   The simulation area will be grid_size × grid_size.\n"
            "   Example: 15\n> "
        )
    )
    print()
    init_infections = int(
        input(
            "\nEnter the *initial number of infections*:\n"
            "   The count of initially infected individuals.\n"
            "   Example: 4\n> "
        )
    )

    print("< < P A R A M E T E R S  A C C E P T E D ! > >")
    print()

    (
        grid,
        recovery_grid,
        was_ever_aware_grid,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    ) = make_grid(grid_size, init_infections)

    simulation(
        grid,
        recovery_grid,
        was_ever_aware_grid,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
        infection_prob,
        recovery_mean,
        recovery_var,
        awareness_rate,
        quarantine_chance,
        awareness_efficacy,
    )


if __name__ == "__main__":
    main()
