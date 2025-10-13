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
        randindex = np.random.randint(0, new.size)  # chooses a random index
        new[randindex] = 1  # update the index
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

    quarantine_duration_grid = np.zeros((grid_size, grid_size), dtype=np.int_)

    infection_day_grid = np.zeros_like(grid, dtype=np.int64)
    infection_day_grid[...] = -1

    return (
        grid,
        recovery_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,

    )


# Used for generating neighbours for a individual
# Input -
#   x - the position of a individual
#   gridsize - square root of the size of the grid (15 x 15 - 15)
# Output -
#   neighbours - a two dimensional array of positions of the neighbours
def get_neighbours(x: np.ndarray, gridsize: int) -> np.ndarray:
    if x.shape == (0,):  # prevents generating neighbours invalid individual position
        return x
    neighbours = np.zeros((1, 2))
    possible_neighbours = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],[1,1],[1,-1],[-1,1],[-1,-1]])
    neighbours = x + possible_neighbours
    mask1 = (neighbours[:, 0] < gridsize) & (
            neighbours[:, 1] < gridsize
    )  # Ensures the neighbours generated are valid
    mask2 = (neighbours[:, 0] >= 0) & (
            neighbours[:, 1] >= 0
    )  # Ensures the neighbours generated are valid
    row_mask = mask1 & mask2
    neighbours = neighbours[row_mask]
    return neighbours


# Used for getting the positions of a certain label
# Input -
#   grid - a N x N grid to check for labels
#   label - the label number [0,1,2,3,4]
# Output -
#   positions - a two dimensional array of the positions
def get_pos(grid: np.ndarray, label: int) -> np.ndarray:
    it = np.nditer(
        grid, order="C", flags=["multi_index"]
    )  # used to itterate through the grid entry by entry
    positions = np.zeros((1, 2))  # Used instead of an empty array
    first = True
    for y in it:
        if first and label == y:  # for the first position identified
            positions = np.array(
                it.multi_index
            )  # redefine positons to include a valid one
            first = False
        elif label == y:
            positions = np.vstack([positions, it.multi_index])

    if (
            positions.ndim == 1 and not first
    ):  # Makes positons a two dimensional array in case there was only one position identified
        positions = positions[np.newaxis, ...]
    elif positions.dtype == np.float64:  # For cases where no positons were found
        positions = np.array([])
    return positions


# Used for updating the grid, in particular to update status of individuals who are in quarantine status and spread awareness among the individuals who are aware
# Input -
#   grid - a N x N grid to update and check for neighbours and labels
#   was_ever_aware_grid - a boolean two dimensional array capturing the state of every individual over the simulation
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   awareness_rate - the chance a susceptible indivdual can gain awareness from aware individuals | None - in case the model chosen in non behavioral
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
    quarantine_pos = get_pos(grid, 3)
    for pos in quarantine_pos:
        if pos.size == 0:
            continue
        if (
                quarantine_duration_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)]
                == 0
        ):
            grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = 4
            was_ever_aware_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = True
        else:
            quarantine_duration_grid[
                pos[0].astype(np.int_), pos[1].astype(np.int_)
            ] -= 1



    infected_pos = get_pos(grid,1)
    infected_aware_pos = get_pos(grid,5)
    if infected_aware_pos.size != 0:
        infected_pos = np.vstack([infected_pos,infected_aware_pos])
    for pos in infected_pos:
        score = get_score(grid,get_neighbours(pos,gridsize),1)
        if score >= .5:
            chance = np.random.random()
            if chance < quarantine_chance:
                grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = 6
                min_days = 7
                max_days = 14
                days = np.random.randint(min_days, max_days + 1)
                quarantine_duration_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = days
                was_ever_quarantined[pos[0].astype(np.int_), pos[1].astype(np.int_)] = True


        infected_quarantined_pos = get_pos(grid,6)
        for pos in infected_quarantined_pos:
            if pos.size == 0:
                continue
            if (
                    quarantine_duration_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)]
                    == 0
            ):
                grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = 5
                was_ever_aware_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = True
            else:
                quarantine_duration_grid[
                    pos[0].astype(np.int_), pos[1].astype(np.int_)
                ] -= 1



    positions = get_pos(grid, 4)
    positions2 = get_pos(grid, 5)
    if  positions.size !=0 and positions2.size != 0:
        positions = np.vstack([positions, get_pos(grid,5)])

    N_total = gridsize * gridsize
    N_aware = was_ever_aware_grid.sum()

    infection_percent = (grid[grid == 1].size + grid[grid == 5].size + grid[grid == 6].size) / N_total

    base_cycle = 4
    aplha = 1.5
    min_cycle = 2

    n_float = base_cycle * np.exp(-aplha* infection_percent)

    n = int(np.clip(np.round(n_float,1),min_cycle,base_cycle))

    # spread the awareness to susceptible neighbours of aware individuals
    if day % n == 0:
        for position in positions:
            for neighbour in get_neighbours(position, gridsize):
                spread_awareness(
                grid, was_ever_aware_grid, neighbour, gridsize, awareness_rate
                )
    """positions = get_pos(grid, 5)
    for position in positions:
        for neighbour in get_neighbours(position, gridsize):
            spread_awareness(grid, was_ever_aware_grid, neighbour, gridsize, awareness_rate)"""



# Used to spread awareness to a position based on function parameters and individual spatiality
# Input -
#   grid - a N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a boolean two dimensional array capturing the state of every individual over the simulation
#   pos - a 1-dimensional array containg the position of the individual to spread awareness to
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   awareness_rate - the chance a susceptible indivdual can gain awareness from aware individuals | None - in case the model chosen in non behavioral
# Output -
#   None
def spread_awareness(
        grid: np.ndarray,
        was_ever_aware_grid: np.ndarray,
        pos: np.ndarray,
        gridsize: int,
        awareness_rate: float = None,
):
    # global awareness
    infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize ** 2
    awareness_percent = (grid[grid == 4].size + grid[grid == 5].size) / gridsize ** 2


    global_awareness_sensitivity = 10
    base_awareness_rate = 0.002 # minimum awareness rate
    max_awareness_rate = 0.01 # maximum awareness rate
    infection_sensitivity = 4.0 # steepness of the infection response curve
    infection_inflection = 0.7 # where infection triggers strongest awareness response
    awareness_sensitivity = 13.0 # how strongly high awareness slow new adoption
    awareness_inflection = 0.1 # where slowdown starts

    # Infection driven term
    infection_term = 1 / (1+ np.exp(-infection_sensitivity * (infection_percent - infection_inflection)))
    
    # Awareness-driven term 
    awareness_term = 1 / (1 + np.exp(-awareness_sensitivity * (awareness_percent - awareness_inflection)))


    global_awareness_factor = base_awareness_rate + (max_awareness_rate - base_awareness_rate) * infection_term * awareness_term

    tmp_grid = grid[grid == 0].copy()
    reached = np.random.random(size=tmp_grid.shape) < 0.025

    global_chance = np.random.random(size=tmp_grid.shape)

    tmp_grid[(global_chance < global_awareness_factor) & reached] = 4

    grid[grid == 0] = tmp_grid
    was_ever_aware_grid[np.logical_and(grid == 4,was_ever_aware_grid == False)] = True

    # local awareness
    # if the individual is susceptible
    if grid[pos[0], pos[1]] == 0 | grid[pos[0], pos[1]] == 1:
        infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize ** 2
        chance = np.random.random()  # produces a random chance
        score = get_score(
            grid, get_neighbours(pos, gridsize), 4
        ) + get_score(grid,get_neighbours(pos,gridsize),5)  # gets the percentage of aware neighbours
        # Appling a sigmoid function for awareness probability

        # Steepness parameter
        k_min = 1 # min steepness
        k_max = 33 # max steepness
        lda = 2 # steepness growth rate
        k = k_min + (k_max - k_min) * score ** lda  # steepness

        # threshold parameter (awareness threshold)
        b_max = 0.24 # min threshold
        b_min = 0.74 # max threshold
        lda = 1 # threshold decay rate
        b = b_min + (b_max - b_min) * np.exp(-lda * score)

        alpha = 1.5
        x = score + alpha * infection_percent

        # Sigmoid function
        sigmoid = 1 / (1 + np.exp(-k * (x - b)))  # the probability

        if (
                awareness_rate and chance < awareness_rate * sigmoid
        ):  # If the indivdual falls in the chance of being aware
            if grid[pos[0], pos[1]] == 1:
                grid[pos[0], pos[1]] = 5
                was_ever_aware_grid[pos[0], pos[1]] = True
            else:
                grid[pos[0], pos[1]] = 4
                was_ever_aware_grid[pos[0], pos[1]] = True
            """neighbours = get_neighbours(pos, grid.shape[0])
            # to spread awareness to the individuals neighbours
            for neighbour in neighbours:
                spread_awareness(
                    grid, was_ever_aware_grid, neighbour, gridsize, awareness_rate
                )"""


# Used to obtain a percentage of a label among neighbours
# Input -
#   grid - a N x N grid to analyze individual spatiality
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


# Used to infect new individuals around the grid who are amongst infected individuals with a probabiility of infection, and precautionary
# parameters such as probability of awareness,quaritine and awareness efficacy
# Input -
#   grid - a N x N grid to analyze individual spatiality
#   was_ever_aware_grid - a grid to track every individual of whether they were ever aware
#   was_ever_quarantined - a grid to track every individual of whether they ever quarantined
#   infection_prob - the probability of a susceptible individual acquiring the disease around infected neighbours
#   awareness_rate - the probability that an susceptible individual adopts protective measures
#   quarantine_chance - the probability that an individual quarantines when surrounded by infected individuals
#   awareness_efficacy - the extent to which awareness reduces infection risk
# Output -
#   None
def infect(
        grid: np.ndarray,
        was_ever_aware_grid: np.ndarray,
        was_ever_quarantined: np.ndarray,
        quarantine_duration_grid: np.ndarray,
        infection_day_grid: np.ndarray,
        recovery_grid: np.ndarray,
        day: int,
        infection_prob: float,
        awareness_rate: float = None,
        quarantine_chance: float = None,
        awareness_efficacy: float = None,
) -> int:
    infected_pos = get_pos(grid, 1)
    infected_aware_pos = get_pos(grid, 5)
    if infected_aware_pos.size != 0 and infected_pos.size != 0:
        infected_pos = np.concat([infected_pos, infected_aware_pos], axis=0)

    recovered_pos = get_pos(grid, 2)


    awareness_pos = get_pos(grid, 4)  # --

    gridsize = grid.shape[0]

    sum = 0

    infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize ** 2

    n_min, n_max = 1, 4
    p0, a = 0.25, 8.0  # pivot at 25% infected; a = steepness
    n_float = n_min + (n_max - n_min) / (1.0 + np.exp(a * (infection_percent - p0)))
    n = max(1, int(round(n_float)))

    if not (day % n == 0):
        return sum

    for x in infected_pos:  # for each infected individual

        neighbours = get_neighbours(x, gridsize)

        diseased = np.array([])

        aware_diseased = np.array([])

        aware_neighbours = np.array([])
        for neighbour in neighbours:
            if grid[neighbour[0], neighbour[1]] == 4:
                if aware_neighbours.size == 0:
                    aware_neighbours = np.array([neighbour])
                else:
                    aware_neighbours = np.vstack([aware_neighbours,neighbour])

        if infected_pos.size != 0:  # --
            # to remove already infected neighbours from being
            for y in infected_pos:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]
        if recovered_pos.size != 0:  # --
            for y in recovered_pos:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]
        if aware_neighbours.size != 0:
            for y in aware_neighbours:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]
            chance_of_disease = np.random.random((aware_neighbours.size // 2,))
            infection_prob_aware = infection_prob * (1 - awareness_efficacy)
            if grid[x[0], x[1]] == 5:
                infection_prob_aware = infection_prob_aware * 0.3
            diseased = chance_of_disease < infection_prob_aware
            aware_diseased = aware_neighbours[diseased]


        chance_of_disease = np.random.random(( neighbours.size// 2,))
        if grid[x[0], x[1]] == 5:
            reduced_prob = infection_prob * 0.3
            diseased = chance_of_disease < reduced_prob
        else:
            diseased = chance_of_disease < infection_prob
        if aware_diseased.size != 0:
            diseased = np.vstack([aware_diseased,neighbours[diseased]])
        else:
            diseased = neighbours[diseased]

        # for spreading the infection in grid
        for y in diseased:
            score = get_score(grid, get_neighbours(y, gridsize), 1)
            infection_percent = (grid[grid == 1].size + grid[grid == 5].size) / gridsize ** 2

            """n_min, n_max = 1, 8
            n_float = n_min + (n_max - n_min) * infection_percent
            n = np.max([1, int(round(n_float))])"""

            # When surrounded by fully infected individuals
            if score >= .5:

                if awareness_rate:
                    # if not already in quarantine status
                    if not grid[y[0], y[1]] == 3:
                        chance = np.random.random()
                        if chance > quarantine_chance:  # infect the individual
                            if grid[y[0], y[1]] == 4:
                                grid[y[0], y[1]] = 5
                                sum+=1
                                latent_period =  np.random.randint(1,4)
                                infection_day_grid[y[0], y[1]] = day
                            else:
                                grid[y[0], y[1]] = 1
                                sum+=1
                                latent_period = np.random.randint(1, 4)
                                infection_day_grid[y[0], y[1]] = day
                        else:  # quarantine the individual
                            grid[y[0], y[1]] = 3
                            was_ever_quarantined[y[0], y[1]] = True

                            min_days = 7
                            max_days = 14
                            days = np.random.randint(min_days, max_days + 1)
                            quarantine_duration_grid[y[0], y[1]] = days
                    else:
                        quarantine_infection_reduction = 0.9
                        chance = np.random.random()
                        new_chance_infection = infection_prob * (
                                1 - quarantine_infection_reduction
                        )
                        if new_chance_infection > chance:
                            grid[y[0], y[1]] = 1
                            sum+=1
                            latent_period = np.random.randint(1, 4)
                            infection_day_grid[y[0], y[1]] = day
                        else:
                            continue
                else:
                    # infect
                    grid[y[0], y[1]] = 1
                    sum+=1
                    latent_period = np.random.randint(1, 4)
                    infection_day_grid[y[0], y[1]] = day
            else:
                #
                if grid[y[0], y[1]] != 4:
                    if awareness_rate:
                        scoreA = get_score(grid, get_neighbours(y, gridsize), 4) + get_score(grid,get_neighbours(y,gridsize),5)
                        chance = np.random.random()
                        alpha = 1.5
                        x = 0.6 * scoreA + 0.4 * score
                        x = np.clip(x,0.0,1.0)
                        k = 6
                        b = 0.45 - 0.15 * infection_percent
                        sigmoid = 1 / (1 + np.exp(-k * (x - b)))
                        baseline_spont = 0.01
                        raw = baseline_spont + (1 - baseline_spont) * sigmoid
                        P_aware = np.clip(awareness_rate * raw,0.0,1.0)
                        if chance > P_aware:
                            grid[y[0], y[1]] = 1
                            sum+=1
                            latent_period = np.random.randint(1, 4)
                            infection_day_grid[y[0], y[1]] = day
                        else:
                            was_ever_aware_grid[y[0], y[1]] = True
                            grid[y[0], y[1]] = 4
                    else:
                        grid[y[0], y[1]] = 1
                        sum+=1
                        latent_period = np.random.randint(1, 4)
                        infection_day_grid[y[0], y[1]] = day
                else:
                    if awareness_rate:
                        """chance = np.random.random()
                        if (
                                awareness_efficacy
                                and score * (1 - awareness_efficacy) > chance
                        ):
                            grid[y[0], y[1]] = 1
                        elif not awareness_efficacy:
                            grid[y[0], y[1]] = 1
                        else:
                            pass
                    else:"""
                        grid[y[0], y[1]] = 1
                        sum+=1
                        latent_period = np.random.randint(1, 4)
                        infection_day_grid[y[0], y[1]] = day
    return sum



def recover(
        day: int,
        grid: np.ndarray,
        recovery_grid: np.ndarray,
        recovery_mean: int,
        recovery_var: int,
        recovery_times: np.ndarray,
) -> np.ndarray:
    k = recovery_mean ** 2 / recovery_var
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
        columns=["Day", "Susceptible", "Infected", "Recovered", "Aware","New_Infections"],dtype=np.int64
    )

    while grid[grid == 1].size > 0:
        before_quarantined = quarantined
        grid_status_update(
            grid, was_ever_aware_grid, was_ever_quarantined, quarantine_duration_grid, day,gridsize, awareness_rate, quarantine_chance
        )
        new_infections = infect(
            grid,
            was_ever_aware_grid,
            was_ever_quarantined,
            quarantine_duration_grid,
            infection_day_grid,
            recovery_grid,
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
        total_susceptible = grid[grid == 0].size + grid[grid == 3].size + grid[grid == 4].size
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
    date_range = pd.date_range(start=date_now, freq="D", periods=day - 1, name = "Date")
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
        total_grid = gridsize ** 2
        average_awareness_efficacy = (total_ever_aware / total_grid) * awareness_efficacy
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
            "> Average awareness Efficacy:", np.round(average_awareness_efficacy * 100, 2)
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
        time_now = np.datetime64("now").astype(str).replace(":", "-").replace(" ","_")
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
