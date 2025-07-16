import numpy as np


def largest_infection_cluster(grid):
    pass


def make_grid(
    grid_size: int, init_infections: int, a: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    np.random.seed(a)

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)

    total_no = grid_size**2

    for i in range(init_infections):
        mask = grid == 0
        view = grid[mask]
        new = np.zeros(view.shape)
        randindex = np.random.randint(0, new.size)
        new[randindex] = 1
        grid[mask] = new

    recovery_grid = np.zeros_like(grid)
    recovery_grid[...] = -1
    recovery_grid = recovery_grid.astype(np.int64)
    was_ever_aware = grid != grid
    was_ever_qurantined = grid != grid

    return grid, recovery_grid, was_ever_aware, was_ever_qurantined


def get_neighbours(x: np.ndarray, gridsize: int) -> np.ndarray:
    if x.shape == (0,):
        return x
    neighbours = np.zeros((1, 2))
    possible_neighbours = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    neighbours = x + possible_neighbours
    mask1 = (neighbours[:, 0] < gridsize) & (neighbours[:, 1] < gridsize)
    mask2 = (neighbours[:, 0] >= 0) & (neighbours[:, 1] >= 0)
    row_mask = mask1 & mask2
    neighbours = neighbours[row_mask]
    return neighbours


def get_pos(grid: np.ndarray, x: int) -> np.ndarray:
    it = np.nditer(grid, order="C", flags=["multi_index"])
    positions = np.zeros((1, 2))
    first = True
    for y in it:
        if first and x == y:
            positions = np.array(it.multi_index)
            first = False
        elif x == y:
            positions = np.vstack([positions, it.multi_index])
    if positions.ndim == 1 and not first:
        positions = positions[np.newaxis, ...]
    elif positions.dtype == np.float64:
        positions = np.array([[]])
    return positions


def grid_status_update(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    gridsize: int,
    awarness_rate: int = None,
):
    qurantine_pos = get_pos(grid, 3)
    for pos in qurantine_pos:
        score = get_score(grid, get_neighbours(pos, gridsize), 1)
        if score == -1:
            continue
        elif score != 1:
            grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = 4
            was_ever_aware_grid[pos[0].astype(np.int_), pos[1].astype(np.int_)] = True

    prev = grid[grid == 4].size
    current = 0
    while prev != current:
        prev = grid[grid == 4].size
        positions = get_pos(grid, 4)
        for position in positions:
            for neighbour in get_neighbours(position, gridsize):
                spread_awarness(
                    grid, was_ever_aware_grid, neighbour, gridsize, awarness_rate
                )

        current = grid[grid == 4].size


def spread_awarness(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    pos: np.ndarray,
    gridsize: int,
    awarness_rate: float = None,
):
    if grid[pos[0], pos[1]] == 0:
        chance = np.random.random()
        score = get_score(grid, get_neighbours(pos, gridsize), 4)
        infected_percent = (grid[grid == 2].size) / (gridsize**2)
        k_min = 1
        k_max = 33
        lda = 2
        k = k_min + (k_max - k_min) * score**lda  # steepness
        b_max = 0.74
        b_min = 0.24
        lda = 1
        b = b_min + (b_max - b_min) * np.exp(-lda * score)
        sigmoid = 1 / (1 + np.exp(-k * (score - b)))
        if awarness_rate and chance < awarness_rate * sigmoid:
            grid[pos[0], pos[1]] = 4
            was_ever_aware_grid[pos[0], pos[1]] = True
            neighbours = get_neighbours(pos, grid.shape[0])
            for neighbour in neighbours:
                spread_awarness(
                    grid, was_ever_aware_grid, neighbour, gridsize, awarness_rate
                )


def get_score(grid: np.ndarray, neighbours: np.ndarray, x: int) -> float:
    if neighbours.shape == (0,):
        return -1
    neighbour_tag = grid[
        neighbours.T.astype(np.int_)[0], neighbours.T.astype(np.int_)[1]
    ]
    if neighbour_tag.size == 4:
        score = 0.25 * neighbour_tag[neighbour_tag == x].size
        return score
    elif neighbour_tag.size == 3:
        score = 0.33333333333333335 * neighbour_tag[neighbour_tag == x].size
        return score
    else:
        score = 0.5 * neighbour_tag[neighbour_tag == x].size
        return score


def infect(
    grid: np.ndarray,
    was_ever_aware_grid: np.ndarray,
    was_ever_qurantined: np.ndarray,
    infection_prob: float,
    awarness_rate: float = None,
    qurantine_chance: float = None,
    awarness_efficacy: float = None,
):

    infected_pos = get_pos(grid, 1)

    recovered_pos = get_pos(grid, 2)

    awarness_pos = get_pos(grid, 4)

    gridsize = grid.shape[0]
    for x in infected_pos:

        neighbours = get_neighbours(x, gridsize)

        if infected_pos.size != 0:
            for y in infected_pos:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]
        if recovered_pos.size != 0:
            for y in recovered_pos:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]

        chance_of_disease = np.random.random((neighbours.size // 2,))
        diseased = chance_of_disease < infection_prob
        diseased = neighbours[diseased]
        for y in diseased:
            score = get_score(grid, get_neighbours(y, gridsize), 1)
            infection_percent = grid[grid == 1].size / gridsize**2
            if score == 1:
                if awarness_rate:
                    if not grid[y[0], y[1]] == 3:
                        chance = np.random.random()
                        if chance > qurantine_chance:
                            grid[y[0], y[1]] = 1
                        else:
                            grid[y[0], y[1]] = 3
                            was_ever_qurantined[y[0], y[1]] = True
                else:
                    grid[y[0], y[1]] = 1
            else:
                if grid[y[0], y[1]] != 4:
                    if awarness_rate:
                        chance = np.random.random()
                        alpha = 1.5
                        x = score * (1 + alpha * infection_percent)
                        k = 10
                        b = 0.3
                        sigmoid = 1 / (1 + np.exp(-k * (x - b)))
                        P_aware = awarness_rate * sigmoid
                        if chance > P_aware:
                            grid[y[0], y[1]] = 1
                        else:
                            was_ever_aware_grid[y[0], y[1]] = True
                            grid[y[0], y[1]] = 4
                    else:
                        grid[y[0], y[1]] = 1
                else:
                    if awarness_rate:
                        chance = np.random.random()
                        if (
                            awarness_efficacy
                            and score * (1 - awarness_efficacy) > chance
                        ):
                            grid[y[0], y[1]] = 1
                        elif not awarness_efficacy:
                            grid[y[0], y[1]] = 1
                        else:
                            pass
                    else:
                        grid[y[0], y[1]] = 1


def recover(
    day: int,
    grid: np.ndarray,
    recovery_grid: np.ndarray,
    recovery_mean: int,
    recovery_var: int,
    recovery_times: np.ndarray,
) -> np.ndarray:
    k = recovery_mean**2 / recovery_var**2
    theta = recovery_var**2 / recovery_mean
    infected = (grid == 1) & (recovery_grid == -1)
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
    was_ever_qurantined: np.ndarray,
    infection_prob: float,
    recovery_mean: int,
    recovery_var: int,
    awarness_rate: float = None,
    qurantine_chance: float = None,
    awarness_efficacy: float = None,
    print_: bool = True,
):

    total_infected = grid[grid == 1].size
    total_recovered = 0
    prev_peak_infection = total_infected
    peak_infection = total_infected
    peak_infection_day = 0
    peak_awarness = grid[grid == 4].size
    gridsize = grid.shape[0]
    qurantined = 0

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
    3 - Qurantined (immune)
    4 - Aware and susceptible\n"""
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

    while grid[grid == 1].size > 0:
        prev_grid = grid
        before_qurantined = qurantined
        grid_status_update(grid, was_ever_aware_grid, gridsize, awarness_rate)
        infect(
            grid,
            was_ever_aware_grid,
            was_ever_qurantined,
            infection_prob,
            awarness_rate,
            qurantine_chance,
            awarness_efficacy,
        )
        recovery_times = recover(
            day,
            grid,
            recovery_grid,
            recovery_mean,
            recovery_var,
            recovery_times,
        )
        after_qurantined = grid[grid == 3].size
        qurantined = max(after_qurantined - before_qurantined, qurantined)
        total_recovered = grid[grid == 2].size
        total_infected = grid[grid == 1].size
        prev_peak_infection = peak_infection
        peak_infection = max(peak_infection, grid[grid == 1].size)
        if prev_peak_infection != peak_infection:
            peak_infection_day = day
        peak_awarness = max(peak_awarness, grid[grid == 4].size)
        if prev_peak_infection != peak_infection:
            max_grid = grid
        i += 1
        day += 1

    never_infected = grid[grid == 0].size + grid[grid == 3].size + grid[grid == 4].size
    sum_recovery_duration = np.sum(recovery_times)
    if awarness_efficacy:
        total_ever_aware = np.sum(was_ever_aware_grid)
        total_grid = gridsize**2
        average_awarness_efficacy = (total_ever_aware / total_grid) * awarness_efficacy
        total_ever_qurantined = np.sum(was_ever_qurantined)

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

        simulation_no_awarness = None

        print("> Epidemic Duration:", day)
        print(
            f"> Peak Active Infections: {peak_infection} (on Day {peak_infection_day})"
        )
        print("> Total Infected:", total_recovered)
        print("> Total Recovered:", total_recovered)
        print("> Never Infected:", never_infected)
        print()
        print("--- Behavior and Awarness ---")
        print("> Individuals who became aware:", peak_awarness)
        print(
            "> Average Awarness Efficacy:", np.round(average_awarness_efficacy * 100, 2)
        )
        print("> Individuals who qurantined:", total_ever_qurantined)
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
        awarness_rate = (
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
        qurantine_chance = (
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
        awarness_efficacy = (
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
        awarness_rate = None
        qurantine_chance = None
        awarness_efficacy = None

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

    t = np.random.randint(1, 101)

    grid, recovery_grid, was_ever_aware_grid, was_ever_qurantined = make_grid(
        grid_size, init_infections, t
    )

    simulation(
        grid,
        recovery_grid,
        was_ever_aware_grid,
        was_ever_qurantined,
        infection_prob,
        recovery_mean,
        recovery_var,
        awarness_rate,
        qurantine_chance,
        awarness_efficacy,
    )


if __name__ == "__main__":
    main()
