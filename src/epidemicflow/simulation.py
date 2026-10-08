import numpy as np
import pandas as pd
from pathlib import Path

from .awareness import generate_awareness_day_cycle_parameters, spread_awareness
from .grid_utils import get_neighbours, get_pos, get_score
from .infection import infect, recover



# The main simulation function that runs the epidemic simulation over a grid for a number of days, updating the grid state each day based on infection, recovery, awareness spread, and quarantine dynamics
# Input -
#   grid - an N x N grid to analyze individual spatiality
#   recovery_grid - a grid to track the day each individual is set to recover
#   was_ever_aware_grid - a boolean two-dimensional array capturing the awareness state of every individual over the simulation
#   was_ever_quarantined - a boolean two-dimensional array capturing the quarantine state of every individual over the simulation
#   quarantine_duration_grid - a grid to track how many days are left for each quarantined individual
#   infection_day_grid - a grid to track the day each individual got infected
#   infection_prob - the probability of a susceptible individual acquiring the disease around infected neighbours
#   recovery_mean - the mean number of days for recovery
#   recovery_var - the variance in number of days for recovery
#   awareness_rate - the probability that an susceptible individual adopts protective measures | None - in case an awareness model is not used
#   quarantine_chance - the probability that an individual quarantines  | None - in case an awareness model is not used
#   awareness_efficacy - the extent to which awareness reduces infection risk | None - in case an awareness model is not used
#   output_path - path to save the simulation log | Default - data/results/
# Output -
#   None
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
    output_path: str = "default",
):

    print(
        "-" * 23,
        ">",
        " E P I D E M I C  S I M U L A T I O N ",
        "<",
        "-" * 23,
        "\n\n",
        sep="",
    )

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

    gridsize = grid.shape[
        0
    ]  # size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)

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

    # to print the initial grid
    for row in grid:
        print(" " * 21, end="")
        print(row)
    print()
    print("-" * 86 + "\n")

    # to initialize variables for the simulation
    recovery_times = np.array([])
    day = 0
    day += 1
    i = 0
    quarantined = 0  # initial number of quarantined individuals

    # to log the simulation results
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

    # to run the simulation while there are still infected individuals
    while grid[grid == 1].size > 0:
        grid_status_update(
            grid,
            was_ever_aware_grid,
            was_ever_quarantined,
            quarantine_duration_grid,
            day,
            gridsize,
            infection_prob,
            awareness_rate,
            quarantine_chance,
        )  # update the grid state
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
        )  # infect new individuals
        recovery_times = recover(
            day,
            grid,
            recovery_grid,
            recovery_mean,
            recovery_var,
            recovery_times,
        )  # recover individuals
        quarantined = np.sum(
            was_ever_quarantined
        )  # total number of individuals who have ever quarantined
        total_recovered = grid[grid == 2].size  # total number of recovered individuals
        total_infected = (
            grid[grid == 1].size + grid[grid == 5].size + grid[grid == 6].size
        )  # total number of currently infected individuals
        total_susceptible = (
            grid[grid == 0].size + grid[grid == 3].size + grid[grid == 4].size
        )  # total number of susceptible individuals
        total_aware = (
            grid[grid == 4].size + grid[grid == 5].size
        )  # total number of aware individuals
        simulation_log.loc[day] = [
            day,
            total_susceptible,
            total_infected,
            total_recovered,
            total_aware,
            new_infections,
        ]  # log the day's results
        i += 1  # increment day counter
        day += 1  # increment day
        # to print the grid every iteration
        for row in grid:
            print(" " * 21, end="")
            print(row)
        print()

    total_recovered = grid[
        grid == 2
    ].size  # final total number of recovered individuals
    date_range = pd.date_range(
        start=date_now, freq="D", periods=day - 1, name="Date"
    )  # create a date range for the simulation log
    simulation_log.index = date_range  # set the date range as the index of the log
    peak_infection = simulation_log[
        "Infected"
    ].max()  # peak number of active infections
    peak_infection_day = simulation_log.query("Infected == Infected.max()")["Day"].iloc[
        0
    ]  # day of peak infection
    peak_awareness = simulation_log["Aware"].max()  # peak number of aware individuals

    never_infected = (gridsize ** 2) - total_recovered # number of individuals who were never infected
    sum_recovery_duration = np.sum(
        recovery_times
    )  # total recovery duration for all infected individuals
    if awareness_efficacy:  # if awareness model is used
        total_ever_aware = np.sum(
            was_ever_aware_grid
        )  # total number of individuals who became aware
        total_grid = gridsize**2  # total number of individuals in the grid
        average_awareness_efficacy = (
            total_ever_aware / total_grid
        ) * awareness_efficacy  # average awareness efficacy across the grid
        total_ever_quarantined = np.sum(
            was_ever_quarantined
        )  # total number of individuals who ever quarantined

    print(
        "-" * 32,
        ">",
        " F I N A L  G R I D ",
        "<",
        "-" * 32,
        "\n",
        sep="",
    )
    # to print the final grid
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

    print("> Epidemic Duration:", day)
    print(f"> Peak Active Infections: {peak_infection} (on Day {peak_infection_day})")
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


    # Compute project root relative to this file (src/)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    RESULTS_DIR = ROOT_DIR / "data" / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # to save the simulation log as a csv file

    time_now = np.datetime64("now").astype(str).replace(":", "-").replace(" ", "_")
    file_name = f"Sim_results-{time_now}.csv"
    if output_path != "default":
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        simulation_log.to_csv(output_path / file_name)
    else:
        simulation_log.to_csv(RESULTS_DIR / file_name)


# Used for updating the grid, in particular to update status of individuals who are in quarantine status and spread awareness among the individuals who are aware
# Input -
#   grid - an N x N grid to update and check for neighbours and labels
#   was_ever_aware_grid - a boolean two-dimensional array capturing the awareness state of every individual over the simulation
#   was_ever_quarantined - a boolean two-dimensional array capturing the quarantine state of every individual over the simulation
#   quarantine_duration_grid - a grid to track how many days are left for each quarantined individual
#   day - the current day of the simulation
#   gridsize - square root of the size of the grid (e.g. for a grid of shape 15 x 15, gridsize = 15)
#   infection_prob - the probability of a susceptible individual acquiring the disease around infected neighbours
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
    infection_prob: float,
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


    alpha,beta,min_cycle,max_cycle = generate_awareness_day_cycle_parameters(infection_prob=infection_prob, awareness_rate=awareness_rate)

    n_float = max_cycle * np.exp(
        -alpha * infection_percent + beta * awareness_percent
    )  # calculate the cycle length based on infection and awareness percentages

    n = int(
        np.clip(np.round(n_float, 1), min_cycle, max_cycle)
    )  # ensure the cycle length is within the defined bounds
    """n_min, n_max, p0, a = generate_awareness_day_cycle_parameters(infection_prob=infection_prob,
                                                                  awareness_rate=awareness_rate) # get parameters based on infection and awareness rates
    n_float = n_min + (n_max - n_min) / (1.0 + np.exp(a * (awareness_percent - p0))) # calculate the cycle length based on awareness percentages
    n = max(1, int(round(n_float)))"""


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
