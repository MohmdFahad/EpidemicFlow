import numpy as np
from .grid_utils import *


def generate_infection_day_cycle_parameters(
    infection_prob: float, awareness_rate: float
):

    if infection_prob >= 0.8 and awareness_rate <= 0.3:
        n_min = 2
        n_max = 5
        p0 = 0.65
        a = 11.0
    elif infection_prob >= 0.7 and awareness_rate <= 0.5:  # current
        n_min = 1  # 2 1
        n_max = 6  # !6 7
        p0 = 0.65  # >0.75 >0.85 >>0.90 !0.60 0.80
        a = 13.0  # >11.0 10.0 !()13.0 9.0 11.0
    elif infection_prob <= 0.4 and awareness_rate >= 0.6:
        n_min = 4
        n_max = 8
        p0 = 0.3
        a = 9.0
    elif infection_prob >= 0.7 and awareness_rate >= 0.6:
        n_min = 3
        n_max = 7
        p0 = 0.5
        a = 10.5
    else:
        n_min = 1  # !1 >2
        n_max = 5 # >()7 >6
        p0 = 0.7  # ()0.5 !0.3 >0.4 >0.3 >0.2 0.4 !0.7 >0.6
        a = 2  # ()3 >()2 !3 >4 >5 !6 !7 9 !13 12 !10

    return n_min, n_max, round(p0, 2), round(a, 2)


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
        grid[grid == 1].size + grid[grid == 5].size + grid[grid == 6].size
    ) / gridsize**2  # percentage of infected individuals in the grid

    n_min, n_max, p0, a = generate_infection_day_cycle_parameters(
        infection_prob=infection_prob, awareness_rate=awareness_rate
    )
    n_float = n_min + (n_max - n_min) / (1.0 + np.exp(a * (infection_percent - p0)))
    n = max(1, int(round(n_float)))

    if not (day % n == 0):  # to control infection cycle
        return 0  # no new infections this day

    infected_pos = get_pos(grid, 1)  # get positions of infected unaware individuals
    infected_aware_pos = get_pos(grid, 5)  # get positions of infected aware individuals
    if (
        infected_aware_pos.size != 0 and infected_pos.size != 0
    ):  # combine the two arrays if both have values
        infected_pos = np.concat([infected_pos, infected_aware_pos], axis=0)

    recovered_pos = get_pos(grid, 2)  # get positions of recovered individuals

    sum = 0  # to count new infections

    for x in infected_pos:  # for each infected individual

        neighbours = get_neighbours(
            x, gridsize
        )  # get the neighbours of the infected individual

        aware_diseased = np.array([])  # to store aware neighbours who get infected

        aware_neighbours = np.array([])  # to store aware neighbours

        for neighbour in neighbours:  # to get aware neighbours
            if grid[neighbour[0], neighbour[1]] == 4:  # aware and susceptible
                if aware_neighbours.size == 0:
                    aware_neighbours = np.array([neighbour])
                else:
                    aware_neighbours = np.vstack([aware_neighbours, neighbour])

        # Filter out neighbours who are not susceptible
        for y in infected_pos:  # filter out infected neighbours
            mask3 = neighbours[:, 0] != y[0]
            mask4 = neighbours[:, 1] != y[1]
            neighbours = neighbours[np.logical_or(mask3, mask4)]
        if recovered_pos.size != 0:  # filter out recovered neighbours
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

        if aware_neighbours.size != 0:  # if there are aware neighbours
            # filter out aware neighbours from the main neighbours array
            for y in aware_neighbours:
                mask3 = neighbours[:, 0] != y[0]
                mask4 = neighbours[:, 1] != y[1]
                neighbours = neighbours[np.logical_or(mask3, mask4)]

            chance_of_disease = np.random.random(
                (aware_neighbours.size // 2,)
            )  # random chance for each aware neighbour

            infection_prob_aware = infection_prob * (
                1 - awareness_efficacy
            )  # reduced infection probability for aware individuals

            if grid[x[0], x[1]] == 5:  # if the infected individual is aware
                infection_prob_aware = (
                    infection_prob_aware * 0.3
                )  # further reduce infection probability

            diseased = (
                chance_of_disease < infection_prob_aware
            )  # determine which aware neighbours get infected
            aware_diseased = aware_neighbours[
                diseased
            ]  # store the newly infected aware neighbours

        chance_of_disease = np.random.random(
            (neighbours.size // 2,)
        )  # random chance for each unaware susceptible neighbour
        if grid[x[0], x[1]] == 5:  # if the infected individual is aware
            reduced_prob = infection_prob * 0.3  # further reduce infection probability
            diseased = (
                chance_of_disease < reduced_prob
            )  # determine which susceptible neighbours get infected
        else:
            diseased = chance_of_disease < infection_prob

        if aware_diseased.size != 0:  # if there are newly infected aware neighbours
            diseased = np.vstack(
                [aware_diseased, neighbours[diseased]]
            )  # combine with newly infected unaware neighbours
        else:
            diseased = neighbours[diseased]

        # for spreading the infection in grid
        for y in diseased:
            score = get_score(
                grid, get_neighbours(y, gridsize), 1
            )  # get the percentage of infected neighbours
            infection_percent = (
                grid[grid == 1].size + grid[grid == 5].size
            ) / gridsize**2  # percentage of infected individuals in the grid

            if score >= 0.5:  # if over half the neighbours are infected

                if awareness_rate:
                    if (
                        not grid[y[0], y[1]] == 3
                    ):  # if the individual is not already quarantined
                        chance = np.random.random()
                        if chance > quarantine_chance:  # does not quarantine
                            if grid[y[0], y[1]] == 4:  # if the individual is aware
                                grid[y[0], y[1]] = (
                                    5  # make the individual aware and infected
                                )
                                sum += 1  # increment new infections count
                                infection_day_grid[y[0], y[1]] = (
                                    day  # set the infection day
                                )
                            else:
                                grid[y[0], y[1]] = 1  # infect the individual
                                sum += 1  # increment new infections count
                                infection_day_grid[y[0], y[1]] = (
                                    day  # set the infection day
                                )
                        else:  # quarantines

                            grid[y[0], y[1]] = (
                                3  # make the individual quarantined and susceptible
                            )
                            was_ever_quarantined[y[0], y[1]] = (
                                True  # update the individual to having been quarantined
                            )

                            min_days = 7  # minimum quarantine duration
                            max_days = 14  # maximum quarantine duration

                            days = np.random.randint(
                                min_days, max_days + 1
                            )  # randomly choose a quarantine duration
                            quarantine_duration_grid[y[0], y[1]] = (
                                days  # set the quarantine duration
                            )
                    else:
                        quarantine_infection_reduction = (
                            0.9  # 90% reduction in infection risk due to quarantine
                        )
                        chance = np.random.random()
                        new_chance_infection = infection_prob * (
                            1 - quarantine_infection_reduction
                        )  # reduced infection probability due to quarantine
                        if new_chance_infection > chance:  # infects despite quarantine
                            grid[y[0], y[1]] = 1  # infect the individual
                            sum += 1  # increment new infections count
                            infection_day_grid[y[0], y[1]] = (
                                day  # set the infection day
                            )
                        else:  # does not infect due to quarantine
                            continue
                else:  # no awareness model
                    grid[y[0], y[1]] = 1  # infect the individual
                    sum += 1  # increment new infections count
                    infection_day_grid[y[0], y[1]] = day  # set the infection day
            else:  # if less than half the neighbours are infected

                if grid[y[0], y[1]] != 4:  # if the individual is not already aware

                    if awareness_rate:  # if awareness model is active

                        # Applying a sigmoid function for awareness probability

                        scoreA = get_score(
                            grid, get_neighbours(y, gridsize), 4
                        ) + get_score(
                            grid, get_neighbours(y, gridsize), 5
                        )  # get the percentage of aware neighbours
                        chance = np.random.random()

                        x = (
                            0.6 * scoreA + 0.4 * score
                        )  # final score combining local awareness and infection factors (weights are adjustable)
                        x = np.clip(x, 0.0, 1.0)  # ensure x is within [0, 1]

                        k = 6  # steepness parameter

                        b = (
                            0.45 - 0.15 * infection_percent
                        )  # threshold parameter (awareness threshold)
                        sigmoid = 1 / (1 + np.exp(-k * (x - b)))  # sigmoid function
                        baseline_spont = (
                            0.01  # baseline spontaneous awareness adoption rate
                        )
                        raw = (
                            baseline_spont + (1 - baseline_spont) * sigmoid
                        )  # raw awareness probability before scaling
                        P_aware = np.clip(
                            awareness_rate * raw, 0.0, 1.0
                        )  # final awareness probability scaled by awareness_rate

                        if chance > P_aware:  # does not become aware

                            grid[y[0], y[1]] = 1  # infect the individual
                            sum += 1  # increment new infections count
                            infection_day_grid[y[0], y[1]] = (
                                day  # set the infection day
                            )

                        else:  # becomes aware
                            was_ever_aware_grid[y[0], y[1]] = (
                                True  # update the individual to having been aware
                            )
                            grid[y[0], y[1]] = (
                                4  # make the individual aware and susceptible
                            )
                    else:  # no awareness model

                        grid[y[0], y[1]] = 1  # infect the individual
                        sum += 1  # increment new infections count
                        infection_day_grid[y[0], y[1]] = day  # set the infection day
                else:  # if the individual is already aware
                    if awareness_rate:  # if awareness model is active
                        grid[y[0], y[1]] = 1  # infect the individual
                        sum += 1  # increment new infections count
                        infection_day_grid[y[0], y[1]] = day  # set the infection day
    return sum  # return the number of new infections


# Used to assign a random number of recovery days to newly infected individuals and recover infected individuals across the grid based on their date of recovery
# Input -
#   day - the current day of the simulation
#   grid - an N x N grid to analyze individual spatiality
#   recovery_grid - a grid to track the day each individual is set to recover
#   recovery_mean - the mean number of days for recovery
#   recovery_var - the variance in number of days for recovery
#   recovery_times - an array to store the recovery times of all infected individuals
# Output -
#   recovery_times - an updated array with the recovery times of all newly infected individuals
def recover(
    day: int,
    grid: np.ndarray,
    recovery_grid: np.ndarray,
    recovery_mean: int,
    recovery_var: int,
    recovery_times: np.ndarray,
) -> np.ndarray:

    k = recovery_mean**2 / recovery_var  # shape parameter for gamma distribution
    theta = recovery_var / recovery_mean  # scale parameter for gamma distribution
    infected = (grid == 1) & (recovery_grid == -1) | (grid == 5) & (
        recovery_grid == -1
    )  # identify newly infected individuals without assigned recovery days
    random_recovery_days = np.round(
        np.random.gamma(shape=k, scale=theta, size=recovery_grid[infected].shape), 2
    ).astype(
        np.int64
    )  # generate random recovery days from gamma distribution
    if recovery_times.size == 0:
        recovery_times = random_recovery_days  # initialize recovery_times if empty
    else:
        recovery_times = np.hstack(
            [recovery_times, random_recovery_days]
        )  # append new recovery times
    recovery_grid[infected] = (random_recovery_days + day).astype(
        np.int64
    )  # set recovery day for newly infected individuals
    grid[day == recovery_grid] = 2  # recover individuals whose recovery day is today
    recovery_grid[day == recovery_grid] = (
        -1
    )  # reset recovery day for recovered individuals
    return recovery_times  # return updated recovery times
