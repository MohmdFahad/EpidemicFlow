from .simulation import *

# The main function is used to run the epidemic simulation program, gather user inputs for simulation parameters, initialize the grid, and start the simulation process
def main():
    print("-" * 29, ">", " E P I D E M I C  F L O W ", "<", "-" * 29, "\n\n", sep="")
    print(" " * 25, "Welcome to the Epidemic Flow Program!\n")
    print(
        """AIM: The Aim of this project is to simulate the spread of a infectious disease throught 
     a 2D grid of individuals over time in order to conlude the total infected,total 
     recovered, peak infection count and Duration of the epidemic, all over time.\n"""
    )
    print(">> E P I D E M I C   S I M U L A T I O N   P A R A M E T E R S <<\n")
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