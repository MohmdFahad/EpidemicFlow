from src.epidemicflow import make_grid, simulation


def main():
    # Scenario 3 — Childhood Disease–like (e.g., measles)
    infection_prob = .85
    avg_recovery = 9
    variance = 9
    PBRR = .20
    quarantine_prob = .40
    awareness_efficacy = .20
    grid_size = 30
    initial_infections = 8

    output_path = "examples/results/"

    (
        grid,
        recover_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    ) = make_grid(grid_size, initial_infections)
    simulation(
        grid,
        recover_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
        infection_prob,
        avg_recovery,
        variance,
        PBRR,
        quarantine_prob,
        awareness_efficacy,
        output_path,
    )


if __name__ == "__main__":
    main()
