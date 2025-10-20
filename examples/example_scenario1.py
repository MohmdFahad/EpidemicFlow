from src.epidemicflow import make_grid, simulation


def main():
    # Scenario 1 — Seasonal Influenza–like
    infection_prob = .70
    avg_recovery = 10
    variance = 9
    PBRR = .30
    quarantine_prob = .60
    awareness_efficacy = .30
    grid_size = 40
    initial_infections = 10

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
