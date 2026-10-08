import numpy as np

from epidemicflow.grid_utils import make_grid

from epidemicflow.simulation import simulation

def main():
    # Scenario 1 — Seasonal Influenza–like
    infection_prob = .55
    avg_recovery = 8
    variance = 4
    PBRR = .45
    quarantine_prob = .60
    awareness_efficacy = .35
    grid_size = 35
    initial_infections = 6

    output_path = "examples/results/"
    rng = np.random.default_rng(1)  # fixed seed: same result every run

    (
        grid,
        recover_grid,
        was_ever_aware,
        was_ever_quarantined,
        quarantine_duration_grid,
        infection_day_grid,
    ) = make_grid(grid_size, initial_infections, rng)
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
        rng=rng,
    )


if __name__ == "__main__":
    main()
