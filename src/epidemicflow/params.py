"""Simulation parameters with validation, plus preset scenarios."""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class SimulationParams:
    infection_prob: float  # chance a susceptible neighbour of an infected person is infected (0-1)
    recovery_mean: float  # mean recovery time in days
    recovery_var: float  # variance of recovery time
    awareness_rate: float  # chance an individual adopts protective behaviour (0-1); 0 turns behaviour off
    quarantine_chance: float  # chance an individual quarantines when surrounded by infections (0-1)
    awareness_efficacy: float  # how much awareness reduces infection risk (0-1)
    grid_size: int = 35
    init_infections: int = 6

    def __post_init__(self):
        for name in ("infection_prob", "awareness_rate", "quarantine_chance", "awareness_efficacy"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}")
        if self.recovery_mean <= 0 or self.recovery_var <= 0:
            raise ValueError("recovery_mean and recovery_var must be positive")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least 2")
        if not 0 < self.init_infections <= self.grid_size**2:
            raise ValueError(f"init_infections must be between 1 and {self.grid_size ** 2}")

    @property
    def behavioural(self) -> bool:
        """True when the awareness / quarantine model is switched on."""
        return self.awareness_rate > 0

    def to_dict(self) -> dict:
        return asdict(self)


SCENARIOS = {
    "influenza": SimulationParams(
        infection_prob=0.55, recovery_mean=8, recovery_var=4, awareness_rate=0.45,
        quarantine_chance=0.60, awareness_efficacy=0.35, grid_size=35, init_infections=6,
    ),
    "covid": SimulationParams(
        infection_prob=0.70, recovery_mean=10, recovery_var=9, awareness_rate=0.30,
        quarantine_chance=0.70, awareness_efficacy=0.30, grid_size=40, init_infections=10,
    ),
    "measles": SimulationParams(
        infection_prob=0.85, recovery_mean=9, recovery_var=9, awareness_rate=0.20,
        quarantine_chance=0.40, awareness_efficacy=0.20, grid_size=30, init_infections=8,
    ),
}
