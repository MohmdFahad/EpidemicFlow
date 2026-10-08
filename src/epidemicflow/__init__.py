"""EpidemicFlow: a behavioural, agent-based epidemic simulator on a grid."""

from .params import SCENARIOS, SimulationParams
from .simulation import SimulationResult, format_summary, run_simulation

__version__ = "1.1.0"
__all__ = ["SCENARIOS", "SimulationParams", "SimulationResult", "format_summary", "run_simulation"]
