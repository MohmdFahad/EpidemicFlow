"""Command-line interface: `epidemicflow --help`."""

import argparse
import dataclasses
from datetime import datetime
from pathlib import Path

from .params import SCENARIOS, SimulationParams
from .simulation import format_summary, run_simulation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epidemicflow",
        description="Behavioural agent-based epidemic simulation on a grid.",
    )
    p.add_argument("--scenario", choices=SCENARIOS, default="influenza",
                   help="preset parameters to start from (default: influenza)")
    p.add_argument("--infection-prob", type=float, help="infection probability, 0-1")
    p.add_argument("--recovery-mean", type=float, help="mean recovery time (days)")
    p.add_argument("--recovery-var", type=float, help="variance of recovery time")
    p.add_argument("--awareness-rate", type=float, help="protective behaviour rate, 0-1 (0 disables behaviour)")
    p.add_argument("--quarantine-chance", type=float, help="quarantine probability, 0-1")
    p.add_argument("--awareness-efficacy", type=float, help="risk reduction from awareness, 0-1")
    p.add_argument("--grid-size", type=int, help="grid is N x N")
    p.add_argument("--init-infections", type=int, help="number of initially infected individuals")
    p.add_argument("--seed", type=int, default=None, help="random seed for a reproducible run")
    p.add_argument("--runs", type=int, default=1, help="run many seeds and report mean and spread")
    p.add_argument("--plot", type=Path, help="with --runs > 1: save a plot of the curves to this file")
    p.add_argument("--output", type=Path, default=Path("data/results"), help="folder for CSV output")
    p.add_argument("--verbose", action="store_true", help="print the grid every day")
    return p


def params_from_args(args: argparse.Namespace) -> SimulationParams:
    overrides = {
        f.name: getattr(args, f.name)
        for f in dataclasses.fields(SimulationParams)
        if getattr(args, f.name) is not None
    }
    return dataclasses.replace(SCENARIOS[args.scenario], **overrides)


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        params = params_from_args(args)
    except ValueError as err:
        parser.error(str(err))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    if args.runs == 1:
        result = run_simulation(params, seed=args.seed, verbose=args.verbose)
        print(format_summary(result))
        path = result.to_csv(args.output / f"{args.scenario}-{stamp}.csv")
        print(f"\nDaily log saved to {path}")
        return

    from .experiments import describe, plot_runs, run_many

    first_seed = args.seed if args.seed is not None else 0
    daily, summaries = run_many(params, runs=args.runs, first_seed=first_seed)
    print(f"{args.runs} runs of '{args.scenario}' (seeds {first_seed}-{first_seed + args.runs - 1}):\n")
    print(describe(summaries).round(2).to_string())
    args.output.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output / f"{args.scenario}-{args.runs}runs-{stamp}.csv")
    if args.plot:
        print(f"\nPlot saved to {plot_runs(daily, args.scenario.title(), args.plot)}")


if __name__ == "__main__":
    main()
