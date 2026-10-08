"""Run every preset scenario over many seeds and save summaries and plots."""

import argparse
from pathlib import Path

from epidemicflow import SCENARIOS
from epidemicflow.experiments import describe, plot_runs, run_many

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    for name, params in SCENARIOS.items():
        daily, summaries = run_many(params, runs=args.runs)
        print(f"\n== {name} ({args.runs} runs) ==")
        print(describe(summaries).round(2).to_string())
        summaries.to_csv(ROOT / "examples" / "results" / f"{name}_summaries.csv")
        plot_runs(daily, name.title(), ROOT / "docs" / "images" / f"{name}.png")


if __name__ == "__main__":
    main()
