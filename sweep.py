"""
sweep.py
========
Runs the Riemann solver across a range of right-state values (rho_R or u_R),
dumping each run's x/t overlay plot and 3D phase portrait into a dedicated
sweep subfolder of output/.

Usage:
    python sweep.py --var rho_R --start 3.0 --end 8.0 --steps 10
    python sweep.py --var u_R   --start 5.0 --end 9.0 --steps 15

Output layout:
    output/sweep_<var>_<start>_<end>/
        xt_plots/
            Case1_alpha0.46_L(...)_R(...).png
            ...
        phase_plots_3d/
            Case1_alpha0.46_L(...)_R(...)_3d.html
            ...

Everything in SolverConfig other than the swept variable is left at its
dataclass default. --steps is the total number of runs (inclusive of both
start and end), i.e. values = np.linspace(start, end, steps).
"""

import argparse
import os
import sys
import time

import numpy as np

from config import SolverConfig
from main import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep rho_R or u_R across a range and run the solver for each value."
    )
    parser.add_argument(
        "--var", required=True, choices=["rho_R", "u_R"],
        help="Which right-state variable to sweep."
    )
    parser.add_argument(
        "--start", required=True, type=float,
        help="Starting value of the swept variable."
    )
    parser.add_argument(
        "--end", required=True, type=float,
        help="Ending value of the swept variable."
    )
    parser.add_argument(
        "--steps", required=True, type=int,
        help="Total number of runs (inclusive of both start and end)."
    )
    parser.add_argument(
        "--plot-every", type=int, default=1000,
        help="How many inner Lax-Friedrichs iterations between overlay curve plots (default: 1000)."
    )
    parser.add_argument(
        "--curve-mode", choices=["surface", "line", "both", "none"], default="surface",
        help="How to render the R1/S1/R3/S3 wave curves in the 3D phase plot: "
             "'surface' (translucent extruded ribbons, default), 'line' (flat dashed lines), "
             "'both', or 'none'."
    )
    parser.add_argument(
        "--output-root", default="output",
        help="Root output directory the sweep subfolder is created under (default: output)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.steps < 1:
        sys.exit("--steps must be >= 1")

    values = np.linspace(args.start, args.end, args.steps)

    sweep_dirname = f"sweep_{args.var}_{args.start}_{args.end}"
    sweep_dir = os.path.join(args.output_root, sweep_dirname)
    xt_dir = os.path.join(sweep_dir, "xt_plots")
    phase_dir = os.path.join(sweep_dir, "phase_plots_3d")
    os.makedirs(xt_dir, exist_ok=True)
    os.makedirs(phase_dir, exist_ok=True)

    print(f"Sweeping {args.var} from {args.start} to {args.end} over {args.steps} run(s).")
    print(f"Output -> {sweep_dir}")
    print(f"  x/t plots      -> {xt_dir}")
    print(f"  3D phase plots -> {phase_dir}")
    print()

    t_start = time.time()
    for i, val in enumerate(values, start=1):
        cfg = SolverConfig()
        setattr(cfg, args.var, float(val))

        print(f"[{i}/{args.steps}] {args.var} = {val:.6g}")
        try:
            run_simulation(cfg, xt_dir=xt_dir, phase_dir=phase_dir,
                            plot_every=args.plot_every, curve_mode=args.curve_mode)
        except Exception as exc:
            print(f"  FAILED for {args.var} = {val:.6g}: {exc}", file=sys.stderr)
        print()

    elapsed = time.time() - t_start
    print(f"Sweep complete: {args.steps} run(s) in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()