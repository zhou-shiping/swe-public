#!/usr/bin/env python3
"""Generate the Example 4.3 linear-velocity moment-model data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from MSWME import SWME_Solver


HERE = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--nx", type=int, default=400)
    result.add_argument("--ny", type=int, default=400)
    result.add_argument("--cfl", type=float, default=0.7)
    result.add_argument(
        "--times", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0]
    )
    result.add_argument("--var-num", type=int, nargs="+", default=[7, 5, 3])
    result.add_argument(
        "--model-type",
        choices=["original", "modified"],
        nargs="+",
        default=["original", "modified"],
    )
    result.add_argument("--velocity-slope", type=float, default=0.25)
    result.add_argument("--output-dir", type=Path, default=HERE / "data")
    return result


def model_filename(model_type: str, var_num: int, nx: int, kappa: float) -> str:
    moment_num = int((var_num - 3) / 2)
    prefix = "HSWME" if model_type == "original" else "M-HSWME"
    return f"nonzero{prefix}{moment_num}_N{nx}_k{kappa:.1e}.npy"


def main() -> None:
    args = parser().parse_args()
    times = np.asarray(sorted(set(float(value) for value in args.times)))
    if args.nx < 10 or args.ny < 10:
        raise ValueError("--nx and --ny must be at least 10")
    if not 0.0 < args.cfl <= 1.0:
        raise ValueError("--cfl must lie in (0, 1]")
    if times.size == 0 or times[0] != 0.0 or np.any(np.diff(times) <= 0.0):
        raise ValueError("--times must be strictly increasing and begin at 0")
    if any(value not in {3, 5, 7} for value in args.var_num):
        raise ValueError("--var-num choices are 3, 5, and 7")

    gravity = 9.81
    domain_length = 1.0
    center_height = 1.0
    rest_height = 2.0 / 3.0
    horizontal_scale = 100.0
    vertical_scale = 1.5
    velocity_scale = horizontal_scale
    kinematic_viscosity = 1.0e-6
    density = 1000.0
    kappa = 1.0e10
    epsilon = vertical_scale / horizontal_scale
    gravity_number = gravity * vertical_scale / velocity_scale**2
    inverse_reynolds = kinematic_viscosity / (velocity_scale * vertical_scale)
    gamma = kappa / (density * velocity_scale)
    dx = domain_length / args.nx
    args.output_dir.mkdir(parents=True, exist_ok=True)

    completed = []
    for var_num in args.var_num:
        for model_type in args.model_type:
            print(f"Running {model_type}, var_num={var_num}", flush=True)
            states, saved_dt = SWME_Solver(
                velocity_scale,
                vertical_scale,
                center_height,
                rest_height,
                domain_length,
                gravity_number,
                args.nx,
                args.ny,
                args.cfl,
                float(times[-1]),
                gamma,
                inverse_reynolds,
                epsilon,
                model_type,
                var_num,
                times,
                "nonzero",
                args.velocity_slope,
            )
            states_array = np.asarray(states, dtype=float)
            expected_shape = (len(times), args.nx, args.ny, var_num)
            if states_array.shape != expected_shape:
                raise ValueError(
                    f"saved state shape {states_array.shape}, expected {expected_shape}"
                )
            if not np.all(np.isfinite(states_array)):
                raise FloatingPointError("non-finite saved state")
            filename = model_filename(model_type, var_num, args.nx, kappa)
            output_path = args.output_dir / filename
            payload = {
                "U_history": states_array,
                "dt_history": np.asarray(saved_dt, dtype=float),
                "time_history": times,
                "dx": dx,
                "time_end": float(times[-1]),
                "L": domain_length,
                "k_coe": kappa,
                "model_type": model_type,
                "var_num": var_num,
                "initial_case": "nonzero",
                "velocity_slope_per_s": args.velocity_slope,
            }
            np.save(output_path, payload)
            completed.append(filename)
            print(f"Saved {output_path}", flush=True)

    metadata = {
        "example": "4.3 water collapse with linear initial velocity",
        "nx": args.nx,
        "ny": args.ny,
        "cfl": args.cfl,
        "saved_times_s": times.tolist(),
        "velocity_slope_per_s": args.velocity_slope,
        "files": completed,
    }
    metadata_path = args.output_dir / "simulation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
