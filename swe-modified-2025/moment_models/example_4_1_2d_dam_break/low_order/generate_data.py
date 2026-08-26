#!/usr/bin/env python3
"""Generate the low-order Example 4.1 moment-model data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from MSWME import SWME_Solver


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CaseConfig:
    key: str
    display_name: str
    kappa: float
    initial_case: str
    initial_profile: str


@dataclass(frozen=True)
class ModelSpec:
    label: str
    filename_prefix: str
    var_num: int
    model_type: str


CASES = {
    "perfect_slip": CaseConfig(
        key="perfect_slip",
        display_name="perfect slip",
        kappa=0.0,
        initial_case="cubic_perfect_slip",
        initial_profile="u=A(3*zeta^2-2*zeta^3), A=1/3 m/s",
    ),
    "no_slip": CaseConfig(
        key="no_slip",
        display_name="no slip",
        kappa=1.0e10,
        initial_case="quadratic_no_slip",
        initial_profile="u=S(2*zeta-zeta^2), S=0.25 m/s",
    ),
}

MODELS = (
    ModelSpec("SWE", "SWE", 2, "original"),
    ModelSpec("HSWME (N=1)", "SWME", 3, "original"),
    ModelSpec("HSWME (N=2)", "SWME2", 4, "original"),
    ModelSpec("HSWME (N=3)", "SWME3", 5, "original"),
    ModelSpec("MSWE", "M-SWE", 2, "modified"),
    ModelSpec("MHSWME (N=1)", "M-SWME", 3, "modified"),
    ModelSpec("MHSWME (N=2)", "M-SWME2", 4, "modified"),
    ModelSpec("MHSWME (N=3)", "M-SWME3", 5, "modified"),
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--case",
        choices=[*CASES, "all"],
        default="all",
    )
    result.add_argument("--nx", type=int, default=4000)
    result.add_argument("--cfl", type=float, default=0.7)
    result.add_argument("--times", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    result.add_argument("--output-dir", type=Path, default=HERE / "data")
    result.add_argument(
        "--models",
        nargs="+",
        choices=[spec.filename_prefix for spec in MODELS],
        default=None,
    )
    return result


def run_case(case: CaseConfig, args: argparse.Namespace) -> None:
    times = sorted(set(float(value) for value in args.times))
    gravity = 9.81
    density = 1000.0
    kinematic_viscosity = 1.0e-6
    horizontal_scale = 100.0
    vertical_scale = 1.5
    velocity_scale = 100.0
    epsilon = vertical_scale / horizontal_scale
    gravity_number = gravity * vertical_scale / velocity_scale**2
    inverse_reynolds = kinematic_viscosity / (velocity_scale * vertical_scale)
    inverse_reynolds_0 = inverse_reynolds / epsilon
    gamma = case.kappa / (density * velocity_scale)
    domain_length = 1.0
    dx = domain_length / args.nx
    saved_times = np.asarray([0.0, *times], dtype=float)
    case_output = args.output_dir / case.key
    case_output.mkdir(parents=True, exist_ok=True)

    selected = (
        MODELS
        if args.models is None
        else tuple(spec for spec in MODELS if spec.filename_prefix in args.models)
    )
    metadata = {
        "case": case.key,
        "bottom_boundary": case.display_name,
        "kappa_kg_per_m2_s": case.kappa,
        "gamma": gamma,
        "initial_profile": case.initial_profile,
        "nx": args.nx,
        "cfl": args.cfl,
        "saved_times_s": saved_times.tolist(),
        "source_treatment": "backward Euler",
        "models_run": [spec.label for spec in selected],
    }

    for spec in selected:
        print(f"Running {case.display_name}: {spec.label}", flush=True)
        state_history, numerical_dt = SWME_Solver(
            velocity_scale,
            1.0,
            2.0 / 3.0,
            domain_length,
            gravity_number,
            args.nx,
            args.cfl,
            times[-1],
            gamma,
            inverse_reynolds_0,
            epsilon,
            spec.model_type,
            spec.var_num,
            times,
            case.initial_case,
            False,
        )
        states = np.asarray(state_history, dtype=float)
        expected_shape = (len(saved_times), args.nx, spec.var_num)
        if states.shape != expected_shape:
            raise ValueError(
                f"{spec.label}: saved state shape {states.shape}, "
                f"expected {expected_shape}"
            )
        if not np.all(np.isfinite(states)) or np.any(states[:, :, 0] <= 0.0):
            raise FloatingPointError(f"{spec.label}: invalid saved solution")
        payload = {
            "U_history": states,
            "time_s": saved_times,
            "numerical_dt_history": np.asarray(numerical_dt, dtype=float),
            "dx": dx,
            "L": domain_length,
            "kappa": case.kappa,
            "gamma": gamma,
            "model": spec.label,
            "model_type": spec.model_type,
            "source_treatment": "backward Euler",
            "var_num": spec.var_num,
            "initial_case": case.initial_case,
            "initial_profile": case.initial_profile,
        }
        output_path = case_output / (
            f"{spec.filename_prefix}_data_kappa{case.kappa:.1e}.npy"
        )
        np.save(output_path, payload)
        print(f"Saved {output_path}", flush=True)

    metadata_path = case_output / "simulation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata_path}")


def main() -> None:
    args = parser().parse_args()
    if args.nx < 10:
        raise ValueError("--nx must be at least 10")
    if not 0.0 < args.cfl <= 1.0:
        raise ValueError("--cfl must lie in (0, 1]")
    times = sorted(set(float(value) for value in args.times))
    if not times or times[0] <= 0.0:
        raise ValueError("--times must contain positive output times")
    requested_cases = CASES.values() if args.case == "all" else [CASES[args.case]]
    for case in requested_cases:
        run_case(case, args)


if __name__ == "__main__":
    main()
