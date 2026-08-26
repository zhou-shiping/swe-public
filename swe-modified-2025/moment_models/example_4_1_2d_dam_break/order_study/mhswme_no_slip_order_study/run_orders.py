#!/usr/bin/env python3
"""Run the no-slip MHSWME retained-order study."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from high_order_mhswme import solve_mhswme


HERE = Path(__file__).resolve().parent


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--orders", type=int, nargs="+", default=[1, *range(3, 16)]
    )
    result.add_argument("--nx", type=int, default=4000)
    result.add_argument("--cfl", type=float, default=0.7)
    result.add_argument("--times", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    result.add_argument("--output-dir", type=Path, default=HERE / "data")
    result.add_argument("--overwrite", action="store_true")
    return result


def output_path(directory: Path, order: int) -> Path:
    return directory / f"MHSWME_N{order}_data_kappa1.0e+10.npy"


def main() -> None:
    args = parser().parse_args()
    orders = sorted(set(args.orders))
    if not orders or orders[0] < 1:
        raise ValueError("orders must be positive")
    if args.nx < 10:
        raise ValueError("nx must be at least 10")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    scales = {"Lchar_m": 100.0, "Hchar_m": 1.5, "Uchar_m_per_s": 100.0}
    epsilon = scales["Hchar_m"] / scales["Lchar_m"]
    inverse_reynolds = 1.0e-6 / (
        scales["Uchar_m_per_s"] * scales["Hchar_m"]
    )
    inverse_reynolds_0 = inverse_reynolds / epsilon
    gamma = 1.0e10 / (1000.0 * scales["Uchar_m_per_s"])
    run_records = []

    for order in orders:
        path = output_path(args.output_dir, order)
        if path.exists() and not args.overwrite:
            existing = np.load(path, allow_pickle=True).item()
            existing_treatment = existing.get(
                "wall_source_treatment", "unspecified"
            )
            if existing_treatment != "backward Euler":
                raise RuntimeError(
                    f"Existing {path} uses source treatment "
                    f"{existing_treatment!r}; rerun with --overwrite to "
                    "generate backward-Euler data."
                )
            print(f"Skipping existing {path}", flush=True)
            continue

        print(f"Running no-slip MHSWME N={order}", flush=True)

        def progress(step: int, time_value: float, dt: float, speed: float) -> None:
            print(
                f"  N={order:2d}, step={step:4d}, t={time_value:.6f}, "
                f"dt={dt:.3e}, max_speed={speed:.3e}",
                flush=True,
            )

        started = time.perf_counter()
        result = solve_mhswme(
            order=order,
            nx=args.nx,
            cfl=args.cfl,
            output_times=args.times,
            progress=progress,
        )
        elapsed = time.perf_counter() - started
        payload = {
            "U_history": result.states,
            "time_s": result.saved_times,
            "numerical_dt_history": result.dt_history,
            "dx": 1.0 / args.nx,
            "L": 1.0,
            "kappa": 1.0e10,
            "gamma": gamma,
            "model": f"MHSWME (N={order})",
            "model_type": "modified",
            "order": order,
            "var_num": order + 2,
            "initial_case": "quadratic_no_slip",
            "initial_profile": "u=S(2*zeta-zeta^2), S=0.25 m/s",
            "scales": scales,
            "parameters": {
                "epsilon": epsilon,
                "gravity_number": 9.81 * scales["Hchar_m"]
                / scales["Uchar_m_per_s"] ** 2,
                "inverse_reynolds": inverse_reynolds,
                "inverse_reynolds_0": inverse_reynolds_0,
            },
            "wall_source_treatment": "backward Euler",
            "elapsed_wall_seconds": elapsed,
        }
        np.save(path, payload)
        record = {
            "order": order,
            "path": str(path),
            "steps": int(len(result.dt_history)),
            "elapsed_wall_seconds": elapsed,
            "minimum_dt": float(np.min(result.dt_history)),
            "maximum_dt": float(np.max(result.dt_history)),
        }
        run_records.append(record)
        print(
            f"Saved {path} ({record['steps']} steps, {elapsed:.2f} s)",
            flush=True,
        )

    available_runs = []
    for path in sorted(args.output_dir.glob("MHSWME_N*_data_kappa1.0e+10.npy")):
        data = np.load(path, allow_pickle=True).item()
        available_runs.append(
            {
                "order": int(data["order"]),
                "path": str(path),
                "steps": int(len(data["numerical_dt_history"])),
                "elapsed_wall_seconds": float(data["elapsed_wall_seconds"]),
                "wall_source_treatment": data.get(
                    "wall_source_treatment", "unspecified"
                ),
            }
        )
    available_runs.sort(key=lambda record: record["order"])

    metadata = {
        "study": "MHSWME retained-order sweep, no-slip bottom",
        "orders_requested": orders,
        "nx": args.nx,
        "cfl": args.cfl,
        "saved_times_s": [0.0, *sorted(set(float(t) for t in args.times))],
        "kappa_kg_per_m2_s": 1.0e10,
        "wall_source_treatment": "backward Euler",
        "available_runs": available_runs,
        "runs_completed_in_this_call": run_records,
    }
    metadata_path = args.output_dir / "simulation_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {metadata_path}")


if __name__ == "__main__":
    main()
