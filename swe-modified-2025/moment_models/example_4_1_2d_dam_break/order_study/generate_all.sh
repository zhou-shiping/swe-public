#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$script_dir/hswme_no_slip_order_study/run_orders.py" "$@"
python3 "$script_dir/mhswme_no_slip_order_study/run_orders.py" "$@"
