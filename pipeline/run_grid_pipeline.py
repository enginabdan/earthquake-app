import argparse
import os
import shlex
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run grid risk prediction and heatmap rendering in one command")
    p.add_argument("--time", type=str, required=True, help="UTC, e.g. 2026-04-01T00:00:00Z")
    p.add_argument("--lat-min", type=str, default="60.00.00S")
    p.add_argument("--lat-max", type=str, default="60.00.00N")
    p.add_argument("--lon-min", type=str, default="180.00.00W")
    p.add_argument("--lon-max", type=str, default="180.00.00E")
    p.add_argument("--step", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--out-png", type=str, default="")
    p.add_argument("--title", type=str, default="Earthquake Grid Risk Heatmap")
    return p.parse_args()


def main():
    args = parse_args()
    this_dir = Path(__file__).resolve().parent

    out_csv = Path(args.out_csv) if args.out_csv else (this_dir.parent / "models" / "grid_risk_map.csv")
    out_png = Path(args.out_png) if args.out_png else out_csv.with_suffix(".png")

    def q(v: object) -> str:
        return shlex.quote(str(v))

    cmd_pred = (
        f"python3 {q(this_dir / 'predict_grid_risk.py')} "
        f"--time {q(args.time)} "
        f"--lat-min {q(args.lat_min)} --lat-max {q(args.lat_max)} "
        f"--lon-min {q(args.lon_min)} --lon-max {q(args.lon_max)} "
        f"--step {q(args.step)} --top-k {q(args.top_k)} --out {q(out_csv)}"
    )

    cmd_plot = (
        f"MPLBACKEND=Agg MPLCONFIGDIR={q(this_dir.parent / '.mplconfig')} "
        f"python3 {q(this_dir / 'render_grid_heatmap.py')} "
        f"--in {q(out_csv)} --out {q(out_png)} --title {q(args.title)}"
    )

    print(cmd_pred)
    rc1 = os.system(cmd_pred)
    if rc1 != 0:
        raise RuntimeError("predict_grid_risk failed")

    print(cmd_plot)
    rc2 = os.system(cmd_plot)
    if rc2 != 0:
        raise RuntimeError("render_grid_heatmap failed")

    print(f"pipeline done: csv={out_csv} png={out_png}")


if __name__ == "__main__":
    main()
