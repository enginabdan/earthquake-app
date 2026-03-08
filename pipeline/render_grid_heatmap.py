import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Render heatmap from grid_risk CSV")
    p.add_argument("--in", dest="in_csv", type=str, required=True, help="Path to grid risk csv")
    p.add_argument("--out", dest="out_png", type=str, default="", help="Output PNG path")
    p.add_argument("--title", type=str, default="Earthquake Risk Heatmap")
    p.add_argument("--cmap", type=str, default="inferno")
    return p.parse_args()


def main():
    args = parse_args()
    in_csv = Path(args.in_csv)
    if not in_csv.exists():
        raise FileNotFoundError(f"missing input: {in_csv}")

    df = pd.read_csv(in_csv)
    required = {"latitude", "longitude", "p_eq_m4_plus"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"input csv missing columns: {sorted(miss)}")

    out_png = Path(args.out_png) if args.out_png else in_csv.with_suffix(".png")

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(
        df["longitude"],
        df["latitude"],
        c=df["p_eq_m4_plus"],
        cmap=args.cmap,
        s=55,
        alpha=0.9,
        edgecolors="none",
    )

    ax.set_title(args.title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.25, linewidth=0.5)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Calibrated P(M>=4)")

    plt.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print(f"saved heatmap -> {out_png}")


if __name__ == "__main__":
    main()
