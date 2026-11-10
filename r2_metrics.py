"""Calculate counting and maturity R-squared directly from per_image.csv.

R2 = 1 - sum((y_true - y_pred)^2) / sum((y_true - mean(y_true))^2).
M_true = ripe_true / total_true; M_pred = ripe_pred / total_pred, computed per image.
Counts use all images. Maturity uses images with both totals > 0; excluded image names are reported.
R2 is undefined (JSON null) for fewer than two samples or constant ground truth.
Run: python r2_metrics.py [--csv PATH] [--output PATH]
"""

import argparse
import csv
import json
from math import fsum
from pathlib import Path


def r_squared(y_true, y_pred):
    """Return R2 and its numerator/denominator using the ground-truth mean."""
    mean_true = fsum(y_true) / len(y_true) if y_true else None
    sse = fsum((true - pred) ** 2 for true, pred in zip(y_true, y_pred))
    sst = fsum((true - mean_true) ** 2 for true in y_true)
    return {
        "n": len(y_true),
        "mean_true": mean_true,
        "sse": sse,
        "sst": sst,
        "r2": 1 - sse / sst if len(y_true) >= 2 and sst > 0 else None,
    }


def main():
    """Read saved counts, compute quantity/maturity R2, and save a separate JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(__file__).resolve().parent / "runs/count_metrics/yolov8n_moca_nwd_300e/per_image.csv",
    )
    parser.add_argument("--output", type=Path, help="Defaults to r2_summary.json beside the input CSV")
    args = parser.parse_args()
    with args.csv.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("The input CSV contains no images")
    for row in rows:
        for name in ("total", "ripe", "unripe"):
            for kind in ("true", "pred"):
                key = f"{name}_{kind}"
                row[key] = int(row[key])
                if row[key] < 0:
                    raise ValueError(f"Negative count in {row['image']}: {key}")
        for kind in ("true", "pred"):
            if row[f"total_{kind}"] != row[f"ripe_{kind}"] + row[f"unripe_{kind}"]:
                raise ValueError(f"Class counts do not sum to the total in {row['image']}: {kind}")
    counts = {
        name: r_squared([row[f"{name}_true"] for row in rows], [row[f"{name}_pred"] for row in rows])
        for name in ("total", "ripe", "unripe")
    }
    valid = [row for row in rows if row["total_true"] > 0 and row["total_pred"] > 0]
    maturity = r_squared(
        [row["ripe_true"] / row["total_true"] for row in valid],
        [row["ripe_pred"] / row["total_pred"] for row in valid],
    )
    summary = {
        "source_csv": str(args.csv.resolve()),
        "images": len(rows),
        "r2_definition": "1 - sum((true - pred)^2) / sum((true - mean(true))^2)",
        "count": counts,
        "maturity_definition": "M_true = ripe_true / total_true; M_pred = ripe_pred / total_pred (per image)",
        "maturity": maturity,
        "maturity_excluded_images": [row["image"] for row in rows if row["total_true"] == 0 or row["total_pred"] == 0],
    }
    output = args.output or args.csv.with_name("r2_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
