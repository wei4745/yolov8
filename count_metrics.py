"""Evaluate YOLO per-image counting RMSE and MAPE on a labeled validation or test split.

Run with the blueberry defaults: python count_metrics.py
Use --split val to evaluate validation images; test is the default.
Override thresholds with --conf, --iou, and --max-det; keep them fixed when comparing models.
The input size defaults to the checkpoint's training size (800). The detection cap defaults to 1000
because this test split contains an image with 307 annotated objects.

RMSE = sqrt(mean((predicted_count - true_count)^2)), including all images in the selected split.
MAPE (%) = 100 * mean(abs(predicted_count - true_count) / true_count), excluding zero true counts.
Compute these separately for total counts and each class; do not divide dataset-wide totals.
Outputs: per_image.csv (counts) and summary.json (metrics, valid sample counts, and inference settings).
Missing or invalid labels fail evaluation; an empty label file represents zero objects.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, img2label_paths


def load_counts(root, split, class_ids):
    """Read each image's total and class counts directly from its YOLO label file."""
    images = sorted(p for p in (root / "images" / split).iterdir() if p.suffix[1:].lower() in IMG_FORMATS)
    if not images:
        raise FileNotFoundError(f"No images in {root / 'images' / split}")
    truth = {}
    valid_ids = {str(i) for i in class_ids}
    for image, label in zip(images, img2label_paths([str(p) for p in images])):
        labels = [line.split() for line in Path(label).read_text(encoding="utf-8").splitlines() if line.strip()]
        if any(len(row) != 5 or row[0] not in valid_ids for row in labels):
            raise ValueError(f"Invalid YOLO detection label: {label}")
        counts = np.array([sum(int(row[0]) == i for row in labels) for i in class_ids], dtype=np.int64)
        truth[image.resolve()] = np.r_[counts.sum(), counts]
    return images, truth


def count_errors(true, pred):
    """Calculate counting errors; zero true counts are excluded only from MAPE."""
    error, valid = pred - true, true > 0
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_percent": float(np.mean(np.abs(error[valid]) / true[valid]) * 100) if valid.any() else None,
        "mape_valid_images": int(valid.sum()),
        "zero_true_images": int((~valid).sum()),
        "true_count": int(true.sum()),
        "pred_count": int(pred.sum()),
    }


def main():
    """Predict labeled images and save total and per-class counting errors."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "D:/pythonProject/newultralytics/runs/detect/runs/blueberry/yolov8n_moca_nwd_300e/weights/best.pt"
        ),
    )
    parser.add_argument("--data", type=Path, default=Path("D:/blueberry_dj"), help="YOLO dataset root")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Output directory for per_image.csv and summary.json")
    args = parser.parse_args()
    model = YOLO(str(args.model.resolve()))
    root = args.data.resolve()
    class_ids = sorted(model.names)
    names = ["total", *[model.names[i] for i in class_ids]]
    images, truth = load_counts(root, args.split, class_ids)
    settings = {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "batch": 1,
        "augment": False,
        "agnostic_nms": False,
    }
    rows = []
    for result in model.predict(
        source=str(root / "images" / args.split), stream=True, save=False, verbose=False, **settings
    ):
        image = Path(result.path).resolve()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        counts = np.array([np.count_nonzero(classes == i) for i in class_ids])
        pred = np.r_[counts.sum(), counts]
        rows.append([image.name, *np.column_stack((truth[image], pred)).ravel().tolist()])
    if len(rows) != len(images) or {row[0] for row in rows} != {image.name for image in images}:
        raise ValueError(f"Expected predictions for all {len(images)} images, received {len(rows)}")
    counts = np.asarray([row[1:] for row in rows], dtype=np.int64)
    metrics = {}
    for i, name in enumerate(names):
        metrics[name] = count_errors(counts[:, 2 * i], counts[:, 2 * i + 1])
    output = args.output or Path(__file__).resolve().parent / "runs/count_metrics" / args.model.parent.parent.name
    if args.output is None and args.split != "test":
        output /= args.split
    output.mkdir(parents=True, exist_ok=True)
    with (output / "per_image.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["image", *[f"{name}_{kind}" for name in names for kind in ("true", "pred")]])
        writer.writerows(rows)
    summary = {
        "model": str(args.model.resolve()),
        "dataset": str(root),
        "split": args.split,
        "images": len(rows),
        "prediction": settings,
        "device": str(model.predictor.device),
        "metrics": metrics,
        "rmse_definition": "sqrt(mean((pred - true)^2)), over all images in the selected split",
        "mape_definition": "100 * mean(abs(pred - true) / true), over images with true > 0",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
