"""Search validation conf/IoU thresholds with equal weights for three metrics.

Minimize the mean of min-max normalized RMSE, MAPE, and (1 - maturity R2).
Normalization ranges come from eligible candidates in this grid; each metric has weight 1/3.
RMSE uses all validation images. MAPE and maturity R2 use a fixed population with true total > 0.
A candidate with zero predictions in that population is ineligible: its maturity is undefined.
The model runs once to cache raw outputs; greedy NMS at the lowest confidence is reused for higher thresholds.
The winning counts are verified by a separate normal prediction pass on the full validation split.
Run: python search_count_thresholds.py
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torchvision  # noqa: F401 -- enable the repository's compiled torchvision NMS dispatcher

from count_metrics import count_errors, load_counts
from r2_metrics import r_squared
from ultralytics import YOLO
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils.nms import non_max_suppression


class RawPredictionPredictor(DetectionPredictor):
    """Preserve the raw tensor before the normal postprocessor changes its box coordinates."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Copy raw predictions, then use the existing detection postprocessor."""
        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
        self.raw_prediction = raw.detach().clone()
        return super().postprocess(preds, img, orig_imgs, **kwargs)


def threshold_grid(values):
    """Expand a start/end/step triple into inclusive thresholds."""
    start, end, step = values
    if not (0 < start <= end < 1 and step > 0):
        raise ValueError("Threshold ranges require 0 < start <= end < 1 and step > 0")
    values = np.round(np.arange(start, end + step * 0.5, step), 6)
    return values[values <= end]


def evaluate_counts(true, pred):
    """Evaluate the three objectives on their fixed validation populations."""
    errors = count_errors(true[:, 0], pred[:, 0])
    valid = true[:, 0] > 0
    missing = int(np.count_nonzero(pred[valid, 0] == 0))
    maturity_r2 = None
    if missing == 0:
        maturity_r2 = r_squared(
            (true[valid, 1] / true[valid, 0]).tolist(),
            (pred[valid, 1] / pred[valid, 0]).tolist(),
        )["r2"]
    return {
        "rmse": errors["rmse"],
        "mape_percent": errors["mape_percent"],
        "maturity_r2": maturity_r2,
        "zero_pred_on_positive_gt": missing,
        "eligible": maturity_r2 is not None and errors["mape_percent"] is not None,
    }


def main():
    """Search val only, verify the winner, and save all candidates plus their normalized scores."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "D:/pythonProject/newultralytics/runs/detect/runs/blueberry/yolov8n_moca_nwd_300e/weights/best.pt"
        ),
    )
    parser.add_argument("--data", type=Path, default=Path("D:/blueberry_dj"))
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--max-det", type=int, default=1000)
    parser.add_argument(
        "--conf-range", type=float, nargs=3, default=(0.01, 0.80, 0.01), metavar=("START", "END", "STEP")
    )
    parser.add_argument(
        "--iou-range", type=float, nargs=3, default=(0.30, 0.90, 0.01), metavar=("START", "END", "STEP")
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    confs, ious = threshold_grid(args.conf_range), threshold_grid(args.iou_range)
    model = YOLO(str(args.model.resolve()))
    if model.task != "detect" or model.names != {0: "ripe", 1: "unripe"} or getattr(model.model, "end2end", False):
        raise ValueError("This search requires a standard YOLO detection model with ripe/unripe classes")
    root = args.data.resolve()
    images, truth = load_counts(root, "val", sorted(model.names))
    true = np.stack([truth[image.resolve()] for image in images])
    paths = {image.resolve(): i for i, image in enumerate(images)}
    positive = true[:, 0] > 0
    if positive.sum() < 2 or np.var(true[positive, 1] / true[positive, 0]) == 0:
        raise ValueError("Validation maturity R2 requires at least two positive totals with varying maturity")
    settings = {
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "batch": 1,
        "save": False,
        "verbose": False,
        "augment": False,
        "agnostic_nms": False,
        "rect": True,
        "quantize": None,
    }
    raw, reference = {}, np.zeros_like(true)
    print(f"Caching raw predictions for {len(images)} validation images...", flush=True)
    for result in model.predict(
        source=str(root / "images/val"), predictor=RawPredictionPredictor, stream=True, conf=0.25, iou=0.7, **settings
    ):
        image = Path(result.path).resolve()
        raw[image] = model.predictor.raw_prediction
        classes = result.boxes.cls.cpu().numpy().astype(int)
        reference[paths[image]] = [len(classes), np.count_nonzero(classes == 0), np.count_nonzero(classes == 1)]
    if set(raw) != set(paths):
        raise ValueError("The predictor did not return every validation image")
    predictions = np.zeros((len(confs), len(ious), len(images), 3), dtype=np.int64)
    for j, iou in enumerate(ious):
        for image, tensor in raw.items():
            det = (
                non_max_suppression(
                    tensor.clone(), conf_thres=float(confs.min()), iou_thres=float(iou), max_det=args.max_det, nc=2
                )[0]
                .cpu()
                .numpy()
            )
            keep = det[None, :, 4] > confs.astype(det.dtype)[:, None]
            predictions[:, j, paths[image], 0] = keep.sum(axis=1)
            predictions[:, j, paths[image], 1] = (keep & (det[None, :, 5] == 0)).sum(axis=1)
            predictions[:, j, paths[image], 2] = (keep & (det[None, :, 5] == 1)).sum(axis=1)
        print(f"NMS IoU {iou:.2f}: {j + 1}/{len(ious)} complete", flush=True)
    baseline_c, baseline_i = np.flatnonzero(np.isclose(confs, 0.25)), np.flatnonzero(np.isclose(ious, 0.7))
    if baseline_c.size and baseline_i.size and not np.array_equal(predictions[baseline_c[0], baseline_i[0]], reference):
        raise ValueError("Cached NMS counts differ from normal baseline predictions")
    candidates = []
    for i, conf in enumerate(confs):
        for j, iou in enumerate(ious):
            candidates.append(
                {
                    "conf": float(conf),
                    "iou": float(iou),
                    **evaluate_counts(true, predictions[i, j]),
                    "rmse_loss": None,
                    "mape_loss": None,
                    "maturity_loss": None,
                    "score": None,
                }
            )
    eligible = [row for row in candidates if row["eligible"]]
    if not eligible:
        raise ValueError("No eligible candidates: lower the confidence range or inspect images with zero predictions")
    losses = np.array([[row["rmse"], row["mape_percent"], 1 - row["maturity_r2"]] for row in eligible])
    low, high = losses.min(axis=0), losses.max(axis=0)
    normalized = np.divide(losses - low, high - low, out=np.zeros_like(losses), where=high > low)
    for row, values in zip(eligible, normalized):
        row.update(zip(("rmse_loss", "mape_loss", "maturity_loss"), values.tolist()))
        row["score"] = float(values.mean())
    best = min(eligible, key=lambda row: row["score"])
    best_i, best_j = (
        np.flatnonzero(np.isclose(confs, best["conf"]))[0],
        np.flatnonzero(np.isclose(ious, best["iou"]))[0],
    )
    best_counts = predictions[best_i, best_j]
    print(f"Verifying winning conf={best['conf']:.2f}, iou={best['iou']:.2f} with normal inference...", flush=True)
    verified = {}
    for result in model.predict(
        source=str(root / "images/val"), stream=True, conf=best["conf"], iou=best["iou"], **settings
    ):
        classes = result.boxes.cls.cpu().numpy().astype(int)
        verified[Path(result.path).resolve()] = [
            len(classes),
            np.count_nonzero(classes == 0),
            np.count_nonzero(classes == 1),
        ]
    if set(verified) != set(paths) or not np.array_equal(
        np.array([verified[p.resolve()] for p in images]), best_counts
    ):
        raise ValueError("Cached winning counts differ from normal predictions")
    output = args.output or Path(__file__).resolve().parent / "runs/threshold_search" / args.model.parent.parent.name
    output.mkdir(parents=True, exist_ok=True)
    with (output / "search_results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(candidates[0]))
        writer.writeheader()
        writer.writerows(sorted(candidates, key=lambda row: row["score"] if row["eligible"] else float("inf")))
    for name, pred in (("best_per_image.csv", best_counts), ("baseline_per_image.csv", reference)):
        with (output / name).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)
            writer.writerow(
                ["image", "total_true", "total_pred", "ripe_true", "ripe_pred", "unripe_true", "unripe_pred"]
            )
            writer.writerows(
                [image.name, *np.column_stack((gt, pp)).ravel().tolist()] for image, gt, pp in zip(images, true, pred)
            )
    summary = {
        "model": str(args.model.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "dataset": str(root),
        "split": "val",
        "images": len(images),
        "positive_gt_images": int(positive.sum()),
        "zero_gt_images": [image.name for image, valid in zip(images, positive) if not valid],
        "prediction": settings,
        "device": str(model.predictor.device),
        "grid": {
            "confs": confs.tolist(),
            "ious": ious.tolist(),
            "candidates": len(candidates),
            "eligible": len(eligible),
        },
        "objective": {
            "weights": [1 / 3, 1 / 3, 1 / 3],
            "losses": ["RMSE", "MAPE (%)", "1 - maturity R2"],
            "normalization": "(loss - min) / (max - min) over eligible grid candidates; constant loss maps to 0",
            "loss_min": low.tolist(),
            "loss_max": high.tolist(),
            "score": "mean of the three normalized losses",
        },
        "baseline": {"conf": 0.25, "iou": 0.7, **evaluate_counts(true, reference)},
        "best": best,
        "best_quantity_r2": r_squared(true[:, 0].tolist(), best_counts[:, 0].tolist()),
        "verified_against_normal_prediction": True,
    }
    (output / "search_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary["baseline"], indent=2), flush=True)
    print(json.dumps(summary["best"], indent=2), flush=True)
    print(f"Results saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
