"""
Evaluate a torchvision Faster R-CNN (COCO-pretrained) baseline for P1.
====================================================================

Same image-level protocol as evaluate_yolo.py (obstructed/clear classifier,
COCO->obstruction class mapping from coco_obstruction_map.py), but using
torchvision's detection API instead of ultralytics, since Faster R-CNN is not
a YOLO model. Sample listing (DB / copy-paste) and the CSV/DB writers are
reused directly from evaluate_yolo.py.

Two backbone variants available via --variant:
  resnet50   : fasterrcnn_resnet50_fpn        - accurate, VERY slow on CPU
               (~18 s/image measured on this machine: 990+110 img would take
               ~5-6h, so by default this variant only runs on small sets like
               the poli test - use --copy-paste at your own risk/time budget).
  mobilenet  : fasterrcnn_mobilenet_v3_large_320_fpn - "mobile/edge" variant,
               lower input resolution (320px), designed for speed. Much
               faster on CPU, more comparable to yolo11n/yolo26n for an
               edge-relevant comparison.

Usage:
  python pipeline1/src/evaluate_rcnn.py --variant resnet50 --split test \
      --out-csv pipeline1/results/rcnn_resnet50_poli_imagelevel.csv

  python pipeline1/src/evaluate_rcnn.py --variant mobilenet --split test \
      --out-csv pipeline1/results/rcnn_mobilenet_poli_imagelevel.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torchvision.io import ImageReadMode, read_image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
    fasterrcnn_resnet50_fpn,
)
from torchvision.transforms.functional import convert_image_dtype

sys.path.insert(0, str(Path(__file__).parent))
from coco_obstruction_map import MIN_BOX_AREA_FRAC, OBSTRUCTION_COCO_NAMES  # noqa: E402
from evaluate_yolo import (  # noqa: E402
    _write_csv,
    samples_copypaste,
    samples_from_db,
)

VARIANTS = {
    "resnet50": (fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights.DEFAULT),
    "mobilenet": (fasterrcnn_mobilenet_v3_large_320_fpn,
                  FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT),
}


def load_model(variant: str, conf: float):
    build_fn, weights = VARIANTS[variant]
    # box_score_thresh filtra gia' dentro al modello le detection deboli,
    # cosi' non dobbiamo rifarlo a mano (equivalente al parametro conf di YOLO).
    model = build_fn(weights=weights, box_score_thresh=conf)
    model.eval()
    categories = weights.meta["categories"]  # lista nomi, indice = classe
    obstruction_ids = {i for i, n in enumerate(categories) if n in OBSTRUCTION_COCO_NAMES}
    return model, obstruction_ids


def predict_is_obstructed(model, obstruction_ids: set[int], img_path: Path,
                          min_area: float) -> bool:
    img = read_image(str(img_path), mode=ImageReadMode.RGB)
    img = convert_image_dtype(img, torch.float32)
    _, h, w = img.shape
    with torch.no_grad():
        out = model([img])[0]
    boxes = out["boxes"].tolist()   # [x1,y1,x2,y2] in pixel, gia' filtrate per score
    labels = out["labels"].tolist()
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        area_frac = (x2 - x1) * (y2 - y1) / (w * h)
        if label in obstruction_ids and area_frac >= min_area:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=list(VARIANTS), default="resnet50")
    ap.add_argument("--db", default="Aggregated_dataset_db/occlusion.db")
    ap.add_argument("--dataset-root", default="Dataset")
    ap.add_argument("--split", default=None, help="DB split, e.g. 'test'")
    ap.add_argument("--copy-paste", action="store_true",
                    help="evaluate on the copy-paste set (990+110 img - SLOW "
                         "with --variant resnet50, ~5-6h on CPU)")
    ap.add_argument("--pruned-file",
                    default="Dataset/ostruzioni_reali/pruned_backgrounds.txt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-area", type=float, default=MIN_BOX_AREA_FRAC)
    ap.add_argument("--out-csv", default="pipeline1/results/rcnn_imagelevel.csv")
    ap.add_argument("--no-db", action="store_true")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    if args.copy_paste:
        samples = samples_copypaste(args.db, dataset_root, Path(args.pruned_file))
    elif args.split:
        samples = samples_from_db(args.db, dataset_root, args.split)
    else:
        raise SystemExit("use --copy-paste or --split")

    print(f"loading fasterrcnn ({args.variant}) ...")
    t0 = time.time()
    model, obstruction_ids = load_model(args.variant, args.conf)
    print(f"loaded in {time.time() - t0:.1f}s, "
          f"{len(obstruction_ids)} obstruction classes, "
          f"{len(samples)} images to evaluate")

    tp = fp = tn = fn = 0
    missing = 0
    t_start = time.time()
    for i, (img, is_pos) in enumerate(samples, 1):
        if not img.exists():
            missing += 1
            continue
        pred = predict_is_obstructed(model, obstruction_ids, img, args.min_area)
        if is_pos and pred:
            tp += 1
        elif is_pos and not pred:
            fn += 1
        elif not is_pos and pred:
            fp += 1
        else:
            tn += 1
        if i % 10 == 0 or i == len(samples):
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(samples) - i)
            print(f"  [{i}/{len(samples)}] {elapsed:.0f}s elapsed, "
                  f"ETA {eta:.0f}s ({eta/60:.1f} min)")

    n_pos, n_neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    fnr = fn / n_pos if n_pos else 0.0

    metrics = {
        "model": f"fasterrcnn-{args.variant}",
        "n_pos": n_pos, "n_neg": n_neg, "conf": args.conf,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "F1": round(f1, 4),
        "FPR": round(fpr, 4), "FNR": round(fnr, 4),
    }
    if missing:
        metrics["missing_files"] = missing

    print("\n== image-level metrics ==")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v}")

    _write_csv(Path(args.out_csv), [metrics])
    if not args.no_db:
        _write_db(args, metrics)


def _write_db(args, metrics: dict) -> None:
    import json
    import sqlite3
    dataset = "copy-paste" if args.copy_paste else args.split
    conn = sqlite3.connect(args.db)
    cur = conn.execute(
        "INSERT INTO experiments (pipeline, model_variant, dataset_filter, "
        "hyperparams, status) VALUES ('P1', ?, ?, ?, 'done')",
        (f"image-level:{metrics['model']}",
         json.dumps({"dataset": dataset, "n_pos": metrics["n_pos"],
                     "n_neg": metrics["n_neg"]}),
         json.dumps({"conf": args.conf, "min_area": args.min_area,
                     "metrics": metrics})),
    )
    exp_id = cur.lastrowid
    conn.execute(
        "INSERT INTO results (exp_id, precision, recall, F1, "
        "false_alarm_rate, false_negative_rate) VALUES (?, ?, ?, ?, ?, ?)",
        (exp_id, metrics["precision"], metrics["recall"], metrics["F1"],
         metrics["FPR"], metrics["FNR"]),
    )
    conn.commit()
    conn.close()
    print(f"wrote experiment {exp_id} (P1, image-level) to DB")


if __name__ == "__main__":
    main()
