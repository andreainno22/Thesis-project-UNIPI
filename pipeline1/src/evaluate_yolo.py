"""
Evaluate P1 obstruction detection: detection mAP (CV) + image-level metrics.
====================================================================

Two sub-commands:

  image-level : run a detector over a positive set and a negative set and
                score it as a binary "obstructed / clear" classifier
                (accuracy, precision, recall, F1, FPR, FNR). This is the
                metric shared with P3 for cross-pipeline comparison.
                Works for a fine-tuned single-class model OR a COCO baseline
                (--weights coco enables the COCO->obstruction mapping).

  cv-map      : aggregate detection mAP@0.5 and mAP@0.5:0.95 across the CV
                folds trained by train_yolo.py (needs the copy-paste labels).

Examples:
  # baseline (non fine-tuned COCO) on the poli real test set, from the DB
  python pipeline1/src/evaluate_yolo.py image-level \
      --weights coco --split test --conf 0.25 \
      --out-csv pipeline1/results/baseline_imagelevel.csv

  # fine-tuned model on the same test set
  python pipeline1/src/evaluate_yolo.py image-level \
      --weights pipeline1/runs/final/weights/best.pt --split test \
      --out-csv pipeline1/results/finetuned_imagelevel.csv

  # cross-validation detection mAP
  python pipeline1/src/evaluate_yolo.py cv-map \
      --runs-dir pipeline1/runs --name cv --folds-dir pipeline1/data \
      --out-csv pipeline1/results/cv_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from coco_obstruction_map import (  # noqa: E402
    MIN_BOX_AREA_FRAC,
    is_obstruction_detection,
    obstruction_class_ids,
)


# --------------------------------------------------------------------------- #
# sample listing
# --------------------------------------------------------------------------- #
def samples_from_db(db: str, dataset_root: Path, split: str) -> list[tuple[Path, int]]:
    """Return [(image_path, is_positive)] for a DB split (positive = obstructed)."""
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT file_path, is_normal FROM frames WHERE split = ?", (split,)
    ).fetchall()
    conn.close()
    out = []
    for file_path, is_normal in rows:
        out.append((dataset_root / file_path, 0 if is_normal else 1))
    return out


def samples_from_dirs(pos_dir: Path | None, neg_dir: Path | None) -> list[tuple[Path, int]]:
    exts = {".jpg", ".jpeg", ".png"}
    out: list[tuple[Path, int]] = []
    for d, lab in ((pos_dir, 1), (neg_dir, 0)):
        if d is None:
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                out.append((p, lab))
    return out


# --------------------------------------------------------------------------- #
# image-level scoring
# --------------------------------------------------------------------------- #
def predict_is_obstructed(
    model: YOLO,
    img: Path,
    conf: float,
    coco_map: bool,
    obstruction_ids: set[int],
    min_area: float,
) -> bool:
    r = model.predict(str(img), conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return False
    if not coco_map:
        return True  # single-class model: any detection = obstruction
    # baseline: keep only mapped classes above the area threshold
    cls = r.boxes.cls.tolist()
    wh = r.boxes.xywhn[:, 2:4].tolist()  # normalized w,h -> area frac = w*h
    for c, (w, h) in zip(cls, wh):
        if is_obstruction_detection(int(c), w * h, obstruction_ids, min_area):
            return True
    return False


def image_level(args) -> None:
    dataset_root = Path(args.dataset_root)
    if args.split:
        samples = samples_from_db(args.db, dataset_root, args.split)
    else:
        samples = samples_from_dirs(
            Path(args.pos_dir) if args.pos_dir else None,
            Path(args.neg_dir) if args.neg_dir else None,
        )
    if not samples:
        raise SystemExit("no samples found (check --split or --pos/neg-dir)")

    coco_map = args.weights.lower() == "coco"
    weights = "yolo11n.pt" if coco_map else args.weights
    model = YOLO(weights)
    obstruction_ids = obstruction_class_ids(model.names) if coco_map else set()
    if coco_map:
        print(f"baseline COCO mapping -> obstruction ids {sorted(obstruction_ids)}, "
              f"min area frac {args.min_area}")

    tp = fp = tn = fn = 0
    missing = 0
    for img, is_pos in samples:
        if not img.exists():
            missing += 1
            continue
        pred = predict_is_obstructed(
            model, img, args.conf, coco_map, obstruction_ids, args.min_area
        )
        if is_pos and pred:
            tp += 1
        elif is_pos and not pred:
            fn += 1
        elif not is_pos and pred:
            fp += 1
        else:
            tn += 1

    n_pos, n_neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_pos if n_pos else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    fpr = fp / n_neg if n_neg else 0.0
    fnr = fn / n_pos if n_pos else 0.0

    metrics = {
        "model": "coco-baseline" if coco_map else Path(args.weights).name,
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
    if args.write_db:
        _write_db(args.db, "image-level", metrics["model"], args, {
            "precision": precision, "recall": recall, "F1": f1,
            "false_alarm_rate": fpr, "false_negative_rate": fnr,
        })


# --------------------------------------------------------------------------- #
# cross-validation detection mAP
# --------------------------------------------------------------------------- #
def cv_map(args) -> None:
    runs_dir = Path(args.runs_dir)
    folds_dir = Path(args.folds_dir)
    fold_yamls = sorted(folds_dir.glob("fold*.yaml"))
    if not fold_yamls:
        raise SystemExit(f"no fold*.yaml in {folds_dir}")

    per_fold = []
    for y in fold_yamls:
        k = y.stem.replace("fold", "")
        best = runs_dir / f"{args.name}_fold{k}" / "weights" / "best.pt"
        if not best.exists():
            print(f"[skip] fold {k}: missing {best}")
            continue
        model = YOLO(str(best))
        res = model.val(data=str(y.resolve()), split="val", verbose=False)
        row = {
            "fold": k,
            "mAP50": round(res.box.map50, 4),
            "mAP50_95": round(res.box.map, 4),
            "precision": round(res.box.mp, 4),
            "recall": round(res.box.mr, 4),
        }
        per_fold.append(row)
        print(f"  fold {k}: mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}  "
              f"P={row['precision']}  R={row['recall']}")

    if not per_fold:
        raise SystemExit("no fold weights found - train first")

    def mean(key):
        return sum(r[key] for r in per_fold) / len(per_fold)

    def std(key):
        m = mean(key)
        return (sum((r[key] - m) ** 2 for r in per_fold) / len(per_fold)) ** 0.5

    summary = {"fold": "mean+-std"}
    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        summary[key] = f"{mean(key):.4f}+-{std(key):.4f}"
    per_fold.append(summary)
    print(f"\n== CV mAP over {len(per_fold) - 1} folds ==")
    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        print(f"  {key:10s}: {summary[key]}")

    _write_csv(Path(args.out_csv), per_fold)


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def _write_db(db: str, variant: str, model_name: str, args, metrics: dict) -> None:
    conn = sqlite3.connect(db)
    cur = conn.execute(
        "INSERT INTO experiments (pipeline, model_variant, dataset_filter, "
        "hyperparams, status) VALUES ('P1', ?, ?, ?, 'done')",
        (f"{variant}:{model_name}",
         json.dumps({"split": args.split}),
         json.dumps({"conf": args.conf})),
    )
    exp_id = cur.lastrowid
    conn.execute(
        "INSERT INTO results (exp_id, precision, recall, F1, "
        "false_alarm_rate, false_negative_rate) VALUES (?, ?, ?, ?, ?, ?)",
        (exp_id, metrics["precision"], metrics["recall"], metrics["F1"],
         metrics["false_alarm_rate"], metrics["false_negative_rate"]),
    )
    conn.commit()
    conn.close()
    print(f"wrote experiment {exp_id} to DB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("image-level", help="binary obstructed/clear metrics")
    a.add_argument("--weights", required=True,
                   help="path to .pt, or 'coco' for the mapped baseline")
    a.add_argument("--db", default="Aggregated_dataset_db/occlusion.db")
    a.add_argument("--dataset-root", default="Dataset")
    a.add_argument("--split", default=None, help="DB split, e.g. 'test'")
    a.add_argument("--pos-dir", default=None)
    a.add_argument("--neg-dir", default=None)
    a.add_argument("--conf", type=float, default=0.25)
    a.add_argument("--min-area", type=float, default=MIN_BOX_AREA_FRAC)
    a.add_argument("--out-csv", default="pipeline1/results/image_level.csv")
    a.add_argument("--write-db", action="store_true")
    a.set_defaults(func=image_level)

    b = sub.add_parser("cv-map", help="cross-validation detection mAP")
    b.add_argument("--runs-dir", default="pipeline1/runs")
    b.add_argument("--name", default="cv")
    b.add_argument("--folds-dir", default="pipeline1/data")
    b.add_argument("--out-csv", default="pipeline1/results/cv_map.csv")
    b.set_defaults(func=cv_map)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
