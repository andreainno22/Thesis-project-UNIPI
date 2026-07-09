"""
Fine-tune a pretrained YOLO on the single-class obstruction dataset (P1).
====================================================================

Trains either a single data.yaml or loops over all CV folds produced by
prepare_yolo_dataset.py. Weights land in <project>/<name>/weights/best.pt.

Examples:
  # one fold
  python pipeline1/src/train_yolo.py --data pipeline1/data/fold0.yaml \
      --model yolo11n.pt --epochs 100 --imgsz 640 --name fold0

  # all 5 folds (cross-validation)
  python pipeline1/src/train_yolo.py --folds-dir pipeline1/data \
      --model yolo11n.pt --epochs 100 --name cv

  # final model on all data
  python pipeline1/src/train_yolo.py --data pipeline1/data/all.yaml \
      --model yolo11n.pt --epochs 100 --name final

Edge target (Raspberry Pi / Jetson Nano): keep --model at nano/small sizes
(yolo11n.pt / yolo11s.pt). See Miglionico et al. (IJCNN 2025).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def train_one(data: Path, args, name: str) -> Path:
    model = YOLO(args.model)
    model.train(
        data=str(data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        freeze=args.freeze,
        device=args.device,
        project=str(Path(args.project).resolve()),
        name=name,
        exist_ok=True,
        seed=args.seed,
        # single-class problem: let YOLO treat all boxes as one class
        single_cls=True,
        verbose=True,
    )
    best = Path(args.project) / name / "weights" / "best.pt"
    print(f"[done] {name}: {best}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="single data.yaml")
    src.add_argument("--folds-dir", help="dir with fold0.yaml ... foldK.yaml")

    ap.add_argument("--model", default="yolo11n.pt",
                    help="pretrained weights (COCO). nano/small for edge.")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1,
                    help="fixed batch size, or -1 for Ultralytics AutoBatch "
                         "(picks the largest batch that fits ~60%% of free "
                         "GPU memory - use this when the GPU is unknown).")
    ap.add_argument("--patience", type=int, default=100,
                    help="early-stop if val fitness doesn't improve for N "
                         "epochs (default 100 = effectively no early stop).")
    ap.add_argument("--freeze", type=int, default=None,
                    help="freeze the first N backbone layers (COCO features) "
                         "instead of fine-tuning the whole net. Try this only "
                         "if CV shows overfitting (val loss rising while "
                         "train loss keeps dropping).")
    ap.add_argument("--device", default=None,
                    help="'0' for GPU, 'cpu', or None to auto-pick")
    ap.add_argument("--project", default="pipeline1/runs")
    ap.add_argument("--name", default="train")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.data:
        train_one(Path(args.data), args, args.name)
    else:
        fold_yamls = sorted(Path(args.folds_dir).glob("fold*.yaml"))
        if not fold_yamls:
            raise SystemExit(f"no fold*.yaml in {args.folds_dir}")
        print(f"cross-validation over {len(fold_yamls)} folds")
        for y in fold_yamls:
            k = y.stem.replace("fold", "")
            train_one(y, args, f"{args.name}_fold{k}")


if __name__ == "__main__":
    main()
