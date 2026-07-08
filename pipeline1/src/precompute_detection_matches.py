"""
Precompute, for every pasted object in the copy-paste dataset, WHICH COCO class
the detector assigns to it (or MISSED). Object-level analysis via IoU matching
between the known pasted-object boxes (objects_metadata.json) and the raw COCO
detections. Output feeds the confusion-matrix notebook (detection_analysis.ipynb).

Heavy step (runs the model on all 990 positives), so it caches to JSON; the
notebook just loads the cache. Re-run this to refresh.

Usage:
  python pipeline1/src/precompute_detection_matches.py --model yolo11n.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.3,
                    help="min IoU to say a detection matches a pasted object "
                         "(lenient: captures loosely-aligned wrong-class boxes)")
    ap.add_argument("--dataset-root", default="Dataset")
    ap.add_argument("--out",
                    default="pipeline1/results/copypaste_detection_matches.json")
    args = ap.parse_args()

    root = Path(args.dataset_root) / "ostruzioni_reali"
    meta = json.loads((root / "objects_metadata.json").read_text(encoding="utf-8"))["images"]
    images_dir = root / "images"

    model = YOLO(args.model)
    names = model.names

    records = []
    total = len(meta)
    for i, (img_name, objs) in enumerate(meta.items(), 1):
        path = images_dir / img_name
        if not path.exists():
            continue
        r = model.predict(str(path), conf=args.conf, verbose=False)[0]
        # raw detections: (coco_class_name, conf, xyxy-normalized)
        dets = []
        if len(r.boxes):
            cls = r.boxes.cls.tolist()
            conf = r.boxes.conf.tolist()
            xywhn = r.boxes.xywhn.tolist()
            for c, cf, (cx, cy, w, h) in zip(cls, conf, xywhn):
                dets.append((names[int(c)], float(cf), to_xyxy(cx, cy, w, h)))

        for o in objs:
            gt = to_xyxy(*o["bbox_yolo"])
            best_iou, best_cls, best_conf = 0.0, "MISSED", 0.0
            for cn, cf, box in dets:
                iv = iou(gt, box)
                if iv > best_iou and iv >= args.iou_thr:
                    best_iou, best_cls, best_conf = iv, cn, cf
            records.append({
                "image": img_name,
                "true_category": o["category"],
                "cut": bool(o["cut_edge"]),
                "pred_class": best_cls,
                "iou": round(best_iou, 3),
                "conf": round(best_conf, 3),
            })
        if i % 50 == 0 or i == total:
            print(f"  [{i}/{total}] images processed")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_doc = {
        "_meta": {"model": args.model, "conf": args.conf, "iou_thr": args.iou_thr},
        "records": records,
    }
    out.write_text(json.dumps(meta_doc), encoding="utf-8")
    print(f"\nwrote {len(records)} object records to {out}")


if __name__ == "__main__":
    main()
