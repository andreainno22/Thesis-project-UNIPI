"""
Failure-mode analysis on the gemini test set (P1).
====================================================================

Two independent analyses, both driven by the GT boxes already annotated
(single-class) and the fine-tuned model's predictions:

1. Merge detection (works today, no category annotation needed): for each
   image, greedily match predictions to GT at IoU>=0.5 (the mAP50 rule).
   Any GT box left unmatched (a false negative) is checked against every
   OTHER GT box in the same image; if it overlaps one (IoU >= --cluster-iou)
   that overlapping box, in turn, IS matched to a prediction, the miss is
   attributed to "fusion" (one predicted box likely covers both). Otherwise
   the miss is "isolated" (a genuine detection failure at that location).

2. Per-category recall (needs pipeline1/src/annotate_gemini_categories.py
   output): if a categories/<stem>.json sidecar exists for an image, GT
   boxes are grouped by category and matched/unmatched counts are
   aggregated into a per-category recall table.

Usage:
  python pipeline1/src/analyze_gemini_failures.py \
      --weights pipeline1/runs/yolo11n_final/weights/best.pt
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

VENUES = {
    "corridoi": ("Dataset/ostruzioni_gemini/corridoi/images/corridoi",
                 "Dataset/ostruzioni_gemini/corridoi/labels/corridoi",
                 "Dataset/ostruzioni_gemini/corridoi/categories"),
    "porte": ("Dataset/ostruzioni_gemini/porte/images/porte",
              "Dataset/ostruzioni_gemini/porte/labels/porte",
              "Dataset/ostruzioni_gemini/porte/categories"),
}


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_gt_xyxy(label_path: Path, w: int, h: int) -> list[list[float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cx, cy, bw, bh = (float(x) for x in parts[1:])
        boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
    return boxes


def load_categories(cat_path: Path) -> list[str | None]:
    if not cat_path.exists():
        return []
    data = json.loads(cat_path.read_text())["boxes"]
    return [b["category"] for b in data]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--conf", type=float, default=0.25,
                    help="operating confidence for matching (not for mAP)")
    ap.add_argument("--match-iou", type=float, default=0.5)
    ap.add_argument("--cluster-iou", type=float, default=0.1,
                    help="GT-GT overlap above which two boxes are considered "
                         "'stacked/adjacent' for the fusion heuristic")
    args = ap.parse_args()

    model = YOLO(args.weights)

    n_fn_fused, n_fn_loose, n_fn_isolated, n_tp, n_gt = 0, 0, 0, 0, 0
    per_cat = {}  # category -> [n_gt, n_tp]
    per_cat_fn = {}  # category -> [fused, loose, isolated]

    for venue, (img_dir, lbl_dir, cat_dir) in VENUES.items():
        img_paths = sorted(glob.glob(str(Path(img_dir) / "*.jpg")))
        for img_path in img_paths:
            img_path = Path(img_path)
            im = Image.open(img_path)
            w, h = im.size
            stem = img_path.stem
            gts = load_gt_xyxy(Path(lbl_dir) / f"{stem}.txt", w, h)
            if not gts:
                continue
            cats = load_categories(Path(cat_dir) / f"{stem}.json")
            if len(cats) != len(gts):
                cats = [None] * len(gts)

            r = model.predict(str(img_path), conf=args.conf, device="cpu", verbose=False)[0]
            preds = r.boxes.xyxy.cpu().numpy().tolist() if len(r.boxes) else []
            confs = r.boxes.conf.cpu().numpy().tolist() if len(r.boxes) else []
            order = sorted(range(len(preds)), key=lambda i: -confs[i])

            # greedy one-to-one match (mAP50 rule): each GT can be claimed by
            # at most one prediction, at the prediction's best-IoU GT.
            matched_gt: set[int] = set()
            matched_pred: dict[int, int] = {}  # pred idx -> gt idx
            for pi in order:
                best_j, best_iou = -1, 0.0
                for j, g in enumerate(gts):
                    if j in matched_gt:
                        continue
                    v = iou(preds[pi], g)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= args.match_iou:
                    matched_gt.add(best_j)
                    matched_pred[pi] = best_j

            n_gt += len(gts)
            n_tp += len(matched_gt)
            for j, cat in enumerate(cats):
                if cat is None:
                    continue
                per_cat.setdefault(cat, [0, 0])
                per_cat[cat][0] += 1
                if j in matched_gt:
                    per_cat[cat][1] += 1

            # For each missed GT, find the prediction (any, not just
            # unmatched) that covers it best - the direct test of "did the
            # model draw one box spanning several GT instances".
            for j in range(len(gts)):
                if j in matched_gt:
                    continue
                best_pi, best_pred_iou = -1, 0.0
                for pi, pbox in enumerate(preds):
                    v = iou(gts[j], pbox)
                    if v > best_pred_iou:
                        best_pred_iou, best_pi = v, pi
                cat = cats[j] if j < len(cats) else None
                if cat is not None:
                    per_cat_fn.setdefault(cat, [0, 0, 0])
                if best_pred_iou >= args.cluster_iou and best_pi in matched_pred:
                    n_fn_fused += 1  # a prediction covers it AND already claimed another GT
                    if cat is not None:
                        per_cat_fn[cat][0] += 1
                elif best_pred_iou >= args.cluster_iou:
                    n_fn_loose += 1  # a prediction overlaps it but matched nothing (imprecise/low-conf box)
                    if cat is not None:
                        per_cat_fn[cat][1] += 1
                else:
                    n_fn_isolated += 1  # no prediction anywhere near this GT
                    if cat is not None:
                        per_cat_fn[cat][2] += 1

    print("== merge-detection analysis (geometric, no categories needed) ==")
    print(f"  GT instances       : {n_gt}")
    print(f"  matched (TP)       : {n_tp}")
    print(f"  FN, fused into a box claimed by another GT (stacking suspect) : {n_fn_fused}")
    print(f"  FN, covered by an unmatched/loose prediction                 : {n_fn_loose}")
    print(f"  FN, isolated - no prediction nearby at all                   : {n_fn_isolated}")

    if per_cat:
        print("\n== per-category recall (from annotate_gemini_categories.py) ==")
        for cat, (tot, tp) in sorted(per_cat.items()):
            print(f"  {cat:12s}: {tp}/{tot}  (recall {tp / tot:.3f})")

        print("\n== per-category FN breakdown (fused / loose / isolated) ==")
        for cat, (fused, loose, isolated) in sorted(per_cat_fn.items()):
            tot_fn = fused + loose + isolated
            print(f"  {cat:12s}: fused={fused} loose={loose} isolated={isolated}  (total FN {tot_fn})")
    else:
        print("\nNo category sidecars found yet - run "
              "pipeline1/src/annotate_gemini_categories.py first for the "
              "per-category breakdown.")


if __name__ == "__main__":
    main()
