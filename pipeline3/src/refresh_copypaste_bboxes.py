"""
Refresh the annotations_bbox rows for the (re)generated copy-paste dataset.
====================================================================

After regenerating Dataset/ostruzioni_reali with generate_synthetic_dataset.py
the old bounding boxes in the DB are stale. This script replaces them, reading
the new YOLO label files and - if objects_metadata.json is available - storing
the pasted object CATEGORY (carrello/sedia/scatola/barella/cestino) in
annotations_bbox.label_class instead of the generic 'obstacle'.

Only annotations_bbox is touched: frames rows (splits, source_group, ...) are
left untouched, since filenames are unchanged between v1 and v2.

Usage:
  python pipeline3/src/refresh_copypaste_bboxes.py \
      --db Aggregated_dataset_db/occlusion.db --dataset-root Dataset
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default="Aggregated_dataset_db/occlusion.db")
    ap.add_argument("--dataset-root", default="Dataset")
    args = ap.parse_args()

    root = Path(args.dataset_root) / "ostruzioni_reali"
    labels_dir = root / "labels"
    meta_path = root / "objects_metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))["images"]
        print(f"metadata: {len(metadata)} images")
    else:
        print("[warn] objects_metadata.json not found: label_class='obstacle'")

    conn = sqlite3.connect(args.db)
    frame_ids = [r[0] for r in conn.execute(
        "SELECT frame_id FROM frames WHERE occlusion_type = 'synthetic_copypaste'"
    )]
    print(f"copy-paste frames in DB: {len(frame_ids)}")

    old = conn.execute(
        "SELECT COUNT(*) FROM annotations_bbox WHERE frame_id IN "
        "(SELECT frame_id FROM frames WHERE occlusion_type='synthetic_copypaste')"
    ).fetchone()[0]

    inserted = 0
    missing_labels = 0
    misaligned = 0
    for fid in frame_ids:
        img_name = Path(fid).name                      # corridoio_001_ostruita.jpg
        label_path = labels_dir / (Path(fid).stem + ".txt")
        conn.execute("DELETE FROM annotations_bbox WHERE frame_id = ?", (fid,))
        if not label_path.exists():
            missing_labels += 1
            continue

        lines = [ln for ln in label_path.read_text().strip().splitlines() if ln]
        objs = metadata.get(img_name, [])
        if objs and len(objs) != len(lines):
            misaligned += 1
            objs = []                                   # fallback: generic class

        txt_rel = f"ostruzioni_reali/labels/{label_path.name}"
        for i, ln in enumerate(lines):
            parts = ln.split()
            cx, cy, w, h = (float(v) for v in parts[1:5])
            label_class = objs[i]["category"] if objs else "obstacle"
            conn.execute(
                "INSERT INTO annotations_bbox (frame_id, label_class, cx, cy, w, h, "
                "txt_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fid, label_class, cx, cy, w, h, txt_rel),
            )
            inserted += 1

    conn.commit()

    print(f"bboxes: {old} old deleted -> {inserted} inserted")
    if missing_labels:
        print(f"[warn] {missing_labels} frames without label file")
    if misaligned:
        print(f"[warn] {misaligned} frames metadata/label misaligned (generic class used)")
    for cls, n in conn.execute(
        "SELECT label_class, COUNT(*) FROM annotations_bbox WHERE frame_id IN "
        "(SELECT frame_id FROM frames WHERE occlusion_type='synthetic_copypaste') "
        "GROUP BY label_class ORDER BY label_class"
    ):
        print(f"  {cls:10s}: {n}")
    conn.close()


if __name__ == "__main__":
    main()
