"""
Import door-frame ROI annotations (JSON) into the SQLite database.
====================================================================

Reads every `*.json` file produced by `annotate_door_roi.py` from a
directory and writes its polygons into the `roi_polygons` table,
matching each JSON to a frame via the relative `file_path` stored in
`frames`.

The matching strategy is:
  1. The JSON contains `image_path` (absolute path of the annotated image).
  2. We compute `rel = image_path.relative_to(dataset_root)`.
  3. We look up `frames` where `file_path == rel` (with forward slashes).

If a JSON references an image not present in `frames`, a warning is
printed and that JSON is skipped (no auto-create). To make a new image
visible, run `load_dataset_to_db.py` first.

By default, importing a JSON REPLACES any existing `roi_polygons` rows
for that frame. Pass `--append` to keep the old rows.

Usage:
    python import_roi_to_db.py \
        --db Aggregated_dataset_db/occlusion.db \
        --roi-dir Dataset/roi \
        --dataset-root Dataset
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def to_rel_posix(path: Path, root: Path) -> str | None:
    """Return path relative to root using forward slashes, or None if outside."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return rel.as_posix()


def import_one(
    conn: sqlite3.Connection,
    json_path: Path,
    dataset_root: Path,
    append: bool,
) -> tuple[int, str]:
    """Import a single JSON. Returns (n_polys_inserted, status_msg)."""
    with json_path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    image_path = Path(d.get("image_path", ""))
    if not image_path.is_absolute():
        image_path = (json_path.parent / image_path).resolve()
    rel = to_rel_posix(image_path, dataset_root)
    if rel is None:
        return 0, f"SKIP (path outside dataset_root): {image_path}"

    row = conn.execute(
        "SELECT frame_id FROM frames WHERE file_path = ?", (rel,)
    ).fetchone()
    if row is None:
        return 0, f"SKIP (no frame for {rel})"
    frame_id = row[0]

    img_w = int(d.get("img_w", 0))
    img_h = int(d.get("img_h", 0))
    polygons = d.get("polygons", []) or []
    if img_w <= 0 or img_h <= 0:
        return 0, f"SKIP (invalid img dims) in {json_path.name}"

    if not append:
        conn.execute("DELETE FROM roi_polygons WHERE frame_id = ?", (frame_id,))

    n_inserted = 0
    for entry in polygons:
        label = entry.get("label")
        pts = entry.get("polygon")
        if not label or not pts or len(pts) < 3:
            continue
        conn.execute(
            "INSERT INTO roi_polygons "
            "(frame_id, label, polygon, img_w, img_h, source_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                frame_id, label, json.dumps(pts), img_w, img_h,
                str(json_path.name),
            ),
        )
        n_inserted += 1
    return n_inserted, f"OK {frame_id} ({n_inserted} polys)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import ROI polygons (JSON) into the SQLite DB.",
    )
    parser.add_argument("--db", required=True, help="Path to occlusion.db")
    parser.add_argument(
        "--roi-dir", required=True,
        help="Directory containing *.json ROI files.",
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Root of the Dataset/ folder (used to compute relative paths).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing polygons instead of replacing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    db_path = Path(args.db)
    roi_dir = Path(args.roi_dir)
    dataset_root = Path(args.dataset_root)

    if not db_path.exists():
        print(f"[ERR] DB not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not roi_dir.is_dir():
        print(f"[ERR] roi-dir not found: {roi_dir}", file=sys.stderr)
        sys.exit(1)
    if not dataset_root.is_dir():
        print(f"[ERR] dataset-root not found: {dataset_root}", file=sys.stderr)
        sys.exit(1)

    jsons = sorted(roi_dir.glob("*.json"))
    if not jsons:
        print(f"[WARN] no JSON files in {roi_dir}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    total_polys = 0
    total_frames = 0
    skipped = 0
    try:
        for jp in jsons:
            n, msg = import_one(conn, jp, dataset_root, append=args.append)
            print(f"  [{jp.name}] {msg}")
            if n > 0:
                total_polys += n
                total_frames += 1
            else:
                skipped += 1
        conn.commit()
    finally:
        conn.close()

    print()
    print(f"[DONE] frames updated: {total_frames}, "
          f"polygons inserted: {total_polys}, skipped JSONs: {skipped}")


if __name__ == "__main__":
    main()
