"""Load the poli_ingegneria real test set into the SQLite DB.

This dataset has a structure not covered by load_dataset_to_db.py:

  Dataset/non_ostruite/poli ingegneria/
      porta_X_ref.jpg     -> reference image for door X (PatchCore memory bank)
      porta_X_N.jpg       -> additional normal shots of door X (FPR negatives)
  Dataset/ostruzioni_poli_ingegneria/
      porta_X_N.jpg       -> real obstructed shots of door X (positives)
  Dataset/roi/
      porta_X_ref.json    -> 5 door-frame polygons annotated on the reference

DB mapping (decided with the user):
  - porta_X_ref.jpg      : is_normal=1, source='poli_ingegneria', reference_frame_id=NULL
                           (the per-door reference; evaluate-db builds the bank from it,
                            ROI gating reads polygons from it)
  - porta_X_N.jpg normal : is_normal=1, source='background_change', reference_frame_id=ref
                           (negatives; picked up by evaluate-db default --bg-source)
  - porta_X_N.jpg obstr  : is_normal=0, source='poli_ingegneria', occlusion_type='real',
                           reference_frame_id=ref
  - source_group         : 'poli_ingegneria' for all
  - split                : 'test' for all
  - ROI                  : the 5 polygons of each JSON go onto the porta_X_ref.jpg frame.
                           Matching is by JSON filename stem (the internal image_path is
                           inconsistent across files), so it is robust.

Idempotent: frames are upserted; ROI rows for the reference frames are replaced.

Usage:
    python src/load_poli_to_db.py
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "Aggregated_dataset_db" / "occlusion.db"
DATA = ROOT / "Dataset"

REF_DIR   = DATA / "non_ostruite" / "poli ingegneria"
OBSTR_DIR = DATA / "ostruzioni_poli_ingegneria"
ROI_DIR   = DATA / "roi"

VENUE        = "porta"
SOURCE_GROUP = "poli_ingegneria"
SOURCE_REF   = "poli_ingegneria"      # reference + obstructed
SOURCE_BG    = "background_change"    # numbered normals -> default --bg-source of evaluate-db
SPLIT        = "test"
EXTS         = (".jpg", ".jpeg", ".png")

UPSERT_FRAME = (
    "INSERT INTO frames "
    "(frame_id, file_path, venue_type, is_normal, obstacle_class, "
    "occlusion_type, occlusion_level, split, source, source_group, "
    "reference_frame_id) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
    "ON CONFLICT(frame_id) DO UPDATE SET "
    "file_path=excluded.file_path, venue_type=excluded.venue_type, "
    "is_normal=excluded.is_normal, obstacle_class=excluded.obstacle_class, "
    "occlusion_type=excluded.occlusion_type, occlusion_level=excluded.occlusion_level, "
    "split=excluded.split, source=excluded.source, source_group=excluded.source_group, "
    "reference_frame_id=excluded.reference_frame_id"
)


def rel(p: Path) -> str:
    return p.relative_to(DATA).as_posix()


def door_of(stem: str) -> str:
    """porta_3_ref / porta_3_5 -> 'porta_3'."""
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "porta":
        raise ValueError(f"unexpected filename stem: {stem}")
    return f"{parts[0]}_{parts[1]}"


def is_ref(stem: str) -> bool:
    return stem.split("_")[-1] == "ref"


def main() -> None:
    for d in (REF_DIR, OBSTR_DIR, ROI_DIR):
        if not d.is_dir():
            sys.exit(f"Missing directory: {d}")

    # Backup
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = DB.with_name(f"occlusion.backup_{stamp}.db")
    shutil.copy2(DB, backup)
    print(f"Backup: {backup}")

    ref_imgs = sorted(p for p in REF_DIR.iterdir() if p.suffix.lower() in EXTS)
    obstr_imgs = sorted(p for p in OBSTR_DIR.iterdir() if p.suffix.lower() in EXTS)

    # door -> reference frame_id
    ref_map: dict[str, str] = {}
    for p in ref_imgs:
        if is_ref(p.stem):
            ref_map[door_of(p.stem)] = rel(p)

    missing_ref = {door_of(p.stem) for p in (ref_imgs + obstr_imgs)} - set(ref_map)
    if missing_ref:
        sys.exit(f"Doors without a _ref image: {sorted(missing_ref)}")

    ref_rows, normal_rows, obstr_rows = [], [], []

    for p in ref_imgs:
        door = door_of(p.stem)
        r = rel(p)
        if is_ref(p.stem):
            ref_rows.append((r, r, VENUE, 1, None, "none", "none",
                             SPLIT, SOURCE_REF, SOURCE_GROUP, None))
        else:
            normal_rows.append((r, r, VENUE, 1, None, "none", "none",
                                SPLIT, SOURCE_BG, SOURCE_GROUP, ref_map[door]))

    for p in obstr_imgs:
        door = door_of(p.stem)
        r = rel(p)
        obstr_rows.append((r, r, VENUE, 0, "obstacle", "real", "partial",
                           SPLIT, SOURCE_REF, SOURCE_GROUP, ref_map[door]))

    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        # References first (FK parents), then children.
        conn.executemany(UPSERT_FRAME, ref_rows)
        conn.executemany(UPSERT_FRAME, normal_rows + obstr_rows)

        # ROI -> reference frames, matched by JSON filename stem.
        roi_inserted = 0
        roi_frames = 0
        for jp in sorted(ROI_DIR.glob("*.json")):
            door = door_of(jp.stem)
            if door not in ref_map:
                print(f"  ROI skip (no door): {jp.name}")
                continue
            frame_id = ref_map[door]
            d = json.load(jp.open(encoding="utf-8"))
            img_w, img_h = int(d.get("img_w", 0)), int(d.get("img_h", 0))
            polys = d.get("polygons", []) or []
            if img_w <= 0 or img_h <= 0 or not polys:
                print(f"  ROI skip (invalid): {jp.name}")
                continue
            conn.execute("DELETE FROM roi_polygons WHERE frame_id = ?", (frame_id,))
            n = 0
            for entry in polys:
                label, pts = entry.get("label"), entry.get("polygon")
                if not label or not pts or len(pts) < 3:
                    continue
                conn.execute(
                    "INSERT INTO roi_polygons "
                    "(frame_id, label, polygon, img_w, img_h, source_json) "
                    "VALUES (?,?,?,?,?,?)",
                    (frame_id, label, json.dumps(pts), img_w, img_h, jp.name),
                )
                n += 1
            roi_inserted += n
            roi_frames += 1
            print(f"  ROI {jp.name} -> {frame_id}  ({n} polys)")

        conn.commit()
    finally:
        conn.close()

    print()
    print(f"References (is_normal=1, ref=NULL) : {len(ref_rows)}")
    print(f"Normal negatives (source={SOURCE_BG}) : {len(normal_rows)}")
    print(f"Obstructed (is_normal=0, real)     : {len(obstr_rows)}")
    print(f"ROI: {roi_inserted} polygons on {roi_frames} reference frames")
    print(f"Total frames upserted: {len(ref_rows) + len(normal_rows) + len(obstr_rows)}")


if __name__ == "__main__":
    main()
