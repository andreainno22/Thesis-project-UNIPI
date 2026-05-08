"""
Load Dataset contents into the SQLite DB.

Usage:
    python load_dataset_to_db.py --db path\to\occlusion.db
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from init_db import init_db


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT_DIR / "occlusion.db"
DEFAULT_SCHEMA_PATH = ROOT_DIR / "Aggregated_dataset_db" / "db_schema.sql"
DEFAULT_DATASET_ROOT = ROOT_DIR / "Dataset"

SUPPORTED_IMAGE_EXT = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
    ".avif",
    ".heic",
}

NON_OSTRUITE_VENUE_MAP = {
    "porte": "porta",
    "corridoi": "corridoio",
}

SYNTHETIC_PREFIX_VENUE = {
    "porte": "porta",
    "corridoi": "corridoio",
}

SYNTHETIC_GROUP_RE = re.compile(
    r"^(corridoi|porte)_\d{4}_(?P<base>(corridoio|porta)_\d+)_v\d+$"
)
RENAMED_OBSTRUCTED_RE = re.compile(
    r"^(?P<base>(corridoio|porta)_\d+)_ostruita$"
)


@dataclass(frozen=True)
class FrameRecord:
    frame_id: str
    file_path: str
    venue_type: str
    is_normal: int
    obstacle_class: str | None
    occlusion_type: str
    occlusion_level: str
    split: str
    source: str
    source_group: str | None
    reference_frame_id: str | None


@dataclass(frozen=True)
class BBoxRecord:
    frame_id: str
    label_class: str
    cx: float
    cy: float
    w: float
    h: float
    txt_path: str | None


def normalize_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXT


def stable_splitter(train_ratio: float, val_ratio: float, seed: str) -> Callable[[str], str]:
    test_ratio = max(0.0, 1.0 - train_ratio - val_ratio)

    def pick_split(group_key: str) -> str:
        digest = hashlib.sha1(f"{seed}:{group_key}".encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        if value < train_ratio:
            return "train"
        if value < train_ratio + val_ratio:
            return "val"
        if test_ratio > 0:
            return "test"
        return "val"

    return pick_split


def fixed_splitter(split_name: str) -> Callable[[str], str]:
    def pick_split(_: str) -> str:
        return split_name

    return pick_split


def parse_synthetic_group(stem: str) -> str | None:
    match = SYNTHETIC_GROUP_RE.match(stem)
    if not match:
        match = RENAMED_OBSTRUCTED_RE.match(stem)
        if not match:
            return None
    return match.group("base")


def infer_venue_from_stem(stem: str) -> str | None:
    for prefix, venue in SYNTHETIC_PREFIX_VENUE.items():
        if stem.startswith(f"{prefix}_"):
            return venue
    if stem.startswith("porta_"):
        return "porta"
    if stem.startswith("corridoio_"):
        return "corridoio"
    return None


def read_yolo_labels(label_path: Path, label_class: str, txt_relpath: str) -> list[BBoxRecord]:
    records: list[BBoxRecord] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
            except ValueError:
                continue
            records.append(
                BBoxRecord(
                    frame_id="",
                    label_class=label_class,
                    cx=cx,
                    cy=cy,
                    w=w,
                    h=h,
                    txt_path=txt_relpath,
                )
            )
    return records


def collect_non_ostruite(
    dataset_root: Path,
    split_for_group: Callable[[str], str],
    source_value: str,
) -> tuple[list[FrameRecord], dict[str, str]]:
    frames: list[FrameRecord] = []
    group_to_frame_id: dict[str, str] = {}

    base_dir = dataset_root / "non_ostruite"
    for folder_name, venue in NON_OSTRUITE_VENUE_MAP.items():
        venue_dir = base_dir / folder_name
        if not venue_dir.exists():
            continue
        for img_path in sorted(venue_dir.iterdir()):
            if not is_image(img_path):
                continue
            group = img_path.stem
            split = split_for_group(group)
            rel_path = normalize_relpath(img_path, dataset_root)
            record = FrameRecord(
                frame_id=rel_path,
                file_path=rel_path,
                venue_type=venue,
                is_normal=1,
                obstacle_class=None,
                occlusion_type="none",
                occlusion_level="none",
                split=split,
                source=source_value,
                source_group=group,
                reference_frame_id=None,
            )
            frames.append(record)
            group_to_frame_id[group] = rel_path

    return frames, group_to_frame_id


def collect_ostruzioni_reali(
    dataset_root: Path,
    split_for_group: Callable[[str], str],
    source_value: str,
    group_to_frame_id: dict[str, str],
    label_class: str,
    occlusion_level: str,
) -> tuple[list[FrameRecord], list[BBoxRecord], list[str]]:
    frames: list[FrameRecord] = []
    annotations: list[BBoxRecord] = []
    missing_labels: list[str] = []

    images_dir = dataset_root / "ostruzioni_reali" / "images"
    labels_dir = dataset_root / "ostruzioni_reali" / "labels"
    if not images_dir.exists() or not labels_dir.exists():
        return frames, annotations, missing_labels

    for img_path in sorted(images_dir.iterdir()):
        if not is_image(img_path):
            continue

        stem = img_path.stem
        group = parse_synthetic_group(stem) or stem
        venue = infer_venue_from_stem(stem)
        if venue is None:
            continue

        split = split_for_group(group)
        rel_path = normalize_relpath(img_path, dataset_root)
        label_path = labels_dir / f"{stem}.txt"
        txt_relpath = normalize_relpath(label_path, dataset_root) if label_path.exists() else None

        reference_id = group_to_frame_id.get(group)

        record = FrameRecord(
            frame_id=rel_path,
            file_path=rel_path,
            venue_type=venue,
            is_normal=0,
            obstacle_class=label_class,
            occlusion_type="synthetic_copypaste",
            occlusion_level=occlusion_level,
            split=split,
            source=source_value,
            source_group=group,
            reference_frame_id=reference_id,
        )
        frames.append(record)

        if not label_path.exists():
            missing_labels.append(rel_path)
            continue

        bbox_records = read_yolo_labels(label_path, label_class, txt_relpath)
        for bbox in bbox_records:
            annotations.append(
                BBoxRecord(
                    frame_id=rel_path,
                    label_class=bbox.label_class,
                    cx=bbox.cx,
                    cy=bbox.cy,
                    w=bbox.w,
                    h=bbox.h,
                    txt_path=bbox.txt_path,
                )
            )

    return frames, annotations, missing_labels


def delete_bbox_for_frames(conn: sqlite3.Connection, frame_ids: Iterable[str]) -> None:
    conn.executemany(
        "DELETE FROM annotations_bbox WHERE frame_id = ?",
        [(frame_id,) for frame_id in frame_ids],
    )


def upsert_frames(conn: sqlite3.Connection, frames: Iterable[FrameRecord]) -> None:
    sql = (
        "INSERT INTO frames ("
        "frame_id, file_path, venue_type, is_normal, obstacle_class, occlusion_type, "
        "occlusion_level, split, source, source_group, reference_frame_id"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(frame_id) DO UPDATE SET "
        "file_path=excluded.file_path, "
        "venue_type=excluded.venue_type, "
        "is_normal=excluded.is_normal, "
        "obstacle_class=excluded.obstacle_class, "
        "occlusion_type=excluded.occlusion_type, "
        "occlusion_level=excluded.occlusion_level, "
        "split=excluded.split, "
        "source=excluded.source, "
        "source_group=excluded.source_group, "
        "reference_frame_id=excluded.reference_frame_id"
    )
    conn.executemany(
        sql,
        [
            (
                frame.frame_id,
                frame.file_path,
                frame.venue_type,
                frame.is_normal,
                frame.obstacle_class,
                frame.occlusion_type,
                frame.occlusion_level,
                frame.split,
                frame.source,
                frame.source_group,
                frame.reference_frame_id,
            )
            for frame in frames
        ],
    )


def insert_bboxes(conn: sqlite3.Connection, bboxes: Iterable[BBoxRecord]) -> None:
    sql = (
        "INSERT INTO annotations_bbox (frame_id, label_class, cx, cy, w, h, txt_path) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)"
    )
    conn.executemany(
        sql,
        [
            (
                bbox.frame_id,
                bbox.label_class,
                bbox.cx,
                bbox.cy,
                bbox.w,
                bbox.h,
                bbox.txt_path,
            )
            for bbox in bboxes
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Dataset into SQLite DB.")
    parser.add_argument(
        "--db",
        type=str,
        default=str(DEFAULT_DB_PATH),
        help="Path to the SQLite database file.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default=str(DEFAULT_SCHEMA_PATH),
        help="Path to the SQL schema file.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the Dataset root folder.",
    )
    parser.add_argument(
        "--split-mode",
        type=str,
        default="hash",
        choices=["hash", "all-train"],
        help="Split strategy: hash for deterministic 80/20, all-train to defer splitting.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--split-seed",
        type=str,
        default="42",
        help="Seed used for deterministic split hashing.",
    )
    parser.add_argument(
        "--source-normal",
        type=str,
        default="Pinterest",
        help="Source label for non_ostruite frames.",
    )
    parser.add_argument(
        "--source-synthetic",
        type=str,
        default="synthetic",
        help="Source label for ostruzioni_reali frames.",
    )
    parser.add_argument(
        "--label-class",
        type=str,
        default="obstacle",
        help="Label class used for bbox annotations.",
    )
    parser.add_argument(
        "--occlusion-level",
        type=str,
        default="partial",
        choices=["partial", "full"],
        help="Occlusion level for synthetic frames.",
    )
    parser.add_argument(
        "--purge-bboxes",
        action="store_true",
        help="Delete existing bbox annotations for frames being reloaded.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    db_path = Path(args.db)
    schema_path = Path(args.schema)

    init_db(db_path, schema_path)

    if args.split_mode == "all-train":
        split_for_group = fixed_splitter("train")
    else:
        split_for_group = stable_splitter(args.train_ratio, args.val_ratio, args.split_seed)

    normal_frames, group_to_frame_id = collect_non_ostruite(
        dataset_root, split_for_group, args.source_normal
    )
    synthetic_frames, bboxes, missing_labels = collect_ostruzioni_reali(
        dataset_root,
        split_for_group,
        args.source_synthetic,
        group_to_frame_id,
        args.label_class,
        args.occlusion_level,
    )

    all_frames = normal_frames + synthetic_frames

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")

        if args.purge_bboxes:
            delete_bbox_for_frames(conn, [f.frame_id for f in synthetic_frames])

        upsert_frames(conn, all_frames)
        insert_bboxes(conn, bboxes)
        conn.commit()
    finally:
        conn.close()

    print("Load complete.")
    print(f"  Frames inserted/updated: {len(all_frames)}")
    print(f"    Normal: {len(normal_frames)}")
    print(f"    Synthetic: {len(synthetic_frames)}")
    print(f"  BBox annotations inserted: {len(bboxes)}")
    if missing_labels:
        print(f"  Missing label files: {len(missing_labels)}")


if __name__ == "__main__":
    main()
