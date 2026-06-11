"""ECC-register the poli_ingegneria test set to its references.

For each obstructed scenario (davanti_centro/destra/sinistra) and the bg_normal
(dietro_vetro_o_poster), compute an ECC warp that aligns the test image to its
reference porta_N.jpg, then save the warped image to a parallel directory.

Output layout (mirrors the original):
  Dataset/ostruzioni_poli_ingegneria_aligned/davanti_centro/porta_N.jpg
  Dataset/ostruzioni_poli_ingegneria_aligned/davanti_destra/porta_N.jpg
  Dataset/ostruzioni_poli_ingegneria_aligned/davanti_sinistra/porta_N.jpg
  Dataset/ostruzioni_poli_ingegneria_aligned/dietro_vetro_o_poster/porta_N.jpg

DB entries are also added with new frame_id (path with `_aligned`) and source
suffixed `_aligned`. Original entries remain untouched, so you can run
`evaluate-db` on either version by toggling --ob-source and --bg-source.

Usage:
    python pipeline1/src/align_poli_test_set.py
"""
from __future__ import annotations
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DB   = ROOT / "Aggregated_dataset_db" / "occlusion.db"
DATA = ROOT / "Dataset"

REF_DIR  = DATA / "non_ostruite" / "poli ingegneria"
ORIG_ROOT = DATA / "ostruzioni_poli_ingegneria"
ALIGNED_ROOT = DATA / "ostruzioni_poli_ingegneria_aligned"

SCENARIOS_OBSTR = ("davanti_centro", "davanti_destra", "davanti_sinistra")
SCENARIOS_BG    = ("dietro_vetro_o_poster",)

SOURCE_GROUP = "poli_ingegneria"
SOURCE_REAL_ALIGNED = "poli_ingegneria_aligned"
SOURCE_BG_ALIGNED   = "background_change_aligned"

ECC_SCALE = 0.25     # downscale factor for ECC fit (sub-pixel result extrapolates back)
ECC_ITERS = 300
ECC_EPS   = 1e-6


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread workaround for Unicode paths on Windows."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except FileNotFoundError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    """cv2.imwrite workaround for Unicode paths on Windows."""
    ext = path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    elif ext == ".png":
        ok, buf = cv2.imencode(".png", img)
    else:
        return False
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def ecc_align(ref_bgr: np.ndarray, test_bgr: np.ndarray) -> tuple[np.ndarray, float, tuple[float, float], float]:
    """Returns (warp_full_res, ECC confidence, (tx_full, ty_full), rotation_deg).

    Computes the ECC EUCLIDEAN warp on downscaled grayscale, then rescales the
    translation back to full resolution. The rotation is scale-invariant.
    """
    ref_gray  = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2GRAY)
    test_gray = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2GRAY)
    ref_s  = cv2.resize(ref_gray,  None, fx=ECC_SCALE, fy=ECC_SCALE, interpolation=cv2.INTER_AREA)
    test_s = cv2.resize(test_gray, None, fx=ECC_SCALE, fy=ECC_SCALE, interpolation=cv2.INTER_AREA)

    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ECC_ITERS, ECC_EPS)
    cc, warp = cv2.findTransformECC(ref_s, test_s, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5)

    # Reconstruct the full-resolution warp by rescaling the translation.
    warp_full = warp.copy()
    warp_full[0, 2] = warp[0, 2] / ECC_SCALE
    warp_full[1, 2] = warp[1, 2] / ECC_SCALE
    tx_full = float(warp_full[0, 2])
    ty_full = float(warp_full[1, 2])
    rot_deg = float(np.degrees(np.arctan2(-warp[0, 1], warp[0, 0])))
    return warp_full, float(cc), (tx_full, ty_full), rot_deg


def warp_to_reference(test_bgr: np.ndarray, warp_full: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Apply WARP_INVERSE_MAP to bring the test image into the reference frame.

    `findTransformECC(ref, test, warp, ...)` finds warp such that ref ~ warpAffine(test, warp).
    So forward-warping test with warp produces an image aligned to ref.
    Important: ECC convention requires WARP_INVERSE_MAP if we want the inverse direction,
    but for our use case we want ref-aligned test, which is direct warp here.
    """
    aligned = cv2.warpAffine(
        test_bgr, warp_full, (out_w, out_h),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return aligned


def upsert_frame(conn: sqlite3.Connection, row: tuple) -> None:
    sql = (
        "INSERT INTO frames "
        "(frame_id, file_path, venue_type, is_normal, obstacle_class, "
        "occlusion_type, occlusion_level, split, source, source_group, "
        "reference_frame_id) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
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
    conn.execute(sql, row)


def main() -> None:
    if not REF_DIR.exists():
        sys.exit(f"Reference dir not found: {REF_DIR}")
    if not ORIG_ROOT.exists():
        sys.exit(f"Original obstructed root not found: {ORIG_ROOT}")

    refs = sorted(p for p in REF_DIR.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not refs:
        sys.exit("No reference images found.")

    print(f"Aligning {len(refs)} reference doors to:")
    print(f"  {ALIGNED_ROOT}")

    conn = sqlite3.connect(str(DB))
    conn.execute("PRAGMA foreign_keys = ON;")

    stats = {"obstr": 0, "bg": 0, "skipped": 0, "warps": []}

    try:
        for ref_path in refs:
            ref_stem = ref_path.stem
            ref_rel  = ref_path.relative_to(DATA).as_posix()
            ref_bgr  = imread_unicode(ref_path)
            if ref_bgr is None:
                print(f"  WARN: cannot read {ref_path}")
                continue
            h, w = ref_bgr.shape[:2]
            print(f"\n[{ref_stem}] reference ({w}x{h})")

            # Obstructed scenarios
            for scenario in SCENARIOS_OBSTR + SCENARIOS_BG:
                test_path = ORIG_ROOT / scenario / f"{ref_stem}.jpg"
                if not test_path.exists():
                    print(f"  WARN: missing {test_path}")
                    stats["skipped"] += 1
                    continue
                test_bgr = imread_unicode(test_path)
                if test_bgr is None:
                    print(f"  WARN: cannot read {test_path}")
                    stats["skipped"] += 1
                    continue
                try:
                    warp_full, cc, (tx, ty), rot = ecc_align(ref_bgr, test_bgr)
                except cv2.error as e:
                    print(f"  ERR ECC {scenario}/{ref_stem}: {e}")
                    stats["skipped"] += 1
                    continue
                aligned = warp_to_reference(test_bgr, warp_full, h, w)

                out_dir = ALIGNED_ROOT / scenario
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{ref_stem}.jpg"
                ok = imwrite_unicode(out_path, aligned, quality=95)
                if not ok:
                    print(f"  ERR write {out_path}")
                    stats["skipped"] += 1
                    continue

                out_rel = out_path.relative_to(DATA).as_posix()
                # DB entry
                is_bg = scenario in SCENARIOS_BG
                if is_bg:
                    row = (
                        out_rel, out_rel, "porta", 1, None,
                        "none", "none", "test", SOURCE_BG_ALIGNED,
                        SOURCE_GROUP, ref_rel,
                    )
                    stats["bg"] += 1
                else:
                    row = (
                        out_rel, out_rel, "porta", 0, "obstacle",
                        "real", "partial", "test", SOURCE_REAL_ALIGNED,
                        SOURCE_GROUP, ref_rel,
                    )
                    stats["obstr"] += 1
                upsert_frame(conn, row)
                stats["warps"].append((scenario, ref_stem, cc, tx, ty, rot))
                print(f"  {scenario:25s} cc={cc:.3f}  dx={tx:+7.1f}  dy={ty:+7.1f}  rot={rot:+.3f}°")

        conn.commit()
    finally:
        conn.close()

    print()
    print(f"Done. Aligned obstr: {stats['obstr']}, bg: {stats['bg']}, skipped: {stats['skipped']}")
    print(f"  Aligned files in: {ALIGNED_ROOT}")
    print(f"  DB entries added with source='{SOURCE_REAL_ALIGNED}' and "
          f"source='{SOURCE_BG_ALIGNED}'.")
    print()
    print("Use the aligned test set with:")
    print(f"  --ob-source {SOURCE_REAL_ALIGNED} --bg-source {SOURCE_BG_ALIGNED}")


if __name__ == "__main__":
    main()
