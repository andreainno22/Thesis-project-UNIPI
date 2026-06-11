"""
ROI utilities for door-frame anomaly scoring.
====================================================================

Carica i poligoni del telaio porta (annotati con annotate_door_roi.py
o presenti nella tabella `roi_polygons` del DB), li converte in maschere
binarie alla risoluzione richiesta, e fornisce la regola di scoring
"connected component must touch frame" descritta nel report.

Il modulo non dipende da PyTorch: lavora su array NumPy.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


VALID_LABELS = {
    "left_jamb", "right_jamb", "center_mullion",
    "threshold", "architrave", "other",
    "door_region",
}

# Labels that define structural frame regions (used for connected-component
# gating). door_region is excluded because it spans the whole door including
# the panel/glass area, where background changes happen.
FRAME_LABELS = {
    "left_jamb", "right_jamb", "center_mullion",
    "threshold", "architrave",
}

# Labels usable as a build-time ROI (patches inside enter the memory bank).
BUILD_ROI_LABELS = {"door_region"}


@dataclass
class FrameRoi:
    """Polygons + image dimensions for a single reference image."""
    img_w: int
    img_h: int
    polygons: list[tuple[str, np.ndarray]]   # (label, Nx2 int array)

    @property
    def is_empty(self) -> bool:
        return len(self.polygons) == 0


def load_roi_from_json(json_path: str | Path) -> FrameRoi:
    """Load a ROI annotation file produced by annotate_door_roi.py."""
    p = Path(json_path)
    with p.open("r", encoding="utf-8") as f:
        d = json.load(f)
    polys: list[tuple[str, np.ndarray]] = []
    for entry in d.get("polygons", []):
        label = entry["label"]
        pts = np.asarray(entry["polygon"], dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            continue
        polys.append((label, pts))
    return FrameRoi(
        img_w=int(d["img_w"]),
        img_h=int(d["img_h"]),
        polygons=polys,
    )


def load_roi_from_db(
    db_path: str | Path,
    frame_id: str,
) -> Optional[FrameRoi]:
    """Load ROI polygons for a given frame_id from the SQLite DB.

    Returns None if no rows exist for the frame.
    """
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT label, polygon, img_w, img_h "
            "FROM roi_polygons WHERE frame_id = ?",
            (frame_id,),
        ).fetchall()
    finally:
        conn.close()
    if not rows:
        return None
    polys: list[tuple[str, np.ndarray]] = []
    img_w = img_h = 0
    for label, polygon_json, w, h in rows:
        img_w, img_h = int(w), int(h)
        pts = np.asarray(json.loads(polygon_json), dtype=np.int32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
            continue
        polys.append((label, pts))
    return FrameRoi(img_w=img_w, img_h=img_h, polygons=polys)


def rasterize_frame_mask(
    roi: FrameRoi,
    target_h: int,
    target_w: int,
    labels: Optional[set[str]] = None,
) -> np.ndarray:
    """Rasterize the polygons into a binary mask at the target resolution.

    Parameters
    ----------
    roi : FrameRoi
        Polygons defined on the original image grid (roi.img_w x roi.img_h).
    target_h, target_w : int
        Output mask shape.
    labels : set[str] or None
        If set, only polygons whose label is in this set are rasterized.
        Default: all polygons (i.e. the union of frame regions).

    Returns
    -------
    mask : np.ndarray of shape (target_h, target_w), dtype uint8 (0/1)
    """
    mask = np.zeros((target_h, target_w), dtype=np.uint8)
    if roi.is_empty:
        return mask
    sx = target_w / float(roi.img_w)
    sy = target_h / float(roi.img_h)
    for label, pts in roi.polygons:
        if labels is not None and label not in labels:
            continue
        scaled = np.empty_like(pts, dtype=np.int32)
        scaled[:, 0] = np.clip(np.round(pts[:, 0] * sx), 0, target_w - 1)
        scaled[:, 1] = np.clip(np.round(pts[:, 1] * sy), 0, target_h - 1)
        cv2.fillPoly(mask, [scaled], 1)
    return mask


def gated_anomaly_score(
    anomaly_map: np.ndarray,
    threshold: float,
    frame_mask: np.ndarray,
    min_overlap_px: int = 10,
) -> tuple[float, np.ndarray]:
    """Apply the connected-component gating rule.

    A component of pixels with score >= threshold is "valid" only if it
    overlaps the frame mask in at least `min_overlap_px` pixels. The
    returned image score is the maximum of the anomaly map restricted to
    valid components, or 0 if none survive.

    Parameters
    ----------
    anomaly_map : np.ndarray (H, W) float
        Smoothed anomaly map at the same resolution as `frame_mask`.
    threshold : float
        Cutoff to binarise the anomaly map.
    frame_mask : np.ndarray (H, W) uint8
        Binary mask (0/1) of the union of frame regions.
    min_overlap_px : int
        Minimum number of pixels a component must share with the frame
        mask to be considered a valid detection.

    Returns
    -------
    image_score : float
        Max anomaly value over valid components; 0.0 if none.
    valid_mask : np.ndarray (H, W) uint8
        Binary mask of the surviving components (useful for heatmap viz).
    """
    if anomaly_map.shape != frame_mask.shape:
        raise ValueError(
            f"shape mismatch: anomaly_map {anomaly_map.shape} vs "
            f"frame_mask {frame_mask.shape}"
        )
    binary = (anomaly_map >= threshold).astype(np.uint8)
    if binary.sum() == 0:
        return 0.0, np.zeros_like(binary)

    num_labels, comp_labels = cv2.connectedComponents(binary, connectivity=8)
    valid_mask = np.zeros_like(binary)
    best_score = 0.0
    for k in range(1, num_labels):
        comp = (comp_labels == k)
        overlap = int(np.logical_and(comp, frame_mask > 0).sum())
        if overlap < min_overlap_px:
            continue
        valid_mask[comp] = 1
        cmax = float(anomaly_map[comp].max())
        if cmax > best_score:
            best_score = cmax
    return best_score, valid_mask


def polygon_to_patch_mask(
    roi: FrameRoi,
    img_size: int = 224,
    patch_h: int = 28,
    patch_w: int = 28,
    labels: Optional[set[str]] = None,
) -> Optional[np.ndarray]:
    """Convert polygon ROI to a boolean patch-grid mask.

    A patch at grid position (pi, pj) is included if its center, mapped back
    to the original image coordinates, falls inside the union of polygons
    matching `labels`. Returns a flat boolean array of shape (patch_h*patch_w,)
    suitable for indexing patch-level feature tensors.

    Returns None if no matching polygon exists (no filtering should be applied).
    """
    if labels is None:
        labels = BUILD_ROI_LABELS
    if roi.is_empty:
        return None
    polys = [pts for label, pts in roi.polygons if label in labels]
    if not polys:
        return None

    # Build mask on the img_size grid first via fillPoly, then sample the
    # patch grid centers. Coordinates of polygons are in the original image
    # frame; we scale them to img_size.
    sx = img_size / float(roi.img_w)
    sy = img_size / float(roi.img_h)
    img_mask = np.zeros((img_size, img_size), dtype=np.uint8)
    for pts in polys:
        scaled = np.empty_like(pts, dtype=np.int32)
        scaled[:, 0] = np.clip(np.round(pts[:, 0] * sx), 0, img_size - 1)
        scaled[:, 1] = np.clip(np.round(pts[:, 1] * sy), 0, img_size - 1)
        cv2.fillPoly(img_mask, [scaled], 1)

    cell_w = img_size / patch_w
    cell_h = img_size / patch_h
    out = np.zeros(patch_h * patch_w, dtype=bool)
    for pi in range(patch_h):
        cy = int((pi + 0.5) * cell_h)
        cy = min(cy, img_size - 1)
        for pj in range(patch_w):
            cx = int((pj + 0.5) * cell_w)
            cx = min(cx, img_size - 1)
            if img_mask[cy, cx]:
                out[pi * patch_w + pj] = True
    return out


def find_roi_json_for_image(
    image_path: str | Path,
    roi_dir: str | Path,
) -> Optional[Path]:
    """Lookup convention: <roi_dir>/<image_stem>.json."""
    image_path = Path(image_path)
    candidate = Path(roi_dir) / (image_path.stem + ".json")
    return candidate if candidate.exists() else None
