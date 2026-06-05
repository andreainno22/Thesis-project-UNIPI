"""
Diagnostic visualization for ROI gating decisions.

Shows which anomaly components are accepted or rejected by the
connected-component gating rule, and how they overlap the frame mask.

Usage:
    python pipeline1/src/debug_gating.py \
        --bank memory_bank.pt \
        --img path/to/test.jpg \
        --roi-json Dataset/roi/reference.json \
        --k 3.0 --min-overlap-px 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from anomaly_detection_patch_core import (
    DEVICE,
    FeatureExtractor,
    compute_patch_scores,
    extract_patch_features,
    load_image_tensor,
)
from roi_utils import load_roi_from_json, rasterize_frame_mask


def debug_gating(
    bank_path: str,
    img_path: str,
    roi_json_path: str,
    k_sigma: float = 3.0,
    min_overlap_px: int = 10,
) -> None:
    # ── Load memory bank ──────────────────────────────────────────────────
    try:
        ck = torch.load(bank_path, map_location="cpu", weights_only=True)
    except TypeError:
        ck = torch.load(bank_path, map_location="cpu")
    memory_bank = ck["memory_bank"]
    cal_mean = ck.get("cal_mean", 0.0)
    cal_std  = ck.get("cal_std",  1.0)
    threshold = cal_mean + k_sigma * cal_std
    print(f"Threshold: {cal_mean:.4f} + {k_sigma} * {cal_std:.4f} = {threshold:.4f}")

    # ── Compute anomaly map ───────────────────────────────────────────────
    model = FeatureExtractor().to(DEVICE).eval()
    img_pil = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img_pil.size
    print(f"Image size: {orig_w}x{orig_h}")

    tensor = load_image_tensor(img_pil)
    feats, H, W = extract_patch_features(model, tensor)
    scores = compute_patch_scores(feats, memory_bank)

    anomaly_map = scores.reshape(H, W).numpy()
    anomaly_map_full = cv2.resize(anomaly_map, (orig_w, orig_h),
                                  interpolation=cv2.INTER_LINEAR)
    anomaly_map_smooth = cv2.GaussianBlur(anomaly_map_full, (0, 0), sigmaX=4)

    # ── Load frame mask ───────────────────────────────────────────────────
    roi_obj = load_roi_from_json(roi_json_path)
    print(f"ROI source img size: {roi_obj.img_w}x{roi_obj.img_h}, "
          f"{len(roi_obj.polygons)} polygons")
    frame_mask = rasterize_frame_mask(roi_obj, target_h=orig_h, target_w=orig_w)
    print(f"Frame mask coverage: {int(frame_mask.sum())} px "
          f"({frame_mask.sum() / frame_mask.size * 100:.1f}%)")

    # ── Connected components ──────────────────────────────────────────────
    binary = (anomaly_map_smooth >= threshold).astype(np.uint8)
    print(f"\nPixels above threshold: {int(binary.sum())} "
          f"({binary.sum() / binary.size * 100:.2f}%)")

    n_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    valid_comps:   list[tuple[int, int, int]] = []
    rejected_comps: list[tuple[int, int, int]] = []
    for comp_id in range(1, n_labels):
        comp_mask = (labels_map == comp_id).astype(np.uint8)
        overlap = int((comp_mask & frame_mask).sum())
        size    = int(stats[comp_id, cv2.CC_STAT_AREA])
        if overlap >= min_overlap_px:
            valid_comps.append((comp_id, overlap, size))
        else:
            rejected_comps.append((comp_id, overlap, size))

    print(f"\nComponents above threshold: {n_labels - 1}")
    print(f"  Accepted (overlap >= {min_overlap_px} px): {len(valid_comps)}")
    for cid, ov, sz in sorted(valid_comps, key=lambda x: -x[2]):
        print(f"    comp {cid:3d}: area={sz:6d} px,  overlap={ov:5d} px")
    print(f"  Rejected: {len(rejected_comps)}")
    for cid, ov, sz in sorted(rejected_comps, key=lambda x: -x[2])[:20]:
        print(f"    comp {cid:3d}: area={sz:6d} px,  overlap={ov:5d} px")

    image_score = 0.0
    if valid_comps:
        valid_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
        for cid, _, _ in valid_comps:
            valid_mask |= (labels_map == cid).astype(np.uint8)
        image_score = float(anomaly_map_smooth[valid_mask > 0].max())
    print(f"\nFinal gated score: {image_score:.4f}  "
          f"({'ANOMALY' if image_score > threshold else 'NORMAL'})")

    # ── Build visualizations ──────────────────────────────────────────────
    img_np = np.array(img_pil)

    # Panel 1: frame mask overlay
    p1 = img_np.copy().astype(np.float32)
    p1[frame_mask > 0] = p1[frame_mask > 0] * 0.4 + np.array([0, 220, 0]) * 0.6
    p1 = p1.clip(0, 255).astype(np.uint8)

    # Panel 2: components colored by validity
    p2 = img_np.copy().astype(np.float32)
    for cid, _, _ in valid_comps:
        m = labels_map == cid
        p2[m] = p2[m] * 0.3 + np.array([0, 230, 0]) * 0.7
    for cid, _, _ in rejected_comps:
        m = labels_map == cid
        p2[m] = p2[m] * 0.3 + np.array([230, 0, 0]) * 0.7
    p2 = p2.clip(0, 255).astype(np.uint8)

    # Panel 3: overlap detail (frame=green, component=orange, overlap=yellow)
    p3 = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    p3[frame_mask > 0]       = [0,   200,  0]
    all_comp_mask = (labels_map > 0).astype(np.uint8)
    p3[all_comp_mask > 0]    = [255, 120,  0]
    p3[(all_comp_mask & frame_mask) > 0] = [255, 255, 0]

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    axes[0].imshow(p1)
    axes[0].set_title("Frame mask  (green = annotated frame regions)")
    axes[0].axis("off")

    axes[1].imshow(p2)
    legend_handles = [
        mpatches.Patch(color="green", label=f"Accepted: {len(valid_comps)}"),
        mpatches.Patch(color="red",   label=f"Rejected: {len(rejected_comps)}"),
    ]
    axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9)
    axes[1].set_title(f"Components  (threshold={threshold:.3f},  "
                      f"min_overlap={min_overlap_px} px)")
    axes[1].axis("off")

    axes[2].imshow(p3)
    legend_handles2 = [
        mpatches.Patch(color=[0,   200/255, 0],      label="Frame mask"),
        mpatches.Patch(color=[1,   120/255, 0],      label="Anomaly components"),
        mpatches.Patch(color=[1,   1,       0],      label="Overlap"),
    ]
    axes[2].legend(handles=legend_handles2, loc="lower right", fontsize=9)
    axes[2].set_title("Overlap detail")
    axes[2].axis("off")

    verdict = "ANOMALY DETECTED" if image_score > threshold else "REJECTED (score=0.0)"
    fig.suptitle(
        f"{Path(img_path).name}  |  gated score={image_score:.4f}  |  {verdict}",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()

    out_dir = Path(img_path).parent / "patchcore_output"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / (Path(img_path).stem + "_debug_gating.jpg")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualize ROI gating decisions for a single test image."
    )
    p.add_argument("--bank",          required=True, help="Memory bank (.pt)")
    p.add_argument("--img",           required=True, help="Test image path")
    p.add_argument("--roi-json",      required=True, help="ROI JSON (annotated frame)")
    p.add_argument("--k",    type=float, default=3.0,
                   help="k for threshold mu+k*sigma (default: 3.0)")
    p.add_argument("--min-overlap-px", type=int, default=10,
                   help="Min overlap pixels to accept a component (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    debug_gating(
        bank_path=args.bank,
        img_path=args.img,
        roi_json_path=args.roi_json,
        k_sigma=args.k,
        min_overlap_px=args.min_overlap_px,
    )
