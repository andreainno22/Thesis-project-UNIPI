"""Explanatory figure for the connected-component gating mechanism (P3).

Builds a schematic-but-faithful illustration of the four stages of the gating
rule used at test time (see roi_utils.gated_anomaly_score):

  1. continuous anomaly map
  2. binarisation at the per-camera threshold theta
  3. 8-connectivity connected components
  4. gating: a component is valid only if it overlaps the door-frame mask in at
     least min_overlap_px pixels; the image score is the max anomaly over valid
     components (0 otherwise).

The anomaly map is synthetic, but every downstream step (threshold, cv2
connected components, overlap test, score) runs the *real* algorithm, so the
figure faithfully reproduces the pipeline logic. One obstruction is grounded and
crosses the threshold region (accepted); a second blob sits entirely on the door
panel with no frame contact (rejected) - the prototypical thin-stemmed failure
(e.g. the floor-fan head).

Output: template_tesi/images/patchCore/p3_gating_mechanism.{pdf,png}

Run: conda run -n tesi_env python pipeline3/src/make_gating_mechanism_figure.py
"""
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "template_tesi" / "images" / "patchCore"

C_ACCEPT = "#2ca02c"   # green
C_REJECT = "#d62728"   # red
C_FRAME = "#1f77b4"    # blue

MIN_OVERLAP_PX = 10

# ── Door geometry (portrait canvas) ──────────────────────────────────────────
H, W = 420, 300
X0, X1, Y0, Y1 = 60, 240, 40, 380           # door outer box
JAMB, ARCH, THR = 22, 22, 26                # frame thicknesses

# Frame mask = structural ring (jambs + architrave + threshold), panel excluded.
frame_mask = np.zeros((H, W), dtype=np.uint8)
frame_mask[Y0:Y1, X0:X0 + JAMB] = 1          # left jamb
frame_mask[Y0:Y1, X1 - JAMB:X1] = 1          # right jamb
frame_mask[Y0:Y0 + ARCH, X0:X1] = 1          # architrave
frame_mask[Y1 - THR:Y1, X0:X1] = 1           # threshold

PANEL = (X0 + JAMB, X1 - JAMB, Y0 + ARCH, Y1 - THR)  # interior bounds


def gauss(cy, cx, sy, sx, amp):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-(((yy - cy) / sy) ** 2 + ((xx - cx) / sx) ** 2) / 2.0)


# ── Synthetic anomaly map ────────────────────────────────────────────────────
amap = np.zeros((H, W), dtype=np.float32)
# Grounded obstruction: panel -> threshold, overlaps the frame (accepted).
amap += gauss(345, 150, 44, 34, 1.00)
# Thin-stemmed object head: isolated on the panel, no frame contact (rejected).
amap += gauss(175, 150, 30, 26, 0.95)
# Low background texture (kept below theta).
rng = np.random.default_rng(3)
bg = rng.standard_normal((H, W)).astype(np.float32)
bg = cv2.GaussianBlur(bg, (0, 0), sigmaX=18)
bg = 0.12 * (bg - bg.min()) / (bg.max() - bg.min())
amap = cv2.GaussianBlur(amap + bg, (0, 0), sigmaX=4)
amap = amap / float(amap.max())   # normalise peak to 1.0 for a clean score

THETA = 0.45  # per-camera threshold mu + k*sigma (schematic value)

# ── Real algorithm: binarise -> connected components -> overlap gating ───────
binary = (amap >= THETA).astype(np.uint8)
n_labels, labels = cv2.connectedComponents(binary, connectivity=8)

accepted, rejected = [], []
for cid in range(1, n_labels):
    comp = labels == cid
    overlap = int(np.logical_and(comp, frame_mask > 0).sum())
    (accepted if overlap >= MIN_OVERLAP_PX else rejected).append((cid, overlap))

valid_mask = np.zeros((H, W), dtype=bool)
for cid, _ in accepted:
    valid_mask |= labels == cid
score = float(amap[valid_mask].max()) if valid_mask.any() else 0.0

print(f"components={n_labels - 1} accepted={len(accepted)} rejected={len(rejected)}")
for cid, ov in accepted + rejected:
    print(f"  comp {cid}: overlap={ov}px -> {'ACCEPT' if ov >= MIN_OVERLAP_PX else 'reject'}")
print(f"gated score = {score:.3f}  (theta={THETA})")
assert n_labels - 1 == 2 and len(accepted) == 1 and len(rejected) == 1, "geometry drifted"


# ── Drawing helpers ──────────────────────────────────────────────────────────
def draw_frame_outline(ax, color=C_FRAME, lw=1.4, ls="-"):
    for (x, y, w, h) in [
        (X0, Y0, X1 - X0, ARCH),
        (X0, Y1 - THR, X1 - X0, THR),
        (X0, Y0, JAMB, Y1 - Y0),
        (X1 - JAMB, Y0, JAMB, Y1 - Y0),
    ]:
        ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                        edgecolor=color, lw=lw, ls=ls))
    ax.add_patch(mpatches.Rectangle((PANEL[0], PANEL[2]),
                                    PANEL[1] - PANEL[0], PANEL[3] - PANEL[2],
                                    fill=False, edgecolor=color, lw=lw, ls=ls))


def style(ax, title):
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_aspect("equal"); ax.axis("off")


# ── Plot: four stages ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.2))

# (1) anomaly map
im = axes[0].imshow(amap, cmap="jet", vmin=0, vmax=1)
draw_frame_outline(axes[0], color="white", lw=1.0)
style(axes[0], "(a) Anomaly map")

# (2) binarisation
bin_cmap = ListedColormap(["#f0f0f0", "#303030"])
axes[1].imshow(binary, cmap=bin_cmap, vmin=0, vmax=1)
draw_frame_outline(axes[1], color=C_FRAME, lw=1.2)
style(axes[1], r"(b) Binarisation  ($\geq \theta$)")

# (3) connected components (frame mask shaded)
comp_rgb = np.ones((H, W, 3), dtype=np.float32)
comp_rgb[frame_mask > 0] = [0.80, 0.88, 0.97]          # frame region shading
palette = [np.array([0.85, 0.50, 0.10]), np.array([0.45, 0.25, 0.70])]
for i, cid in enumerate(range(1, n_labels)):
    comp_rgb[labels == cid] = palette[i % len(palette)]
axes[2].imshow(comp_rgb)
draw_frame_outline(axes[2], color=C_FRAME, lw=1.2)
style(axes[2], "(c) Connected components")

# (4) gating decision
gate_rgb = np.ones((H, W, 3), dtype=np.float32)
gate_rgb[frame_mask > 0] = [0.80, 0.88, 0.97]
for cid, _ in accepted:
    gate_rgb[labels == cid] = matplotlib.colors.to_rgb(C_ACCEPT)
for cid, _ in rejected:
    gate_rgb[labels == cid] = matplotlib.colors.to_rgb(C_REJECT)
axes[3].imshow(gate_rgb)
draw_frame_outline(axes[3], color=C_FRAME, lw=1.2)
style(axes[3], "(d) Gating decision")

# annotate decision arrows
ay, ax_ = 330, 150
axes[3].annotate("accepted\n(overlaps frame)", xy=(ax_, 360), xytext=(150, 408),
                 ha="center", va="top", fontsize=8, color=C_ACCEPT,
                 arrowprops=dict(arrowstyle="->", color=C_ACCEPT, lw=1.0))
axes[3].annotate("rejected\n(no frame contact)", xy=(150, 180), xytext=(150, 95),
                 ha="center", va="bottom", fontsize=8, color=C_REJECT,
                 arrowprops=dict(arrowstyle="->", color=C_REJECT, lw=1.0))

handles = [
    mpatches.Patch(color="#cce0f7", label="Frame mask (jambs, architrave, threshold)"),
    mpatches.Patch(color=C_ACCEPT, label=fr"Accepted: overlap $\geq$ {MIN_OVERLAP_PX} px"),
    mpatches.Patch(color=C_REJECT, label="Rejected: overlap below limit"),
]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5,
           frameon=False, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(
    r"Image score $=\max$ over accepted components "
    fr"$= {score:.2f}$  (0 if none survive)",
    fontsize=10, y=1.0)
fig.tight_layout(rect=(0, 0.04, 1, 0.97))

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"p3_gating_mechanism.{ext}", dpi=150, bbox_inches="tight")
print("saved p3_gating_mechanism.pdf/.png")
