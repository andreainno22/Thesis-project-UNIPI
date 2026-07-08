"""
Interactive bounding-box annotation tool for the P1 test set.
====================================================================

Draw obstruction bounding boxes on the real `ostruzioni_poli_ingegneria`
images (which ship without labels) and save them in YOLO format
(class 0 = obstruction; normalized cx cy w h), one .txt per image next to
an `images/` -> `labels/` layout, so Ultralytics can compute mAP on them.

Controls:
  - Left-drag        : draw a box
  - 'u' / Backspace  : remove the last box on the current image
  - 's'              : save the current image's label file
  - 'n' / 'p'        : next / previous image (auto-saves)
  - 'q' / close      : save and quit

Example:
  python pipeline1/src/annotate_bboxes.py \
      --images "Dataset/ostruzioni_poli_ingegneria/*.jpg" \
      --out-dir Dataset/ostruzioni_poli_ingegneria/labels
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RectangleSelector
from PIL import Image


class BoxAnnotator:
    def __init__(self, images: list[Path], out_dir: Path):
        self.images = images
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        # boxes[i] = list of (cx, cy, w, h) normalized for image i
        self.boxes: dict[int, list[tuple[float, float, float, float]]] = {}

        # Disable matplotlib default keybindings that conflict with our
        # shortcuts: 'p' = pan, 's' = save dialog, 'backspace'/'left' = back
        # view, 'right' = forward view, 'q' = quit.
        for km in ("keymap.pan", "keymap.save", "keymap.back",
                   "keymap.forward", "keymap.quit"):
            plt.rcParams[km] = []

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.selector = RectangleSelector(
            self.ax, self._on_select, useblit=True,
            button=[1], minspanx=5, minspany=5, interactive=False,
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._load()
        plt.show()

    # ---- persistence -----------------------------------------------------
    def _label_path(self, i: int) -> Path:
        return self.out_dir / (self.images[i].stem + ".txt")

    def _load_existing(self, i: int) -> None:
        p = self._label_path(i)
        self.boxes[i] = []
        if p.exists():
            for line in p.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    self.boxes[i].append(tuple(float(x) for x in parts[1:]))

    def _save(self, i: int) -> None:
        lines = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                 for (cx, cy, w, h) in self.boxes.get(i, [])]
        self._label_path(i).write_text("\n".join(lines) + ("\n" if lines else ""))

    # ---- rendering -------------------------------------------------------
    def _load(self) -> None:
        if self.idx not in self.boxes:
            self._load_existing(self.idx)
        self.img = np.array(Image.open(self.images[self.idx]).convert("RGB"))
        self.H, self.W = self.img.shape[:2]
        self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.img)
        for (cx, cy, w, h) in self.boxes.get(self.idx, []):
            x = (cx - w / 2) * self.W
            y = (cy - h / 2) * self.H
            self.ax.add_patch(plt.Rectangle(
                (x, y), w * self.W, h * self.H,
                fill=False, edgecolor="#e74c3c", linewidth=2))
        n = len(self.boxes.get(self.idx, []))
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}"
            f"  -  {n} box  (drag=add, u=undo, n/p=nav, q=quit)")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    # ---- events ----------------------------------------------------------
    def _on_select(self, eclick, erelease) -> None:
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        cx = (x0 + x1) / 2 / self.W
        cy = (y0 + y1) / 2 / self.H
        w = (x1 - x0) / self.W
        h = (y1 - y0) / self.H
        self.boxes.setdefault(self.idx, []).append((cx, cy, w, h))
        self._redraw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        print(f"key event received: {event.key!r}")
        if key in ("u", "backspace"):
            if self.boxes.get(self.idx):
                self.boxes[self.idx].pop()
                self._redraw()
        elif key == "s":
            self._save(self.idx)
            print(f"saved {self._label_path(self.idx).name}")
        elif key in ("n", "right"):
            self._save(self.idx)
            if self.idx == len(self.images) - 1:
                print("already at last image")
            self.idx = min(self.idx + 1, len(self.images) - 1)
            self._load()
        elif key in ("p", "left"):
            self._save(self.idx)
            self.idx = max(self.idx - 1, 0)
            self._load()
        elif key == "q":
            self._save(self.idx)
            plt.close(self.fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, help="glob of images to annotate")
    ap.add_argument("--out-dir", required=True, help="dir for YOLO .txt labels")
    args = ap.parse_args()

    images = [Path(p) for p in sorted(glob.glob(args.images))]
    if not images:
        raise SystemExit(f"no images matched {args.images}")
    print(f"annotating {len(images)} images -> {args.out_dir}")
    BoxAnnotator(images, Path(args.out_dir))


if __name__ == "__main__":
    main()
