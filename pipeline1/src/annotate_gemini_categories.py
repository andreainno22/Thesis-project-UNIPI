"""
Interactive per-object category annotation for the gemini test set (P1).
====================================================================

The gemini bboxes are already drawn (single-class "obstruction", used for
detection mAP via gemini_test.yaml). This tool does NOT touch those .txt
label files - it adds a category on top of each existing box, saved to a
sidecar JSON, so the mAP evaluation pipeline (nc=1) stays untouched.

Purpose: break down detection performance by object category (which
categories the model finds vs misses) and flag cases where the model fuses
several ground-truth boxes of the same category (e.g. stacked cardboard
boxes) into one prediction.

Controls:
  - Click near a box       : select it (highlighted yellow)
  - '1'..'7'                : assign category to the selected box
                              (1 cart, 2 wheelchair, 3 stretcher, 4 box,
                               5 waste_bin, 6 fan, 7 other)
  - 'l'                     : cycle the selected box's category
  - 's'                     : save the current image's categories
  - 'n' / 'p'               : next / previous image (auto-saves)
  - 'q' / close             : save and quit

Example:
  python pipeline1/src/annotate_gemini_categories.py \
      --images "Dataset/ostruzioni_gemini/corridoi/images/corridoi/*.jpg" \
      --labels Dataset/ostruzioni_gemini/corridoi/labels/corridoi \
      --out-dir Dataset/ostruzioni_gemini/corridoi/categories

  python pipeline1/src/annotate_gemini_categories.py \
      --images "Dataset/ostruzioni_gemini/porte/images/porte/*.jpg" \
      --labels Dataset/ostruzioni_gemini/porte/labels/porte \
      --out-dir Dataset/ostruzioni_gemini/porte/categories
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CATEGORIES = ["cart", "wheelchair", "stretcher", "box", "waste_bin", "fan", "other"]
COLORS = {
    "cart": "#3498db", "wheelchair": "#9b59b6", "stretcher": "#e67e22",
    "box": "#2ecc71", "waste_bin": "#1abc9c", "fan": "#e74c3c",
    "other": "#95a5a6", None: "#bdc3c7",
}


class CategoryAnnotator:
    def __init__(self, images: list[Path], labels_dir: Path, out_dir: Path):
        self.images = images
        self.labels_dir = labels_dir
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.idx = 0
        self.selected = -1
        # boxes[i] = list of dicts {cx, cy, w, h, category}
        self.boxes: dict[int, list[dict]] = {}

        for km in ("keymap.pan", "keymap.save", "keymap.back",
                   "keymap.forward", "keymap.quit"):
            plt.rcParams[km] = []

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self._load()
        plt.show()

    # ---- persistence -----------------------------------------------------
    def _gt_label_path(self, i: int) -> Path:
        return self.labels_dir / (self.images[i].stem + ".txt")

    def _cat_path(self, i: int) -> Path:
        return self.out_dir / (self.images[i].stem + ".json")

    def _load_boxes(self, i: int) -> None:
        cat_path = self._cat_path(i)
        if cat_path.exists():
            self.boxes[i] = json.loads(cat_path.read_text())["boxes"]
            return
        gt_path = self._gt_label_path(i)
        boxes = []
        if gt_path.exists():
            for line in gt_path.read_text().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    cx, cy, w, h = (float(x) for x in parts[1:])
                    boxes.append({"cx": cx, "cy": cy, "w": w, "h": h, "category": None})
        self.boxes[i] = boxes

    def _save(self, i: int) -> None:
        self._cat_path(i).write_text(json.dumps({"boxes": self.boxes.get(i, [])}, indent=1))

    # ---- rendering -------------------------------------------------------
    def _load(self) -> None:
        if self.idx not in self.boxes:
            self._load_boxes(self.idx)
        self.img = np.array(Image.open(self.images[self.idx]).convert("RGB"))
        self.H, self.W = self.img.shape[:2]
        self.selected = -1
        self._redraw()

    def _redraw(self) -> None:
        self.ax.clear()
        self.ax.imshow(self.img)
        for j, b in enumerate(self.boxes.get(self.idx, [])):
            x = (b["cx"] - b["w"] / 2) * self.W
            y = (b["cy"] - b["h"] / 2) * self.H
            w, h = b["w"] * self.W, b["h"] * self.H
            edge = "#f1c40f" if j == self.selected else COLORS[b["category"]]
            lw = 3 if j == self.selected else 2
            self.ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                                             edgecolor=edge, linewidth=lw))
            self.ax.text(x, max(y - 4, 0), b["category"] or "?",
                         color=edge, fontsize=9, weight="bold")
        n_done = sum(1 for b in self.boxes.get(self.idx, []) if b["category"])
        n_tot = len(self.boxes.get(self.idx, []))
        self.ax.set_title(
            f"[{self.idx + 1}/{len(self.images)}] {self.images[self.idx].name}"
            f"  -  {n_done}/{n_tot} categorized  "
            f"(click=select, 1-7=category, l=cycle, n/p=nav, q=quit)")
        self.ax.axis("off")
        self.fig.canvas.draw_idle()

    # ---- events ------------------------------------------------------
    def _on_click(self, event) -> None:
        if event.xdata is None or event.ydata is None:
            return
        boxes = self.boxes.get(self.idx, [])
        best, best_d = -1, float("inf")
        for j, b in enumerate(boxes):
            x, y = b["cx"] * self.W, b["cy"] * self.H
            hw, hh = b["w"] * self.W / 2, b["h"] * self.H / 2
            inside = (abs(event.xdata - x) <= hw) and (abs(event.ydata - y) <= hh)
            d = (event.xdata - x) ** 2 + (event.ydata - y) ** 2
            if inside and d < best_d:
                best, best_d = j, d
        self.selected = best
        self._redraw()

    def _set_category(self, cat: str) -> None:
        if self.selected < 0:
            return
        self.boxes[self.idx][self.selected]["category"] = cat
        self._redraw()

    def _on_key(self, event) -> None:
        key = (event.key or "").lower()
        if key in "1234567":
            self._set_category(CATEGORIES[int(key) - 1])
        elif key == "l" and self.selected >= 0:
            cur = self.boxes[self.idx][self.selected]["category"]
            i = (CATEGORIES.index(cur) + 1) if cur in CATEGORIES else 0
            self._set_category(CATEGORIES[i % len(CATEGORIES)])
        elif key == "s":
            self._save(self.idx)
            print(f"saved {self._cat_path(self.idx).name}")
        elif key in ("n", "right"):
            self._save(self.idx)
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
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True, help="glob of images to annotate")
    ap.add_argument("--labels", required=True,
                    help="dir with existing YOLO .txt GT boxes (not modified)")
    ap.add_argument("--out-dir", required=True, help="dir for per-image category JSON")
    args = ap.parse_args()

    images = [Path(p) for p in sorted(glob.glob(args.images))]
    if not images:
        raise SystemExit(f"no images matched {args.images}")
    print(f"categorizing {len(images)} images -> {args.out_dir}")
    print(f"categories: {CATEGORIES}")
    CategoryAnnotator(images, Path(args.labels), Path(args.out_dir))


if __name__ == "__main__":
    main()
