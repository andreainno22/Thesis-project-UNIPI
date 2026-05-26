"""
Interactive rectangular ROI annotation tool.
=============================================

Permette di definire una ROI rettangolare (x,y,w,h) su ogni immagine
trascinando un rettangolo con il mouse. L'output e un JSON per immagine
compatibile con il flag --roi di build e test di anomaly_detection_patch_core.py.

Comandi:
  - Trascina il mouse       : disegna il rettangolo
  - 's'                     : salva il JSON
  - 'r'                     : resetta il rettangolo corrente
  - 'n'                     : immagine successiva (auto-save)
  - 'p'                     : immagine precedente (auto-save)
  - 'q' / chiusura finestra : esci (auto-save)

Output JSON (in --out-dir):
  {
    "image_path": "<percorso assoluto>",
    "img_w": W,
    "img_h": H,
    "roi": "x,y,w,h"   <- compatibile con --roi di build e test
  }

Esempio:
  python annotate_rect_roi.py \\
      --images "Dataset/non_ostruite/poli ingegneria/*.jpg" \\
      --out-dir Dataset/rect_roi
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import RectangleSelector
from PIL import Image


def json_path_for(image_path: Path, out_dir: Path) -> Path:
    return out_dir / (image_path.stem + "_rect_roi.json")


def load_existing(image_path: Path, out_dir: Path) -> Optional[tuple[int, int, int, int]]:
    jp = json_path_for(image_path, out_dir)
    if not jp.exists():
        return None
    with jp.open("r", encoding="utf-8") as f:
        d = json.load(f)
    roi_str = d.get("roi")
    if not roi_str:
        return None
    try:
        x, y, w, h = map(int, roi_str.split(","))
        return x, y, w, h
    except Exception:
        return None


def save_roi(image_path: Path, img_w: int, img_h: int,
             roi: tuple[int, int, int, int], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = json_path_for(image_path, out_dir)
    d = {
        "image_path": str(image_path.resolve()),
        "img_w": img_w,
        "img_h": img_h,
        "roi": f"{roi[0]},{roi[1]},{roi[2]},{roi[3]}",
    }
    with jp.open("w", encoding="utf-8") as f:
        json.dump(d, f, indent=2, ensure_ascii=False)
    return jp


class RectRoiAnnotator:
    def __init__(self, image_paths: list[Path], out_dir: Path):
        if not image_paths:
            raise ValueError("Nessuna immagine da annotare.")
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.idx = 0
        self.current_roi: Optional[tuple[int, int, int, int]] = None  # x,y,w,h
        self.image_np: Optional[np.ndarray] = None
        self.img_w = 0
        self.img_h = 0

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.fig.subplots_adjust(bottom=0.05)

        # Disable conflicting matplotlib keybindings
        for km, keys in [
            ("keymap.save",   ["s"]),
            ("keymap.quit",   ["q"]),
            ("keymap.pan",    ["p"]),
            ("keymap.yscale", ["l"]),
            ("keymap.back",   ["c"]),
        ]:
            for k in keys:
                try:
                    plt.rcParams[km].remove(k)
                except (ValueError, KeyError):
                    pass

        self.selector = RectangleSelector(
            self.ax,
            self._on_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords="pixels",
            interactive=True,
            props=dict(facecolor="cyan", edgecolor="cyan", alpha=0.25, fill=True),
        )

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._load_current()
        plt.show()

    def _load_current(self):
        ip = self.image_paths[self.idx]
        img = Image.open(ip).convert("RGB")
        self.image_np = np.array(img)
        self.img_h, self.img_w = self.image_np.shape[:2]
        self.current_roi = load_existing(ip, self.out_dir)
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image_np)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        ip = self.image_paths[self.idx]
        roi_str = (f"{self.current_roi[0]},{self.current_roi[1]},"
                   f"{self.current_roi[2]},{self.current_roi[3]}"
                   if self.current_roi else "nessuna")
        title = (
            f"[{self.idx+1}/{len(self.image_paths)}] {ip.name}\n"
            f"ROI: {roi_str}   |   trascina per disegnare, 's'=salva, 'r'=reset, 'n'/'p'=naviga"
        )
        self.ax.set_title(title, fontsize=9)

        if self.current_roi:
            x, y, w, h = self.current_roi
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=2, edgecolor="cyan", facecolor="cyan", alpha=0.20,
            )
            self.ax.add_patch(rect)
            self.ax.text(
                x + w / 2, y + h / 2,
                f"ROI\n{w}×{h} px",
                color="white", fontsize=10, ha="center", va="center",
                bbox=dict(facecolor="cyan", alpha=0.5, edgecolor="none",
                          boxstyle="round,pad=0.3"),
            )

        self.fig.canvas.draw_idle()
        # Re-attach selector to new axes
        self.selector.ax = self.ax
        self.selector.set_active(True)

    def _on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if any(v is None for v in [x1, y1, x2, y2]):
            return
        x = int(round(min(x1, x2)))
        y = int(round(min(y1, y2)))
        w = int(round(abs(x2 - x1)))
        h = int(round(abs(y2 - y1)))
        x = max(0, min(self.img_w - 1, x))
        y = max(0, min(self.img_h - 1, y))
        w = min(w, self.img_w - x)
        h = min(h, self.img_h - y)
        if w < 5 or h < 5:
            return
        self.current_roi = (x, y, w, h)
        print(f"[INFO] ROI selezionata: x={x}, y={y}, w={w}, h={h}")
        self._redraw()

    def on_key(self, event: KeyEvent):
        key = (event.key or "").lower()
        if key == "s":
            self._save()
        elif key == "r":
            self.current_roi = None
            print("[INFO] ROI resettata.")
            self._redraw()
        elif key == "n":
            self._save()
            if self.idx + 1 < len(self.image_paths):
                self.idx += 1
                self._load_current()
            else:
                print("[INFO] gia' all'ultima immagine.")
        elif key == "p":
            self._save()
            if self.idx > 0:
                self.idx -= 1
                self._load_current()
            else:
                print("[INFO] gia' alla prima immagine.")
        elif key == "q":
            self._save()
            plt.close(self.fig)

    def _save(self):
        if self.current_roi is None:
            print("[WARN] nessuna ROI definita, skip salvataggio.")
            return
        ip = self.image_paths[self.idx]
        jp = save_roi(ip, self.img_w, self.img_h, self.current_roi, self.out_dir)
        print(f"[OK] salvato {jp}  ->  roi={self.current_roi[0]},{self.current_roi[1]},"
              f"{self.current_roi[2]},{self.current_roi[3]}")


def expand_image_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for pat in patterns:
        matches = sorted(glob.glob(pat))
        if not matches and Path(pat).exists():
            matches = [pat]
        for m in matches:
            mp = Path(m).resolve()
            if mp.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            if mp in seen:
                continue
            seen.add(mp)
            out.append(mp)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive rectangular ROI annotation for build/test.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help='Glob pattern(s) o lista di immagini (es: "Dataset/non_ostruite/poli ingegneria/*.jpg").',
    )
    parser.add_argument(
        "--out-dir", default="Dataset/rect_roi",
        help="Cartella output JSON (default: Dataset/rect_roi).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = expand_image_paths(args.images)
    if not paths:
        raise SystemExit("Nessuna immagine trovata.")
    out_dir = Path(args.out_dir)
    print(f"[INFO] {len(paths)} immagini, output in {out_dir.resolve()}")
    print("[INFO] trascina per disegnare la ROI, 's'=salva, 'r'=reset, 'n'/'p'=naviga, 'q'=esci")
    RectRoiAnnotator(paths, out_dir)


if __name__ == "__main__":
    main()
