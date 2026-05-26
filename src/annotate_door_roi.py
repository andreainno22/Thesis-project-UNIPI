"""
Interactive ROI annotation tool for door frame regions.
====================================================================

Annota su un'immagine di riferimento i poligoni del telaio della porta
(stipiti, montante centrale, soglia, architrave). Le ROI vengono salvate
come JSON per essere poi importate nel DB e usate dalla pipeline PatchCore.

Comandi (interattivi):
  - Click sinistro            : aggiunge un vertice al poligono corrente
  - Tasto destro / 'c'         : chiude il poligono corrente
  - 'l'                        : cambia la label del prossimo poligono
  - 'u' / Backspace            : rimuove l'ultimo vertice
  - 'd'                        : elimina l'ultimo poligono chiuso
  - 's'                        : salva il JSON corrente
  - 'n'                        : passa all'immagine successiva (auto-save)
  - 'p'                        : passa all'immagine precedente (auto-save)
  - 'q' / chiusura finestra    : esci (auto-save)

Esempio:
  python annotate_door_roi.py \
      --images "Dataset/non_ostruite/poli ingegneria/*.jpg" \
      --out-dir Dataset/roi
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.patches import Polygon as MplPolygon
from PIL import Image


# Label valide, in ordine in cui vengono ciclate con il tasto 'l'.
LABEL_CYCLE = [
    "left_jamb",
    "center_mullion",
    "right_jamb",
    "threshold",
    "architrave",
    "other",
]

LABEL_COLORS = {
    "left_jamb":      "#e74c3c",
    "right_jamb":     "#3498db",
    "center_mullion": "#9b59b6",
    "threshold":      "#2ecc71",
    "architrave":     "#f39c12",
    "other":          "#7f8c8d",
}


@dataclass
class AnnotatedPolygon:
    label: str
    points: list[list[int]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"label": self.label, "polygon": self.points}


@dataclass
class ImageAnnotation:
    image_path: str
    img_w: int
    img_h: int
    polygons: list[AnnotatedPolygon] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "image_path": self.image_path,
            "img_w": self.img_w,
            "img_h": self.img_h,
            "polygons": [p.to_dict() for p in self.polygons],
        }


def json_path_for(image_path: Path, out_dir: Path) -> Path:
    return out_dir / (image_path.stem + ".json")


def load_existing(image_path: Path, out_dir: Path) -> Optional[ImageAnnotation]:
    jp = json_path_for(image_path, out_dir)
    if not jp.exists():
        return None
    with jp.open("r", encoding="utf-8") as f:
        d = json.load(f)
    return ImageAnnotation(
        image_path=d["image_path"],
        img_w=int(d["img_w"]),
        img_h=int(d["img_h"]),
        polygons=[AnnotatedPolygon(label=p["label"], points=p["polygon"])
                  for p in d.get("polygons", [])],
    )


def save_annotation(ann: ImageAnnotation, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp = json_path_for(Path(ann.image_path), out_dir)
    with jp.open("w", encoding="utf-8") as f:
        json.dump(ann.to_dict(), f, indent=2, ensure_ascii=False)
    return jp


class RoiAnnotator:
    def __init__(self, image_paths: list[Path], out_dir: Path):
        if not image_paths:
            raise ValueError("Nessuna immagine da annotare.")
        self.image_paths = image_paths
        self.out_dir = out_dir
        self.idx = 0
        self.current_label_idx = 0
        self.current_points: list[list[int]] = []
        self.annotation: Optional[ImageAnnotation] = None

        self.fig, self.ax = plt.subplots(figsize=(10, 12))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self._load_current()
        plt.show()

    def _load_current(self):
        ip = self.image_paths[self.idx]
        img = Image.open(ip).convert("RGB")
        self.image_np = np.array(img)
        h, w = self.image_np.shape[:2]
        existing = load_existing(ip, self.out_dir)
        if existing and (existing.img_w != w or existing.img_h != h):
            print(f"[WARN] Dimensioni JSON ({existing.img_w}x{existing.img_h}) "
                  f"diverse dall'immagine ({w}x{h}); ignoro l'esistente.")
            existing = None
        self.annotation = existing or ImageAnnotation(
            image_path=str(ip), img_w=w, img_h=h, polygons=[]
        )
        self.current_points = []
        self._redraw()

    def _redraw(self):
        self.ax.clear()
        self.ax.imshow(self.image_np)
        label = LABEL_CYCLE[self.current_label_idx]
        title = (
            f"[{self.idx+1}/{len(self.image_paths)}] "
            f"{Path(self.annotation.image_path).name}\n"
            f"label corrente: {label}   |   "
            f"poligoni: {len(self.annotation.polygons)}   |   "
            f"vertici parziali: {len(self.current_points)}"
        )
        self.ax.set_title(title, fontsize=10)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Poligoni chiusi
        for ap in self.annotation.polygons:
            if len(ap.points) >= 3:
                color = LABEL_COLORS.get(ap.label, "#000000")
                patch = MplPolygon(
                    ap.points, closed=True, fill=True,
                    facecolor=color, edgecolor=color, alpha=0.30, linewidth=2,
                )
                self.ax.add_patch(patch)
                cx = float(np.mean([p[0] for p in ap.points]))
                cy = float(np.mean([p[1] for p in ap.points]))
                self.ax.text(cx, cy, ap.label, color="white", fontsize=9,
                             ha="center", va="center",
                             bbox=dict(facecolor=color, alpha=0.7,
                                       edgecolor="none", boxstyle="round,pad=0.2"))

        # Poligono in costruzione
        if self.current_points:
            xs = [p[0] for p in self.current_points]
            ys = [p[1] for p in self.current_points]
            color = LABEL_COLORS.get(label, "#000000")
            self.ax.plot(xs + xs[:1], ys + ys[:1],
                        marker="o", color=color, linewidth=1.5, linestyle="--")

        self.fig.canvas.draw_idle()

    def on_click(self, event: MouseEvent):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if event.button == 1:  # left click -> add vertex
            x = int(round(event.xdata))
            y = int(round(event.ydata))
            x = max(0, min(self.annotation.img_w - 1, x))
            y = max(0, min(self.annotation.img_h - 1, y))
            self.current_points.append([x, y])
            self._redraw()
        elif event.button == 3:  # right click -> close polygon
            self._close_polygon()

    def on_key(self, event: KeyEvent):
        key = (event.key or "").lower()
        if key in ("c", "enter"):
            self._close_polygon()
        elif key in ("u", "backspace"):
            if self.current_points:
                self.current_points.pop()
                self._redraw()
        elif key == "d":
            if self.annotation.polygons:
                removed = self.annotation.polygons.pop()
                print(f"[INFO] rimosso poligono: {removed.label}")
                self._redraw()
        elif key == "l":
            self.current_label_idx = (self.current_label_idx + 1) % len(LABEL_CYCLE)
            print(f"[INFO] label corrente: {LABEL_CYCLE[self.current_label_idx]}")
            self._redraw()
        elif key == "s":
            self._save()
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

    def _close_polygon(self):
        if len(self.current_points) < 3:
            print(f"[WARN] poligono ignorato: servono almeno 3 vertici "
                  f"(ne hai {len(self.current_points)}).")
            return
        ap = AnnotatedPolygon(
            label=LABEL_CYCLE[self.current_label_idx],
            points=self.current_points.copy(),
        )
        self.annotation.polygons.append(ap)
        print(f"[INFO] poligono '{ap.label}' chiuso con {len(ap.points)} vertici.")
        self.current_points = []
        self._redraw()

    def _save(self):
        if self.annotation is None:
            return
        jp = save_annotation(self.annotation, self.out_dir)
        print(f"[OK] salvato {jp}")


def expand_image_paths(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    seen = set()
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
        description="Interactive ROI annotation tool for door frames.",
    )
    parser.add_argument(
        "--images", nargs="+", required=True,
        help='Glob pattern(s) o lista di immagini da annotare '
             '(es: "Dataset/non_ostruite/porte/*.jpg").',
    )
    parser.add_argument(
        "--out-dir", default="Dataset/roi",
        help="Cartella di output per i JSON delle ROI (default: Dataset/roi).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = expand_image_paths(args.images)
    if not paths:
        raise SystemExit("Nessuna immagine trovata per i pattern forniti.")
    out_dir = Path(args.out_dir)
    print(f"[INFO] {len(paths)} immagini, output in {out_dir.resolve()}")
    print("[INFO] click sx = vertice, tasto destro = chiudi poligono, "
          "'l' = cambia label, 'n'/'p' = avanti/indietro, 'q' = esci.")
    RoiAnnotator(paths, out_dir)


if __name__ == "__main__":
    main()
