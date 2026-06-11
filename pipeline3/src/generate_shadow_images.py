"""
Genera varianti con ombre fittizie per testare la robustezza di PatchCore.

Tipi di ombra disponibili:
  - ellittica   : ellisse scura con bordi sfumati, simula ombra proiettata da
                  un oggetto (es. carrello, persona) sul pavimento/parete.
  - direzionale : gradiente scuro da un lato dell'immagine, simula luce
                  parzialmente bloccata (finestra laterale, lampada a parete).
  - striscia    : banda scura orizzontale/verticale con bordi sfumati, simula
                  l'ombra di una trave del soffitto, uno scaffale a muro, un
                  traverso architettonico - comune in corridoi industriali.
  - random      : applica k ombre casuali dal pool {ellittica, direzionale,
                  striscia}, con k ∈ [--shadows-min, --shadows-max] (default).
  - combined    : direzionale + ellittica in sequenza (backward compat).

Uso tipico:
  # Genera + inserisci nel DB (consigliato)
  python pipeline1/src/generate_shadow_images.py \\
    --input-dir  Dataset/non_ostruite/porte \\
    --output-dir Dataset/shadow_test/normali \\
    --n-variants 3 --shadow-type random \\
    --shadows-min 1 --shadows-max 3 \\
    --db Aggregated_dataset_db/occlusion.db \\
    --dataset-root Dataset

  # Solo genera immagini, senza DB
  python pipeline1/src/generate_shadow_images.py \\
    --input-dir Dataset/non_ostruite/porte \\
    --output-dir Dataset/shadow_test/normali
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


# ── Generatori di ombre ───────────────────────────────────────────────────────

def add_elliptical_shadow(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.35, 0.65),
    blur_sigma_range: tuple[float, float] = (12.0, 35.0),
) -> Image.Image:
    """
    Aggiunge un'ombra ellittica con bordi morbidi.
    L'ellisse è ruotata di un angolo casuale per maggiore varietà.
    """
    arr   = np.array(img_pil).astype(np.float32)
    h, w  = arr.shape[:2]

    cx        = int(rng.integers(w // 5, 4 * w // 5))
    cy        = int(rng.integers(h // 5, 4 * h // 5))
    a         = int(rng.integers(w // 10, w // 3))
    b         = int(rng.integers(h // 10, h // 3))
    angle     = rng.uniform(0.0, np.pi)
    intensity = rng.uniform(*intensity_range)

    Y, X    = np.ogrid[:h, :w]
    cos_a   = np.cos(angle);  sin_a = np.sin(angle)
    x_rot   =  (X - cx) * cos_a + (Y - cy) * sin_a
    y_rot   = -(X - cx) * sin_a + (Y - cy) * cos_a
    mask    = ((x_rot / (a + 1e-6)) ** 2 + (y_rot / (b + 1e-6)) ** 2 <= 1).astype(np.float32)

    sigma     = rng.uniform(*blur_sigma_range)
    mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(sigma))

    factor = 1.0 - intensity * mask_soft[:, :, np.newaxis]
    arr    = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_directional_shadow(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.20, 0.50),
    coverage_range:  tuple[float, float] = (0.20, 0.55),
) -> Image.Image:
    """
    Aggiunge un gradiente scuro da uno dei quattro lati.
    Simula luce parzialmente bloccata (finestra, porta semiaperta).
    """
    arr       = np.array(img_pil).astype(np.float32)
    h, w      = arr.shape[:2]
    intensity = rng.uniform(*intensity_range)
    coverage  = rng.uniform(*coverage_range)
    direction = int(rng.integers(0, 4))   # 0=sinistra 1=destra 2=alto 3=basso

    if direction in (0, 1):
        n    = int(w * coverage)
        grad = np.linspace(1.0, 0.0, max(n, 1), dtype=np.float32)
        if direction == 0:
            row = np.concatenate([grad, np.zeros(w - n, dtype=np.float32)])
        else:
            row = np.concatenate([np.zeros(w - n, dtype=np.float32), grad[::-1]])
        mask = np.tile(row, (h, 1))
    else:
        n    = int(h * coverage)
        grad = np.linspace(1.0, 0.0, max(n, 1), dtype=np.float32)
        if direction == 2:
            col = np.concatenate([grad, np.zeros(h - n, dtype=np.float32)])
        else:
            col = np.concatenate([np.zeros(h - n, dtype=np.float32), grad[::-1]])
        mask = np.tile(col.reshape(-1, 1), (1, w))

    factor = 1.0 - intensity * mask[:, :, np.newaxis]
    arr    = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_stripe_shadow(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.25, 0.55),
    width_range:     tuple[float, float] = (0.05, 0.20),
    blur_sigma_range: tuple[float, float] = (8.0, 22.0),
    orientation: str = "random",
    position: float | None = None,
) -> Image.Image:
    """
    Aggiunge una striscia scura orizzontale o verticale con bordi sfumati.
    orientation: 'horizontal' | 'vertical' | 'random'
    position: posizione relativa del centro [0.0-1.0]; None = casuale
    """
    arr       = np.array(img_pil).astype(np.float32)
    h, w      = arr.shape[:2]
    intensity = rng.uniform(*intensity_range)
    if orientation == "horizontal":
        horizontal = True
    elif orientation == "vertical":
        horizontal = False
    else:
        horizontal = bool(rng.integers(0, 2))

    mask = np.zeros((h, w), dtype=np.float32)
    if horizontal:
        band_half = max(2, int(h * rng.uniform(*width_range) / 2))
        center    = int(position * h) if position is not None else int(rng.integers(band_half, h - band_half))
        center    = int(np.clip(center, band_half, h - band_half))
        mask[max(0, center - band_half):min(h, center + band_half), :] = 1.0
    else:
        band_half = max(2, int(w * rng.uniform(*width_range) / 2))
        center    = int(position * w) if position is not None else int(rng.integers(band_half, w - band_half))
        center    = int(np.clip(center, band_half, w - band_half))
        mask[:, max(0, center - band_half):min(w, center + band_half)] = 1.0

    sigma     = rng.uniform(*blur_sigma_range)
    mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(sigma))

    factor = 1.0 - intensity * mask_soft[:, :, np.newaxis]
    arr    = np.clip(arr * factor, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def add_combined_shadow(img_pil: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Applica ombra direzionale + ellittica in sequenza (backward compat)."""
    img = add_directional_shadow(img_pil, rng)
    img = add_elliptical_shadow(img, rng)
    return img


# Pool atomico usato dalla modalità random (ombre)
_SHADOW_POOL = [add_elliptical_shadow, add_directional_shadow, add_stripe_shadow]

_SHADOW_FN = {
    "elliptical":  add_elliptical_shadow,
    "directional": add_directional_shadow,
    "stripe":      add_stripe_shadow,
    "combined":    add_combined_shadow,
    "random":      None,   # gestito esplicitamente in generate_shadow_variants
}


# ── Generatori di luce ────────────────────────────────────────────────────────

def add_elliptical_light(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.20, 0.50),
    blur_sigma_range: tuple[float, float] = (12.0, 35.0),
) -> Image.Image:
    """
    Aggiunge un fascio luminoso ellittico con bordi morbidi.
    Simula luce da lucernario, lampada a soffitto puntata, o riflesso luminoso.
    Applica una leggera tinta calda (R↑, B↓) per simulare luce solare/alogena.
    """
    arr  = np.array(img_pil).astype(np.float32)
    h, w = arr.shape[:2]

    cx        = int(rng.integers(w // 5, 4 * w // 5))
    cy        = int(rng.integers(h // 5, 4 * h // 5))
    a         = int(rng.integers(w // 10, w // 3))
    b         = int(rng.integers(h // 10, h // 3))
    angle     = rng.uniform(0.0, np.pi)
    intensity = rng.uniform(*intensity_range)

    Y, X    = np.ogrid[:h, :w]
    cos_a   = np.cos(angle);  sin_a = np.sin(angle)
    x_rot   =  (X - cx) * cos_a + (Y - cy) * sin_a
    y_rot   = -(X - cx) * sin_a + (Y - cy) * cos_a
    mask    = ((x_rot / (a + 1e-6)) ** 2 + (y_rot / (b + 1e-6)) ** 2 <= 1).astype(np.float32)

    sigma     = rng.uniform(*blur_sigma_range)
    mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(sigma))

    r_extra = float(rng.uniform(0.0, 0.15))
    b_less  = float(rng.uniform(0.0, 0.08))

    arr_out = arr.copy()
    arr_out[:, :, 0] = arr[:, :, 0] * (1.0 + (intensity + r_extra) * mask_soft)
    arr_out[:, :, 1] = arr[:, :, 1] * (1.0 + intensity * mask_soft)
    arr_out[:, :, 2] = arr[:, :, 2] * (1.0 + max(0.0, intensity - b_less) * mask_soft)
    return Image.fromarray(np.clip(arr_out, 0, 255).astype(np.uint8))


def add_directional_light(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range: tuple[float, float] = (0.15, 0.40),
    coverage_range:  tuple[float, float] = (0.20, 0.55),
) -> Image.Image:
    """
    Aggiunge un gradiente luminoso progressivo da uno dei quattro lati.
    Simula luce diffusa da una finestra laterale, una porta aperta su ambiente
    illuminato, o una lampada a parete che inonda un lato del corridoio.
    """
    arr       = np.array(img_pil).astype(np.float32)
    h, w      = arr.shape[:2]
    intensity = rng.uniform(*intensity_range)
    coverage  = rng.uniform(*coverage_range)
    direction = int(rng.integers(0, 4))   # 0=sinistra 1=destra 2=alto 3=basso

    if direction in (0, 1):
        n    = int(w * coverage)
        grad = np.linspace(1.0, 0.0, max(n, 1), dtype=np.float32)
        if direction == 0:
            row = np.concatenate([grad, np.zeros(w - n, dtype=np.float32)])
        else:
            row = np.concatenate([np.zeros(w - n, dtype=np.float32), grad[::-1]])
        mask = np.tile(row, (h, 1))
    else:
        n    = int(h * coverage)
        grad = np.linspace(1.0, 0.0, max(n, 1), dtype=np.float32)
        if direction == 2:
            col = np.concatenate([grad, np.zeros(h - n, dtype=np.float32)])
        else:
            col = np.concatenate([np.zeros(h - n, dtype=np.float32), grad[::-1]])
        mask = np.tile(col.reshape(-1, 1), (1, w))

    factor = 1.0 + intensity * mask[:, :, np.newaxis]
    return Image.fromarray(np.clip(arr * factor, 0, 255).astype(np.uint8))


def add_stripe_light(
    img_pil: Image.Image,
    rng: np.random.Generator,
    intensity_range:  tuple[float, float] = (0.25, 0.55),
    width_range:      tuple[float, float] = (0.05, 0.20),
    blur_sigma_range: tuple[float, float] = (6.0, 18.0),
    orientation: str = "random",
    position: float | None = None,
) -> Image.Image:
    """
    Aggiunge una striscia luminosa con bordi sfumati.
    orientation: 'horizontal' | 'vertical' | 'diagonal' | 'random'
    position: posizione relativa del centro [0.0-1.0]; None = casuale
    """
    arr  = np.array(img_pil).astype(np.float32)
    h, w = arr.shape[:2]
    intensity = rng.uniform(*intensity_range)
    if orientation == "horizontal":
        ori_idx = 0
    elif orientation == "vertical":
        ori_idx = 1
    elif orientation == "diagonal":
        ori_idx = 2
    else:
        ori_idx = int(rng.integers(0, 3))   # 0=horizontal, 1=vertical, 2=diagonal

    if ori_idx == 0:   # orizzontale
        band_half = max(2, int(h * rng.uniform(*width_range) / 2))
        center    = int(position * h) if position is not None else int(rng.integers(band_half, h - band_half))
        center    = int(np.clip(center, band_half, h - band_half))
        mask      = np.zeros((h, w), dtype=np.float32)
        mask[max(0, center - band_half):min(h, center + band_half), :] = 1.0
        sigma     = rng.uniform(*blur_sigma_range)
        mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(sigma))
        arr_out   = np.clip(arr * (1.0 + intensity * mask_soft[:, :, np.newaxis]), 0, 255)

    elif ori_idx == 1:   # verticale
        band_half = max(2, int(w * rng.uniform(*width_range) / 2))
        center    = int(position * w) if position is not None else int(rng.integers(band_half, w - band_half))
        center    = int(np.clip(center, band_half, w - band_half))
        mask      = np.zeros((h, w), dtype=np.float32)
        mask[:, max(0, center - band_half):min(w, center + band_half)] = 1.0
        sigma     = rng.uniform(*blur_sigma_range)
        mask_soft = cv2.GaussianBlur(mask, (0, 0), sigmaX=float(sigma))
        arr_out   = np.clip(arr * (1.0 + intensity * mask_soft[:, :, np.newaxis]), 0, 255)

    else:   # diagonale: fascio solare
        if rng.random() < 0.5:
            angle_deg = rng.uniform(25.0, 65.0)    # sale da sinistra a destra
        else:
            angle_deg = rng.uniform(115.0, 155.0)  # sale da destra a sinistra
        angle_rad = np.radians(float(angle_deg))
        sin_a = float(np.sin(angle_rad))
        cos_a = float(np.cos(angle_rad))

        Y_grid, X_grid = np.mgrid[:h, :w]
        perp_map = (-X_grid.astype(np.float32) * sin_a
                    + Y_grid.astype(np.float32) * cos_a)
        p_min    = float(perp_map.min())
        p_max    = float(perp_map.max())
        p_range  = p_max - p_min
        bh       = max(4.0, p_range * rng.uniform(*width_range) / 2)
        center_p = (float(position) * (p_max - p_min) + p_min
                    if position is not None
                    else float(rng.uniform(p_min + bh, p_max - bh)))

        mask_soft = np.clip(1.0 - np.abs(perp_map - center_p) / (bh + 1e-6),
                            0.0, 1.0).astype(np.float32)
        sigma     = rng.uniform(*blur_sigma_range)
        mask_soft = cv2.GaussianBlur(mask_soft, (0, 0), sigmaX=float(sigma))

        r_extra = float(rng.uniform(0.05, 0.18))
        b_less  = float(rng.uniform(0.03, 0.10))
        arr_out = arr.copy()
        arr_out[:, :, 0] = arr[:, :, 0] * (1.0 + (intensity + r_extra) * mask_soft)
        arr_out[:, :, 1] = arr[:, :, 1] * (1.0 + intensity * mask_soft)
        arr_out[:, :, 2] = arr[:, :, 2] * (1.0 + max(0.0, intensity - b_less) * mask_soft)
        arr_out = np.clip(arr_out, 0, 255)

    return Image.fromarray(arr_out.astype(np.uint8))


# Pool atomico usato dalla modalità random (luci)
_LIGHT_POOL = [add_elliptical_light, add_directional_light, add_stripe_light]

_LIGHT_FN = {
    "light_elliptical":  add_elliptical_light,
    "light_directional": add_directional_light,
    "light_stripe":      add_stripe_light,
    "light_random":      None,   # gestito esplicitamente in generate_shadow_variants
}


# ── DB loading ────────────────────────────────────────────────────────────────

def _normalize_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_shadow_frames_to_db(
    pairs: list[tuple[Path, Path]],   # (shadow_path, original_path)
    dataset_root: Path,
    db_path: Path,
    source: str = "shadow",
) -> tuple[int, int]:
    """
    Inserisce i frame shadow normali nel DB, ereditando i metadati dall'originale.

    reference_frame_id = frame_id dell'originale normale, così evaluate_from_db
    può raccoglierli via query_shadow_normal_frames per il test FP con ombre.
    """
    conn = sqlite3.connect(str(db_path))
    inserted = skipped = 0
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        for shadow_path, orig_path in pairs:
            try:
                orig_rel = _normalize_relpath(orig_path, dataset_root)
            except ValueError:
                print(f"  ERRORE: {orig_path} non è dentro dataset_root ({dataset_root}). "
                      f"Usa --dataset-root correttamente.")
                sys.exit(1)
            row = conn.execute(
                "SELECT frame_id, venue_type, is_normal, occlusion_type, "
                "occlusion_level, split, source_group, reference_frame_id "
                "FROM frames WHERE file_path = ?",
                (orig_rel,),
            ).fetchone()

            if row is None:
                print(f"  WARN: non trovato nel DB: {orig_rel}  (skip)")
                skipped += 1
                continue

            (orig_id, venue_type, is_normal, occlusion_type,
             occlusion_level, split, source_group, orig_ref_id) = row

            try:
                shadow_rel = _normalize_relpath(shadow_path, dataset_root)
            except ValueError:
                print(f"  ERRORE: --output-dir ({shadow_path.parent}) deve essere "
                      f"dentro --dataset-root ({dataset_root}).")
                sys.exit(1)

            ref_id = orig_id if is_normal == 1 else orig_ref_id

            conn.execute(
                "INSERT INTO frames "
                "(frame_id, file_path, venue_type, is_normal, obstacle_class, "
                "occlusion_type, occlusion_level, split, source, source_group, "
                "reference_frame_id) VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(frame_id) DO UPDATE SET "
                "file_path=excluded.file_path, venue_type=excluded.venue_type, "
                "is_normal=excluded.is_normal, occlusion_type=excluded.occlusion_type, "
                "occlusion_level=excluded.occlusion_level, split=excluded.split, "
                "source=excluded.source, source_group=excluded.source_group, "
                "reference_frame_id=excluded.reference_frame_id",
                (
                    shadow_rel, shadow_rel,
                    venue_type, is_normal,
                    None,        # obstacle_class - non rilevante per shadow/light
                    occlusion_type, occlusion_level,
                    split, source,
                    source_group, ref_id,
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return inserted, skipped


# ── Generazione immagini ──────────────────────────────────────────────────────

_ALL_FN: dict = {**_SHADOW_FN, **_LIGHT_FN}


def generate_shadow_variants(
    input_dir: str,
    output_dir: str,
    n_variants: int = 1,
    shadow_type: str = "random",
    n_shadows_min: int = 1,
    n_shadows_max: int = 3,
    seed: int = 42,
    intensity_min: float | None = None,
    intensity_max: float | None = None,
    stripe_orientation: str = "random",
    stripe_position: float | None = None,
    db_path: str | None = None,
    dataset_root: str | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> None:
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    is_light = shadow_type.startswith("light_")
    pool     = _LIGHT_POOL if is_light else _SHADOW_POOL
    db_source = "light" if is_light else "shadow"
    suffix_tag = "light" if is_light else "shadow"

    if shadow_type not in _ALL_FN:
        print(f"ERRORE: tipo '{shadow_type}' non valido. Scegli fra: {list(_ALL_FN)}")
        sys.exit(1)

    if n_shadows_min < 1 or n_shadows_max < n_shadows_min:
        print("ERRORE: --shadows-min deve essere >= 1 e <= --shadows-max")
        sys.exit(1)

    disturbance_fn = _ALL_FN[shadow_type]   # None per random/light_random

    intensity_override: tuple[float, float] | None = None
    if intensity_min is not None or intensity_max is not None:
        lo = intensity_min if intensity_min is not None else 0.20
        hi = intensity_max if intensity_max is not None else 0.80
        intensity_override = (lo, hi)

    img_paths = sorted(p for p in in_path.iterdir() if p.suffix.lower() in extensions)
    if not img_paths:
        print(f"ERRORE: nessuna immagine trovata in {in_path}")
        sys.exit(1)

    load_db = db_path is not None and dataset_root is not None

    random_types = ("random", "light_random")
    type_desc = (
        f"{shadow_type} [{n_shadows_min}–{n_shadows_max}] da pool "
        f"{{{', '.join(f.__name__ for f in pool)}}}"
        if shadow_type in random_types
        else shadow_type
    )
    print(f"\n{'='*60}")
    print(f"GENERATE {'LIGHT' if is_light else 'SHADOW'} IMAGES")
    print(f"  Input    : {in_path}  ({len(img_paths)} immagini)")
    print(f"  Output   : {out_path}")
    print(f"  Tipo     : {type_desc}")
    print(f"  Varianti : {n_variants}")
    print(f"  Seed     : {seed}")
    print(f"  DB       : {db_path or 'non usato'}  (source='{db_source}')")
    if intensity_override:
        print(f"  Intensity: {intensity_override[0]:.2f}–{intensity_override[1]:.2f}  (override)")
    print(f"{'='*60}\n")

    rng   = np.random.default_rng(seed)
    pairs: list[tuple[Path, Path]] = []   # (out_path, orig_path)

    for img_path in img_paths:
        img_pil = Image.open(img_path).convert("RGB")
        for k in range(n_variants):
            out_name = f"{img_path.stem}_{suffix_tag}{k:02d}{img_path.suffix}"
            out_file = out_path / out_name

            def _call(fn, img):
                kwargs = {}
                if intensity_override:
                    kwargs["intensity_range"] = intensity_override
                if fn in (add_stripe_shadow, add_stripe_light):
                    if stripe_orientation != "random":
                        kwargs["orientation"] = stripe_orientation
                    if stripe_position is not None:
                        kwargs["position"] = stripe_position
                return fn(img, rng, **kwargs)

            if shadow_type in random_types:
                n_d = int(rng.integers(n_shadows_min, n_shadows_max + 1))
                out_img = img_pil
                for _ in range(n_d):
                    fn = pool[int(rng.integers(0, len(pool)))]
                    out_img = _call(fn, out_img)
            else:
                out_img = _call(disturbance_fn, img_pil)

            out_img.save(out_file)
            pairs.append((out_file, img_path))
        print(f"  {img_path.name}  →  {n_variants} varianti")

    total = len(pairs)
    print(f"\nGenerati {total} file in {out_path}/")

    if load_db:
        ds_root = Path(dataset_root)
        print(f"\nInserimento nel DB: {db_path} ...")
        inserted, skipped = load_shadow_frames_to_db(
            pairs, ds_root, Path(db_path), source=db_source
        )
        print(f"  Inseriti : {inserted}")
        if skipped:
            print(f"  Saltati  : {skipped}  (originale non trovato nel DB)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera immagini con ombre fittizie per test di robustezza PatchCore"
    )
    parser.add_argument("--input-dir",    required=True,
                        help="Directory sorgente con immagini .jpg/.png")
    parser.add_argument("--output-dir",   required=True,
                        help="Directory destinazione per le varianti con ombra")
    parser.add_argument("--n-variants",   type=int, default=1,
                        help="Numero di varianti per immagine (default: 1)")
    parser.add_argument("--shadow-type",  default="random",
                        choices=list(_ALL_FN),
                        help="Tipo: elliptical|directional|stripe|random|combined "
                             "(ombre) oppure light_elliptical|light_directional|"
                             "light_stripe|light_random (luci). Default: random")
    parser.add_argument("--shadows-min",  type=int, default=1,
                        help="Numero minimo di ombre per variante, solo con --shadow-type random "
                             "(default: 1)")
    parser.add_argument("--shadows-max",  type=int, default=3,
                        help="Numero massimo di ombre per variante, solo con --shadow-type random "
                             "(default: 3)")
    parser.add_argument("--seed",          type=int, default=42,
                        help="Seme RNG (default: 42)")
    parser.add_argument("--intensity-min", type=float, default=None,
                        help="Intensita' minima ombra [0.0-1.0] (override; default: valore del tipo)")
    parser.add_argument("--intensity-max", type=float, default=None,
                        help="Intensita' massima ombra [0.0-1.0] (override; default: valore del tipo)")
    parser.add_argument("--stripe-orientation", default="random",
                        choices=["horizontal", "vertical", "diagonal", "random"],
                        help="Orientamento striscia per stripe/light_stripe (default: random); "
                             "'diagonal' solo per light_stripe")
    parser.add_argument("--stripe-position", type=float, default=None,
                        help="Posizione centro striscia [0.0-1.0] (es. 0.5 = centro; default: casuale)")
    parser.add_argument("--db",           default=None,
                        help="Path al DB SQLite (necessario per --load-db)")
    parser.add_argument("--dataset-root", default=None,
                        help="Root della cartella Dataset (necessario per --load-db)")
    args = parser.parse_args()

    if (args.db is None) != (args.dataset_root is None):
        parser.error("--db e --dataset-root devono essere forniti insieme oppure nessuno dei due.")

    generate_shadow_variants(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_variants=args.n_variants,
        shadow_type=args.shadow_type,
        n_shadows_min=args.shadows_min,
        n_shadows_max=args.shadows_max,
        seed=args.seed,
        intensity_min=args.intensity_min,
        intensity_max=args.intensity_max,
        stripe_orientation=args.stripe_orientation,
        stripe_position=args.stripe_position,
        db_path=args.db,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
