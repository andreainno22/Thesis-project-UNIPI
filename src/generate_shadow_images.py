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
  python src/generate_shadow_images.py \\
    --input-dir  Dataset/non_ostruite/porte \\
    --output-dir Dataset/shadow_test/normali \\
    --n-variants 3 --shadow-type random \\
    --shadows-min 1 --shadows-max 3 \\
    --db Aggregated_dataset_db/occlusion.db \\
    --dataset-root Dataset

  # Solo genera immagini, senza DB
  python src/generate_shadow_images.py \\
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
) -> Image.Image:
    """
    Aggiunge una striscia scura orizzontale o verticale con bordi sfumati.
    Simula l'ombra proiettata da una trave del soffitto, uno scaffale a muro
    o un traverso architettonico - pattern molto comune in corridoi industriali.
    """
    arr       = np.array(img_pil).astype(np.float32)
    h, w      = arr.shape[:2]
    intensity = rng.uniform(*intensity_range)
    horizontal = bool(rng.integers(0, 2))

    mask = np.zeros((h, w), dtype=np.float32)
    if horizontal:
        band_half = max(2, int(h * rng.uniform(*width_range) / 2))
        center    = int(rng.integers(band_half, h - band_half))
        mask[max(0, center - band_half):min(h, center + band_half), :] = 1.0
    else:
        band_half = max(2, int(w * rng.uniform(*width_range) / 2))
        center    = int(rng.integers(band_half, w - band_half))
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


# Pool atomico usato dalla modalità random
_SHADOW_POOL = [add_elliptical_shadow, add_directional_shadow, add_stripe_shadow]

_SHADOW_FN = {
    "elliptical":  add_elliptical_shadow,
    "directional": add_directional_shadow,
    "stripe":      add_stripe_shadow,
    "combined":    add_combined_shadow,
    "random":      None,   # gestito esplicitamente in generate_shadow_variants
}


# ── DB loading ────────────────────────────────────────────────────────────────

def _normalize_relpath(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def load_shadow_frames_to_db(
    pairs: list[tuple[Path, Path]],   # (shadow_path, original_path)
    dataset_root: Path,
    db_path: Path,
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
                    None,        # obstacle_class - non rilevante per shadow
                    occlusion_type, occlusion_level,
                    split, "shadow",
                    source_group, ref_id,
                ),
            )
            inserted += 1

        conn.commit()
    finally:
        conn.close()

    return inserted, skipped


# ── Generazione immagini ──────────────────────────────────────────────────────

def generate_shadow_variants(
    input_dir: str,
    output_dir: str,
    n_variants: int = 1,
    shadow_type: str = "random",
    n_shadows_min: int = 1,
    n_shadows_max: int = 3,
    seed: int = 42,
    db_path: str | None = None,
    dataset_root: str | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> None:
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if shadow_type not in _SHADOW_FN:
        print(f"ERRORE: shadow_type '{shadow_type}' non valido. Scegli fra: {list(_SHADOW_FN)}")
        sys.exit(1)

    if n_shadows_min < 1 or n_shadows_max < n_shadows_min:
        print("ERRORE: --shadows-min deve essere >= 1 e <= --shadows-max")
        sys.exit(1)

    shadow_fn = _SHADOW_FN[shadow_type]   # None per random, gestito sotto

    img_paths = sorted(p for p in in_path.iterdir() if p.suffix.lower() in extensions)
    if not img_paths:
        print(f"ERRORE: nessuna immagine trovata in {in_path}")
        sys.exit(1)

    load_db = db_path is not None and dataset_root is not None

    shadows_desc = (
        f"random [{n_shadows_min}–{n_shadows_max}] da pool "
        f"{{ellittica, direzionale, striscia}}"
        if shadow_type == "random"
        else shadow_type
    )
    print(f"\n{'='*60}")
    print(f"GENERATE SHADOW IMAGES")
    print(f"  Input    : {in_path}  ({len(img_paths)} immagini)")
    print(f"  Output   : {out_path}")
    print(f"  Tipo     : {shadows_desc}")
    print(f"  Varianti : {n_variants}")
    print(f"  Seed     : {seed}")
    print(f"  DB       : {db_path or 'non usato'}")
    print(f"{'='*60}\n")

    rng   = np.random.default_rng(seed)
    pairs: list[tuple[Path, Path]] = []   # (shadow_path, orig_path)

    for img_path in img_paths:
        img_pil = Image.open(img_path).convert("RGB")
        for k in range(n_variants):
            out_name   = f"{img_path.stem}_shadow{k:02d}{img_path.suffix}"
            out_file   = out_path / out_name

            if shadow_type == "random":
                n_sh = int(rng.integers(n_shadows_min, n_shadows_max + 1))
                shadow_img = img_pil
                for _ in range(n_sh):
                    fn = _SHADOW_POOL[int(rng.integers(0, len(_SHADOW_POOL)))]
                    shadow_img = fn(shadow_img, rng)
            else:
                shadow_img = shadow_fn(img_pil, rng)

            shadow_img.save(out_file)
            pairs.append((out_file, img_path))
        print(f"  {img_path.name}  →  {n_variants} varianti")

    total = len(pairs)
    print(f"\nGenerati {total} file in {out_path}/")

    if load_db:
        ds_root = Path(dataset_root)
        print(f"\nInserimento nel DB: {db_path} ...")
        inserted, skipped = load_shadow_frames_to_db(pairs, ds_root, Path(db_path))
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
                        choices=list(_SHADOW_FN),
                        help="Tipo ombra: elliptical | directional | stripe | "
                             "random (default) | combined")
    parser.add_argument("--shadows-min",  type=int, default=1,
                        help="Numero minimo di ombre per variante, solo con --shadow-type random "
                             "(default: 1)")
    parser.add_argument("--shadows-max",  type=int, default=3,
                        help="Numero massimo di ombre per variante, solo con --shadow-type random "
                             "(default: 3)")
    parser.add_argument("--seed",         type=int, default=42,
                        help="Seme RNG (default: 42)")
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
        db_path=args.db,
        dataset_root=args.dataset_root,
    )


if __name__ == "__main__":
    main()
