"""
convert_to_jpg.py
-----------------
Converte tutte le immagini in una cartella (e sottocartelle) in formato JPG.
Formati supportati: PNG, BMP, TIFF, WEBP, GIF, ecc.
Le immagini già in JPG/JPEG vengono saltate.

MI SERVE SOLO PER CONVERTIRE IN JPG LE IMMAGINI DI BACKGROUND, NON LE IMMAGINI 
OSTRUITE, NE' GLI OGGETTI DI FINE TUNING

Uso:
    python convert_to_jpg.py --input_dir <cartella> [--quality 95] [--delete_original]
"""

import argparse
import os
from pathlib import Path
from PIL import Image
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("ATTENZIONE: pillow-heif non trovato, i file .heic verranno saltati. Installa con: pip install pillow-heif")

# Formati da convertire
SUPPORTED_FORMATS = {".png", ".bmp", ".tiff", ".tif", ".webp", ".gif", ".ppm", ".pgm", ".heic"}
SKIP_FORMATS = {".jpg", ".jpeg"}


def convert_image(src_path: Path, quality: int, delete_original: bool) -> bool:
    """Converte una singola immagine in JPG. Ritorna True se convertita."""
    dst_path = src_path.with_suffix(".jpg")

    try:
        with Image.open(src_path) as img:
            # Converti in RGB (rimuove canale alpha se presente, es. PNG con trasparenza)
            if img.mode in ("RGBA", "LA", "P"):
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                background.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            img.save(dst_path, "JPEG", quality=quality, optimize=True)

        if delete_original:
            src_path.unlink()
            print(f"  ✓ Convertita e originale eliminato: {src_path.name} → {dst_path.name}")
        else:
            print(f"  ✓ Convertita: {src_path.name} → {dst_path.name}")
        return True

    except Exception as e:
        print(f"  ✗ Errore su {src_path.name}: {e}")
        return False


def convert_folder(input_dir: str, quality: int, delete_original: bool, recursive: bool):
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"Errore: la cartella '{input_dir}' non esiste.")
        return

    pattern = "**/*" if recursive else "*"
    all_files = list(input_path.glob(pattern))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS]
    skipped = [f for f in all_files if f.is_file() and f.suffix.lower() in SKIP_FORMATS]

    print(f"\n Cartella: {input_path.resolve()}")
    print(f"   Immagini da convertire : {len(image_files)}")
    print(f"   JPG già presenti (skip): {len(skipped)}\n")

    converted, errors = 0, 0
    for img_path in image_files:
        ok = convert_image(img_path, quality, delete_original)
        if ok:
            converted += 1
        else:
            errors += 1

    print(f"\n Completato: {converted} convertite, {errors} errori, {len(skipped)} saltate.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converte immagini in JPG.")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Cartella contenente le immagini da convertire"
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        help="Qualità JPG (1-100, default: 95)"
    )
    parser.add_argument(
        "--delete_original", action="store_true",
        help="Elimina i file originali dopo la conversione"
    )
    parser.add_argument(
        "--no_recursive", action="store_true",
        help="Non cercare nelle sottocartelle"
    )

    args = parser.parse_args()
    convert_folder(
        input_dir=args.input_dir,
        quality=args.quality,
        delete_original=args.delete_original,
        recursive=not args.no_recursive
    )
