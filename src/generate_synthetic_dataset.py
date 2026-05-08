import os
import random
import glob
import hashlib
from PIL import Image

# ========================================================
# CONFIGURAZIONE DEI PERCORSI E PARAMETRI
# ========================================================

# 1. Cartelle contenenti le immagini di sfondo (venue diverse)
BACKGROUND_VENUES = {
    "porte": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\non_ostruite\porte",
    "corridoi": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\non_ostruite\corridoi",
}

# 2. Sfondi campionati per venue (None = usa tutti gli sfondi)
BACKGROUNDS_PER_VENUE = None

# 3. Cartelle degli oggetti per categoria (classe unica YOLO)
OBJECT_CATEGORIES = {
    "carrello": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\cart.v1i.yolo26\train",
    "sedia": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\only-wheelchair.v1i.yolo26\train",
    "scatola": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\boxes.v1i.yolo26\train",
    "barella": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\Stretcher.v1i.yolo26\train",
    "cestino": r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\waste_bin.v1i.yolo26\train",
}

CLASS_ID = 0

# 4. Cartella dove verrà salvato il nuovo dataset pronto per l'addestramento YOLO
OUTPUT_DIR = r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\ostruzioni_reali"

# 5. Oggetti per immagine (1-3 random)
MIN_OBJECTS_PER_IMAGE = 1
MAX_OBJECTS_PER_IMAGE = 3

# 6. Variazioni per sfondo (scala + posizione diverse)
VARIATIONS_PER_BACKGROUND_RANGE = (1, 1)

# 7. Parametri di posizionamento e overlap
SCALE_RANGE = (0.15, 0.50)
MIN_Y_FRACTION = 0.50
MAX_IOU = 0.10
MAX_PLACEMENT_ATTEMPTS = 30

# 8. Riproducibilita (None = casuale)
RNG_SEED = None

# 9. Percentuale di background da lasciare puliti (negative samples)
NEGATIVE_BG_RATIO = 0.10
NEGATIVE_BG_SEED = 42

# ========================================================
# FUNZIONI PRINCIPALI
# ========================================================

def load_backgrounds_by_venue():
    """Carica i percorsi di tutti gli sfondi, campionandoli per venue."""
    backgrounds_by_venue = {}
    for venue, venue_dir in BACKGROUND_VENUES.items():
        venue_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            venue_paths.extend(glob.glob(os.path.join(venue_dir, ext)))

        if not venue_paths:
            print(f"ATTENZIONE: Nessuno sfondo trovato in {venue_dir}")
            backgrounds_by_venue[venue] = []
            continue

        total_paths = len(venue_paths)
        venue_paths = sorted(venue_paths)
        if BACKGROUNDS_PER_VENUE and total_paths > BACKGROUNDS_PER_VENUE:
            venue_paths = random.sample(venue_paths, BACKGROUNDS_PER_VENUE)

        print(f"Venue {venue}: selezionati {len(venue_paths)}/{total_paths} sfondi.")
        backgrounds_by_venue[venue] = venue_paths

    return backgrounds_by_venue


def seed_from_value(value: str) -> int:
    return int(hashlib.sha1(value.encode("utf-8")).hexdigest()[:8], 16)


def select_negative_backgrounds(venue_paths, venue_name: str) -> set[str]:
    if NEGATIVE_BG_RATIO <= 0:
        return set()

    bg_stems = [os.path.splitext(os.path.basename(p))[0] for p in venue_paths]
    count = int(len(bg_stems) * NEGATIVE_BG_RATIO)
    if count <= 0:
        return set()

    seed = NEGATIVE_BG_SEED if NEGATIVE_BG_SEED is not None else (RNG_SEED or 42)
    rng = random.Random(seed_from_value(f"{seed}:{venue_name}"))
    return set(rng.sample(bg_stems, count))


def load_objects_by_category():
    """Carica i percorsi di tutti gli oggetti per categoria."""
    objects_by_category = {}
    for category, obj_dir in OBJECT_CATEGORIES.items():
        obj_paths = []
        for ext in ["*.png"]:  # Gli oggetti scontornati sono PNG
            obj_paths.extend(glob.glob(os.path.join(obj_dir, ext)))

        objects_by_category[category] = obj_paths
        print(f"Categoria {category}: trovati {len(obj_paths)} oggetti ritagliati.")

    return objects_by_category

def create_yolo_label(bg_w, bg_h, fg_x, fg_y, fg_w, fg_h, class_id):
    """Calcola le coordinate YOLO (normalizzate) per l'oggetto incollato."""
    # Calcola il centro dell'oggetto
    center_x = fg_x + (fg_w / 2.0)
    center_y = fg_y + (fg_h / 2.0)
    
    # Normalizza rispetto alle dimensioni dello sfondo (valori tra 0 e 1)
    norm_x = center_x / bg_w
    norm_y = center_y / bg_h
    norm_w = fg_w / bg_w
    norm_h = fg_h / bg_h
    
    # Assicurati che i valori siano formattati a 6 cifre decimali
    return f"{class_id} {norm_x:.6f} {norm_y:.6f} {norm_w:.6f} {norm_h:.6f}"


def compute_iou(box_a, box_b):
    """Calcola la IoU tra due bounding box (x, y, w, h)."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2 = ax + aw
    ay2 = ay + ah
    bx2 = bx + bw
    by2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0:
        return 0.0

    union_area = (aw * ah) + (bw * bh) - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def find_non_overlapping_placement(bg_w, bg_h, fg_img, existing_boxes):
    """Trova una posizione che rispetti la soglia di overlap."""
    aspect_ratio = fg_img.width / fg_img.height

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        scale_factor = random.uniform(SCALE_RANGE[0], SCALE_RANGE[1])
        new_fg_h = int(bg_h * scale_factor)
        if new_fg_h <= 1:
            continue

        new_fg_w = int(new_fg_h * aspect_ratio)
        if new_fg_w <= 1:
            continue

        if new_fg_w >= bg_w or new_fg_h >= bg_h:
            continue

        min_y = int(bg_h * MIN_Y_FRACTION)
        max_y = bg_h - new_fg_h
        if max_y < 0:
            continue

        if max_y <= min_y:
            paste_y = max(0, max_y)
        else:
            paste_y = random.randint(min_y, max_y)

        max_x = bg_w - new_fg_w
        if max_x < 0:
            continue

        paste_x = random.randint(0, max_x)

        candidate_box = (paste_x, paste_y, new_fg_w, new_fg_h)
        if all(compute_iou(candidate_box, box) < MAX_IOU for box in existing_boxes):
            resized = fg_img.resize((new_fg_w, new_fg_h), Image.Resampling.LANCZOS)
            return resized, candidate_box

    return None

def generate_dataset():
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    # Prepara le cartelle di output per immagini e label (stile YOLO)
    images_out_dir = os.path.join(OUTPUT_DIR, "images")
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)

    backgrounds_by_venue = load_backgrounds_by_venue()
    objects_by_category = load_objects_by_category()

    available_categories = [c for c, paths in objects_by_category.items() if paths]
    if not any(backgrounds_by_venue.values()) or not available_categories:
        print("Errore: Sfondi o oggetti mancanti. Controlla i percorsi in alto.")
        return

    total_generated = 0
    total_skipped = 0

    for venue, bg_paths in backgrounds_by_venue.items():
        if not bg_paths:
            continue

        generated_for_venue = 0
        skipped_backgrounds = 0
        negative_bg_stems = select_negative_backgrounds(bg_paths, venue)
        if negative_bg_stems:
            print(
                f"Venue {venue}: {len(negative_bg_stems)} background lasciati puliti (negative)"
            )

        for bg_idx, bg_path in enumerate(bg_paths, start=1):
            try:
                bg_img = Image.open(bg_path).convert("RGB")
            except Exception as e:
                print(f"Errore caricamento sfondo {bg_path}: {e}")
                continue

            bg_w, bg_h = bg_img.size
            bg_filename = os.path.splitext(os.path.basename(bg_path))[0]
            if bg_filename in negative_bg_stems:
                skipped_backgrounds += 1
                bg_img.close()
                continue
            num_variations = random.randint(
                VARIATIONS_PER_BACKGROUND_RANGE[0],
                VARIATIONS_PER_BACKGROUND_RANGE[1],
            )

            for variation_idx in range(1, num_variations + 1):
                composed_img = bg_img.copy()
                labels_list = []
                placed_boxes = []

                num_objects = random.randint(MIN_OBJECTS_PER_IMAGE, MAX_OBJECTS_PER_IMAGE)

                for _ in range(num_objects):
                    category = random.choice(available_categories)
                    fg_path = random.choice(objects_by_category[category])

                    try:
                        with Image.open(fg_path) as fg_img:
                            fg_img = fg_img.convert("RGBA")
                            placement = find_non_overlapping_placement(
                                bg_w, bg_h, fg_img, placed_boxes
                            )
                    except Exception:
                        continue

                    if placement is None:
                        continue

                    resized_img, (paste_x, paste_y, new_fg_w, new_fg_h) = placement
                    composed_img.paste(resized_img, (paste_x, paste_y), resized_img)
                    placed_boxes.append((paste_x, paste_y, new_fg_w, new_fg_h))

                    yolo_label = create_yolo_label(
                        bg_w, bg_h, paste_x, paste_y, new_fg_w, new_fg_h, CLASS_ID
                    )
                    labels_list.append(yolo_label)

                if not labels_list:
                    total_skipped += 1
                    continue

                output_filename = (
                    f"{venue}_{bg_idx:04d}_{bg_filename}_v{variation_idx:02d}"
                )
                img_save_path = os.path.join(images_out_dir, f"{output_filename}.jpg")
                composed_img.save(img_save_path, "JPEG", quality=95)

                txt_save_path = os.path.join(labels_out_dir, f"{output_filename}.txt")
                with open(txt_save_path, "w") as f:
                    f.write("\n".join(labels_list))

                total_generated += 1
                generated_for_venue += 1

                if total_generated % 50 == 0:
                    print(f"Generate {total_generated} immagini sintetiche...")

            bg_img.close()

        print(f"Venue {venue}: generate {generated_for_venue} immagini.")
        if skipped_backgrounds:
            print(f"Venue {venue}: background puliti (skip) {skipped_backgrounds}.")

    print(f"\nCOMPLETATO! Dataset sintetico salvato in: {OUTPUT_DIR}")
    print(f"Immagini generate: {total_generated}")
    if total_skipped:
        print(f"Immagini scartate (nessun oggetto piazzato): {total_skipped}")
    print("Pronto per l'addestramento con YOLO!")

if __name__ == "__main__":
    generate_dataset()
