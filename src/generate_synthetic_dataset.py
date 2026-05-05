import os
import random
import glob
from PIL import Image

# ========================================================
# CONFIGURAZIONE DEI PERCORSI E PARAMETRI
# ========================================================

# 1. Cartella contenente le immagini di sfondo (corridoi, porte vuote)
BACKGROUNDS_DIR = r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\non_ostruite\porte"

# 2. Dizionario che mappa la cartella degli oggetti al loro Class ID (YOLO)

OBJECT_CLASSES = {
    r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\cart.v1i.yolo26\train": 0, # Ostacolo generico
    r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\only-wheelchair.v1i.yolo26\train": 0, # Ostacolo generico   
    r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\Stretcher.v1i.yolo26\train": 0, # Ostacolo generico
    r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\oggetti_fine_tuning\waste_bin.v1i.yolo26\train": 0 # Ostacolo generico
}

# 3. Cartella dove verrà salvato il nuovo dataset pronto per l'addestramento YOLO
OUTPUT_DIR = r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\Dataset\ostruzioni_reali\porte"

# 4. Quante immagini sintetiche vuoi generare in totale?
NUM_IMAGES_TO_GENERATE = 10

# 5. Numero massimo di oggetti da inserire in una singola foto
MAX_OBJECTS_PER_IMAGE = 1

# ========================================================
# FUNZIONI PRINCIPALI
# ========================================================

def load_paths():
    """Carica i percorsi di tutti gli sfondi e di tutti gli oggetti divisi per classe."""
    # Carica sfondi
    bg_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        bg_paths.extend(glob.glob(os.path.join(BACKGROUNDS_DIR, ext)))
    
    if not bg_paths:
        print(f"ATTENZIONE: Nessuno sfondo trovato in {BACKGROUNDS_DIR}")
        
    # Carica oggetti
    objects_dict = {}
    for obj_dir, class_id in OBJECT_CLASSES.items():
        obj_paths = []
        for ext in ["*.png"]: # Gli oggetti scontornati sono PNG
            obj_paths.extend(glob.glob(os.path.join(obj_dir, ext)))
        
        objects_dict[class_id] = obj_paths
        print(f"Classe {class_id}: trovati {len(obj_paths)} oggetti ritagliati.")
        
    return bg_paths, objects_dict

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

def generate_dataset():
    # Prepara le cartelle di output per immagini e label (stile YOLO)
    images_out_dir = os.path.join(OUTPUT_DIR, "images")
    labels_out_dir = os.path.join(OUTPUT_DIR, "labels")
    os.makedirs(images_out_dir, exist_ok=True)
    os.makedirs(labels_out_dir, exist_ok=True)
    
    bg_paths, objects_dict = load_paths()
    
    if not bg_paths or not any(objects_dict.values()):
        print("Errore: Sfondi o oggetti mancanti. Controlla i percorsi in alto.")
        return

    # Estrai tutte le classi disponibili che hanno almeno un'immagine
    available_classes = [c for c, paths in objects_dict.items() if paths]

    # Ordina i percorsi degli sfondi per avere un ordine deterministico
    bg_paths = sorted(bg_paths)

    for i in range(NUM_IMAGES_TO_GENERATE):
        # 1. Scegli uno sfondo in ordine
        bg_path = bg_paths[i % len(bg_paths)]
        try:
            bg_img = Image.open(bg_path).convert("RGB")
        except Exception as e:
            print(f"Errore caricamento sfondo {bg_path}: {e}")
            continue
            
        bg_w, bg_h = bg_img.size
        
        # Scegli quanti oggetti incollare in questa immagine (da 1 a MAX)
        num_objects = random.randint(1, MAX_OBJECTS_PER_IMAGE)
        
        labels_list = []
        
        # 2. Incolla gli oggetti
        for _ in range(num_objects):
            class_id = random.choice(available_classes)
            fg_path = random.choice(objects_dict[class_id])
            
            try:
                fg_img = Image.open(fg_path).convert("RGBA") # Importante: RGBA per la trasparenza
            except Exception as e:
                continue
                
            # --- LOGICA DI RIDIMENSIONAMENTO E POSIZIONAMENTO ---
            # Scala l'oggetto casualmente per simulare distanza (es. dal 30% all'80% dell'altezza dello sfondo)
            scale_factor = random.uniform(0.15, 0.50)
            new_fg_h = int(bg_h * scale_factor)
            
            # Mantieni le proporzioni originali dell'oggetto
            aspect_ratio = fg_img.width / fg_img.height
            new_fg_w = int(new_fg_h * aspect_ratio)
            
            fg_img = fg_img.resize((new_fg_w, new_fg_h), Image.Resampling.LANCZOS)
            
            # Posizionamento: Varietà massima sul pavimento
            # Y: Mettiamolo nella metà inferiore per evitare oggetti sul soffitto
            min_y = int(bg_h * 0.5) # Parte dal 50% dell'altezza
            max_y = bg_h - new_fg_h
            
            if max_y <= min_y:
                # Se l'oggetto è troppo grande per lo sfondo, mettilo alla base
                paste_y = bg_h - new_fg_h
            else:
                paste_y = random.randint(min_y, max_y)
                
            # X: Casuale su tutta la larghezza dell'immagine (per massima varianza)
            max_x = bg_w - new_fg_w
            paste_x = random.randint(0, max(0, max_x))
            
            # 3. Incolla usando il canale Alpha (trasparenza) come maschera
            bg_img.paste(fg_img, (paste_x, paste_y), fg_img)
            
            # 4. Genera la label YOLO
            yolo_label = create_yolo_label(bg_w, bg_h, paste_x, paste_y, new_fg_w, new_fg_h, class_id)
            labels_list.append(yolo_label)
            
        # 5. Salva l'immagine generata e il file txt
        bg_filename = os.path.splitext(os.path.basename(bg_path))[0]
        
        # Nome base
        output_filename = f"{bg_filename}_ostruita"
        # Se stiamo ciclando di nuovo sugli sfondi, aggiungiamo un numero per evitare sovrascritture
        if i >= len(bg_paths):
            output_filename = f"{output_filename}_{i // len(bg_paths)}"
        
        img_save_path = os.path.join(images_out_dir, f"{output_filename}.jpg")
        bg_img.save(img_save_path, "JPEG", quality=95)
        
        txt_save_path = os.path.join(labels_out_dir, f"{output_filename}.txt")
        with open(txt_save_path, "w") as f:
            f.write("\n".join(labels_list))
            
        if (i+1) % 10 == 0:
            print(f"Generate {i+1}/{NUM_IMAGES_TO_GENERATE} immagini sintetiche...")

    print(f"\nCOMPLETATO! Dataset sintetico salvato in: {OUTPUT_DIR}")
    print(f"Pronto per l'addestramento con YOLO!")

if __name__ == "__main__":
    generate_dataset()
