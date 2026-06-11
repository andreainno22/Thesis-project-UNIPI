import os
from PIL import Image
from rembg import remove
import glob

# ========================================================
# CONFIGURAZIONE DEI PERCORSI
# ========================================================
# Inserisci qui il percorso della cartella "train" del dataset scaricato da Roboflow
DATASET_DIR = r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\gallery-dl\oggetti_fine_tuning\Waste container.v1i.yolo26\train"

# Cartella dove verranno salvati i PNG trasparenti ritagliati
OUTPUT_DIR = r"C:\Users\andrea\OneDrive - University of Pisa\Università\Unipi\tesi\gallery-dl\oggetti_fine_tuning\Waste container.v1i.yolo26\train\ritagliati"

def process_yolo_dataset(dataset_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Trova tutte le immagini (cerca jpg, jpeg, png)
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(glob.glob(os.path.join(dataset_dir, "images", ext)))
    
    print(f"Trovate {len(image_paths)} immagini da analizzare.")
    counter = 0

    for img_path in image_paths:
        # Trova il file di testo (label) corrispondente
        base_name = os.path.basename(img_path)
        name_without_ext = os.path.splitext(base_name)[0]
        label_path = os.path.join(dataset_dir, "labels", name_without_ext + ".txt")
        
        if not os.path.exists(label_path):
            continue
            
        try:
            # Apri l'immagine
            img = Image.open(img_path).convert("RGB")
            img_w, img_h = img.size
            
            # Leggi il file delle label (potrebbero esserci più oggetti in una foto)
            with open(label_path, "r") as f:
                lines = f.readlines()
                
            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                # YOLO format: class_id x_center y_center width height (normalizzati da 0 a 1)
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                
                # Converti le coordinate YOLO in pixel (x_min, y_min, x_max, y_max)
                x_min = int((x_center - w / 2) * img_w)
                y_min = int((y_center - h / 2) * img_h)
                x_max = int((x_center + w / 2) * img_w)
                y_max = int((y_center + h / 2) * img_h)
                
                # Evita di andare fuori dai bordi dell'immagine
                x_min, y_min = max(0, x_min), max(0, y_min)
                x_max, y_max = min(img_w, x_max), min(img_h, y_max)
                
                # Ritaglia l'oggetto dall'immagine originale
                cropped_img = img.crop((x_min, y_min, x_max, y_max))
                
                # Rimuovi lo sfondo usando l'AI di rembg!
                # Questo restituirà un'immagine con sfondo trasparente (canale Alpha)
                img_transparent = remove(cropped_img)
                
                # Salva l'immagine finale
                output_filename = f"{name_without_ext}_obj{idx}.png"
                output_path = os.path.join(output_dir, output_filename)
                img_transparent.save(output_path, "PNG")
                
                counter += 1
                if counter % 50 == 0:
                    print(f"Elaborati {counter} oggetti...")
                    
        except Exception as e:
            print(f"Errore con l'immagine {img_path}: {e}")

    print(f"\nFINITO! Generati {counter} PNG trasparenti nella cartella: {output_dir}")

if __name__ == "__main__":
    process_yolo_dataset(DATASET_DIR, OUTPUT_DIR)
