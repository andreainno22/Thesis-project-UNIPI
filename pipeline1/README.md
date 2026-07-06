# P1 - Object detection puro (obstruction / no-obstruction)

YOLO fine-tunato a **classe singola** ("obstruction"): rileva la *presenza* di
un'ostruzione su una via di fuga, non il tipo di oggetto. Confrontato con una
baseline COCO non fine-tunata (mappata a "obstruction") e valutato in
cross-validation sul dataset copy-paste + sul test set reale `poli ingegneria`.

## Dati

- **Positivi (train)**: 990 immagini copy-paste (`ostruzioni_reali`), label YOLO
  gia presenti, classe 0.
- **Negativi (train)**: 110 sfondi di riferimento *held-out* (i pruned del
  copy-paste, `Dataset/ostruzioni_reali/pruned_backgrounds.txt`), usati come
  background images (nessun file label). Gli sfondi usati dal copy-paste NON
  sono usabili come negativi (leakage).
- **Test**: `ostruzioni_poli_ingegneria` (30 reali) + normali poli (`split=test`
  nel DB). Le 30 reali vanno **annotate** (vedi `annotate_bboxes.py`) per il mAP;
  le metriche image-level non richiedono le bbox.

Il grouping della cross-validation usa `source_group` (id sfondo) cosi lo stesso
sfondo non finisce sia in train che in val.

## Pipeline

```bash
PY="C:/Users/andrea/anaconda3/envs/tesi_env/python.exe"   # env: tesi_env

# 1) prepara i fold (5-fold grouped CV) - scrive pipeline1/data/*.yaml
$PY pipeline1/src/prepare_yolo_dataset.py --folds 5 --seed 42

# 2a) baseline COCO (non fine-tunata) sul test set, image-level
$PY pipeline1/src/evaluate_yolo.py image-level --weights coco --split test \
    --out-csv pipeline1/results/baseline_imagelevel.csv

# 2b) annota le bbox delle 30 immagini reali (per il mAP sul test)
$PY pipeline1/src/annotate_bboxes.py \
    --images "Dataset/ostruzioni_poli_ingegneria/*.jpg" \
    --out-dir Dataset/ostruzioni_poli_ingegneria/labels

# 3) fine-tuning in cross-validation (yolo11n per edge; --device 0 se hai GPU)
$PY pipeline1/src/train_yolo.py --folds-dir pipeline1/data \
    --model yolo11n.pt --epochs 100 --name cv

# 4) metriche detection (mAP) aggregate sui fold
$PY pipeline1/src/evaluate_yolo.py cv-map --name cv \
    --out-csv pipeline1/results/cv_map.csv

# 5) modello finale su tutti i dati, poi image-level sul test
$PY pipeline1/src/train_yolo.py --data pipeline1/data/all.yaml \
    --model yolo11n.pt --epochs 100 --name final
$PY pipeline1/src/evaluate_yolo.py image-level \
    --weights pipeline1/runs/final/weights/best.pt --split test \
    --out-csv pipeline1/results/finetuned_imagelevel.csv
```

## Note

- **Env**: usare sempre `tesi_env`. `cv2.imread` fallisce sui path accentati del
  progetto (`Universita`); ultralytics usa un suo `imread` (imdecode) e funziona.
- **Edge target** (Raspberry Pi / Jetson Nano): restare su size nano/small.
  Rif. Miglionico, Ducange et al. (IJCNN 2025): YOLOv11n ~10.6 FPS su Jetson
  Nano con TensorRT.
- **Baseline mapping**: `coco_obstruction_map.py` definisce quali classi COCO
  contano come ostruzione (esclusi persona, oggetti piccoli, ...). E' una
  scelta di design, facilmente editabile.
- `evaluate_yolo.py ... --write-db` traccia l'esperimento nelle tabelle
  `experiments`/`results` (pipeline='P1').
