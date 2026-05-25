# DB Operativo SQLite - Schema Finale

## DDL Completo

### `frames` - indice dei frame fisici

```sql
CREATE TABLE frames (
    frame_id               TEXT PRIMARY KEY,
    file_path              TEXT NOT NULL,
    venue_type             TEXT NOT NULL CHECK(venue_type IN ('porta','corridoio','scala')),
    is_normal              BOOLEAN NOT NULL,
    obstacle_class         TEXT,                    -- NULL se is_normal=True
    occlusion_type         TEXT NOT NULL CHECK(occlusion_type IN
                               ('none','synthetic_geometric','synthetic_copypaste','real')),
    occlusion_level        TEXT NOT NULL CHECK(occlusion_level IN ('none','partial','full')),
    split                  TEXT NOT NULL CHECK(split IN ('train','val','test')),
    source                 TEXT NOT NULL,            -- CDNet / OpenImages / custom / synthetic
    source_group           TEXT,                     -- raggruppa frame dalla stessa scena/sequenza
                                                     -- es. "cdnet_office_001", NULL per img indipendenti
    reference_frame_id     TEXT REFERENCES frames(frame_id)    -- frame normale di riferimento (P2/P3)
);

CREATE INDEX idx_frames_split       ON frames(split);
CREATE INDEX idx_frames_venue       ON frames(venue_type);
CREATE INDEX idx_frames_normal      ON frames(is_normal);
CREATE INDEX idx_frames_source_grp  ON frames(source_group);
```

> [!NOTE]
> **`source_group`** - valorizzato solo quando i frame hanno correlazione visiva
> (es. stessa sequenza CDNet, stessa sessione foto custom). Frame da Open Images o
> Pinterest → NULL. Regola di split: frame con lo stesso `source_group` finiscono
> tutti nello stesso split.

---

### `annotations_bbox` - per P1 e P2

```sql
CREATE TABLE annotations_bbox (
    ann_id        INTEGER PRIMARY KEY,
    frame_id      TEXT NOT NULL REFERENCES frames(frame_id),
    label_class   TEXT NOT NULL,         -- obstacle / door
    cx            REAL NOT NULL,         -- YOLO normalized
    cy            REAL NOT NULL,
    w             REAL NOT NULL,
    h             REAL NOT NULL,
    txt_path      TEXT                   -- path al .txt YOLO
);

CREATE INDEX idx_bbox_frame ON annotations_bbox(frame_id);
```

---

### `annotations_masks` - per P4

```sql
CREATE TABLE annotations_masks (
    mask_id       INTEGER PRIMARY KEY,
    frame_id      TEXT NOT NULL REFERENCES frames(frame_id),
    ref_frame_id  TEXT NOT NULL REFERENCES frames(frame_id),  -- frame normale di riferimento
    mask_path     TEXT NOT NULL           -- path alla PNG binaria
);

CREATE INDEX idx_mask_frame ON annotations_masks(frame_id);
```

---

### `experiments` - tracking delle run

```sql
CREATE TABLE experiments (
    exp_id          INTEGER PRIMARY KEY,
    pipeline        TEXT NOT NULL CHECK(pipeline IN ('P1','P2','P3','P4')),
    model_variant   TEXT NOT NULL,        -- yolov8n / yolov11s / efficientnet_b0 / mobilenetv2 ...
    dataset_filter  JSON,                 -- query usata per selezionare i frame
    hyperparams     JSON,                 -- include aug config on-the-fly per P3
    run_date        DATETIME NOT NULL DEFAULT (datetime('now')),
    artifact_path   TEXT,                 -- P1/P2/P4: best.pt - P3: memory_bank.pt
    status          TEXT NOT NULL CHECK(status IN ('running','done','failed'))
                        DEFAULT 'running'
);
```

> [!IMPORTANT]
> **`artifact_path`** sostituisce il vecchio `weights_path`.
> PatchCore (P3) non salva pesi ma un memory bank (`.pt` o `.pkl`).

---

### `results` - metriche per esperimento

```sql
CREATE TABLE results (
    result_id          INTEGER PRIMARY KEY,
    exp_id             INTEGER NOT NULL REFERENCES experiments(exp_id),
    fold               TEXT,                -- NULL per pipeline senza CV; 'fold_1'..'fold_k' per P1
    venue_type         TEXT CHECK(venue_type IN ('porta','corridoio','scala')),  -- NULL = globale
    -- metriche detection (P1, P2, P4)
    mAP50              REAL,
    mAP50_95           REAL,
    precision          REAL,
    recall             REAL,
    F1                 REAL,
    -- metriche anomaly detection (P3)
    auroc              REAL,               -- AUROC su score normalizzati (z-score per-camera)
    pro_score          REAL,
    anomaly_threshold  REAL,
    -- metriche deployment
    latency_ms         REAL,
    false_alarm_rate   REAL,
    false_negative_rate REAL               -- FN / (FN+TP) = 1 − recall
);

CREATE INDEX idx_results_exp ON results(exp_id);
```

> [!NOTE]
> **`fold`** è NULL per P3 (unsupervised, nessun train/val/test) e P2 (baseline).
> Per P1 con cross-validation viene valorizzato con l'etichetta del fold (es. `'fold_1'`).
> Il contesto completo dei dati usati in ogni run è in `experiments.dataset_filter` (JSON).
>
> **`auroc` in P3** è calcolato su score normalizzati per camera: `(score − μ) / σ`,
> dove μ e σ provengono dalla calibrazione sulla scena normale. Questo rende l'AUROC
> confrontabile tra camere diverse e tra pipeline diverse.

---

## Campi rimossi vs schema originale

| Campo rimosso | Tabella | Motivo |
|---|---|---|
| `camera_id` | `frames` | Non pertinente: dataset di immagini singole, non stream video |
| `sequence_id` | `frames` | Sostituito da `source_group` - più leggero, copre il caso CDNet |
| `split` | `results` | Ridondante: P3 è unsupervised (nessun split), P1 usa CV (fold dinamici). Il contesto dati è in `experiments.dataset_filter` |

## Campi aggiunti vs schema originale

| Campo aggiunto | Tabella | Motivo |
|---|---|---|
| `source_group` | `frames` | Anti-leakage per frame correlati (CDNet, sessioni foto) |
| `reference_frame_id` | `frames` | Calibration frame P2 e riferimento P3 per frame sintetici |
| `artifact_path` | `experiments` | Rinominato da `weights_path` (P3 salva memory bank) |
| `fold` | `results` | Etichetta fold per P1 con cross-validation; NULL per P2/P3 |
| `auroc` | `results` | Metrica primaria PatchCore (score normalizzati z-score) |
| `pro_score` | `results` | Metrica pixel-level anomaly detection |
| `anomaly_threshold` | `results` | Soglia usata per binarizzare anomaly score |

---

## Augmentation P3 - riepilogo strategia

P3 (PatchCore) è **unsupervised**: non ha fase di training nel senso classico.
L'augmentazione serve a costruire la memory bank e a calibrare la soglia
a partire da una singola immagine di riferimento per camera.

```
ref_img  (singola immagine normale per camera)
│
├── augment_reference(seed=42,  n=15, include_original=False)
│       → varianti fotometriche: luminosità, contrasto, saturazione,
│         temperatura colore (shift R/B), rumore gaussiano, blur leggero
│       → NON flip/rotazione (camera fissa)
│       → usate per costruire la MEMORY BANK
│
├── augment_reference(seed=1000, n=10, include_original=False)
│       → stesse trasformazioni, seed diverso → statisticamente indipendenti
│       → usate per calibrare μ e σ della soglia (UNSEEN dalla bank)
│
└── ref_img originale (non entra nella bank né nella calibrazione)
        → usato come CAMPIONE NORMALE nel test (genuinamente unseen)
```

> [!NOTE]
> Nessun file di augmentazione viene scritto in `frames`. I parametri
> (n_augments, n_cal, seed, coreset_p, k_sigma) sono salvati in
> `experiments.hyperparams` (JSON). La memory bank serializzata (`.pt`)
> è referenziata da `experiments.artifact_path`.

---

## Utilizzo frame per pipeline - query rapide

```sql
-- P1: frame con bbox per obstacle detection (CV: filtrare per fold esternamente)
SELECT f.*, b.* FROM frames f
JOIN annotations_bbox b ON f.frame_id = b.frame_id
WHERE f.is_normal = 0 AND b.label_class = 'obstacle';

-- P2: frame con bbox porta
SELECT f.*, b.* FROM frames f
JOIN annotations_bbox b ON f.frame_id = b.frame_id
WHERE b.label_class = 'door';

-- P3: reference frames che hanno almeno una ostruita accoppiata
SELECT f.frame_id, f.file_path, f.venue_type FROM frames f
WHERE f.is_normal = 1
AND EXISTS (
    SELECT 1 FROM frames o
    WHERE o.is_normal = 0 AND o.reference_frame_id = f.frame_id
);

-- P3: immagini ostruite con la loro reference (per evaluate_from_db)
SELECT o.frame_id, o.file_path, o.venue_type, o.reference_frame_id
FROM frames o
WHERE o.is_normal = 0 AND o.reference_frame_id IS NOT NULL;

-- P4: training siamese (coppie con change mask)
SELECT f.*, m.ref_frame_id, m.mask_path FROM frames f
JOIN annotations_masks m ON f.frame_id = m.frame_id;

-- Anti-leakage: verifica che source_group non sia diviso tra split
SELECT source_group, GROUP_CONCAT(DISTINCT split) AS splits
FROM frames
WHERE source_group IS NOT NULL
GROUP BY source_group
HAVING COUNT(DISTINCT split) > 1;  -- deve restituire 0 righe
```
