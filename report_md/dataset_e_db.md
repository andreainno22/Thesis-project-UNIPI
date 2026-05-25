# Costruzione del Dataset e del Database

Questo documento racconta come è stato costruito il dataset usato per addestrare e valutare le pipeline di rilevamento di ostruzione delle vie di fuga, e come i frame e le annotazioni sono organizzati nel database operativo.

L'obiettivo non è solo descrivere il *cosa* (cartelle, formati, comandi), ma anche il *perché* di alcune scelte progettuali - in particolare quelle legate alla generazione sintetica e alla strategia di split.

---

## 1. Inquadramento e motivazione

Il problema affrontato dalla tesi (rilevare se una via di fuga è ostruita da un oggetto) non ha, allo stato attuale, un dataset pubblico di riferimento. La letteratura su anomaly detection lavora prevalentemente su MVTec-AD (difetti industriali) o VisA, che non contengono né porte di sicurezza né corridoi reali. I dataset di object detection su sedie a rotelle, carrelli o stretcher esistono, ma sono pensati per la classificazione dell'oggetto in sé, non per la sua presenza come ostacolo in un contesto specifico.

La scelta è stata quindi quella di **costruire un dataset proprietario per copia-incolla sintetica**, partendo da due fonti distinte:

1. **Background** di porte e corridoi senza ostruzione (la "scena normale" attesa).
2. **Oggetti scontornati** appartenenti alle categorie di ostacolo più rilevanti (carrelli, sedie a rotelle, scatole, barelle, cestini).

Questa separazione tra background e oggetti permette di generare un numero arbitrario di immagini ostruite preservando il controllo su variabili come scala, posizione e densità degli oggetti. Una conseguenza importante è che, per ogni immagine ostruita, esiste sempre una corrispondente immagine *non* ostruita della stessa scena: una proprietà che le pipeline P3 (PatchCore) e P4 (Siamese) sfruttano come riferimento.

---

## 2. Panoramica del flusso

```
                ┌──────────────────────────────┐
   Pinterest →  │  Dataset/non_ostruite/       │ ── background normali (porte, corridoi)
                └──────────┬───────────────────┘
                           │
                           │ + oggetti scontornati
                           ▼
   Roboflow  →  ┌──────────────────────────────┐
                │  Dataset/oggetti_fine_tuning │ ── 5 categorie YOLO
                └──────────┬───────────────────┘
                           │ extract_objects.py (rembg)
                           ▼
                ┌──────────────────────────────┐
                │  …/<categoria>/ritagliati/   │ ── PNG con canale α
                └──────────┬───────────────────┘
                           │ generate_synthetic_dataset.py
                           ▼
                ┌──────────────────────────────┐
                │  Dataset/ostruzioni_reali/   │ ── immagini sintetiche + .txt YOLO
                └──────────┬───────────────────┘
                           │ load_dataset_to_db.py
                           ▼
                ┌──────────────────────────────┐
                │  occlusion.db (SQLite)       │ ── frames + annotations_bbox
                └──────────────────────────────┘
```

Un quarto ramo, indipendente, alimenta il set di test per la robustezza alle ombre (vedi sez. 5).

---

## 3. Raccolta delle immagini di base

### 3.1 Background normali

Le immagini di porte di sicurezza e corridoi non ostruiti sono state raccolte tramite scraping di Pinterest, perché è una delle poche fonti che offre una grande varietà di scene reali con illuminazione e prospettive eterogenee. Sono state filtrate manualmente per scartare:

- Inquadrature troppo vicine (close-up di maniglia / pomello).
- Scene con ostruzioni.
- Render fotorealistici e immagini chiaramente generate da AI.
- Duplicati e quasi-duplicati.

Il risultato è organizzato per *venue*:

| Cartella                   | Venue       | Numero di immagini |
|----------------------------|-------------|--------------------|
| `non_ostruite/porte/`      | porta       | 550                |
| `non_ostruite/corridoi/`   | corridoio   | 550                |

I file sono rinominati nella forma `porta_NNN.jpg` / `corridoio_NNN.jpg`. Il nome stem (`porta_001`) diventa la *chiave di gruppo* (`source_group`) usata dal database per legare l'immagine pulita a tutte le sue varianti ostruite (vedi sez. 6.3).

Per uniformare il formato è stato usato `convert_to_jpg.py`, che converte PNG/WebP/HEIC in JPEG appiattendo l'eventuale canale alpha su sfondo bianco.

### 3.2 Oggetti

Le cinque categorie di ostacolo sono state scaricate come dataset YOLO indipendenti da Roboflow (versione `v1i.yolo26` per ciascuna):

| Categoria | Sorgente Roboflow                  | Cartella                           |
|-----------|------------------------------------|------------------------------------|
| carrello  | `cart.v1i.yolo26`                  | `oggetti_fine_tuning/cart/`        |
| sedia     | `only-wheelchair.v1i.yolo26`       | `oggetti_fine_tuning/only-wheelchair/` |
| scatola   | `boxes.v1i.yolo26`                 | `oggetti_fine_tuning/boxes/`       |
| barella   | `Stretcher.v1i.yolo26`             | `oggetti_fine_tuning/Stretcher/`   |
| cestino   | `waste_bin.v1i.yolo26`             | `oggetti_fine_tuning/waste_bin/`   |

Ogni dataset arriva con `images/` e `labels/` in formato YOLO. Le label sono state mantenute così come fornite (ognuna con la propria `class_id` interna), ma al momento della composizione sintetica tutte le categorie collassano su un'unica classe `obstacle` (class_id 0): per il problema in esame il *tipo* di ostacolo non è rilevante, conta solo la sua presenza nell'area di interesse.

---

## 4. Estrazione e composizione sintetica

### 4.1 Estrazione degli oggetti (`extract_objects.py`)

Per poter incollare un oggetto su uno sfondo arbitrario è necessario rimuovere il background originale. Lo script `extract_objects.py` esegue questo passaggio:

1. Legge ogni immagine di una categoria e il rispettivo `.txt` YOLO.
2. Per ogni bounding box, ritaglia il rettangolo dall'immagine originale.
3. Passa il ritaglio a **rembg** (modello U²-Net), che restituisce un PNG a 4 canali con il soggetto isolato dallo sfondo.
4. Salva il risultato in `oggetti_fine_tuning/<categoria>/train/ritagliati/`.

L'output è un insieme di PNG trasparenti, una per istanza, nominati `<nome_originale>_obj<idx>.png`.

Limiti noti del passaggio:

- rembg non è perfetto sui bordi sottili (ruote di un carrello, cinghie di una barella): in alcuni casi residua alone bianco.
- Oggetti con riflessi sul pavimento vengono inclusi nel ritaglio: questo in pratica aiuta il realismo perché preserva un'ombra di base, ma rende l'oggetto meno trasportabile su sfondi con illuminazione molto diversa.
- E' stato eseguito un filtraggio a mano delle immagini per rimuovere gli oggetti ritagliati male.

### 4.2 Composizione copy-paste (`generate_synthetic_dataset.py`)

Lo script di composizione produce le immagini ostruite finali. Per ogni background:

1. Si campionano da 1 a 3 oggetti (uniformemente, tra le 5 categorie).
2. Per ogni oggetto si cerca una posizione e una scala valide (fino a 30 tentativi).
3. L'oggetto è incollato preservando il canale alpha; le coordinate del bbox finale sono salvate in formato YOLO normalizzato.

I vincoli di posizionamento riflettono assunzioni semantiche sul problema:

| Parametro              | Valore           | Razionale |
|------------------------|------------------|-----------|
| `SCALE_RANGE`          | (0.15, 0.50)     | L'altezza dell'oggetto è 15–50 % dell'altezza del frame: copre sia ostacoli vicini sia ostacoli a distanza media. Oltre il 50 % l'oggetto invaderebbe il soffitto. |
| `MIN_Y_FRACTION`       | 0.50             | Il bordo superiore dell'oggetto non può essere sopra la metà del frame: un carrello o un cestino *appoggiati a terra* hanno il vertice basso vicino al pavimento. Evita "oggetti che galleggiano". |
| `MAX_IOU`              | 0.10             | Tra oggetti diversi nella stessa immagine si ammette al massimo il 10 % di sovrapposizione: assicura che ogni istanza sia chiaramente distinguibile. |
| `MAX_PLACEMENT_ATTEMPTS` | 30             | Numero di tentativi prima di rinunciare a piazzare l'oggetto. Tarato empiricamente: oltre, il guadagno marginale è trascurabile. |
| `NEGATIVE_BG_RATIO`    | 0.10             | Il 10 % dei background viene esplicitamente lasciato pulito (nessun oggetto incollato): serve a generare campioni negativi nello stesso formato di output. |

Il nome del file di output codifica il venue, l'indice del background e la variazione: `porte_0042_porta_042_v01.jpg`. Questa struttura è fondamentale per il database, perché un'espressione regolare nel loader (`SYNTHETIC_GROUP_RE`) estrae lo stem dell'immagine sorgente (`porta_042`) e lo usa come `source_group` per associare l'immagine sintetica al suo background normale.

| Cartella                       | Contenuto                                | Numero di immagini |
|--------------------------------|------------------------------------------|--------------------|
| `ostruzioni_reali/images/`     | composizioni copy-paste (.jpg)           | 990                |
| `ostruzioni_reali/labels/`     | bbox YOLO normalizzati (.txt)            | 990                |

### 4.3 Limiti del copy-paste rispetto a una scena reale

Il copy-paste introduce per costruzione alcune semplificazioni che vanno tenute presenti in fase di valutazione:

- **Coerenza illuminazione/colore**: l'oggetto porta con sé l'illuminazione della scena di provenienza, che raramente coincide con quella dello sfondo. Pipeline che si appoggiano su micro-discontinuità di bordo (es. PatchCore su feature WideResNet) possono trovare l'oggetto "troppo facile" rispetto a un'ostruzione reale.
- **Assenza di ombre proiettate**: l'oggetto non genera l'ombra che proietterebbe nel mondo reale. Questo è in parte un *bias di facilità* (manca un cue) ma anche un *bias di confusione* (un'ombra residua dall'immagine originale può trovarsi in posizione semanticamente sbagliata sul nuovo sfondo).
- **Assenza di occlusioni parziali del bordo**: l'oggetto è sempre interamente visibile, mai parzialmente uscito dal frame o nascosto da altri elementi della scena.

Questi limiti motivano l'introduzione del modulo di test descritto nella prossima sezione.

---

## 5. Test di robustezza: shadow injection

Una delle modalità di fallimento osservate durante lo sviluppo di P3 è il **falso positivo da ombra**: ombre nette proiettate da elementi architettonici (travi, scaffali, infissi) producono pattern che PatchCore considera anomali rispetto al riferimento. Per quantificare e mitigare il problema è stato aggiunto un terzo "tipo" di dato - varianti di immagini *normali* con ombre fittizie sovrapposte.

Lo script `generate_shadow_images.py` produce queste varianti applicando uno o più di tre tipi di ombra elementari:

- **Ellittica**: ellisse scura con bordi gaussiani sfumati, simula l'ombra di un oggetto sul pavimento o sulla parete.
- **Direzionale**: gradiente scuro proveniente da uno dei quattro lati, simula luce parzialmente bloccata (porta socchiusa, finestra laterale).
- **Striscia**: banda orizzontale o verticale con bordi sfumati, simula un'ombra da trave del soffitto o da scaffalatura.

La modalità `random` ne combina 1–3 a caso per ogni variante. Le varianti sono salvate in `Dataset/shadow_test/{normali_porte, normali_corridoi}/` e - opzionalmente - inserite nel database come frame normali (`is_normal = 1`, `source = "shadow"`), ereditando split e venue dall'immagine originale tramite il `file_path`. In questo modo il modulo `evaluate-db` può misurare il *false positive rate* sulle ombre senza alcuna modifica al codice di valutazione.

---

## 6. Database operativo

### 6.1 Perché un database

Le immagini in `Dataset/` sono organizzate per tipologia (normali / sintetiche / shadow) e per venue, ma le pipeline hanno bisogno di interrogare i dati lungo altre dimensioni: per split (train/val/test), per categoria di ostacolo, per reference frame, per source group. Mantenere queste relazioni come metadata di filesystem (cartelle annidate, suffissi nel nome) diventa rapidamente fragile.

La scelta è quindi un database SQLite singolo (`Aggregated_dataset_db/occlusion.db`), che funge da indice canonico: tutte le query usate dagli script di training/valutazione passano da qui, e i file in `Dataset/` sono puntati per path relativo.

SQLite è stato preferito a una soluzione più pesante (Postgres) per tre ragioni: il dataset sta nell'ordine delle migliaia di righe, il database è single-writer (lo script di caricamento), e un singolo file `.db` è portabile come parte della repository.

### 6.2 Schema

Lo schema completo è in [Aggregated_dataset_db/db_schema.sql](../Aggregated_dataset_db/db_schema.sql). Le tabelle rilevanti per il dataset sono due:

**`frames`** - un record per ogni immagine fisica sul disco:

| Colonna                | Significato                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `frame_id`             | path relativo da `Dataset/` - chiave naturale, stabile fra macchine.        |
| `venue_type`           | `porta` / `corridoio` / `scala`. Determinato dalla cartella o dal prefisso. |
| `is_normal`            | 1 se l'immagine non contiene ostruzioni, 0 altrimenti.                      |
| `occlusion_type`       | `none` / `synthetic_geometric` / `synthetic_copypaste` / `real`.            |
| `occlusion_level`      | `none` / `partial` / `full`.                                                |
| `split`                | `train` / `val` / `test`.                                                   |
| `source`               | provenienza del frame: `Pinterest`, `synthetic`, `shadow`, ecc.             |
| `source_group`         | chiave di gruppo per anti-leakage (vedi sez. 6.3).                          |
| `reference_frame_id`   | per le immagini sintetiche: punta al background normale di partenza.        |

**`annotations_bbox`** - bounding box in formato YOLO normalizzato (`cx, cy, w, h`), legati al frame tramite `frame_id` e con `label_class = obstacle` per tutte le ostruzioni sintetiche.

Le tabelle `annotations_masks`, `experiments` e `results` esistono già nello schema per supportare le pipeline P4, l'esecuzione di esperimenti e la raccolta di metriche, ma non sono popolate dal flusso di costruzione del dataset.

### 6.3 Split deterministico e anti-leakage

Il rischio principale in un dataset sintetico è il **leakage**: se la stessa scena di background appare in `train` come immagine normale e in `test` come immagine ostruita, una pipeline che usa il riferimento (P3, P4) vede di fatto durante il test informazione che ha visto in training.

Lo script `load_dataset_to_db.py` evita questo problema in due passaggi:

1. **Costruzione della chiave di gruppo (`source_group`)**:
   - Per le immagini normali, è semplicemente lo stem del file (`porta_042`).
   - Per le immagini sintetiche, è estratto dal nome con regex (`porte_0042_porta_042_v01` → `porta_042`).
   - Le immagini shadow ereditano il `source_group` del frame originale.
   
2. **Split via hash deterministico**: il gruppo viene hashato con SHA1, e il valore (proiettato in [0,1]) decide il bucket di split secondo i ratio configurati (default 80/20 train/val). Lo stesso gruppo finisce sempre nello stesso split, indipendentemente dall'ordine di caricamento o dall'aggiunta di nuove varianti. Una query di verifica nello schema review controlla che `COUNT(DISTINCT split)` per ogni `source_group` sia 1.

Questa strategia ha un effetto pratico importante: aggiungere domani nuove varianti sintetiche per `porta_042` non sposta il frame fra split. La distribuzione resta stabile e gli esperimenti sono confrontabili nel tempo.

---

## 7. Statistiche finali

A fine pipeline il dataset si compone di:

| Categoria                       | Quantità | Note                                  |
|---------------------------------|----------|---------------------------------------|
| Background normali (porte)      | 550      | da Pinterest                          |
| Background normali (corridoi)   | 550      | da Pinterest                          |
| Oggetti scontornati (5 classi)  | ~migliaia| varianti per categoria                |
| Immagini sintetiche ostruite    | 990      | copy-paste, 1–3 oggetti per frame     |
| Varianti shadow                 | variabile| generate on-demand per test FP        |

Lo split predefinito (80/20 per hash) produce ~880 frame in train e ~220 in val per le immagini normali, con un partizionamento analogo lato sintetico. Il `test` split è disponibile riducendo `--train-ratio + --val-ratio` sotto 1.

---

## 8. Riproducibilità - comandi essenziali

Tutti i comandi sono pensati per essere lanciati dalla root della repository.

**Estrazione oggetti** (una volta per ogni dataset Roboflow):
```bash
python src/extract_objects.py
# (modificare DATASET_DIR e OUTPUT_DIR nel sorgente, oppure parametrizzarlo)
```

**Generazione immagini sintetiche**:
```bash
python src/generate_synthetic_dataset.py
# parametri configurati nelle costanti in testa al file
```

**Inizializzazione e caricamento del DB**:
```bash
python src/init_db.py --db Aggregated_dataset_db/occlusion.db

python src/load_dataset_to_db.py \
  --db Aggregated_dataset_db/occlusion.db \
  --dataset-root Dataset \
  --split-mode hash --train-ratio 0.80 --val-ratio 0.20
```

**Generazione varianti shadow + inserimento nel DB**:
```bash
python src/generate_shadow_images.py \
  --input-dir  Dataset/non_ostruite/porte \
  --output-dir Dataset/shadow_test/normali_porte \
  --n-variants 3 --shadow-type random \
  --db Aggregated_dataset_db/occlusion.db \
  --dataset-root Dataset
```

---

## Riferimenti incrociati

- Schema completo del DB: [Aggregated_dataset_db/db_schema.sql](../Aggregated_dataset_db/db_schema.sql)
- Review motivata dello schema: [Aggregated_dataset_db/db_schema_review.md](../Aggregated_dataset_db/db_schema_review.md)
- Loader Python: [src/load_dataset_to_db.py](../src/load_dataset_to_db.py)
- Generatore sintetico: [src/generate_synthetic_dataset.py](../src/generate_synthetic_dataset.py)
- Estrattore oggetti: [src/extract_objects.py](../src/extract_objects.py)
- Generatore ombre: [src/generate_shadow_images.py](../src/generate_shadow_images.py)
