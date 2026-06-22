# Costruzione del Dataset e del Database

Questo documento racconta come è stato costruito il dataset usato per addestrare e valutare le pipeline di rilevamento di ostruzione delle vie di fuga, e come i frame e le annotazioni sono organizzati nel database operativo.

L'obiettivo non è solo descrivere il *cosa* (cartelle, formati, comandi), ma anche il *perché* di alcune scelte progettuali - in particolare quelle legate alla generazione sintetica, alla strategia di split e alla separazione tra dati di costruzione e test reale.

Una distinzione attraversa tutto il documento: da un lato il **materiale di costruzione** (background da Pinterest, oggetti scontornati, composizioni copy-paste, varianti di robustezza a ombre e luce), sintetico o raccolto dal web, che serve a costruire, calibrare e validare i modelli; dall'altro un **test set reale**, fotografato sul campo (sez. 6), che è l'unica vera misura finale ed è comune a tutte le pipeline della tesi.

---

## 1. Inquadramento e motivazione

Il problema affrontato dalla tesi (rilevare se una via di fuga è ostruita da un oggetto) non ha, allo stato attuale, un dataset pubblico di riferimento. La letteratura su anomaly detection lavora prevalentemente su MVTec-AD (difetti industriali) o VisA, che non contengono né porte di sicurezza né corridoi reali. I dataset di object detection su sedie a rotelle, carrelli o stretcher esistono, ma sono pensati per la classificazione dell'oggetto in sé, non per la sua presenza come ostacolo in un contesto specifico.

La scelta è stata quindi quella di **costruire un dataset proprietario per copia-incolla sintetica**, partendo da due fonti distinte:

1. **Background** di porte e corridoi senza ostruzione (la "scena normale" attesa).
2. **Oggetti scontornati** appartenenti alle categorie di ostacolo più rilevanti (carrelli, sedie a rotelle, scatole, barelle, cestini).

Questa separazione tra background e oggetti permette di generare un numero arbitrario di immagini ostruite preservando il controllo su variabili come scala, posizione e densità degli oggetti. Una conseguenza importante è che, per ogni immagine ostruita, esiste sempre una corrispondente immagine *non* ostruita della stessa scena: una proprietà che le pipeline P3 (PatchCore) e P4 (Siamese) sfruttano come riferimento.

Questo materiale sintetico, però, resta una *approssimazione* del mondo reale (vedi i limiti in sez. 4.3). Per questo motivo la valutazione finale non avviene su di esso, ma su un dataset reale acquisito sul campo (sez. 6), comune a tutte le pipeline.

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

A questo flusso principale (che produce il pool di **training/validazione**, `split = train`) si affiancano due rami indipendenti:

- **(a) Varianti di robustezza** - a partire dalle immagini normali si generano versioni con ombre e illuminazione alterata, per misurare e mitigare i falsi positivi (sez. 5).
- **(b) Test set reale «poli ingegneria»** - foto scattate sul campo, l'unico vero set di test (`split = test`), comune a tutte le pipeline (sez. 6).

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

I file sono rinominati nella forma `porta_NNN.jpg` / `corridoio_NNN.jpg`. Il nome stem (`porta_001`) diventa la *chiave di gruppo* (`source_group`) usata dal database per legare l'immagine pulita a tutte le sue varianti ostruite (vedi sez. 8.3).

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

Questi limiti motivano sia le varianti di robustezza della prossima sezione, sia - soprattutto - la scelta di valutare le pipeline su un test set reale (sez. 6).

---

## 5. Dati di robustezza: ombre e illuminazione

I metodi *reference-based* (P3, P4) confrontano una scena con un riferimento e segnalano come anomalo ciò che si discosta. Questo li rende sensibili a variazioni che cambiano l'aspetto della scena **senza** introdurre un'ostruzione: ombre nette e cambi di illuminazione sono le due principali. Per quantificare e mitigare questi falsi positivi sono state generate due famiglie di varianti a partire dalle immagini *normali*.

### 5.1 Shadow injection

Una delle modalità di fallimento osservate durante lo sviluppo di P3 è il **falso positivo da ombra**: ombre nette proiettate da elementi architettonici (travi, scaffali, infissi) producono pattern che PatchCore considera anomali rispetto al riferimento. Per quantificare e mitigare il problema è stato aggiunto un terzo "tipo" di dato - varianti di immagini *normali* con ombre fittizie sovrapposte.

Lo script `generate_shadow_images.py` produce queste varianti applicando uno o più di tre tipi di ombra elementari:

- **Ellittica**: ellisse scura con bordi gaussiani sfumati, simula l'ombra di un oggetto sul pavimento o sulla parete.
- **Direzionale**: gradiente scuro proveniente da uno dei quattro lati, simula luce parzialmente bloccata (porta socchiusa, finestra laterale).
- **Striscia**: banda orizzontale o verticale con bordi sfumati, simula un'ombra da trave del soffitto o da scaffalatura.

La modalità `random` ne combina 1–3 a caso per ogni variante. Le varianti sono salvate in `Dataset/shadow_test/{normali_porte, normali_corridoi}/` e - opzionalmente - inserite nel database come frame normali (`is_normal = 1`, `source = "shadow"`), ereditando split e venue dall'immagine originale tramite il `file_path`. In questo modo il modulo `evaluate-db` può misurare il *false positive rate* sulle ombre senza alcuna modifica al codice di valutazione. Nel DB corrente sono presenti 990 varianti shadow (495 porte + 495 corridoi).

### 5.2 Light injection - varianti di illuminazione

Le varianti di **illuminazione** sono generate dallo **stesso** script `generate_shadow_images.py`: oltre al pool di ombre, lo script contiene un pool speculare di "luci" che si attiva passando `--shadow-type light_random` (o un tipo specifico `light_elliptical` / `light_directional` / `light_stripe`). Concettualmente sono il complemento delle ombre - invece di moltiplicare i pixel per un fattore < 1, aggiungono regioni più chiare (fattore > 1):

- **`light_elliptical`**: fascio luminoso ellittico con bordi morbidi (lucernario, lampada puntata, riflesso), con una leggera tinta calda (R↑, B↓).
- **`light_directional`**: gradiente luminoso progressivo da uno dei quattro lati (finestra laterale, porta aperta su ambiente illuminato, lampada a parete).
- **`light_stripe`**: striscia luminosa orizzontale, verticale o **diagonale**; quest'ultima simula un fascio di luce solare, anch'esso con tinta calda.

La modalità `light_random` combina 1–3 di queste primitive per variante. Le varianti sono salvate in `Dataset/light_test/{normali_porte, normali_corridoi}/` (suffisso `_light00`) e registrate nel DB come frame normali (`is_normal = 1`, `source = "light"`), ereditando venue e split dall'immagine originale - esattamente come le ombre (la stessa funzione `load_shadow_frames_to_db` distingue le due famiglie tramite il campo `source`). Servono a misurare e ridurre i falsi positivi dovuti a cambi di luce e alimentano gli esperimenti di ablazione sulla robustezza (es. `pipeline3/results/ablation_L1_light_only.csv`). Nel DB corrente sono presenti 1100 varianti light (550 porte + 550 corridoi).

---

## 6. Test set reale sul campo: «poli ingegneria»

Tutto il materiale descritto finora (background da Pinterest, oggetti scontornati, composizioni copy-paste, varianti shadow/light) è sintetico o raccolto dal web: serve a *costruire* e *validare* i modelli, ma non rappresenta una vera condizione operativa. Per la valutazione finale è stato quindi costruito un secondo dataset, **fotografato direttamente sul campo presso i poli di Ingegneria dell'Università di Pisa**. Questo set è l'unico vero test della tesi ed è **comune a tutte le pipeline (P1-P4)**: tutti i metodi vengono confrontati esattamente sugli stessi frame reali.

### 6.1 Acquisizione e struttura

Sono state riprese **6 porte/uscite distinte** (`porta_1` … `porta_6`), tutte di tipo `venue = porta`, a risoluzione **960×1280** (orientamento verticale). Per ciascuna scena il protocollo di acquisizione prevede tre tipi di frame:

- **1 immagine di riferimento** (`porta_N_ref.jpg`): la "scena normale attesa", senza ostruzioni, usata come baseline dai metodi reference-based (P3, P4) e come àncora per l'annotazione del telaio della porta.
- **più varianti non ostruite** della stessa scena (`porta_N_k.jpg`, cartella `non_ostruite/poli ingegneria/`).
- **più varianti ostruite**, con un oggetto reale fisicamente collocato davanti all'uscita (cartella `ostruzioni_poli_ingegneria/`).

La numerosità per scena non è uniforme: `porta_3` e `porta_4` hanno 9+9 varianti, le altre quattro 3+3.

| Scena    | Riferimenti | Non ostruite (hard neg) | Ostruite (reali) |
|----------|:-----------:|:-----------------------:|:----------------:|
| porta_1  | 1           | 3                       | 3                |
| porta_2  | 1           | 3                       | 3                |
| porta_3  | 1           | 9                       | 9                |
| porta_4  | 1           | 9                       | 9                |
| porta_5  | 1           | 3                       | 3                |
| porta_6  | 1           | 3                       | 3                |
| **Totale** | **6**     | **30**                  | **30**           |

Il dataset è quindi **bilanciato**: 30 frame normali (non ostruiti) contro 30 frame ostruiti. I 6 riferimenti sono anch'essi normali, ma hanno ruolo di baseline e non vengono usati come query di test; la valutazione si misura sui 30 vs 30.

### 6.2 Hard negatives e ostruzioni reali

La distinzione tra le due classi è progettata per essere realistica e non banale:

- I frame non ostruiti **non** sono semplici ri-scatti del riferimento: sono **hard negatives**, cioè la stessa porta ripresa con variazioni di disturbo che *non* costituiscono un'ostruzione - cambi di punto di vista e inquadratura, illuminazione diversa, un cartello/avviso affisso sull'anta, persone di passaggio. Il loro scopo è misurare il **false positive rate** in condizioni in cui la scena cambia ma la via di fuga resta libera. (Per esempio, in `porta_1` una variante mostra un poster affisso sull'anta e un leggero cambio di prospettiva, ma la porta resta sgombra.)
- I frame ostruiti contengono un **ostacolo reale** appoggiato davanti all'uscita (es. il cestino "CARTA PAPER" spostato contro la porta, carrelli, ecc.): nessun copy-paste, nessuna ombra sintetica, con illuminazione e ombre coerenti con la scena. Sono quindi molto più rappresentativi di un'ostruzione reale rispetto alle composizioni copy-paste della sez. 4.

Questo rende il set adatto a stressare entrambe le code del problema: la **detection** delle ostruzioni vere (recall) e la **robustezza ai falsi allarmi** (specificità) sotto variazioni legittime della scena.

### 6.3 Annotazione del telaio e registrazione nel DB

Sui 6 riferimenti è stato annotato il **telaio della porta** come insieme di poligoni ROI (`left_jamb`, `right_jamb`, `architrave`, `threshold` e `door_region`), importati nella tabella `roi_polygons` (`img_w = 960`, `img_h = 1280`). In valutazione questo abilita il *gating per componenti connesse* descritto nelle istruzioni operative: un'anomalia viene considerata valida solo se cade nella regione della porta, riducendo i falsi positivi dovuti allo sfondo variabile dietro le porte vetrate (rilevante per P3).

La registrazione nel database segue convenzioni precise:

| Gruppo                    | Cartella / pattern                              | `is_normal` | `occlusion_type` | `source`            | `reference_frame_id`      |
|---------------------------|-------------------------------------------------|:-----------:|------------------|---------------------|---------------------------|
| Riferimenti               | `non_ostruite/poli ingegneria/porta_N_ref.jpg`  | 1           | `none`           | `poli_ingegneria`   | (nessuno)                 |
| Non ostruite (hard neg)   | `non_ostruite/poli ingegneria/porta_N_k.jpg`    | 1           | `none`           | `background_change` | → `porta_N_ref.jpg`       |
| Ostruite (reali)          | `ostruzioni_poli_ingegneria/porta_N_k.jpg`      | 0           | `real`           | `poli_ingegneria`   | → `porta_N_ref.jpg`       |

Punti chiave:

- Tutti i **66 frame** (6 ref + 30 hard neg + 30 ostruiti) hanno `split = test` e `source_group = poli_ingegneria`. Avere un unico `source_group` garantisce che l'intero set resti nel test e non finisca mai nel pool di training: nessun *leakage* tra il materiale sintetico di costruzione e il test reale.
- Ogni variante (ostruita o meno) punta al riferimento della propria scena via `reference_frame_id`, così i metodi reference-based sanno quale baseline usare.
- Le ostruzioni reali **non** hanno bounding box (`annotations_bbox` vuota per questo set): la ground truth è a livello di frame (ostruito / libero), non di box. Le bbox presenti nel DB provengono unicamente dalle composizioni copy-paste.

---

## 7. Uso dei dataset nelle pipeline

Le diverse fonti di dati hanno ruoli distinti a seconda della pipeline. Il principio comune è che la **valutazione finale avvenga sempre sul test set reale «poli ingegneria»** (sez. 6), mentre i dati sintetici/scrappati servono a costruire, calibrare o validare preliminarmente i singoli metodi.

- **Test comune (tutte le pipeline)**: il set reale `poli ingegneria`. Tutte le pipeline vengono confrontate sugli stessi 30 frame ostruiti / 30 liberi, con i 6 riferimenti per-scena come baseline.

- **P1 - YOLO (object detection)**: gli oggetti scontornati (5 categorie Roboflow) e, se necessario, le composizioni copy-paste (`ostruzioni_reali`) verranno usati per il **fine-tuning** del detector sulle categorie di ostacolo. *(L'impiego del copy-paste per il fine-tuning è ancora da confermare; qui lo si assume.)*

- **P2 - baseline door-centric (Talebi et al.)**: **non** usa il dataset copy-paste. Seguendo l'approccio di Talebi et al., a partire dalle immagini di riferimento vengono generate ostruzioni sintetiche tipo "nuvole di pixel" sovrapposte alla regione della porta; la generazione è quindi specifica della pipeline e non attinge alle composizioni della sez. 4.

- **P3 - PatchCore (anomaly detection)**: il dataset copy-paste (`ostruzioni_reali`) è usato come **dataset preliminare di validazione teorica** del modello, per verificare che PatchCore separi correttamente normale e anomalo in condizioni controllate *prima* del test reale (questo è descritto nella relativa sezione di tesi). Le varianti shadow e light (sez. 5) servono a quantificare e mitigare i falsi positivi da ombre e cambi di luce; il riferimento per-scena di `poli ingegneria` alimenta il memory bank in fase di test.

- **P4 - Siamese change detection**: approccio e dati ancora da definire. Condividerà comunque il test set reale comune.

---

## 8. Database operativo

### 8.1 Perché un database

Le immagini in `Dataset/` sono organizzate per tipologia (normali / sintetiche / shadow / light / reali) e per venue, ma le pipeline hanno bisogno di interrogare i dati lungo altre dimensioni: per split (train/val/test), per categoria di ostacolo, per reference frame, per source group. Mantenere queste relazioni come metadata di filesystem (cartelle annidate, suffissi nel nome) diventa rapidamente fragile.

La scelta è quindi un database SQLite singolo (`Aggregated_dataset_db/occlusion.db`), che funge da indice canonico: tutte le query usate dagli script di training/valutazione passano da qui, e i file in `Dataset/` sono puntati per path relativo.

SQLite è stato preferito a una soluzione più pesante (Postgres) per tre ragioni: il dataset sta nell'ordine delle migliaia di righe, il database è single-writer (lo script di caricamento), e un singolo file `.db` è portabile come parte della repository.

### 8.2 Schema

Lo schema completo è in [Aggregated_dataset_db/db_schema.sql](../Aggregated_dataset_db/db_schema.sql). Le tabelle rilevanti per il dataset sono tre:

**`frames`** - un record per ogni immagine fisica sul disco:

| Colonna                | Significato                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `frame_id`             | path relativo da `Dataset/` - chiave naturale, stabile fra macchine.        |
| `venue_type`           | `porta` / `corridoio` / `scala`. Determinato dalla cartella o dal prefisso. |
| `is_normal`            | 1 se l'immagine non contiene ostruzioni, 0 altrimenti.                      |
| `occlusion_type`       | `none` / `synthetic_geometric` / `synthetic_copypaste` / `real`.            |
| `occlusion_level`      | `none` / `partial` / `full`.                                                |
| `split`                | `train` / `val` / `test`.                                                   |
| `source`               | provenienza del frame: `Pinterest`, `synthetic`, `shadow`, `light`, `background_change`, `poli_ingegneria`. |
| `source_group`         | chiave di gruppo per anti-leakage (vedi sez. 8.3).                          |
| `reference_frame_id`   | per le immagini sintetiche e per le varianti di `poli ingegneria`: punta al background/riferimento normale di partenza. |

**`annotations_bbox`** - bounding box in formato YOLO normalizzato (`cx, cy, w, h`), legati al frame tramite `frame_id` e con `label_class = obstacle` per tutte le ostruzioni sintetiche. Sono popolati **solo** dalle composizioni copy-paste; le ostruzioni reali di `poli ingegneria` non hanno bbox.

**`roi_polygons`** - poligoni del telaio porta (`left_jamb`, `right_jamb`, `architrave`, `threshold`, `door_region`, …) annotati sui riferimenti, usati per il gating in valutazione (vedi sez. 6.3).

Le tabelle `annotations_masks`, `experiments` e `results` esistono già nello schema per supportare le pipeline P4, l'esecuzione di esperimenti e la raccolta di metriche, ma non sono popolate dal flusso di costruzione del dataset.

### 8.3 Split deterministico e anti-leakage

Il rischio principale in un dataset sintetico è il **leakage**: se la stessa scena di background appare in `train` come immagine normale e in `test` come immagine ostruita, una pipeline che usa il riferimento (P3, P4) vede di fatto durante il test informazione che ha visto in training.

Lo script `load_dataset_to_db.py` evita questo problema in due passaggi:

1. **Costruzione della chiave di gruppo (`source_group`)**:
   - Per le immagini normali, è semplicemente lo stem del file (`porta_042`).
   - Per le immagini sintetiche, è estratto dal nome con regex (`porte_0042_porta_042_v01` → `porta_042`).
   - Le immagini shadow ereditano il `source_group` del frame originale.
   - Tutto il set `poli ingegneria` condivide un unico `source_group` (`poli_ingegneria`), che lo tiene compatto nel `test`.

2. **Split via hash deterministico**: il gruppo viene hashato con SHA1, e il valore (proiettato in [0,1]) decide il bucket di split secondo i ratio configurati. Lo stesso gruppo finisce sempre nello stesso split, indipendentemente dall'ordine di caricamento o dall'aggiunta di nuove varianti. Una query di verifica nello schema review controlla che `COUNT(DISTINCT split)` per ogni `source_group` sia 1.

**Stato operativo attuale del DB**: il meccanismo di hashing resta disponibile nel loader, ma nella configurazione corrente tutto il materiale di costruzione (Pinterest, light, shadow, synthetic) è assegnato a `split = train`, mentre il test reale `poli ingegneria` è l'unico `split = test`. Non esiste uno split di **validazione** fisso (`val`) **per scelta**: la validazione viene effettuata via *cross-validation* sul pool di training, quindi i fold sono costruiti a runtime e non materializzati come colonna nel DB. In altre parole, oggi la separazione operativa non è "80/20 train/val sul sintetico" ma "tutto il sintetico in training (con cross-validation per la validazione), test solo sul reale".

Questa strategia ha un effetto pratico importante: aggiungere domani nuove varianti sintetiche per `porta_042` non sposta il frame fra split, e il test reale resta sempre disgiunto dal materiale di costruzione. La distribuzione resta stabile e gli esperimenti sono confrontabili nel tempo.

---

## 9. Statistiche finali

Lo stato corrente del database (`occlusion.db`) si compone di **4246 frame** e **1900 bounding box**. Lo split operativo separa nettamente il materiale di costruzione (tutto in `train`) dal test reale (`poli ingegneria`, in `test`); non esiste uno split `val` fisso perché la validazione è effettuata via cross-validation sul pool di training (vedi sez. 8.3).

| Fonte (`source`)             | Venue              | `is_normal` | `occlusion_type`      | `split` | N        |
|------------------------------|--------------------|:-----------:|-----------------------|---------|---------:|
| Pinterest (background)       | porta + corridoio  | 1           | `none`                | train   | 1100     |
| light (varianti luce)        | porta + corridoio  | 1           | `none`                | train   | 1100     |
| shadow (varianti ombra)      | porta + corridoio  | 1           | `none`                | train   | 990      |
| synthetic (copy-paste)       | porta + corridoio  | 0           | `synthetic_copypaste` | train   | 990      |
| poli_ingegneria (riferimenti)| porta              | 1           | `none`                | test    | 6        |
| background_change (hard neg) | porta              | 1           | `none`                | test    | 30       |
| poli_ingegneria (ostruzioni) | porta              | 0           | `real`                | test    | 30       |
| **Totale**                   |                    |             |                       |         | **4246** |

Riepilogo per split: **train = 4180** (1100 + 1100 + 990 + 990), **test = 66** (6 + 30 + 30); nessuno split `val` fisso (validazione via cross-validation sul training).

Riepilogo per `occlusion_type`: `none` = 3226, `synthetic_copypaste` = 990, `real` = 30.

Materiale a monte non indicizzato nel DB come frame: gli oggetti scontornati delle 5 categorie (`oggetti_fine_tuning/<categoria>/train/ritagliati/`), nell'ordine delle migliaia di PNG trasparenti, che alimentano la composizione copy-paste e l'eventuale fine-tuning di P1.

---

## 10. Riproducibilità - comandi essenziali

Tutti i comandi sono pensati per essere lanciati dalla root della repository; gli script vivono in `pipeline3/src/`.

**Estrazione oggetti** (una volta per ogni dataset Roboflow):
```bash
python pipeline3/src/extract_objects.py
# (modificare DATASET_DIR e OUTPUT_DIR nel sorgente, oppure parametrizzarlo)
```

**Generazione immagini sintetiche**:
```bash
python pipeline3/src/generate_synthetic_dataset.py
# parametri configurati nelle costanti in testa al file
```

**Inizializzazione e caricamento del DB**:
```bash
python pipeline3/src/init_db.py --db Aggregated_dataset_db/occlusion.db

python pipeline3/src/load_dataset_to_db.py \
  --db Aggregated_dataset_db/occlusion.db \
  --dataset-root Dataset \
  --split-mode hash --train-ratio 0.80 --val-ratio 0.20
```

**Generazione varianti shadow + inserimento nel DB**:
```bash
python pipeline3/src/generate_shadow_images.py \
  --input-dir  Dataset/non_ostruite/porte \
  --output-dir Dataset/shadow_test/normali_porte \
  --n-variants 3 --shadow-type random \
  --db Aggregated_dataset_db/occlusion.db \
  --dataset-root Dataset
```

**Generazione varianti light (stesso script, pool luci)**:
```bash
python pipeline3/src/generate_shadow_images.py \
  --input-dir  Dataset/non_ostruite/porte \
  --output-dir Dataset/light_test/normali_porte \
  --n-variants 1 --shadow-type light_random \
  --db Aggregated_dataset_db/occlusion.db \
  --dataset-root Dataset
```

**Annotazione + import del telaio porta (poli ingegneria)**:
```bash
python pipeline3/src/annotate_door_roi.py \
  --images "Dataset/non_ostruite/poli ingegneria/*_ref.jpg" \
  --out-dir Dataset/roi

python pipeline3/src/import_roi_to_db.py \
  --db Aggregated_dataset_db/occlusion.db \
  --roi-dir Dataset/roi \
  --dataset-root Dataset
```

---

## Riferimenti incrociati

- Schema completo del DB: [Aggregated_dataset_db/db_schema.sql](../Aggregated_dataset_db/db_schema.sql)
- Review motivata dello schema: [Aggregated_dataset_db/db_schema_review.md](../Aggregated_dataset_db/db_schema_review.md)
- Loader Python: [pipeline3/src/load_dataset_to_db.py](../pipeline3/src/load_dataset_to_db.py)
- Generatore sintetico: [pipeline3/src/generate_synthetic_dataset.py](../pipeline3/src/generate_synthetic_dataset.py)
- Estrattore oggetti: [pipeline3/src/extract_objects.py](../pipeline3/src/extract_objects.py)
- Generatore ombre: [pipeline3/src/generate_shadow_images.py](../pipeline3/src/generate_shadow_images.py)
- Annotatore telaio porta: [pipeline3/src/annotate_door_roi.py](../pipeline3/src/annotate_door_roi.py)
- Import ROI nel DB: [pipeline3/src/import_roi_to_db.py](../pipeline3/src/import_roi_to_db.py)
