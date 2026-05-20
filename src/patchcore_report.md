# PatchCore per il Rilevamento di Occlusioni — Report Metodologico

## 1. Pipeline PatchCore: funzionamento dettagliato

### 1.1 Architettura generale

PatchCore è un sistema di **anomaly detection non supervisionato** basato su memoria. Non richiede immagini anomale durante la fase di costruzione: apprende esclusivamente dalla scena normale di riferimento. Il principio fondamentale è che un'immagine è anomala se le sue feature locali si trovano lontane da quelle memorizzate durante la fase di build.

Nel contesto applicativo, ogni via di fuga è monitorata da una telecamera dedicata. Per ciascuna telecamera viene costruita una **memory bank indipendente** a partire da un singolo frame di riferimento della scena normale. Questa scelta riflette esattamente il caso reale di deploy: la camera non cambia inquadratura, e la scena normale ha caratteristiche fotometriche stabili.

---

### 1.2 Fase di Build — Costruzione della Memory Bank

#### Augmentazione fotometrica

A partire dalla singola immagine di riferimento `ref_img`, vengono generate `n_augments` varianti fotometriche (default: 15) usando un RNG con seed fisso (seed=42). Le augmentazioni applicate sono:

| Trasformazione | Range | Motivazione |
|---|---|---|
| Luminosità | ×[0.85, 1.15] | Mattina / sera / luce artificiale |
| Contrasto | ×[0.90, 1.10] | Luce diretta vs diffusa |
| Saturazione | ×[0.90, 1.10] | LED vs incandescente vs fluorescente |
| Shift canale R | [−8, +8] | Temperatura colore |
| Shift canale B | [−20, +20] | Temperatura colore |
| Rumore gaussiano | σ ∈ [1, 4] | Rumore sensore |
| Blur gaussiano | r ∈ [0.5, 1.5], p=0.3 | Camera leggermente sfocata |

**Non** vengono applicate flip, rotazioni o crop: la telecamera è fissa e l'orientamento è un'informazione semantica rilevante.

**Accortezza chiave**: la `ref_img` originale **non entra nella memory bank** — solo le varianti augmentate vengono usate. Questo garantisce che il frame di riferimento possa essere usato come campione normale indipendente nella fase di test.

#### Estrazione delle feature

Ogni variante augmentata viene processata da un **WideResNet50-2 pre-addestrato su ImageNet** (pesi congelati, nessun fine-tuning). Il backbone estrae feature map dai livelli intermedi `layer2` e `layer3`:

```
Input: [1, 3, 224, 224]
layer2 output f2: [1,  512, 28, 28]
layer3 output f3: [1, 1024, 14, 14]

f3 upscalato bilinearmente → [1, 1024, 28, 28]
Concatenazione:              [1, 1536, 28, 28]
```

La scelta di `layer2` e `layer3` è quella originale del paper PatchCore: `layer1` cattura feature troppo locali (texture fine), `layer4` è troppo astratto e ImageNet-biased. La combinazione dei due livelli bilancia dettaglio e semantica.

#### Neighbourhood Aggregation

Prima di flattenare le feature, viene applicato un **average pooling locale 3×3** (padding=1, stride=1) su ciascun canale della feature map concatenata. Questo implementa l'aggregazione di vicinato descritta nell'equazione 2 del paper: ogni patch feature diventa una media pesata del suo intorno 3×3, aumentando il contesto locale e riducendo la sensibilità al rumore pixel-level.

```
[1, 1536, 28, 28] → avg_pool2d(k=3, s=1, p=1) → [1, 1536, 28, 28]
→ reshape → [784, 1536]   (784 = 28×28 patch per immagine)
```

Le feature di tutte le `n_augments` varianti vengono concatenate:

```
Memory bank raw: [784 × 15, 1536] = [11760, 1536]
```

#### Coreset Subsampling

La memory bank grezza viene ridotta tramite **greedy minimax coreset** al `coreset_p = 1%` (default), mantenendo almeno 50 punti. L'algoritmo seleziona iterativamente il punto che massimizza la distanza minima dal coreset corrente (facility location), garantendo che la copertura dello spazio delle feature sia uniforme con il minimo numero di punti.

Per efficienza, la selezione greedy viene eseguita su una **proiezione casuale sparsa** a 128 dimensioni (Johnson-Lindenstrauss), ma il coreset finale è estratto dalle feature originali a 1536d.

```
Memory bank finale: [~117, 1536]  (1% di 11760)
```

---

### 1.3 Calibrazione della Soglia

La soglia operativa `θ = μ + k·σ` viene calibrata su `n_cal` varianti augmentate della reference (default: 10), generate con seed=1000, **diverso** da quello usato per la bank (seed=42). Questo garantisce che le varianti di calibrazione siano statisticamente indipendenti da quelle della bank.

Anche le varianti di calibrazione non includono la `ref_img` originale.

Per ciascuna variante di calibrazione si calcola il **max anomaly score** dell'immagine. La distribuzione dei 10 score normali fornisce `μ` e `σ`, e la soglia è `θ = μ + k·σ` dove `k` è un iperparametro (default: 3.0) che controlla il trade-off FPR/recall.

---

### 1.4 Fase di Test — Inferenza

Data un'immagine di test, la pipeline calcola un **anomaly map** e un **scalar anomaly score**.

#### Calcolo degli score per patch

Per ogni patch dell'immagine di test si calcola la distanza al nearest neighbour nella memory bank, con **re-weighting** secondo l'equazione 7 del paper PatchCore:

```
s* = distanza al nearest neighbour m*
Nb(m*) = k nearest neighbours di m* nella bank  (k=9)

w = 1 - exp(||test - m*||) / Σ_{m ∈ Nb(m*)} exp(||test - m||)
score_patch = w · s*
```

Il re-weighting riduce lo score quando il nearest neighbour si trova in una regione densa della bank (alta confidenza che sia normale), e lo amplifica quando si trova in una zona isolata. Le distanze al denominatore sono tra il test patch e i vicini di m*, **non** tra m* e i suoi vicini — questo è un punto critico per la correttezza dell'equazione 7.

#### Anomaly map e scalar score

Le score per patch vengono reshapate nella griglia spaziale 28×28, poi:
1. Upscalate bilinearmente alla risoluzione originale
2. Smoothate con Gaussian blur (σ=4)

L'**anomaly score scalare** è il massimo dell'anomaly map (o nella ROI se specificata):

```python
image_score = anomaly_map_smooth.max()
```

La decisione finale è `image_score > θ`.

---

## 2. Cosa misuriamo e accortezze per un test fair

### 2.1 Setup di valutazione

La valutazione viene eseguita tramite `evaluate_from_db`. Per ciascuna immagine di riferimento `ref_i` nel DB:

1. Si costruisce la memory bank dalle sole varianti augmentate di `ref_i` (non da `ref_i` stessa)
2. Si calibra la soglia `θ_i` sulle varianti di calibrazione (seed diverso, anch'esse senza l'originale)
3. Si testa `ref_i` come campione normale — è l'unico campione genuinamente unseen rispetto alla bank
4. Si testa la corrispondente immagine ostruita come campione anomalo

### 2.2 Separazione bank / calibrazione / test (no data leakage)

| Componente | Seed | Contiene ref_img? | Ruolo |
|---|---|---|---|
| Bank variants (n=15) | 42 | No | Costruisce lo spazio normale |
| Cal variants (n=10) | 1000 | No | Calibra μ e σ della soglia |
| Normal test | — | Sì (è ref_img) | Campione normale indipendente |
| Anomaly test | — | — | Immagine ostruita sintetica |

I tre insiemi sono **disgiunti**: nessun dato usato per costruire il bank o calibrare la soglia entra nella fase di test. La `ref_img` è riservata al test perché è il solo campione normale che il modello non ha mai visto, nemmeno indirettamente tramite augmentazione.

### 2.3 Metriche e loro interpretazione

**Metriche threshold-based** (calcolate con `θ_i = μ_i + k·σ_i` locale per camera):

- `recall` = TP / (TP + FN) — frazione di ostruzioni rilevate
- `false_alarm_rate` = FP / (FP + TN) — frazione di scene normali erroneamente classificate come anomalie

Queste metriche aggregano decisioni binarie già thresholdate per camera, quindi sono **comparabili tra camere diverse** indipendentemente dalla scala assoluta degli score.

**AUROC normalizzato** (per confronto con le altre pipeline P1, P2, P4):

Prima di aggregare gli score di camere diverse per il calcolo dell'AUROC, ogni score viene normalizzato rispetto alla distribuzione normale della propria camera:

```python
normalized_score = (score - μ_i) / σ_i
```

Così l'AUROC misura la separabilità delle distribuzioni normale/anomala in unità di σ, confrontabile tra camere e tra pipeline diverse. Un AUROC su score grezzi non normalizzati sarebbe dominato dalle differenze di scala tra bank diverse e non rifletterebbe la qualità intrinseca del modello.

### 2.4 Difficoltà del task e validità del test

Il test è **conservativo** (difficile per il modello) per il seguente motivo strutturale: le immagini ostruite sono generate per copy-paste sullo stesso sfondo del reference, quindi il background è già perfettamente rappresentato nella memory bank. Solo le patch dell'ostacolo contribuiscono all'anomaly score.

Questo è il caso più difficile di anomaly detection: il contesto è normale, solo l'oggetto è anomalo. Un sistema che supera questa valutazione con recall alto e false_alarm_rate basso ha dimostrato di essere sensibile alle feature degli ostacoli anche in presenza di un background familiare, che è esattamente la capacità richiesta nel caso di deploy reale.

---

## 3. Test di robustezza con ombre sintetiche

### 3.1 Motivazione

Il sistema di anomaly detection è basato sulla distanza tra feature di test e memory bank. Una preoccupazione legittima è che variazioni di illuminazione — in particolare **ombre proiettate** da persone o oggetti che transitano fuori inquadratura — possano deformare le feature in misura sufficiente a superare la soglia di anomalia, generando **falsi positivi**.

Il failure mode di interesse è quindi:

> scena normale + ombra → score > θ → falso allarme

### 3.2 Perché le ombre su scene ostruite non sono utili

Si potrebbe pensare di testare anche immagini ostruite con ombra, per verificare che il modello mantenga il recall. Questo test è **non informativo** per il seguente motivo strutturale: l'occlusione (un carrello, un bidone, una barella) introduce nella patch map un set di feature completamente assenti dalla memory bank — il segnale è già molto sopra la soglia. Aggiungere un'ombra su un'immagine ostruita può solo aumentare il segnale anomalo, non abbassarlo.

Il caso di failure realistico — immagine classificata come normale quando non lo è — non può essere causato da un'ombra aggiuntiva su un ostacolo già presente. Lo scenario che merita attenzione è esclusivamente l'FP sulla scena normale.

**Conseguenza di design**: le ombre sintetiche vengono generate e valutate **solo sulle scene normali**.

### 3.3 Perché le ombre non entrano nella memory bank

Un'alternativa sarebbe includere varianti con ombra nell'augmentazione della bank, rendendola esplicitamente invariante alle ombre. Questa scelta ha un difetto fondamentale: la bank apprenderebbe a ignorare ombre di **forma e posizione specifiche** (quelle generate dalla augmentation), ma non generalizzerebbe a ombre di forma diversa in inferenza.

Includere ombre nella bank renderebbe il modello cieco a ombre del tipo addestrato, senza fornire alcuna garanzia sulle ombre reali. La robustezza documentabile è quella che il modello mostra **senza** aver visto ombre durante il build — e questa è la metrica che ha senso riportare.

**Conseguenza di design**: la memory bank rimane costruita solo da varianti fotometriche shadow-free (luminosità, contrasto, saturazione, temperatura colore, rumore). Le ombre sono esclusivamente nel test set.

### 3.4 Tipi di ombre sintetiche

Sono implementati tre tipi atomici, combinabili in sequenza tramite la modalità `random`:

| Tipo | Descrizione | Scenario simulato |
|---|---|---|
| **Ellittica** | Ellisse scura con bordi sfumati (Gaussian blur), posizione/dimensione/rotazione casuali | Ombra proiettata sul pavimento o sulla parete da un oggetto fuori inquadratura |
| **Direzionale** | Gradiente scuro progressivo da un lato dell'immagine, direzione e copertura casuali | Luce parzialmente bloccata: finestra laterale, lampada a parete, porta socchiusa |
| **Striscia** | Banda orizzontale o verticale con bordi sfumati, larghezza 5–20% dell'immagine | Ombra di trave del soffitto, scaffale a muro, traverso architettonico |

Le ombre direzionali sono le più comuni negli ambienti reali di corridoi e uscite di emergenza: una luce che cambia posizione nel corso della giornata o una porta aperta che proietta un cono d'ombra creano esattamente questo tipo di variazione. Le ombre ellittiche simulano la proiezione di oggetti che transitano (es. trolley spinto da una persona nel corridoio adiacente). Le strisce sono particolarmente frequenti in ambienti industriali con illuminazione a soffitto parzialmente ostruita da elementi strutturali.

Tutti i bordi sono **sfumati** (Gaussian blur sulla maschera): i bordi duri sarebbero irrealistici per un'ombra di scena chiusa e creerebbero segnali di alta frequenza che il backbone ResNet non vede in condizioni normali.

#### Composizione delle ombre per variante

Ogni variante shadow è generata in modalità `random`: vengono estratte casualmente k ombre dal pool dei tre tipi, con k ∈ [1, 3], e applicate in sequenza sulla stessa immagine. Per ogni reference viene prodotta **una sola variante** ombreggiata.

La scelta di k casuale serve a coprire scenari di complessità diversa: k=1 simula una singola fonte di disturbo, k=2–3 simula la sovrapposizione di più elementi (es. luce laterale + ombra di trave). La selezione casuale del tipo ad ogni estrazione garantisce che le varianti non siano sistematicamente dominate da un solo pattern, aumentando la diversità del test set con un numero minimo di immagini per reference.

### 3.5 Integrazione nel DB e nella pipeline di valutazione

Le immagini shadow vengono inserite nel DB con:

| Campo | Valore |
|---|---|
| `is_normal` | 1 |
| `source` | `'shadow'` |
| `occlusion_type` | `'none'` (la scena rimane non ostruita) |
| `reference_frame_id` | frame_id dell'immagine originale |
| `split` | ereditato dall'originale |

Il campo `reference_frame_id` — normalmente usato per collegare immagini ostruite al loro background di riferimento — viene qui riutilizzato per collegare ogni variante shadow alla scena originale da cui deriva. Questo permette a `evaluate_from_db` di recuperarle via `query_shadow_normal_frames` e costruire la memory bank dall'originale shadow-free, garantendo un test genuinamente indipendente dalla bank.

La metrica prodotta è la **shadow FPR**: frazione di varianti shadow normali classificate erroneamente come anomalie. Viene riportata separatamente dalla FPR standard (calcolata sul frame originale) perché misura una diversa fonte di errore.

```
shadow_FPR = shadow_fp / n_shadow_normal_tested
```

Un sistema robusto deve avere shadow_FPR comparabile alla FPR standard: se i due valori sono simili, l'ombra non costituisce una fonte di FP aggiuntiva rispetto alla variabilità fotometrica già coperta dall'augmentazione della bank.
