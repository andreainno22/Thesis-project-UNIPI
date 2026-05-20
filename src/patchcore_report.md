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

---

### 3.6 Risultati empirici sul test shadow

La valutazione su 990 campioni per categoria (corridoi e porte, split test) produce la seguente distribuzione degli anomaly score:

| Categoria | Score medio | Std | Min | Max | FPR / TPR |
|---|---|---|---|---|---|
| `normal` | 2.309 | 0.156 | 1.871 | 3.129 | **FPR 0.5%** |
| `shadow_normal` | 3.189 | 0.495 | 2.097 | 4.507 | **FPR 90.1%** |
| `obstructed` | 4.648 | 0.348 | 3.542 | 5.866 | **TPR 100%** |

Il modello si comporta correttamente sui campioni normali e ostruiti, ma **classifica come anomalia il 90.1% delle scene normali con ombra**. Questo è il failure mode esatto descritto in §3.1: le ombre, pur non essendo ostruzioni, generano feature distanti dalla bank e superano la soglia calibrata.

Il risultato è coerente con la motivazione teorica di §3.3: la bank non contiene patch con ombre, quindi qualsiasi ombra produce distanze elevate rispetto ai vicini nella bank — il meccanismo nearest-neighbour non riesce a distinguere l'anomalia reale dalla variazione fotometrica non coperta dall'augmentazione.

#### Zona di overlap e analisi della threshold

La zona critica è il range di score **3.54–4.51**, in cui le code delle due distribuzioni si sovrappongono (shadow_normal max = 4.507; obstructed min = 3.542). Una threshold globale fissa non può separare le due classi senza sacrificare recall:

| Threshold | Shadow FPR | Obstr TPR |
|---|---|---|
| 3.75 | 13.1% | 99.4% |
| **4.00** | **4.8%** | **97.3%** |
| 4.25 | 1.4% | 87.3% |
| 4.50 | 0.1% | 67.4% |

A threshold 4.00 si ottiene un compromesso accettabile (4.8% shadow FPR, 97.3% TPR), ma questo richiede alzare il valore assoluto della soglia in modo fisso — non è compatibile con il sistema di calibrazione adattiva per-camera già in uso, che determina la threshold a partire dalla distribuzione dei campioni normali di ciascuna bank.

La soluzione non può essere solo una ricalibrazione di `k`: il problema è strutturale e risiede nel **mancato allineamento tra la distribuzione di calibrazione (shadow-free) e la distribuzione di test (con ombre)**.

---

### 3.7 Proposta di miglioramento: shadow augmentation in fase di build

#### Motivazione

La diagnosi del §3.6 indica che la calibrazione e la bank vengono costruite su una distribuzione diversa da quella che si incontra al test. Questo è un caso di **distribution mismatch** tra training e deployment: le ombre sono variazioni fotometriche attese nella scena reale, ma completamente assenti dalla fase di build.

Il rimedio naturale all'interno del framework PatchCore è estendere `augment_reference` con ombre sintetiche. In questo modo:

1. **Memory bank con ombre**: le patch d'ombra nel campione di test trovano corrispondenti nella bank → distanza ridotta → score basso → nessun FP
2. **Calibrazione con ombre**: la distribuzione dei 10 score di calibrazione include già varianti con ombra → `μ_cal` si sposta verso l'alto, `σ_cal` aumenta → la threshold `μ + k·σ` si adatta naturalmente → le ombre non la superano

L'effetto è ottenuto senza modificare `k`, senza aggiungere soglie separate e senza nessun componente esterno: il framework PatchCore si ricalibra autonomamente.

Questa revisione modifica la scelta di design enunciata in §3.3. La preoccupazione originale — che la bank apprenda a ignorare solo le ombre specifiche dell'augmentation — è fondata in linea di principio, ma i dati mostrano che il costo dell'approccio shadow-free è un FPR del 90%, inaccettabile in un sistema safety-critical. La generalizzazione alle ombre reali è una questione empirica che può essere valutata estendendo il test set con varianti non viste.

#### Implementazione proposta

```python
# In augment_reference, aggiungere con probabilità shadow_prob:
from generate_shadow_images import (
    add_elliptical_shadow, add_directional_shadow, add_stripe_shadow
)
_SHADOW_POOL = [add_elliptical_shadow, add_directional_shadow, add_stripe_shadow]

# Per ogni variante augmentata:
if rng.random() < shadow_prob:   # default: 0.4
    shadow_fn = rng.choice(_SHADOW_POOL)
    img_aug = shadow_fn(img_aug, rng)
```

Il parametro `shadow_prob` (default consigliato: 0.4) controlla la frazione di varianti che ricevono un'ombra, bilanciando copertura dello spazio shadow e preservazione della variabilità shadow-free nella bank. Per ciascuna variante che riceve ombra, vengono applicate k ∈ [1, 3] ombre in sequenza (stessa logica di `generate_shadow_images.py` in modalità `random`), coprendo scenari di complessità diversa.

#### Separazione formale tra ombre di bank e ombre di test

Le ombre di test nel DB sono generate da `generate_shadow_images.py` con `seed=42` applicato sull'immagine originale pulita. Per garantire disgiunzione formale rispetto alle ombre generate in `augment_reference`, le shadow augmentation della bank usano un **RNG dedicato** con seed disgiunto:

```python
shadow_rng = np.random.default_rng(seed + 99999)
```

I tre spazi di seed risultano formalmente disgiunti:

| Insieme | Seed effettivo per le ombre |
|---|---|
| Bank variants (seed=42) | `default_rng(100041)` |
| Cal variants (seed=1000) | `default_rng(100999)` |
| Test shadows nel DB | `default_rng(42)` su immagine originale |

Anche senza questa separazione esplicita, la disgiunzione sarebbe di fatto garantita: l'RNG principale in `augment_reference` consuma circa 8 chiamate random (brightness, contrast, saturation, shift R/B, noise, blur check, blur radius) prima di raggiungere il punto ombra, rendendo il suo stato completamente diverso da un `default_rng(42)` fresco. Il seed dedicato aggiunge una garanzia formale documentabile indipendentemente dall'implementazione interna dell'RNG.

---

### 3.8 Contesto letterario

#### Precisazione sul tipo di problema

Il problema osservato non è un caso di domain shift tra training e test nel senso classico. Nel nostro sistema ogni bank è costruita dalla stessa telecamera che poi viene testata: non esiste un dataset di training globale con condizioni di illuminazione diverse da quelle di deploy. Il failure mode è invece un **mismatch di copertura dell'augmentazione**: la procedura `augment_reference` non genera ombre, quindi le ombre sono fuori dalla distribuzione coperta dalla bank pur provenendo dalla stessa camera.

Questa distinzione è rilevante per la scelta delle citazioni: lavori come PIAD (Yang et al., CVPR 2025) affrontano un problema strutturalmente diverso (training set multi-oggetto sotto illuminazione controllata vs. test sotto illuminazione diversa) e non si mappano direttamente al nostro scenario.

#### Letteratura pertinente

**M²AD: Visual Anomaly Detection under Complex View-Illumination Interplay** (Cheng et al., arxiv 2505.10996, maggio 2025) rimane rilevante: valuta metodi VAD esistenti — inclusi approcci memory-based analoghi a PatchCore — sotto variazioni di illuminazione sistematiche, e mostra crolli significativi di performance. Il risultato è coerente con il 90.1% di shadow FPR osservato, anche se il loro setup (12 viste × 10 illuminazioni su dataset industriale) è più ampio del nostro. Utile per motivare il gap nella letteratura.

**Shadow Augmentation for Handwashing Action Recognition** (arxiv 2410.03984, 2024) fornisce il precedente più diretto alla proposta del §3.7: ombre sintetiche aggiunte durante la fase di costruzione del modello riducono i falsi positivi a test time. Il dominio (sorveglianza indoor clinica, telecamere fisse, illuminazione artificiale variabile) è analogo al nostro.

**Pose-Agnostic Anomaly Detection with Retinex-based Illumination Alignment (R3-PAD)** (Wang et al., 2025) propone una strategia alternativa che agisce sul lato test anziché sul lato training: una Retinex-UNet normalizza l'illuminazione dell'immagine di test prima dell'estrazione delle feature, rimuovendo l'effetto dell'ombra prima ancora che raggiunga il backbone. Il vantaggio principale rispetto all'approccio di §3.7 è che la bank rimane invariata e la generalizzazione a tipi di ombra non visti durante il build non dipende dalla copertura dell'augmentazione. Tuttavia, questo approccio introduce un passo di preprocessing aggiuntivo seriale che aumenta il tempo di inferenza — un costo rilevante nel contesto di deploy su device edge. Va inoltre verificato empiricamente che la normalizzazione Retinex non attenui anche le feature degli ostacoli, riducendo il recall. Per questi motivi viene considerato come direzione alternativa da esplorare piuttosto che come soluzione primaria.

**Enhancing Anomaly Detection Generalization through Knowledge Exposure: Dual Effects of Augmentation** analizza teoricamente il rapporto tra copertura dell'augmentazione e generalizzazione nel contesto dell'anomaly detection. Supporta la scelta di una `shadow_prob` controllata: un'augmentazione troppo aggressiva può ridurre la sensibilità alle anomalie vere, mentre una copertura parziale (40–50%) amplia la distribuzione normale senza degradare il recall.

Per il contesto classico del rilevamento di ombre in sorveglianza, il riferimento primario è **Sanin et al. (2012)** *"Shadow Detection: A Survey and Comparative Evaluation of Recent Methods"* (Pattern Recognition), che classifica i metodi in quattro categorie: cromaticità, fisici, geometrici e tessiturali. Per ambienti indoor, il metodo basato su cromaticità HSV (Cucchiara et al., 2003) rimane il riferimento storico più citato nel contesto di sistemi di videosorveglianza.
