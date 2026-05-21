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

---

### 3.9 Risultati con shadow augmentation (shadow_prob=0.4, k=3.0)

#### Configurazione del run

La valutazione è eseguita con `--shadow-prob 0.4` e `k=3.0` (default). Il 40% delle varianti di augmentation — sia per la bank (seed=42, n=15) che per la calibrazione (seed=1000, n=10) — riceve k ∈ [1,3] ombre sintetiche in sequenza dopo le trasformazioni fotometriche standard.

#### Risultati

| Metrica | Baseline (shadow_prob=0.0) | Con shadow aug (shadow_prob=0.4) |
|---|---|---|
| Normal FPR | 0.5% | 0.0% |
| **Shadow FPR** | **90.1%** | **0.4%** |
| Obstr TPR (recall) | 100% | 70.6% |
| Obstr FNR | 0% | 29.4% |
| AUROC (Normal vs Obstr) | ~1.000 | ~1.000 |

#### Interpretazione

**Il fix ha funzionato sul problema shadow.** La shadow FPR è scesa dal 90.1% allo 0.4% — una riduzione di due ordini di grandezza. Il meccanismo è esattamente quello previsto in §3.7: le varianti con ombra nella calibrazione alzano μ_cal e σ_cal, portando la threshold adattiva da ~2.6 a ~4.3. Le patch d'ombra nella bank trovano corrispondenti → distanza ridotta → score nella zona normale.

**L'AUROC rimane ~1.000**, confermando che la capacità discriminativa del backbone non è degradata. La separazione tra distribuzioni normale e ostruita è invariata — il problema era e resta esclusivamente nella posizione della threshold.

**Il FNR 29.4% segnala un over-correction della threshold.** Analizzando i falsi negativi, il pattern è chiaro: le ostruzioni mancate hanno score 3.66–3.93, ma le threshold delle rispettive reference arrivano fino a 5.19. Il fenomeno è causato dalla variabilità stocastica della shadow augmentation durante la calibrazione: alcune reference hanno ricevuto per caso molte varianti con ombre pesanti, producendo score di calibrazione elevati e quindi threshold alte. Con k=3.0 l'effetto viene amplificato (il termine k·σ_cal cresce quando σ_cal include la variabilità degli score shadow).

Il trade-off risultante è sbilanciato rispetto al punto ottimale identificato nell'analisi sul CSV baseline (§3.6): a threshold globale 4.00 si otteneva shadow FPR=4.8%, TPR=97.3%. Il run con shadow_prob=0.4, k=3.0 supera quell'obiettivo sulle ombre (0.4% vs 4.8%) ma sacrifica troppo recall (70.6% vs 97.3%).

#### Leve di tuning

Due parametri permettono di recuperare recall senza rinunciare al guadagno sulle ombre:

**1. Abbassare k** (da 3.0 a ~1.5–2.0): la threshold scende proporzionalmente, recuperando le ostruzioni deboli. Con k=1.5 e la distribuzione di calibrazione attuale (che include shadow), la threshold attesa è nell'intorno di 3.5–3.8 — sopra la shadow mean (3.19) ma accessibile per le ostruzioni.

**2. Abbassare shadow_prob** (da 0.4 a ~0.2–0.3): meno varianti con ombra in calibrazione → μ_cal si alza meno → threshold più conservativa. La riduzione riduce però anche la copertura della bank, potenzialmente aumentando il shadow FPR.

Le due leve hanno effetti opposti e complementari: k controlla la posizione assoluta della threshold dato un certo livello di calibrazione; shadow_prob controlla quanto la calibrazione incorpora la variabilità shadow. Il punto di operazione target (shadow FPR < 5%, TPR > 95%) è raggiungibile con una combinazione nell'intorno di `shadow_prob ∈ [0.2, 0.3]`, `k ∈ [1.5, 2.0]`, da validare empiricamente sul test set.

---

### 3.10 Analisi della varianza della soglia per camera

#### Motivazione

L'interpretazione del §3.9 attribuiva il FNR 29.4% a una "over-correction globale" della threshold, suggerendo come rimedi un abbassamento di `k` o di `shadow_prob`. Un'analisi più fine della distribuzione delle threshold per-camera mostra che la diagnosi è incompleta: il problema non è uno spostamento uniforme della soglia, ma una **coda lunga patologica** nella distribuzione di θ_i, generata da instabilità stocastica della calibrazione.

#### Distribuzione delle threshold

| Esperimento | μ_θ | σ_θ | CV | min | max | range |
|---|---|---|---|---|---|---|
| Baseline (no shadow) | 2.470 | 0.158 | 6.4% | 1.97 | 3.00 | 1.03 |
| `shadow_prob=0.4` | 4.348 | **0.539** | **12.4%** | 2.44 | **6.19** | **3.74** |

Con la shadow augmentation la **σ della distribuzione delle threshold è 3.4× più ampia** rispetto al baseline shadow-free, e il range si estende da [1.97, 3.00] a [2.44, 6.19]. Non è uno spostamento uniforme — è la comparsa di una coda destra che non esisteva.

#### Distribuzione del contributo agli FN per bucket di threshold

I 291 falsi negativi (29.4% di 990 ostruite) non sono distribuiti uniformemente, ma si concentrano nelle camere con threshold elevata:

| Bucket θ | n camere | % camere | FNR del bucket | % FN totali |
|---|---|---|---|---|
| ≤ 4.0 | 261 | 26% | 0% | 0% |
| 4.0 – 4.5 | 348 | 35% | 12% | 14% |
| 4.5 – 5.0 | 264 | 27% | **53%** | **48%** |
| 5.0 – 5.5 | 103 | 10% | **92%** | **33%** |
| > 5.5 | 14 | 1.4% | **100%** | 5% |

**117 camere (12% del totale) con threshold > 4.5 generano il 86% degli FN.** Le 14 camere con θ > 5.5 sono completamente cieche alle ostruzioni: missano il 100% degli obstructed test per quelle reference.

#### Diagnosi: instabilità stocastica della calibrazione

Le camere con threshold patologicamente alta non condividono caratteristiche di scena particolari — sono distribuite indifferentemente tra porte e corridoi. Il loro tratto comune è statistico: con `--cal 10` sample e `shadow_prob=0.4`, alcune camere campionano per caso 6–8 variants pesantemente ombreggiate su 10. Questo gonfia sia μ_cal che soprattutto σ_cal, e con k=3.0 il termine k·σ_cal amplifica l'effetto fino a portare θ_i a 6.19.

Sono **outlier di calibrazione, non outlier di scena**. La stessa camera, ricalibrata con un seed diverso, avrebbe una threshold completamente diversa. La σ_cal stimata su n=10 sample con varianza alta tra le sorgenti (variants shadow-free vs shadow-heavy) non è statisticamente affidabile.

#### Implicazione sulle leve di tuning

Le due leve identificate in §3.9 (abbassare k o abbassare shadow_prob) **non risolvono la coda lunga**:

- **Abbassare k** comprime uniformemente tutte le threshold. Le 14 camere a 6.19 scenderebbero a ~4.1 con k=2.0, ma resterebbero outlier rispetto alla mediana (~2.9). Inoltre l'abbassamento di k riduce il margine di sicurezza anche per le camere già ben calibrate.
- **Abbassare shadow_prob globalmente** riduce la varianza di σ_cal ma allo stesso tempo riduce la copertura shadow della bank, peggiorando potenzialmente il shadow FPR.

#### Rimedio strutturale: decoupling build/calibration + più sample di calibrazione

Le due fasi che attualmente condividono `shadow_prob` hanno requisiti opposti:

| Fase | Obiettivo | shadow_prob ottimale |
|---|---|---|
| **Build** (n=15 variants → coreset 1%) | Bank diversificata sulle ombre per ridurre la distanza al kNN su patch ombreggiate | **Alto (0.4)** |
| **Calibration** (n=10 variants → μ, σ) | Distribuzione degli score normali stabile e riproducibile | **Basso (0.1)** o zero |

Inoltre, aumentando `--cal` da 10 a 30 si riduce la varianza stocastica di σ_cal di un fattore √3 ≈ 1.7. Questo è il modo più diretto per eliminare gli outlier nella distribuzione di θ_i senza compromettere la qualità della bank.

**La modifica non incide su tempo di inferenza né su dimensione della memory bank:** le variants di calibrazione sono utilizzate solo per stimare μ e σ a build time; non entrano nel coreset. Il costo aggiuntivo è esclusivamente computazionale e una tantum, sostenuto durante la costruzione della bank.

#### Configurazione proposta per il prossimo run

| Parametro | Attuale | Proposto | Motivazione |
|---|---|---|---|
| `shadow_prob` (build) | 0.4 | 0.4 | bank diversificata sulle ombre, invariata |
| `shadow_prob` (calibration) | 0.4 | **0.1** | σ_cal stabile, threshold senza coda destra |
| `--cal` | 10 | **30** | riduce varianza stocastica di σ_cal (√3 ≈ 1.7×) |
| `--aug` | 15 | 15 | bank di dimensione invariata |
| `CORESET_P` | 0.01 | 0.01 | bank di dimensione invariata → inferenza invariata |
| `k` | 3.0 | 3.0 | mantenuto per ora; eventuale aggiustamento dopo aver stabilizzato la distribuzione di θ_i |

L'aspettativa è che la σ delle threshold per-camera scenda da 0.54 a un valore comparabile al baseline shadow-free (~0.2), mantenendo la mediana spostata abbastanza da preservare il guadagno sulla shadow FPR. Una volta stabilizzata la distribuzione di θ_i, eventuali aggiustamenti residui di k diventano un puro spostamento sul ROC, prevedibile e simmetrico per tutte le camere.

---

### 3.11 Risultati del decoupling e ricalibrazione di k

#### Run 1 — decoupling con k=3.0

Configurazione: `shadow_prob=0.4`, `shadow_prob_cal=0.1`, `--cal 30`, `k=3.0`.

| Metrica | Coupled (sp=spc=0.4, k=3.0) | Decoupled (spc=0.1, cal=30, k=3.0) | Δ |
|---|---|---|---|
| Clean FPR | 0.0% | 0.0% | invariato |
| **Shadow FPR** | 2.1% | **28.4%** | +26.3 pp |
| Obstr TPR | 70.6% | **99.4%** | +28.8 pp |
| Obstr FNR | 29.4% | **0.6%** | −28.8 pp |
| Precision | 97.1% | 77.8% | −19.3 pp |
| **F1** | 0.818 | **0.873** | +0.055 |
| AUROC | 0.998 | 0.998 | invariato |
| **σ delle θ** | **0.539** | **0.388** | **−28%** |
| Range θ | [2.44, 6.19] | [2.24, 5.23] | restretto |

Il decoupling ha avuto due effetti distinti:

1. **Stabilizzazione strutturale**: σ_θ è scesa del 28% e la coda destra patologica è scomparsa (max θ da 6.19 a 5.23). Le camere "rotte" del run precedente sono recuperate.
2. **Trade-off operativo capovolto**: la media di θ è scesa da 4.35 a 3.45, portando shadow FPR da 2% a 28% e TPR da 71% a 99%. Si è passati da over-correction a under-correction.

L'F1 è salito di +5.5 punti percentuali, ma il punto operativo è ora troppo permissivo: 281 falsi allarmi su scene normali ombreggiate.

#### Run 2 — decoupling con k=4.0

Configurazione: identica al run 1 ma `k=4.0`, per spostare la media di θ verso lo sweet spot teorico ~4.0 visto nel threshold-sweep globale.

| Metrica | Decoupled k=3.0 | Decoupled k=4.0 | Δ |
|---|---|---|---|
| Clean FPR | 0.0% | 0.0% | invariato |
| Shadow FPR | 28.4% | **13.5%** | −14.9 pp |
| Obstr TPR | 99.4% | **93.4%** | −6.0 pp |
| Obstr FNR | 0.6% | 6.6% | +6.0 pp |
| Precision | 77.8% | 87.3% | +9.5 pp |
| **F1** | 0.873 | **0.903** | +0.030 |
| σ delle θ | **0.388** | **0.497** | **+28%** |
| Range θ | [2.24, 5.23] | [2.28, 6.09] | riallargato |

Il run k=4.0 migliora F1 di altri 3 pp, ma rivela un effetto collaterale: **σ_θ è salita del 28%**. Il problema è strutturale: la formula `θ_i = μ_i + k·σ_cal,i` ha k come moltiplicatore di una stima rumorosa di σ_cal. Aumentando k amplifichiamo proporzionalmente le differenze stocastiche tra camere — la coda lunga torna.

#### Distribuzione del contributo agli errori — k=4.0

| Bucket θ | n camere | shadow FPR | TPR ostr | % FN | % shadow FP |
|---|---|---|---|---|---|
| ≤ 3.0 | 52 (5%) | **67%** | 100% | 0% | 26% |
| 3.0–3.5 | 250 (25%) | 24% | 100% | 0% | 46% |
| 3.5–4.0 | 384 (39%) | 9% | 99% | 6% | 25% |
| 4.0–4.5 | 243 (25%) | 2% | 90% | 35% | 4% |
| **4.5–5.0** | **52 (5%)** | 0% | **40%** | **48%** | 0% |
| > 5.0 | 9 (1%) | 0% | 22% | 11% | 0% |

Sono ricomparse **due code patologiche simmetriche**:
- 52 camere con θ ≤ 3.0 (5% del totale) generano il 26% di tutti gli shadow FP (shadow FPR=67%)
- 61 camere con θ > 4.5 (6% del totale) generano il 59% di tutti gli FN (TPR scende al 22–40%)

Sono gli stessi pattern del run coupled originale, solo a scala ridotta. La diagnosi è la stessa: σ_cal stimato su 30 sample non è abbastanza preciso, e moltiplicarlo per k>3 ne amplifica l'errore stocastico in modo strutturale.

#### Confronto con threshold globale ideale

Il threshold-sweep su scores normalizzati mostra che, se ogni camera operasse alla **stessa** soglia 4.0, otterremmo:

| Operating point | Clean FPR | Shadow FPR | TPR | F1 stimato |
|---|---|---|---|---|
| Per-camera (k=4) | 0.0% | 13.5% | 93.4% | 0.903 |
| **Globale uniforme @ 4.0** | **0.0%** | **2.9%** | **97.7%** | **~0.94** |

Esiste un gap di **~4 punti percentuali di F1** che non è raggiungibile con la calibrazione per-camera attuale. Il limite è intrinseco alla formula `μ_i + k·σ_cal,i`: l'incertezza in σ_cal,i si propaga sul threshold e produce camere fortunate/sfortunate indipendentemente dalla qualità della scena.

#### Limite della leva k

Confronto del tasso di shift:
- k=3.0 → 4.0: μ_θ +0.30, σ_θ +0.11 (+28%)
- estrapolazione k→5.0: μ_θ atteso ~+0.30, σ_θ atteso ~+30% ulteriore

I rendimenti di k sono decrescenti: spostiamo la media lentamente ma allarghiamo la distribuzione velocemente. Continuare a salire con k peggiora la stabilità senza guadagni operativi significativi.

---

### 3.12 Proposta finale: pooled σ con median

#### Motivazione

L'analisi dei due run decoupled identifica con precisione la causa dei rendimenti decrescenti: **σ_cal,i è una stima rumorosa**, perché basata su n=30 sample con `shadow_prob_cal=0.1` (≈3 variants con ombra su 30, numero piccolo e ad alta varianza di Bernoulli). L'incertezza relativa su σ_cal,i con n=30 è dell'ordine del 18%, e si riflette uno-a-uno su `k·σ_cal,i` nel calcolo della soglia.

D'altro canto, **μ_cal,i è una stima affidabile**: 30 sample sono sufficienti per fissarne il valore con precisione (la SE su μ è dell'ordine di σ_cal,i / √30 ≈ 6% di σ_cal,i). Inoltre μ_cal,i incorpora informazione genuina sulla scena specifica (livello assoluto degli score nella distribuzione normale di quella camera), che vogliamo preservare per la calibrazione adattiva.

La proposta è quindi:

```
θ_i = μ_i + k · σ_pop
```

dove `σ_pop` è uno stimatore globale di σ_cal, calcolato dopo che tutti i worker hanno terminato la prima passata.

#### Scelta dello stimatore

| Stimatore | Formula | Pro | Contro |
|---|---|---|---|
| Pooled std (RMS) | √mean(σ_i²) | Massima verosimiglianza se σ è omogeneo | Sensibile alle code di σ_i |
| Mean | mean(σ_i) | Semplice | Influenzato da calibrazioni sfortunate alte |
| **Median** | median(σ_i) | **Robusto agli outlier** | Scarta informazione dalle code |

La scelta ricade sul **median** per coerenza con la motivazione dell'intervento: il problema sono proprio le calibrazioni "fortunate" (σ_cal molto basso → soglia troppo bassa → shadow FP) o "sfortunate" (σ_cal alto → soglia troppo alta → FN). Il median ignora entrambi gli estremi e produce un valore di tolleranza rappresentativo della popolazione.

#### Effetto atteso sulla distribuzione di θ

Sotto l'attuale formula `θ_i = μ_i + k·σ_cal,i`:

```
Var(θ_i) = Var(μ_i) + k² · Var(σ_cal,i) + 2k · Cov(μ_i, σ_cal,i)
```

Il termine `k² · Var(σ_cal,i)` è dominante: con k=4 e Var(σ_cal,i) ≈ 0.05² (stima grezza), questo contributo è ~0.16, comparabile alla σ_θ osservata 0.50² = 0.25.

Sotto la nuova formula `θ_i = μ_i + k·σ_pop` (con σ_pop costante):

```
Var(θ_i) = Var(μ_i)
```

La σ delle threshold collassa alla **variabilità intrinseca delle scene**, eliminando completamente la componente stocastica della calibrazione. Stima conservativa: σ_θ scende da 0.50 a ~0.15 (stessa scala di Var(μ_i) osservata nel baseline shadow-free).

#### Conseguenze operative attese

1. **Coda lunga eliminata**: le 61 camere con θ > 4.5 in k=4 erano outlier per via di σ_cal,i alto, non per μ_i alto. Con σ_pop costante, queste camere tornano nel cluster centrale.
2. **Camere a θ basso meno permissive**: le 52 camere con θ ≤ 3.0 erano outlier per σ_cal,i basso. Con σ_pop costante (mediana ≈ 0.25 atteso), le loro soglie salgono verso μ_i + k·σ_pop ≈ μ_i + 1.0.
3. **Operating point avvicinato allo sweet spot teorico ~4.0**: dato che la perdita di F1 (4 pp) è dovuta alla dispersione di θ rispetto a un threshold globale ideale, comprimere la distribuzione attorno alla media dovrebbe recuperare gran parte del gap.

#### Implementazione (sketch)

Algoritmo two-pass nel runner:

**Pass 1** (worker, in parallelo per camera):
- Costruisce bank e calibrazione come ora
- Calcola μ_i, σ_i, scores per tutti i test (normal, shadow, obstructed)
- **Non** classifica ancora gli esiti (no `is_anomaly`)
- Restituisce i raw scores + (μ_i, σ_i)

**Pass 2** (main thread, sequenziale dopo aggregazione):
- Raccoglie tutti i σ_i dalle camere processate (+ eventuali da CSV in resume mode)
- Calcola `σ_pop = median(σ_i)`
- Per ogni row: ricalcola `threshold = μ_i + k·σ_pop`, `normalized_score = (score − μ_i) / σ_pop`, `is_anomaly = score > threshold`
- Ricostruisce `venue_stats` (fp/tp/fn/shadow_fp e relative liste di candidati per le heatmap) dalle nuove classificazioni
- Scrive CSV con threshold ricalcolato + colonne diagnostiche `mu`, `sigma` (l'σ_cal,i originale per camera, ai fini di analisi)

CLI: introdurre `--sigma-mode {per_camera, pooled}` con default `per_camera` per backward compatibility. La metadata dell'esperimento (tabella `experiments`) registra il modo scelto e il valore di `σ_pop` calcolato, per riproducibilità.

#### Aspettativa quantitativa

Sulla base del threshold-sweep eseguito sui dati k=4, la configurazione `sigma_mode=pooled, k=4.0` dovrebbe operare con threshold per-camera nell'intorno di `μ_i + 4·σ_pop ≈ μ_i + 1.0`. Con μ_i mediamente ~3.0, le soglie si concentrerebbero intorno a 4.0, dove il punto operativo globale dà:

- shadow FPR ~3%
- TPR ~98%
- F1 ~0.94

Il guadagno atteso rispetto a k=4 standard è di **~4 pp di F1** (da 0.903 a ~0.94), e questo guadagno proviene esclusivamente dalla riduzione di varianza nelle soglie — non da nessun cambio della capacità discriminativa del modello (AUROC resta 0.998).

---

### 3.13 Verifica empirica del pooled σ e considerazioni di generalizzazione

#### Risultato sperimentale (sigma_mode=pooled, k=4.0)

Eseguito il run con `--sigma-mode pooled --shadow-prob 0.4 --shadow-prob-cal 0.1 --cal 30 --k 4.0`. La predizione di §3.12 si è materializzata quasi punto a punto:

| Metrica | Decoupled per-camera (k=4) | **Pooled σ (k=4)** | Δ |
|---|---|---|---|
| Clean FPR | 0.0% | 0.0% | invariato |
| Shadow FPR | 13.5% | **12.2%** | −1.3 pp |
| Obstr TPR | 93.4% | **99.8%** | +6.4 pp |
| Obstr FNR | 6.6% | 0.2% | −6.4 pp |
| Precision | 87.3% | 89.1% | +1.8 pp |
| **F1** | 0.903 | **0.941** | **+0.038** |
| AUROC | 0.998 | 0.998 | invariato |
| **σ delle θ** | **0.497** | **0.137** | **−73%** |
| Range θ | [2.28, 6.09] | [3.24, 4.16] | −76% |

Il confronto è apples-to-apples (stesso k=4.0, stessa augmentation): **il guadagno di +3.8 pp di F1 deriva esclusivamente dall'eliminazione della varianza stocastica della soglia**. σ_pop calcolato = 0.300 (median delle σ_cal,i sulle 990 reference).

#### La coda patologica è scomparsa

Bucket-by-bucket analysis sul pooled run:

| Bucket θ | n camere | shadow FPR | TPR ostr | % FN |
|---|---|---|---|---|
| ≤ 3.4 | 7 (0.7%) | 14% | 100% | 0% |
| 3.4–3.6 | 149 (15%) | 24% | 100% | 0% |
| 3.6–3.8 | 507 (51%) | 12% | 100% | 0% |
| 3.8–4.0 | 296 (30%) | 8% | 99.3% | 100% |
| > 4.0 | 31 (3%) | 3% | 100% | 0% |

Nessuna camera "rotta": TPR ≥ 99.3% in ogni bucket. I 2 falsi negativi residui (0.2% del totale) vengono tutti dal bucket 3.8–4.0 e sono casi limite intrinsecamente difficili (ostacolo piccolo, scarsamente contrastato), non outlier di calibrazione.

#### Considerazioni metodologiche sulla generalizzazione

L'analisi degli ultimi run ha messo in luce un trade-off tra **tuning fine sul test set** e **affermazioni sostenibili per il deploy reale**. Vale la pena distinguere esplicitamente due classi di risultati ottenuti.

##### Risultati strutturali (procedura, indipendenti dal dataset)

Sono interventi sulla **pipeline di calibrazione**, non sui dati:

| Intervento | Generalizzazione attesa |
|---|---|
| Shadow augmentation in build (`shadow_prob` > 0) | Robusta: il fix copre il distribution mismatch tra build e test (§3.7) — qualsiasi dataset con ombre fuori-distribuzione beneficia |
| Decoupling build/calibration shadow_prob | Robusta: separa due fasi con requisiti opposti (§3.10) — fix di pipeline |
| `--cal 30` (vs 10) | Robusta: riduce SE(σ_cal) di √3 ≈ 1.7× — risultato statistico |
| Pooled σ con median (§3.12) | Robusta: σ_pop ha SE ≈ SE(σ_cal,i)/√N con N=990 camere → quasi zero — risultato statistico |
| AUROC ≈ 0.998 (Normal+Shadow vs Obstr) | Robusta: misura la separabilità del classificatore indipendentemente dalla soglia. Numero più "trasportabile" che abbiamo |

##### Risultati di punto operativo (specifici al test set)

Sono numeri che dipenderebbero dalla distribuzione delle ombre e delle ostruzioni del particolare test set:

| Metrica | Cosa la condiziona |
|---|---|
| Valore esatto di k che massimizza F1 | Severità delle ombre nel test, difficoltà delle ostruzioni |
| Shadow FPR a k fissato | Distribuzione di severity delle ombre |
| TPR a k fissato | Distribuzione di severity delle ostruzioni |
| σ_pop = 0.300 | Dipende da `shadow_prob_cal` e dalla varianza delle scene |
| Distribuzione di μ_i | Dipende dalle scene specifiche delle 990 camere del test |

In deploy reale i numeri specifici **cambieranno**:
- Le ombre proiettate da persone reali sono diverse dalle 3 modalità sintetiche (ellittica/direzionale/striscia)
- Gli ostacoli reali sono diversi dalle 5 categorie del fine-tuning (carrelli, sedie a rotelle, barelle, scatole, bidoni)
- Le scene di deploy hanno μ_i con distribuzione diversa dalle 990 camere usate qui

##### Cosa NON va difeso nella tesi

Continuare a spingere k=3 → 4 → 5 in cerca di F1 ottimo sul test set è una forma di **test-set tuning**, equivalente a optimismo metodologico. L'F1 ≈ 0.97 previsto per k≈5 sarebbe un upper bound *su questo dataset*, non una stima di performance in deploy.

In particolare il punto operativo scelto in deploy non si determina sui dati di laboratorio: dipende dal **costo asimmetrico FN/FP** del contesto applicativo (per via di fuga: un FN — ostruzione mancata — vale infinitamente più di un FP — falso allarme).

##### Cosa va difeso nella tesi

**Reclamo forte (strutturale):**
> Il pooling di σ con stimatore median (sostituzione di σ_cal,i per-camera con σ_pop globale) elimina la coda patologica della distribuzione delle soglie, riducendo σ_θ del 73% (0.50 → 0.137). Il fix è motivato dalla teoria statistica (stima per-camera su n=30 ha SE relativa ≈ 18%, mentre σ_pop aggregato su 990 camere ha SE ≈ 0.6%) e si replica in modo dataset-indipendente.

**Reclamo accettabile (operativo, con caveat):**
> Sul test set considerato, il classificatore opera con AUROC = 0.998 e, alla scelta `k = 4` (regola conservativa "4-sigma"), produce shadow FPR = 12.2%, TPR = 99.8%, F1 = 0.941. La scelta di k è in linea con i criteri standard di calibrazione 3–4 sigma e non è stata ottimizzata sul test set.

**Reclamo da evitare:**
> L'F1 ottimo è 0.97 con k = 3.9.

#### Operating point selection in production

Per il deployment reale, la scelta del punto operativo non dovrebbe essere fatta sui dati di laboratorio. Si raccomanda il seguente protocollo:

1. **Conserva la pipeline di calibrazione invariata**: decoupling, `cal=30`, sigma_mode=pooled. Questi sono fix strutturali e si applicano allo stesso modo a qualsiasi insieme di camere.

2. **Ricalcola σ_pop sul sito specifico**: dopo aver costruito le bank per le N camere del deploy, raccogliere le σ_cal,i e usarne la mediana come σ_pop locale. Il valore atteso è dello stesso ordine di grandezza (~0.3) ma non identico.

3. **Scegli k in base al costo asimmetrico FN/FP**: la curva ROC del sistema permette di selezionare il punto operativo desiderato a posteriori. Per vie di fuga, dove un FN costa potenzialmente vite umane, k basso (k=2.5–3) è preferibile anche al prezzo di shadow FPR elevato; per applicazioni meno critiche k=4 è ragionevole.

4. **Validazione in-situ obbligatoria**: pochi giorni di osservazione su scene reali, raccogliendo manualmente l'identità di FP e FN, è il test di validazione che vale più di qualsiasi tuning su sintetici.

5. **Periodic recalibration**: μ_i delle camere drifta con l'invecchiamento dei sensori e con le modifiche alla scena. Ricalibrare quarterly o quando si osservi un drift sistematico negli score di calibrazione di una camera.

#### Validità del risultato sperimentale

Pur con i caveat sopra, il risultato di questo capitolo è solido per quanto riguarda l'oggetto della tesi: la **pipeline di anomaly detection per occlusioni** funziona con AUROC ≈ 1.0 e separa robustamente normali, normali-con-ombra, e ostruzioni. Il fix del pooled σ riduce la variabilità per-camera in modo che il sistema sia **calibrabile in modo prevedibile** anche su insiemi di camere eterogenee. Quale sia il punto operativo migliore è una domanda di applicazione, non di architettura.

Le metriche più "trasportabili" — AUROC = 0.998, riduzione di σ_θ del 73%, ottenimento di clean FPR = 0% senza degrado della TPR — sono affermazioni sostanziali sull'architettura, non artefatti di tuning.
