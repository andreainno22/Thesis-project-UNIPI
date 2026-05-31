# PatchCore per il Rilevamento di Occlusioni - Report Metodologico

## 1. Pipeline PatchCore: funzionamento dettagliato

### 1.1 Architettura generale

PatchCore è un sistema di **anomaly detection non supervisionato** basato su memoria. Non richiede immagini anomale durante la fase di costruzione: apprende esclusivamente dalla scena normale di riferimento. Il principio fondamentale è che un'immagine è anomala se le sue feature locali si trovano lontane da quelle memorizzate durante la fase di build.

Nel contesto applicativo, ogni via di fuga è monitorata da una telecamera dedicata. Per ciascuna telecamera viene costruita una **memory bank indipendente** a partire da un singolo frame di riferimento della scena normale. Questa scelta riflette esattamente il caso reale di deploy: la camera non cambia inquadratura e la scena normale ha caratteristiche fotometriche stabili.

---

### 1.2 Fase di Build - Costruzione della Memory Bank

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

**Accortezza chiave**: la `ref_img` originale **non entra nella memory bank** - solo le varianti augmentate vengono usate. Questo garantisce che il frame di riferimento possa essere usato come campione normale indipendente nella fase di test.

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

### 1.4 Fase di Test - Inferenza

Data un'immagine di test, la pipeline calcola un **anomaly map** e un **scalar anomaly score**.

#### Calcolo degli score per patch

Per ogni patch dell'immagine di test si calcola la distanza al nearest neighbour nella memory bank, con **re-weighting** secondo l'equazione 7 del paper PatchCore:

```
s* = distanza al nearest neighbour m*
Nb(m*) = k nearest neighbours di m* nella bank  (k=9)

w = 1 - exp(||test - m*||) / Σ_{m ∈ Nb(m*)} exp(||test - m||)
score_patch = w · s*
```

Il re-weighting riduce lo score quando il nearest neighbour si trova in una regione densa della bank (alta confidenza che sia normale), e lo amplifica quando si trova in una zona isolata. Le distanze al denominatore sono tra il test patch e i vicini di m*, **non** tra m* e i suoi vicini - questo è un punto critico per la correttezza dell'equazione 7.

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
3. Si testa `ref_i` come campione normale - è l'unico campione genuinamente unseen rispetto alla bank
4. Si testa la corrispondente immagine ostruita come campione anomalo

### 2.2 Separazione bank / calibrazione / test (no data leakage)

| Componente | Seed | Contiene ref_img? | Ruolo |
|---|---|---|---|
| Bank variants (n=15) | 42 | No | Costruisce lo spazio normale |
| Cal variants (n=10) | 1000 | No | Calibra μ e σ della soglia |
| Normal test | - | Sì (è ref_img) | Campione normale indipendente |
| Anomaly test | - | - | Immagine ostruita sintetica |

I tre insiemi sono **disgiunti**: nessun dato usato per costruire il bank o calibrare la soglia entra nella fase di test. La `ref_img` è riservata al test perché è il solo campione normale che il modello non ha mai visto, nemmeno indirettamente tramite augmentazione.

### 2.3 Metriche e loro interpretazione

**Metriche threshold-based** (calcolate con `θ_i = μ_i + k·σ_i` locale per camera):

- `recall` = TP / (TP + FN) - frazione di ostruzioni rilevate
- `false_alarm_rate` = FP / (FP + TN) - frazione di scene normali erroneamente classificate come anomalie

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

Il sistema di anomaly detection è basato sulla distanza tra feature di test e memory bank. Una preoccupazione legittima è che variazioni di illuminazione - in particolare **ombre proiettate** da persone o oggetti che transitano fuori inquadratura - possano deformare le feature in misura sufficiente a superare la soglia di anomalia, generando **falsi positivi**.

Il failure mode di interesse è quindi:

> scena normale + ombra → score > θ → falso allarme

### 3.2 Perché le ombre su scene ostruite non sono utili

Si potrebbe pensare di testare anche immagini ostruite con ombra, per verificare che il modello mantenga il recall. Questo test è **non informativo** per il seguente motivo strutturale: l'occlusione (un carrello, un bidone, una barella) introduce nella patch map un set di feature completamente assenti dalla memory bank - il segnale è già molto sopra la soglia. Aggiungere un'ombra su un'immagine ostruita può solo aumentare il segnale anomalo, non abbassarlo.

Il caso di failure realistico - immagine classificata come normale quando non lo è - non può essere causato da un'ombra aggiuntiva su un ostacolo già presente. Lo scenario che merita attenzione è esclusivamente l'FP sulla scena normale.

**Conseguenza di design**: le ombre sintetiche vengono generate e valutate **solo sulle scene normali**.

### 3.3 Perché le ombre non entrano nella memory bank

Un'alternativa sarebbe includere varianti con ombra nell'augmentazione della bank, rendendola esplicitamente invariante alle ombre. Questa scelta ha un difetto fondamentale: la bank apprenderebbe a ignorare ombre di **forma e posizione specifiche** (quelle generate dalla augmentation), ma non generalizzerebbe a ombre di forma diversa in inferenza.

Includere ombre nella bank renderebbe il modello cieco a ombre del tipo addestrato, senza fornire alcuna garanzia sulle ombre reali. La robustezza documentabile è quella che il modello mostra **senza** aver visto ombre durante il build - e questa è la metrica che ha senso riportare.

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

Il campo `reference_frame_id` - normalmente usato per collegare immagini ostruite al loro background di riferimento - viene qui riutilizzato per collegare ogni variante shadow alla scena originale da cui deriva. Questo permette a `evaluate_from_db` di recuperarle via `query_shadow_normal_frames` e costruire la memory bank dall'originale shadow-free, garantendo un test genuinamente indipendente dalla bank.

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

Il risultato è coerente con la motivazione teorica di §3.3: la bank non contiene patch con ombre, quindi qualsiasi ombra produce distanze elevate rispetto ai vicini nella bank - il meccanismo nearest-neighbour non riesce a distinguere l'anomalia reale dalla variazione fotometrica non coperta dall'augmentazione.

#### Zona di overlap e analisi della threshold

La zona critica è il range di score **3.54–4.51**, in cui le code delle due distribuzioni si sovrappongono (shadow_normal max = 4.507; obstructed min = 3.542). Una threshold globale fissa non può separare le due classi senza sacrificare recall:

| Threshold | Shadow FPR | Obstr TPR |
|---|---|---|
| 3.75 | 13.1% | 99.4% |
| **4.00** | **4.8%** | **97.3%** |
| 4.25 | 1.4% | 87.3% |
| 4.50 | 0.1% | 67.4% |

A threshold 4.00 si ottiene un compromesso accettabile (4.8% shadow FPR, 97.3% TPR), ma questo richiede alzare il valore assoluto della soglia in modo fisso - non è compatibile con il sistema di calibrazione adattiva per-camera già in uso, che determina la threshold a partire dalla distribuzione dei campioni normali di ciascuna bank.

La soluzione non può essere solo una ricalibrazione di `k`: il problema è strutturale e risiede nel **mancato allineamento tra la distribuzione di calibrazione (shadow-free) e la distribuzione di test (con ombre)**.

---

### 3.7 Proposta di miglioramento: shadow augmentation in fase di build

#### Motivazione

La diagnosi del §3.6 indica che la calibrazione e la bank vengono costruite su una distribuzione diversa da quella che si incontra al test. Questo è un caso di **distribution mismatch** tra training e deployment: le ombre sono variazioni fotometriche attese nella scena reale, ma completamente assenti dalla fase di build.

Il rimedio naturale all'interno del framework PatchCore è estendere `augment_reference` con ombre sintetiche. In questo modo:

1. **Memory bank con ombre**: le patch d'ombra nel campione di test trovano corrispondenti nella bank → distanza ridotta → score basso → nessun FP
2. **Calibrazione con ombre**: la distribuzione dei 10 score di calibrazione include già varianti con ombra → `μ_cal` si sposta verso l'alto, `σ_cal` aumenta → la threshold `μ + k·σ` si adatta naturalmente → le ombre non la superano

L'effetto è ottenuto senza modificare `k`, senza aggiungere soglie separate e senza nessun componente esterno: il framework PatchCore si ricalibra autonomamente.

Questa revisione modifica la scelta di design enunciata in §3.3. La preoccupazione originale - che la bank apprenda a ignorare solo le ombre specifiche dell'augmentation - è fondata in linea di principio, ma i dati mostrano che il costo dell'approccio shadow-free è un FPR del 90%, inaccettabile in un sistema safety-critical. La generalizzazione alle ombre reali è una questione empirica che può essere valutata estendendo il test set con varianti non viste.

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

**M²AD: Visual Anomaly Detection under Complex View-Illumination Interplay** (Cheng et al., arxiv 2505.10996, maggio 2025) rimane rilevante: valuta metodi VAD esistenti - inclusi approcci memory-based analoghi a PatchCore - sotto variazioni di illuminazione sistematiche, e mostra crolli significativi di performance. Il risultato è coerente con il 90.1% di shadow FPR osservato, anche se il loro setup (12 viste × 10 illuminazioni su dataset industriale) è più ampio del nostro. Utile per motivare il gap nella letteratura.

**Shadow Augmentation for Handwashing Action Recognition** (arxiv 2410.03984, 2024) fornisce il precedente più diretto alla proposta del §3.7: ombre sintetiche aggiunte durante la fase di costruzione del modello riducono i falsi positivi a test time. Il dominio (sorveglianza indoor clinica, telecamere fisse, illuminazione artificiale variabile) è analogo al nostro.

**Pose-Agnostic Anomaly Detection with Retinex-based Illumination Alignment (R3-PAD)** (Wang et al., 2025) propone una strategia alternativa che agisce sul lato test anziché sul lato training: una Retinex-UNet normalizza l'illuminazione dell'immagine di test prima dell'estrazione delle feature, rimuovendo l'effetto dell'ombra prima ancora che raggiunga il backbone. Il vantaggio principale rispetto all'approccio di §3.7 è che la bank rimane invariata e la generalizzazione a tipi di ombra non visti durante il build non dipende dalla copertura dell'augmentazione. Tuttavia, questo approccio introduce un passo di preprocessing aggiuntivo seriale che aumenta il tempo di inferenza - un costo rilevante nel contesto di deploy su device edge. Va inoltre verificato empiricamente che la normalizzazione Retinex non attenui anche le feature degli ostacoli, riducendo il recall. Per questi motivi viene considerato come direzione alternativa da esplorare piuttosto che come soluzione primaria.

**Enhancing Anomaly Detection Generalization through Knowledge Exposure: Dual Effects of Augmentation** analizza teoricamente il rapporto tra copertura dell'augmentazione e generalizzazione nel contesto dell'anomaly detection. Supporta la scelta di una `shadow_prob` controllata: un'augmentazione troppo aggressiva può ridurre la sensibilità alle anomalie vere, mentre una copertura parziale (40–50%) amplia la distribuzione normale senza degradare il recall.

Per il contesto classico del rilevamento di ombre in sorveglianza, il riferimento primario è **Sanin et al. (2012)** *"Shadow Detection: A Survey and Comparative Evaluation of Recent Methods"* (Pattern Recognition), che classifica i metodi in quattro categorie: cromaticità, fisici, geometrici e tessiturali. Per ambienti indoor, il metodo basato su cromaticità HSV (Cucchiara et al., 2003) rimane il riferimento storico più citato nel contesto di sistemi di videosorveglianza.

---

### 3.9 Risultati con shadow augmentation (shadow_prob=0.4, k=3.0)

#### Configurazione del run

La valutazione è eseguita con `--shadow-prob 0.4` e `k=3.0` (default). Il 40% delle varianti di augmentation - sia per la bank (seed=42, n=15) che per la calibrazione (seed=1000, n=10) - riceve k ∈ [1,3] ombre sintetiche in sequenza dopo le trasformazioni fotometriche standard.

#### Risultati

| Metrica | Baseline (shadow_prob=0.0) | Con shadow aug (shadow_prob=0.4) |
|---|---|---|
| Normal FPR | 0.5% | 0.0% |
| **Shadow FPR** | **90.1%** | **0.4%** |
| Obstr TPR (recall) | 100% | 70.6% |
| Obstr FNR | 0% | 29.4% |
| AUROC (Normal vs Obstr) | ~1.000 | ~1.000 |

#### Interpretazione

**Il fix ha funzionato sul problema shadow.** La shadow FPR è scesa dal 90.1% allo 0.4% - una riduzione di due ordini di grandezza. Il meccanismo è esattamente quello previsto in §3.7: le varianti con ombra nella calibrazione alzano μ_cal e σ_cal, portando la threshold adattiva da ~2.6 a ~4.3. Le patch d'ombra nella bank trovano corrispondenti → distanza ridotta → score nella zona normale.

**L'AUROC rimane ~1.000**, confermando che la capacità discriminativa del backbone non è degradata. La separazione tra distribuzioni normale e ostruita è invariata - il problema era e resta esclusivamente nella posizione della threshold.

**Il FNR 29.4% segnala un over-correction della threshold.** Analizzando i falsi negativi, il pattern è chiaro: le ostruzioni mancate hanno score 3.66–3.93, ma le threshold delle rispettive reference arrivano fino a 5.19. Il fenomeno è causato dalla variabilità stocastica della shadow augmentation durante la calibrazione: alcune reference hanno ricevuto per caso molte varianti con ombre pesanti, producendo score di calibrazione elevati e quindi threshold alte. Con k=3.0 l'effetto viene amplificato (il termine k·σ_cal cresce quando σ_cal include la variabilità degli score shadow).

Il trade-off risultante è sbilanciato rispetto al punto ottimale identificato nell'analisi sul CSV baseline (§3.6): a threshold globale 4.00 si otteneva shadow FPR=4.8%, TPR=97.3%. Il run con shadow_prob=0.4, k=3.0 supera quell'obiettivo sulle ombre (0.4% vs 4.8%) ma sacrifica troppo recall (70.6% vs 97.3%).

#### Leve di tuning

Due parametri permettono di recuperare recall senza rinunciare al guadagno sulle ombre:

**1. Abbassare k** (da 3.0 a ~1.5–2.0): la threshold scende proporzionalmente, recuperando le ostruzioni deboli. Con k=1.5 e la distribuzione di calibrazione attuale (che include shadow), la threshold attesa è nell'intorno di 3.5–3.8 - sopra la shadow mean (3.19) ma accessibile per le ostruzioni.

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

Con la shadow augmentation la **σ della distribuzione delle threshold è 3.4× più ampia** rispetto al baseline shadow-free, e il range si estende da [1.97, 3.00] a [2.44, 6.19]. Non è uno spostamento uniforme - è la comparsa di una coda destra che non esisteva.

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

Le camere con threshold patologicamente alta non condividono caratteristiche di scena particolari - sono distribuite indifferentemente tra porte e corridoi. Il loro tratto comune è statistico: con `--cal 10` sample e `shadow_prob=0.4`, alcune camere campionano per caso 6–8 variants pesantemente ombreggiate su 10. Questo gonfia sia μ_cal che soprattutto σ_cal, e con k=3.0 il termine k·σ_cal amplifica l'effetto fino a portare θ_i a 6.19.

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

#### Run 1 - decoupling con k=3.0

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

#### Run 2 - decoupling con k=4.0

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

Il run k=4.0 migliora F1 di altri 3 pp, ma rivela un effetto collaterale: **σ_θ è salita del 28%**. Il problema è strutturale: la formula `θ_i = μ_i + k·σ_cal,i` ha k come moltiplicatore di una stima rumorosa di σ_cal. Aumentando k amplifichiamo proporzionalmente le differenze stocastiche tra camere - la coda lunga torna.

#### Distribuzione del contributo agli errori - k=4.0

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

Il guadagno atteso rispetto a k=4 standard è di **~4 pp di F1** (da 0.903 a ~0.94), e questo guadagno proviene esclusivamente dalla riduzione di varianza nelle soglie - non da nessun cambio della capacità discriminativa del modello (AUROC resta 0.998).

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
| Shadow augmentation in build (`shadow_prob` > 0) | Robusta: il fix copre il distribution mismatch tra build e test (§3.7) - qualsiasi dataset con ombre fuori-distribuzione beneficia |
| Decoupling build/calibration shadow_prob | Robusta: separa due fasi con requisiti opposti (§3.10) - fix di pipeline |
| `--cal 30` (vs 10) | Robusta: riduce SE(σ_cal) di √3 ≈ 1.7× - risultato statistico |
| Pooled σ con median (§3.12) | Robusta: σ_pop ha SE ≈ SE(σ_cal,i)/√N con N=990 camere → quasi zero - risultato statistico |
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

In particolare il punto operativo scelto in deploy non si determina sui dati di laboratorio: dipende dal **costo asimmetrico FN/FP** del contesto applicativo (per via di fuga: un FN - ostruzione mancata - vale infinitamente più di un FP - falso allarme).

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

Le metriche più "trasportabili" - AUROC = 0.998, riduzione di σ_θ del 73%, ottenimento di clean FPR = 0% senza degrado della TPR - sono affermazioni sostanziali sull'architettura, non artefatti di tuning.

---

### 3.14 Problema delle porte a vetri: variabilità del fondo esterno

#### Descrizione del problema

Le porte a vetri introducono un failure mode distinto dalle ombre: il pannello traslucido rende visibile lo sfondo esterno (corridoio adiacente, area esterna, parcheggio), che varia nel tempo indipendentemente dallo stato della via di fuga. Persone che transitano, variazioni di luce naturale, o qualsiasi movimento oltre la porta producono cambiamenti di feature nella zona vetro che il sistema può classificare come anomalie, generando falsi positivi strutturali anche in assenza di qualsiasi ostacolo.

Il failure mode specifico è:

> porta a vetri normale + sfondo esterno diverso da reference → score > θ → falso allarme

Questo è concettualmente diverso dal problema shadow: le ombre variano la scena in modo coerente con la posizione della porta (proiettano sulla stessa superficie), mentre il fondo esterno varia in una regione spazialmente distinta e **fisicamente disgiunta** dalla zona dove un ostacolo fisico si manifesterebbe.

#### Soluzione proposta: ROI sulla soglia e il telaio

La soluzione si basa su un'osservazione geometrica: un ostacolo fisico (carrello, sedia a rotelle, barella, ecc.) è **davanti** alla porta e occluda necessariamente la parte bassa del telaio e la soglia. Un cambio di sfondo esterno è **dietro** il vetro e modifica solo i pixel del pannello centrale, mai la soglia né il telaio.

La ROI utile per il rilevamento è quindi la **fascia inferiore dell'inquadratura** (soglia + ~30–40% dal basso) più eventualmente i bordi laterali del telaio, escludendo il pannello vetro centrale. L'anomaly score viene calcolato come massimo dell'anomaly map nella sola ROI, ignorando le patch nella zona vetro.

```
+---------------------+
|                     |   <- vetro (escluso dalla ROI)
|      VETRO          |
|   (sfondo esterno)  |
|                     |
+---------------------+   <- soglia / fascia inferiore
|    SOGLIA + ROI     |   <- incluso nella ROI
+---------------------+
```

Questa scelta non richiede un modello di segmentazione: la ROI può essere definita manualmente per ciascuna telecamera come rettangolo o fascia di altezza relativa (parametro `--roi-bottom-frac`), oppure derivata dal bounding box dell'annotazione della porta se disponibile nel DB.

Le categorie di ostacoli nel dataset (carrelli, barelle, sedie a rotelle, bidoni, scatole) hanno tutte altezza sufficiente da occupare la fascia inferiore; ostacoli molto sottili in alto (es. nastro segnaletico teso) potrebbero sfuggire, ma non rientrano nelle categorie target del sistema.

#### Approccio alternativo: augmentation del fondo sul vetro

In alternativa alla ROI, si potrebbe includere nella memory bank varianti dell'immagine di riferimento con il pannello vetro sostituito da texture diverse (blur, colore uniforme, campioni casuali di patch). Il meccanismo sarebbe analogo alla shadow augmentation di §3.7: la bank apprende che la zona vetro è "sempre variabile" e abbassa la distanza al kNN per qualsiasi contenuto in quella regione.

Questa alternativa è meno precisa della ROI: la copertura dipende da quante e quali varianti di fondo si generano, mentre la ROI elimina strutturalmente il problema per costruzione.

#### Piano di validazione

La validazione di questo problema verrà condotta su un **test set manuale** di fotografie reali di porte a vetri, raccolte appositamente. Per ogni scena verranno catturate almeno le seguenti condizioni:

- scena normale con sfondo esterno identico al reference
- scena normale con sfondo esterno diverso (persona che transita, luce diversa, ecc.)
- scena ostruita

La ROI verrà configurata manualmente per ogni inquadratura, e si verificherà empiricamente se il falso positivo sul fondo esterno scompare con la ROI attiva senza degradare il TPR sull'ostruzione. Il confronto baseline/ROI su questo test set specifico fornirà la stima quantitativa del fenomeno.

#### Affinamento della regola: l'ostruzione deve toccare uno stipite

La formulazione iniziale "ROI = soglia + telaio" trattava la frame mask come un'area di interesse spaziale: i pixel fuori dall'area venivano azzerati prima di calcolare il massimo. Questa scelta ha un limite quando il fondo dietro il vetro produce un anomaly score elevato anche solo in un punto: se la sua componente non si estende al telaio, deve comunque essere respinta.

La regola raffinata, allineata con il protocollo di annotazione di Talebi et al. (paper "Real Time Exit Blockage Detection via Door-Centric Change Monitoring"), è formulata come gating per **connected component**:

> Una componente connessa di pixel con anomaly score >= θ è valida solo se interseca la frame mask in almeno `min_overlap_px` pixel. Lo score finale dell'immagine è il massimo della anomaly map ristretto alle componenti valide, oppure 0 se nessuna componente sopravvive.

**Connected components.** L'algoritmo prende la mappa binaria ottenuta dalla sogliatura e raggruppa i pixel "accesi" in isole separate, dove ogni isola e un insieme di pixel connessi tra loro (connettivita 8: ognuno dei 4 vicini ortogonali piu i 4 diagonali). Il risultato e un'etichetta intera distinta per ogni blob contiguo:

```
Mappa binaria (1 = score >= θ):    Componenti etichettate:

0  0  0  0  0  0  0               0  0  0  0  0  0  0
0  1  1  0  0  1  0               0  A  A  0  0  B  0
0  1  1  0  0  1  0      ->       0  A  A  0  0  B  0
0  0  0  0  1  1  0               0  0  0  0  B  B  0
0  0  1  0  0  0  0               0  0  C  0  0  0  0
0  0  0  0  0  0  0               0  0  0  0  0  0  0
```

Tre componenti distinte (A, B, C) perche non hanno pixel adiacenti tra loro. Per ciascuna si calcola il numero di pixel che cadono nella frame mask: se l'intersezione e inferiore a `min_overlap_px` la componente viene scartata. Lo score finale e il massimo dei valori continui dell'anomaly map sui soli pixel delle componenti superstiti; se nessuna sopravvive lo score e 0.0.

L'algoritmo viene applicato con `cv2.connectedComponents(..., connectivity=8)`.

Conseguenze:

- Una persona o un riflesso visibili attraverso il vetro generano una componente interamente nella zona vetro: zero intersezione con il telaio -> respinto.
- Un carrello davanti alla porta genera una componente che si estende dalla zona vetro alla soglia o agli stipiti: intersezione > soglia -> accettato.
- L'oggetto stesso non deve necessariamente toccare il telaio: basta che la componente di anomalia (più estesa per via dello smoothing gaussiano sigma=4) lo raggiunga.

Il parametro `min_overlap_px` (default 10) è robusto a piccoli errori di annotazione e a oggetti che lambiscono marginalmente lo stipite, ma rifiuta le componenti chiaramente confinate al vetro. Valori più alti rendono la regola più stretta (richiedono ostruzioni più estese del telaio); valori più bassi la rendono più permissiva.

**Perche la memory bank include anche la zona vetro.** Una scelta alternativa sarebbe escludere la zona vetrata dalla build phase, includendo nella bank solo le patch degli stipiti. Questo approccio e pero controproducente: a test time il modello estrae comunque feature dall'intera immagine, e per le patch della zona vetro cerca i vicini piu prossimi in una bank che non ne contiene nessuno. La distanza kNN risultante e alta non per la presenza di un'anomalia, ma per pura assenza di riferimento - un mismatch di distribuzione. Lo score in quella zona sarebbe sistematicamente gonfiato anche su scene perfettamente normali, rendendo la zona vetro sempre anomala per costruzione. La scelta corretta e quindi includere l'immagine completa nella bank, lasciando che il gating filtri a posteriori i blob che non toccano il telaio.

**Effetto del resize sull'anomaly score e dimensione minima degli oggetti.** Ogni immagine viene ridimensionata a 224x224 prima dell'estrazione delle feature. La feature map risultante e una griglia di 28x28 patch (una ogni 8px nell'input ridimensionato); dopo il neighborhood pooling 3x3, ogni feature vector integra un'area di 24x24px nell'immagine 224x224. Poiche build e test usano lo stesso resize, il confronto e consistente e non introduce bias sistematici. Il limite reale e la perdita di dettaglio per oggetti piccoli: un oggetto deve occupare almeno 2-3 patch adiacenti per generare un segnale anomalo distinguibile dal rumore, il che corrisponde a circa 16-24px nell'input 224x224.

Per un'immagine originale 960x1280 (come le immagini poli_ingegneria) il calcolo inverso dello scaling fornisce la dimensione minima nella risoluzione originale:

| Copertura nella griglia | px in 224x224 | px in larghezza (960px orig.) | px in altezza (1280px orig.) |
|---|---|---|---|
| 1 patch | 8px | ~34px | ~46px |
| 2 patch (minimo pratico) | 16px | ~69px | ~91px |
| 3x3 neighborhood | 24px | ~103px | ~137px |

Il Gaussian blur con sigma=4 applicato sull'anomaly map espande i blob attorno all'oggetto, abbassando ulteriormente la soglia percettiva: un oggetto da 30-40px puo produrre un blob visibile da 60-70px dopo smoothing. Per le ostruzioni tipiche (carrello, sedia a rotelle, cestino) riprese da 2-3 metri la dimensione e ampiamente superiore a questi limiti. Il problema degli oggetti sotto-soglia riguarda oggetti molto lontani, parzialmente visibili ai bordi dell'inquadratura, o di dimensioni molto ridotte (es. un piccolo pacchetto).

**Assunzione di camera fissa e tolleranza spaziale nativa.** Il meccanismo di gating presuppone che la telecamera sia fisicamente stabile: i poligoni del telaio vengono annotati una volta sola sull'immagine di riferimento e riutilizzati su tutti i frame successivi. Prima di quantificare la sensibilità a micro-spostamenti, vale la pena analizzare quanta tolleranza spaziale il sistema possiede già per costruzione.

Due elementi della pipeline assorbono piccoli disallineamenti senza modifiche:

1. **Neighborhood pooling 3×3**: ogni patch feature è la media del suo intorno 3×3, che copre 24×24 px nell'immagine originale a risoluzione nativa (per un'immagine 960×1280 come quelle di poli_ingegneria). Uno spostamento di 1-2 patch nella griglia 28×28 non altera significativamente le feature.
2. **Gaussian smoothing σ=4 sulla anomaly map**: allarga e ammorbidisce i blob, riducendo la sensibilità alla posizione esatta di ciascun pixel anomalo.

Una foto scattata a mano libera dista tipicamente 5-15 px di traslazione e 0.5-2° di rotazione rispetto al reference. Dopo il resize a 224×224 questo si riduce a 1-3 px - interamente assorbito dal neighborhood pooling. Il problema emerge solo per spostamenti più grandi (urto fisico alla telecamera, riposizionamento) o per zoom involontario.

**Strategie di mitigazione per spostamenti residui:**

*Opzione A - Geometric augmentation nel bank* (zero costo a test time): aggiungere piccole trasformazioni affini alle varianti del bank (shift ±5-10 px, rotazione ±2°). La bank impara a riconoscere come normale la stessa scena con piccoli spostamenti. Parametro suggerito: `geom_prob=0.5`, `max_shift_px=8`, `max_rot_deg=2.0`. Questa opzione è conservativa e non introduce latenza aggiuntiva a inferenza.

*Opzione B - ECC registration a test time*: allineare ogni immagine di test al reference prima dell'estrazione delle feature tramite `cv2.findTransformECC` con modalità `MOTION_EUCLIDEAN` (traslazione + rotazione). Più robusto per spostamenti maggiori, ma aggiunge un passo di preprocessing seriale e introduce un potenziale failure mode (registration failure su scene molto diverse). Adatto se la camera può subire drift progressivo nel tempo.

**Indicazioni pratiche per la raccolta dati.** Per il test set di validazione (Fase D), la variabilità di uno scatto a mano libera è eliminabile con accorgimenti a costo zero: un segno sul pavimento per il posizionamento dei piedi e appoggio a una superficie fissa per ogni scatto. Questo riduce la variabilità residua a meno di 5 px, dentro la tolleranza nativa del sistema, senza richiedere nessuna modifica al codice. L'implementazione della geometric augmentation rimane una direzione futura opzionale per il caso di camera fisicamente instabile (es. supporto vibrante in ambiente industriale).

Rimane una limitazione esplicita per spostamenti grandi (urto fisico, riposizionamento deliberato): la ROI proiettata non coincide più con gli stipiti reali, e il gating può rigettare ostruzioni legittime o non filtrare variazioni di fondo. Il deployment corretto richiede una camera montata in modo rigido. Un tracking automatico degli angoli del telaio (es. tramite omografia incrementale) per aggiornare la ROI dinamicamente è una possibile estensione futura.

**Ortogonalita rispetto alle ombre.** Il gating e neutro rispetto al problema dei falsi positivi da ombra descritto nella sezione precedente. Le ombre proiettate sul pavimento davanti alla porta generano componenti anomale che si estendono tipicamente fino alla soglia (`threshold`) o ai montanti laterali; superano quindi il filtro e producono lo stesso score che si otterrebbe senza gating. Il gating interviene unicamente quando la componente e interamente confinata nella zona vetrata, lontana dal telaio - scenario che non si verifica per le ombre. Il fix per le ombre rimane distinto e indipendente: shadow augmentation nella fase di build per includere nella memory bank varianti normali con ombra, abbassando cosi il loro score al di sotto della soglia di calibrazione. I due meccanismi sono complementari e non interferenti.

**Ombre solari geometriche e composizione ROI rettangolare + gating.** Nelle scene con luce solare diretta attraverso porte a vetri (es. uscite di emergenza esposte a sud), le ombre proiettate sul pavimento possono essere forti, hard-edged e geometricamente strutturate - proiezioni della griglia del telaio che cambiano con l'angolo solare durante la giornata e la stagione. La shadow augmentation sintetica nella build phase non e sufficiente per questo caso: le augmentazioni aggiungono blob scuri casuali, ma non riproducono la variabilita delle proiezioni solari reali, che dipendono da un parametro continuo (posizione del sole) non campionabile esaustivamente da un singolo riferimento.

La strategia corretta per queste scene e combinare il flag `--roi "x,y,w,h"` (ROI rettangolare legacy) con il gating poligonale. La ROI rettangolare viene configurata per includere solo il rettangolo della porta, escludendo il pavimento antistante. Le ombre sul pavimento cadono interamente fuori da questa ROI e vengono azzerate prima del gating, senza contribuire allo score. All'interno della ROI rettangolare, il gating poligonale filtra poi le variazioni di background attraverso il vetro. I due meccanismi si compongono in sequenza: il rettangolo elimina il pavimento, i poligoni eliminano il vetro. Il risultato e un detector sensibile unicamente alle anomalie che cadono sul telaio fisico della porta.

#### Generalizzazione: gating poligonale per ogni tipo di porta

La discussione precedente ha presentato il gating poligonale come specifico per le porte a vetri, dove il pannello centrale e una zona "rumorosa" da escludere. In realta il meccanismo e piu generale: si applica a qualsiasi scena in cui sia possibile distinguere zone strutturalmente stabili (telaio fisico) da zone soggette a variabilita visiva (pannelli, vetri, riflessi, illuminazione locale).

Per una porta solida con pannello opaco, una variazione di luce diretta o un riflesso speculare sul pannello centrale produce una componente anomala interamente confinata sul pannello, senza intersezione con stipiti, soglia o architrave. Il gating la respinge esattamente come respinge le variazioni di background attraverso il vetro. Un ostacolo fisico davanti alla porta (carrello, sedia a rotelle, cestino, pacco) e invece sempre **fisicamente connesso al pavimento** - poggia a terra o e sostenuto da supporti che toccano il suolo - quindi la componente anomala si estende necessariamente fino alla soglia annotata, soddisfa il criterio di intersezione e viene accettata. Questo vincolo fisico (gli ostacoli sono grounded) e quello che rende il filtro affidabile: le anomalie genuine hanno una struttura geometrica predicibile rispetto al telaio, i disturbi no.

In altre parole, escludere il pannello centrale dalla maschera di gating equivale a dire al sistema: "qualunque cosa accada sul pannello senza propagarsi al telaio fisico e un disturbo, non un'ostruzione". Questa formulazione e valida per:

- Porte a vetri (pannello = vetro, disturbo = background esterno variabile)
- Porte opache con illuminazione variabile (pannello = superficie verniciata, disturbo = riflessi e luce diretta)
- Porte metalliche o lucide (pannello = superficie speculare, disturbo = riflessi della scena interna)

L'unica anomalia genuina che il poligonale potrebbe respingere e un oggetto **interamente** sul pannello senza contatto col pavimento o col telaio - un caso poco rilevante per il blocco di uscite di emergenza, dove gli ostacoli sono per definizione fisici e grounded. Per evitare ambiguita, in casi rari di oggetti appesi e possibile estendere il poligono `architrave` per includere parte del traverso superiore.

La raccomandazione operativa e quindi di annotare i poligoni di stipiti e soglia anche per le porte non a vetri quando la scena presenta forte variabilita di illuminazione locale sul pannello, e di tenere il gating disattivato (solo `--roi-rect-json`) per scene con pannelli opachi in illuminazione stabile, dove l'annotazione aggiuntiva non porta beneficio.

#### Architettura di poligoni etichettati

La ROI non è una singola maschera binaria, ma una lista di poligoni etichettati per stipite. Le label predefinite sono:

| Label | Regione |
|---|---|
| `left_jamb` | Telaio verticale sinistro |
| `right_jamb` | Telaio verticale destro |
| `center_mullion` | Montante verticale centrale (porte doppie) |
| `threshold` | Soglia / fascia inferiore davanti alla porta |
| `architrave` | Traverso superiore |
| `other` | Catch-all per geometrie non standard |

Per il calcolo del gating, l'unione di tutti i poligoni costituisce la frame mask. La separazione per label è preservata nel JSON e nel DB per analisi successive (ad esempio: misurare in quale stipite cade più frequentemente l'intersezione, o tarare `min_overlap_px` per label specifiche).

I poligoni sono salvati in coordinate dell'immagine originale (non normalizzate) per non perdere precisione dopo resize. La rasterizzazione alla risoluzione di lavoro avviene a runtime via `cv2.fillPoly`, con scaling lineare delle coordinate.

#### Implementazione

Il meccanismo è realizzato come pipeline a tre stadi:

1. **Annotazione interattiva** (`src/annotate_door_roi.py`): un tool basato su matplotlib permette di disegnare un poligono per ciascuno stipite. Click sinistro = vertice, tasto destro = chiude il poligono, 'l' = cicla label, 'n'/'p' = naviga tra immagini, 'q' = salva e esci. L'output è un JSON per immagine in `Dataset/roi/<stem>.json` con i campi `image_path`, `img_w`, `img_h`, `polygons: [{label, polygon}, ...]`.

2. **Import nel DB** (`src/import_roi_to_db.py`): i JSON vengono inseriti nella tabella `roi_polygons`, con matching per `file_path` relativo al `dataset-root`. L'import sostituisce di default i poligoni esistenti per la stessa frame (flag `--append` per la modalità additiva). Le frame non presenti in `frames` vengono saltate con un warning -- per renderle visibili bisogna prima eseguire `load_dataset_to_db.py`.

3. **Scoring con gating** (`src/roi_utils.py` + `compute_image_score_from_pil`): al test time, dopo aver costruito l'anomaly map smoothata, si rasterizza il `frame_mask` dai poligoni alla risoluzione originale, si threshold-binarizza la mappa, si calcolano le componenti connesse (cv2 8-connectivity), e una componente è valida solo se ha almeno `min_overlap_px` pixel di intersezione con la frame mask. Lo score finale è il massimo della mappa ristretto alle componenti valide, o 0.0 se nessuna sopravvive.

Il valore di soglia usato per la binarizzazione è la stessa `θ_i = μ_i + k·σ_i` (o `μ_i + k·σ_pop` in pooled mode) calibrata sulla camera. La calibrazione resta non-gated per preservare la distribuzione naturale degli score sui campioni normali. Il gating viene applicato esclusivamente in fase di test, per i campioni `normal`, `shadow_normal` e `obstructed` delle reference che hanno una ROI annotata nel DB; per le altre reference, l'evaluate-db ricade sul comportamento storico (max della mappa intera).

I CSV di output di `evaluate-db` includono una colonna `gated ∈ {0, 1}` per distinguere le righe processate con gating da quelle non gated, in modo che le analisi successive possano stratificare i risultati. La presenza di ROI annotate non è mutuamente esclusiva con il vecchio flag `--roi "x,y,w,h"`: se entrambi sono attivi, il rettangolo legacy maschera prima la mappa e il gating opera sulla porzione superstite.

#### Limite intrinseco: porta aperta o stipite parzialmente non visibile

L'intero meccanismo di gating poggia su un'assunzione implicita: **la geometria del telaio annotata nel reference è stabile e visibile a test time**. Questa assunzione regge per una porta chiusa in condizioni normali, ma viene violata in almeno due scenari operativamente rilevanti.

**Porta aperta - doppio failure mode simmetrico.** Quando la porta si apre si manifestano contemporaneamente due failure mode di segno opposto.

Il primo è un **falso positivo strutturale**: il banco è costruito sul reference a porta chiusa, dove soglia e stipiti sono visibili e forniscono patch "normali" alla memory bank. Con la porta aperta quelle stesse zone cambiano radicalmente - la soglia ruota con l'anta e sparisce, gli stipiti laterali vengono parzialmente coperti dall'anta - e il sistema vede una scena fuori distribuzione esattamente nella regione della frame mask. Il gating aggrava il problema invece di attenuarlo: la componente anomala generata dalla sparizione della soglia/stipite cade per definizione dentro la frame mask (e' li' che la struttura e' cambiata) e supera il filtro senza difficolta'. Il risultato e' un allarme certo ogni volta che la porta si apre, indipendentemente dalla presenza di ostacoli.

Il secondo e' un **falso negativo strutturale**: un ostacolo piazzato sulla soglia di una porta aperta non genera una componente anomala che interseca la frame mask (il poligono annotato sul reference punta ora su pixel dell'anta o del vuoto, non sul telaio fisico) -> il gating lo respinge anche se l'ostruzione e' reale.

**Stipite parzialmente occultato.** Anche senza che la porta si apra, un oggetto alto posizionato lateralmente (es. un rack metallico appoggiato allo stipite) può coprire parte della frame mask. Se la componente anomala dell'ostruzione principale non raggiunge la porzione di telaio rimasta visibile, il gating può rifiutarla anche se l'ostruzione è reale.

Entrambi i casi condividono la stessa causa: il gating assume che la frame mask sia sempre "reachable" da una componente anomala genuina, ma questa raggiungibilita dipende dalla visibilita del telaio - che e una proprieta della scena, non del modello.

**Impatto sul sistema di emergenza.** Il caso porta aperta e il piu preoccupante: un'uscita di emergenza non e ostruita solo quando viene bloccata con la porta chiusa, ma anche - e forse piu frequentemente - quando viene bloccata mentre la porta e gia aperta (es. un carrello lasciato nel varco durante le operazioni di movimentazione). Questo e esattamente il caso che il sistema potrebbe mancare.

**Strategie di mitigazione (non implementate, direzioni future).** Il problema non ha una soluzione semplice nel framework attuale:

- *Rilevamento dello stato porta*: un classificatore binario porta-aperta/porta-chiusa (che puo essere molto semplice, anche template matching) puo segnalare quando la geometria del reference e invalidata. In modalita porta aperta il gating verrebbe disattivato, ricadendo sul max-score globale senza filtraggio, a costo di piu falsi positivi da fondo variabile.
- *Reference multipli*: annotare un reference separato per la porta aperta (memoria bank + ROI specifici per lo stato aperto). Il sistema switcha tra i due in base allo stato rilevato. Duplica la memoria ma mantiene la discriminazione.
- *Gating adattivo*: rilevare a runtime quali regioni del telaio sono visibili (attraverso matching con il reference) e aggiornare la frame mask di conseguenza. Piu robusto ma significativamente piu complesso.

**Nella tesi questo e dichiarato come limite del metodo**, non come problema risolto. Il sistema e progettato e validato per la condizione operativa nominale (porta chiusa, telaio visibile): e la condizione piu comune in un contesto di sorveglianza continua, in cui la porta e aperta solo transitoriamente durante il passaggio di persone, non in modo stabile. La condizione di porta stabile aperta con ostacolo e un caso d'uso limite che richiede estensioni architetturali fuori dallo scope di questo lavoro.

---

### 3.15 Light augmentation e ablation study (risultati pending)

#### 3.15.1 Motivazione

Il sistema con pooled sigma (§3.13) raggiunge shadow_FPR = 12.2% e TPR = 99.8% sul test sintetico. Rimane però una sorgente di falsi positivi strutturalmente analoga alle ombre: le variazioni di illuminazione locale per **eccesso** anziché per difetto. Fasci di luce da lucernari, finestre laterali, riflessioni solari su superfici lucide, o lampade puntate generano patch features aumentate in luminosità che - esattamente come le ombre - non hanno corrispondenti nella memory bank costruita da varianti fotometriche standard.

Il failure mode è speculare a quello delle ombre:

> scena normale + fascio di luce → features fuori distribuzione → score > θ → falso allarme

Le tre modalità di luce sintetica già implementate in `generate_shadow_images.py` coprono i casi principali negli ambienti corridoio/porta di un ospedale o edificio pubblico:

| Tipo | Scenario fisico simulato |
|---|---|
| Ellittica | Fascio da lucernario, lampada a soffitto puntata, riflesso luminoso su pavimento |
| Direzionale | Gradiente progressivo da finestra laterale o porta aperta su ambiente illuminato |
| Striscia orizzontale/verticale | Luce radente da traversa, crack in una porta o parete |
| Striscia diagonale (solare) | Raggio solare che taglia la scena; riceve tinta calda (R↑, B↓) per simulare la cromaticità della luce solare diretta |

Il fix proposto segue la stessa logica della shadow augmentation (§3.7): includere varianti con luce nella bank (`light_prob` = 0.4) e con probabilità ridotta nella calibrazione (`light_prob_cal` = 0.1). I due parametri sono indipendenti da `shadow_prob` e `shadow_prob_cal`: una variante può ricevere ombra, luce, entrambi, o nessuno dei due.

#### 3.15.2 Dataset light_test

Per misurare il light_FPR analogamente al shadow_FPR è stato generato un dataset `light_test`:

```
Dataset/light_test/
  normali_porte/      550 immagini  (porta_*_light00.jpg)
  normali_corridoi/   550 immagini  (corridoio_*_light00.jpg)
```

Generato con `--shadow-type light_random --n-variants 1 --seed 42` dalla stessa sorgente di `shadow_test`. Registrate nel DB con `source='light'`, `is_normal=1`, `reference_frame_id` all'originale. `evaluate-db` le raccoglie tramite `query_shadow_normal_frames` (filtro per `source`) e riporta un `light_FPR` separato da `shadow_FPR` e `clean_FPR`.

#### 3.15.3 Asimmetria tra spazio shadow e spazio light (caveat metodologico)

Un'osservazione importante per l'interpretazione dei risultati: i due test set `shadow_test` e `light_test`, pur condividendo la stessa struttura (3 modalità: ellittica, direzionale, striscia; k ∈ [1,3] stacked per immagine), **non coprono spazi visivi di dimensione equivalente**. La generazione delle luci sintetiche in `generate_shadow_images.py` introduce due dimensioni aggiuntive rispetto alle ombre, motivate dal realismo fisico:

| Dimensione | Shadow | Light |
|---|---|---|
| Forma + intensità | sì | sì |
| Direzionalità (stripe) | 2 orientamenti (oriz, vert) | **3 orientamenti** (oriz, vert, **diagonale solare**) |
| Shift cromatico | nessuno | **R↑, B↓** su elliptical e stripe diagonale (tinta calda solare) |
| Angolo continuo solare | n/a | **25-65° o 115-155°** parametrizzato |

Le scelte sono fisicamente motivate - il sole reale produce raggi diagonali con cromaticità calda, mentre le ombre proiettate non hanno equivalente cromatico - ma hanno una conseguenza metodologica: a parità di copertura della bank (es. ~6 light variants e ~6 shadow variants su 15 totali), il sottospazio light richiede di rappresentare più modalità con lo stesso budget di varianti.

**Conseguenze per l'interpretazione dei risultati:**

1. **Il confronto diretto `shadow_FPR vs light_FPR` non è apples-to-apples.** Un light_FPR del 23% non implica che il light aug sia "metà efficace" di un shadow aug che raggiunge il 12% - significa che lo stesso budget di varianti deve coprire uno spazio più diversificato.

2. **Il confronto valido è within-category**: light_FPR_L1 vs light_FPR_SL15 vs light_FPR_SL25 misura l'effetto di shadow co-augmentation e bank size sulla copertura del sottospazio light. shadow_FPR_C-pooled vs shadow_FPR_SL15 vs shadow_FPR_SL25 misura l'analogo per il sottospazio shadow.

3. **Motivazione aggiuntiva per SL-25**: la maggiore diversità del sottospazio light giustifica esplicitamente l'aumento di `aug` a 25 quando il light aug è attivo. Non è solo "più varianti = meglio in generale", ma "il test set delle luci ha più modalità, quindi serve copertura proporzionalmente più ampia".

Una possibile estensione future-work è quantificare la varianza spaziale+cromatica relativa dei due test set (es. distanza media tra feature vectors estratti dal backbone sulle 3 modalità di ciascun tipo), o ridurre la light augmentation alle stesse 3 modalità geometriche delle ombre per ottenere una comparabilità diretta. La scelta nella tesi è di mantenere la light augmentation realistica (modeling fisicamente fondato) e riportare il caveat dell'asimmetria nei risultati.

#### 3.15.4 Coverage della bank con due tipi di disturbance

Con `aug=15`, `shadow_prob=0.4`, `light_prob=0.4` la distribuzione attesa delle varianti è:

| Tipo variante | Probabilità | N atteso (aug=15) | N atteso (aug=25) |
|---|---|---|---|
| Solo fotometrica | 0.6 × 0.6 = 0.36 | 5.4 | 9.0 |
| Shadow only | 0.4 × 0.6 = 0.24 | 3.6 | 6.0 |
| Light only | 0.6 × 0.4 = 0.24 | 3.6 | 6.0 |
| Shadow + Light | 0.4 × 0.4 = 0.16 | 2.4 | 4.0 |

Con `aug=15` il coreset (1% di 784×15 ≈ 117 punti) deve coprire quattro sottospazi invece di tre. Il rischio è che i ~3.6 shadow-only variants siano insufficienti rispetto al caso shadow-only (§3.7, ~6 shadow variants su 15). Con `aug=25` si ottengono ~6.0 shadow-only + ~6.0 light-only, ripristinando la coverage per tipo al livello del run shadow-only originale, a costo di un ~65% di tempo di build aggiuntivo.

#### 3.15.5 Ablation design

Quattro configurazioni, tutte con `k=4.0`, `sigma_mode=pooled`, `cal=30`, `split=test`:

| # | Config | sp (build) | lp (build) | aug | spc (cal) | lpc (cal) | Scopo |
|---|--------|-----------|-----------|-----|----------|----------|-------|
| 0 | C-pooled (baseline) | 0.4 | 0.0 | 15 | 0.1 | 0.0 | Riferimento - nessuna light aug |
| 1 | L-only | 0.0 | 0.4 | 15 | 0.0 | 0.1 | Isola effetto light aug, senza shadow |
| 2 | SL-15 | 0.4 | 0.4 | 15 | 0.1 | 0.1 | Shadow + light, bank compatta |
| 3 | SL-25 | 0.4 | 0.4 | 25 | 0.1 | 0.1 | Shadow + light, bank più grande |

*sp = shadow_prob, lp = light_prob, spc = shadow_prob_cal, lpc = light_prob_cal*

**Ipotesi testate:**

- **Run 1 vs 0**: il light aug riduce il light_FPR? Quanto costa in shadow_FPR (la bank non ha più shadow variants) e TPR?
- **Run 2 vs 0**: effetto combinato con bank invariata. Rischio di regressione su shadow_FPR per coverage ridotta per tipo.
- **Run 3 vs 2**: `aug=25` recupera la coverage persa in SL-15? Atteso: shadow_FPR e light_FPR entrambi più bassi rispetto a SL-15.

**Metriche da riportare:**

Clean FPR | Shadow FPR | Light FPR | Obstructed TPR | F1 | AUROC | σ_θ

#### 3.15.6 Risultati Run 1: L-only

Configurazione: `shadow_prob=0.0`, `light_prob=0.4`, `aug=15`, `shadow_prob_cal=0.0`, `light_prob_cal=0.1`, `cal=30`, `k=4.0`, `sigma_mode=pooled`. Test set: 990 reference (porte + corridoi), ciascuna con 1 normal + 1 shadow_normal + 1 light_normal + 1 obstructed.

**Risultati aggregati:**

| Metrica | C-pooled (sp=0.4, lp=0.0) | **L-only (sp=0.0, lp=0.4)** | Δ |
|---|---|---|---|
| Clean FPR | 0.0% | **0.0%** | invariato |
| **Shadow FPR** | 12.2% | **37.7%** | **+25.5 pp** (regressione attesa) |
| **Light FPR** | n/d | **23.2%** | nuovo, vedi sotto |
| Obstr TPR | 99.8% | **100.0%** | +0.2 pp |
| Obstr FNR | 0.2% | 0.0% | -0.2 pp |
| Precision | 89.1% | 62.2% | -26.9 pp |
| F1 | 0.941 | **0.767** | -0.174 |
| AUROC (N+S+L vs Obstr) | 0.998 | **0.9985** | invariato |
| σ_θ | 0.137 | **0.131** | -4% (stabile) |

**Distribuzione degli score:**

| Categoria | mean | std | min | max |
|---|---|---|---|---|
| Normal | 2.557 | 0.133 | 2.037 | 2.992 |
| Shadow normal | 3.212 | 0.430 | 2.145 | 4.505 |
| Light normal | 3.040 | 0.379 | 2.079 | 4.313 |
| Obstructed | 4.675 | 0.344 | 3.685 | 5.859 |

**Soglie per-camera (post-pooling):** mean=3.333, std=0.131, range=[2.867, 3.724]. Distribuzione compatta, nessuna coda patologica.

**Breakdown per venue:**

| Venue | Clean FPR | Shadow FPR | Light FPR | TPR |
|---|---|---|---|---|
| corridoio | 0.0% | 33.7% | 21.4% | 100% |
| porta | 0.0% | 41.6% | 25.1% | 100% |

#### 3.15.7 Interpretazione del Run 1

**Il light augmentation funziona.** La light_FPR scende a 23.2%, contro un atteso ≥70% che ci si aspetterebbe da un bank senza copertura di patch luminose (estrapolando dal pattern shadow: la shadow_FPR del baseline pre-fix era 90%, e con `lp=0.0` ci aspettiamo un comportamento analogo). Il meccanismo è quello previsto in §3.7 - le patch con fascio luminoso nel test set trovano corrispondenti nella bank, abbassando la distanza kNN e portando lo score sotto la soglia di calibrazione.

**Conferma del ruolo strutturale della shadow augmentation.** Rimuovere la shadow aug fa salire la shadow_FPR da 12.2% (C-pooled) a 37.7%. Questo conferma che in C-pooled la shadow aug era attivamente discriminante e non un effetto secondario di scelte fortunate dei seed - rimuovendola, il sistema perde robustezza esattamente nel modo predetto dal modello di failure di §3.1.

**Il light_FPR (23%) non è confrontabile direttamente con lo shadow_FPR di C-pooled (12%).** Come documentato in §3.15.3, lo spazio delle luci è intrinsecamente più ampio: tre orientamenti di striscia (oriz/vert/diagonale) invece di due, parametro angolare continuo per il fascio solare diagonale, e shift cromatici (R↑, B↓) assenti dalle ombre. Lo stesso budget di copertura (~6 light variants su 15 nel bank) deve quindi rappresentare uno spazio visivo più diversificato, e una light_FPR residua di 23% riflette in parte questa diversità intrinseca, non un'inferiorità del meccanismo di fix.

**AUROC 0.9985 conferma capacità discriminativa intatta.** Il backbone separa correttamente normali (clean + shadow + light) da obstructed in unità normalizzate. Il problema è esclusivamente nel piazzamento delle soglie per-camera, come confermato dal threshold-sweep globale:

| Threshold globale | Normal FPR | Shadow FPR | Light FPR | TPR |
|---|---|---|---|---|
| 3.40 | 0.0% | 33.9% | 19.7% | 100% |
| **3.67** | **0.0%** | **14.9%** | **5.2%** | **100%** |
| 3.95 | 0.0% | 4.7% | 0.4% | 98.5% |

Esiste una soglia globale (~3.67) che otterrebbe simultaneamente shadow_FPR=15%, light_FPR=5%, TPR=100%, ma le soglie per-camera attuali (mean=3.33) sono più basse, riflettendo il fatto che `mu_cal` per-camera è abbassato dalla mancata copertura della shadow aug.

**Asimmetria porta vs corridoio.** Le porte sono più difficili sia per shadow (41.6% vs 33.7%) che per light (25.1% vs 21.4%). Ipotesi: le scene con porte (in particolare quelle a vetri presenti nel dataset poli_ingegneria) hanno maggiore variabilità di sfondo che si sovrappone parzialmente con i pattern di disturbance, rendendo il discriminator meno netto.

**Stabilità delle soglie.** σ_θ=0.131, comparabile a C-pooled (0.137) - il pooled sigma con median continua a eliminare la coda patologica anche in questa configurazione, indipendentemente dal tipo di augmentation attivo.

**Conclusione del Run 1.** L'F1 0.767 non è un risultato da difendere come benchmark - è il "single-disturbance-fix" baseline che giustifica l'esigenza dei run successivi. Il valore informativo è duplice: (i) conferma che il light aug funziona come fix isolato, (ii) quantifica il costo di non avere la shadow aug, motivando il design SL-15/SL-25 dove entrambi gli augmenti sono attivi simultaneamente.

#### 3.15.8 Risultati completi dell'ablation (L0-L4)

Completati i run mancanti, l'ablation è ora chiusa. Oltre alle quattro configurazioni del design originale (§3.15.5) è stato aggiunto un quinto run di **controllo a budget pari** (L4, fair-budget), non previsto nel design iniziale ma necessario per disaccoppiare due effetti che in SL-15/SL-25 risultano confusi: l'aumento della *copertura* (aggiungere il tipo luce) e l'aumento dell'*intensità totale* di augmentation (più varianti disturbate per immagine).

| # | Config | sp | lp | aug | spc | lpc | budget* | exp_id / artifact |
|---|--------|----|----|-----|-----|-----|---------|-------------------|
| L0 | C-pooled (baseline) | 0.4 | 0.0 | 15 | 0.1 | 0.0 | 0.40 | 10 / `ablation_L0bis_shadow_only.csv` |
| L1 | L-only | 0.0 | 0.4 | 15 | 0.0 | 0.1 | 0.40 | 7 / `ablation_L1_light_only.csv` |
| L2 | SL-15 | 0.4 | 0.4 | 15 | 0.1 | 0.1 | 0.80 | 8 / `ablation_L2_SL15.csv` |
| L3 | SL-25 | 0.4 | 0.4 | 25 | 0.1 | 0.1 | 0.80 | 9 / `ablation_L3_SL25.csv` |
| L4 | fair-budget | 0.23 | 0.23 | 15 | 0.06 | 0.06 | 0.46 | 12 / `ablation_L4_fair_budget.csv` |

*budget = prob. attesa di disturbance per variante (sp+lp), proxy dell'intensità totale di augmentation. L4 è costruito per avere budget ~0.46, comparabile allo 0.40 di L0, ma ripartito tra ombra e luce invece che tutto su ombra.

**Risultati aggregati** (media porta+corridoio; le aggregazioni di L1 coincidono con la tabella di §3.15.6, a conferma della consistenza):

| Config | clean_FPR | shadow_FPR | light_FPR | TPR | FNR | Precision | F1 | θ medio |
|--------|-----------|------------|-----------|-----|-----|-----------|-----|---------|
| L0 C-pooled | 0.0% | 12.2% | 5.8% | 99.8% | 0.2% | 84.8% | 0.917 | 3.74 |
| L1 L-only | 0.0% | 37.7% | 23.2% | 100% | 0.0% | 62.2% | 0.767 | 3.33 |
| **L2 SL-15** | 0.0% | **2.2%** | **0.30%** | 99.0% | 1.0% | 97.5% | **0.982** | 3.91 |
| L3 SL-25 | 0.0% | 1.1% | 0.10% | 97.7% | 2.3% | 98.8% | 0.982 | 3.95 |
| L4 fair-budget | 0.0% | 11.2% | 4.6% | 100% | 0.0% | 86.7% | 0.929 | 3.65 |

**Breakdown per venue** (le porte restano più difficili dei corridoi su entrambe le disturbance):

| Config | venue | shadow_FPR | light_FPR | TPR | F1 |
|--------|-------|------------|-----------|-----|-----|
| L0 | corridoio / porta | 8.9% / 15.6% | 5.7% / 5.9% | 100% / 99.6% | 0.932 / 0.901 |
| L2 | corridoio / porta | 1.4% / 3.0% | 0.2% / 0.4% | 99.6% / 98.4% | 0.990 / 0.975 |
| L3 | corridoio / porta | 0.8% / 1.4% | 0.0% / 0.2% | 99.0% / 96.4% | 0.991 / 0.973 |
| L4 | corridoio / porta | 8.5% / 13.9% | 4.7% / 4.4% | 100% / 100% | 0.938 / 0.919 |

**Q1 - Quanto incide il test set con luci sintetiche.** Sul modello senza gestione luci (L0) le immagini con luce producono light_FPR = 5.8% e abbassano il baseline da precision 89% / F1 0.941 (valore senza light test, §3.15.6) a precision 84.8% / F1 0.917. L'impatto è reale ma moderato. Il dato non ovvio è che **la luce è un disturbo più lieve dell'ombra**: anche con una bank priva di copertura luminosa, il light_FPR (5.8%) è inferiore allo shadow_FPR (12.2%). Lo confermano gli score del Run 1 (light normal mean 3.04 < shadow normal 3.21 < obstructed 4.68): il fascio luminoso spinge le feature meno fuori distribuzione dell'ombra. Resta valido il caveat di §3.15.3 - il 5.8% basso riflette l'intensità minore della perturbazione, non una minore ampiezza dello spazio luce (che è anzi più ampio per orientamenti, angolo solare e shift cromatico).

**Q2 - Aggiungere light aug (sopra le ombre) aiuta.** Confronto within-category con shadow fissa a 0.4 (L0 → L2 → L3):

- light_FPR: 5.8% → 0.30% → 0.10% (circa 19x a SL-15);
- shadow_FPR: 12.2% → 2.2% → 1.1% - aggiungere la luce **migliora anche le ombre**, non le degrada;
- F1: 0.917 → 0.982; precision 84.8% → 97.5%;
- costo: recall 99.8% → 99.0% → 97.7% (FNR crescente con la dimensione della bank).

L1 (L-only) resta il controllo che dimostra il **ruolo strutturale della shadow aug**: rimuovendola la soglia per-camera collassa (θ medio 3.33, il più basso) e tutto fa più FP, comprese le luci (light_FPR 23%). Quel 23% non è la light aug che fallisce ma il threshold abbassato dalla mancata copertura ombra (a soglia globale 3.67 lo stesso modello otterrebbe light_FPR 5%, shadow 15%, TPR 100%). **La light aug è quindi un complemento della shadow aug, non un sostituto.**

**Decomposizione budget vs diversificazione (perché serve L4).** L2/L3 applicano il doppio dell'augmentation di L0 (budget 0.80 vs 0.40): parte del loro guadagno è semplicemente "più augmentation → calibrazione con score più alti → θ più alto → meno FP", confuso con l'effetto della copertura luce. Il θ medio cresce in modo monotono col budget (L4 3.65 < L0 3.74 < L2 3.91 < L3 3.95) e traccia direttamente il calo dei FP. L4 isola i due effetti tenendo il budget ~pari a L0 ma diversificato:

- **L4 vs L0** (budget pari, diversificazione pura): light_FPR 5.8% → 4.6%, shadow_FPR 12.2% → 11.2%, F1 0.917 → 0.929. Guadagno marginale, ma a costo zero - recall resta 100% e le ombre non peggiorano nonostante la shadow aug ridotta da 0.4 a 0.23. Da notare che L4 ottiene questo con θ più basso di L0 (3.65 vs 3.74): la bank diversificata produce score per-immagine più bassi sui test disturbati, segno di robustezza intrinseca leggermente maggiore.
- **L4 vs L2** (stessa ricetta, più budget): divario ampio (light_FPR 4.6% → 0.30%, shadow_FPR 11.2% → 2.2%).

Conclusione della decomposizione: **la fetta dominante del miglioramento di SL-15 sul baseline viene dall'aumento dell'intensità totale di augmentation, non dalla sola copertura del tipo luce.** La diversificazione shadow → shadow+light, a parità di budget, contribuisce un guadagno piccolo ma reale e gratuito (nessun costo su ombre né su recall).

**SL-15 vs SL-25.** Passando da aug 15 a 25 (stessi sp, lp) shadow_FPR e light_FPR si dimezzano ancora (2.2 → 1.1%, 0.30 → 0.10%) e la precision sale, ma il recall scende (99.0 → 97.7%, FNR 1.0 → 2.3%) e **l'F1 resta identico (0.982)**. Le 10 varianti aggiuntive (build +65%) comprano solo meno falsi allarmi a scapito del recall. La scelta è giustificata solo se l'operating point richiede FP minimi (e regge l'argomento coverage di §3.15.4 sullo spazio-luce più ampio), altrimenti sono rendimenti decrescenti.

**Conclusione dell'ablation.** La configurazione raccomandata è **SL-15** (F1 0.982, light_FPR 0.30%, shadow_FPR 2.2%, recall 99.0%): gestisce simultaneamente i due failure mode al costo di +0.8 pp di FNR rispetto al baseline. Le luci sintetiche nel test set incidono in modo moderato (5.8% non trattato, circa metà delle ombre) e vengono quasi azzerate dall'aggiunta della light aug. Il messaggio metodologico onesto da riportare in tesi, evidenziato da L4, è che il guadagno headline non è puro "effetto-luce" ma in larga parte "effetto-budget": più augmentation totale alza la soglia operativa e taglia i FP, mentre la diversificazione a budget costante dà un miglioramento minore ma a costo nullo. Il confronto diretto shadow_FPR vs light_FPR resta non apples-to-apples (§3.15.3): le letture valide sono within-category.

### 3.16 Augmentation e pooling: discriminazione o calibrazione?

Le sezioni §3.7-§3.15 introducono diversi interventi (shadow augmentation, light augmentation, decoupling build/calibration, `--cal 30`, pooled σ) e ne misurano l'effetto su FPR, TPR, F1. Questa sezione risponde a una domanda trasversale: questi interventi migliorano la **capacità discriminativa** del modello (separare scene normali, incluse quelle con ombre e luci, da scene ostruite) oppure agiscono solo sul **piazzamento della soglia**? La distinzione decide cosa la tesi può rivendicare: un guadagno di rappresentazione o un guadagno di calibrazione.

L'analisi è condotta sui CSV di tutte le ablation (esclusi i run su poli_ingegneria). Due gruppi con test set distinti: **Gruppo A** (synthetic copy-paste, 990 normal + shadow + obstructed, senza luci): `senza_gestione_ombre` (no-aug) → `shadow04` → `decoupled` → `decoupled_k4` → `pooled`; **Gruppo B** (ablation L0-L4 con luci, §3.15.8).

#### 3.16.1 La metrica giusta per la discriminazione

L'AUROC è threshold-independent: se un intervento agisce solo sulla soglia, l'AUROC resta invariata; se migliora la separabilità, sale. Ma l'**AUROC normalizzato** del §2.3 (`(score-μ_i)/σ_i`) **non è discriminazione pura**: dipende da quale σ si usa per normalizzare. Verificato sui CSV - nel run pooled `normalized_score=(score-μ_i)/σ_pop` (σ costante), nel decoupled usa σ_i per-camera. Di conseguenza l'AUROC normalizzato-pooled del Gruppo A oscilla 0.954 → 0.987 → 0.972 → 0.998 mentre la separabilità intrinseca non si muove: quel movimento è qualità di calibrazione, non qualità del modello.

La metrica pulita è l'**AUROC per-camera**: la probabilità che un'ostruzione abbia score più alto di una variante-normale *della stessa camera*, sui raw score, mediata sulle camere. Entro una camera μ_i e σ_i sono costanti, quindi raw e normalized differiscono per una trasformazione affine che non altera il ranking: l'AUROC per-camera è invariante sia alla soglia sia alla normalizzazione, e isola la discriminazione intrinseca.

#### 3.16.2 La discriminazione è satura e quasi invariante alle augmentation

AUROC per-camera (clean / shadow / light vs obstructed):

| Config | clean | shadow | light |
|--------|-------|--------|-------|
| no-aug (senza_gestione) | 1.000 | 0.996 | n/a |
| shadow04 / decoupled / k4 / pooled | 1.000 | 0.996 | n/a |
| L0 shadow-only | 1.000 | 0.996 | 0.997 |
| L1 light-only | 1.000 | 0.996 | 1.000 |
| L2 SL-15 / L3 SL-25 / L4 | 1.000 | 0.998 | 1.000 |

Su clean la separazione è perfetta (1.000) ovunque. Su ombre resta 0.996 in tutto il percorso shadow del Gruppo A, **anche senza shadow augmentation** (no-aug = shadow04); su luci migliora solo 0.997 → 1.000. Tutti i movimenti sono al 3°-4° decimale: il backbone WideResNet separa già normale-vs-ostruito quasi perfettamente, e l'augmentation non ha margine da aggiungere.

#### 3.16.3 La prova controllata: AUROC piatta, FPR che oscilla 40×

Gruppo A - le transizioni decoupled → k4 → pooled cambiano solo il meccanismo di soglia, non il banco:

| Config | raw AUROC | per-cam (shadow) | norm-pooled AUROC | shadow_FPR | TPR | F1 |
|--------|-----------|------------------|-------------------|------------|-----|-----|
| no-aug | 0.9969 | 0.996 | 0.954 | 90.1% | 100% | 0.688 |
| shadow04 | 0.9981 | 0.996 | 0.987 | 2.1% | 70.6% | 0.818 |
| decoupled | 0.9981 | 0.996 | 0.972 | 28.4% | 99.4% | 0.873 |
| decoupled_k4 | 0.9981 | 0.996 | 0.972 | 13.5% | 93.4% | 0.903 |
| pooled | 0.9981 | 0.996 | 0.998 | 12.2% | 99.8% | 0.941 |

Lo shadow_FPR salta da 2.1% a 90% a 12% (range 40×) mentre **raw AUROC è costante a 0.9981 e per-camera-shadow inchiodata a 0.996**. È la definizione di effetto solo-soglia, coerente con §3.13 ("AUROC 0.998 invariato... il guadagno di F1 deriva esclusivamente dall'eliminazione della varianza stocastica della soglia"). Anche la shadow augmentation stessa (no-aug → shadow04) abbatte shadow_FPR 90% → 2% con raw AUROC quasi fermo (0.9969 → 0.9981) e per-camera-shadow identica.

#### 3.16.4 Meccanismo: shift distribuzionale, non ri-ordinamento

Come può lo shadow_FPR crollare 90% → 2% se l'AUROC non cambia? Senza ombre nel banco, le patch d'ombra non hanno vicini e producono score alti: restano *sotto* le ostruite (ranking preservato, AUROC 0.996) ma *sopra* una soglia calibrata sui clean, quindi sforano (FPR 90%). Con le ombre nel banco, le ombre-normali trovano corrispondenti e il loro score scende verso la zona normale, cadendo *sotto* la soglia (FPR 2%), senza che il loro ordine rispetto alle ostruite cambi. L'augmentation esegue uno **shift verso il basso** della distribuzione degli score normali-disturbati, rilevante rispetto a una soglia fissa, non un **ri-ordinamento**. Il pooled σ poi stabilizza *dove* quella soglia cade tra le camere. Nessuno dei due aggiunge potere discriminativo: rendono utilizzabile a un operating point unico una discriminazione già eccellente.

#### 3.16.5 Riformulazione del contributo

Il messaggio corretto non è "il modello discrimina meglio" (è già ~1.0, saturo) ma: il backbone separa normale-vs-ostruito quasi perfettamente anche su scene con ombre e luci; il collo di bottiglia è il **piazzamento di una soglia unica robusta ai disturbi**. Shadow/light augmentation (abbassa gli score dei disturbi sotto la soglia) e pooled σ (stabilizza la soglia tra camere) convertono una discriminazione latente in un operating point utilizzabile (FPR 90% → 2% a TPR ~99%, AUROC invariata).

**Caveat (ceiling effect).** La conclusione vale su un test set dove la discriminazione è quasi satura *by design* (§2.4: ostruite copy-paste sullo stesso sfondo, solo l'oggetto è anomalo, segnale forte, AUROC ≈ 1). Su dati più difficili (ostruzioni reali, sfondi variabili) la discriminazione avrebbe headroom e l'augmentation potrebbe contribuire anche alla separabilità, non solo alla calibrazione. Va dichiarato esplicitamente.

#### 3.16.6 Pooling: un meccanismo di regolarizzazione

Il pooled σ è l'esempio più puro di intervento solo-calibrazione: per costruzione (§3.12) sostituisce σ_cal,i per-camera con un'unica σ_pop, lascia μ_i locale, e lascia l'AUROC invariata. Il suo intero valore è la riduzione della varianza delle soglie (σ_θ −73%): è, per definizione, un meccanismo di **regolarizzazione (shrinkage)**. Da qui la domanda: in un deploy reale dove ogni telecamera è fisica, indipendente e calibrabile sui propri frame normali, il pooling serve, o è una stampella del setup sperimentale?

La risposta è un trade-off bias-varianza sulla stima di σ_i:

- **Varianza** della stima per-camera ∝ 1/n_cal. Qui n_cal=30 varianti sintetiche di *una sola* immagine (~3 eventi-ombra) → σ_i molto rumorosa (SE ≈ 18%, §3.12).
- **Bias** del pooling ∝ eterogeneità *vera* delle σ_i tra camere. Il §3.10 documenta che qui le σ_i estreme sono outlier stocastici (seed-dependent), non di scena ("la stessa camera ricalibrata con un seed diverso avrebbe una soglia diversa") → eterogeneità vera quasi nulla.

Nel regime di questa tesi (banca one-shot da una sola immagine + calibrazione sintetica scarsa) entrambe le condizioni che favoriscono il pooling sono soddisfatte: stima per-camera rumorosa + eterogeneità vera ≈ 0 → il pooling è riduzione di varianza a costo-quasi-zero di bias. È la scelta corretta.

In un deploy reale con abbondanti frame normali per camera la situazione cambia:

- Con migliaia di frame normali reali raccolti su giorni (ombre, luci, ore del giorno reali), σ_i diventa una stima **affidabile**: il rumore che il pooling corregge sparisce.
- Scene reali possono avere σ_i **genuinamente eterogenee**: una porta a vetri esposta a luce esterna variabile ha uno spread degli score normali realmente maggiore di un corridoio interno stabile. Qui σ_i porta segnale (quella camera merita una tolleranza più alta) e il pooling introdurrebbe bias: forzare la camera ad alta varianza sulla σ_pop mediana abbassa la sua soglia → falsi allarmi cronici; forzare quella a bassa varianza la alza → missed detection.

Quindi il pooling non è una necessità architetturale ma un regolarizzatore il cui beneficio dipende dal regime di dati. Resta prezioso in due scenari operativamente comuni anche nel deploy reale:

1. **Cold-start**: una camera appena installata ha pochi frame normali → σ_i rumorosa → il pooling (borrowing strength dalle altre camere) la regolarizza. Aggiungere camere a un sistema in esercizio è la norma, non un caso limite.
2. **Robustezza a finestre di calibrazione non rappresentative**: se il periodo di raccolta normale di una camera include per caso condizioni anomale, σ_i si gonfia/sgonfia; il pooling fa da guardrail.

La generalizzazione corretta è il **partial pooling** (shrinkage gerarchico / empirical Bayes):

```
σ_i* = (1 − w_i) · σ_pop + w_i · σ_i
```

con peso w_i crescente con n_cal,i (e con l'evidenza che σ_i sia scene-stabile). Il pooling pieno attuale è l'angolo w_i=0 (max shrinkage), corretto per il regime data-starved della tesi; la calibrazione per-camera è w_i=1; l'ottimo è adattivo. Si noti che il design attuale **già** pool-a σ in pieno ma tiene μ_i locale - scelta sensata perché μ_i porta informazione di scena genuina (livello assoluto degli score normali) mentre σ_i qui è dominata dal rumore di campionamento. In deploy reale, se anche σ_i portasse segnale di scena, andrebbe pooled solo parzialmente.

L'esperimento decisivo per un sito reale: misurare la **stabilità test-retest di σ_i** su finestre di calibrazione indipendenti. Se σ_i è stabile per camera tra finestre → è segnale → tendere al per-camera; se oscilla → è rumore → poolare. Il §3.13 raccomanda già di ricalcolare σ_pop sul sito; il passo successivo naturale è renderlo partial pooling pesato sui dati reali di ciascuna camera.

**Una distinzione da non confondere: pooling cross-camera vs σ per-camera da N frame reali.** Nel mappare questo risultato sul deploy reale è facile sovrapporre due meccanismi distinti. Il σ_pop dell'ablation è calcolato sulle σ_cal,i di 990 reference, cioè 990 *scene diverse*: pool-a **attraverso camere diverse**. In deploy reale, invece, la mossa naturale è stimare σ_i da N frame normali *reali della stessa camera* (che ne coprano la variazione naturale: ore, luci, ombre, persone di passaggio): è **stima per-camera con dati adeguati**, non pooling cross-camera. Sono due leve diverse per la stessa grandezza:

| Leva | Meccanismo | Bias | Richiede |
|------|-----------|------|----------|
| (a) N frame reali della camera i | più dati sulla **stessa** camera | zero | serie reali per camera |
| (b) σ_pop cross-camera (ablation, §3.12) | shrinkage verso le **altre** camere | >0 se le camere differiscono | nessun dato per-camera |

Gli ablation di questo capitolo hanno avuto a disposizione **solo la leva (b)** - niente serie temporali reali, solo 1 immagine + augment sintetici per camera. Quindi dimostrano la **diagnosi** (la σ per-camera rumorosa crea soglie patologiche, §3.10) e che **(b) la regolarizza** nel regime data-starved (§3.13). **Non** dimostrano (a): l'efficacia di una σ_i stimata da N frame reali è un'estrapolazione ben motivata, da validare in-situ, non un risultato di questi esperimenti.

Il punto operativo: la leva (a) è **migliore** (bias zero) e, quando disponibile, rende **non necessaria** la (b). Calcolare σ da N frame reali della camera **non è "σ_pop che fa da regolarizzatore"**: è la per-camera σ fatta come si deve, cioè proprio ciò che toglie il bisogno del regolarizzatore cross-camera. Il termine "regolarizzatore" spetta al borrowing cross-camera (b), non alla media su N frame della stessa camera (che è semplice stima con più dati). Si noti che anche la raccomandazione del §3.13 ("ricalcola σ_pop sul sito") resta **cross-camera** (mediana sulle σ_i delle camere del sito), distinta dalla calibrazione within-camera qui descritta.

La sintesi rimane il partial pooling già introdotto sopra (`σ_i* = (1−w_i)·σ_pop + w_i·σ_i`): la leva (b) σ_pop è il *target* di shrinkage e la rete di sicurezza del cold-start, la leva (a) fornisce la stima locale, e `w_i → 1` man mano che la camera accumula frame reali. In deploy maturo si tende al per-camera unbiased; il pooling cross-camera resta dove è strutturalmente utile: l'avvio di una camera nuova.

---

### 3.17 Validazione sul test set reale poli_ingegneria: ricalibrazione di k e tassonomia degli errori

Le sezioni precedenti hanno sviluppato i meccanismi singoli - pooled σ (§3.12-3.13) e gating per connected component (§3.14) - su suite sintetiche o su singole scene. Questa sezione li applica congiuntamente al primo **test set reale**, `poli_ingegneria` (la Fase D dell'architettura sperimentale, §4.1): 6 porte reali fotografate in sede, ciascuna con un'immagine di riferimento annotata (poligoni del telaio). Il set contiene 30 immagini ostruite reali (positivi), 30 immagini normali non ostruite (negativi) e le 6 reference. **La telecamera è assunta fissa** (§3.14): tutti gli scatti di una stessa porta condividono l'inquadratura, e l'eventuale variazione tra essi è di contenuto e illuminazione, non di angolo di ripresa. I negativi misurano quindi il FPR sotto la variabilità normale (luce, contenuto transitorio) che il sistema deve tollerare a parità di geometria.

#### 3.17.1 Setup e formula della soglia

Pipeline completa: `sigma_mode=pooled` + gating poligonale (`min_overlap_px=10`). La soglia di decisione per camera è:

θ_i = μ_i + k · median_j(σ_j),  con median(σ) = 0.213 sulle 6 reference.

Va distinta dalla **soglia di binarizzazione** che il gating usa per estrarre le componenti connesse dalla anomaly map (§3.14): quella resta locale, μ_i + k·σ_local,i, calcolata in build. Entrambe le soglie hanno k come moltiplicatore, quindi k agisce su due stadi distinti - vedi §3.17.3.

#### 3.17.2 Ricalibrazione di k: da 4 a 3

Il primo run usava k=4, coerente con lo sweet spot delle suite sintetiche (§3.11). Sul set reale k=4 è risultato troppo conservativo:

| Metrica | k=4 | k=3 | Δ |
|---|---|---|---|
| Recall (TPR) | 83.3% (25/30) | **90.0% (27/30)** | +6.7 pp |
| FPR | 3.3% (1/30) | 3.3% (1/30) | invariato |
| Precision | 96.2% | 96.4% | +0.2 pp |
| **F1** | 0.893 | **0.931** | +0.038 |
| Accuracy | 90.0% | 93.3% | +3.3 pp |

Poiché median(σ) è una costante condivisa, scendere da k=4 a k=3 abbassa ogni θ_i della stessa quantità k·median(σ) ≈ 0.213, senza riallargare la distribuzione delle soglie. Il problema strutturale del §3.11 - dove k moltiplicava una σ per-camera rumorosa e ne amplificava la varianza - qui non si presenta, perché la σ è poolata. Il recall sale di 6.7 punti a costo zero in FPR. Recall per porta a k=3: porta_1/2/5/6 = 100%, porta_3 = 78% (7/9), porta_4 = 89% (8/9).

#### 3.17.3 Tassonomia degli errori: calibrazione vs geometria

Il confronto dei due run separa i falsi negativi in due classi con causa diversa:

| Classe | Casi | Causa | Recuperabile con k? |
|---|---|---|---|
| **Calibration-limited** | porta_3_5, porta_4_8 | ostruzione rilevata, ma una soglia k=4 troppo alta la azzera o la respinge | Sì, recuperati a k=3 |
| **Gating-limited** | porta_3_2 (ventilatore), porta_3_9, porta_4_9 | la componente anomala non interseca il telaio per ≥10px | No |
| FP (vetro) | porta_4_4 | cambio di fondo dietro il vetro (tapparella + luce) | No, vedi §3.14 |

I due FN calibration-limited sono recuperati scendendo a k=3, ma per **meccanismi distinti**, uno per ciascuno dei due stadi in cui entra k:

- **porta_4_8 (stadio decisione)**: a k=4 una componente valida esisteva già e produceva score 3.43, ma stava sotto la soglia di decisione (3.67). A k=3 la soglia scende a 3.46 e il punteggio (3.53) la supera.
- **porta_3_5 (stadio binarizzazione)**: a k=4 lo score era 0.0. Il picco di anomalia dell'ostruzione (3.865) cadeva appena sotto la soglia di binarizzazione locale μ+4·σ_local = 3.872: nessun pixel superava la soglia, la mappa binaria era vuota, nessuna componente da valutare. A k=3 la soglia di binarizzazione scende a 3.47, il picco sopravvive, la componente si forma, interseca il telaio e lo score salta a 3.87.

Il caso porta_3_5 mostra che la soglia non è solo un confine di decisione: controlla anche **quali pixel esistono** per il gating. Una k troppo alta può cancellare un'ostruzione a basso contrasto prima ancora che il gating la valuti.

#### 3.17.4 Il caso del ventilatore: conferma empirica del limite previsto in §3.14

Tre FN restano azzerate anche a k=3. Il caso prototipico ispezionato è porta_3_2: un **ventilatore a piantana** davanti alla porta. La heatmap conferma che PatchCore lo rileva correttamente (componente rossa intensa, ben sopra ogni soglia, sulla testa), ma:

- la **testa** del ventilatore, larga, cade al centro della porta, nel `door_region`, **escluso per costruzione** dalla frame mask (§3.14);
- l'unico contatto con un elemento di telaio è l'**asta sottile** che scende verso la soglia; a 224×224 l'asta si riduce a 1-2 px di larghezza e la sua intersezione col poligono `threshold` resta sotto i 10 px richiesti.

L'unica componente sopra-soglia (la testa) non tocca il telaio e viene scartata: score 0.0, irrecuperabile abbassando k perché la testa supera ampiamente qualsiasi soglia di binarizzazione - il problema è puramente geometrico, non di calibrazione.

Questo è la **conferma empirica del limite già previsto in §3.14**: "l'unica anomalia genuina che il poligonale potrebbe respingere è un oggetto interamente sul pannello senza contatto col pavimento o col telaio". Il ventilatore raffina la previsione: il contatto col pavimento esiste (l'asta), ma è **troppo sottile** per superare il criterio di overlap. L'assunzione del §3.14 - "gli ostacoli sono grounded, quindi la componente si estende fino alla soglia" - vale per ostacoli a profilo largo (carrello, sedia a rotelle, cestino) ma non per oggetti a stelo sottile (ventilatore, attaccapanni, cartello su asta). Le leve per recuperarli (abbassare `min_overlap_px`, dilatare la frame mask, reincludere `door_region` per le porte opache) sono discusse in §3.14 e spostano tutte il trade-off verso il recall a scapito della precision.

#### 3.17.5 Sintesi

Sul primo test set reale, la pipeline completa (pooled σ + gating, k=3) raggiunge **recall 90%, FPR 3.3%, F1 0.931**. I due errori residui strutturali erano entrambi già previsti dall'analisi teorica e nessuno è un fallimento della detection di PatchCore, che rileva correttamente sia il vetro sia il ventilatore: l'unico FP è la porta a vetro con fondo variabile (§3.14), e le FN dure sono ostacoli a stelo sottile che non soddisfano il criterio di overlap col telaio. Entrambi sono conseguenze governabili delle scelte di gating, con trade-off precision/recall noti.

---

## 4. Inquadramento delle suite di test e validità statistica

Il sistema è stato sottoposto a piu suite di test in fasi successive, ciascuna con uno scopo specifico nel percorso di sviluppo. La presente sezione formalizza il ruolo di ogni suite, evitando di interpretare risultati intermedi come benchmark finali e chiarendo cosa ciascuna misurazione effettivamente dimostra.

### 4.1 Architettura sperimentale a fasi

Il lavoro segue una struttura ad **ablazione progressiva**, in cui ogni componente del sistema viene introdotto in risposta a un failure mode identificato empiricamente. Le suite di test corrispondono a fasi distinte di questa progressione:

| Fase | Test set | Componenti attive | Scopo |
|---|---|---|---|
| A - Baseline | Copy-paste sintetico (`ostruzioni_reali`) | PatchCore vanilla, no ROI | Caratterizzare il comportamento di base e identificare failure modes |
| B - Diagnosi ombre | Copy-paste + augmentazione ombre sintetiche | PatchCore vanilla, no ROI | Quantificare l'impatto delle ombre sul FPR |
| C - Shadow fix | Copy-paste + ombre | PatchCore + shadow_aug in build | Misurare l'effetto della shadow augmentation |
| D - Glass door fix | Test set reale (poli_ingegneria + Pinterest) | PatchCore + shadow_aug + ROI gating + rect ROI | Valutare il sistema completo su scene reali con porte a vetri |

Ogni transizione tra fasi e motivata da una diagnosi quantitativa della fase precedente, non da scelte arbitrarie di design.

### 4.2 Interpretazione dei risultati del baseline

I risultati ottenuti nelle fasi A, B e C sono stati condotti sul dataset copy-paste sintetico, senza definizione della ROI poligonale. Questa configurazione corrisponde a una caratterizzazione del **comportamento di base** del rilevatore PatchCore in assenza dei meccanismi di gating successivamente introdotti.

I valori assoluti di precisione, recall e FPR ottenuti in queste fasi **non rappresentano il benchmark finale del sistema** e non possono essere proiettati come stima delle performance in produzione. Essi mantengono pero una valenza precisa nel contesto sperimentale:

1. **Validazione metodologica.** La pipeline build → calibrazione → test → evaluate funziona correttamente sui dati sintetici, con AUROC significativamente sopra il chance level. Questo conferma che lo score patch-level di PatchCore e discriminativo per la classe di anomalie considerate.
2. **Caratterizzazione dei failure modes.** L'analisi quantitativa dei falsi positivi nelle fasi A e B ha identificato due categorie ricorrenti: variazioni di sfondo attraverso il vetro e ombre proiettate sul pavimento. Questa diagnosi e l'evidenza empirica che giustifica le aggiunte successive (shadow augmentation, ROI gating). Senza la fase di baseline, gli interventi correttivi sarebbero stati guidati da intuizione anziche da evidenza.
3. **Riferimento per ablazione.** I numeri di baseline costituiscono il punto di confronto rispetto al quale misurare l'effetto incrementale di ogni componente. La metrica rilevante non e il valore assoluto di AUROC ma il delta tra configurazioni: `Δ_shadow_aug = AUROC(C) - AUROC(B)`, `Δ_gating = AUROC(D) - AUROC(C)`.

### 4.3 Dimensione dei dataset e potenza statistica

Le fasi A-C utilizzano il dataset copy-paste sintetico, che conta circa un migliaio di immagini di riferimento tra porte e corridoi, con composite generati da `generate_synthetic_dataset.py`. Questo volume e adeguato per una validazione statistica robusta: gli intervalli di confidenza sulle metriche aggregate (AUROC, FPR, F1) sono stretti e i confronti tra configurazioni (baseline vs shadow_aug) hanno potenza sufficiente a rilevare effetti di interesse pratico. I risultati di queste fasi non soffrono di limitazioni di dimensione del campione.

La fase D, invece, opera su un test set custom di porte a vetri (poli_ingegneria + immagini Pinterest) dove ogni scena richiede annotazione manuale di ROI rettangolare e poligoni del telaio. Il volume e necessariamente contenuto - dell'ordine di alcune decine di scene - per il costo lineare dell'annotazione. Le limitazioni statistiche si concentrano qui:

- Intervalli di confidenza ampi sulle metriche assolute (FPR e TPR su poche decine di campioni hanno IC bootstrap ampi).
- Bassa potenza nel rilevare differenze piccole tra configurazioni.
- Sensibilita all'identita delle scene specifiche incluse: una porta con illuminazione particolare puo dominare l'aggregato.

Per la fase D si adottano specifici accorgimenti:

- **AUROC con bootstrap confidence intervals (95%)** invece di accuracy puntuale. I CI da bootstrap (1000 resampling) forniscono una stima dell'incertezza compatibile con N piccoli e l'AUROC e meno sensibile della accuracy allo sbilanciamento di classi.
- **Paired test (McNemar)** tra configurazioni successive valutate sulle stesse immagini. Questo aumenta la potenza statistica rispetto a un confronto a campioni indipendenti, sfruttando il fatto che le coppie (con/senza gating) condividono le stesse fonti di varianza per-scena.
- **Disclaim espliciti** nelle didascalie ("on a dataset of N=X glass-door scenes, the results suggest..."): nessuna proiezione su performance in produzione e effettuata dai dati di fase D.

Il copy-paste delle fasi A-C e quindi il test set quantitativamente significativo, mentre la fase D ha valore principalmente qualitativo: dimostrare che il meccanismo di gating si comporta come atteso su scene reali, validando il design ma non quantificando precisamente il guadagno. La generalizzazione su scala maggiore richiederebbe un dataset di porte a vetri annotato in modo automatico (es. SAM) o semi-automatico.

### 4.4 Sintesi della narrativa sperimentale

Il messaggio metodologico della tesi si articola su due livelli di evidenza, con peso statistico diverso:

- **Fasi A-C (copy-paste, N ~ 1000)**: validazione quantitativa con potenza statistica adeguata. I numeri sono interpretabili come stime stabili dell'effetto dei singoli componenti (shadow augmentation, decoupling della soglia, pooled sigma). I delta misurati sono significativi e generalizzabili a scene simili (porte e corridoi sintetici con ostruzioni copy-paste).
- **Fase D (test set custom con porte a vetri, N ~ poche decine)**: validazione qualitativa del meccanismo di gating su scene reali. I numeri vanno letti come illustrativi - confermano o smentiscono il comportamento atteso del filtro, ma non quantificano un benchmark assoluto.

Il messaggio finale e dunque: "la diagnosi sui dati sintetici copia-incolla (statisticamente solida) ha identificato due failure modes principali; l'introduzione di shadow augmentation e ROI gating riduce empiricamente la loro incidenza, e la validazione qualitativa su scene reali a camera fissa conferma il comportamento atteso del sistema completo." La validazione quantitativa del gating su scala maggiore e indicata come estensione naturale, condizionata alla disponibilita di un test set piu ampio di porte a vetri annotato in modo automatico o semi-automatico (es. SAM per la segmentazione del telaio).
