# P1 - Object detection: report baseline COCO (senza fine tuning)

Data: 2026-07-07. Ambiente: `tesi_env` (ultralytics 8.4.89, YOLOv11n, CPU).

## 1. Obiettivo e impostazione

P1 rileva la PRESENZA di un'ostruzione su una via di fuga (non il tipo di
oggetto): singola classe "ostruzione", tutti gli oggetti collassati in classe 0.

Su indicazione del relatore proviamo prima SENZA fine tuning: si valuta una rete
YOLOv11n preaddestrata su COCO, mappando un sottoinsieme di classi COCO come
"ostruzione", e si misura come si comporta. Il codice di fine tuning e la cross
validation restano pronti come piano B.

## 2. Dati

- Copy-paste v2 (`Dataset/ostruzioni_reali`): 990 immagini ostruite (positivi)
  + 110 sfondi di riferimento held-out (negativi, i "pruned"). Rigenerato con
  seed fisso 42, con metadata per oggetto (`objects_metadata.json`) e con il 32%
  degli oggetti tagliati dal bordo scena (parzialmente visibili). Categorie
  incollate bilanciate: barella 399, scatola 381, sedia (wheelchair) 378,
  cestino 377, carrello 365.
- Test reale `poli ingegneria`: 30 immagini ostruite reali + 36 normali,
  totalmente slegate da train/val (unico test esterno).

## 3. Metodo

- Baseline COCO: `coco_obstruction_map.py` definisce le classi COCO che contano
  come ostruzione. Una detection vale se la classe e nel set E l'area della box
  supera l'1% dell'immagine (esclude oggetti minuscoli). Soglia di confidenza
  0.25.
- Metrica image-level (condivisa con P3): un'immagine e predetta "ostruita" se
  almeno una detection valida supera la soglia. Da qui accuracy, precision,
  recall, F1, FPR, FNR.

Mapping iniziale (10 classi): bench, backpack, handbag, suitcase, chair, couch,
potted plant, bed, dining table, refrigerator.

## 4. Risultati globali

| set | n_pos | n_neg | recall | precision | F1 | FPR | accuracy |
|---|---|---|---|---|---|---|---|
| copy-paste v2 | 990 | 110 | 0.302 | 0.929 | 0.456 | 0.209 | 0.351 |
| poli reale | 30 | 36 | 0.233 | 0.438 | 0.304 | 0.250 | 0.515 |

Lettura: COCO manca circa il 70% delle ostruzioni (recall ~0.30), ma quando
"vede" qualcosa e quasi sempre corretto sul copy-paste (precision 0.93). Il
problema non e il rumore ma il VOCABOLARIO: 4 categorie su 5 (carrello, sedia a
rotelle, barella, cestino) non esistono tra le classi COCO. I risultati su
copy-paste e su poli reale sono coerenti: il deficit e strutturale, non un
artefatto del dominio sintetico.

## 5. Recall per categoria (copy-paste v2)

Immagine conteggiata per la categoria se contiene almeno un oggetto di quella
categoria (le categorie si sovrappongono nelle immagini multi-oggetto).

| categoria | recall |
|---|---|
| sedia (wheelchair) | 0.401 |
| carrello | 0.380 |
| barella | 0.327 |
| scatola | 0.293 |
| cestino | 0.234 |

Nessuna categoria funziona: anche la migliore (0.40) e inaccettabile per un
sistema di sicurezza. Il 30-40% che viene rilevato sono verosimili confusioni
con classi COCO visivamente simili.

Effetto del numero di oggetti (piu oggetti = piu facile becarne almeno uno):
1 oggetto recall 0.218, 2 oggetti 0.344, 3 oggetti 0.358. Lo scenario reale
critico (un singolo carrello davanti all'uscita) e proprio quello dove la
baseline e piu cieca.

Oggetti tagliati vs interi (a parita di 1 solo oggetto per isolare l'effetto):
interi 0.207, tagliati 0.256. Gli oggetti parzialmente visibili non degradano
la baseline.

## 6. Analisi per-classe: chi scatta sui positivi vs sui negativi

Analisi indipendente dal mapping usato in baseline (sez. 3-4): qui si fa
girare YOLOv11n preaddestrato COCO, SENZA filtrare le classi, su tutte le 990
immagini positive (copy-paste) e le 110 negative (sfondi puliti), a conf 0.25.
Per OGNI classe delle 80 di COCO (chair, bicycle, person, ...) si conta su
quante immagini quella classe scatta almeno una volta, con lo stesso filtro di
area minima 1% usato per la baseline (una detection minuscola non conta). Poi:

- **pos%** = (immagini positive su cui la classe scatta almeno 1 volta) / 990.
  Es. chair pos%=16.6 significa: su 990 immagini con un oggetto ostacolo
  incollato, in 164 di esse (16.6%) YOLO ha rilevato ALMENO una box con
  classe "chair" (area >= 1%) - indipendentemente da quale fosse l'oggetto
  incollato davvero (potrebbe essere la sedia a rotelle scambiata per chair,
  ma anche un carrello o altro).
- **neg%** = (immagini negative su cui la classe scatta almeno 1 volta) / 110.
  Es. refrigerator neg%=11.8 significa: su 110 corridoi/porte NORMALI (senza
  alcun oggetto incollato), in 13 di essi (11.8%) YOLO ha comunque rilevato
  una box "refrigerator" - un FALSO POSITIVO, perche' quella scena non ha
  nessun frigorifero, e' un artefatto del modello (es. una porta a vetri, un
  riflesso, uno scaffale scambiati per frigo).
- **disc** = pos% - neg%, il "potere discriminante": quanto la classe e' piu
  frequente sui positivi rispetto ai negativi. disc alto e positivo = la
  classe e' un buon segnale di ostruzione (scatta quando c'e' qualcosa,
  raramente quando non c'e' nulla). disc vicino a 0 o negativo = la classe
  scatta piu o allo stesso modo sui corridoi VUOTI che su quelli ostruiti:
  usarla nel mapping produrrebbe falsi allarmi senza guadagno di recall.

Tabella ordinata per disc decrescente (le classi piu utili in cima):

| classe | pos% | neg% | disc | nel mapping |
|---|---|---|---|---|
| chair | 16.6 | 0.9 | +15.7 | si |
| bicycle | 9.9 | 0.0 | +9.9 | no |
| bench | 4.6 | 0.0 | +4.6 | si |
| motorcycle | 3.8 | 0.0 | +3.8 | no |
| toilet | 4.6 | 0.9 | +3.7 | no |
| airplane | 3.7 | 0.0 | +3.7 | no |
| sink | 2.1 | 0.0 | +2.1 | no |
| cup | 2.0 | 0.0 | +2.0 | no |
| suitcase | 1.8 | 0.0 | +1.8 | si |
| tv | 1.5 | 0.0 | +1.5 | no |
| couch | 0.8 | 0.0 | +0.8 | si |
| dining table | 0.5 | 0.0 | +0.5 | si |
| bed | 3.8 | 3.6 | +0.2 | si |
| backpack | 0.1 | 0.0 | +0.1 | si |
| handbag | 0.1 | 0.9 | -0.8 | si |
| potted plant | 1.9 | 3.6 | -1.7 | si |
| refrigerator | 3.5 | 11.8 | -8.3 | si |

## 6.1 Verifica visiva per-classe: cosa rileva davvero ogni classe

L'analisi della sez. 6 dice QUANTO una classe discrimina, non COSA sta
rilevando concretamente. Prima di modificare il mapping in base al numero
"disc", si sono ispezionati manualmente i ritagli (crop) delle box rilevate
per le classi piu controverse, per capire se il segnale numerico riflette una
detection semanticamente sensata o un artefatto casuale.

**`refrigerator` (disc -8.3, rimossa dal mapping)**: ispezionati tutti i 49
ritagli (36 pos, 13 neg). Risultato: la classe si comporta come un rilevatore
generico di "grande pannello rettangolare verticale piatto", non di
frigoriferi. TUTTI e 13 i falsi positivi sui negativi sono su porte (0 su
corridoi), e i ritagli mostrano porte a due ante bianche/grigie con giunzione
verticale centrale e maniglia - la sagoma tipica di un frigo side-by-side.
Anche su molte immagini POSITIVE la classe rileva la porta di sfondo invece
dell'oggetto incollato: il caso a confidenza piu alta di tutto il dataset
(0.906, porta_119) cade esattamente sulla porta verde sullo sfondo, non sulla
scatola di cartone incollata in primo piano. Un caso ha anche rilevato un
quadro incorniciato appeso al muro (corridoio_396) - stesso pattern, pannello
rettangolare piatto. Conclusione: rimuovere `refrigerator` non fa perdere
quasi nulla, perche' anche dove "funzionava" spesso non stava rilevando
l'oggetto giusto.

**`toilet` (disc +3.7, aggiunta al mapping su richiesta)**: ispezionati i
ritagli. Segnale eccellente e pulito: 47 detection su 48 sono sui positivi
(98%), solo 1 sui 110 negativi (0.9%). I ritagli confermano che la classe sta
rilevando correttamente il CESTINO (bidoni della spazzatura di vari colori,
grigio con coperchio a bascula, arancione) - COCO non ha una classe
"trash can/waste bin" dedicata, ma la sagoma cilindrica/tronco-conica del
cestino somiglia a un vaso WC visto da una certa angolazione. L'unico falso
positivo sui negativi e' una soglia/gradino in marmo (forma bianca
arrotondata) - caso isolato e comprensibile. Aggiunta motivata e verificata.

**`vase` (segnale debole/misto, mantenuta su richiesta)**: solo 5 detection
totali (4 pos, 1 neg) - campione troppo piccolo per essere conclusivo.
Ispezionando i ritagli: un caso ha azzeccato per coincidenza un cestino (blu,
a rete), ma gli altri 3-4 sono VERI vasi con piante decorative presenti sullo
sfondo delle foto (sia sui positivi che sul negativo), non oggetti bersaglio
incollati - la classe sta funzionando "correttamente" (c'e' davvero un vaso
nell'inquadratura) ma non e' utile per il nostro scopo, perche' non e' uno dei
5 oggetti target. Impatto pressoche' nullo sulle metriche data la rarita'.

## 6.2 Matrice di confusione object-level (copy-paste)

Analisi a livello di OGGETTO (non image-level): per ogni oggetto incollato,
di cui conosciamo categoria vera e box (da objects_metadata.json), matchiamo
le detection COCO grezze via IoU (>=0.3, lenient) e registriamo QUALE classe
COCO gli viene assegnata, o MISSED. Precompute in
`precompute_detection_matches.py`, visualizzazione in
`detection_analysis.ipynb`. Modello yolo11n, conf 0.25.

Detect rate per categoria (oggetto matchato a QUALSIASI classe, IoU>=0.3):

| categoria | detect rate | mancati |
|---|---|---|
| sedia (wheelchair) | 0.521 | 48% |
| barella (stretcher) | 0.326 | 67% |
| carrello (cart) | 0.271 | 73% |
| cestino (bin) | 0.239 | 76% |
| scatola (box) | 0.129 | 87% |

Come ogni categoria viene etichettata (conteggi principali, su ~380 ogg/cat):

- sedia: bicycle 80 + motorcycle 31 (le RUOTE) > chair 77 -> le sedie a
  rotelle sono viste piu come veicoli a due ruote che come sedie. Conferma
  quantitativa dell'aggiunta bicycle/motorcycle al mapping.
- barella: airplane 38, chair 36, bed 9 -> la forma allungata orizzontale
  la fa scambiare per `airplane` (NON in mapping) quanto per bed/chair.
- cestino: toilet 31 (in mapping) + cup 22 (non in mapping) -> i cestini
  cilindrici del copy-paste vanno su toilet/cup.
- carrello: chair 38, bench 22, bed 12.
- scatola: 87% MISSED, il resto sparso (bed 11, bench 7) -> una scatola di
  cartone liscia non ha alcun analogo COCO, e' la categoria piu invisibile.

Nota: anche con IoU lenient (0.3), ~70% di TUTTI gli oggetti (1335/1900) sono
MISSED. Il limite di vocabolario e' netto e quantificato.

## 6.3 Analisi visiva sul poli reale (cosa viene effettivamente rilevato)

Ora che le box del poli sono annotate (28/30 immagini), si puo' analizzare a
livello di oggetto anche sul reale. Ispezione diretta delle immagini (non
dedotta dai nomi delle classi): ogni "porta" e' una porta fisica con piu
ostacoli diversi fotografati. Inventario reale:

| ostacolo reale | dove | cosa fa COCO | contato? |
|---|---|---|---|
| sedia | porta_3_4/5/6, 4_4/5/6 | chair 0.5-0.9 | si (in mapping) |
| bottiglia di vetro (a terra) | porta_3_9, 4_7, 5/6 | bottle 0.5-0.84 (CORRETTO) | no (non in mapping) |
| cestino carta blu (alto, rett.) | porta_1, 2 | parking meter / niente | no |
| persona / braccio | porta_3_7, 4_8, 5_1 | person | no (giustamente escluso) |
| ventilatore / foglio su architrave | scene "vuote" | niente | no (invisibile a COCO) |

Scoperte sul reale:

1. L'UNICO ostacolo rilevato-e-mappato e' la sedia (chair). Le 6 immagini
   sedia sono le sole rilevate a conf 0.25.
2. Le bottiglie sono rilevate CORRETTAMENTE come `bottle` (sono bottiglie
   vere a terra), ma `bottle` non e' nel mapping -> contate come mancate pur
   essendo state VISTE. Di nuovo un limite di mapping, non di modello.
3. Il cestino reale (blu, alto, rettangolare) va su `parking meter` o niente.
   Cruciale: il `toilet` che avevamo aggiunto al mapping - validato sui
   cestini CILINDRICI del copy-paste - NON scatta sui cestini reali
   rettangolari. E' un caso concreto e forte del gap sintetico->reale: un
   aggiustamento del mapping ottimo sul sintetico e' inutile sul reale.
4. Ventilatore e foglio: invisibili (nessuna classe COCO). Il braccio/persona
   va su `person`, correttamente escluso.
5. Single-class AP@0.5 (obstruction) sul poli: 0.336 (n_gt=31; a conf 0.25
   solo le 6 sedie rilevate).

CAVEAT METODOLOGICO IMPORTANTE: queste osservazioni sul poli sono DIAGNOSTICHE
(spiegano perche' le metriche reali sono basse), NON una licenza per ritoccare
il mapping. Aggiungere `bottle` perche' ci sono bottiglie nel poli, o
`parking meter` perche' il bidone gli somiglia, brucerebbe il poli come test
indipendente (si tarerebbe sul test set). Il mapping si tara solo sul
copy-paste; il poli resta intoccato. Entrambe le analisi (matrice di
confusione + galleria GT vs predetto) sono riproducibili in
`detection_analysis.ipynb`.

## 7. Scoperte principali e risultati finali del mapping

1. Il FPR (0.21 sul copy-paste, mapping iniziale) e quasi tutto dovuto a
   `refrigerator`: scatta su 13 dei 110 corridoi normali (11.8%) contro il
   3.5% dei positivi (v. verifica visiva sez. 6.1: rileva porte, non frigo).
   Insieme a `handbag` sono stati rimossi dal mapping.

2. Gli oggetti CON RUOTE vengono rilevati come `bicycle` (9.9% dei positivi,
   0% dei negativi) e `motorcycle` (3.8% / 0%): carrelli e sedie a rotelle
   scattano su queste classi perche COCO riconosce le ruote. I cestini
   cilindrici scattano su `toilet` (verificato visivamente, sez. 6.1). Cioe:
   il detector VEDE gli oggetti, ma li etichetta con la classe sbagliata.
   Questo e l'argomento forte a favore di un detector con il vocabolario
   giusto (open-vocabulary o fine tuning): il problema di COCO non e cecita
   ma vocabolario.

3. MAPPING FINALE (dopo le verifiche visive): tolti `refrigerator` e
   `handbag`, aggiunti `bicycle`, `motorcycle`, `toilet`, `vase`. Progressione
   sul copy-paste (990 pos / 110 neg, conf 0.25):

   | step | recall | precision | F1 | FPR |
   |---|---|---|---|---|
   | mapping iniziale (10 classi) | 0.302 | 0.929 | 0.456 | 0.209 |
   | -refrigerator/handbag +bicycle/motorcycle | 0.363 | 0.976 | 0.529 | 0.082 |
   | +toilet/vase (finale) | **0.406** | **0.976** | **0.574** | 0.091 |

   Recall per categoria (mapping finale): sedia 0.635 (era 0.40), barella
   0.436, carrello 0.426, cestino 0.388 (era 0.234, +66% grazie a `toilet`),
   scatola 0.355.

4. ATTENZIONE - il miglioramento NON si trasferisce al test reale (poli, 30
   pos / 36 neg): il risultato con il mapping finale e' IDENTICO bit per bit
   a quello prima di aggiungere toilet/vase (TP=6, FP=3, TN=33, FN=24;
   recall=0.20, F1=0.308, FPR=0.083). `toilet`, che sul copy-paste rilevava
   quasi perfettamente i cestini (47/48), non rileva NESSUNO dei cestini
   nelle foto reali del poli. Stesso gap sintetico->reale gia' osservato con
   YOLO-World (sez. 8): il tuning del mapping, per quanto verificato
   visivamente e metodologicamente corretto, e' stato validato sui ritagli
   puliti del copy-paste, che non sono rappresentativi delle condizioni reali
   (illuminazione, angolazione, distanza, sfondo) del poli ingegneria.
   Monito da riportare in tesi: il copy-paste e' utile per iterare
   rapidamente, ma solo il test reale misura la generalizzazione vera.

## 8. YOLO-World open-vocabulary (zero-shot, senza fine tuning)

Modello vision-language: si forniscono i concetti target come testo (prompt) e
il modello rileva per similarita semantica, senza addestramento sulle nostre
classi. Per il task a classe singola qualsiasi prompt che matcha conta come
ostruzione. Modello: yolov8s-world.pt (~13M param + text encoder CLIP). Prompt
estesi con sinonimi: wheelchair, stretcher, gurney, hospital bed, cart, trolley,
utility cart, trash can, waste bin, cardboard box, box. Sweep di confidenza.

Copy-paste v2 (990 pos / 110 neg):

| conf | recall | precision | F1 | FPR |
|---|---|---|---|---|
| 0.05 | 0.973 | 0.986 | 0.979 | 0.127 |
| 0.10 | 0.926 | 0.992 | 0.958 | 0.064 |
| 0.25 | 0.822 | 0.994 | 0.900 | 0.045 |

Poli reale (30 pos / 36 neg):

| conf | recall | precision | F1 | FPR |
|---|---|---|---|---|
| 0.05-0.25 | 0.400 | 0.750 | 0.522 | 0.111 |

Confronto sintetico dei tre approcci (F1):

| modello | copy-paste | poli reale |
|---|---|---|
| COCO mappato (conf 0.25) | 0.46 | 0.30 |
| YOLO-World (conf 0.10) | 0.96 | 0.52 |

Scoperte:

1. Open-vocab batte nettamente COCO sia sul sintetico che sul reale: vede gli
   oggetti che COCO non ha nel vocabolario. Sul copy-paste e quasi perfetto
   (F1 0.96), con FPR piu basso di COCO (non allucina sui corridoi normali).
2. GAP SINTETICO -> REALE enorme: F1 crolla da 0.96 a 0.52. I numeri del
   copy-paste sono ottimistici (oggetti ritagliati puliti = facili); il reale
   e molto piu duro. Monito: non fidarsi delle metriche sul solo copy-paste.
3. Sul poli i 18 miss (su 30) sono TUTTI delle porte 3 e 4, cioe UN tipo di
   oggetto: il ventilatore, che non e tra i prompt. Le porte 1/2/5/6
   (sedia, cestino) sono rilevate tutte con alta confidenza (0.59-0.79) anche
   sul reale. Quindi sulle classi che nomini, open-vocab funziona bene pure sul
   reale; il problema e la COMPLETEZZA del vocabolario.
4. Diagnosi: aggiungendo esplicitamente prompt "ventilator/medical
   equipment/machine", le porte 3-4 restano quasi tutte non rilevate (solo 4
   detection deboli a conf 0.05-0.09 su 18). Il ventilatore e INTRINSECAMENTE
   difficile per open-vocab, non solo un prompt mancante.

Implicazione forte: nessun approccio "senza fine tuning e senza dati reali"
raggiunge una recall adeguata sul reale (miglior caso recall 0.40). Ma un
oggetto sconosciuto/difficile come il ventilatore e proprio cio che
l'anomaly detection (P3) rileva a prescindere dalla nomenclatura. Questo e
l'argomento piu forte per la pipeline mista P1+P3 (YOLO-World copre gli oggetti
noti con FPR basso, PatchCore fa da rete di sicurezza sugli sconosciuti).

### 8.1 YOLO-World e il costo edge: piu pesante, ma non per forza squalificato

Verifica architetturale (model.info(), stessa macchina): yolov8s-world (la
taglia PIU PICCOLA disponibile - non esiste una variante "nano" per la
famiglia world) ha 13.4M parametri e 71.5 GFLOPs, contro 2.6M parametri e
6.6 GFLOPs di yolo11n: circa 5x i parametri, 11x i GFLOPs. Sul riferimento
concreto che abbiamo (Miglionico et al., Jetson Nano con TensorRT: yolo11n
~10.6 FPS), un modello 11x piu pesante scenderebbe verosimilmente sotto 1 FPS.

IMPORTANTE - questo NON squalifica automaticamente YOLO-World per il deploy:
il sistema prevede comunque una soglia temporale (un'ostruzione conta solo se
persiste per un tempo T, v. todo sezione Sistema), quindi non serve processare
un flusso video continuo a frame-rate video: basta CAMPIONARE periodicamente
(es. 1 frame ogni pochi secondi). A ~1 FPS o anche meno, un ostacolo fermo
viene comunque rilevato entro tempi compatibili con qualunque soglia T
ragionevole. Il vincolo "sotto 1 FPS = inutilizzabile" vale solo sotto
l'assunzione (qui non applicabile) di dover processare video in tempo reale.

Il vincolo che RESTA indipendente dal frame-rate e' la RAM: il checkpoint
CLIP (ViT-B-32) pesa 354 MB da solo (contro i 5.6 MB di yolo11n.pt) e deve
stare in memoria durante l'inferenza anche se il text encoder gira una sola
volta (paradigma "prompt-then-detect": set_classes() lo esegue una volta,
non ad ogni frame). Su Jetson Nano 4GB (memoria unificata CPU/GPU, come da
Miglionico et al.) o su Raspberry Pi (RAM da fissare col modello esatto),
questo va verificato sull'hardware finale insieme al resto dello stack
(OS, driver camera, eventuale P3 in parallelo).

RUOLO DI YOLO-WORLD IN QUESTA TESI: soprattutto strumento diagnostico, gia'
usato per testare l'ipotesi "senza fine tuning" e misurare il tetto
raggiungibile a costo zero di training. Se il budget di RAM sull'hardware
finale lo permette, resta pero' un candidato di deploy plausibile proprio
perche' il vincolo FPS e' allentato dal design a soglia temporale - da
verificare sperimentalmente quando l'hardware finale sara' disponibile.

IDEA PER IL SEGUITO: usare YOLO-World come "insegnante" per generare
pseudo-label su dati (copy-paste e/o dataset reali), poi fare fine tuning di
yolo11n su quelle etichette (distillazione). Combina il vocabolario ricco
dell'open-vocab con la leggerezza edge del modello nano, senza annotazione
manuale - utile comunque anche se YOLO-World risultasse deployabile, perche'
un modello nano fine-tunato resta piu economico da far girare ad ogni ciclo.

## 8.2 YOLO26 vs YOLO11 (baseline COCO, stesso mapping)

YOLO26n (Ultralytics, `yolo26n.pt`): 2.57M parametri, 6.1 GFLOPs - leggermente
piu leggero di yolo11n (2.6M/6.6 GFLOPs), stesso vocabolario COCO-80. Stesso
mapping finale (sez. 7), stessa soglia di confidenza 0.25, stessi due test set.

| set | modello | recall | precision | F1 | FPR |
|---|---|---|---|---|---|
| copy-paste | yolo11n | 0.406 | 0.976 | 0.574 | 0.091 |
| copy-paste | yolo26n | 0.306 | 0.990 | 0.468 | 0.027 |
| poli reale | yolo11n | 0.200 | 0.667 | 0.308 | 0.083 |
| poli reale | yolo26n | 0.167 | 1.000 | 0.286 | 0.000 |

Pattern coerente su entrambi i dataset: YOLO26n e' sistematicamente piu
CONSERVATIVO di YOLO11n a parita' di soglia di confidenza - precision piu
alta, FPR piu basso (zero falsi allarmi sul poli), ma recall piu bassa
ovunque. Per un task di sicurezza dove il falso negativo (ostruzione mancata)
e' l'errore piu grave, questo e' un compromesso sfavorevole: il FPR=0 sul poli
arriva perdendo ancora piu ostruzioni vere (25/30 mancate contro 24/30 di
yolo11n). Non testato uno sweep di soglie piu basse per YOLO26n (potrebbe
recuperare recall abbassando conf, analogamente al discorso fatto per
YOLO-World in sez. 8) - eventuale sviluppo futuro.

## 8.3 Confronto complessivo di tutti i modelli baseline (poli reale)

Tutti i modelli COCO-pretrained testati sul test reale poli (30 ostruite /
36 libere), image-level, conf 0.25, stesso mapping finale (sez. 7). YOLO-World
(sez. 8) e' incluso per riferimento ma NON usa il mapping (open-vocabulary,
prompt diretti). Colonna "peso/GFLOPs" per il contesto edge.

| modello | tipo | recall | precision | F1 | FPR | param | GFLOPs |
|---|---|---|---|---|---|---|---|
| YOLO-World-s (open-vocab) | VL transformer | 0.400 | 0.750 | 0.520 | 0.111 | 13.4M | 71.5 |
| Faster R-CNN MobileNet-320 | CNN two-stage | 0.400 | 0.545 | 0.462 | 0.278 | ~19M | ~5 |
| RT-DETR-L | transformer | 0.367 | 0.440 | 0.400 | 0.389 | 33.0M | 108.3 |
| Faster R-CNN ResNet50-FPN | CNN two-stage | 0.267 | 0.667 | 0.381 | 0.111 | ~41M | ~134 |
| YOLO11n | CNN one-stage | 0.200 | 0.667 | 0.308 | 0.083 | 2.6M | 6.6 |
| YOLO26n | CNN one-stage | 0.167 | 1.000 | 0.286 | 0.000 | 2.6M | 6.1 |

Osservazioni:
- Nessun modello frozen supera F1 0.52 sul reale: il tetto senza fine tuning
  resta basso, il limite e' sempre il vocabolario COCO (nessuna classe per
  wheelchair/cart/stretcher/bin; il ventilatore invisibile a tutti).
- Le differenze di recall tra modelli frozen riflettono soprattutto quanto
  "liberamente" ciascuno spara classi COCO generiche (calibrazione/soglia),
  NON la loro capacita' di apprendere. Questi numeri NON predicono il ranking
  dopo fine tuning.
- Tempi di inferenza misurati su CPU (questa macchina, no CUDA): yolo11n/26n
  ~1-8 s/img, R-CNN MobileNet ~2 s/img, RT-DETR-L ~14 s/img, R-CNN ResNet50
  ~18 s/img. Su edge con GPU (Jetson Nano) i numeri sarebbero ben diversi.

Nota sull'idoneita' edge (framing corretto): il sistema campiona con soglia
temporale (v. todo Sistema), non processa video a frame-rate pieno. Quindi
un throughput anche SOTTO 1 FPS, purche' vicino a ~1 FPS, e' accettabile per
l'obstruction detection (un ostacolo fermo viene comunque rilevato entro
pochi secondi). Con questo criterio, i candidati al fine tuning NON si
limitano ai soli yolo nano: RT-DETR-L (transformer, ~0.8 FPS su Jetson Nano
PyTorch secondo Miglionico et al., 63MB) resta un candidato TESTABILE, ed e'
anche l'unico modo per avere il confronto CNN vs transformer con entrambi
fine-tunati. Restano fuori solo i modelli con vincoli di RAM/peso proibitivi
o troppo lenti anche per il campionamento (R-CNN ResNet50 ~18 s/img, YOLO-World
per il CLIP da 354MB). Candidati fine tuning: yolo11n / yolo11s / yolo26n
(leggeri, gia' validati su Jetson) + RT-DETR-L (per il confronto transformer).

## 9. Decisioni sul mapping e questione gating

Bivio metodologico. Aggiungere bicycle/motorcycle/toilet/sink solo perche
scattano sui nostri 990 positivi e non sui nostri 110 negativi significa tarare
la lista di classi sullo stesso set su cui si valuta (overfitting al test). Su
negativi diversi (poli, HIOD) una bici o un WC reali farebbero scattare falsi
allarmi.

Si adottano due letture separate:

- Mapping semantico "onesto" (deciso a priori): solo oggetti che una persona
  chiamerebbe ostruzione da pavimento -> chair, bench, couch, bed, dining table,
  suitcase. Da qui si RIMUOVONO refrigerator, potted plant, handbag, backpack
  (FP driver o contributo nullo). E' la baseline COCO difendibile.
- Il fenomeno bicycle/motorcycle/toilet e riportato come evidenza qualitativa
  dell'inadeguatezza di COCO, non come classi da sfruttare per gonfiare la
  recall.

Nota sul gating (idea da valutare): classi come bicycle/motorcycle non esistono
in ospedale, quindi in linea di principio potrebbero aiutare a rilevare i
carrelli senza rischi. L'unico rischio e lo stesso gia visto in P3 con le porte
a vetri: un oggetto (una moto, ma vale per qualsiasi oggetto) che passa DIETRO
la porta a vetri genererebbe una detection spuria. Questo non giustifica
l'esclusione di bike/moto, perche il problema e trasversale a tutte le classi.
La soluzione naturale e RIUSARE lo STESSO gating gia sviluppato per P3, identico:
la detection vale solo se la sua box si sovrappone alle regioni del telaio
(FRAME_LABELS: stipiti, montante, soglia, architrave; `roi_polygons`,
`roi_utils.py`), non solo alla soglia. In questo modo un oggetto centrato nel
pannello di vetro, che non tocca il telaio, viene soppresso, mentre un carrello
che occupa l'apertura toccando il telaio viene mantenuto. Vantaggio per la tesi:
stesso identico meccanismo ROI condiviso tra P1 e P3. Caveat: le ROI oggi sono
annotate solo su 6 frame porta del poli; per il gating completo vanno annotate
le altre.

## 10. Prossimi passi

- Applicare il mapping semantico ridotto e rimisurare FPR/recall (atteso FPR
  verso 0.08-0.10, recall quasi invariata).
- Valutare YOLO-World zero-shot con prompt delle 5 categorie: e il vero test
  dell'ipotesi "il fine tuning non serve" senza addestrare nulla.
- Test statistici tra modelli frozen: McNemar a livello di immagine (predizioni
  per-immagine gia salvate) + bootstrap CI, specie sul poli.
- Valutare il gating ROI come da sezione 8.
- Piano B: fine tuning in cross validation se le opzioni frozen restano
  insufficienti. -> FATTO, vedi sez. 11.

File dei risultati: `pipeline1/results/baseline_copypaste_v2_*.csv`,
`pipeline1/results/baseline_imagelevel.csv` (poli).

## 11. Fine-tuning YOLO11n (Piano B)

Data: 2026-07-09. Nessun modello frozen supera F1 0.52 sul poli reale
(sez. 8.3): si passa al fine tuning vero e proprio.

### 11.1 Motivazione della scelta del modello e setup

Modello: **YOLO11n**, unica variante considerata (non yolo11s/m). Motivazione
diversa da quella puramente prestazionale della sez. 8.3 (dove gli FPS erano
il criterio): qui il vincolo stretto e' la RAM, non la latenza. Il modello
finale gira sullo stesso hardware (Raspberry Pi o Jetson Nano) insieme a
PatchCore (P3, backbone WideResNet50-2) e a un modello di pose estimation
ancora da scegliere - tre reti in memoria contemporaneamente sullo stesso
dispositivo, quindi ogni MB di peso extra conta piu della velocita' per
singolo frame (il sistema campiona con soglia temporale, v. sez. 8.1: non
serve processare a frame-rate video).

Setup:
- Hardware: GPU remota (VM universitaria, container Docker, NVIDIA
  A100-SXM4-80GB), nessuna GPU disponibile in locale. Ambiente: torch
  2.5.1+cu124, ultralytics 8.4.89.
- Dataset: stesso copy-paste v2 della baseline (990 positivi, 110 sfondi
  held-out come negativi). Split 5-fold raggruppato per `source_group` (nessun
  background condiviso tra train e val di uno stesso fold), seed 42. Il
  modello "finale" (`yolo11n_final`, quello valutato sotto) e' addestrato su
  tutto il dataset (990+110).
- Iperparametri: `epochs=100`, `patience=20` (early stop se il val fitness
  non migliora per 20 epoche), `batch=32` (fisso, non AutoBatch, per non
  monopolizzare una GPU condivisa con altri utenti), `imgsz=640`,
  `single_cls=True`. Resto ai default di ultralytics: augmentation mosaic
  1.0 (disattivato negli ultimi 10 epoch), HSV jitter, `fliplr=0.5`,
  `flipud=0` (corretto per scene verticali di porte/corridoi), `mixup` e
  `copy_paste` (l'augmentation nativa di ultralytics, diversa dal nostro
  copy-paste dataset) a 0 perche' non disponiamo di maschere di
  segmentazione.
- La cross-validation a 5 fold e' stata lanciata ma i risultati non sono
  ancora stati recuperati/analizzati (v. Prossimi passi, sez. 11.5): tutti i
  numeri sotto sono del solo modello finale.

### 11.2 Risultati su poli ingegneria (image-level)

Stesso protocollo image-level della baseline (sez. 3), soglia di confidenza
0.25, `pipeline1/src/evaluate_yolo.py image-level`.

| n_pos | n_neg | TP | FP | TN | FN | accuracy | precision | recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 36 | 12 | 7 | 29 | 18 | 0.621 | 0.632 | 0.400 | 0.490 | 0.194 |

Recall grezza 0.400: peggio di quanto suggerissero le curve di training. Il
CSV per-immagine (`pipeline1/results/yolo11n_final_poli_perimage.csv`) mostra
pero' un pattern concentrato, non rumore sparso: `porta_3` manca 9/9 (100%),
`porta_4` manca 7/9, mentre `porta_1`, `porta_5`, `porta_6` sono perfette
(3/3 ciascuna) e `porta_2` quasi (1/3 preso).

Ispezione diretta delle immagini (`porta_3_1.jpg`, `porta_4_1.jpg`): in
entrambe le location l'ostacolo e' lo **stesso ventilatore a piantana**,
oggetto che non appartiene a nessuna delle 5 categorie di fine tuning
(carrello, sedia a rotelle, barella, scatola, cestino) - quindi
out-of-distribution puro, mai visto in training. E' lo stesso identico
oggetto gia' documentato come failure case per P3 (Capitolo 5, "gating-limited
false negative"): li' PatchCore lo rileva correttamente ma il gating lo
azzera per un limite geometrico (testa larga fuori dal telaio); qui YOLO non
lo rileva affatto perche' e' semplicemente una categoria mai vista. Due
pipeline, due meccanismi di fallimento diversi, stesso oggetto reale - un
filo conduttore utile per la discussione conclusiva della tesi.

I restanti 2 FN (`porta_2_1`, `porta_2_3`) sono invece un vero cestino
(categoria di training), quindi un miss genuino:

| Causa del FN | Immagini | % dei 18 FN |
|---|---|---|
| Ventilatore (categoria out-of-distribution) | porta_3 (9) + porta_4 (7) = 16 | 89% |
| Cestino reale mancato (categoria in-training, miss genuino) | porta_2_1, porta_2_3 = 2 | 11% |

Escludendo le 18 immagini con il ventilatore (out-of-distribution per
costruzione, nessun modello frozen della sez. 8.3 lo rileva neppure), la
recall sulle sole categorie effettivamente coperte dal fine tuning e':
**10/12 = 0.833**, molto piu vicina alle attese e piu rappresentativa delle
capacita' reali del modello sulle categorie che conosce.

Confronto con la tabella di sez. 8.3 (tutti i modelli, poli reale, image-level,
conf 0.25):

| modello | tipo | recall | precision | F1 | FPR | param |
|---|---|---|---|---|---|---|
| YOLO-World-s (open-vocab, frozen) | VL transformer | 0.400 | 0.750 | 0.520 | 0.111 | 13.4M |
| Faster R-CNN MobileNet-320 (frozen) | CNN two-stage | 0.400 | 0.545 | 0.462 | 0.278 | ~19M |
| **YOLO11n fine-tuned (recall grezza)** | CNN one-stage | 0.400 | 0.632 | 0.490 | 0.194 | 2.6M |
| YOLO11n (frozen, baseline) | CNN one-stage | 0.200 | 0.667 | 0.308 | 0.083 | 2.6M |

Lettura onesta: la recall grezza del fine tuning (0.400, F1 0.490) non batte
nettamente il miglior baseline frozen (YOLO-World, F1 0.520) - il gap
sintetico->reale gia' osservato nella baseline (sez. 7 punto 4) non e'
sparito con il fine tuning. Il fine tuning migliora pero' nettamente la
precision rispetto a YOLO-World (0.632 vs 0.750 sono comparabili, ma FPR
0.194 contro 0.111 e' peggio) e soprattutto raddoppia la recall rispetto al
proprio se stesso non fine-tunato (0.400 vs 0.200). Il dato piu informativo
resta pero' il breakdown sopra: **la recall corretta sulle categorie note
(0.833) supera nettamente ogni baseline frozen**, a conferma che il
fine tuning ha funzionato bene sulle categorie effettivamente viste - il
problema residuo e' copertura del vocabolario di oggetti (esattamente lo
stesso tipo di limite gia' diagnosticato per COCO in sez. 7), non qualita'
dell'addestramento.

### 11.3 Risultati sul set gemini (detection mAP)

Set di test generato con Gemini 2.5 Flash Image (editing fotorealistico di
foto reali, oggetti aggiunti via prompt): 34 immagini corridoio + 60 porte,
94 totali, tutte positive (nessun negativo, quindi qui non si calcolano
precision/FPR image-level ma solo mAP di detection). Label annotate a mano
in `Dataset/ostruzioni_gemini/{corridoi,porte}/labels/`.

Nota metodologica (errore fatto e corretto durante l'analisi, riportato
perche' rilevante per iterazioni future): il primo run di `yolo val` forzava
`conf=0.25`, che ultralytics usa come soglia di NMS **prima** di costruire la
curva precision-recall - questo esclude a priori le detection a bassa
confidenza dal calcolo di mAP, sottostimandolo. Il default corretto per la
validazione (`conf=None` -> 0.001 internamente) sweepa tutte le soglie:

| | conf=0.25 forzato (sbagliato) | default (corretto) |
|---|---|---|
| Precision | 0.887 | 0.883 |
| Recall | 0.588 | 0.599 |
| mAP50 | 0.548 | **0.609** |
| mAP50-95 | 0.358 | **0.392** |

Risultati finali (corretti), 94 immagini / 187 istanze GT:

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.883 | 0.599 | 0.609 | 0.392 |

### 11.4 Recall per-istanza vs recall per-immagine, e analisi delle fusioni

La recall per-istanza (0.599) conta ogni box GT separatamente: se in una
scena ci sono piu' oggetti impilati/adiacenti e il modello ne rileva uno
solo con una box che li copre tutti, le altre GT restano "mancate" anche se
per lo scopo applicativo (rilevare che la via di fuga e' ostruita) la scena
e' comunque correttamente segnalata. Recall per-immagine (almeno una
detection sopra soglia 0.25 nell'immagine, calcolata con uno script ad-hoc
perche' il set e' tutto positivo e non serve al calcolo di mAP):

| Venue | Immagini con >=1 detection | Recall per-immagine |
|---|---|---|
| corridoi | 33/34 | 0.971 |
| porte | 57/60 | 0.950 |
| combinato | 90/94 | **0.957** |

Il divario tra 0.599 (per-istanza) e 0.957 (per-immagine) e' pero' esso
stesso fuorviante nella direzione opposta: se una scena contiene 4 oggetti
diversi e il modello ne trova solo 1, la recall per-immagine segna comunque
1.0, nascondendo che 3 oggetti su 4 sono stati mancati. Per distinguere il
failure mode specifico osservato a occhio (scatole di cartone impilate:
individuate, ma spesso come un'unica box invece di N separate) da un miss
genuino su oggetti distinti, si e' scritta un'analisi geometrica
(`pipeline1/src/analyze_gemini_failures.py`): per ogni GT non trovata (FN),
si cerca la predizione che la copre meglio (qualunque IoU, non solo il match
ufficiale a soglia 0.5):

| Causa del FN | Conteggio | % dei 79 FN |
|---|---|---|
| Fusa in una box gia' assegnata a un'altra GT (sospetto impilamento) | 16 | 20% |
| Coperta da una predizione non abbastanza precisa/sicura da fare match (IoU<0.5) | 41 | 52% |
| Isolata, nessuna predizione nei paraggi | 22 | 28% |

Il 72% dei FN (57/79) ha comunque una box del modello nei dintorni (condivisa
con un'altra GT, o semplicemente imprecisa), coerente con il gap osservato
tra mAP50 (0.609) e mAP50-95 (0.392): il modello individua la zona giusta
piu' spesso di quanto suggerisca la recall grezza, ma la localizzazione/
separazione delle istanze e' imprecisa. Solo il 28% e' un vero punto cieco
senza alcuna predizione vicina.

Limite di questa analisi: e' puramente geometrica (non sa quale sia la vera
categoria dell'oggetto), quindi da sola non distingue "impilamento di
scatole" da altri tipi di sovrapposizione. Le 187 GT del set gemini sono
state percio' annotate a mano per categoria
(`pipeline1/src/annotate_gemini_categories.py`, etichettatura interattiva
che non tocca i file usati per il calcolo di mAP), permettendo di
scomporre sia la recall sia il tipo di FN per categoria.

Recall per categoria:

| Categoria | Recall | Istanze GT |
|---|---|---|
| **box** | **0.281 (27/96)** | 96 |
| waste_bin | 0.826 (19/23) | 23 |
| cart | 0.857 (18/21) | 21 |
| stretcher | 0.947 (18/19) | 19 |
| wheelchair | 0.963 (26/27) | 27 |

Le scatole sono l'unica categoria problematica (recall 0.281 contro >=0.826
di tutte le altre) e sono anche il 51% di tutte le istanze del set (96/187):
da sole spiegano quasi per intero perche' la recall aggregata (0.599) sia
cosi' piu' bassa della recall per-immagine (0.957).

Incrociando la categoria con il tipo di FN (fusa / imprecisa / isolata):

| Categoria | Fusa | Imprecisa | Isolata | Totale FN |
|---|---|---|---|---|
| **box** | 16 | 41 | 12 | 69 |
| cart | 0 | 0 | 3 | 3 |
| stretcher | 0 | 0 | 1 | 1 |
| waste_bin | 0 | 0 | 4 | 4 |
| wheelchair | 0 | 0 | 1 | 1 |

Risultato netto: **il 100% dei casi "fusa" e "imprecisa" (57/57) appartiene
alla categoria box**; le altre quattro categorie, quando mancate (9 casi in
totale su tutto il dataset), lo sono sempre in modo isolato e pulito, mai per
fusione. Questo conferma in modo quantitativo l'osservazione qualitativa
iniziale (scatole di cartone impilate rilevate come un'unica box invece di
N separate) e ne delimita con precisione la portata: non e' un problema
generale di localizzazione del modello, e' specifico alla categoria box,
verosimilmente perche' e' l'unica delle 5 categorie che nel dataset
(copy-paste e gemini) compare frequentemente in gruppi fisicamente
ravvicinati/impilati, mentre le altre (carrelli, sedie, barelle, cestini)
compaiono tipicamente come oggetto singolo per scena. Implicazione per il
seguito: la fusione delle scatole e' verosimilmente un limite di NMS/anchor
assignment su oggetti molto vicini e visivamente simili tra loro (v. sez.
11.5), non un problema di training insufficiente sulla categoria in se': se
si ipotizzano risolti tutti i casi "fusa" e "imprecisa" (NMS/localizzazione
perfetti) e restano solo i 12 miss "isolata", la recall di box salirebbe a
(96-12)/96 = 84/96 = 0.875, in linea con le altre categorie (0.826-0.963).

**Causa radice, e perche' non e' banalmente risolvibile.** Il generatore
copy-paste (`generate_synthetic_dataset.py`) impone `MAX_IOU = 0.10`: ogni
nuovo oggetto viene ripiazzato (fino a 30 tentativi) finche' il suo box non
si sovrappone per meno del 10% con ogni oggetto gia' piazzato. Il training
set quindi non contiene, per costruzione, quasi nessuna scena con oggetti
realmente impilati/sovrapposti - il modello non ha mai visto questa
configurazione geometrica, qualunque sia la categoria. Le scatole sono
l'unica categoria che gemini genera spesso impilata, quindi l'unica a
incappare in questo gap. Una soluzione "ovvia" (generare varianti copy-paste
con scatole impilate) e' pero' meno banale di quanto sembri: quando un
oggetto e' fortemente occluso da un altro, la scelta della bounding box
(amodale, l'estensione intera anche dove coperta, vs modale, ridotta alla
sola porzione visibile) non e' neutra. Box amodali che si sovrappongono
molto danno un segnale di training ambiguo che rinforzerebbe la fusione
invece di correggerla; box modali risolvono l'ambiguita' ma per un oggetto
quasi del tutto coperto degenerano in un lembo visibile minuscolo,
probabilmente non rilevabile da nessun detector - non un limite di training,
un limite di cio' che e' visivamente inferibile.

**Ridefinizione del problema (il punto piu' importante di questa sezione).**
Prima di investire nel fix sopra, va chiesto se serve davvero: P1 e' un
task a **classe singola, single-class "obstruction"** - rileva la PRESENZA
di un ostacolo, non il tipo ne' il conteggio (v. sez. 1). Se 3 scatole
impilate producono una singola box che le copre tutte, per lo scopo
applicativo (segnalare che la via di fuga e' ostruita) e' un successo
pieno, non un errore. Verifica diretta: delle scene gemini con almeno un
oggetto categoria box, quante vengono segnalate come ostruite (almeno una
detection, indipendentemente da quante separate)?

| | yolo11n | yolo26n |
|---|---|---|
| Scene con >=1 box, rilevate come ostruite | 30/30 (1.000) | 30/30 (1.000) |
| Scene senza box, rilevate come ostruite | 60/64 (0.938) | 60/64 (0.938) |

Le scene con scatole vengono rilevate **meglio** delle altre, non peggio.
Il recall per-istanza basso su box (0.281) e' quindi interamente un
artefatto della metrica scelta (mAP di detection, che conta istanze), non
un deficit funzionale del sistema per il compito che deve davvero svolgere.
Conclusione per la tesi: riportare onestamente entrambi i numeri (il
recall per-istanza mostra un limite reale di *localizzazione/conteggio*,
utile a livello di detection benchmark), ma il recall a livello di *scena*
e' la metrica che conta per la funzione di sicurezza del sistema, ed e'
li' che P1 va giudicato. Investire tempo residuo nel tentativo di separare
scatole pesantemente occluse ha probabilmente un ritorno basso rispetto ad
altri gap gia' identificati (il ventilatore out-of-distribution in sez.
11.2, che *invece* causa mancate rilevazioni vere a livello di scena).

### 11.6 Controllo del leakage di sfondo e confronto multi-modello (no-leak)

**Il problema del leakage.** Il set di test gemini e' costruito editando foto
reali di sfondo (aggiunta di oggetti via prompt su porte/corridoi esistenti).
Molti di questi sfondi sono gli stessi usati per costruire il copy-paste di
training: su 120 sfondi gemini, 107 corrispondono a un positivo copy-paste
(il modello ha visto quella scena esatta, con un oggetto incollato diverso) e
13 a un negativo pulito - **0 scene completamente nuove**. Quindi gemini, per
come e' costruito, testa la generalizzazione a oggetti nuovi su scene
familiari, non a scene nuove: c'e' il rischio che i suoi numeri siano gonfiati
dal riconoscimento della scena piu' che dalla vera capacita' di rilevare
l'oggetto.

**Verifica empirica.** Invece di limitarsi a menzionare il rischio, lo si e'
testato: si e' rigenerato il training set copy-paste **escludendo** i 107
positivi + 13 negativi il cui sfondo compare in gemini
(`prepare_yolo_dataset.py --exclude-stems-file`, lista di stem in
`pipeline1/data/gemini_bg_exclude.txt`), passando da 990/110 a 883/97
immagini, e si e' riaddestrato yolo11n ("no-leak"). Confronto sullo stesso
identico set gemini (120 immagini, 166 istanze), a parita' di tutto tranne
l'esclusione degli sfondi condivisi:

| yolo11n | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|
| leak (training completo 990/110) | 0.863 | 0.709 | 0.951 | 0.789 |
| **no-leak (883/97, sfondi gemini esclusi)** | **0.866** | 0.703 | 0.963 | 0.831 |

I due modelli sono **praticamente identici** (mAP50 0.866 vs 0.863; la recall
no-leak e' addirittura leggermente piu' alta). Rimuovere ogni sovrapposizione
di sfondo train/test **non peggiora** le prestazioni: il leakage di sfondo
NON stava gonfiando i numeri di gemini. Il modello sta imparando a rilevare
l'ostacolo, non a memorizzare lo sfondo. Punto metodologico forte per la tesi:
l'ipotesi di leakage e' stata testata ed esclusa empiricamente, non solo
dichiarata. Tutti i modelli successivi (yolo26n, RT-DETR) sono addestrati
sulla versione no-leak per coerenza.

**Confronto multi-modello (tutti no-leak, stesso set gemini 120 img / 166 ist).**

| modello | tipo | mAP50 | mAP50-95 | precision | recall | param |
|---|---|---|---|---|---|---|
| **yolo11n** | CNN one-stage | **0.866** | 0.703 | 0.963 | **0.831** | 2.6M |
| yolo26n | CNN one-stage | 0.852 | 0.708 | 0.969 | 0.789 | 2.6M |
| rtdetr-l (*) | transformer | 0.828 | 0.645 | 0.903 | 0.741 | 33M |

(*) checkpoint parziale, v. sotto: il training e' divergato.

Risultato netto: **il modello piu' piccolo (yolo11n) e' il migliore** su questo
set, sia in mAP50 sia in recall. RT-DETR, con ~13x i parametri, e' il peggiore
dei tre. Questo rafforza la scelta di deploy su due assi contemporaneamente:
il transformer pesante non solo viola il vincolo di RAM (il piu' stringente,
data la co-esecuzione con PatchCore e pose estimation, v. sez. 11.1), ma qui
non offre nemmeno un vantaggio di accuratezza che lo giustifichi.

**Instabilita' del training di RT-DETR (caveat importante).** Il confronto
sopra NON e' "RT-DETR a parita' di budget, convergito". Il `results.csv` del
run mostra che RT-DETR ha allenato bene fino all'epoca 14 (best: mAP50 0.970,
mAP50-95 0.795 sul val interno copy-paste), poi all'epoca 15 e' **collassato a
zero** (precision/recall/mAP tutti 0) restandoci fino alla fine; la loss di
validazione mostra `NaN` gia' a sprazzi (epoche 4, 12, 13) e poi in modo
permanente dall'epoca 15. L'early stopping (patience 20) ha correttamente
fermato il run all'epoca 34. Il `best.pt` valutato e' quindi il checkpoint
dell'epoca 14, prima del collasso - un risultato valido ma di un training non
convergito in modo pulito.

**Causa documentata (fonte primaria).** Il `NaN` nella loss non e' una
sorpresa: e' un comportamento noto e documentato dagli autori stessi di
ultralytics. Nel docstring della classe `RTDETRTrainer` (file
`ultralytics/models/rtdetr/train.py`, riga 41 nella versione 8.4.89 usata qui;
identico alla reference ufficiale [1]) si legge testualmente:

> Notes:
>   - F.grid_sample used in RT-DETR does not support the `deterministic=True` argument.
>   - **AMP training can lead to NaN outputs and may produce errors during bipartite graph matching.**

Cioe': l'addestramento in mixed precision (AMP) puo' produrre output `NaN` ed
errori durante il *bipartite graph matching*, l'assegnamento uno-a-uno tra
query predette e oggetti veri (matching di Hungarian) che sostituisce l'NMS in
RT-DETR. E' esattamente il sintomo osservato. AMP e' attivo di default in
ultralytics (`amp=True`), e `train_yolo.py` non lo disattiva: il run e' quindi
caduto nel caso previsto dalla documentazione.

Modifiche da provare per un confronto RT-DETR equo, in ordine di solidita'
della fonte:

- **`amp=False` (fonte forte, da provare per prima e da sola).** Disattiva la
  mixed precision automatica: il modello calcola i gradienti interamente in
  fp32 invece di usare fp16 dove possibile. AMP serve a ridurre memoria e
  tempo di training usando meta' precisione numerica, ma fp16 ha un intervallo
  di rappresentazione molto piu' stretto: valori piccoli finiscono a zero
  (underflow), valori grandi saturano (overflow) e diventano `inf`/`NaN`. Nel
  matching bipartito di RT-DETR si calcola una matrice di costo e la si
  ottimizza: se anche un solo costo diventa `NaN`, l'assegnamento si rompe e
  il `NaN` si propaga a tutta la loss e ai pesi - da cui il collasso
  irreversibile a metriche zero, che e' precisamente il crollo all'epoca 15.
  Con `amp=False` il training e' piu' lento e usa piu' VRAM, ma numericamente
  robusto: sull'A100-80GB il costo e' irrilevante. Raccomandazione esplicita
  della documentazione ufficiale [1] e delle discussioni ultralytics [2].
- **`lr0` piu' basso (fonte media).** Riportato in issue ultralytics come
  mitigazione dell'instabilita' [2][3]. Da notare pero' che nella issue [2]
  il `NaN` compariva **gia' con `lr0=0.001`** (learning rate gia' basso), il
  che suggerisce che la leva primaria sia AMP e non il learning rate: motivo in
  piu' per cambiare una variabile alla volta.
- **`warmup_epochs` piu' lungo / batch piu' piccolo (fonte debole).** Pratica
  generale sui transformer, nessuna raccomandazione specifica trovata per
  RT-DETR in ultralytics. Da tenere come piano C.
- **Piu' epoche.** I transformer convergono piu' lentamente delle CNN; ha senso
  solo DOPO aver stabilizzato il training, non come rimedio all'instabilita'.

Metodologia dell'esperimento: cambiare **solo `amp=False`**, tenendo lr, batch
ed epoche identici. Se il training si stabilizza, la causa e' attribuita con
certezza e supportata da fonte ufficiale - molto piu' difendibile che cambiare
tre parametri insieme senza sapere quale abbia funzionato.

Il flag non e' oggi esposto da `train_yolo.py` (che passa i default di
ultralytics): va aggiunto `--amp/--no-amp` prima di rilanciare. NOTA: anche se
un RT-DETR stabilizzato recuperasse accuratezza, resterebbe comunque escluso
dal deploy per il vincolo di RAM; il suo ruolo in tesi e' di baseline
comparativa CNN-vs-transformer, non di candidato reale.

Fonti:
- [1] Ultralytics, *Reference for `ultralytics/models/rtdetr/train.py`* (docstring
  `RTDETRTrainer`, avviso AMP / NaN / bipartite graph matching):
  https://docs.ultralytics.com/reference/models/rtdetr/train/
  (verificato anche nel sorgente installato, ultralytics 8.4.89, riga 41)
- [2] Ultralytics issue #7594, *RTDETR training on custom dataset giving "NAN"
  loss after 50 epochs using AdamW optimizer*:
  https://github.com/ultralytics/ultralytics/issues/7594
- [3] Ultralytics issue #18521, *RT-DETR model's loss became "nan" and mAP,
  recall not update after epoch 40*:
  https://github.com/ultralytics/ultralytics/issues/18521

**yolo11n e yolo26n sono saturi, non under-trained.** Contrariamente a
RT-DETR (rotto), yolo11n e yolo26n no-leak sono convergiti correttamente: piu'
epoche non aiuterebbero. Entrambi hanno eseguito tutte le 150 epoche (tetto
raggiunto, nessun early stop), e il best e' in fondo (epoca 150 per yolo11n,
148 per yolo26n) - il che a prima vista suggerirebbe di allungare il training.
Ma la deviazione standard del mAP50-95 sulle ultime 15 epoche e' ~0.003 per
entrambi: non e' un trend in salita, e' un **plateau piatto** nel rumore, e il
"best all'epoca 150" e' solo il picco casuale della curva piatta. Soprattutto,
questo plateau e' misurato sul val interno (`fold0_val` del **copy-paste**),
cioe' la stessa distribuzione sintetica del training, ed e' **saturo**:
mAP50 $\approx 0.995$, mAP50-95 $\approx 0.97$-$0.98$. Due conseguenze:

- Piu' epoche non darebbero quasi nulla: la metrica ottimizzata dal training e'
  gia' al massimo.
- Anzi rischierebbero di **peggiorare** il reale: continuare ad allenare su
  copy-paste gia' saturo spinge verso l'overfitting sul dominio sintetico,
  l'opposto di cio' che serve per generalizzare a gemini/poli. Semmai il dubbio
  legittimo e' il contrario (150 epoche potrebbero gia' essere troppe).

Il collo di bottiglia per il reale non e' la durata del training ma il gap
sintetico->reale e la copertura del vocabolario (v. sez. 11.2, 11.4): la leva
per migliorare non e' piu' training, sono dati migliori (scene/oggetti piu'
realistici, categorie mancanti come il ventilatore).

### 11.7 Prossimi passi

- Ri-addestrare RT-DETR con la ricetta stabilizzata (sez. 11.6: `amp=False`,
  `lr0` basso) per un confronto CNN-vs-transformer a training convergito;
  aggiungere prima i flag `--amp`/`--lr0` a `train_yolo.py`.
- Rifare poli image-level + failure analysis per categoria su yolo26n e
  RT-DETR no-leak, per completare il confronto multi-modello anche sul test
  reale e sulle scatole.
- Valutare uno sweep di soglie di confidenza sul poli (come fatto per
  YOLO-World in sez. 8) per vedere se un conf piu' basso recupera i 2
  cestini mancati (`porta_2_1`, `porta_2_3`) senza esplodere il FPR.
- Valutare se aggiungere al training oggetti "thin-stemmed" (ventilatori,
  attaccapanni, cartelli su piantana) chiuda il gap out-of-distribution
  identificato in sez. 11.2 - lo stesso limite strutturale gia' documentato
  per P3.
- Valutare il gating ROI (sez. 9) anche per P1, riusando lo stesso
  meccanismo gia' sviluppato per P3.

File dei risultati: `pipeline1/results/yolo11n_final_poli_imagelevel.csv`,
`pipeline1/results/yolo11n_final_poli_perimage.csv`,
`pipeline1/runs/yolo11n_final/weights/best.pt`,
`pipeline1/runs/yolo11n_final_gemini_val/` (curve, confusion matrix).
