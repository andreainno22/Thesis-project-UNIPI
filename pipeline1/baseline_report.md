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
  insufficienti.

File dei risultati: `pipeline1/results/baseline_copypaste_v2_*.csv`,
`pipeline1/results/baseline_imagelevel.csv` (poli).
