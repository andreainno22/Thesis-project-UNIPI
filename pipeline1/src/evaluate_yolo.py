"""
Evaluate P1 obstruction detection: detection mAP (CV) + image-level metrics.
====================================================================

Two sub-commands:

  image-level : run a detector over a positive set and a negative set and
                score it as a binary "obstructed / clear" classifier
                (accuracy, precision, recall, F1, FPR, FNR). This is the
                metric shared with P3 for cross-pipeline comparison.
                Works for a fine-tuned single-class model OR a COCO baseline
                (--weights coco enables the COCO->obstruction mapping).

  cv-map      : aggregate detection mAP@0.5 and mAP@0.5:0.95 across the CV
                folds trained by train_yolo.py (needs the copy-paste labels).

Examples:
  # baseline (non fine-tuned COCO) on the poli real test set, from the DB
  python pipeline1/src/evaluate_yolo.py image-level \
      --weights coco --split test --conf 0.25 \
      --out-csv pipeline1/results/baseline_imagelevel.csv

  # fine-tuned model on the same test set
  python pipeline1/src/evaluate_yolo.py image-level \
      --weights pipeline1/runs/final/weights/best.pt --split test \
      --out-csv pipeline1/results/finetuned_imagelevel.csv

  # cross-validation detection mAP
  python pipeline1/src/evaluate_yolo.py cv-map \
      --runs-dir pipeline1/runs --name cv --folds-dir pipeline1/data \
      --out-csv pipeline1/results/cv_map.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent))
from coco_obstruction_map import (  # noqa: E402
    MIN_BOX_AREA_FRAC,
    is_obstruction_detection,
    obstruction_class_ids,
)
from prepare_yolo_dataset import parse_pruned_backgrounds  # noqa: E402


# --------------------------------------------------------------------------- #
# sample listing
# --------------------------------------------------------------------------- #
# Ogni funzione qui sotto produce la stessa cosa: una lista di coppie
# (percorso_immagine, is_positive), dove is_positive=1 significa "ostruita" e
# is_positive=0 significa "libera/normale". Tre modi diversi di ottenerla, a
# seconda di dove vivono i dati che vuoi valutare.

def samples_from_db(db: str, dataset_root: Path, split: str) -> list[tuple[Path, int]]:
    """Return [(image_path, is_positive)] for a DB split (positive = obstructed).

    Usata per il test set 'poli ingegneria': legge dalla tabella frames tutte
    le immagini con quello split (es. 'test'), e deduce l'etichetta da
    is_normal (che nel DB e' 1 per le immagini pulite, 0 per quelle ostruite:
    per questo il valore restituito e' l'OPPOSTO, 0 if is_normal else 1).
    """
    conn = sqlite3.connect(db)
    rows = conn.execute(
        "SELECT file_path, is_normal FROM frames WHERE split = ?", (split,)
    ).fetchall()
    conn.close()
    out = []
    for file_path, is_normal in rows:
        out.append((dataset_root / file_path, 0 if is_normal else 1))
    return out


def samples_copypaste(db: str, dataset_root: Path, pruned_file: Path) -> list[tuple[Path, int]]:
    """Copy-paste evaluation set: 990 composites as positives, the 110
    pruned held-out backgrounds as negatives (same sources as prepare).

    Non usa la colonna 'split' del DB (li' il copy-paste e' tutto marcato
    'train', perche' e' pensato per il training/CV, non per un test fisso).
    Qui invece ricostruiamo lo STESSO set di positivi/negativi che usa
    prepare_yolo_dataset.py per costruire i fold, cosi' possiamo valutare un
    modello frozen (COCO, YOLO-World) sull'intero copy-paste in un colpo solo:
      - positivi: tutte le 990 immagini con occlusion_type='synthetic_copypaste'
      - negativi: i 110 sfondi "pruned" (mai usati per il copy-paste, quindi
        non ci sarebbe leakage) elencati in pruned_backgrounds.txt
    parse_pruned_backgrounds() legge quel file e restituisce l'insieme di stem
    (es. 'corridoio_015'); la IN (...) con i placeholder e' costruita a mano
    perche' sqlite3 non accetta liste Python direttamente in una query.
    """
    conn = sqlite3.connect(db)
    pos = [r[0] for r in conn.execute(
        "SELECT file_path FROM frames WHERE occlusion_type = 'synthetic_copypaste'"
    )]
    stems = parse_pruned_backgrounds(pruned_file)
    placeholders = ",".join("?" * len(stems))  # "?,?,?,...", uno per stem
    neg = [r[0] for r in conn.execute(
        f"SELECT file_path FROM frames WHERE is_normal = 1 "
        f"AND source = 'Pinterest' AND source_group IN ({placeholders})",
        tuple(stems),
    )]
    conn.close()
    return ([(dataset_root / p, 1) for p in pos]
            + [(dataset_root / p, 0) for p in neg])


def samples_from_dirs(pos_dir: Path | None, neg_dir: Path | None) -> list[tuple[Path, int]]:
    """Fallback generico: leggi tutte le immagini da due cartelle qualsiasi
    (una di positivi, una di negativi), senza passare dal DB. Utile per
    valutazioni ad hoc su dataset esterni."""
    exts = {".jpg", ".jpeg", ".png"}
    out: list[tuple[Path, int]] = []
    for d, lab in ((pos_dir, 1), (neg_dir, 0)):
        if d is None:
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in exts:
                out.append((p, lab))
    return out


# --------------------------------------------------------------------------- #
# image-level scoring
# --------------------------------------------------------------------------- #
def predict_is_obstructed(
    model: YOLO,
    img: Path,
    conf: float,
    coco_map: bool,
    obstruction_ids: set[int],
    min_area: float,
) -> bool:
    """Un'immagine e' "ostruita" se il modello ci trova sopra ALMENO UNA
    detection valida. Due modi di essere "valida", a seconda del modello:

    - modello fine-tunato a classe singola: ha UNA sola classe possibile
      ("obstruction"), quindi qualsiasi box rilevata sopra la soglia conf
      conta gia' come ostruzione (coco_map=False).
    - baseline COCO (coco_map=True): il modello ha 80 classi generiche, quindi
      bisogna FILTRARE le detection e tenere solo quelle la cui classe e' nel
      nostro mapping (v. coco_obstruction_map.py) e la cui area e' abbastanza
      grande da non essere un dettaglio trascurabile (min_area, es. un piccolo
      oggetto sullo sfondo). is_obstruction_detection() fa questo controllo.
    """
    r = model.predict(str(img), conf=conf, verbose=False)[0]
    if len(r.boxes) == 0:
        return False
    if not coco_map:
        return True  # single-class model: any detection = obstruction
    # baseline: keep only mapped classes above the area threshold
    cls = r.boxes.cls.tolist()
    wh = r.boxes.xywhn[:, 2:4].tolist()  # normalized w,h -> area frac = w*h
    for c, (w, h) in zip(cls, wh):
        if is_obstruction_detection(int(c), w * h, obstruction_ids, min_area):
            return True
    return False


def image_level(args) -> None:
    """Valuta un modello come classificatore binario ostruita/libera su un
    intero dataset, e stampa/salva le metriche condivise con P3 (accuracy,
    precision, recall, F1, FPR, FNR)."""

    # 1) Scegli la sorgente delle immagini in base ai flag passati da riga di
    #    comando: copy-paste (990+110), uno split del DB (es. 'test' = poli),
    #    oppure due cartelle qualsiasi.
    dataset_root = Path(args.dataset_root)
    if args.copy_paste:
        samples = samples_copypaste(args.db, dataset_root, Path(args.pruned_file))
    elif args.split:
        samples = samples_from_db(args.db, dataset_root, args.split)
    else:
        samples = samples_from_dirs(
            Path(args.pos_dir) if args.pos_dir else None,
            Path(args.neg_dir) if args.neg_dir else None,
        )
    if not samples:
        raise SystemExit("no samples found (check --split or --pos/neg-dir)")

    # 2) Carica il modello. --weights coco e' un valore speciale (shortcut
    #    storico): invece di un path a un checkpoint, attiva la modalita'
    #    "baseline COCO non fine-tunata" scaricando yolo11n.pt. Per testare
    #    un ALTRO checkpoint pretrained-COCO (es. yolo26n.pt) con lo stesso
    #    mapping di classi, si passa il path esplicito + --coco-map.
    coco_map = args.weights.lower() == "coco" or args.coco_map
    weights = "yolo11n.pt" if args.weights.lower() == "coco" else args.weights
    model = YOLO(weights)
    # model.names e' il dizionario {id_classe: nome} del modello (per COCO,
    # {0:'person', 1:'bicycle', ...}); obstruction_class_ids() lo intersects
    # con OBSTRUCTION_COCO_NAMES per ottenere gli ID numerici da tenere.
    obstruction_ids = obstruction_class_ids(model.names) if coco_map else set()
    if coco_map:
        print(f"baseline COCO mapping -> obstruction ids {sorted(obstruction_ids)}, "
              f"min area frac {args.min_area}")

    # 3) Passa ogni immagine nel modello e accumula la matrice di confusione
    #    (TP/FP/TN/FN) confrontando l'etichetta vera (is_pos, dal DB) con la
    #    predizione del modello (pred, da predict_is_obstructed).
    tp = fp = tn = fn = 0
    missing = 0
    per_image: list[dict] = []  # un record per immagine, per CSV/analisi extra
    for img, is_pos in samples:
        if not img.exists():
            missing += 1
            continue
        pred = predict_is_obstructed(
            model, img, args.conf, coco_map, obstruction_ids, args.min_area
        )
        per_image.append({"image": img.name, "is_obstructed": is_pos,
                          "predicted": int(pred)})
        if is_pos and pred:
            tp += 1          # ostruita, rilevata           -> vero positivo
        elif is_pos and not pred:
            fn += 1          # ostruita, NON rilevata        -> falso negativo (pericoloso!)
        elif not is_pos and pred:
            fp += 1          # libera, rilevata per errore   -> falso positivo (falso allarme)
        else:
            tn += 1          # libera, correttamente ignorata -> vero negativo

    # 4) Dalla matrice di confusione derivano tutte le metriche standard.
    #    n_pos/n_neg sono i conteggi REALI (non le predizioni): TP+FN = tutte
    #    le immagini che erano davvero ostruite, TN+FP = tutte quelle libere.
    n_pos, n_neg = tp + fn, tn + fp
    precision = tp / (tp + fp) if (tp + fp) else 0.0   # tra le "ostruita" predette, quante lo erano davvero
    recall = tp / n_pos if n_pos else 0.0              # tra le ostruzioni vere, quante ne abbiamo trovate
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) else 0.0)            # media armonica di precision e recall
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    fpr = fp / n_neg if n_neg else 0.0                 # falso-allarme rate (quanto "grida al lupo" a vuoto)
    fnr = fn / n_pos if n_pos else 0.0                 # tasso di ostruzioni mancate (il piu' critico per la sicurezza)

    # Nome del modello per report/DB: "coco-baseline" resta il nome storico
    # SOLO per lo shortcut --weights coco (yolo11n), cosi' i risultati gia'
    # salvati restano confrontabili; qualsiasi altro checkpoint (es. yolo26n.pt
    # con --coco-map) e' etichettato con il proprio nome file, con "+coco-map"
    # per ricordare che e' comunque stato applicato il filtro di classi.
    if args.weights.lower() == "coco":
        model_label = "coco-baseline"
    elif coco_map:
        model_label = f"{Path(args.weights).name}+coco-map"
    else:
        model_label = Path(args.weights).name

    metrics = {
        "model": model_label,
        "n_pos": n_pos, "n_neg": n_neg, "conf": args.conf,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round(accuracy, 4), "precision": round(precision, 4),
        "recall": round(recall, 4), "F1": round(f1, 4),
        "FPR": round(fpr, 4), "FNR": round(fnr, 4),
    }
    if missing:
        metrics["missing_files"] = missing

    print("\n== image-level metrics ==")
    for k, v in metrics.items():
        print(f"  {k:14s}: {v}")

    _write_csv(Path(args.out_csv), [metrics])
    if args.per_image_csv:
        _write_csv(Path(args.per_image_csv), per_image)
    if args.metadata:
        _category_breakdown(Path(args.metadata), per_image, Path(args.out_csv))
    if not args.no_db:
        mapping = sorted(model.names[i] for i in obstruction_ids) if coco_map else None
        _write_db(args, metrics, mapping)


def _category_breakdown(meta_path: Path, per_image: list[dict],
                        out_csv: Path) -> None:
    """Per-category (and cut/whole) recall over the positive images, using the
    generator metadata. An image counts for category c if it contains at least
    one pasted object of c; categories overlap on multi-object images.

    Questo e' cio' che permette di rispondere a "il modello vede meglio le
    sedie a rotelle o i cestini?" invece che solo "il modello vede il 30% delle
    ostruzioni" senza sapere QUALI. Il file objects_metadata.json (scritto da
    generate_synthetic_dataset.py) e' l'unico posto dove sappiamo quale
    categoria di oggetto e' stata incollata in quale immagine: la tabella
    annotations_bbox del DB ha solo le coordinate, non la categoria per ogni
    box, quindi serve rileggere quel file JSON esterno.
    """
    if not meta_path.exists():
        print(f"[warn] metadata not found: {meta_path}")
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))["images"]

    # Mappa nome-file -> 0/1 (rilevata o no), solo per le immagini positive
    # (le negative non hanno oggetti incollati, quindi non hanno categoria).
    pred_of = {r["image"]: r["predicted"] for r in per_image
               if r["is_obstructed"]}

    # "buckets" raggruppa le predizioni (0/1) per categoria: buckets['sedia']
    # e' la lista di predizioni su TUTTE le immagini che contengono almeno una
    # sedia. Un'immagine con 2 oggetti diversi finisce in 2 bucket diversi
    # (da qui "categories overlap" nella docstring) - e' voluto: vogliamo
    # sapere "quanto bene troviamo le sedie", non partizionare le immagini.
    buckets: dict[str, list[int]] = {}
    for name, objs in meta.items():
        if name not in pred_of:
            continue
        cats = {o["category"] for o in objs}   # set: dedup se stessa categoria 2 volte
        for c in cats:
            buckets.setdefault(c, []).append(pred_of[name])
        # bucket aggiuntivo trasversale: l'immagine ha ALMENO un oggetto
        # tagliato dal bordo, o sono tutti interi? Utile per capire se gli
        # oggetti parzialmente visibili sono piu difficili da rilevare.
        has_cut = any(o["cut_edge"] for o in objs)
        buckets.setdefault("with_cut_object" if has_cut else "whole_objects_only",
                           []).append(pred_of[name])

    rows = []
    print("\n== per-category recall (image contains >=1 object of the category) ==")
    for key in sorted(buckets):
        hits = buckets[key]
        rec = sum(hits) / len(hits)   # media delle predizioni 0/1 = recall di quel bucket
        rows.append({"category": key, "n_images": len(hits),
                     "detected": sum(hits), "recall": round(rec, 4)})
        print(f"  {key:20s}: {sum(hits):4d}/{len(hits):4d}  recall={rec:.3f}")

    _write_csv(out_csv.with_name(out_csv.stem + "_per_category.csv"), rows)


# --------------------------------------------------------------------------- #
# open-vocabulary (YOLO-World) zero-shot, with a confidence sweep
# --------------------------------------------------------------------------- #
# YOLO-World non ha classi fisse: invece di un elenco di ID come COCO, gli si
# passano dei PROMPT testuali (set_classes) e lui rileva qualsiasi box che
# somiglia semanticamente a uno di quei concetti. Qui, per ogni categoria
# target, includiamo piu' sinonimi (es. "cart"/"trolley"/"utility cart") per
# aumentare le chance che almeno uno "agganci" l'oggetto nell'immagine.
DEFAULT_OPENVOCAB_PROMPTS = [
    "wheelchair",
    "stretcher", "gurney", "hospital bed",
    "cart", "trolley", "utility cart",
    "trash can", "waste bin",
    "cardboard box", "box",
]


def _metrics_from_counts(tp, fp, tn, fn) -> dict:
    """Stesse formule di image_level() ma fattorizzate a parte, perche' qui
    vanno ricalcolate una volta per ogni soglia di confidenza dello sweep
    (vedi open_vocab sotto), non una volta sola."""
    n_pos, n_neg = tp + fn, tn + fp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / n_pos if n_pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "n_pos": n_pos, "n_neg": n_neg, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": round((tp + tn) / (tp + tn + fp + fn), 4) if (tp+tn+fp+fn) else 0.0,
        "precision": round(prec, 4), "recall": round(rec, 4), "F1": round(f1, 4),
        "FPR": round(fp / n_neg, 4) if n_neg else 0.0,
        "FNR": round(fn / n_pos, 4) if n_pos else 0.0,
    }


def open_vocab(args) -> None:
    """Zero-shot open-vocabulary detection (YOLO-World). Runs inference ONCE at
    the lowest sweep confidence, records the max area-filtered detection
    confidence per image, then re-thresholds for every conf in the sweep. For
    the single-class task any prompt match counts as an obstruction.

    L'IDEA CHIAVE (perche' non rifare l'inferenza per ogni soglia):
    la soglia di confidenza serve solo a scartare le detection "deboli" PRIMA
    di restituirle. Ma se io chiedo al modello conf=0.05 (la soglia PIU'
    BASSA dello sweep), ottengo gia' TUTTE le detection che avrei ottenuto con
    conf=0.10 o conf=0.25 (che sono sottoinsiemi di quelle a 0.05), piu' in
    piu' quelle deboli tra 0.05 e le soglie piu' alte. Percio' mi basta girare
    il modello UNA VOLTA a conf=0.05, salvare per ogni immagine la confidenza
    massima tra le sue detection (max_conf), e poi per ogni soglia t dello
    sweep basta confrontare max_conf >= t: e' identico a rifare l'inferenza a
    conf=t, ma senza il costo di re-inferire (il modello e' lento su CPU).
    """
    dataset_root = Path(args.dataset_root)
    if args.copy_paste:
        samples = samples_copypaste(args.db, dataset_root, Path(args.pruned_file))
    elif args.split:
        samples = samples_from_db(args.db, dataset_root, args.split)
    else:
        raise SystemExit("use --copy-paste or --split")

    prompts = ([p.strip() for p in args.prompts.split(",")]
               if args.prompts else DEFAULT_OPENVOCAB_PROMPTS)
    # set() dedup + sorted(): se l'utente scrive "0.25,0.05,0.25" lo sweep
    # diventa comunque [0.05, 0.25], in ordine crescente per la stampa.
    sweep = sorted({float(x) for x in args.sweep.split(",")})
    min_conf = min(sweep)   # la soglia piu' permissiva: qui gira l'inferenza vera

    model = YOLO(args.weights)
    model.set_classes(prompts)   # imposta il vocabolario testuale (vedi sopra)
    print(f"open-vocab model: {args.weights}")
    print(f"prompts ({len(prompts)}): {prompts}")
    print(f"conf sweep: {sweep}  (inference run once at {min_conf})\n")

    # Passata UNICA sul dataset: per ogni immagine teniamo solo il numero
    # max_conf (la confidenza della detection piu' sicura, tra quelle la cui
    # area supera min_area - lo stesso filtro "escludi oggetti minuscoli"
    # usato per COCO). Se non c'e' nessuna detection valida, max_conf resta 0.
    per_image = []
    missing = 0
    for img, is_pos in samples:
        if not img.exists():
            missing += 1
            continue
        r = model.predict(str(img), conf=min_conf, verbose=False)[0]
        max_conf = 0.0
        if len(r.boxes):
            confs = r.boxes.conf.tolist()
            wh = r.boxes.xywhn[:, 2:4].tolist()
            for cf, (w, h) in zip(confs, wh):
                if w * h >= args.min_area:
                    max_conf = max(max_conf, cf)
        per_image.append({"image": img.name, "is_obstructed": is_pos,
                          "max_conf": round(max_conf, 4)})

    # Ora, PER OGNI soglia dello sweep, ricalcoliamo TP/FP/TN/FN da zero
    # ri-usando gli stessi max_conf gia' calcolati (nessuna nuova inferenza):
    # un'immagine e' "predetta ostruita a soglia t" se max_conf >= t.
    # Alzando t, ci si aspetta: recall giu' (piu' selettivi, si perdono
    # detection deboli ma vere), precision su (si scartano i falsi allarmi
    # deboli) - il classico compromesso precision/recall.
    rows = []
    print("== conf sweep (open-vocab) ==")
    print(f"{'conf':>5s} {'recall':>7s} {'precision':>9s} {'F1':>6s} {'FPR':>6s} {'FNR':>6s}")
    for t in sweep:
        tp = fp = tn = fn = 0
        for r in per_image:
            pred = r["max_conf"] >= t
            if r["is_obstructed"] and pred:
                tp += 1
            elif r["is_obstructed"]:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        m = _metrics_from_counts(tp, fp, tn, fn)
        m = {"conf": t, **m}
        rows.append(m)
        print(f"{t:5.2f} {m['recall']:7.3f} {m['precision']:9.3f} "
              f"{m['F1']:6.3f} {m['FPR']:6.3f} {m['FNR']:6.3f}")

    _write_csv(Path(args.out_csv), rows)
    if args.per_image_csv:
        _write_csv(Path(args.per_image_csv), per_image)
    if missing:
        print(f"[warn] {missing} missing image files")
    if not args.no_db:
        _write_db_openvocab(args, prompts, rows)


def _write_db_openvocab(args, prompts, rows) -> None:
    """Salva la sweep sul DB: UN esperimento (experiments) con i prompt e i
    parametri usati, e PIU' righe in results (una per soglia dello sweep,
    cosi' si puo' confrontare come cambiano le metriche al variare della
    soglia senza dover rilanciare nulla). anomaly_threshold e' la colonna
    generica gia' usata da P3 per lo stesso scopo (soglia di score); qui la
    riusiamo per la soglia di confidenza YOLO-World."""
    dataset = "copy-paste" if args.copy_paste else args.split
    conn = sqlite3.connect(args.db)
    cur = conn.execute(
        "INSERT INTO experiments (pipeline, model_variant, dataset_filter, "
        "hyperparams, status) VALUES ('P1', ?, ?, ?, 'done')",
        (f"open-vocab:{Path(args.weights).stem}",
         json.dumps({"dataset": dataset,
                     "n_pos": rows[0]["n_pos"], "n_neg": rows[0]["n_neg"]}),
         json.dumps({"prompts": prompts, "min_area": args.min_area,
                     "sweep": [r["conf"] for r in rows]})),
    )
    exp_id = cur.lastrowid
    for m in rows:  # one results row per confidence threshold
        conn.execute(
            "INSERT INTO results (exp_id, precision, recall, F1, "
            "anomaly_threshold, false_alarm_rate, false_negative_rate) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (exp_id, m["precision"], m["recall"], m["F1"], m["conf"],
             m["FPR"], m["FNR"]),
        )
    conn.commit()
    conn.close()
    print(f"\nwrote experiment {exp_id} (P1, open-vocab, {len(rows)} thresholds) to DB")


# --------------------------------------------------------------------------- #
# cross-validation detection mAP
# --------------------------------------------------------------------------- #
def cv_map(args) -> None:
    """Aggrega il mAP di DETECTION (non image-level) sui 5 modelli allenati in
    cross validation da train_yolo.py. Serve SOLO per il piano B (fine tuning);
    con modelli frozen (COCO, YOLO-World) questo comando non si applica, perche'
    non esiste un 'best.pt per fold' - v. discussione nel todo sul perche' i
    fold non servono per valutare modelli non allenati."""
    runs_dir = Path(args.runs_dir)
    folds_dir = Path(args.folds_dir)
    fold_yamls = sorted(folds_dir.glob("fold*.yaml"))
    if not fold_yamls:
        raise SystemExit(f"no fold*.yaml in {folds_dir}")

    per_fold = []
    for y in fold_yamls:
        k = y.stem.replace("fold", "")
        # train_yolo.py salva i pesi in runs/<name>_fold<k>/weights/best.pt:
        # qui ricostruiamo lo stesso path per ritrovare il modello di quel fold.
        best = runs_dir / f"{args.name}_fold{k}" / "weights" / "best.pt"
        if not best.exists():
            print(f"[skip] fold {k}: missing {best}")
            continue
        model = YOLO(str(best))
        # model.val() e' la validazione DETECTION nativa di ultralytics: valuta
        # il modello sul suo stesso val set (definito nel data.yaml del fold)
        # e calcola mAP/precision/recall SULLE BOUNDING BOX (non a livello di
        # immagine come image_level() sopra - qui conta anche la posizione,
        # non solo "c'e' un oggetto si/no").
        res = model.val(data=str(y.resolve()), split="val", verbose=False)
        row = {
            "fold": k,
            "mAP50": round(res.box.map50, 4),      # mAP a soglia IoU 0.5
            "mAP50_95": round(res.box.map, 4),     # mAP mediato su IoU 0.5-0.95 (piu' severo)
            "precision": round(res.box.mp, 4),
            "recall": round(res.box.mr, 4),
        }
        per_fold.append(row)
        print(f"  fold {k}: mAP50={row['mAP50']}  mAP50-95={row['mAP50_95']}  "
              f"P={row['precision']}  R={row['recall']}")

    if not per_fold:
        raise SystemExit("no fold weights found - train first")

    # Media e deviazione standard sui fold: la deviazione standard e' cio' che
    # dice "quanto e' stabile la stima" - se e' alta, un singolo fold da solo
    # sarebbe stato fuorviante, ed e' proprio il motivo per cui si fa CV
    # invece di un solo split train/val.
    def mean(key):
        return sum(r[key] for r in per_fold) / len(per_fold)

    def std(key):
        m = mean(key)
        return (sum((r[key] - m) ** 2 for r in per_fold) / len(per_fold)) ** 0.5

    summary = {"fold": "mean+-std"}
    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        summary[key] = f"{mean(key):.4f}+-{std(key):.4f}"
    per_fold.append(summary)
    print(f"\n== CV mAP over {len(per_fold) - 1} folds ==")
    for key in ("mAP50", "mAP50_95", "precision", "recall"):
        print(f"  {key:10s}: {summary[key]}")

    _write_csv(Path(args.out_csv), per_fold)


# --------------------------------------------------------------------------- #
# output helpers
# --------------------------------------------------------------------------- #
def _write_csv(path: Path, rows: list[dict]) -> None:
    """Scrive una lista di dict come CSV. Le colonne (keys) sono raccolte
    scorrendo TUTTE le righe, non solo la prima: nel caso di image_level(),
    ad esempio, la riga della baseline COCO ha anche una chiave 'missing_files'
    che le altre righe non hanno, e questo evita un KeyError o colonne perse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path}")


def _write_db(args, metrics: dict, mapping: list[str] | None) -> None:
    """Persist an image-level run to experiments/results (pipeline='P1').

    The full metric dict (accuracy, counts, ...) and the COCO->obstruction
    class list are stored in experiments.hyperparams so nothing is lost, even
    though the results table only has columns for the shared headline metrics.
    """
    dataset = "copy-paste" if args.copy_paste else (args.split or "dirs")
    conn = sqlite3.connect(args.db)
    cur = conn.execute(
        "INSERT INTO experiments (pipeline, model_variant, dataset_filter, "
        "hyperparams, status) VALUES ('P1', ?, ?, ?, 'done')",
        (f"image-level:{metrics['model']}",
         json.dumps({"dataset": dataset,
                     "n_pos": metrics["n_pos"], "n_neg": metrics["n_neg"]}),
         json.dumps({"conf": args.conf, "min_area": args.min_area,
                     "mapping": mapping, "metrics": metrics})),
    )
    exp_id = cur.lastrowid
    conn.execute(
        "INSERT INTO results (exp_id, precision, recall, F1, "
        "false_alarm_rate, false_negative_rate) VALUES (?, ?, ?, ?, ?, ?)",
        (exp_id, metrics["precision"], metrics["recall"], metrics["F1"],
         metrics["FPR"], metrics["FNR"]),
    )
    conn.commit()
    conn.close()
    print(f"wrote experiment {exp_id} (P1, image-level) to DB")


# --------------------------------------------------------------------------- #
# CLI: tre sotto-comandi (image-level / cv-map / open-vocab) che condividono
# molti flag ma hanno funzioni target diverse (args.func, impostata da
# set_defaults). argparse chiama automaticamente args.func(args) in fondo.
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("image-level", help="binary obstructed/clear metrics")
    a.add_argument("--weights", required=True,
                   help="path to .pt, or 'coco' for the mapped yolo11n baseline")
    a.add_argument("--coco-map", action="store_true",
                   help="apply the COCO->obstruction class mapping to --weights "
                        "(for any other COCO-pretrained checkpoint, e.g. yolo26n.pt)")
    a.add_argument("--db", default="Aggregated_dataset_db/occlusion.db")
    a.add_argument("--dataset-root", default="Dataset")
    a.add_argument("--split", default=None, help="DB split, e.g. 'test'")
    a.add_argument("--copy-paste", action="store_true",
                   help="evaluate on the copy-paste set (990 pos + 110 pruned neg)")
    a.add_argument("--pruned-file",
                   default="Dataset/ostruzioni_reali/pruned_backgrounds.txt")
    a.add_argument("--pos-dir", default=None)
    a.add_argument("--neg-dir", default=None)
    a.add_argument("--conf", type=float, default=0.25)
    a.add_argument("--min-area", type=float, default=MIN_BOX_AREA_FRAC)
    a.add_argument("--out-csv", default="pipeline1/results/image_level.csv")
    a.add_argument("--per-image-csv", default=None,
                   help="also dump one row per image (path, label, prediction)")
    a.add_argument("--metadata", default=None,
                   help="objects_metadata.json of the copy-paste generator: "
                        "adds per-category recall breakdown")
    a.add_argument("--no-db", action="store_true",
                   help="skip writing the run to experiments/results")
    a.set_defaults(func=image_level)

    b = sub.add_parser("cv-map", help="cross-validation detection mAP")
    b.add_argument("--runs-dir", default="pipeline1/runs")
    b.add_argument("--name", default="cv")
    b.add_argument("--folds-dir", default="pipeline1/data")
    b.add_argument("--out-csv", default="pipeline1/results/cv_map.csv")
    b.set_defaults(func=cv_map)

    c = sub.add_parser("open-vocab",
                       help="YOLO-World zero-shot, obstructed/clear, conf sweep")
    c.add_argument("--weights", default="yolov8s-world.pt",
                   help="YOLO-World checkpoint")
    c.add_argument("--db", default="Aggregated_dataset_db/occlusion.db")
    c.add_argument("--dataset-root", default="Dataset")
    c.add_argument("--split", default=None, help="DB split, e.g. 'test'")
    c.add_argument("--copy-paste", action="store_true",
                   help="evaluate on the copy-paste set (990 pos + 110 pruned neg)")
    c.add_argument("--pruned-file",
                   default="Dataset/ostruzioni_reali/pruned_backgrounds.txt")
    c.add_argument("--prompts", default=None,
                   help="comma-separated prompts (default: extended hospital set)")
    c.add_argument("--sweep", default="0.05,0.10,0.25",
                   help="comma-separated confidence thresholds")
    c.add_argument("--min-area", type=float, default=MIN_BOX_AREA_FRAC)
    c.add_argument("--out-csv", default="pipeline1/results/openvocab_sweep.csv")
    c.add_argument("--per-image-csv", default=None)
    c.add_argument("--no-db", action="store_true")
    c.set_defaults(func=open_vocab)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
