"""
PatchCore — Anomaly Detection
====================================================================
Uso:
  1. BUILD memory bank da una singola immagine di riferimento:
       python anomaly_detection_patch_core.py build --ref porta.jpg --bank memory_bank.pt

  2. TEST su un'immagine (con o senza occlusione):
       python anomaly_detection_patch_core.py test --bank memory_bank.pt --img test.jpg

  3. TEST con ROI (opzionale): usando --roi "x,y,w,h" in pixel
       python anomaly_detection_patch_core.py test --bank memory_bank.pt --img test.jpg --roi "100,50,300,400"

Dipendenze: torch torchvision pillow numpy scikit-learn opencv-python
"""

# TODO: capire se adattare il re- weighting come nel paper o lasciare così com'è
# TODO: rimandare test su vm, salvare csv e rivedere la tabella results del db
# TODO: capire come attuare la visualizzazione di heatmap su i vari test (media di heatmap (?) o heatmap di un test specifico (es. quello con occlusione più evidente)
import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from sklearn.random_projection import SparseRandomProjection


# ── Configurazione globale ──────────────────────────────────────────────────

IMG_SIZE   = 224          # dimensione input al backbone
PATCH_SIZE = 3            # neighbourhood aggregation (paper: p=3)
LAYERS     = [2, 3]       # livelli intermedi WideResNet50 (paper: j=2,3)
CORESET_P  = 0.01         # fraction of the memory bank retained after coreset reduction (1%)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT_DIR / "Dataset"
DEFAULT_DB_PATH = ROOT_DIR / "Aggregated_dataset_db" / "occlusion.db"


# ── Augmentation fotometrica ────────────────────────────────────────────────

def augment_reference(img_pil: Image.Image, n: int = 15,
                      seed: int = 42,
                      include_original: bool = True) -> list[Image.Image]:
    """
    Genera n varianti fotometriche di una singola immagine.
    Varia: luminosità, contrasto, saturazione, temperatura colore,
           rumore gaussiano, blur leggero.
    NON applica flip/rotazioni forti: la camera è fissa.

    Args:
        seed: seme RNG. Usare semi diversi per bank vs calibrazione vs test
              in modo da garantire varianti statisticamente indipendenti.
        include_original: se True, l'originale è incluso come primo elemento
                          e vengono generate n-1 augmentazioni (totale n).
                          Se False, vengono generate n augmentazioni pure
                          senza l'originale — usare in evaluate_from_db per
                          riservare ref_img come campione di test indipendente.
    """
    variants = [img_pil] if include_original else []

    rng = np.random.default_rng(seed)

    for _ in range(n - 1 if include_original else n):
        img = img_pil.copy()

        # Luminosità: simula mattina/sera/notte/luce artificiale
        brightness = rng.uniform(0.85, 1.15)
        img = ImageEnhance.Brightness(img).enhance(brightness)

        # Contrasto: luce diretta vs diffusa
        contrast = rng.uniform(0.90, 1.10)
        img = ImageEnhance.Contrast(img).enhance(contrast)

        # Saturazione: led vs incandescente vs fluorescente
        saturation = rng.uniform(0.90, 1.10)
        img = ImageEnhance.Color(img).enhance(saturation)

        # Temperatura colore (shift canali R/B)
        arr = np.array(img).astype(np.float32)
        r_shift = rng.uniform(-8, 8)
        b_shift = rng.uniform(-20, 20)
        arr[:, :, 0] = np.clip(arr[:, :, 0] + r_shift, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] + b_shift, 0, 255)

        # Rumore gaussiano leggero
        noise = rng.normal(0, rng.uniform(1, 4), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

        # Blur occasionale (es. camera leggermente sfocata)
        if rng.random() < 0.3:
            radius = rng.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        variants.append(img)

    return variants


# ── Backbone e estrazione feature ───────────────────────────────────────────

class FeatureExtractor(nn.Module):
    """
    WideResNet50-2 pre-addestrato su ImageNet.
    Estrae feature map dai livelli intermedi (layer2, layer3).
    """
    def __init__(self):
        super().__init__()
        try:
            weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1
            net = wide_resnet50_2(weights=weights)
            print("      Backbone: WideResNet50-2 (ImageNet pretrained) ")
        except Exception:
            print("         Download ImageNet weights non riuscito.")
            print("         Uso backbone random (solo per test strutturale).")
            print("         Sulla tua macchina con accesso internet funzionerà.")
            net = wide_resnet50_2(weights=None)
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        # Non usiamo layer4 (troppo abstract / ImageNet-biased)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        f2 = self.layer2(x)   # feature map livello 2
        f3 = self.layer3(f2)  # feature map livello 3
        return f2, f3


def extract_patch_features(model: FeatureExtractor,
                            imgs_tensor: torch.Tensor,
                            patch_size: int = PATCH_SIZE) -> torch.Tensor:
    """
    Estrae locally-aware patch features da una batch di immagini.
    - Prende f2 e f3, upscala f3 alla risoluzione di f2
    - Applica average pooling locale (neighbourhood aggregation)
    - Restituisce tensore [N_patches_totali, D]
    """
    with torch.no_grad():
        f2, f3 = model(imgs_tensor.to(DEVICE))

        # Upsample f3 to f2 spatial resolution before concatenation
        f3_up = nn.functional.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )

        # Concatenate along the channel dimension
        feat = torch.cat([f2, f3_up], dim=1)  # [B, C2+C3, H, W]

        # Local neighbourhood aggregation via average pooling (paper eq. 2, p=3).
        # padding=pad keeps the spatial resolution unchanged (same-padding).
        if patch_size > 1:
            pad = patch_size // 2
            feat = nn.functional.avg_pool2d(
                feat, kernel_size=patch_size, stride=1, padding=pad
            )

        # Flatten spatial dimensions: [B, C, H, W] -> [B*H*W, C]
        B, C, H, W = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(-1, C)

    return feat.cpu(), H, W


# ── Trasformazione immagini ─────────────────────────────────────────────────

preprocess = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])


def load_image_tensor(path_or_pil) -> torch.Tensor:
    if isinstance(path_or_pil, (str, Path)):
        img = Image.open(path_or_pil).convert("RGB")
    else:
        img = path_or_pil.convert("RGB")
    return preprocess(img).unsqueeze(0)  # [1, 3, H, W]


def parse_roi(roi: str | None) -> tuple[int, int, int, int] | None:
    if not roi:
        return None
    parts = roi.split(",")
    if len(parts) != 4:
        raise ValueError("ROI must be in format x,y,w,h")
    x, y, w, h = (int(p) for p in parts)
    return x, y, w, h


def compute_image_score_from_pil(
    model: FeatureExtractor,
    img_pil: Image.Image,
    memory_bank: torch.Tensor,
    roi: tuple[int, int, int, int] | None,
) -> float:
    orig_w, orig_h = img_pil.size
    tensor = load_image_tensor(img_pil)
    test_feats, H_t, W_t = extract_patch_features(model, tensor)
    scores = compute_patch_scores(test_feats, memory_bank)
    anomaly_map = scores.reshape(H_t, W_t).numpy()
    anomaly_map_full = cv2.resize(
        anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )
    anomaly_map_smooth = cv2.GaussianBlur(anomaly_map_full, (0, 0), sigmaX=4)

    if roi:
        x, y, w, h = roi
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(orig_w, x + w)
        y1 = min(orig_h, y + h)
        roi_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
        if x1 > x0 and y1 > y0:
            roi_mask[y0:y1, x0:x1] = 1
        anomaly_map_smooth = anomaly_map_smooth * roi_mask

    return float(anomaly_map_smooth.max())


@dataclass(frozen=True)
class DBFrame:
    frame_id: str
    file_path: str
    venue_type: str
    split: str
    source: str
    is_normal: int
    reference_frame_id: str | None


def chunked(items: list[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def resolve_dataset_path(dataset_root: Path, file_path: str) -> Path:
    return dataset_root / Path(file_path)


def query_reference_frames(
    conn: sqlite3.Connection,
    split: str | None,
    venue: str | None,
    ref_source: str | None,
) -> list[DBFrame]:
    sql = (
        "SELECT f.frame_id, f.file_path, f.venue_type, f.split, f.source, "
        "f.is_normal, f.reference_frame_id "
        "FROM frames f "
        "WHERE f.is_normal = 1 "
        "AND EXISTS ("
        "  SELECT 1 FROM frames o "
        "  WHERE o.is_normal = 0 AND o.reference_frame_id = f.frame_id"
        ")"
    )
    params: list[str] = []
    if split:
        sql += " AND f.split = ?"
        params.append(split)
    if venue:
        sql += " AND f.venue_type = ?"
        params.append(venue)
    if ref_source:
        sql += " AND f.source = ?"
        params.append(ref_source)
    sql += " ORDER BY f.venue_type, f.frame_id"

    rows = conn.execute(sql, params).fetchall()
    return [
        DBFrame(
            frame_id=row[0],
            file_path=row[1],
            venue_type=row[2],
            split=row[3],
            source=row[4],
            is_normal=row[5],
            reference_frame_id=row[6],
        )
        for row in rows
    ]


def query_obstructed_frames(
    conn: sqlite3.Connection,
    ref_ids: list[str],
    split: str | None,
    ob_source: str | None,
) -> dict[str, list[DBFrame]]:
    if not ref_ids:
        return {}

    mapping: dict[str, list[DBFrame]] = {}
    for chunk in chunked(ref_ids, 900):
        placeholders = ",".join("?" for _ in chunk)
        sql = (
            "SELECT frame_id, file_path, venue_type, split, source, is_normal, "
            "reference_frame_id "
            "FROM frames "
            "WHERE is_normal = 0 "
            "AND reference_frame_id IN ("
            + placeholders
            + ")"
        )
        params: list[str] = list(chunk)
        if split:
            sql += " AND split = ?"
            params.append(split)
        if ob_source:
            sql += " AND source = ?"
            params.append(ob_source)

        rows = conn.execute(sql, params).fetchall()
        for row in rows:
            frame = DBFrame(
                frame_id=row[0],
                file_path=row[1],
                venue_type=row[2],
                split=row[3],
                source=row[4],
                is_normal=row[5],
                reference_frame_id=row[6],
            )
            if frame.reference_frame_id is None:
                continue
            mapping.setdefault(frame.reference_frame_id, []).append(frame)

    return mapping


# ── Calcolo anomaly score ───────────────────────────────────────────────

def compute_patch_scores(test_feats: torch.Tensor,
                         memory_bank: torch.Tensor) -> torch.Tensor:
    """
    Calcola l'anomaly score per ogni patch, usando kNN con re-weighting (eq. 7 paper).
    Restituisce un tensore [H*W] di score.
    """
    # The paper uses raw Euclidean distances on unnormalised features.
    # L2-normalising would project everything onto the unit sphere, turning
    # the distance into a cosine-based measure, which is not what eq. 6/7 assume.
    batch_size = 256
    scores = []
    for i in range(0, len(test_feats), batch_size):
        batch = test_feats[i:i+batch_size]                        # [b, D]
        dists = torch.cdist(batch.unsqueeze(0),
                            memory_bank.unsqueeze(0)).squeeze(0)  # [b, N]
        min_dists, nn_idx = dists.min(dim=1)                      # [b], [b]

        k_neighbors = min(9, len(memory_bank))

        # Re-weighting as in paper eq. 7:
        #   s = (1 - exp(||m_test - m*||) /
        #            sum_{m in Nb(m*)} exp(||m_test - m||)) * s*
        #
        # Nb(m*) = k nearest neighbours of m* inside the memory bank.
        # The distances in the denominator are between the TEST patch and
        # each member of Nb(m*), NOT between m* and its neighbours.
        # The previous implementation used ||m* - m|| in the denominator,
        # which mixed two different distance spaces.

        # Step 1: find Nb(m*) -- k nearest neighbours of m* in the bank.
        nn_feats = memory_bank[nn_idx]                            # [b, D]
        bank_dists_from_nn = torch.cdist(
            nn_feats.unsqueeze(0), memory_bank.unsqueeze(0)
        ).squeeze(0)                                              # [b, N]
        topk_nb_idx = bank_dists_from_nn.topk(
            k_neighbors, dim=1, largest=False
        ).indices                                                 # [b, k]

        # Step 2: distances from each test patch to the members of Nb(m*).
        nb_feats = memory_bank[topk_nb_idx]                      # [b, k, D]
        dists_test_to_nb = (
            batch.unsqueeze(1) - nb_feats                        # [b, k, D]
        ).norm(dim=2)                                            # [b, k]

        # Step 3: compute the softmax weight and scale the raw distance.
        # Shifted exponentials for numerical stability.
        all_exp_args = torch.cat([min_dists.unsqueeze(1), dists_test_to_nb], dim=1)
        shift = all_exp_args.max(dim=1, keepdim=True).values     # [b, 1]
        exp_min = torch.exp(min_dists - shift.squeeze(1))        # [b]
        exp_sum = torch.exp(dists_test_to_nb - shift).sum(dim=1) # [b]
        w = 1.0 - exp_min / (exp_sum + 1e-8)
        weighted_scores = w * min_dists
        scores.append(weighted_scores)

    return torch.cat(scores)


# ── Coreset subsampling ─────────────────────────────────────────────────────

def greedy_coreset(features: np.ndarray, target_size: int) -> np.ndarray:
    """
    Greedy minimax facility location coreset.
    Seleziona target_size punti che massimizzano la copertura dello spazio.
    Usa proiezione casuale per efficienza (Johnson-Lindenstrauss).
    """
    print(f"  Coreset: {len(features)} → {target_size} punti...")

    # Proiezione casuale per ridurre dimensionalità (JL theorem)
    proj_dim = min(128, features.shape[1])
    projector = SparseRandomProjection(n_components=proj_dim, random_state=42)
    features_proj = projector.fit_transform(features)

    # Greedy selection
    selected = [0]
    # Distanza minima di ogni punto dal coreset corrente
    min_dists = np.full(len(features_proj), np.inf)

    for _ in range(target_size - 1):
        last = features_proj[selected[-1]]
        dists = np.linalg.norm(features_proj - last, axis=1)
        min_dists = np.minimum(min_dists, dists)
        selected.append(int(np.argmax(min_dists)))

    return features[np.array(selected)]


# ── BUILD: costruzione memory bank ──────────────────────────────────────────

def build_memory_bank(ref_path: str, bank_path: str, n_augments: int = 15,
                      n_cal: int = 10, coreset_p: float = CORESET_P):
    """
    Costruisce la memory bank da una singola immagine di riferimento.

    FIX anti data-leakage: le varianti per la calibrazione di mu/sigma
    sono generate con un seed separato (seed=1000) e NON entrano nella
    memory bank. Questo garantisce che la soglia mu+k*sigma sia calibrata
    su campioni davvero unseen, simulando il comportamento in produzione.

    Args:
        n_augments : varianti usate per costruire la bank  (seed=42)
        n_cal      : varianti per calibrare mu/sigma       (seed=1000)
        coreset_p  : frazione del bank da mantenere dopo coreset
    """
    print(f"\n{'='*60}")
    print(f"BUILD memory bank")
    print(f"  Immagine di riferimento : {ref_path}")
    print(f"  Augmentazioni bank      : {n_augments}  (seed=42)")
    print(f"  Augmentazioni cal.      : {n_cal}  (seed=1000, unseen dalla bank)")
    print(f"  Coreset fraction        : {coreset_p*100:.1f}%")
    print(f"  Salvataggio in          : {bank_path}")
    print(f"  Device                  : {DEVICE}")
    print(f"{'='*60}\n")

    ref_img = Image.open(ref_path).convert("RGB")

    # 1a. Varianti per la memory bank (seed=42)
    bank_variants = augment_reference(ref_img, n=n_augments, seed=42)
    print(f"[1/5] Augmentation bank : {len(bank_variants)} immagini (seed=42)")

    # 1b. Varianti per calibrazione soglia (seed=1000 → statisticamente indipendenti)
    #     NON entrano nella bank: simulano immagini normali future mai viste.
    cal_variants = augment_reference(ref_img, n=n_cal, seed=1000)
    print(f"      Augmentation cal.  : {len(cal_variants)} immagini (seed=1000)")

    # Salva preview
    preview_dir = Path(bank_path).parent / "augmentation_preview"
    preview_dir.mkdir(exist_ok=True)
    for i, v in enumerate(bank_variants):
        v.save(preview_dir / f"bank_variant_{i:02d}.jpg")
    for i, v in enumerate(cal_variants):
        v.save(preview_dir / f"cal_variant_{i:02d}.jpg")
    print(f"      Preview salvate in: {preview_dir}/")

    # 2. Estrazione feature (solo bank_variants)
    print(f"\n[2/5] Estrazione feature con WideResNet50...")
    model = FeatureExtractor().to(DEVICE).eval()

    all_features = []
    for i, img in enumerate(bank_variants):
        tensor = load_image_tensor(img)
        feats, H, W = extract_patch_features(model, tensor)
        all_features.append(feats)
        print(f"      Variante {i+1:02d}/{len(bank_variants)}: "
              f"{feats.shape[0]} patch × {feats.shape[1]}d")

    memory_bank = torch.cat(all_features, dim=0)  # [N_total_patches, D]
    print(f"\n      Memory bank raw: {memory_bank.shape}")

    # 3. Coreset subsampling
    print(f"\n[3/5] Coreset subsampling ({coreset_p*100:.1f}%)...")
    target_size = max(50, int(len(memory_bank) * coreset_p))
    bank_np = memory_bank.numpy()
    bank_coreset = greedy_coreset(bank_np, target_size)
    bank_coreset = torch.from_numpy(bank_coreset)
    print(f"      Memory bank finale: {bank_coreset.shape}")

    # 4. Calibrazione soglia su cal_variants (UNSEEN dalla bank → no leakage)
    print(f"\n[4/5] Calibrazione soglia su {len(cal_variants)} immagini unseen...")
    cal_scores = []
    for i, img in enumerate(cal_variants):
        tensor = load_image_tensor(img)
        feats, _, _ = extract_patch_features(model, tensor)
        patch_scores = compute_patch_scores(feats, bank_coreset)
        max_score = float(patch_scores.max())
        cal_scores.append(max_score)
        print(f"      Cal. variante {i+1:02d}/{len(cal_variants)}: max score = {max_score:.4f}")

    cal_scores = np.array(cal_scores)
    cal_mean = float(cal_scores.mean())
    cal_std  = float(cal_scores.std())
    print(f"\n      Distribuzione score (unseen normals):")
    print(f"        Media (mu)             = {cal_mean:.4f}")
    print(f"        Std   (sigma)          = {cal_std:.4f}")
    print(f"        Soglia mu+2*sigma      = {cal_mean + 2*cal_std:.4f}")
    print(f"        Soglia mu+3*sigma      = {cal_mean + 3*cal_std:.4f}")

    # 5. Salvataggio
    print(f"\n[5/5] Salvataggio memory bank...")
    torch.save({
        "memory_bank": bank_coreset,
        "patch_hw": (H, W),
        "img_size": IMG_SIZE,
        "ref_path": str(ref_path),
        "n_augments": n_augments,
        "n_cal": n_cal,
        "coreset_p": coreset_p,
        "cal_mean": cal_mean,
        "cal_std": cal_std,
        "cal_scores": cal_scores.tolist(),   # utile per diagnosi
    }, bank_path)
    print(f"      Salvato: {bank_path}")
    print(f"\n Memory bank pronta! Soglia calibrata: mu={cal_mean:.4f}, sigma={cal_std:.4f}")


# ── TEST: inferenza su immagine di test ─────────────────────────────────────

def test_image(bank_path: str, img_path: str, roi: str = None,
               threshold: float = None, k_sigma: float = 3.0):
    print(f"\n{'='*60}")
    print(f"TEST anomaly detection")
    print(f"  Memory bank : {bank_path}")
    print(f"  Immagine    : {img_path}")
    print(f"  ROI         : {roi or 'intera immagine'}")
    print(f"{'='*60}\n")

    # 1. Carica memory bank
    try:
        checkpoint = torch.load(bank_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(bank_path, map_location="cpu")
    memory_bank = checkpoint["memory_bank"]  # [N, D]
    H_feat, W_feat = checkpoint["patch_hw"]
    print(f"[1/4] Memory bank caricata: {memory_bank.shape}")

    # 2. Estrai feature dall'immagine di test
    print(f"[2/4] Estrazione feature immagine di test...")
    model = FeatureExtractor().to(DEVICE).eval()
    test_img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = test_img.size

    tensor = load_image_tensor(test_img)
    test_feats, H_t, W_t = extract_patch_features(model, tensor)
    print(f"      Feature test: {test_feats.shape}")

    # 3. Calcolo anomaly score per patch
    print(f"[3/4] Calcolo anomaly score (kNN)...")
    scores = compute_patch_scores(test_feats, memory_bank)

    # 4. Costruzione anomaly map
    print(f"[4/4] Costruzione heatmap e risultati...")
    anomaly_map = scores.reshape(H_t, W_t).numpy()

    # Upscala heatmap alla risoluzione originale
    anomaly_map_full = cv2.resize(
        anomaly_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
    )

    # Gaussian smoothing (paper: sigma=4)
    anomaly_map_smooth = cv2.GaussianBlur(anomaly_map_full, (0, 0), sigmaX=4)

    # ── Applica ROI se specificata ─────────────────────────────────────────
    roi_mask = np.ones((orig_h, orig_w), dtype=np.float32)
    if roi:
        x, y, w, h = map(int, roi.split(","))
        roi_mask[:] = 0
        roi_mask[y:y+h, x:x+w] = 1
        print(f"      ROI applicata: x={x}, y={y}, w={w}, h={h}")

    anomaly_map_roi = anomaly_map_smooth * roi_mask

    # Score immagine = max score nella ROI
    image_score = float(anomaly_map_roi.max())

    # Soglia calibrata: mu + k*sigma dalle immagini di riferimento
    if threshold is None:
        cal_mean = checkpoint.get("cal_mean")
        cal_std = checkpoint.get("cal_std")
        if cal_mean is not None and cal_std is not None:
            threshold = cal_mean + k_sigma * cal_std
            print(f"      Soglia calibrata: mu({cal_mean:.4f}) + {k_sigma}*sigma({cal_std:.4f}) = {threshold:.4f}")
        else:
            # Fallback per memory bank vecchie senza calibrazione
            threshold = float(np.percentile(anomaly_map_smooth[roi_mask > 0], 95))
            print(f"      Memory bank senza calibrazione, uso 95 percentile: {threshold:.4f}")

    is_anomaly = image_score > threshold

    print(f"\n{'─'*40}")
    print(f"  Anomaly score (max in ROI) : {image_score:.4f}")
    print(f"  Soglia                     : {threshold:.4f}")
    print(f"  Esito                      : {'  ANOMALIA RILEVATA' if is_anomaly else '✅ SCENA NORMALE'}")
    print(f"{'─'*40}\n")

    # ── Salva visualizzazioni ──────────────────────────────────────────────
    out_dir = Path(img_path).parent / "patchcore_output"
    out_dir.mkdir(exist_ok=True)
    stem = Path(img_path).stem

    # Heatmap colorata
    norm_map = (anomaly_map_smooth - anomaly_map_smooth.min())
    norm_map = norm_map / (norm_map.max() + 1e-8) * 255
    heatmap_color = cv2.applyColorMap(norm_map.astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay su immagine originale
    orig_pil = Image.open(img_path).convert("RGB")
    orig_cv = cv2.cvtColor(np.array(orig_pil), cv2.COLOR_RGB2BGR)
    orig_cv = cv2.resize(orig_cv, (orig_w, orig_h))
    overlay = cv2.addWeighted(orig_cv, 0.5, heatmap_color, 0.5, 0)

    # Disegna ROI sul risultato
    if roi:
        x, y, w, h = map(int, roi.split(","))
        color = (0, 0, 255) if is_anomaly else (0, 255, 0)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 3)
        label = f"ANOMALIA ({image_score:.3f})" if is_anomaly else f"OK ({image_score:.3f})"
        cv2.putText(overlay, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Salva (usiamo PIL invece di cv2.imwrite per supportare percorsi con caratteri accentati su Windows)
    heatmap_path = out_dir / f"{stem}_heatmap.jpg"
    overlay_path = out_dir / f"{stem}_overlay.jpg"
    Image.fromarray(cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)).save(heatmap_path)
    Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)).save(overlay_path)

    print(f"  Output salvato in: {out_dir}/")
    print(f"    - {stem}_heatmap.jpg  (mappa anomalia)")
    print(f"    - {stem}_overlay.jpg  (overlay su originale)")

    return image_score, is_anomaly, threshold



# ── EVALUATE: test sistematico su dataset di background ─────────────────────

def evaluate(bg_dir: str, obstr_dir: str = None, roi: str = None,
             k_sigma: float = 3.0, n_augments: int = 15, n_cal: int = 10,
             coreset_p: float = CORESET_P, test_aug_seeds: list = None,
             out_csv: str = "eval_results.csv"):
    """
    Valutazione sistematica di PatchCore su un dataset di immagini background.

    Per ogni immagine di background i:
      1. Costruisce la memory bank (leave-one-out: usa TUTTE le altre + aug)
         → nella pratica, build individuale per ogni immagine (più realistico
           per il caso "una camera = una bank")
      2. Testa K augmented versions di i con seed diversi → FPR (atteso: 0)
      3. Se fornita, testa la corrispondente immagine ostruita → TPR (atteso: 1)

    Metriche finali:
      - FPR  @ soglia mu+k*sigma
      - TPR  @ soglia mu+k*sigma (se obstr_dir fornita)
      - Distribuzione score normali e anomali
      - AUROC (se obstr_dir fornita)

    Args:
        bg_dir        : directory con immagini background (.jpg/.png)
        obstr_dir     : directory con immagini ostruite (stesso nome file del bg)
        roi           : ROI opzionale "x,y,w,h"
        k_sigma       : moltiplicatore soglia (default 3.0)
        n_augments    : varianti per la memory bank
        n_cal         : varianti per calibrazione soglia (unseen)
        coreset_p     : frazione coreset
        test_aug_seeds: lista di seed per le augmented test images (default: [2000,2001,2002])
        out_csv       : path CSV con risultati per immagine
    """
    import csv
    import json

    if test_aug_seeds is None:
        test_aug_seeds = [2000, 2001, 2002]

    bg_paths = sorted(Path(bg_dir).glob("*.jpg")) + sorted(Path(bg_dir).glob("*.png"))
    if not bg_paths:
        print(f"ERRORE: nessuna immagine trovata in {bg_dir}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"EVALUATE — valutazione sistematica PatchCore")
    print(f"  Background dir  : {bg_dir}  ({len(bg_paths)} immagini)")
    print(f"  Ostruzioni dir  : {obstr_dir or 'non fornita'}")
    print(f"  ROI             : {roi or 'intera immagine'}")
    print(f"  Soglia          : mu + {k_sigma}*sigma")
    print(f"  Aug bank/cal    : {n_augments} / {n_cal}")
    print(f"  Coreset         : {coreset_p*100:.1f}%")
    print(f"  Test aug seeds  : {test_aug_seeds}")
    print(f"{'='*60}\n")

    model = FeatureExtractor().to(DEVICE).eval()
    print("Backbone caricato.\n")

    # ── Strutture dati risultati ──────────────────────────────────────────────
    results = []          # una riga per immagine
    all_scores_free  = [] # per AUROC / distribuzione globale
    all_scores_obstr = []

    roi_tuple = parse_roi(roi)

    for idx, bg_path in enumerate(bg_paths):
        name = bg_path.stem
        print(f"[{idx+1:04d}/{len(bg_paths)}] {name}")

        ref_img = Image.open(bg_path).convert("RGB")

        # ── Build memory bank per questa immagine ─────────────────────────
        bank_variants = augment_reference(ref_img, n=n_augments, seed=42)
        cal_variants  = augment_reference(ref_img, n=n_cal,       seed=1000)

        all_features = []
        for img in bank_variants:
            tensor = load_image_tensor(img)
            feats, _, _ = extract_patch_features(model, tensor)
            all_features.append(feats)

        memory_bank_raw = torch.cat(all_features, dim=0)
        target_size = max(50, int(len(memory_bank_raw) * coreset_p))
        bank_np = memory_bank_raw.numpy()
        bank = torch.from_numpy(greedy_coreset(bank_np, target_size))

        # ── Calibrazione soglia su cal_variants (unseen) ──────────────────
        cal_scores_img = [
            compute_image_score_from_pil(model, img, bank, roi_tuple)
            for img in cal_variants
        ]

        mu    = float(np.mean(cal_scores_img))
        sigma = float(np.std(cal_scores_img))
        threshold = mu + k_sigma * sigma

        # ── Test su augmented free images (FP check) ─────────────────────
        free_scores = []
        for seed in test_aug_seeds:
            test_variants = augment_reference(ref_img, n=2, seed=seed)
            # Prendi la prima variante (esclude l'originale che è index 0)
            img_test = test_variants[1]
            score = compute_image_score_from_pil(model, img_test, bank, roi_tuple)
            free_scores.append(score)
            all_scores_free.append(score)

        fp_count = sum(s > threshold for s in free_scores)

        # ── Test su immagine ostruita (TP check) ──────────────────────────
        obstr_score = None
        is_tp = None
        if obstr_dir:
            # Cerca file con stesso stem nel obstr_dir
            obstr_candidates = (
                list(Path(obstr_dir).glob(f"{name}.jpg")) +
                list(Path(obstr_dir).glob(f"{name}.png")) +
                list(Path(obstr_dir).glob(f"{name}_*.jpg")) +
                list(Path(obstr_dir).glob(f"{name}_*.png"))
            )
            if obstr_candidates:
                obstr_path = obstr_candidates[0]
                obstr_img = Image.open(obstr_path).convert("RGB")
                obstr_score = compute_image_score_from_pil(model, obstr_img, bank, roi_tuple)
                is_tp = obstr_score > threshold
                all_scores_obstr.append(obstr_score)
            else:
                print(f"           nessuna immagine ostruita per {name}")

        # ── Log riga ──────────────────────────────────────────────────────
        status = "FP!" if fp_count > 0 else "ok"
        tp_str = ("TP" if is_tp else "FN") if is_tp is not None else "n/a"
        obstr_str = f"{obstr_score:.3f}" if obstr_score is not None else "n/a"
        print(f"         mu={mu:.3f} sigma={sigma:.4f} thr={threshold:.3f} | "
              f"free={[f'{s:.3f}' for s in free_scores]} [{status}] | "
              f"obstr={obstr_str} [{tp_str}]")

        results.append({
            "name": name,
            "mu": round(mu, 5),
            "sigma": round(sigma, 6),
            "threshold": round(threshold, 5),
            "free_scores": [round(s, 5) for s in free_scores],
            "fp_count": fp_count,
            "obstr_score": round(obstr_score, 5) if obstr_score is not None else None,
            "is_tp": is_tp,
        })

    # ── Report finale ─────────────────────────────────────────────────────────
    total_free_tests = len(all_scores_free)
    total_fp = sum(r["fp_count"] for r in results)
    fpr = total_fp / total_free_tests if total_free_tests > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"RISULTATI FINALI")
    print(f"  Immagini valutate   : {len(results)}")
    print(f"  Test free totali    : {total_free_tests}")
    print(f"  Falsi positivi      : {total_fp}  (FPR = {fpr*100:.2f}%)")

    if all_scores_obstr:
        total_tp = sum(1 for r in results if r["is_tp"])
        tpr = total_tp / len(all_scores_obstr)
        print(f"  Test ostruiti       : {len(all_scores_obstr)}")
        print(f"  True positivi       : {total_tp}  (TPR = {tpr*100:.2f}%)")
        # AUROC
        try:
            from sklearn.metrics import roc_auc_score
            all_scores = all_scores_free + all_scores_obstr
            all_labels = [0]*len(all_scores_free) + [1]*len(all_scores_obstr)
            auroc = roc_auc_score(all_labels, all_scores)
            print(f"  AUROC               : {auroc:.4f}")
        except ImportError:
            print("  AUROC: sklearn non disponibile")

    print(f"\n  Score free   — media={np.mean(all_scores_free):.4f}  "
          f"std={np.std(all_scores_free):.4f}  "
          f"p95={np.percentile(all_scores_free, 95):.4f}")
    if all_scores_obstr:
        print(f"  Score ostruiti— media={np.mean(all_scores_obstr):.4f}  "
              f"std={np.std(all_scores_obstr):.4f}  "
              f"p5={np.percentile(all_scores_obstr, 5):.4f}")
    print(f"{'='*60}")

    # ── Salva CSV ─────────────────────────────────────────────────────────────
    out_path = Path(out_csv)
    with open(out_path, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["name", "mu", "sigma", "threshold",
                      "free_scores", "fp_count", "obstr_score", "is_tp"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r_csv = r.copy()
            r_csv["free_scores"] = json.dumps(r["free_scores"])
            writer.writerow(r_csv)
    print(f"\n  Risultati salvati in: {out_path}")

    return results


def insert_experiment(
    conn: sqlite3.Connection,
    model_variant: str,
    dataset_filter: dict,
    hyperparams: dict,
    artifact_path: str | None,
    status: str = "running",
) -> int:
    cur = conn.execute(
        "INSERT INTO experiments (pipeline, model_variant, dataset_filter, hyperparams, "
        "artifact_path, status) VALUES (?, ?, ?, ?, ?, ?)",
        (
            "P3",
            model_variant,
            json.dumps(dataset_filter) if dataset_filter else None,
            json.dumps(hyperparams) if hyperparams else None,
            artifact_path,
            status,
        ),
    )
    return int(cur.lastrowid)


def insert_results(
    conn: sqlite3.Connection,
    exp_id: int,
    results_split: str,
    venue_type: str,
    metrics: dict,
) -> None:
    conn.execute(
        "INSERT INTO results (exp_id, split, venue_type, precision, recall, F1, auroc, "
        "anomaly_threshold, false_alarm_rate) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            exp_id,
            results_split,
            venue_type,
            metrics.get("precision"),
            metrics.get("recall"),
            metrics.get("F1"),
            metrics.get("auroc"),
            metrics.get("anomaly_threshold"),
            metrics.get("false_alarm_rate"),
        ),
    )


def evaluate_from_db(
    db_path: str,
    dataset_root: str,
    split: str | None,
    venue: str | None,
    ref_source: str | None,
    ob_source: str | None,
    roi: str | None,
    k_sigma: float,
    n_augments: int,
    n_cal: int,
    coreset_p: float,
    out_csv: str | None,
    model_variant: str,
    test_normal: bool = True,
) -> None:
    results_split = split if split is not None else "test"

    dataset_root_path = Path(dataset_root)
    if not dataset_root_path.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root_path}")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        refs = query_reference_frames(conn, split, venue, ref_source)
        if not refs:
            print("Nessun riferimento trovato nel DB.")
            return

        ref_ids = [ref.frame_id for ref in refs]
        ob_map = query_obstructed_frames(conn, ref_ids, split, ob_source)
        refs = [ref for ref in refs if ref.frame_id in ob_map]
        if not refs:
            print("Nessuna coppia reference/ostruita trovata nel DB.")
            return

        roi_tuple = parse_roi(roi)

        artifact_path = str(Path(out_csv)) if out_csv else None
        dataset_filter = {
            "split": split,
            "venue": venue,
            "ref_source": ref_source,
            "ob_source": ob_source,
            "reference_only_with_pairs": True,
        }
        hyperparams = {
            "n_augments": n_augments,
            "n_cal": n_cal,
            "coreset_p": coreset_p,
            "k_sigma": k_sigma,
            "roi": roi,
            "test_normal": test_normal,
        }
        exp_id = insert_experiment(
            conn,
            model_variant=model_variant,
            dataset_filter=dataset_filter,
            hyperparams=hyperparams,
            artifact_path=artifact_path,
            status="running",
        )

        model = FeatureExtractor().to(DEVICE).eval()

        per_item_results: list[dict] = []
        missing_files: list[str] = []
        venue_stats: dict[str, dict] = {}

        print(f"\n{'='*60}")
        print("EVALUATE-DB — PatchCore con SQLite")
        print(f"  DB           : {db_path}")
        print(f"  Dataset root : {dataset_root_path}")
        print(f"  References   : {len(refs)}")
        print(f"  Split filter : {split or 'none'}")
        print(f"  Venue filter : {venue or 'none'}")
        print(f"  Ref source   : {ref_source or 'none'}")
        print(f"  Ob source    : {ob_source or 'none'}")
        print(f"  ROI          : {roi or 'intera immagine'}")
        print(f"  Soglia       : mu + {k_sigma}*sigma")
        print(f"  Aug bank/cal : {n_augments} / {n_cal}")
        print(f"  Coreset      : {coreset_p*100:.1f}%")
        print(f"{'='*60}\n")

        for idx, ref in enumerate(refs, start=1):
            ref_path = resolve_dataset_path(dataset_root_path, ref.file_path)
            if not ref_path.exists():
                missing_files.append(str(ref_path))
                continue

            ref_img = Image.open(ref_path).convert("RGB")

            bank_variants = augment_reference(ref_img, n=n_augments, seed=42, include_original=False)
            cal_variants = augment_reference(ref_img, n=n_cal, seed=1000, include_original=False)

            all_features = []
            for img in bank_variants:
                tensor = load_image_tensor(img)
                feats, _, _ = extract_patch_features(model, tensor)
                all_features.append(feats)

            memory_bank_raw = torch.cat(all_features, dim=0)
            target_size = max(50, int(len(memory_bank_raw) * coreset_p))
            bank_np = memory_bank_raw.numpy()
            bank = torch.from_numpy(greedy_coreset(bank_np, target_size))

            cal_scores = [
                compute_image_score_from_pil(model, img, bank, roi_tuple)
                for img in cal_variants
            ]
            mu = float(np.mean(cal_scores))
            sigma = float(np.std(cal_scores))
            threshold = mu + k_sigma * sigma

            stats = venue_stats.setdefault(
                ref.venue_type,
                {
                    "normal_scores": [],
                    "ob_scores": [],
                    "normal_scores_norm": [],
                    "ob_scores_norm": [],
                    "thresholds": [],
                    "fp": 0,
                    "tp": 0,
                    "fn": 0,
                },
            )
            stats["thresholds"].append(threshold)

            if test_normal:
                ref_score = compute_image_score_from_pil(model, ref_img, bank, roi_tuple)
                is_fp = ref_score > threshold
                norm_ref_score = (ref_score - mu) / (sigma + 1e-8)
                stats["normal_scores"].append(ref_score)
                stats["normal_scores_norm"].append(norm_ref_score)
                if is_fp:
                    stats["fp"] += 1

                per_item_results.append(
                    {
                        "reference_id": ref.frame_id,
                        "test_id": ref.frame_id,
                        "test_type": "normal",
                        "venue_type": ref.venue_type,
                        "file_path": ref.file_path,
                        "score": round(ref_score, 6),
                        "normalized_score": round(norm_ref_score, 6),
                        "threshold": round(threshold, 6),
                        "is_anomaly": int(is_fp),
                    }
                )

            for ob in ob_map.get(ref.frame_id, []):
                ob_path = resolve_dataset_path(dataset_root_path, ob.file_path)
                if not ob_path.exists():
                    missing_files.append(str(ob_path))
                    continue

                ob_img = Image.open(ob_path).convert("RGB")
                ob_score = compute_image_score_from_pil(model, ob_img, bank, roi_tuple)
                is_tp = ob_score > threshold
                norm_ob_score = (ob_score - mu) / (sigma + 1e-8)
                stats["ob_scores"].append(ob_score)
                stats["ob_scores_norm"].append(norm_ob_score)
                if is_tp:
                    stats["tp"] += 1
                else:
                    stats["fn"] += 1

                per_item_results.append(
                    {
                        "reference_id": ref.frame_id,
                        "test_id": ob.frame_id,
                        "test_type": "obstructed",
                        "venue_type": ob.venue_type,
                        "file_path": ob.file_path,
                        "score": round(ob_score, 6),
                        "normalized_score": round(norm_ob_score, 6),
                        "threshold": round(threshold, 6),
                        "is_anomaly": int(is_tp),
                    }
                )

            if idx % 10 == 0:
                print(f"  Processati {idx}/{len(refs)} riferimenti...")

        if out_csv:
            import csv

            with open(out_csv, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "reference_id",
                    "test_id",
                    "test_type",
                    "venue_type",
                    "file_path",
                    "score",
                    "normalized_score",
                    "threshold",
                    "is_anomaly",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for row in per_item_results:
                    writer.writerow(row)

        for venue_type, stats in venue_stats.items():
            normal_scores = stats["normal_scores"]
            ob_scores = stats["ob_scores"]
            normal_scores_norm = stats["normal_scores_norm"]
            ob_scores_norm = stats["ob_scores_norm"]
            fp = stats["fp"]
            tp = stats["tp"]
            fn = stats["fn"]

            normal_total = len(normal_scores)
            ob_total = len(ob_scores)

            false_alarm_rate = fp / normal_total if normal_total else None
            recall = tp / ob_total if ob_total else None
            precision = tp / (tp + fp) if (tp + fp) else None
            f1 = None
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1 = 2 * precision * recall / (precision + recall)

            auroc = None
            if normal_scores_norm and ob_scores_norm:
                try:
                    from sklearn.metrics import roc_auc_score

                    all_scores_norm = normal_scores_norm + ob_scores_norm
                    all_labels = [0] * len(normal_scores_norm) + [1] * len(ob_scores_norm)
                    auroc = float(roc_auc_score(all_labels, all_scores_norm))
                except Exception:
                    auroc = None

            threshold_mean = float(np.mean(stats["thresholds"])) if stats["thresholds"] else None

            insert_results(
                conn,
                exp_id=exp_id,
                results_split=results_split,
                venue_type=venue_type,
                metrics={
                    "precision": precision,
                    "recall": recall,
                    "F1": f1,
                    "auroc": auroc,
                    "anomaly_threshold": threshold_mean,
                    "false_alarm_rate": false_alarm_rate,
                },
            )

        conn.execute("UPDATE experiments SET status = 'done' WHERE exp_id = ?", (exp_id,))
        conn.commit()

        print("\nValutazione completata.")
        print(f"  Exp ID: {exp_id}")
        for venue_type, stats in venue_stats.items():
            print(
                f"  {venue_type}: normals={len(stats['normal_scores'])} "
                f"obstr={len(stats['ob_scores'])}"
            )
        if missing_files:
            print(f"  File mancanti: {len(missing_files)}")
        if artifact_path:
            print(f"  CSV salvato: {artifact_path}")
    finally:
        conn.close()

# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PatchCore — Rilevamento Occlusioni Vie di Fuga"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── Comando BUILD ─────────────────────────────────────────────────────────
    p_build = sub.add_parser("build", help="Costruisce la memory bank da una immagine")
    p_build.add_argument("--ref",       required=True,
                         help="Immagine di riferimento (scena normale)")
    p_build.add_argument("--bank",      required=True,
                         help="Percorso output memory bank (.pt)")
    p_build.add_argument("--aug",       type=int,   default=15,
                         help="Varianti augmentation per la bank (default: 15)")
    p_build.add_argument("--cal",       type=int,   default=10,
                         help="Varianti per calibrazione soglia, unseen (default: 10)")
    p_build.add_argument("--coreset-p", type=float, default=CORESET_P,
                         help=f"Frazione coreset (default: {CORESET_P}). "
                              f"Aumentare a 0.05-0.10 per bank da singola immagine.")

    # ── Comando TEST ──────────────────────────────────────────────────────────
    p_test = sub.add_parser("test", help="Testa un'immagine per anomalie")
    p_test.add_argument("--bank",      required=True,
                        help="Memory bank (.pt) da usare")
    p_test.add_argument("--img",       required=True,
                        help="Immagine da testare")
    p_test.add_argument("--roi",
                        help='ROI in pixel: "x,y,w,h" (opzionale)')
    p_test.add_argument("--threshold", type=float, default=None,
                        help="Soglia manuale (default: auto calibrata mu+k*sigma)")
    p_test.add_argument("--k",         type=float, default=3.0,
                        help="Moltiplicatore k per soglia mu+k*sigma (default: 3.0)")

    # ── Comando EVALUATE ──────────────────────────────────────────────────────
    p_eval = sub.add_parser("evaluate",
                            help="Valutazione sistematica FPR/TPR su dataset di background")
    p_eval.add_argument("--bg-dir",     required=True,
                        help="Directory con immagini di background (.jpg/.png)")
    p_eval.add_argument("--obstr-dir",  default=None,
                        help="Directory con immagini ostruite (stesso nome file del bg)")
    p_eval.add_argument("--roi",
                        help='ROI in pixel: "x,y,w,h" (opzionale)')
    p_eval.add_argument("--k",          type=float, default=3.0,
                        help="Moltiplicatore k soglia (default: 3.0)")
    p_eval.add_argument("--aug",        type=int,   default=15,
                        help="Varianti augmentation per la bank (default: 15)")
    p_eval.add_argument("--cal",        type=int,   default=10,
                        help="Varianti calibrazione soglia (default: 10)")
    p_eval.add_argument("--coreset-p",  type=float, default=CORESET_P,
                        help=f"Frazione coreset (default: {CORESET_P})")
    p_eval.add_argument("--test-seeds", default="2000,2001,2002",
                        help="Seed CSV per test augmentations (default: 2000,2001,2002)")
    p_eval.add_argument("--out-csv",    default="eval_results.csv",
                        help="Percorso output CSV risultati (default: eval_results.csv)")

    # ── Comando EVALUATE-DB ───────────────────────────────────────────────────
    p_eval_db = sub.add_parser("evaluate-db",
                               help="Valutazione PatchCore con immagini da SQLite DB")
    p_eval_db.add_argument("--db",
                           default=str(DEFAULT_DB_PATH),
                           help="Path al database SQLite (default: Aggregated_dataset_db/occlusion.db)")
    p_eval_db.add_argument("--dataset-root",
                           default=str(DEFAULT_DATASET_ROOT),
                           help="Root della cartella Dataset")
    p_eval_db.add_argument("--split",
                           default=None,
                           help="Filtro split per reference/ostruite (train/val/test)")
    p_eval_db.add_argument("--venue",
                           default=None,
                           help="Filtro venue_type (porta/corridoio/scala)")
    p_eval_db.add_argument("--ref-source",
                           default=None,
                           help="Filtro source per reference")
    p_eval_db.add_argument("--ob-source",
                           default=None,
                           help="Filtro source per ostruite")
    p_eval_db.add_argument("--roi",
                           help='ROI in pixel: "x,y,w,h" (opzionale)')
    p_eval_db.add_argument("--k", type=float, default=3.0,
                           help="Moltiplicatore k soglia (default: 3.0)")
    p_eval_db.add_argument("--aug", type=int, default=15,
                           help="Varianti augmentation per la bank (default: 15)")
    p_eval_db.add_argument("--cal", type=int, default=10,
                           help="Varianti calibrazione soglia (default: 10)")
    p_eval_db.add_argument("--coreset-p", type=float, default=CORESET_P,
                           help=f"Frazione coreset (default: {CORESET_P})")
    p_eval_db.add_argument("--out-csv", default=None,
                           help="Percorso CSV output (opzionale)")
    p_eval_db.add_argument("--model-variant", default="wide_resnet50_2",
                           help="Etichetta modello per tabella experiments")

    args = parser.parse_args()

    if args.command == "build":
        build_memory_bank(
            args.ref, args.bank,
            n_augments=args.aug,
            n_cal=args.cal,
            coreset_p=args.coreset_p,
        )
    elif args.command == "test":
        test_image(args.bank, args.img,
                   roi=args.roi, threshold=args.threshold, k_sigma=args.k)
    elif args.command == "evaluate":
        seeds = [int(s) for s in args.test_seeds.split(",")]
        evaluate(
            bg_dir=args.bg_dir,
            obstr_dir=args.obstr_dir,
            roi=args.roi,
            k_sigma=args.k,
            n_augments=args.aug,
            n_cal=args.cal,
            coreset_p=args.coreset_p,
            test_aug_seeds=seeds,
            out_csv=args.out_csv,
        )
    elif args.command == "evaluate-db":
        evaluate_from_db(
            db_path=args.db,
            dataset_root=args.dataset_root,
            split=args.split,
            venue=args.venue,
            ref_source=args.ref_source,
            ob_source=args.ob_source,
            roi=args.roi,
            k_sigma=args.k,
            n_augments=args.aug,
            n_cal=args.cal,
            coreset_p=args.coreset_p,
            out_csv=args.out_csv,
            model_variant=args.model_variant,
        )


if __name__ == "__main__":
    main()