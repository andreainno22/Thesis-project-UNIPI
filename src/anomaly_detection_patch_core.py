"""
PatchCore — Anomaly Detection
====================================================================
Uso:
  1. BUILD memory bank da una singola immagine di riferimento:
       python patchcore_door.py build --ref porta.jpg --bank memory_bank.pt

  2. TEST su un'immagine (con o senza occlusione):
       python patchcore_door.py test --bank memory_bank.pt --img test.jpg

  3. TEST con ROI (opzionale): usando --roi "x,y,w,h" in pixel
       python patchcore_door.py test --bank memory_bank.pt --img test.jpg --roi "100,50,300,400"

Dipendenze: torch torchvision pillow numpy scikit-learn opencv-python
"""

#TODO: calibrare la soglia di anomaly score, mediando su vari esempi di porta normale aumentata
#TODO: calibrare il coreset
#TODO: testare porta aumentata in modo diverso da quelle nella memory bank 
# per vedere se viene generato un falso positivo

import argparse
import sys
from pathlib import Path

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


# ── Augmentation fotometrica ────────────────────────────────────────────────

def augment_reference(img_pil: Image.Image, n: int = 15) -> list[Image.Image]:
    """
    Genera n varianti fotometriche di una singola immagine.
    Varia: luminosità, contrasto, saturazione, temperatura colore,
           rumore gaussiano, blur leggero.
    NON applica flip/rotazioni forti: la camera è fissa.
    """
    variants = [img_pil]  # includi l'originale

    rng = np.random.default_rng(42)

    for _ in range(n - 1):
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
            print("      Backbone: WideResNet50-2 (ImageNet pretrained) ✅")
        except Exception:
            print("      ⚠️  Download ImageNet weights non riuscito.")
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
        exp_min = torch.exp(min_dists)                           # [b]
        exp_sum = torch.exp(dists_test_to_nb).sum(dim=1)        # [b]
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

def build_memory_bank(ref_path: str, bank_path: str, n_augments: int = 15):
    print(f"\n{'='*60}")
    print(f"BUILD memory bank")
    print(f"  Immagine di riferimento : {ref_path}")
    print(f"  Augmentazioni generate  : {n_augments}")
    print(f"  Salvataggio in          : {bank_path}")
    print(f"  Device                  : {DEVICE}")
    print(f"{'='*60}\n")

    # 1. Augmentation
    ref_img = Image.open(ref_path).convert("RGB")
    variants = augment_reference(ref_img, n=n_augments)
    print(f"[1/5] Augmentation: {len(variants)} immagini generate")

    # Salva preview augmentazioni
    preview_dir = Path(bank_path).parent / "augmentation_preview"
    preview_dir.mkdir(exist_ok=True)
    for i, v in enumerate(variants):
        v.save(preview_dir / f"variant_{i:02d}.jpg")
    print(f"      Preview salvate in: {preview_dir}/")

    # 2. Estrazione feature
    print(f"[2/5] Estrazione feature con WideResNet50...")
    model = FeatureExtractor().to(DEVICE).eval()

    all_features = []
    for i, img in enumerate(variants):
        tensor = load_image_tensor(img)
        feats, H, W = extract_patch_features(model, tensor)
        all_features.append(feats)
        print(f"      Variante {i+1:02d}/{len(variants)}: "
              f"{feats.shape[0]} patch × {feats.shape[1]}d")

    memory_bank = torch.cat(all_features, dim=0)  # [N_total_patches, D]
    print(f"\n      Memory bank raw: {memory_bank.shape}")

    # 3. Coreset subsampling
    print(f"\n[3/5] Coreset subsampling ({CORESET_P*100:.0f}%)...")
    target_size = max(50, int(len(memory_bank) * CORESET_P))
    bank_np = memory_bank.numpy()
    bank_coreset = greedy_coreset(bank_np, target_size)
    bank_coreset = torch.from_numpy(bank_coreset)
    print(f"      Memory bank finale: {bank_coreset.shape}")

    # 4. Calibrazione soglia su immagini di riferimento
    print(f"\n[4/5] Calibrazione soglia su {len(variants)} immagini di riferimento...")
    cal_scores = []
    for i, img in enumerate(variants):
        tensor = load_image_tensor(img)
        feats, _, _ = extract_patch_features(model, tensor)
        patch_scores = compute_patch_scores(feats, bank_coreset)
        max_score = float(patch_scores.max())
        cal_scores.append(max_score)
        print(f"      Variante {i+1:02d}/{len(variants)}: max score = {max_score:.4f}")

    cal_scores = np.array(cal_scores)
    cal_mean = float(cal_scores.mean())
    cal_std = float(cal_scores.std())
    print(f"\n      Distribuzione score normali:")
    print(f"        Media (mu)  = {cal_mean:.4f}")
    print(f"        Std   (sigma)  = {cal_std:.4f}")
    print(f"        Soglia suggerita (mu+2*sigma) = {cal_mean + 2*cal_std:.4f}")
    print(f"        Soglia suggerita (mu+3*sigma) = {cal_mean + 3*cal_std:.4f}")

    # 5. Salvataggio
    print(f"\n[5/5] Salvataggio memory bank...")
    torch.save({
        "memory_bank": bank_coreset,
        "patch_hw": (H, W),
        "img_size": IMG_SIZE,
        "ref_path": str(ref_path),
        "n_augments": n_augments,
        "cal_mean": cal_mean,
        "cal_std": cal_std,
    }, bank_path)
    print(f"      Salvato: {bank_path}")
    print(f"\n Memory bank pronta! Soglia calibrata: mu={cal_mean:.4f}, sigma={cal_std:.4f}")


# ── TEST: inferenza su immagine di test ─────────────────────────────────────

def test_image(bank_path: str, img_path: str, roi: str = None,
               threshold: float = None, k_sigma: float = 2.0):
    print(f"\n{'='*60}")
    print(f"TEST anomaly detection")
    print(f"  Memory bank : {bank_path}")
    print(f"  Immagine    : {img_path}")
    print(f"  ROI         : {roi or 'intera immagine'}")
    print(f"{'='*60}\n")

    # 1. Carica memory bank
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


# ── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PatchCore — Rilevamento Occlusioni Vie di Fuga"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Comando BUILD
    p_build = sub.add_parser("build", help="Costruisce la memory bank")
    p_build.add_argument("--ref",  required=True,
                         help="Immagine di riferimento (scena normale)")
    p_build.add_argument("--bank", required=True,
                         help="Percorso output memory bank (.pt)")
    p_build.add_argument("--aug",  type=int, default=15,
                         help="Numero varianti augmentation (default: 50)")

    # Comando TEST
    p_test = sub.add_parser("test", help="Testa un'immagine per anomalie")
    p_test.add_argument("--bank",  required=True,
                        help="Memory bank (.pt) da usare")
    p_test.add_argument("--img",   required=True,
                        help="Immagine da testare")
    p_test.add_argument("--roi",
                        help='ROI in pixel: "x,y,w,h" (opzionale)')
    p_test.add_argument("--threshold", type=float, default=None,
                        help="Soglia manuale (default: auto calibrata mu+k*sigma)")
    p_test.add_argument("--k", type=float, default=2.0,
                        help="Moltiplicatore k per soglia mu+k*sigma (default: 2.0)")

    args = parser.parse_args()

    if args.command == "build":
        build_memory_bank(args.ref, args.bank, n_augments=args.aug)
    elif args.command == "test":
        test_image(args.bank, args.img, roi=args.roi, threshold=args.threshold, k_sigma=args.k)


if __name__ == "__main__":
    main()