# Pipeline 3: Unsupervised Anomaly Detection with PatchCore

---

## Pipeline Overview

### Motivation and Design Rationale

The detection of emergency exit occlusions presents a fundamental labeling bottleneck: while normal scenes (clear, unobstructed doorways and corridors) are easy to capture at scale, the space of possible obstructions is open-ended. Wheelchairs, carts, stretchers, waste bins, cardboard boxes, and any combination thereof can occupy a doorway in arbitrary configurations, orientations, and lighting conditions. Collecting and labeling a representative sample of anomalous scenes is time-consuming, and any finite labeled set risks leaving coverage gaps on obstacle types or configurations not seen during training.

Pipeline 3 (P3) addresses this bottleneck by framing occlusion detection as **one-class anomaly detection**: the system learns only from normal, unobstructed scenes, and at inference time flags any image whose feature representation deviates significantly from what was memorized as normal. This eliminates the need for labeled anomalous examples entirely.

The choice to adapt PatchCore (Roth et al., 2022) - originally proposed for industrial surface inspection - to the emergency exit domain is motivated by three properties that align well with the deployment context:

1. **One-shot operation**: a single reference photograph of each monitored location is sufficient to build the entire anomaly model. In a real deployment, each fixed camera captures one reference frame of the clear exit; no additional data collection is required.
2. **Spatial resolution**: PatchCore operates at the patch level (28x28 spatial grid at 224x224 input) rather than producing a single image-level embedding. This preserves the spatial structure of the anomaly, enabling heatmap visualization and spatially-aware filtering.
3. **Edge compatibility**: the inference pipeline - feature extraction through a frozen ResNet backbone, followed by nearest-neighbor lookup in a compact coreset - runs entirely on CPU, requires no gradient computation, and can be parallelized per-camera without shared state.

### Position in the Four-Pipeline Architecture

P3 occupies a distinct niche among the four pipelines developed in this thesis. P1 (YOLO-based object detection) requires labeled bounding boxes for every obstacle category and can only detect object classes it has been trained on. P2 (door-centric bounding box baseline) requires a model of the door state itself. P4 (Siamese change detection) requires a pair of images - a reference and a query - and is sensitive to temporal drift in the reference. P3 requires only a single normal reference image and a pre-trained ImageNet backbone, making it the most data-efficient of the four pipelines and the most straightforward to deploy on a new camera location without any annotation effort.

The trade-off is that P3 cannot identify *which* obstacle is present - it only signals that something is anomalous. It is also, in principle, sensitive to any change in the scene, including non-obstacle changes such as lighting variations, shadows, or background movement. Managing these false positive sources is the primary engineering challenge addressed in this chapter.

### Key Intuition

The core intuition of PatchCore is that ImageNet-pretrained convolutional features are broadly transferable: the intermediate representations of a deep ResNet, trained on millions of diverse natural images, capture generic visual patterns (textures, edges, object parts) that are discriminative across domains, including surveillance footage of building interiors. A patch in a normal scene of a fire exit door will have a feature vector that falls close to other normal patches in this embedding space. A patch belonging to an unexpected obstacle - a wheel of a cart, the fabric of a stretcher, the surface of a cardboard box - will produce a feature vector that is distant from all stored normal representations, yielding a high anomaly score.

The memory bank is built from augmented variants of the single reference image rather than from a large dataset, exploiting the fixed-camera assumption: the camera does not move, so the background remains stable across days and the only source of variation is illumination. Photometric augmentations (brightness, contrast, saturation, color temperature, noise) at build time teach the bank to tolerate this variation, making the calibrated threshold robust to normal scene fluctuations without requiring multiple observation days.

---

## Pipeline Implementation

### Feature Extraction

The backbone is **WideResNet50-2** (He et al., 2016; Zagoruyko and Komodakis, 2016), pre-trained on ImageNet and used in frozen mode (no fine-tuning). Following the original PatchCore formulation, features are extracted from two intermediate blocks: `layer2` and `layer3`.

Every input image is resized to 224x224 and normalized with ImageNet statistics before passing through the backbone. The two feature maps produced are:

| Block | Spatial resolution | Channels |
|---|---|---|
| layer2 | 28 x 28 | 512 |
| layer3 | 14 x 14 | 1024 |

`layer1` provides features at 56x56 but encodes mostly low-level texture information (edges, color gradients) with little semantic content; it was not used. `layer4` (7x7, 2048 channels) is too abstract and strongly biased toward the 1000 ImageNet classes, making it a poor discriminator for the domain-specific textures of obstruction objects. The two chosen blocks balance local detail and semantic context.

`layer3` is upsampled bilinearly to 28x28 and concatenated channel-wise with `layer2`, yielding a joint feature tensor of shape [B, 1536, 28, 28]. A **3x3 average pooling** with same-padding is then applied along the spatial dimensions (neighborhood aggregation, equation 2 in Roth et al., 2022). This operation replaces each patch vector with the weighted average of its 3x3 spatial neighborhood, increasing the receptive field of each feature vector and reducing sensitivity to pixel-level noise. The output is flattened to [B x 784, 1536]: 784 patch feature vectors of 1536 dimensions per image.

### Build Phase

The build phase constructs the memory bank from a single normal reference image `ref_img`.

**Photometric augmentation.** To populate the bank with sufficient coverage of normal appearance variation, `n_augments` augmented variants are generated from `ref_img` (default: 15, seed=42). The augmentations applied are:

| Transform | Range | Rationale |
|---|---|---|
| Brightness | x[0.85, 1.15] | Morning vs. evening vs. artificial light |
| Contrast | x[0.90, 1.10] | Direct vs. diffuse illumination |
| Saturation | x[0.90, 1.10] | LED vs. fluorescent vs. incandescent |
| Red channel shift | [-8, +8] | Color temperature variation |
| Blue channel shift | [-20, +20] | Color temperature variation |
| Gaussian noise | sigma in [1, 4] | Sensor noise |
| Gaussian blur | radius in [0.5, 1.5], p=0.3 | Slight lens defocus |

Critically, no geometric augmentations (flips, rotations, crops) are applied: the camera is physically fixed, so orientation is semantically meaningful and should not be treated as a source of invariance.

The original `ref_img` is **excluded** from the bank variants. This reserves it as an independent normal test sample, avoiding any overlap between the bank and the test evaluation.

**Feature extraction and coreset reduction.** Patch features are extracted from each augmented variant and concatenated into a raw memory bank of shape [784 x 15, 1536] = [11760, 1536]. This is then compressed using a **greedy minimax coreset** algorithm (facility location) to `coreset_p` = 1% of the raw bank, retaining at least 50 points. The coreset algorithm iteratively selects the point that maximizes the minimum distance from the already-selected set, ensuring uniform coverage of the feature space. For computational efficiency, the selection is performed on a sparse random projection to 128 dimensions (Johnson-Lindenstrauss), while the final coreset is extracted from the original 1536-dimensional space. The resulting memory bank contains approximately 117 vectors - a compact representation that fits in a few megabytes and enables sub-millisecond nearest-neighbor lookups.

**Rectangular ROI at build time.** When a rectangular ROI is specified via `--roi "x,y,w,h"` or `--roi-rect-json`, the patch mask corresponding to the ROI is computed before feature collection. Only patches whose center falls within the ROI after scaling to 224x224 are included in the bank and in calibration. This ensures consistency between build and test: if the bank is built on door-only patches, test scoring must also be restricted to door patches, avoiding the distribution mismatch that would arise if background patches were scored against a door-only bank.

### Threshold Calibration

#### Design space and rationale

Several threshold strategies are possible for a one-class anomaly detector: a fixed global value, a per-camera percentile of observed scores, a ROC-optimal cut calibrated on labeled anomalies, or a parametric rule derived from the distribution of normal scores. Each has different requirements and properties.

A **fixed global threshold** requires that the absolute scale of anomaly scores be consistent across cameras. This does not hold in practice: the score of a patch from a plain white door against a bank of plain white door variants is structurally lower than the score of a glass-door patch with an outdoor background visible through the glass. A threshold calibrated on the easier scene would generate systematic false positives on the harder scene, and vice versa. Global thresholds are therefore unsuitable for a multi-scene deployment.

A **ROC-optimal threshold** maximizes a chosen metric (e.g., F1) on a labeled set of normal and anomalous examples. This violates the one-class premise: labeled anomalous examples are exactly what P3 is designed to avoid requiring. Moreover, a ROC-optimal threshold on synthetic anomalies (copy-paste composites) would overfit to the specific appearance of those anomalies and potentially underperform on real-world obstacles not represented in the training set.

A **per-camera percentile** (e.g., the 99th percentile of calibration scores) is equivalent to a parametric rule under a specific distributional assumption, but is less interpretable and more sensitive to outliers in the calibration sample - a single anomalously high calibration score inflates the estimate without recourse.

The chosen strategy is `theta = mu + k * sigma` estimated from n_cal augmented variants of the reference image. This is equivalent to the threshold used in statistical process control (Shewhart control charts) for detecting out-of-distribution events in a monitored process: given an empirical estimate of the normal distribution's mean and standard deviation, the threshold is set at k standard deviations above the mean. Under a Gaussian assumption, k=3 corresponds to a theoretical FPR of 0.13% and k=4 to 0.003%. In practice the calibration scores are not exactly Gaussian - they are maxima of correlated patch scores - but the k parameter still provides an intuitive and monotone control over the FPR/recall trade-off.

The key advantage over alternatives is that **the threshold adapts to the scene without requiring any anomalous data**: the only input is the normal reference image and its augmented variants. The parameter k is the sole deployment-time hyperparameter and has a clear semantic meaning - it encodes the asymmetric cost of false negatives relative to false positives. For safety-critical applications (emergency exits where a missed obstruction can be life-threatening), k is set low (2.5-3.0) to prioritize recall at the expense of FPR. For applications where false alarms have high operational cost, k is set higher (4.0-5.0).

#### Calibration procedure and anti-leakage design

The threshold is calibrated on `n_cal` augmented variants of `ref_img` (default: 10, seed=1000). The seed is different from the bank seed (42) to ensure statistical independence between the two augmented sets.

For each calibration variant, the maximum anomaly score over all patches (or over the ROI if specified) is computed. The n_cal resulting scores form the empirical distribution of maximum anomaly scores for a normal scene. Their mean `mu` and standard deviation `sigma` parametrize the threshold: `theta = mu + k * sigma`.

The original `ref_img` is excluded from calibration. Inclusion would bias `mu` downward: the reference is the closest possible image to the bank (the bank is built from its augmentations), so its score is structurally the minimum of the distribution, not a representative sample of it. Reserving `ref_img` as the actual normal test point - entirely unseen by both the bank construction and calibration - produces an honest estimate of FPR.

The separation between the three data sets is strictly enforced:

| Set | Seed | Contains ref_img? | Role |
|---|---|---|---|
| Bank variants (n=15) | 42 | No | Builds the feature memory |
| Calibration variants (n=10) | 1000 | No | Estimates mu and sigma |
| Normal test | - | Yes | Independent normal sample |
| Obstructed test | - | - | Anomalous query |

#### Coverage dependency

The accuracy of the calibration depends directly on the coverage of the augmentation scheme. `mu` and `sigma` are estimated on augmented variants, so they reflect only the score variability that those augmentations induce. If a real-world source of variation (e.g., cast shadows, outdoor background changes through glass) is not represented in the calibration set, `sigma` is underestimated for that scene, and the real test scores under that variation will exceed `theta` more often than expected.

This dependency is the root cause of the shadow false positive problem described in the Results section: baseline calibration produces `sigma` estimated from shadow-free variants, yielding a threshold tuned for shadow-free test conditions. Shadow scenes produce scores well above this threshold because their contribution to `sigma` is absent from the calibration. Extending the augmentation scheme to cover shadows - and carefully managing the shadow fraction in calibration to avoid inflating `sigma` - is precisely the fix developed in Phase C.

### Test Phase and Anomaly Map

At inference, the query image is processed by the same feature extraction pipeline (224x224 resize, WideResNet50-2, neighborhood aggregation) to produce 784 patch vectors of 1536 dimensions. Each patch vector is compared to the memory bank via k-nearest-neighbor search.

The anomaly score for patch `p` follows the re-weighted formulation of equation 7 in Roth et al.:

```
s* = ||p - m*||          (Euclidean distance to nearest bank neighbor m*)
Nb(m*) = 9-NN of m* within the bank
w = 1 - exp(||p - m*||) / sum_{m in Nb(m*)} exp(||p - m||)
score(p) = w * s*
```

The re-weighting factor `w` reduces the score when `m*` lies in a dense region of the bank (high confidence that the neighbor is representative of normal content) and amplifies it when `m*` is isolated (the nearest match itself is atypical). The distances in the denominator are between the test patch `p` and the members of `Nb(m*)`, not between `m*` and its neighbors - a detail that matters for correctness relative to the paper's formulation.

The 784 per-patch scores are reshaped to 28x28, bilinearly upsampled to the original image resolution, and smoothed with a Gaussian filter (sigma=4). The smoothing step serves two purposes: it spatially broadens anomaly blobs (improving detection of small objects that span only 1-2 patch cells), and it reduces high-frequency noise introduced by the upsampling step.

The scalar image score is the **maximum** of the smoothed anomaly map within the active region (the full image, or the rectangular ROI if specified). The binary decision is `score > theta`.

### Spatial Gating via Annotated Polygons

Glass-door scenarios introduce a specific failure mode: the transparent panel makes external backgrounds visible, and any movement or lighting change beyond the door generates anomaly blobs in the glass region without any physical obstruction being present. A rectangular ROI restricted to the door frame mitigates this, but a more principled solution exploits the physical grounding constraint: **any real obstruction must rest on the floor and therefore intersects the door threshold**.

Door-frame polygons (left jamb, right jamb, center mullion, threshold, architrave) are annotated manually using `annotate_door_roi.py`, a lightweight matplotlib-based tool, and stored in the database as `roi_polygons` entries per frame. At evaluation time, the polygon set for the reference frame is loaded, rasterized to the original image resolution, and used to construct a binary `frame_mask`.

The gating rule operates on connected components of the thresholded anomaly map:

1. Binarize the smoothed anomaly map at the calibrated threshold `theta`.
2. Run 8-connected component labeling (`cv2.connectedComponents`).
3. Accept component `C` if `|C AND frame_mask| >= min_overlap_px` (default: 10).
4. Compute the final score as the maximum of the continuous anomaly map over all accepted components. If no component survives, the score is 0.0.

This rule correctly separates background variation behind the glass (components confined to the glass panel, zero overlap with the frame mask) from physical obstructions (components extending to the threshold or jambs). The scoring function over accepted components is a max - not a sum or average - so a valid obstruction component is scored at its peak anomaly intensity regardless of how many pixels overlap the frame mask.

The polygon gating is also applicable to opaque doors when the panel surface exhibits significant illumination variation (specular highlights, direct sunlight patches). In such cases, illumination anomalies are confined to the panel surface and do not reach the annotated frame structure, while physical obstructions - which rest on the floor - necessarily produce anomaly blobs extending to the threshold. For opaque doors with stable illumination, gating adds no benefit and is left disabled.

A rectangular ROI (`--roi`) and polygon gating compose sequentially: the rectangular mask zeroes the anomaly map outside the door bounding box first, then gating is applied to the remaining map. This allows suppressing floor reflections and solar shadows (which fall outside the door rectangle) while retaining the glass-area filtering for pixels inside the door.

**Fixed camera assumption and native spatial tolerance.** The polygon gating mechanism assumes the camera is physically stable: frame polygons are annotated once on the reference image and reused on all subsequent frames without modification. Before treating this as a hard constraint, it is worth quantifying the tolerance the pipeline already provides by construction.

Two components absorb small misalignments without any modification. First, the 3×3 neighborhood average pooling means each patch feature integrates a 24×24 px region of the original image (at native resolution); a displacement of 1-2 patch positions in the 28×28 grid does not meaningfully alter the feature vector. Second, the Gaussian smoothing (σ=4) applied to the anomaly map spatially broadens blobs, reducing sensitivity to exact pixel positions. A hand-held photograph typically differs from the reference by 5-15 px of translation and 0.5-2° of rotation. After resizing to 224×224, this reduces to 1-3 px - well within the neighborhood pooling tolerance.

If larger residual motion must be tolerated, two mitigation strategies are available. The first is **geometric augmentation of the bank**: including variants with small affine transforms (shift ±8 px, rotation ±2°) at build time teaches the bank to recognize the same scene under minor positional variation, with no inference-time cost. The second is **ECC-based image registration** at test time (`cv2.findTransformECC`, `MOTION_EUCLIDEAN`), which aligns each query frame to the reference before feature extraction; this handles larger displacements but adds a preprocessing step and a potential failure mode. For the evaluation in this thesis, neither mitigation is applied: the data collection protocol (fixed foot position, braced camera) keeps hand-held variation within the native tolerance of the pipeline.

For production deployments, a rigidly mounted camera remains the correct design choice. Automatic tracking of frame corner positions (e.g., via incremental homography) to update the ROI annotation dynamically without re-annotation is a natural future extension.

### Shadow Augmentation

The baseline memory bank contains no augmented variants with cast shadows. This leaves a coverage gap: at test time, shadows projected on the floor or walls by persons or objects passing outside the field of view produce patch features that are distant from all bank entries, triggering false positives. The shadow FPR measured on the baseline system is 90.1% (see Results).

The fix is to include synthetic shadow variants in both the bank and calibration augmentations. Three synthetic shadow types are implemented:

| Type | Description |
|---|---|
| Elliptical | Dark ellipse with blurred borders, random position/size/rotation |
| Directional | Progressive dark gradient from one image edge, random direction and width |
| Stripe | Horizontal or vertical dark band with blurred borders, width 5-20% of image |

All borders are Gaussian-blurred to avoid unrealistic hard edges that would generate high-frequency features absent from any real shadow. Each augmented variant receives k in [1, 3] stacked shadows with probability `shadow_prob` (applied after the standard photometric transforms).

A critical design choice is to **decouple the shadow probability between build and calibration**:
- Build (`shadow_prob`=0.4): bank variants include shadows, so test-time shadow patches find near neighbors in the bank and produce low kNN distances.
- Calibration (`shadow_prob_cal`=0.1): only a small fraction of calibration variants carry shadows. This keeps `sigma_cal` stable and prevents over-inflation of the threshold by the high-variance shadow scores.

The motivation for decoupling is that the two phases have opposing requirements: the bank needs maximum coverage to reduce kNN distances on shadow patches; the calibration needs a stable distribution to produce a reliable `mu` and `sigma` for threshold computation. Using `shadow_prob_cal = shadow_prob = 0.4` makes `sigma_cal` high-variance (a few unlucky variants with heavy shadows can produce a sigma outlier), which with `k * sigma` amplification yields per-camera thresholds that are either too high (false negatives) or too low (false positives).

### Light Augmentation

Shadow augmentation addresses darkening disturbances; the symmetric failure mode is localized **brightness excess** - light beams from skylights, lateral windows, specular reflections on shiny surfaces, or direct sunlight entering through a glass door. These generate patch features elevated in luminosity that, like shadows, fall outside the distribution of a photometric-only memory bank.

Three synthetic light types are implemented, each modeling a physically distinct illumination pattern:

| Type | Description |
|---|---|
| Elliptical | Bright ellipse with soft borders; models skylight beam, spotlight, or specular floor reflection |
| Directional | Progressive bright gradient from one edge; models lateral window or open door onto a lit space |
| Stripe | Horizontal, vertical, or diagonal bright band; the diagonal variant receives a warm color tint (R↑, B↓) to simulate the chromaticity of direct sunlight |

The same decoupling principle as shadow augmentation applies: `light_prob=0.4` for build variants (maximum bank coverage for light patches) and `light_prob_cal=0.1` for calibration variants (stable `sigma_cal`). The two parameters are independent of `shadow_prob` and `shadow_prob_cal`: a variant can receive a shadow, a light beam, both, or neither.

**Asymmetry between shadow and light spaces (methodological caveat).** Although the shadow and light augmentation families share the same structural template (three modalities each, with k in [1,3] stacked instances per image), they do not cover visual spaces of equivalent dimensionality. The light augmentation introduces two additional axes of variation motivated by physical realism: the stripe modality has three orientations (horizontal, vertical, and a continuously-parameterized diagonal at 25-65° or 115-155° simulating solar incidence angles), against the two orientations of shadow stripes; and both the elliptical and diagonal-stripe lights apply a warm color shift (R↑, B↓) to model the chromaticity of direct sunlight, which has no analogue in shadow generation since cast shadows are achromatic. As a result, the same fraction of bank variants must cover a richer feature subspace in the light case than in the shadow case. The practical consequence is that **the raw values of `light_FPR` and `shadow_FPR` cannot be directly compared as measures of augmentation effectiveness**; the legitimate within-category comparisons are `light_FPR` across configurations (L-only vs SL-15 vs SL-25) and `shadow_FPR` across configurations (C-pooled vs SL-15 vs SL-25). This asymmetry also provides explicit motivation for the larger `aug=25` configuration when both augmentations are active: the light subspace contains more modalities and therefore benefits proportionally more from increased coverage.

**Bank coverage with two disturbance types.** When both `shadow_prob=0.4` and `light_prob=0.4` are active with `aug=15`, the expected breakdown across 15 variants is: 5.4 pure-photometric, 3.6 shadow-only, 3.6 light-only, and 2.4 shadow-plus-light. The coreset (~117 points) must cover four feature subspaces instead of three. With `aug=15` this leaves ~3.6 shadow-only variants - roughly half the ~6 available in the shadow-only configuration of Phase C - raising the risk of a regression in shadow FPR. Increasing to `aug=25` restores per-type coverage to ~6 shadow-only and ~6 light-only variants, at the cost of a ~65% longer build time per camera. The ablation in Phase E quantifies whether bank size is the limiting factor for dual-augmentation performance.

### Calibration Stabilization: Pooled Sigma

Per-camera threshold calibration using `theta_i = mu_i + k * sigma_cal_i` has a structural instability: `sigma_cal_i`, estimated on n=10-30 samples with `shadow_prob_cal`=0.1, has a relative standard error of approximately 18% (n=30). This translates directly to an 18% relative uncertainty in the `k * sigma_cal_i` term of the threshold.

With k=4 and `sigma_cal_i` around 0.3, the `4 * 0.3 = 1.2` contribution to the threshold has a standard error of about 0.22. Across ~1000 cameras, the resulting distribution of thresholds has a long tail: some cameras receive thresholds as high as 6.0 (missing all obstructions) and some as low as 2.5 (generating shadow false positives), even though the underlying scenes are not fundamentally different.

The fix replaces per-camera `sigma_cal_i` with a global estimator:

```
theta_i = mu_i + k * sigma_pop
sigma_pop = median( sigma_cal_1, ..., sigma_cal_N )
```

`mu_i` remains per-camera, preserving scene-specific adaptation. `sigma_pop` is computed after all cameras have been processed in a first pass and applied uniformly in a second pass. Using the median rather than the mean makes `sigma_pop` robust to the outlier `sigma_cal_i` values that caused the threshold distribution tail.

The variance of `theta_i` under pooled-sigma is `Var(mu_i)`, compared to `Var(mu_i) + k^2 * Var(sigma_cal_i)` under per-camera sigma. Empirically, this reduces `sigma_theta` from 0.497 to 0.137 (73% reduction) and compresses the threshold range from [2.28, 6.09] to [3.24, 4.16], eliminating the pathological tails.

---

## Pipeline Results

### Experimental Setup

All results are obtained via `evaluate-db`, which processes each normal reference frame in the database as follows: (i) build the memory bank from augmented variants of the reference (excluding the original); (ii) calibrate the threshold on independent augmentation variants; (iii) score the original reference frame as the normal test sample; (iv) score all obstructed frames linked to that reference.

Scores are normalized per camera as `(score - mu_i) / sigma_i` before computing AUROC, making the metric comparable across cameras with different absolute score scales. Threshold-based metrics (FPR, TPR, F1) use the per-camera threshold `theta_i` so that each camera's binary decision reflects its own calibration.

All experiments use WideResNet50-2 on the copy-paste synthetic dataset (`ostruzioni_reali`), split `test`, evaluated on both `porta` and `corridoio` venue types together unless noted. The dataset contains approximately 990 reference frames in the test split, each with one or more synthetic obstructed variants generated by copy-pasting obstacle objects onto the reference background.

The evaluation protocol is organized as a progressive ablation:

| Phase | Configuration | Purpose |
|---|---|---|
| A | Baseline (no shadows, no ROI) | Characterize base behavior and identify failure modes |
| B | Baseline + shadow test set | Quantify shadow FPR |
| C | Shadow augmentation (shadow_prob=0.4) | Measure effect of shadow fix |
| C-decoupled | Decoupled shadow_prob (0.4/0.1), cal=30 | Stabilize threshold distribution |
| C-pooled | Pooled sigma, k=4.0 | Eliminate threshold variance tail |
| D | Real glass-door scenes + ROI gating | Validate gating on real scenes (pending) |
| E | Light augmentation ablation (L-only, SL-15, SL-25) | Reduce light FPR; quantify bank size effect (pending) |

### Phase A: Baseline Performance on Copy-Paste Dataset

The baseline configuration (15 bank augmentations, 10 calibration variants, shadow_prob=0.0, k=3.0) achieves the following on the synthetic test set:

| Category | Mean score | Std | FPR / TPR |
|---|---|---|---|
| Normal | 2.309 | 0.156 | FPR 0.5% |
| Obstructed | 4.648 | 0.348 | TPR 100% |
| AUROC | | | ~1.000 |

The near-perfect AUROC and 100% TPR confirm that the copy-paste synthetic dataset is cleanly separable: the obstacle patches produce feature vectors that are substantially outside the distribution of normal door/corridor patches. The 0.5% FPR corresponds to approximately 5 false alarms out of 990 normal reference frames, reflecting the small probability that photometric augmentation variants used in calibration do not fully cover the score range of the original reference.

The score distributions show a gap of over 1.3 standard deviations between the means of the two classes (4.648 vs. 2.309 in absolute score, or roughly 15 sigma in normalized units), indicating highly confident separation. This result establishes PatchCore as a viable detector for the class of occlusions considered and validates the evaluation methodology.

### Phase B: Shadow Vulnerability Diagnosis

A test set of shadow-augmented normal scenes is evaluated against the baseline model. Each of the 990 reference frames is associated with one synthetic shadow variant (k in [1,3] stacked shadows, random types). These variants are normal scenes - no obstruction is present - and should be classified as non-anomalous.

| Category | Mean score | Std | Min | Max | Rate |
|---|---|---|---|---|---|
| Normal | 2.309 | 0.156 | 1.871 | 3.129 | FPR 0.5% |
| Shadow normal | 3.189 | 0.495 | 2.097 | 4.507 | **FPR 90.1%** |
| Obstructed | 4.648 | 0.348 | 3.542 | 5.866 | TPR 100% |

The shadow FPR of 90.1% is the key diagnostic result. Shadow patches - dark regions with smooth, blurred borders - produce feature vectors substantially different from any entry in the bank (which contains only photometrically-varied but shadow-free augmentations). The baseline system cannot distinguish a cast shadow from a physical obstruction at the feature level.

The score distributions of shadow normal and obstructed overlap in the range [3.54, 4.51]: shadow normal max = 4.507, obstructed min = 3.542. A global threshold sweep shows that no fixed threshold achieves simultaneously acceptable shadow FPR and TPR:

| Threshold | Shadow FPR | Obstructed TPR |
|---|---|---|
| 3.75 | 13.1% | 99.4% |
| 4.00 | 4.8% | 97.3% |
| 4.25 | 1.4% | 87.3% |
| 4.50 | 0.1% | 67.4% |

This analysis motivates the shadow augmentation fix: the problem is not threshold calibration but distribution mismatch between the bank (shadow-free) and the test set (shadows present).

### Phase C: Shadow Augmentation Fix

Shadow augmentation is introduced in the build phase (`shadow_prob=0.4`, k=3.0). Each bank variant receives, with 40% probability, a synthetic shadow (1-3 stacked, random type) applied after the standard photometric transforms.

**Initial run (shadow_prob = shadow_prob_cal = 0.4, k=3.0):**

| Metric | Baseline | Shadow aug (sp=0.4) | Delta |
|---|---|---|---|
| Clean FPR | 0.5% | 0.0% | -0.5 pp |
| Shadow FPR | 90.1% | 0.4% | **-89.7 pp** |
| Obstructed TPR | 100% | 70.6% | -29.4 pp |
| AUROC (clean + shadow vs. obstr.) | ~1.000 | ~1.000 | invariant |

The shadow FPR is reduced by two orders of magnitude. However, TPR drops to 70.6%: the threshold is over-inflated because calibration variants also receive heavy shadows, raising `mu_cal` and especially `sigma_cal`, and the `k * sigma_cal` term amplifies the resulting threshold.

Analysis of the per-camera threshold distribution reveals that the issue is not a uniform upward shift but a **long tail**: 12% of cameras receive thresholds above 4.5 (generating 86% of all false negatives), while the remaining 88% perform well. These are calibration outliers - the same camera re-calibrated with a different random seed would produce a normal threshold - not scene outliers.

**Decoupled calibration (shadow_prob=0.4, shadow_prob_cal=0.1, cal=30, k=3.0):**

Separating the shadow probability for build and calibration, and increasing calibration samples from 10 to 30 to reduce the standard error on `sigma_cal`:

| Metric | Coupled (sp=spc=0.4) | Decoupled (spc=0.1, cal=30) | Delta |
|---|---|---|---|
| Clean FPR | 0.0% | 0.0% | invariant |
| Shadow FPR | 2.1% | 28.4% | +26.3 pp |
| Obstructed TPR | 70.6% | 99.4% | +28.8 pp |
| F1 | 0.818 | 0.873 | +0.055 |
| sigma of thresholds | 0.539 | 0.388 | -28% |

Decoupling recovers TPR from 70.6% to 99.4% by stabilizing the threshold distribution, but the operating point is now too permissive: shadow FPR rises to 28.4%. The sigma of per-camera thresholds decreases by 28%, confirming the diagnostic but not yet reaching a stable distribution.

**Pooled sigma (shadow_prob=0.4, shadow_prob_cal=0.1, cal=30, k=4.0, sigma_mode=pooled):**

Replacing per-camera `sigma_cal_i` with the global median `sigma_pop` = 0.300 eliminates the residual variance in the threshold distribution:

| Metric | Decoupled per-camera k=4 | Pooled sigma k=4 | Delta |
|---|---|---|---|
| Clean FPR | 0.0% | 0.0% | invariant |
| Shadow FPR | 13.5% | **12.2%** | -1.3 pp |
| Obstructed TPR | 93.4% | **99.8%** | +6.4 pp |
| Obstructed FNR | 6.6% | 0.2% | -6.4 pp |
| Precision | 87.3% | 89.1% | +1.8 pp |
| **F1** | 0.903 | **0.941** | **+0.038** |
| AUROC | 0.998 | 0.998 | invariant |
| sigma of thresholds | 0.497 | **0.137** | **-73%** |
| Threshold range | [2.28, 6.09] | [3.24, 4.16] | -76% |

The threshold distribution collapses from a range of 3.81 units to 0.92 units. The pathological long tail disappears entirely: in the per-camera k=4 run, 6% of cameras with theta > 4.5 generated 59% of all false negatives; in the pooled run, no camera has theta > 4.16 and TPR is 99.3% or higher across all threshold buckets.

The 3.8 pp F1 gain is attributable entirely to variance reduction - not to any change in the backbone's discriminative capacity, which remains constant at AUROC = 0.998.

**Interpretation of the k=4 operating point.** The choice of k=4 is consistent with a conservative "4-sigma" rule and was not optimized on the test set. Threshold-sweep analysis shows that the F1-optimal k on this dataset is approximately 3.9, but reporting the optimum would constitute test-set tuning. The k=4 result (shadow FPR = 12.2%, TPR = 99.8%, F1 = 0.941) represents a principled point on the ROC curve, defensible on statistical grounds. In a safety-critical deployment where missed occlusions (false negatives) carry asymmetric cost relative to false alarms, k can be lowered toward 2.5-3.0 at the cost of higher shadow FPR.

**Summary of P3 results on copy-paste synthetic dataset (pooled sigma, k=4.0):**

| Metric | Value |
|---|---|
| AUROC | 0.998 |
| Clean FPR | 0.0% |
| Shadow FPR | 12.2% |
| Obstructed TPR | 99.8% |
| Precision | 89.1% |
| F1 | 0.941 |
| Sigma of per-camera thresholds | 0.137 |

### Phase D: Glass-Door Generalization

*(Results pending - test set currently being collected and annotated.)*

The Phase D evaluation targets real glass-door scenarios - a failure mode not present in the copy-paste synthetic dataset. Glass panels make external backgrounds visible, and any background change (persons walking by, lighting variation) generates anomaly components in the glass region. The connected-component polygon gating mechanism described in the Implementation section is designed to suppress these components while preserving detection of physical obstructions.

The test set consists of photographs of glass doors collected from two sources: real glass doors at Poli Ingegneria (field-captured reference and obstructed scenes) and Pinterest images of glass emergency exits (diverse backgrounds and lighting). For each scene, the reference frame is annotated with:
- A rectangular ROI enclosing the door (via `annotate_rect_roi.py`)
- Polygon masks for the structural frame elements - left and right jambs, threshold, and optionally center mullion and architrave (via `annotate_door_roi.py`)

The metrics to be reported include: FPR on glass-background-change scenes (i.e., normal scenes with backgrounds different from the reference), TPR on obstructed scenes, and the delta between configurations with and without gating active. Given the small dataset size (order of tens of scenes), results will be accompanied by 95% bootstrap confidence intervals and interpreted as qualitative validation of the gating mechanism rather than as a production benchmark.

The expected result, based on the physical grounding argument, is that gating substantially reduces glass-background FPR (components confined to the glass panel are rejected) while maintaining TPR for physical obstructions (which extend to the threshold annotation). The quantitative magnitude of the effect depends on the severity of background variation in the collected scenes.

### Phase E: Light Augmentation Ablation

*(Results pending - runs scheduled.)*

Phase E addresses the symmetric counterpart of the shadow failure mode: localized brightness excess from light beams, specular reflections, and direct sunlight entering through glass panels. The evaluation uses a `light_test` dataset of 1100 synthetic light-augmented normal scenes (550 doors, 550 corridors), generated with the three light types described in the Implementation section and registered in the database as `source='light'`.

Four configurations are evaluated as a progressive ablation, all at k=4.0, sigma_mode=pooled, cal=30:

| Config | shadow_prob | light_prob | aug | shadow_prob_cal | light_prob_cal |
|---|---|---|---|---|---|
| C-pooled (ref.) | 0.4 | 0.0 | 15 | 0.1 | 0.0 |
| L-only | 0.0 | 0.4 | 15 | 0.0 | 0.1 |
| SL-15 | 0.4 | 0.4 | 15 | 0.1 | 0.1 |
| SL-25 | 0.4 | 0.4 | 25 | 0.1 | 0.1 |

The L-only configuration isolates the effect of light augmentation without shadow interaction; SL-15 tests the combined case on the existing bank size; SL-25 tests whether increasing the bank from 15 to 25 variants recovers the per-type coverage diluted when two disturbance classes are active simultaneously (see Implementation section).

The primary metric of interest is `light_FPR` - the false positive rate on light-augmented normal scenes - alongside `shadow_FPR` and `TPR` to detect any regression introduced by the change. AUROC is expected to remain at ~0.998 regardless of augmentation configuration, as the backbone's discriminative capacity is invariant to threshold placement.

### Discussion

The progressive ablation demonstrates that the failure modes of the P3 pipeline are diagnosable and fixable within the PatchCore framework without architectural changes:

- **Cast shadows**: addressed by including synthetic shadow variants in the bank augmentation, with decoupled calibration to prevent threshold inflation.
- **Localized brightness excess** (light beams, reflections, direct sunlight): addressed symmetrically to shadows, by adding synthetic light variants to the bank; ablation over bank size (Phase E, pending) quantifies the interaction with shadow augmentation when both are active.
- **Glass-door backgrounds**: addressed by connected-component polygon gating, exploiting the physical constraint that real obstructions are floor-grounded.
- **Threshold instability across cameras**: addressed by pooling the sigma estimate across all cameras in the deployment, replacing a noisy per-camera estimate with a robust population-level value.

Each fix is motivated by a specific, quantified failure mode, and the AUROC (0.998) confirms that the discriminative capability of the backbone is never the limiting factor - the pipeline challenges are exclusively in threshold placement and spatial filtering. This makes the system robust in the following sense: improvements to threshold calibration strategy directly translate to F1 improvements without requiring new training data or backbone changes.

The structural results (shadow augmentation effectiveness, decoupling rationale, pooled-sigma variance reduction) are pipeline properties that generalize beyond the specific test set. The exact operating point metrics (FPR = 12.2%, TPR = 99.8% at k=4) are representative for this dataset and shadow synthesis protocol but should be recalibrated for any new deployment using site-specific `sigma_pop` and an appropriate k choice based on the FN/FP cost ratio of the application.
