"""Verify that the poli_ingegneria obstructed/bg images are aligned to refs.

For each current test image (obstructed davanti_* and bg dietro_vetro_o_poster)
we re-fit an ECC EUCLIDEAN warp against its reference porta_N.jpg. If the image
is already aligned, the residual transform must be ~identity (dx, dy ~ 0 px,
rot ~ 0 deg) and the post-fit correlation high.

Prefers cv2 (same method as align_poli_test_set.py). If cv2 is missing it falls
back to a numpy FFT phase-correlation that estimates translation only.

Usage:
    python src/check_poli_alignment.py
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
REF_DIR = DATA / "non_ostruite" / "poli ingegneria"
TEST_ROOT = DATA / "ostruzioni_poli_ingegneria"   # current (aligned) images
SCEN = ("davanti_centro", "davanti_destra", "davanti_sinistra", "dietro_vetro_o_poster")
SCALE = 0.25
ITERS = 300
EPS = 1e-6

# Alignment acceptance thresholds (residual after re-fit).
TOL_FRAC = 0.005   # |dx|,|dy| must be < 0.5% of width/height
TOL_ROT = 0.5      # |rot| must be < 0.5 deg

try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False


def read_gray(path: Path):
    if HAVE_CV2:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        return img
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit(
            "Neither cv2 nor PIL is available. Run this in the tesi_env "
            "container: conda run -n tesi_env python src/check_poli_alignment.py"
        )
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def fit_ecc_cv2(ref_gray, test_gray):
    rs = cv2.resize(ref_gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    ts = cv2.resize(test_gray, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, ITERS, EPS)
    cc, warp = cv2.findTransformECC(rs, ts, warp, cv2.MOTION_EUCLIDEAN, crit, None, 5)
    dx = float(warp[0, 2] / SCALE)
    dy = float(warp[1, 2] / SCALE)
    rot = float(np.degrees(np.arctan2(-warp[0, 1], warp[0, 0])))
    return float(cc), dx, dy, rot


def fit_phasecorr_numpy(ref_gray, test_gray):
    h = min(ref_gray.shape[0], test_gray.shape[0])
    w = min(ref_gray.shape[1], test_gray.shape[1])
    a = ref_gray[:h, :w].astype(np.float64)
    b = test_gray[:h, :w].astype(np.float64)
    a = (a - a.mean()) / (a.std() + 1e-8)
    b = (b - b.mean()) / (b.std() + 1e-8)
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    Fa = np.fft.fft2(a * win)
    Fb = np.fft.fft2(b * win)
    R = Fa * np.conj(Fb)
    R /= np.abs(R) + 1e-8
    r = np.fft.ifft2(R).real
    peak = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = peak
    if dy > h // 2:
        dy -= h
    if dx > w // 2:
        dx -= w
    cc = float((a * b).mean())
    return cc, float(dx), float(dy), float("nan")


def main():
    if not REF_DIR.exists():
        raise SystemExit(f"Reference dir not found: {REF_DIR}")
    method = "cv2 ECC EUCLIDEAN" if HAVE_CV2 else "numpy FFT phase-corr (translation only)"
    print(f"Alignment check method: {method}")
    print(f"Test images under: {TEST_ROOT}\n")
    print(f"{'scenario':24s} {'img':9s} {'cc':>6s} {'dx_px':>8s} {'dy_px':>8s} "
          f"{'rot':>7s} {'dx%':>6s} {'dy%':>6s}  verdict")

    rows = []
    for n in range(1, 7):
        ref = read_gray(REF_DIR / f"porta_{n}.jpg")
        if ref is None:
            print(f"  MISSING reference porta_{n}.jpg")
            continue
        rh, rw = ref.shape[:2]
        for s in SCEN:
            tp = TEST_ROOT / s / f"porta_{n}.jpg"
            if not tp.exists():
                print(f"{s:24s} porta_{n}  MISSING FILE")
                continue
            test = read_gray(tp)
            if test is None:
                print(f"{s:24s} porta_{n}  UNREADABLE")
                continue
            try:
                if HAVE_CV2:
                    cc, dx, dy, rot = fit_ecc_cv2(ref, test)
                else:
                    cc, dx, dy, rot = fit_phasecorr_numpy(ref, test)
            except Exception as e:  # noqa: BLE001
                print(f"{s:24s} porta_{n}  FIT-FAIL: {e}")
                continue
            dxf = abs(dx) / rw
            dyf = abs(dy) / rh
            rot_ok = np.isnan(rot) or abs(rot) <= TOL_ROT
            aligned = dxf <= TOL_FRAC and dyf <= TOL_FRAC and rot_ok
            verdict = "ALIGNED" if aligned else "OFF"
            rot_str = "  n/a " if np.isnan(rot) else f"{rot:7.2f}"
            print(f"{s:24s} porta_{n}  {cc:6.3f} {dx:8.1f} {dy:8.1f} {rot_str} "
                  f"{100*dxf:6.2f} {100*dyf:6.2f}  {verdict}")
            rows.append((s, n, cc, dx, dy, rot, dxf, dyf, aligned))

    if not rows:
        return
    import statistics as st
    adx = [abs(r[3]) for r in rows]
    ady = [abs(r[4]) for r in rows]
    arot = [abs(r[5]) for r in rows if not np.isnan(r[5])]
    ccs = [r[2] for r in rows]
    n_aligned = sum(1 for r in rows if r[8])
    print(f"\n=== Summary over {len(rows)} test images ({n_aligned} ALIGNED / "
          f"{len(rows)-n_aligned} OFF) ===")
    print(f"  |dx| px : median={st.median(adx):6.2f}  mean={st.mean(adx):6.2f}  max={max(adx):6.2f}")
    print(f"  |dy| px : median={st.median(ady):6.2f}  mean={st.mean(ady):6.2f}  max={max(ady):6.2f}")
    if arot:
        print(f"  |rot| ° : median={st.median(arot):6.2f}  mean={st.mean(arot):6.2f}  max={max(arot):6.2f}")
    print(f"  cc      : median={st.median(ccs):.3f}  min={min(ccs):.3f}")
    # Per-scenario breakdown of mean residual translation.
    print("\n  Per-scenario mean |dx|,|dy| px:")
    for s in SCEN:
        sub = [r for r in rows if r[0] == s]
        if sub:
            mdx = st.mean(abs(r[3]) for r in sub)
            mdy = st.mean(abs(r[4]) for r in sub)
            print(f"    {s:24s} |dx|={mdx:6.1f}  |dy|={mdy:6.1f}")


if __name__ == "__main__":
    main()
