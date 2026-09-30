"""Caliber (vessel width) features + CRAE / CRVE / AVR.

**Caliber map** = 각 skeleton pixel 에서의 vessel **지름** (= 2 × distance-transform 값).
Distance transform 은 vessel mask 안 픽셀에서 가장 가까운 배경까지의 거리 → radius.
Skeleton pixel 에서 sampling 하면 그 지점의 local vessel radius → 2× = diameter.

**두 종류 feature**:

1. **Zone × network 별 caliber 통계** (mean, median, p75, p90, max, n_samples)
   - `vessel / artery / vein` × 11 zones × 6 stats

2. **Zone B 안 CRAE / CRVE / AVR** (Knudtson 2003 formula)
   - Zone B (0.5-1.0 DD from disc margin) 안 상위 K 개 vessel segment 를 골라
     iterative pairing 으로 조합 지름 (equivalent diameter) 계산
   - CRAE: `sqrt(0.87 a1² + 1.01 a2² - 0.22 a1 a2 - 10.76)` — μm
   - CRVE: `sqrt(0.72 v1² + 0.91 v2² + 450.05)` — μm
   - AVR = CRAE / CRVE — 정상 ≈ 0.67, 낮아지면 고혈압 / 심혈관 위험
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

_KERNEL8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


def compute_caliber_map(
    mask: np.ndarray,
    min_object_px: int = 30,
    max_hole_px: int = 20,
) -> np.ndarray:
    """Skeleton 픽셀 위에서 sampling 한 diameter map. 다른 픽셀은 0.

    Returns:
        float32 2D array. skeleton 이 아닌 곳 = 0, skeleton 에서는 2 × dist_transform.
    """
    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any():
        return np.zeros(mask.shape, dtype=np.float32)
    if max_hole_px > 0:
        m = remove_small_holes(m, area_threshold=max_hole_px)
    if min_object_px > 0:
        m = remove_small_objects(m, min_size=min_object_px)
    if not m.any():
        return np.zeros(mask.shape, dtype=np.float32)

    dist = ndi.distance_transform_edt(m)     # radius
    skel = skeletonize(m)
    return np.where(skel, 2.0 * dist, 0.0).astype(np.float32)


def region_caliber_stats(
    caliber_map: np.ndarray,
    roi_mask: np.ndarray,
    mm_per_px: float = 1.0,
    prefix: str = "",
) -> Dict[str, float]:
    """ROI 안 skeleton pixel 들의 caliber 통계 (μm)."""
    values_px = caliber_map[roi_mask & (caliber_map > 0)]
    if len(values_px) == 0:
        return {
            f"{prefix}caliber_mean_um":   np.nan,
            f"{prefix}caliber_median_um": np.nan,
            f"{prefix}caliber_p75_um":    np.nan,
            f"{prefix}caliber_p90_um":    np.nan,
            f"{prefix}caliber_max_um":    np.nan,
            f"{prefix}caliber_n_samples": 0,
        }
    values_um = values_px * mm_per_px * 1000.0    # px → μm
    return {
        f"{prefix}caliber_mean_um":   float(np.mean(values_um)),
        f"{prefix}caliber_median_um": float(np.median(values_um)),
        f"{prefix}caliber_p75_um":    float(np.percentile(values_um, 75)),
        f"{prefix}caliber_p90_um":    float(np.percentile(values_um, 90)),
        f"{prefix}caliber_max_um":    float(np.max(values_um)),
        f"{prefix}caliber_n_samples": int(len(values_px)),
    }


def whole_image_caliber_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    mm_per_px: float = 1.0,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 이미지 (valid_mask 안) 에서 vessel/artery/vein caliber 통계.

    Zone 단위로는 sample 이 너무 적어 noisy → whole-image 로만 뽑는 게 안정적.

    Returns:
        `{prefix}{tag}_caliber_{stat}` 형태 dict. 3 network × 6 stat = **18 features**.
    """
    vessel_cal = compute_caliber_map(vessel_mask > 0)
    artery_cal = compute_caliber_map(np.isin(av_mask, (1, 3)))
    vein_cal   = compute_caliber_map(np.isin(av_mask, (2, 3)))

    # valid_mask 가 있으면 그 영역만, 없으면 전체 이미지
    roi = valid_mask if valid_mask is not None else np.ones_like(vessel_cal, dtype=bool)

    out: Dict[str, float] = {}
    for tag, cal in [("vessel", vessel_cal), ("artery", artery_cal), ("vein", vein_cal)]:
        out.update(region_caliber_stats(
            cal, roi, mm_per_px=mm_per_px, prefix=f"{prefix}{tag}_"))
    return out


def zone_caliber_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    zones: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
    mm_per_px: float = 1.0,
) -> Dict[str, float]:
    """모든 zone × (vessel / artery / vein) 에 대해 caliber 통계 계산.

    Returns:
        `{zone_name}_{tag}_caliber_{stat}` 형태 dict.
        6 stats × 3 networks × N zones.
    """
    vessel_cal = compute_caliber_map(vessel_mask > 0)
    artery_cal = compute_caliber_map(np.isin(av_mask, (1, 3)))
    vein_cal   = compute_caliber_map(np.isin(av_mask, (2, 3)))

    tags = [("vessel", vessel_cal), ("artery", artery_cal), ("vein", vein_cal)]

    out: Dict[str, float] = {}
    for zone_name, zone in zones.items():
        z = zone & valid_mask if valid_mask is not None else zone
        for tag, cal in tags:
            out.update(region_caliber_stats(
                cal, z, mm_per_px=mm_per_px, prefix=f"{zone_name}_{tag}_"))
    return out


# ═════════════════════ CRAE / CRVE / AVR (Knudtson 2003) ═════════════════════


def segment_width_variability(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_length_px: int = 5,
    min_object_px: int = 30,
    max_hole_px: int = 20,
    prefix: str = "",
) -> Dict[str, float]:
    """각 vessel segment 안 caliber 분산 (CoV = std/mean). Focal narrowing marker.

    Returns:
        `{prefix}width_cov_mean, {prefix}width_cov_max`
    """
    from scipy.ndimage import convolve as _conv
    from skimage.morphology import skeletonize as _skel
    _K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    p = prefix
    nan_out = {f"{p}width_cov_mean": np.nan, f"{p}width_cov_max": np.nan}

    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any(): return nan_out
    if max_hole_px > 0:
        m = remove_small_holes(m, area_threshold=max_hole_px)
    if min_object_px > 0:
        m = remove_small_objects(m, min_size=min_object_px)
    if not m.any(): return nan_out

    dist = ndi.distance_transform_edt(m)
    skel = skeletonize(m)
    if valid_mask is not None:
        skel = skel & valid_mask
    if not skel.any(): return nan_out

    neigh = _conv(skel.astype(np.uint8), _K, mode="constant", cval=0)
    branches = skel & (neigh >= 3)
    segments = skel & ~branches
    if not segments.any(): return nan_out

    labels, n = ndi.label(segments, structure=np.ones((3, 3), int))
    covs = []
    for lbl in range(1, n + 1):
        seg = labels == lbl
        if seg.sum() < min_length_px: continue
        widths = 2.0 * dist[seg]
        m_w = float(widths.mean())
        if m_w < 1.0: continue
        covs.append(float(widths.std() / m_w))
    if len(covs) == 0: return nan_out
    return {
        f"{p}width_cov_mean": float(np.mean(covs)),
        f"{p}width_cov_max":  float(np.max(covs)),
    }


def whole_image_width_variability(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """3 network × 2 stat = **6 features**."""
    out: Dict[str, float] = {}
    for tag, m in [("vessel", vessel_mask > 0),
                   ("artery", np.isin(av_mask, (1, 3))),
                   ("vein",   np.isin(av_mask, (2, 3)))]:
        out.update(segment_width_variability(m, valid_mask=valid_mask, prefix=f"{prefix}{tag}_"))
    return out


# ═════════════════ Junction exponent (Murray's law) ═════════════════


def _solve_murray_x(dp: float, d1: float, d2: float,
                     x_search=np.linspace(1.5, 5.0, 71)) -> float:
    """Solve X for D_p^X = D_1^X + D_2^X.  Returns nan if no valid solution."""
    if not (dp > 0 and d1 > 0 and d2 > 0): return np.nan
    if dp < max(d1, d2): return np.nan       # parent 는 최대여야
    residuals = np.abs(dp**x_search - (d1**x_search + d2**x_search))
    return float(x_search[np.argmin(residuals)])


def junction_exponent_analysis(
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    sample_offset: int = 4,
    min_object_px: int = 30,
    max_hole_px: int = 20,
    prefix: str = "",
) -> Dict[str, float]:
    """각 3-arm branch point 의 Murray junction exponent X 분포.

    각 branch point 에서:
    1. 3 arm 을 skeleton connected component 로 식별
    2. 각 arm 의 sample_offset 픽셀 위치에서 diameter (2 × dist_transform) 측정
    3. 가장 큰 arm = parent (D_p), 나머지 = daughters (D_1, D_2)
    4. Solve X 로 Murray exponent 계산 (정상 ≈ 3)

    Returns:
        `{prefix}junc_X_mean, {prefix}junc_X_std`
    """
    from scipy.ndimage import convolve as _conv
    from skimage.morphology import skeletonize as _skel
    _K = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)

    p = prefix
    nan_out = {f"{p}junc_X_mean": np.nan, f"{p}junc_X_std": np.nan,
               f"{p}junc_n": 0}

    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any(): return nan_out
    if max_hole_px > 0:
        m = remove_small_holes(m, area_threshold=max_hole_px)
    if min_object_px > 0:
        m = remove_small_objects(m, min_size=min_object_px)
    if not m.any(): return nan_out

    dist = ndi.distance_transform_edt(m)
    skel = skeletonize(m)
    if valid_mask is not None:
        skel = skel & valid_mask
    if not skel.any(): return nan_out

    neigh = _conv(skel.astype(np.uint8), _K, mode="constant", cval=0)
    branches = skel & (neigh == 3)     # 3-arm junction 만
    if not branches.any(): return nan_out
    br_labels, n_j = ndi.label(branches, structure=np.ones((3, 3), int))
    skel_no_junc = skel & ~branches

    H, W = skel.shape
    xs_list = []
    for jl in range(1, n_j + 1):
        junc = br_labels == jl
        ys, xs = np.where(junc)
        cx = float(xs.mean()); cy = float(ys.mean())
        pad = sample_offset + 3
        y0 = int(max(0, cy - pad)); y1 = int(min(H, cy + pad))
        x0 = int(max(0, cx - pad)); x1 = int(min(W, cx + pad))
        local_skel = skel_no_junc[y0:y1, x0:x1]
        local_dist = dist[y0:y1, x0:x1]
        if not local_skel.any(): continue
        lbl_l, n_l = ndi.label(local_skel, structure=np.ones((3, 3), int))
        if n_l < 3: continue
        lcx = cx - x0; lcy = cy - y0
        diameters = []
        for comp in range(1, n_l + 1):
            cy_pts, cx_pts = np.where(lbl_l == comp)
            d_from_junc = np.hypot(cx_pts - lcx, cy_pts - lcy)
            # sample_offset 픽셀 위치 (없으면 가장 먼 지점)
            idx = int(np.argmin(np.abs(d_from_junc - sample_offset)))
            dia = 2.0 * float(local_dist[cy_pts[idx], cx_pts[idx]])
            diameters.append(dia)
        if len(diameters) < 3: continue
        diameters.sort(reverse=True)   # 최대 = parent
        dp, d1, d2 = diameters[0], diameters[1], diameters[2]
        X = _solve_murray_x(dp, d1, d2)
        if np.isfinite(X):
            xs_list.append(X)

    if len(xs_list) == 0: return nan_out
    xs_arr = np.array(xs_list)
    return {
        f"{p}junc_X_mean": float(np.mean(xs_arr)),
        f"{p}junc_X_std":  float(np.std(xs_arr)),
        f"{p}junc_n":      int(len(xs_arr)),
    }


def whole_image_junction_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """3 network × 3 stat = **9 features** (X_mean, X_std, n)."""
    out: Dict[str, float] = {}
    for tag, m in [("vessel", vessel_mask > 0),
                   ("artery", np.isin(av_mask, (1, 3))),
                   ("vein",   np.isin(av_mask, (2, 3)))]:
        out.update(junction_exponent_analysis(m, valid_mask=valid_mask, prefix=f"{prefix}{tag}_"))
    return out


def _segment_diameters(
    mask: np.ndarray,
    zone: np.ndarray,
    top_k: int = 6,
    min_seg_length_px: int = 5,
) -> List[float]:
    """Zone 을 통과하는 개별 vessel segment 의 대표 지름 (median) 반환, 내림차순 정렬.

    Segment = skeleton 에서 branch pixel 을 제거하고 남는 connected component.
    """
    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any():
        return []
    m = remove_small_holes(m, area_threshold=20)
    m = remove_small_objects(m, min_size=30)
    if not m.any():
        return []

    dist = ndi.distance_transform_edt(m)
    skel = skeletonize(m)
    if not skel.any():
        return []

    # branch = 8-neighbor ≥ 3
    neigh = convolve(skel.astype(np.uint8), _KERNEL8, mode="constant", cval=0)
    branches = skel & (neigh >= 3)
    segments_mask = skel & ~branches
    if not segments_mask.any():
        return []

    labels, n_labels = ndi.label(segments_mask, structure=np.ones((3, 3), int))
    if n_labels == 0:
        return []

    diameters = []
    for lbl in range(1, n_labels + 1):
        seg = labels == lbl
        seg_in_zone = seg & zone
        if seg_in_zone.sum() < min_seg_length_px:
            continue
        radii = dist[seg_in_zone]
        # 대표 지름 = median radius × 2 (robust)
        diameters.append(2.0 * float(np.median(radii)))

    diameters.sort(reverse=True)
    return diameters[:top_k]


def _knudtson_pair(a: float, b: float, formula: str) -> float:
    """Knudtson iterative pairing 한 단계.

    formula='artery': sqrt(0.87 a² + 1.01 b² - 0.22 a b - 10.76)
    formula='vein':   sqrt(0.72 a² + 0.91 b² + 450.05)
    """
    if formula == "artery":
        val = 0.87 * a * a + 1.01 * b * b - 0.22 * a * b - 10.76
    elif formula == "vein":
        val = 0.72 * a * a + 0.91 * b * b + 450.05
    else:
        raise ValueError(formula)
    return float(np.sqrt(max(val, 0.0)))


def knudtson_equivalent(diameters_um: List[float], formula: str) -> float:
    """Iterative Knudtson pairing → 최종 1 개 조합 지름 (μm)."""
    if not diameters_um:
        return float("nan")
    d = sorted(diameters_um, reverse=True)
    while len(d) > 1:
        n = len(d)
        new = []
        for i in range(n // 2):
            new.append(_knudtson_pair(d[i], d[n - 1 - i], formula))
        if n % 2 == 1:
            new.append(d[n // 2])
        d = sorted(new, reverse=True)
    return d[0]


def compute_crae_crve_avr(
    av_mask: np.ndarray,
    zone_b: np.ndarray,
    mm_per_px: float,
    top_k: int = 6,
) -> Dict[str, float]:
    """Zone B (peripapillary) 안에서 CRAE / CRVE / AVR 계산.

    Args:
        av_mask: 4-class AV mask (0/1/2/3)
        zone_b:  Zone B bool mask (disc-margin 기반)
        mm_per_px: 픽셀당 mm (features.zones.mm_per_px_from_disc 로 계산)
        top_k:   상위 몇 개 segment 를 pairing 에 쓸지 (Knudtson 원본 = 6)
    """
    artery_mask = np.isin(av_mask, (1, 3))
    vein_mask   = np.isin(av_mask, (2, 3))

    art_diams_px = _segment_diameters(artery_mask, zone_b, top_k=top_k)
    vein_diams_px = _segment_diameters(vein_mask,   zone_b, top_k=top_k)

    factor = mm_per_px * 1000.0     # px → μm
    art_diams_um  = [d * factor for d in art_diams_px]
    vein_diams_um = [d * factor for d in vein_diams_px]

    crae = knudtson_equivalent(art_diams_um,  "artery")
    crve = knudtson_equivalent(vein_diams_um, "vein")
    avr  = crae / crve if (np.isfinite(crae) and np.isfinite(crve) and crve > 0) else np.nan

    return {
        "CRAE_um": crae if np.isfinite(crae) else np.nan,
        "CRVE_um": crve if np.isfinite(crve) else np.nan,
        "AVR":     avr,
        "n_artery_segments": len(art_diams_px),
        "n_vein_segments":   len(vein_diams_px),
    }
