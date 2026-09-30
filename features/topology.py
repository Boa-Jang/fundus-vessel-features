"""Topological / global vessel features:

- **Fractal dimension** (box-counting) — vessel network 의 self-similarity / complexity
- **Sholl analysis** — fovea/disc 중심 동심원 반경별 skeleton 교차 수 곡선

둘 다 whole-image scalar → per-network (vessel / artery / vein) 각각.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .skeleton import clean_and_skeletonize

# ═════════════════ Fractal dimension ═════════════════


DEFAULT_BOX_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]


def fractal_dim_boxcount(mask: np.ndarray,
                         box_sizes: Sequence[int] = DEFAULT_BOX_SIZES) -> float:
    """Box-counting fractal dimension.

    `log(N(s)) = -D * log(s) + c` → D = -slope

    Args:
        mask: 2D bool (or 0/1) — vessel mask or skeleton
        box_sizes: 여러 box size (픽셀). 큰 것 → 작은 것 순 관계 없음.

    Returns:
        Fractal dimension. 실패 시 NaN.
    """
    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any():
        return float("nan")
    H, W = m.shape
    counts, valid_s = [], []
    for s in box_sizes:
        if s > min(H, W): continue
        nH, nW = H // s, W // s
        if nH == 0 or nW == 0: continue
        block = m[:nH*s, :nW*s].reshape(nH, s, nW, s)
        n_nonempty = int((block.sum(axis=(1, 3)) > 0).sum())
        if n_nonempty > 0:
            counts.append(n_nonempty)
            valid_s.append(s)
    if len(counts) < 3:
        return float("nan")
    log_s = np.log(np.array(valid_s, dtype=float))
    log_n = np.log(np.array(counts, dtype=float))
    slope, _ = np.polyfit(log_s, log_n, 1)
    return float(-slope)


def whole_image_fractal_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 이미지 vessel/artery/vein 의 box-count fractal dim.

    Returns:
        3 features: `{prefix}{tag}_fractal_dim`
    """
    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))
    if valid_mask is not None:
        vessel_bin &= valid_mask
        artery_bin &= valid_mask
        vein_bin   &= valid_mask

    return {
        f"{prefix}vessel_fractal_dim": fractal_dim_boxcount(vessel_bin),
        f"{prefix}artery_fractal_dim": fractal_dim_boxcount(artery_bin),
        f"{prefix}vein_fractal_dim":   fractal_dim_boxcount(vein_bin),
    }


# ═════════════════ Sholl analysis ═════════════════


def sholl_curve(skel: np.ndarray, cx: float, cy: float,
                radii_px: Sequence[float]) -> np.ndarray:
    """반경 r (픽셀) 별 skeleton 교차 수. 각 r 은 ±0.5 픽셀 tolerance 원환.

    Returns:
        len(radii_px) 길이 1D 배열 (int)
    """
    if not skel.any():
        return np.zeros(len(radii_px), dtype=int)
    H, W = skel.shape
    yy, xx = np.mgrid[:H, :W]
    r_map = np.hypot(xx - cx, yy - cy)
    counts = np.empty(len(radii_px), dtype=int)
    for i, r in enumerate(radii_px):
        ring = (r_map >= r - 0.5) & (r_map < r + 0.5)
        counts[i] = int((skel & ring).sum())
    return counts


def sholl_summary(curve: np.ndarray, radii_px: Sequence[float],
                  prefix: str = "") -> Dict[str, float]:
    """Sholl 곡선 → 4 개 요약 stat."""
    p = prefix
    if not curve.any():
        return {
            f"{p}sholl_max":      0,
            f"{p}sholl_r_max_px": np.nan,
            f"{p}sholl_auc":      0.0,
            f"{p}sholl_mean":     0.0,
        }
    radii_arr = np.asarray(radii_px, dtype=float)
    return {
        f"{p}sholl_max":      int(curve.max()),
        f"{p}sholl_r_max_px": float(radii_arr[int(np.argmax(curve))]),
        f"{p}sholl_auc":      float(np.trapezoid(curve, radii_arr)),
        f"{p}sholl_mean":     float(curve.mean()),
    }


def sholl_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    center_xy: tuple,
    valid_mask: Optional[np.ndarray] = None,
    radii_px: Optional[Sequence[float]] = None,
    center_name: str = "center",
    prefix: str = "sholl_",
) -> Dict[str, float]:
    """단일 center 에서 vessel/artery/vein sholl 분석.

    Args:
        center_xy: (cx, cy) 중심 좌표 (픽셀)
        radii_px: 반경 리스트. None → 10 to 240 stepped 5 (image size 기준)
        center_name: 접두어에 붙일 이름 ('fovea', 'disc' 등)
        prefix: 기본 접두어

    Returns:
        3 networks × 4 stat = **12 features**.
        키 형태: `{prefix}{center_name}_{tag}_sholl_{stat}`
    """
    H, W = vessel_mask.shape
    if radii_px is None:
        max_r = min(H, W) // 2 - 5
        radii_px = list(range(10, max_r, 5))

    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))
    if valid_mask is not None:
        vessel_bin &= valid_mask
        artery_bin &= valid_mask
        vein_bin   &= valid_mask

    v_skel = clean_and_skeletonize(vessel_bin)
    a_skel = clean_and_skeletonize(artery_bin)
    n_skel = clean_and_skeletonize(vein_bin)

    cx, cy = center_xy
    out: Dict[str, float] = {}
    for tag, skel in [("vessel", v_skel), ("artery", a_skel), ("vein", n_skel)]:
        curve = sholl_curve(skel, cx, cy, radii_px)
        out.update(sholl_summary(curve, radii_px,
                                 prefix=f"{prefix}{center_name}_{tag}_"))
    return out
