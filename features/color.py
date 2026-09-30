"""Vessel color features — arteriosclerosis (동맥경화) 조기 신호 탐지 목적.

**임상 배경**:
- 정상 동맥: 밝은 빨강, 중앙 광반사 (CLR) 얇음
- Copper wire sign: 벽 비후 → 황갈색 톤 (early sclerosis)
- Silver wire sign: 완전 회백화 (advanced sclerosis)

**아이디어**: Vessel skeleton (중심선) 위에서 RGB sampling → 색감 통계.
Sclerotic vessel 은 색이 탈색되어 achromatic 해짐 → **whiteness ↑ / R-G ratio ↓**.

**추출 지표** (per network: vessel / artery / vein):
- `mean_R, mean_G, mean_B` — 각 채널 평균
- `mean_intensity` — 회색값 평균
- `R_G_ratio, R_B_ratio` — 채널 비율 (achromaticity 지표)
- `whiteness_mean, whiteness_p95` — 색 탈색도 (1 - channel_std/intensity)
- `AV_intensity_ratio, AV_whiteness_ratio` — artery vs vein 상대 지표 (경화 marker)
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .skeleton import clean_and_skeletonize


def color_stats_at_skeleton(
    rgb: np.ndarray,
    mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_object_px: int = 30,
    max_hole_px: int = 20,
    prefix: str = "",
) -> Dict[str, float]:
    """Skeleton pixel 에서 RGB 값 sampling → 색상 통계.

    Returns:
        9 features: mean_R/G/B, mean_intensity, R_G_ratio, R_B_ratio,
                    whiteness_mean, whiteness_p95, n_samples
    """
    skel = clean_and_skeletonize(mask, min_object_px, max_hole_px)
    if valid_mask is not None:
        skel = skel & valid_mask

    nan_out = {
        f"{prefix}mean_R":         np.nan,
        f"{prefix}mean_G":         np.nan,
        f"{prefix}mean_B":         np.nan,
        f"{prefix}mean_intensity": np.nan,
        f"{prefix}R_G_ratio":      np.nan,
        f"{prefix}R_B_ratio":      np.nan,
        f"{prefix}whiteness_mean": np.nan,
        f"{prefix}whiteness_p95":  np.nan,
        f"{prefix}n_samples":      0,
    }
    if not skel.any():
        return nan_out

    r = rgb[..., 0].astype(np.float32)
    g = rgb[..., 1].astype(np.float32)
    b = rgb[..., 2].astype(np.float32)

    r_at = r[skel]
    g_at = g[skel]
    b_at = b[skel]
    intensity = (r_at + g_at + b_at) / 3.0
    if len(intensity) == 0:
        return nan_out

    # Whiteness: achromaticity 지표. 채널 std 가 작을수록 (R≈G≈B) → 회색톤/은백
    channel_std = np.std(np.stack([r_at, g_at, b_at], axis=0), axis=0)
    whiteness   = 1.0 - channel_std / (intensity + 1e-6)

    return {
        f"{prefix}mean_R":         float(r_at.mean()),
        f"{prefix}mean_G":         float(g_at.mean()),
        f"{prefix}mean_B":         float(b_at.mean()),
        f"{prefix}mean_intensity": float(intensity.mean()),
        f"{prefix}R_G_ratio":      float(r_at.mean() / (g_at.mean() + 1e-6)),
        f"{prefix}R_B_ratio":      float(r_at.mean() / (b_at.mean() + 1e-6)),
        f"{prefix}whiteness_mean": float(whiteness.mean()),
        f"{prefix}whiteness_p95":  float(np.percentile(whiteness, 95)),
        f"{prefix}n_samples":      int(skel.sum()),
    }


def whole_image_color_features(
    rgb: np.ndarray,
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 이미지 vessel / artery / vein skeleton 색상 통계 + AV 비율.

    Returns:
        3 network × 9 stat = 27 features
        + AV 비교 3 features (intensity ratio, whiteness ratio, R_G_ratio ratio)
        = **30 features**
    """
    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))

    out: Dict[str, float] = {}
    per_tag: Dict[str, Dict[str, float]] = {}
    for tag, mask in [("vessel", vessel_bin), ("artery", artery_bin), ("vein", vein_bin)]:
        feats = color_stats_at_skeleton(rgb, mask, valid_mask=valid_mask,
                                        prefix=f"{prefix}{tag}_")
        out.update(feats)
        per_tag[tag] = feats

    # AV 비교 지표 — artery / vein
    def _safe_ratio(a, b):
        if not (np.isfinite(a) and np.isfinite(b)) or abs(b) < 1e-6:
            return np.nan
        return float(a / b)

    a_int = per_tag["artery"][f"{prefix}artery_mean_intensity"]
    v_int = per_tag["vein"]  [f"{prefix}vein_mean_intensity"]
    a_wht = per_tag["artery"][f"{prefix}artery_whiteness_mean"]
    v_wht = per_tag["vein"]  [f"{prefix}vein_whiteness_mean"]
    a_rg  = per_tag["artery"][f"{prefix}artery_R_G_ratio"]
    v_rg  = per_tag["vein"]  [f"{prefix}vein_R_G_ratio"]

    out[f"{prefix}AV_intensity_ratio"] = _safe_ratio(a_int, v_int)
    out[f"{prefix}AV_whiteness_ratio"] = _safe_ratio(a_wht, v_wht)
    out[f"{prefix}AV_RG_ratio"]        = _safe_ratio(a_rg,  v_rg)

    return out
