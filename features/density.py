"""Density-based vessel features within arbitrary regions of interest.

기본 3 종 density (per ROI):
    vessel_density  = |vessel ∩ roi|   / |roi|
    artery_density  = |A     ∩ roi|   / |roi|      (av in {1, 3})
    vein_density    = |V     ∩ roi|   / |roi|      (av in {2, 3})
    AV_ratio_area   = |A|   / |V|                   (ROI 안)
    disc_frac       = |disc ∩ roi|    / |roi|
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def region_density_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    disc_mask: np.ndarray,
    roi_mask: np.ndarray,
    prefix: str = "",
) -> Dict[str, float]:
    """단일 ROI 안에서 vessel / A / V / disc density 계산."""
    roi_n = max(int(roi_mask.sum()), 1)
    v_in = (vessel_mask > 0) & roi_mask
    art  = np.isin(av_mask, (1, 3)) & roi_mask
    vei  = np.isin(av_mask, (2, 3)) & roi_mask
    disc_in = (disc_mask > 0) & roi_mask
    p = prefix
    # NOTE: roi_area_px 제거 (zone geometry — aligned 에선 상수, biological signal 아님).
    # vessel_area_px, disc_area_px 는 실제 segmentation mask 픽셀 수라 biological 정보 있음 → 유지.
    return {
        f"{p}vessel_density":  v_in.sum() / roi_n,
        f"{p}artery_density":  art.sum()  / roi_n,
        f"{p}vein_density":    vei.sum()  / roi_n,
        f"{p}AV_ratio_area":   art.sum() / max(int(vei.sum()), 1),
        f"{p}disc_frac":       disc_in.sum() / roi_n,
        f"{p}vessel_area_px":  int(v_in.sum()),
        f"{p}disc_area_px":    int(disc_in.sum()),
    }


def whole_image_density_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    disc_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 fundus (valid_mask) 안 density — zone 무관 전역 지표.

    Returns:
        `{prefix}{metric}` 형태 8 개 core feature.
    """
    roi = valid_mask if valid_mask is not None else np.ones_like(vessel_mask, dtype=bool)
    return region_density_features(vessel_mask, av_mask, disc_mask, roi, prefix=prefix)


def zone_density_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    disc_mask: np.ndarray,
    zones: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """여러 zone 에 대해 loop 로 density 계산.

    Args:
        zones: {zone_name: bool 2D mask}
        valid_mask: 옵션. 있으면 각 zone 을 이거랑 intersection 해서
                    invalid 영역 (예: fundus 밖, common_valid 밖) 제외.

    Returns:
        {"<zone>_vessel_density": ..., "<zone>_artery_density": ..., ...}
    """
    out: Dict[str, float] = {}
    for zone_name, zone in zones.items():
        z = zone & valid_mask if valid_mask is not None else zone
        feats = region_density_features(
            vessel_mask, av_mask, disc_mask, z, prefix=f"{zone_name}_"
        )
        out.update(feats)
    return out
