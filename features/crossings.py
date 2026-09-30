"""AV crossings (동정맥 교차점) 정량 — 고혈압성 망막병증 marker.

**AV mask class 3** = artery 와 vein 이 겹치는 픽셀 (crossing 지점).

**임상**:
- **A-V nicking** = 동맥이 정맥을 압박하는 sign → 만성 고혈압
- **Salus's sign, Bonnet sign, Gunn sign** 등 세부 소견
- Zone B (peripapillary) 안 crossing 수와 크기가 중요

**추출**:
- Whole / Zone B / Zone C 각각:
  - `n_crossings`: 교차점 (connected component) 개수
  - `total_area_px`: 교차 픽셀 총합
  - `mean_cluster_size`: 평균 cluster 크기

**총 3 area × 3 stat = 9 features**.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import ndimage as ndi


def _crossing_stats(bin_mask: np.ndarray, prefix: str = "") -> Dict[str, float]:
    p = prefix
    if not bin_mask.any():
        return {
            f"{p}n":                 0,
            f"{p}total_area_px":     0,
            f"{p}mean_cluster_size": np.nan,
        }
    labels, n = ndi.label(bin_mask, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return {
            f"{p}n":                 0,
            f"{p}total_area_px":     0,
            f"{p}mean_cluster_size": np.nan,
        }
    sizes = np.bincount(labels.ravel())[1:]      # skip label 0 (bg)
    return {
        f"{p}n":                 int(n),
        f"{p}total_area_px":     int(sizes.sum()),
        f"{p}mean_cluster_size": float(sizes.mean()),
    }


def crossings_features(
    av_mask: np.ndarray,
    zones: Optional[Dict[str, np.ndarray]] = None,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "crossings_",
) -> Dict[str, float]:
    """AV crossings 정량 — whole + Zone B + Zone C.

    Returns:
        `{prefix}whole_{n, total_area_px, mean_cluster_size}` × 3 areas = 9 features.
    """
    crossings = av_mask == 3
    if valid_mask is not None:
        crossings = crossings & valid_mask

    out: Dict[str, float] = {}
    out.update(_crossing_stats(crossings, prefix=f"{prefix}whole_"))

    if zones is not None:
        for zn in ["zone_B", "zone_C"]:
            if zn in zones:
                out.update(_crossing_stats(crossings & zones[zn], prefix=f"{prefix}{zn}_"))
            else:
                out.update({
                    f"{prefix}{zn}_n": 0,
                    f"{prefix}{zn}_total_area_px": 0,
                    f"{prefix}{zn}_mean_cluster_size": np.nan,
                })

    return out
