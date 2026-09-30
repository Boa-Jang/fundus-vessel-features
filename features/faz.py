"""Foveal Avascular Zone (FAZ) — DR / DME / macular ischemia gold-standard 지표.

**정의**: Fovea 중심 무혈관 영역 (건강 시 지름 ~500 μm, 면적 ~0.25 mm²).

**계산**:
1. Fovea 근처 (max_radius) 안에서 vessel mask 의 **complement (non-vessel)** 검출
2. Connected components 중 **fovea 좌표를 포함하는 component** = FAZ
3. 그 mask 로부터 area / diameter / circularity / perimeter 계산

**임상 의미**:
- **정상**: FAZ diameter ≈ 500 μm, circularity > 0.8
- **DR/DME**: FAZ 확대 (허혈), 경계 irregular (circularity ↓)
- **AMD/glaucoma**: FAZ 변화 크지 않음 (감별 지점)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import perimeter


def compute_faz(
    vessel_mask: np.ndarray,
    fovea_xy: Tuple[float, float],
    valid_mask: Optional[np.ndarray] = None,
    mm_per_px: float = 1.0,
    max_radius_dd: float = 1.5,
    disc_diameter_px: Optional[float] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """FAZ mask 추출 + 통계.

    Args:
        vessel_mask: binary vessel mask (H, W)
        fovea_xy: (fx, fy) 픽셀 좌표
        valid_mask: fundus 유효 영역 (검은 배경 제외)
        mm_per_px: 픽셀당 mm — μm 환산에 사용
        max_radius_dd: fovea 로부터 이 반경 안에서만 FAZ 검색 (DD 단위 or 픽셀)
        disc_diameter_px: 주어지면 max_radius_dd 를 DD 단위로 해석
        prefix: feature 이름 접두어

    Returns:
        `{prefix}FAZ_{area_px, area_um2, diameter_um, circularity, perimeter_um, found}`
    """
    p = prefix
    empty = {
        f"{p}FAZ_area_px":     np.nan,
        f"{p}FAZ_area_um2":    np.nan,
        f"{p}FAZ_diameter_um": np.nan,
        f"{p}FAZ_circularity": np.nan,
        f"{p}FAZ_perimeter_um": np.nan,
    }

    H, W = vessel_mask.shape
    fx, fy = float(fovea_xy[0]), float(fovea_xy[1])
    if not (0 <= fx < W and 0 <= fy < H) or np.isnan(fx) or np.isnan(fy):
        return empty

    # 검색 영역 (fovea 주변)
    max_r_px = max_radius_dd * disc_diameter_px if disc_diameter_px else max_radius_dd * 100
    yy, xx = np.mgrid[:H, :W]
    search = np.hypot(xx - fx, yy - fy) < max_r_px
    if valid_mask is not None:
        search = search & valid_mask

    # non-vessel area within search
    non_vessel = ~(vessel_mask > 0) & search

    # Connected components of non-vessel
    labels, n = ndi.label(non_vessel, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return empty

    fx_int, fy_int = int(round(fx)), int(round(fy))
    lbl = int(labels[fy_int, fx_int])
    if lbl == 0:
        # fovea 좌표에 vessel 이 있으면 → 근처 non-vessel 픽셀에서 label 조회
        # 3-pixel search radius
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = fy_int + dy, fx_int + dx
                if 0 <= nx < W and 0 <= ny < H and labels[ny, nx] > 0:
                    lbl = int(labels[ny, nx])
                    break
            if lbl > 0:
                break
        if lbl == 0:
            return empty

    faz = labels == lbl
    area_px = int(faz.sum())
    area_um2 = float(area_px * (mm_per_px * 1000.0) ** 2)
    diameter_um = float(2.0 * np.sqrt(area_um2 / np.pi))
    perim_px = float(perimeter(faz))
    perim_um = float(perim_px * mm_per_px * 1000.0)
    circ = float(4.0 * np.pi * area_px / (perim_px ** 2)) if perim_px > 0 else np.nan

    return {
        f"{p}FAZ_area_px":     area_px,
        f"{p}FAZ_area_um2":    area_um2,
        f"{p}FAZ_diameter_um": diameter_um,
        f"{p}FAZ_circularity": circ,
        f"{p}FAZ_perimeter_um": perim_um,
    }
