"""Anatomical zones for retinal feature extraction.

두 종류:

1. **True clinical ETDRS 9-subfield** (Early Treatment Diabetic Retinopathy Study)
   - **mm 단위**, disc 실제 지름 = 1.8 mm 표준 가정
   - Fovea 중심, 3 개 동심원 (지름 1 / 3 / 6 mm)
   - Central + inner 4 quadrant + outer 4 quadrant = **9 subfields**
   - 원 안 4-quadrant 는 ±45° 대각선으로 분할
   - Superior / Inferior 는 image 위/아래 (양쪽 눈 공통)
   - Nasal / Temporal 는 disc 방향에 따라 결정 (eye 정보 필요)
     - Right eye: 이미지 상 disc 는 fovea 왼쪽 → 왼쪽 wedge = Nasal, 오른쪽 = Temporal
     - Left eye:  이미지 상 disc 는 fovea 오른쪽 → 오른쪽 wedge = Nasal, 왼쪽 = Temporal

2. **DD-scaled 3-ring** (scale-invariant, 임상 표준 아님)
   - Disc diameter (DD) 단위, camera / eye 크기 무관
   - central / parafovea / perifovea 3 개 링
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# ─────────────────── mm-based clinical ETDRS ───────────────────

STANDARD_DISC_MM = 1.8   # 성인 disc 평균 지름 (mm) — 대부분의 논문 표준

# 9-subfield ETDRS grid: (r_central_mm, r_inner_mm, r_outer_mm)
DEFAULT_ETDRS_RADII_MM = (0.5, 1.5, 3.0)

# 9-subfield zone 이름 (clinical convention)
ETDRS_9_NAMES = ["C", "S1", "N1", "I1", "T1", "S2", "N2", "I2", "T2"]


def mm_per_px_from_disc(disc_diameter_px: float,
                        standard_disc_mm: float = STANDARD_DISC_MM) -> float:
    """Disc 지름 픽셀 값 → mm/pixel 환산 factor.

    성인 disc 지름 ≈ 1.8 mm 표준 가정. 개별 편차 (근시 등) 는 무시.
    """
    return standard_disc_mm / disc_diameter_px


def make_etdrs_9subfields(
    fovea_x: float,
    fovea_y: float,
    eye: str,
    mm_per_px: float,
    image_shape: Tuple[int, int],
    radii_mm: Tuple[float, float, float] = DEFAULT_ETDRS_RADII_MM,
) -> Dict[str, np.ndarray]:
    """True clinical ETDRS 9-subfield grid.

    Args:
        fovea_x, fovea_y: fovea 중심 픽셀 좌표
        eye: 'L' or 'R' — nasal 방향 결정
        mm_per_px: 픽셀당 mm (`mm_per_px_from_disc()` 로 계산)
        image_shape: (H, W)
        radii_mm: (r_central, r_inner, r_outer) mm. 기본 (0.5, 1.5, 3.0)

    Returns:
        dict[str, bool array] — 9 keys: C, S1, N1, I1, T1, S2, N2, I2, T2
    """
    assert eye in ("L", "R"), f"eye must be 'L' or 'R', got {eye}"
    r_c_mm, r_i_mm, r_o_mm = radii_mm

    H, W = image_shape
    yy, xx = np.mgrid[:H, :W]
    dx = xx - fovea_x
    dy = yy - fovea_y
    r_mm = np.hypot(dx, dy) * mm_per_px

    # 링 마스크
    ring_central = r_mm < r_c_mm
    ring_inner   = (r_mm >= r_c_mm) & (r_mm < r_i_mm)
    ring_outer   = (r_mm >= r_i_mm) & (r_mm < r_o_mm)

    # 4-quadrant 분할: ±45° 대각선으로 4 wedge
    # angle (image coords: y increases downward)
    #   0° = right, 90° = down, ±180° = left, -90° = up
    ang_deg = np.degrees(np.arctan2(dy, dx))

    # wedge 정의 (image coords 기준)
    wedge_up    = (ang_deg >= -135) & (ang_deg < -45)   # Superior
    wedge_down  = (ang_deg >=   45) & (ang_deg < 135)   # Inferior
    wedge_right = (ang_deg >=  -45) & (ang_deg <  45)
    wedge_left  = (ang_deg >= 135) | (ang_deg < -135)

    # eye 에 따라 left/right → nasal/temporal 매핑
    # 본 파이프라인 convention: disc 가 이미지 R 쪽에 있으면 eye="R" (OD), L 쪽에 있으면 eye="L" (OS)
    # Nasal 은 항상 disc 쪽, Temporal 은 반대쪽.
    if eye == "R":
        # Right eye: disc 는 이미지 오른쪽 → 오른쪽 wedge = Nasal
        wedge_nasal, wedge_temporal = wedge_right, wedge_left
    else:  # eye == "L"
        # Left eye: disc 는 이미지 왼쪽 → 왼쪽 wedge = Nasal
        wedge_nasal, wedge_temporal = wedge_left, wedge_right

    return {
        "C":  ring_central,
        # inner ring 4-quadrant
        "S1": ring_inner & wedge_up,
        "N1": ring_inner & wedge_nasal,
        "I1": ring_inner & wedge_down,
        "T1": ring_inner & wedge_temporal,
        # outer ring 4-quadrant
        "S2": ring_outer & wedge_up,
        "N2": ring_outer & wedge_nasal,
        "I2": ring_outer & wedge_down,
        "T2": ring_outer & wedge_temporal,
    }


# ─────────────────── DD-scaled 3-ring (기존, scale-invariant) ───────────────────

DEFAULT_DD_ZONES: Dict[str, Tuple[float, float]] = {
    "central":   (0.0, 0.5),
    "parafovea": (0.5, 1.5),
    "perifovea": (1.5, 3.0),
}

# Wong et al. peripapillary zones — disc **margin** 기준, 겹치지 않게 분리
# (원래 Wong 2004 에서 zone_C 는 0.5-2.0 (B 와 겹침) 이나 리포팅 혼동 방지 위해 non-overlap 사용)
DEFAULT_DISC_MARGIN_ZONES: Dict[str, Tuple[float, float]] = {
    "zone_B": (0.5, 1.0),   # inner peripapillary annulus (Knudtson CRAE/CRVE 측정 영역)
    "zone_C": (1.0, 2.0),   # outer peripapillary annulus
}

DISC_ZONE_NAMES = ["zone_B", "zone_C"]


def make_annular_zones(
    center_xy: Tuple[float, float],
    disc_diameter_px: float,
    image_shape: Tuple[int, int],
    zones: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    """Concentric annular zone masks — DD 단위 (scale-invariant)."""
    if zones is None:
        zones = DEFAULT_DD_ZONES
    H, W = image_shape
    cx, cy = center_xy
    yy, xx = np.mgrid[:H, :W]
    r_dd = np.hypot(xx - cx, yy - cy) / disc_diameter_px
    return {name: (r_dd >= lo) & (r_dd < hi) for name, (lo, hi) in zones.items()}


def make_dd_zones(
    fovea_x: float,
    fovea_y: float,
    disc_diameter_px: float,
    image_shape: Tuple[int, int],
    zones: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    """Fovea-centered DD-scaled 3-ring (scale-invariant)."""
    return make_annular_zones((fovea_x, fovea_y), disc_diameter_px, image_shape, zones)


def make_disc_zones(
    disc_cx: float,
    disc_cy: float,
    disc_diameter_px: float,
    image_shape: Tuple[int, int],
    zones: Dict[str, Tuple[float, float]] | None = None,
) -> Dict[str, np.ndarray]:
    """Disc **margin**-centered zones (radii from disc margin, DD units)."""
    if zones is None:
        zones = DEFAULT_DISC_MARGIN_ZONES
    H, W = image_shape
    yy, xx = np.mgrid[:H, :W]
    r_from_center_dd = np.hypot(xx - disc_cx, yy - disc_cy) / disc_diameter_px
    r_from_margin_dd = r_from_center_dd - 0.5
    return {name: (r_from_margin_dd >= lo) & (r_from_margin_dd < hi)
            for name, (lo, hi) in zones.items()}


# ─────────────────── Standard FOV (DD-scaled) ───────────────────

STANDARD_FOV_RADIUS_DD = 3.0   # 기본 반경: 3 × DD (image-centered, macula-centered 이미지 적합)


def make_standard_fov(
    cx: float, cy: float,
    disc_diameter_px: float,
    image_shape: Tuple[int, int],
    radius_dd: float = STANDARD_FOV_RADIUS_DD,
) -> np.ndarray:
    """DD 배수 반경의 원형 FOV ROI.

    Whole-image feature 계산 시 환자간 **해부학 비율** 통일을 위해 사용.
    - R = radius_dd × disc_diameter_px
    - 환자마다 픽셀 반경은 다르지만 DD 단위 반경은 같음 → 공정한 비교

    Args:
        cx, cy: 중심 좌표 (macula-centered 이미지면 image center 추천)
        disc_diameter_px: 이 환자의 disc 지름 (px)
        image_shape: (H, W)
        radius_dd: 반경을 DD 몇 배로 할지 (default 3.0)

    Returns:
        bool 2D mask
    """
    H, W = image_shape
    yy, xx = np.mgrid[:H, :W]
    R = radius_dd * disc_diameter_px
    return (xx - cx) ** 2 + (yy - cy) ** 2 < R ** 2


# ─────────────────── backward-compat 별칭 ───────────────────

# 이전 03-features.ipynb 에서 make_etdrs_zones 이름으로 호출했지만 실제로는 DD-scaled 였음.
# 새 clinical 함수와 구분되므로 DEFAULT_ETDRS 는 제거하고 이름 재정리:
DEFAULT_ETDRS = DEFAULT_DD_ZONES  # 옛 이름 (혹시 남은 참조 대비)
make_etdrs_zones = make_dd_zones  # 옛 이름 = DD-scaled 3-ring
