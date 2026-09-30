"""Skeleton-based vessel features (length, branch points, endpoints) per zone.

**Skeletonize** = vessel mask 를 1-pixel-wide medial axis 로 축약 → 그래프 형태.
그 위에서 픽셀별 이웃 개수로 branch/endpoint 분류.

**Feature per zone × (vessel, artery, vein)**:
- `skel_length_px`    — skeleton pixel 개수 (in-zone) = vessel 총 길이의 approximation
- `n_branches`        — junction 개수 (skeleton pixel 중 8-neighbor 3 개 이상)
- `n_endpoints`       — leaf 개수 (8-neighbor 1 개)

**주의**:
- Skeletonize 는 **전체 mask 에서 한 번만** 수행하고 zone 은 나중에 intersection.
  Zone 경계에서 skeleton 이 잘리면 인위적 endpoint 가 생기지만, 상대 비교엔 문제없음.
- Class 3 (교차점) 은 artery, vein 양쪽 skeleton 에 포함 (density 와 동일 컨벤션).
- Skeleton 전 cleaning (remove small holes / objects) 으로 노이즈로 인한
  spurious branch 방지 (fundus-vessel-features/retinal_graph/masks.py 참고).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import convolve, label
from skimage.morphology import (
    remove_small_holes,
    remove_small_objects,
    skeletonize,
)

# 3×3 8-connectivity kernel (자기 자신 제외)
_KERNEL8 = np.array([[1, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]], dtype=np.uint8)


def clean_and_skeletonize(
    mask: np.ndarray,
    min_object_px: int = 30,
    max_hole_px: int = 20,
) -> np.ndarray:
    """Binary mask cleaning + skeletonize.

    - `remove_small_holes` — 벽 안 작은 구멍을 채워서 skeleton loop 방지
    - `remove_small_objects` — 고립된 speckle 제거 (spurious endpoint 방지)
    - `skeletonize` — 1-pixel-wide medial axis, 8-connected, homotopy 보존
    """
    m = np.ascontiguousarray(mask.astype(bool))
    if not m.any():
        return np.zeros_like(m)
    if max_hole_px > 0:
        m = remove_small_holes(m, area_threshold=max_hole_px)
    if min_object_px > 0:
        m = remove_small_objects(m, min_size=min_object_px)
    if not m.any():
        return np.zeros_like(m)
    return skeletonize(m)


def branch_endpoint_masks(skel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Skeleton → (branch_mask, endpoint_mask).

    - branch: 8-neighbor 개수가 3 이상 (junction)
    - endpoint: 8-neighbor 개수가 1 (leaf)
    """
    if not skel.any():
        empty = np.zeros_like(skel, dtype=bool)
        return empty, empty
    neigh = convolve(skel.astype(np.uint8), _KERNEL8, mode="constant", cval=0)
    branches  = skel & (neigh >= 3)
    endpoints = skel & (neigh == 1)
    return branches, endpoints


def _count_clusters(bool_mask: np.ndarray) -> int:
    """Count 8-connected components (physical junction 하나 = 라벨 하나)."""
    if not bool_mask.any():
        return 0
    _, n = label(bool_mask, structure=np.ones((3, 3), dtype=int))
    return int(n)


def region_skeleton_features(
    skel: np.ndarray,
    branches: np.ndarray,
    endpoints: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    prefix: str = "",
) -> Dict[str, float]:
    """단일 ROI 안에서 skeleton length + branch/endpoint 개수 계산."""
    if roi_mask is not None:
        skel_z      = skel      & roi_mask
        branches_z  = branches  & roi_mask
        endpoints_z = endpoints & roi_mask
    else:
        skel_z, branches_z, endpoints_z = skel, branches, endpoints

    p = prefix
    return {
        f"{p}skel_length_px": int(skel_z.sum()),
        f"{p}n_branches":     _count_clusters(branches_z),
        f"{p}n_endpoints":    _count_clusters(endpoints_z),
    }


def zone_skeleton_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    zones: Dict[str, np.ndarray],
    valid_mask: Optional[np.ndarray] = None,
    min_object_px: int = 30,
    max_hole_px: int = 20,
) -> Dict[str, float]:
    """모든 zone × (vessel / artery / vein) 에 대해 skeleton feature 계산.

    Skeletonization 은 이미지 당 3 번만 (vessel / A / V 각) 수행 후 zone 은 intersection.

    Returns:
        `{zone_name}_{tag}_{feat}` 형태의 dict.
        3 features × 3 networks × N zones.
    """
    # 3 개 network 준비
    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))

    # 한 번씩 skeletonize
    vessel_skel = clean_and_skeletonize(vessel_bin, min_object_px, max_hole_px)
    artery_skel = clean_and_skeletonize(artery_bin, min_object_px, max_hole_px)
    vein_skel   = clean_and_skeletonize(vein_bin,   min_object_px, max_hole_px)

    # branch / endpoint 마스크
    v_br, v_ep = branch_endpoint_masks(vessel_skel)
    a_br, a_ep = branch_endpoint_masks(artery_skel)
    n_br, n_ep = branch_endpoint_masks(vein_skel)

    tags = [
        ("vessel", vessel_skel, v_br, v_ep),
        ("artery", artery_skel, a_br, a_ep),
        ("vein",   vein_skel,   n_br, n_ep),
    ]

    out: Dict[str, float] = {}
    for zone_name, zone in zones.items():
        z = zone & valid_mask if valid_mask is not None else zone
        for tag, skel, br, ep in tags:
            feats = region_skeleton_features(
                skel, br, ep, roi_mask=z,
                prefix=f"{zone_name}_{tag}_",
            )
            out.update(feats)
    return out


def whole_image_skeleton_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    min_object_px: int = 30,
    max_hole_px: int = 20,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 이미지 (valid_mask 안) 에서 vessel/artery/vein skeleton feature 계산.

    Zone 기반 skeleton 은 sample 부족해서 noisy → whole-image 로만 뽑는 게 안정적.

    Returns:
        `{prefix}{tag}_{feat}` 형태 dict. 3 network × 3 metric = **9 features**.
    """
    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))

    v_skel = clean_and_skeletonize(vessel_bin, min_object_px, max_hole_px)
    a_skel = clean_and_skeletonize(artery_bin, min_object_px, max_hole_px)
    n_skel = clean_and_skeletonize(vein_bin,   min_object_px, max_hole_px)

    v_br, v_ep = branch_endpoint_masks(v_skel)
    a_br, a_ep = branch_endpoint_masks(a_skel)
    n_br, n_ep = branch_endpoint_masks(n_skel)

    out: Dict[str, float] = {}
    for tag, skel, br, ep in [
        ("vessel", v_skel, v_br, v_ep),
        ("artery", a_skel, a_br, a_ep),
        ("vein",   n_skel, n_br, n_ep),
    ]:
        out.update(region_skeleton_features(
            skel, br, ep, roi_mask=valid_mask, prefix=f"{prefix}{tag}_"))
    return out


def _segment_lengths(skel: np.ndarray) -> np.ndarray:
    """Skeleton → branch pixel 제거 → connected components 의 길이 (픽셀 수) 배열."""
    if not skel.any():
        return np.array([], dtype=int)
    neigh = convolve(skel.astype(np.uint8), _KERNEL8, mode="constant", cval=0)
    branches = skel & (neigh >= 3)
    segments = skel & ~branches
    if not segments.any():
        return np.array([], dtype=int)
    labels, n = label(segments, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return np.array([], dtype=int)
    counts = np.bincount(labels.ravel())[1:]   # skip 0 (bg)
    return counts.astype(int)


def segment_length_stats(
    skel: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    prefix: str = "",
    min_length_px: int = 3,
) -> Dict[str, float]:
    """Skeleton segment 별 길이 통계.

    Returns:
        `{prefix}n_segments`, `{prefix}seg_len_mean/median/max/std`
    """
    if roi_mask is not None:
        skel = skel & roi_mask
    lengths = _segment_lengths(skel)
    lengths = lengths[lengths >= min_length_px]
    p = prefix
    if len(lengths) == 0:
        return {
            f"{p}n_segments":     0,
            f"{p}seg_len_mean":   np.nan,
            f"{p}seg_len_median": np.nan,
            f"{p}seg_len_max":    np.nan,
            f"{p}seg_len_std":    np.nan,
        }
    return {
        f"{p}n_segments":     int(len(lengths)),
        f"{p}seg_len_mean":   float(np.mean(lengths)),
        f"{p}seg_len_median": float(np.median(lengths)),
        f"{p}seg_len_max":    int(np.max(lengths)),
        f"{p}seg_len_std":    float(np.std(lengths)),
    }


def whole_image_segment_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """전체 이미지 vessel/artery/vein segment 길이 통계.

    Returns:
        `{prefix}{tag}_{stat}` — 3 network × 5 stat = **15 features**.
    """
    vessel_bin = vessel_mask > 0
    artery_bin = np.isin(av_mask, (1, 3))
    vein_bin   = np.isin(av_mask, (2, 3))

    v_skel = clean_and_skeletonize(vessel_bin)
    a_skel = clean_and_skeletonize(artery_bin)
    n_skel = clean_and_skeletonize(vein_bin)

    out: Dict[str, float] = {}
    for tag, skel in [("vessel", v_skel), ("artery", a_skel), ("vein", n_skel)]:
        out.update(segment_length_stats(skel, roi_mask=valid_mask, prefix=f"{prefix}{tag}_"))
    return out


def segment_tortuosity_list(
    skel: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    min_length_px: int = 5,
) -> np.ndarray:
    """각 segment 의 tortuosity = arc_length / chord_length.

    arc = segment pixel 수 (1-px wide skeleton 이므로 픽셀 수 ≈ 실제 길이)
    chord = segment 안 가장 멀리 떨어진 두 픽셀 거리 (max pairwise distance)
    """
    from scipy.spatial.distance import pdist
    if roi_mask is not None:
        skel = skel & roi_mask
    if not skel.any():
        return np.array([])
    neigh = convolve(skel.astype(np.uint8), _KERNEL8, mode="constant", cval=0)
    branches = skel & (neigh >= 3)
    segments = skel & ~branches
    if not segments.any():
        return np.array([])
    labels, n = label(segments, structure=np.ones((3, 3), dtype=int))
    torts = []
    for lbl in range(1, n + 1):
        ys, xs = np.where(labels == lbl)
        if len(xs) < min_length_px:
            continue
        arc = float(len(xs))
        pts = np.column_stack([xs, ys])
        if len(pts) > 200:
            idx = np.linspace(0, len(pts) - 1, 200).astype(int)
            pts = pts[idx]
        dists = pdist(pts)
        chord = float(dists.max()) if len(dists) > 0 else 1.0
        if chord < 1.0:
            continue
        torts.append(arc / chord)
    return np.array(torts)


def tortuosity_stats(torts: np.ndarray, prefix: str = "") -> Dict[str, float]:
    p = prefix
    if len(torts) == 0:
        return {f"{p}tortuosity_mean": np.nan,
                f"{p}tortuosity_median": np.nan,
                f"{p}tortuosity_max": np.nan,
                f"{p}tortuosity_p90": np.nan}
    return {
        f"{p}tortuosity_mean":   float(np.mean(torts)),
        f"{p}tortuosity_median": float(np.median(torts)),
        f"{p}tortuosity_max":    float(np.max(torts)),
        f"{p}tortuosity_p90":    float(np.percentile(torts, 90)),
    }


def whole_image_tortuosity_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """3 network × 4 stat = **12 features**."""
    v_skel = clean_and_skeletonize(vessel_mask > 0)
    a_skel = clean_and_skeletonize(np.isin(av_mask, (1, 3)))
    n_skel = clean_and_skeletonize(np.isin(av_mask, (2, 3)))
    out: Dict[str, float] = {}
    for tag, skel in [("vessel", v_skel), ("artery", a_skel), ("vein", n_skel)]:
        torts = segment_tortuosity_list(skel, roi_mask=valid_mask)
        out.update(tortuosity_stats(torts, prefix=f"{prefix}{tag}_"))
    return out


# ═════════════════ Bifurcation angle (Murray's law) ═════════════════


OPTIMAL_BIFURCATION_ANGLE = 82.5   # Murray's law 최적각 (°)


def bifurcation_angles(
    skel: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    sample_dist: int = 5,
) -> np.ndarray:
    """Branch point 3-arm 별 pairwise 각도 (°) 리스트."""
    if roi_mask is not None:
        skel = skel & roi_mask
    if not skel.any():
        return np.array([])
    neigh = convolve(skel.astype(np.uint8), _KERNEL8, mode="constant", cval=0)
    branches = skel & (neigh >= 3)
    if not branches.any():
        return np.array([])

    br_labels, n_junc = label(branches, structure=np.ones((3, 3), dtype=int))
    skel_no_junc = skel & ~branches
    H, W = skel.shape
    angles_all = []

    for jl in range(1, n_junc + 1):
        junc = br_labels == jl
        ys, xs = np.where(junc)
        cx = float(xs.mean()); cy = float(ys.mean())
        pad = sample_dist + 3
        y0 = int(max(0, cy - pad)); y1 = int(min(H, cy + pad))
        x0 = int(max(0, cx - pad)); x1 = int(min(W, cx + pad))
        local = skel_no_junc[y0:y1, x0:x1]
        if not local.any():
            continue
        lbl_l, n_l = label(local, structure=np.ones((3, 3), dtype=int))
        if n_l < 3:
            continue
        directions = []
        lcx = cx - x0; lcy = cy - y0
        for comp in range(1, n_l + 1):
            cy_pts, cx_pts = np.where(lbl_l == comp)
            d_from_junc = np.hypot(cx_pts - lcx, cy_pts - lcy)
            # 가장 sample_dist 에 가까운 점을 대표
            far_idx = np.argmin(np.abs(d_from_junc - sample_dist))
            dx = cx_pts[far_idx] - lcx
            dy = cy_pts[far_idx] - lcy
            norm = np.hypot(dx, dy)
            if norm > 0:
                directions.append((dx / norm, dy / norm))
        if len(directions) < 3:
            continue
        # 3 개 daughters — pairwise 각도
        for i in range(3):
            for j in range(i + 1, 3):
                d1, d2 = directions[i], directions[j]
                cos_t = np.clip(d1[0] * d2[0] + d1[1] * d2[1], -1, 1)
                angles_all.append(float(np.degrees(np.arccos(cos_t))))
    return np.array(angles_all)


def bifurcation_angle_stats(angles: np.ndarray, prefix: str = "") -> Dict[str, float]:
    p = prefix
    if len(angles) == 0:
        return {f"{p}bifurc_angle_mean": np.nan,
                f"{p}bifurc_angle_std":  np.nan,
                f"{p}bifurc_dev_from_optimal": np.nan,
                f"{p}bifurc_n": 0}
    return {
        f"{p}bifurc_angle_mean":       float(np.mean(angles)),
        f"{p}bifurc_angle_std":        float(np.std(angles)),
        f"{p}bifurc_dev_from_optimal": float(np.mean(np.abs(angles - OPTIMAL_BIFURCATION_ANGLE))),
        f"{p}bifurc_n":                int(len(angles)),
    }


def whole_image_bifurcation_features(
    vessel_mask: np.ndarray,
    av_mask: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    prefix: str = "whole_",
) -> Dict[str, float]:
    """3 network × 4 stat = **12 features**."""
    v_skel = clean_and_skeletonize(vessel_mask > 0)
    a_skel = clean_and_skeletonize(np.isin(av_mask, (1, 3)))
    n_skel = clean_and_skeletonize(np.isin(av_mask, (2, 3)))
    out: Dict[str, float] = {}
    for tag, skel in [("vessel", v_skel), ("artery", a_skel), ("vein", n_skel)]:
        angles = bifurcation_angles(skel, roi_mask=valid_mask)
        out.update(bifurcation_angle_stats(angles, prefix=f"{prefix}{tag}_"))
    return out


def make_skeletons(vessel_mask: np.ndarray, av_mask: np.ndarray,
                   min_object_px: int = 30, max_hole_px: int = 20
                   ) -> Dict[str, np.ndarray]:
    """편의 함수 — vessel/artery/vein skeleton 3 개 반환 (시각화용)."""
    return {
        "vessel": clean_and_skeletonize(vessel_mask > 0, min_object_px, max_hole_px),
        "artery": clean_and_skeletonize(np.isin(av_mask, (1, 3)), min_object_px, max_hole_px),
        "vein":   clean_and_skeletonize(np.isin(av_mask, (2, 3)), min_object_px, max_hole_px),
    }
