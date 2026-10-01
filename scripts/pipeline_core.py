"""Shared pipeline core — 환자 단위 feature 추출.

파이프라인 단계:
1. Preprocess (원본 → RGB + CE, SQUARE_SIZE)
2. VascX 추론 (7 model: quality, av, vessels, disc, fovea, discedge, odfd)
3. Inventory 구축 (eye side, fovea, disc geometry, quality)
4. Registration (aligned 모드) 또는 identity (naive 모드)
5. Feature 추출 (241 features × visit)
6. Patient JSON 저장

디스크 관리:
- keep_masks=True  → 모든 mask 유지 (재분석 가능)
- keep_masks=False → 환자 처리 후 mask 삭제 (JSON 만 남김)
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi
from tqdm import tqdm

# ── features/ 모듈 (같은 repo 안) ──
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from features.zones import (
    make_etdrs_9subfields, make_disc_zones, make_standard_fov,
    mm_per_px_from_disc,
    DEFAULT_ETDRS_RADII_MM, DEFAULT_DISC_MARGIN_ZONES,
    STANDARD_FOV_RADIUS_DD,
)
from features.density import zone_density_features, whole_image_density_features
from features.skeleton import (
    whole_image_skeleton_features, whole_image_segment_features,
    whole_image_tortuosity_features, whole_image_bifurcation_features,
)
from features.caliber import (
    whole_image_caliber_features, compute_crae_crve_avr,
    whole_image_width_variability, whole_image_junction_features,
)
from features.topology import whole_image_fractal_features, sholl_features
from features.color import whole_image_color_features
from features.faz import compute_faz
from features.crossings import crossings_features


# ═════════════════════════════════════════════════════════════════════
# Configuration
# ═════════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """1 회 실행의 파라미터 집합."""
    # Data / paths
    csv_path: Path
    src_images_dir: Path                # 원본 이미지 폴더
    output_dir: Path                    # 산출물 (preprocessed / inference / aligned / features 하위 자동 생성)

    # Model / grid
    weights_dir: Path                   # ../weights/vascx
    square_size: int = 512
    device: str = "cuda:0"

    # Modes
    mode: str = "aligned"               # "aligned" | "naive"
    keep_masks: bool = False            # False → 환자 끝나면 mask 삭제

    # CSV 컬럼 이름 (기본은 longitudinal_final_260820_reclassified.csv 기준)
    col_patient: str = "ID"
    col_filename: str = "Filename"
    col_date: str = "LAB_DTM"
    col_shot: str = "shot"

    # Filtering
    patient_ids: Optional[List[str]] = None   # None → 전체
    min_visits_per_eye: int = 2               # 이 이하는 스킵 (registration 불가)

    # EyeLiner
    el_size: int = 256

    # 파생 경로 (post-init 에서 세팅)
    prep_root: Path = field(init=False)
    inference_root: Path = field(init=False)
    aligned_root: Path = field(init=False)
    features_root: Path = field(init=False)

    def __post_init__(self):
        self.csv_path = Path(self.csv_path)
        self.src_images_dir = Path(self.src_images_dir)
        self.output_dir = Path(self.output_dir)
        self.weights_dir = Path(self.weights_dir)

        self.prep_root = self.output_dir / f"preprocessed_{self.square_size}"
        self.inference_root = self.output_dir / f"inference_{self.square_size}"
        self.aligned_root = self.output_dir / f"aligned_{self.square_size}"
        self.features_root = self.output_dir / f"features_{self.square_size}_{self.mode}"

        for p in [self.prep_root, self.inference_root, self.features_root]:
            p.mkdir(parents=True, exist_ok=True)
        if self.mode == "aligned":
            self.aligned_root.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# 1. VascX 모델 로더
# ═════════════════════════════════════════════════════════════════════

def load_vascx_models(cfg: PipelineConfig):
    """7 개 ensemble 로드 + square_size override.

    Returns: dict {"quality", "av", "vessels", "disc", "fovea", "discedge", "odfd"}.
    """
    from rtnls_inference import (
        ClassificationEnsemble, SegmentationEnsemble,
        HeatmapRegressionEnsemble, RegressionEnsemble,
    )

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    w = cfg.weights_dir
    models = {
        "quality":  ClassificationEnsemble.from_release(str(w / "quality.pt")).to(device).eval(),
        "av":       SegmentationEnsemble.from_release(str(w / "av_july24.pt")).to(device).eval(),
        "vessels":  SegmentationEnsemble.from_release(str(w / "vessels_july24.pt")).to(device).eval(),
        "disc":     SegmentationEnsemble.from_release(str(w / "disc_july24.pt")).to(device).eval(),
        "fovea":    HeatmapRegressionEnsemble.from_release(str(w / "fovea_july24.pt")).to(device).eval(),
        "discedge": HeatmapRegressionEnsemble.from_release(str(w / "discedge_july24.pt")).to(device).eval(),
        "odfd":     RegressionEnsemble.from_release(str(w / "odfd_march25.pt")).to(device).eval(),
    }
    for name in ["av", "vessels", "disc", "fovea", "discedge", "odfd"]:
        tt = models[name].config["datamodule"].setdefault("test_transform", {})
        tt["square_size"] = cfg.square_size
        tt.setdefault("resize", cfg.square_size)
    return models, device


# ═════════════════════════════════════════════════════════════════════
# 2. Preprocess + Inference (환자별)
# ═════════════════════════════════════════════════════════════════════

def preprocess_patient(cfg: PipelineConfig, patient_id: str, rows: pd.DataFrame) -> List[str]:
    """원본 이미지 → RGB + CE (SQUARE_SIZE).

    Returns: 성공한 image id 리스트 (stem).
    """
    from rtnls_fundusprep.preprocessor import parallel_preprocess
    pid = str(patient_id)
    rgb_dir = cfg.prep_root / pid / "rgb"
    ce_dir  = cfg.prep_root / pid / "ce"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    ce_dir.mkdir(parents=True, exist_ok=True)

    files = [cfg.src_images_dir / fn for fn in rows[cfg.col_filename]]
    ids   = [f.stem for f in files]
    existing = {p.stem for p in rgb_dir.glob("*.png")}
    todo = [f for f in files if f.stem not in existing]
    if todo:
        parallel_preprocess(
            todo, rgb_path=rgb_dir, ce_path=ce_dir,
            square_size=cfg.square_size, n_jobs=4,
        )
    return [i for i in ids if (rgb_dir / f"{i}.png").exists()]


def infer_patient(cfg: PipelineConfig, models: dict, patient_id: str,
                   ids: List[str], device) -> pd.DataFrame:
    """이 환자 이미지들에 대해 7 model 추론 → per-image inventory row."""
    from rtnls_inference.utils import decollate_batch

    pid = str(patient_id)
    rgb_dir = cfg.prep_root / pid / "rgb"
    ce_dir  = cfg.prep_root / pid / "ce"
    inf_dir = cfg.inference_root / pid
    for tag in ["vessels", "av", "discs"]:
        (inf_dir / tag).mkdir(parents=True, exist_ok=True)

    rgb_paths    = [rgb_dir / f"{i}.png" for i in ids]
    ce_paths     = [ce_dir  / f"{i}.png" for i in ids]
    paired_paths = [(str(r), str(c)) for r, c in zip(rgb_paths, ce_paths)]

    # 4-1 keypoints (RGB+CE 페어)
    df_fovea    = models["fovea"].predict_preprocessed(paired_paths, ids=ids, num_workers=2, batch_size=8)
    df_fovea.columns = ["x_fovea", "y_fovea"]
    df_discedge = models["discedge"].predict_preprocessed(paired_paths, ids=ids, num_workers=2, batch_size=8)
    df_discedge.columns = ["x_discedge", "y_discedge"]

    # 4-2 scalar (RGB only)
    df_odfd = models["odfd"].predict_preprocessed([str(p) for p in rgb_paths], ids=ids, num_workers=2, batch_size=8)
    df_odfd.columns = ["v_odfd"]

    # 4-3 quality (RGB only)
    dl_q = models["quality"]._make_inference_dataloader(
        [str(p) for p in rgb_paths], ids=ids, num_workers=2, preprocess=False, batch_size=16,
    )
    qids, qrows = [], []
    with torch.no_grad():
        for batch in dl_q:
            if len(batch) == 0: continue
            im = batch["image"].to(device)
            q = models["quality"].predict_step(im).mean(dim=0)
            for it in decollate_batch({"id": batch["id"], "quality": q}):
                qids.append(it["id"]); qrows.append(it["quality"].tolist())
    df_quality = pd.DataFrame(qrows, index=qids, columns=["q1", "q2", "q3"])

    # 4-4 masks (paired)
    models["av"].predict_preprocessed(paired_paths, ids=ids, dest_path=inf_dir/"av",
                                       num_workers=2, batch_size=8)
    models["vessels"].predict_preprocessed(paired_paths, ids=ids, dest_path=inf_dir/"vessels",
                                            num_workers=2, batch_size=8)
    models["disc"].predict_preprocessed(paired_paths, ids=ids, dest_path=inf_dir/"discs",
                                         num_workers=2, batch_size=8)

    return dict(fovea=df_fovea, discedge=df_discedge, odfd=df_odfd, quality=df_quality)


# ═════════════════════════════════════════════════════════════════════
# 3. Inventory (eye side, disc geometry, meta)
# ═════════════════════════════════════════════════════════════════════

def disc_geometry(disc_mask: np.ndarray, min_area_px: int = 50) -> dict:
    m = disc_mask.astype(bool)
    if m.sum() < min_area_px:
        return dict(cx=np.nan, cy=np.nan, diameter=np.nan, area_px=float(m.sum()), found=False)
    lab, n = ndi.label(m)
    if n > 1:
        sizes = ndi.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    area = float(m.sum())
    cy, cx = ndi.center_of_mass(m)
    return dict(cx=float(cx), cy=float(cy), diameter=2.0*np.sqrt(area/np.pi),
                area_px=area, found=True)


def determine_laterality(x_fovea: float, x_disc: float, W: int, min_frac: float = 0.02) -> Tuple[str, float]:
    sep = x_disc - x_fovea
    if abs(sep) < min_frac * W:
        return "unknown", sep
    return ("L" if sep > 0 else "R"), sep


def build_inventory(cfg: PipelineConfig, patient_id: str, rows: pd.DataFrame,
                    inf_dfs: dict) -> pd.DataFrame:
    """이미지별 meta row DataFrame (id, patient, eye, fovea, disc, quality, ...)."""
    pid = str(patient_id)
    inf_dir = cfg.inference_root / pid
    W = cfg.square_size

    df_fovea = inf_dfs["fovea"]; df_discedge = inf_dfs["discedge"]
    df_odfd = inf_dfs["odfd"];   df_quality = inf_dfs["quality"]

    out_rows = []
    for _, r in rows.iterrows():
        stem = Path(r[cfg.col_filename]).stem
        if stem not in df_fovea.index or stem not in df_discedge.index:
            continue
        fx, fy = df_fovea.loc[stem, ["x_fovea", "y_fovea"]]

        disc_mask_p = inf_dir / "discs" / f"{stem}.png"
        if disc_mask_p.exists():
            dg = disc_geometry(np.array(Image.open(disc_mask_p)))
        else:
            dg = dict(cx=np.nan, cy=np.nan, diameter=np.nan, area_px=0, found=False)

        if dg["found"]:
            dx, dy, disc_source = dg["cx"], dg["cy"], "mask"
        else:
            dx = float(df_discedge.loc[stem, "x_discedge"])
            dy = float(df_discedge.loc[stem, "y_discedge"])
            disc_source = "keypoint_fallback"

        eye, sep = determine_laterality(float(fx), dx, W)
        q = df_quality.loc[stem].tolist() if stem in df_quality.index else [np.nan]*3
        out_rows.append({
            "id": stem, "patient": pid,
            "date": r[cfg.col_date], "shot": r.get(cfg.col_shot, 1),
            "filename": r[cfg.col_filename],
            "eye": eye, "sep_x": sep,
            "x_fovea": float(fx), "y_fovea": float(fy),
            "x_disc": dx, "y_disc": dy,
            "disc_diameter": dg["diameter"], "disc_area_px": dg["area_px"],
            "disc_source": disc_source,
            "v_odfd": df_odfd.loc[stem, "v_odfd"] if stem in df_odfd.index else np.nan,
            "q1": q[0], "q2": q[1], "q3": q[2],
        })
    inv = pd.DataFrame(out_rows)
    inv[cfg.col_date] if cfg.col_date in inv.columns else None
    inv["date"] = pd.to_datetime(inv["date"])
    return inv.sort_values(["eye", "date", "shot"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# 4. Registration (aligned) / identity (naive)
# ═════════════════════════════════════════════════════════════════════

def _setup_eyeliner(cfg: PipelineConfig, device):
    """Local EyeLiner + lightglue 임포트."""
    el_root = _PKG_ROOT / "EyeLiner"
    if str(el_root) not in sys.path:
        sys.path.insert(0, str(el_root))
    if "eyeliner.lightglue" not in sys.modules:
        sys.modules["eyeliner.lightglue"]       = importlib.import_module("lightglue")
        sys.modules["eyeliner.lightglue.utils"] = importlib.import_module("lightglue.utils")
    from eyeliner import EyeLinerP
    return EyeLinerP(kp_method="splg", reg="affine",
                     image_size=(3, cfg.el_size, cfg.el_size), device=device)


def _prep_tensor(img_np, size, device):
    a = cv2.resize(img_np.astype(np.float32) / 255.0, (size, size))
    return torch.tensor(a).permute(2, 0, 1).unsqueeze(0).to(device)


def _rescale_theta(theta_256, H, W, size):
    S = np.array([[W/size,0,0],[0,H/size,0],[0,0,1]], dtype=np.float64)
    return S @ theta_256 @ np.linalg.inv(S)


def warp_image(img, theta, H, W, is_mask=False):
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, theta[:2, :], (W, H),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register_group(cfg: PipelineConfig, eyeliner, device, visits: List[dict],
                    patient_id: str, eye: str) -> List[dict]:
    """(patient, eye) 그룹의 visit 들을 첫 방문 기준 정렬.

    aligned 모드: 실제 EyeLiner 로 theta 계산, mask warp 저장.
    naive 모드: 모두 identity theta, mask 는 inference 폴더 그대로 사용.
    """
    pid = str(patient_id)
    if cfg.mode == "aligned":
        out_dir = cfg.aligned_root / pid / eye
        for tag in ["av", "vessels", "discs"]:
            (out_dir / tag).mkdir(parents=True, exist_ok=True)

    inf_dir = cfg.inference_root / pid
    rgb_dir = cfg.prep_root / pid / "rgb"
    W = cfg.square_size

    records = []
    fixed = visits[0]
    fixed_rgb = np.array(Image.open(rgb_dir / f"{fixed['id']}.png"))

    # fixed 는 identity
    if cfg.mode == "aligned":
        for tag in ["av", "vessels", "discs"]:
            src = inf_dir / tag / f"{fixed['id']}.png"
            if src.exists():
                Image.open(src).save(out_dir / tag / f"{fixed['id']}.png",
                                      format="PNG", optimize=True)
    records.append({**fixed, "is_fixed": True, "n_kp": np.nan,
                    "theta": np.eye(3).flatten().tolist()})

    if cfg.mode == "naive":
        for v in visits[1:]:
            records.append({**v, "is_fixed": False, "n_kp": np.nan,
                            "theta": np.eye(3).flatten().tolist()})
        return records

    # aligned: 나머지 visit registration
    for v in visits[1:]:
        moving_rgb = np.array(Image.open(rgb_dir / f"{v['id']}.png"))
        try:
            tf = _prep_tensor(fixed_rgb, cfg.el_size, device)
            tm = _prep_tensor(moving_rgb, cfg.el_size, device)
            theta, cache = eyeliner({"fixed_input": tf, "moving_input": tm})
            theta = theta.squeeze().cpu().numpy()
            theta = _rescale_theta(theta, W, W, cfg.el_size)
            n_kp = cache["kp_fixed"].shape[1]
        except Exception as e:
            print(f"  [{pid}/{eye}] {v['id']} registration 실패: {e}")
            continue

        for tag in ["av", "vessels", "discs"]:
            src = inf_dir / tag / f"{v['id']}.png"
            if not src.exists(): continue
            m = np.array(Image.open(src))
            wm = warp_image(m, theta, W, W, is_mask=True)
            Image.fromarray(wm).save(out_dir / tag / f"{v['id']}.png",
                                       format="PNG", optimize=True)
        records.append({**v, "is_fixed": False, "n_kp": int(n_kp),
                        "theta": theta.flatten().tolist()})
    return records


# ═════════════════════════════════════════════════════════════════════
# 5. Zones + common_valid + feature extraction
# ═════════════════════════════════════════════════════════════════════

def _fundus_valid(rgb): return rgb.sum(-1) > 10


def build_zones_and_valid(cfg: PipelineConfig, patient_id: str, eye: str,
                           reg_records: List[dict], inv: pd.DataFrame) -> Tuple[dict, np.ndarray, np.ndarray, dict]:
    """Fixed visit 기준으로 zones + common_valid + whole_valid.

    Returns:
        (zones, common_valid, whole_valid, meta)
        - common_valid = 모든 visit fundus 유효 영역 교집합 → zone/spatial feature 용
        - whole_valid  = standard_fov (3×DD image-ctr) ∩ common_valid → whole-image feature 용
                         (환자 DD 비례해서 공정 비교)
    """
    pid = str(patient_id)
    W = cfg.square_size
    grp_inv = inv[(inv["eye"] == eye)].copy()
    if len(grp_inv) == 0:
        return None, None, None

    fixed = reg_records[0]
    fx = float(fixed["x_fovea"]); fy = float(fixed["y_fovea"])
    dcx = float(fixed["x_disc"]); dcy = float(fixed["y_disc"])
    dd_grp = float(grp_inv["disc_diameter"].mean())
    mpp = mm_per_px_from_disc(dd_grp)

    etdrs  = make_etdrs_9subfields(fx, fy, eye, mpp, (W, W), DEFAULT_ETDRS_RADII_MM)
    disc_z = make_disc_zones(dcx, dcy, dd_grp, (W, W), DEFAULT_DISC_MARGIN_ZONES)
    zones = {**etdrs, **disc_z}

    # common_valid — 모든 visit 의 fundus 유효 영역 교집합
    common = np.ones((W, W), dtype=bool)
    for i, v in enumerate(reg_records):
        rgb = np.array(Image.open(cfg.prep_root / pid / "rgb" / f"{v['id']}.png"))
        theta = np.array(v["theta"], dtype=np.float64).reshape(3, 3)
        if cfg.mode == "aligned" and not v.get("is_fixed", False):
            rgb = warp_image(rgb, theta, W, W)
        common &= _fundus_valid(rgb)

    # standard FOV — 환자 DD 비례 원 (image-centered, 3×DD)
    standard_fov = make_standard_fov(
        cx=W // 2, cy=W // 2,
        disc_diameter_px=dd_grp, image_shape=(W, W),
        radius_dd=STANDARD_FOV_RADIUS_DD,
    )
    whole_valid = standard_fov & common

    meta = dict(fovea_x=fx, fovea_y=fy, disc_cx=dcx, disc_cy=dcy,
                disc_diameter_px=dd_grp, mm_per_px=mpp, n_visits=len(reg_records),
                standard_fov_radius_dd=STANDARD_FOV_RADIUS_DD,
                whole_valid_area_px=int(whole_valid.sum()),
                common_valid_area_px=int(common.sum()))
    return zones, common, whole_valid, meta


def _mask_dir_for_visit(cfg: PipelineConfig, pid: str, eye: str, tag: str) -> Path:
    """Mode 에 따라 mask 파일 위치."""
    if cfg.mode == "aligned":
        return cfg.aligned_root / pid / eye / tag
    return cfg.inference_root / pid / tag


def extract_features_for_visit(cfg: PipelineConfig, pid: str, eye: str, visit: dict,
                                zones: dict, common_valid: np.ndarray, whole_valid: np.ndarray,
                                group_meta: dict) -> Optional[dict]:
    """단일 visit 의 241 features 계산.

    valid_mask 분리:
      - whole_valid  = standard_fov (3×DD image-ctr) ∩ common_valid → whole-image 전역 feature
      - common_valid = fundus 유효 영역 교집합 → zone / spatial (fovea/disc anchor) feature
    """
    W = cfg.square_size
    vid = visit["id"]
    try:
        rgb_base = np.array(Image.open(cfg.prep_root / pid / "rgb" / f"{vid}.png"))
        vs_m = np.array(Image.open(_mask_dir_for_visit(cfg, pid, eye, "vessels") / f"{vid}.png"))
        av_m = np.array(Image.open(_mask_dir_for_visit(cfg, pid, eye, "av")      / f"{vid}.png"))
        dc_m = np.array(Image.open(_mask_dir_for_visit(cfg, pid, eye, "discs")   / f"{vid}.png"))
    except FileNotFoundError:
        return None

    dd_v = visit.get("disc_diameter", np.nan)
    if np.isfinite(dd_v) and dd_v >= 5:
        mpp_v = mm_per_px_from_disc(float(dd_v))
        dd_v = float(dd_v)
    else:
        dd_v = float(group_meta["disc_diameter_px"])
        mpp_v = float(group_meta["mm_per_px"])

    # aligned: RGB 도 warp (color feature 정확도)
    if cfg.mode == "aligned" and not visit.get("is_fixed", False):
        theta = np.array(visit["theta"], dtype=np.float64).reshape(3, 3)
        rgb_a = warp_image(rgb_base, theta, W, W)
    else:
        rgb_a = rgb_base

    fx, fy = group_meta["fovea_x"], group_meta["fovea_y"]
    dcx, dcy = group_meta["disc_cx"], group_meta["disc_cy"]

    # whole-image (DD-scaled FOV 안에서 — 환자간 공정)
    w_dens   = whole_image_density_features(vs_m, av_m, dc_m, valid_mask=whole_valid)
    w_skel   = whole_image_skeleton_features(vs_m, av_m, valid_mask=whole_valid)
    w_seg    = whole_image_segment_features(vs_m, av_m, valid_mask=whole_valid)
    w_cal    = whole_image_caliber_features(vs_m, av_m, valid_mask=whole_valid, mm_per_px=mpp_v)
    w_frac   = whole_image_fractal_features(vs_m, av_m, valid_mask=whole_valid)
    w_color  = whole_image_color_features(rgb_a, vs_m, av_m, valid_mask=whole_valid)
    w_tort   = whole_image_tortuosity_features(vs_m, av_m, valid_mask=whole_valid)
    w_bif    = whole_image_bifurcation_features(vs_m, av_m, valid_mask=whole_valid)
    w_wv     = whole_image_width_variability(vs_m, av_m, valid_mask=whole_valid)
    w_junc   = whole_image_junction_features(vs_m, av_m, valid_mask=whole_valid)

    # zone / spatial feature (해부학 anchor 가 명확 — common_valid 로 fundus 밖만 제외)
    z_dens   = zone_density_features(vs_m, av_m, dc_m, zones, valid_mask=common_valid)
    knudtson = compute_crae_crve_avr(av_m, zones["zone_B"], mm_per_px=mpp_v)
    w_sh_f   = sholl_features(vs_m, av_m, (fx, fy),   valid_mask=common_valid, center_name="fovea")
    w_sh_d   = sholl_features(vs_m, av_m, (dcx, dcy), valid_mask=common_valid, center_name="disc")
    faz      = compute_faz(vs_m, (fx, fy), valid_mask=common_valid, mm_per_px=mpp_v, disc_diameter_px=dd_v)
    cross    = crossings_features(av_m, zones=zones, valid_mask=common_valid)

    return {
        "id": vid, "patient": pid, "eye": eye,
        "date": pd.Timestamp(visit["date"]).isoformat(),
        "shot": int(visit.get("shot", 1)),
        "disc_diameter_px": dd_v, "mm_per_px": mpp_v,
        "q1": visit.get("q1"), "q2": visit.get("q2"), "q3": visit.get("q3"),
        "x_fovea": float(fx), "y_fovea": float(fy),
        "x_disc": float(dcx), "y_disc": float(dcy),
        **w_dens, **z_dens, **w_skel, **w_seg, **w_cal, **knudtson,
        **w_frac, **w_sh_f, **w_sh_d, **w_color,
        **w_tort, **faz, **cross, **w_bif, **w_wv, **w_junc,
    }


# ═════════════════════════════════════════════════════════════════════
# 6. Patient JSON I/O
# ═════════════════════════════════════════════════════════════════════

def _json_default(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray): return o.tolist()
    if isinstance(o, pd.Timestamp): return o.isoformat()
    if isinstance(o, (Path,)): return str(o)
    return str(o)


def save_patient_json(cfg: PipelineConfig, patient_id: str,
                       eye_meta: dict, feature_rows: List[dict]) -> Path:
    """Patient JSON: {patient, config, meta_by_eye, features (list of dicts)}."""
    pid = str(patient_id)
    payload = {
        "patient": pid,
        "config": {
            "mode": cfg.mode, "square_size": cfg.square_size,
            "extracted_at": datetime.utcnow().isoformat() + "Z",
        },
        "meta_by_eye": eye_meta,
        "features": feature_rows,
    }
    out_path = cfg.features_root / f"{pid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return out_path


def load_patient_features(json_path: Path) -> pd.DataFrame:
    """Patient JSON → feature DataFrame (feature row 리스트만)."""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["features"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def merge_patient_jsons(features_root: Path) -> pd.DataFrame:
    """모든 patient JSON → single DataFrame."""
    dfs = []
    for p in sorted(features_root.glob("*.json")):
        try:
            dfs.append(load_patient_features(p))
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).sort_values(["patient", "eye", "date"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# 7. Cleanup (mask 삭제)
# ═════════════════════════════════════════════════════════════════════

def cleanup_patient_masks(cfg: PipelineConfig, patient_id: str):
    """환자 처리 완료 후 mask/preprocessed 삭제 (JSON 만 남김)."""
    pid = str(patient_id)
    for base in [cfg.prep_root, cfg.inference_root, cfg.aligned_root]:
        d = base / pid
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════
# 8. Top-level per-patient runner
# ═════════════════════════════════════════════════════════════════════

def _inventory_cache_path(cfg: PipelineConfig, pid: str) -> Path:
    return cfg.inference_root / pid / "inventory.csv"


def _all_masks_exist(cfg: PipelineConfig, pid: str, rows: pd.DataFrame) -> bool:
    """rows 의 모든 stem 에 대해 vessels/av/discs mask 파일이 있는지."""
    inf_dir = cfg.inference_root / pid
    for _, r in rows.iterrows():
        stem = Path(r[cfg.col_filename]).stem
        for tag in ["vessels", "av", "discs"]:
            if not (inf_dir / tag / f"{stem}.png").exists():
                return False
    return True


def run_patient(cfg: PipelineConfig, models: dict, device, eyeliner,
                 patient_id: str, rows: pd.DataFrame,
                 verbose: bool = True, force: bool = False) -> Optional[Path]:
    """한 환자에 대해 전체 파이프라인 실행 → JSON 저장 경로 반환.

    자동 스킵/캐시:
      - `<features_root>/<pid>.json` 존재 → 즉시 반환 (force=True 로 override)
      - `<inference_root>/<pid>/inventory.csv` + 모든 mask 파일 존재
        → preprocess + inference 스킵 (registration + feature 만 재실행)

    실패 시 None, 이미 완료된 JSON 존재 시 그 경로 반환.
    """
    pid = str(patient_id)
    out_json = cfg.features_root / f"{pid}.json"

    if out_json.exists() and not force:
        if verbose: print(f"[{pid}] JSON 존재 → 스킵 ({out_json.name})")
        return out_json

    inv_cache = _inventory_cache_path(cfg, pid)

    # 1. preprocess + inference (캐시 있으면 스킵)
    if inv_cache.exists() and _all_masks_exist(cfg, pid, rows):
        inv = pd.read_csv(inv_cache, parse_dates=["date"])
        if verbose: print(f"[{pid}] inventory + masks 캐시 재사용")
    else:
        ids = preprocess_patient(cfg, pid, rows)
        if len(ids) == 0:
            if verbose: print(f"[{pid}] no images"); return None
        inf_dfs = infer_patient(cfg, models, pid, ids, device)
        inv = build_inventory(cfg, pid, rows, inf_dfs)
        if len(inv) == 0:
            if verbose: print(f"[{pid}] inventory empty"); return None
        inv_cache.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(inv_cache, index=False)

    # 3. per-eye pipeline
    eye_meta = {}
    feature_rows = []
    for eye in ["L", "R"]:
        visits = inv[inv["eye"] == eye].to_dict("records")
        if len(visits) < cfg.min_visits_per_eye:
            if verbose: print(f"[{pid}/{eye}] {len(visits)} visits (skip)")
            continue

        reg_records = register_group(cfg, eyeliner, device, visits, pid, eye)
        zones, common_valid, whole_valid, meta = build_zones_and_valid(cfg, pid, eye, reg_records, inv)
        if zones is None:
            continue
        eye_meta[eye] = meta

        for v in reg_records:
            feat = extract_features_for_visit(cfg, pid, eye, v, zones,
                                                common_valid, whole_valid, meta)
            if feat is not None:
                feature_rows.append(feat)

    if not feature_rows:
        if verbose: print(f"[{pid}] no features extracted")
        return None

    # 4. save JSON
    out_path = save_patient_json(cfg, pid, eye_meta, feature_rows)
    if verbose:
        print(f"[{pid}] saved {out_path.name} ({len(feature_rows)} visits, {len(eye_meta)} eyes)")

    # 5. cleanup
    if not cfg.keep_masks:
        cleanup_patient_masks(cfg, pid)

    return out_path
