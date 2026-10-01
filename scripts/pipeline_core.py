"""Shared pipeline core — per-patient aligned feature extraction.

Steps:
    1. Preprocess raw image → RGB + CE at SQUARE_SIZE.
    2. VascX inference (7 models: quality, av, vessels, disc, fovea, discedge, odfd).
    3. Build inventory (eye side, fovea / disc geometry, quality).
    4. EyeLiner registration — first visit = fixed, rest warped to its frame.
    5. Feature extraction (~241 features per visit).
    6. Save per-patient JSON.

Disk management:
    keep_masks=True  → keep every mask (preprocessed / inference / aligned) for re-analysis.
    keep_masks=False → delete masks after the patient's JSON is written (JSON is self-contained).
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage as ndi

# features/ is a sibling package in this repo
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from features.zones import (
    make_etdrs_9subfields, make_disc_zones, mm_per_px_from_disc,
    DEFAULT_ETDRS_RADII_MM, DEFAULT_DISC_MARGIN_ZONES,
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
    """Parameters for one pipeline invocation."""
    csv_path: Path
    src_images_dir: Path
    output_dir: Path

    weights_dir: Path
    square_size: int = 512
    device: str = "cuda:0"

    keep_masks: bool = False

    col_patient: str = "ID"
    col_filename: str = "Filename"
    col_date: str = "LAB_DTM"
    col_shot: str = "shot"

    patient_ids: Optional[List[str]] = None
    min_visits_per_eye: int = 2

    el_size: int = 256

    # Derived paths (set in __post_init__)
    prep_root: Path = field(init=False)
    inference_root: Path = field(init=False)
    aligned_root: Path = field(init=False)
    features_root: Path = field(init=False)

    def __post_init__(self):
        self.csv_path = Path(self.csv_path)
        self.src_images_dir = Path(self.src_images_dir)
        self.output_dir = Path(self.output_dir)
        self.weights_dir = Path(self.weights_dir)

        self.prep_root      = self.output_dir / f"preprocessed_{self.square_size}"
        self.inference_root = self.output_dir / f"inference_{self.square_size}"
        self.aligned_root   = self.output_dir / f"aligned_{self.square_size}"
        self.features_root  = self.output_dir / f"features_{self.square_size}_aligned"

        for p in [self.prep_root, self.inference_root, self.aligned_root, self.features_root]:
            p.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════
# 1. VascX model loader
# ═════════════════════════════════════════════════════════════════════

def load_vascx_models(cfg: PipelineConfig):
    """Load 7 ensembles and override their `square_size` to cfg.square_size."""
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
# 2. Preprocess + inference (per patient)
# ═════════════════════════════════════════════════════════════════════

def preprocess_patient(cfg: PipelineConfig, patient_id: str, rows: pd.DataFrame) -> List[str]:
    """Raw image → RGB + CE at SQUARE_SIZE. Returns the list of successful image stems."""
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
                  ids: List[str], device) -> dict:
    """Run all 7 models on the patient's images."""
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

    df_fovea    = models["fovea"].predict_preprocessed(paired_paths, ids=ids, num_workers=2, batch_size=8)
    df_fovea.columns = ["x_fovea", "y_fovea"]
    df_discedge = models["discedge"].predict_preprocessed(paired_paths, ids=ids, num_workers=2, batch_size=8)
    df_discedge.columns = ["x_discedge", "y_discedge"]

    df_odfd = models["odfd"].predict_preprocessed([str(p) for p in rgb_paths], ids=ids, num_workers=2, batch_size=8)
    df_odfd.columns = ["v_odfd"]

    dl_q = models["quality"]._make_inference_dataloader(
        [str(p) for p in rgb_paths], ids=ids, num_workers=2, preprocess=False, batch_size=16,
    )
    qids, qrows = [], []
    with torch.no_grad():
        for batch in dl_q:
            if len(batch) == 0:
                continue
            im = batch["image"].to(device)
            q = models["quality"].predict_step(im).mean(dim=0)
            for it in decollate_batch({"id": batch["id"], "quality": q}):
                qids.append(it["id"])
                qrows.append(it["quality"].tolist())
    df_quality = pd.DataFrame(qrows, index=qids, columns=["q1", "q2", "q3"])

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
    return dict(cx=float(cx), cy=float(cy), diameter=2.0 * np.sqrt(area / np.pi),
                area_px=area, found=True)


def determine_laterality(x_fovea: float, x_disc: float, W: int, min_frac: float = 0.02) -> Tuple[str, float]:
    sep = x_disc - x_fovea
    if abs(sep) < min_frac * W:
        return "unknown", sep
    return ("L" if sep > 0 else "R"), sep


def build_inventory(cfg: PipelineConfig, patient_id: str, rows: pd.DataFrame,
                    inf_dfs: dict) -> pd.DataFrame:
    """One row per image: id, patient, eye, fovea/disc coords, quality, …"""
    pid = str(patient_id)
    inf_dir = cfg.inference_root / pid
    W = cfg.square_size

    df_fovea    = inf_dfs["fovea"]
    df_discedge = inf_dfs["discedge"]
    df_odfd     = inf_dfs["odfd"]
    df_quality  = inf_dfs["quality"]

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
        q = df_quality.loc[stem].tolist() if stem in df_quality.index else [np.nan] * 3
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
    inv["date"] = pd.to_datetime(inv["date"])
    return inv.sort_values(["eye", "date", "shot"]).reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════
# 4. Registration
# ═════════════════════════════════════════════════════════════════════

def _setup_eyeliner(cfg: PipelineConfig, device):
    """Import the vendored EyeLiner + LightGlue and build an `EyeLinerP`."""
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
    S = np.array([[W/size, 0, 0], [0, H/size, 0], [0, 0, 1]], dtype=np.float64)
    return S @ theta_256 @ np.linalg.inv(S)


def warp_image(img, theta, H, W, is_mask=False):
    flags = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    return cv2.warpAffine(img, theta[:2, :], (W, H),
                          flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def register_group(cfg: PipelineConfig, eyeliner, device, visits: List[dict],
                   patient_id: str, eye: str) -> List[dict]:
    """Align every follow-up visit in a (patient, eye) group to the first visit."""
    pid = str(patient_id)
    out_dir = cfg.aligned_root / pid / eye
    for tag in ["av", "vessels", "discs"]:
        (out_dir / tag).mkdir(parents=True, exist_ok=True)

    inf_dir = cfg.inference_root / pid
    rgb_dir = cfg.prep_root / pid / "rgb"
    W = cfg.square_size

    records = []
    fixed = visits[0]
    fixed_rgb = np.array(Image.open(rgb_dir / f"{fixed['id']}.png"))

    for tag in ["av", "vessels", "discs"]:
        src = inf_dir / tag / f"{fixed['id']}.png"
        if src.exists():
            Image.open(src).save(out_dir / tag / f"{fixed['id']}.png",
                                 format="PNG", optimize=True)
    records.append({**fixed, "is_fixed": True, "n_kp": np.nan,
                    "theta": np.eye(3).flatten().tolist()})

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
            print(f"  [{pid}/{eye}] {v['id']} registration failed: {e}")
            continue

        for tag in ["av", "vessels", "discs"]:
            src = inf_dir / tag / f"{v['id']}.png"
            if not src.exists():
                continue
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

def _fundus_valid(rgb):
    return rgb.sum(-1) > 10


def build_zones_and_valid(cfg: PipelineConfig, patient_id: str, eye: str,
                          reg_records: List[dict], inv: pd.DataFrame) -> Tuple[Optional[dict], Optional[np.ndarray], Optional[dict]]:
    """Build fixed-frame zones + common_valid (intersection across visits)."""
    pid = str(patient_id)
    W = cfg.square_size
    grp_inv = inv[inv["eye"] == eye].copy()
    if len(grp_inv) == 0:
        return None, None, None

    fixed = reg_records[0]
    fx, fy   = float(fixed["x_fovea"]), float(fixed["y_fovea"])
    dcx, dcy = float(fixed["x_disc"]),  float(fixed["y_disc"])
    dd_grp   = float(grp_inv["disc_diameter"].mean())
    mpp      = mm_per_px_from_disc(dd_grp)

    etdrs  = make_etdrs_9subfields(fx, fy, eye, mpp, (W, W), DEFAULT_ETDRS_RADII_MM)
    disc_z = make_disc_zones(dcx, dcy, dd_grp, (W, W), DEFAULT_DISC_MARGIN_ZONES)
    zones  = {**etdrs, **disc_z}

    common = np.ones((W, W), dtype=bool)
    for v in reg_records:
        rgb = np.array(Image.open(cfg.prep_root / pid / "rgb" / f"{v['id']}.png"))
        if not v.get("is_fixed", False):
            theta = np.array(v["theta"], dtype=np.float64).reshape(3, 3)
            rgb = warp_image(rgb, theta, W, W)
        common &= _fundus_valid(rgb)

    meta = dict(fovea_x=fx, fovea_y=fy, disc_cx=dcx, disc_cy=dcy,
                disc_diameter_px=dd_grp, mm_per_px=mpp, n_visits=len(reg_records))
    return zones, common, meta


def extract_features_for_visit(cfg: PipelineConfig, pid: str, eye: str, visit: dict,
                               zones: dict, valid: np.ndarray, group_meta: dict) -> Optional[dict]:
    """Compute all ~241 features for a single visit."""
    W = cfg.square_size
    vid = visit["id"]
    aligned_mask_dir = cfg.aligned_root / pid / eye
    try:
        rgb_base = np.array(Image.open(cfg.prep_root / pid / "rgb" / f"{vid}.png"))
        vs_m = np.array(Image.open(aligned_mask_dir / "vessels" / f"{vid}.png"))
        av_m = np.array(Image.open(aligned_mask_dir / "av"      / f"{vid}.png"))
        dc_m = np.array(Image.open(aligned_mask_dir / "discs"   / f"{vid}.png"))
    except FileNotFoundError:
        return None

    dd_v = visit.get("disc_diameter", np.nan)
    if np.isfinite(dd_v) and dd_v >= 5:
        mpp_v = mm_per_px_from_disc(float(dd_v))
        dd_v = float(dd_v)
    else:
        dd_v = float(group_meta["disc_diameter_px"])
        mpp_v = float(group_meta["mm_per_px"])

    if not visit.get("is_fixed", False):
        theta = np.array(visit["theta"], dtype=np.float64).reshape(3, 3)
        rgb_a = warp_image(rgb_base, theta, W, W)
    else:
        rgb_a = rgb_base

    fx, fy   = group_meta["fovea_x"], group_meta["fovea_y"]
    dcx, dcy = group_meta["disc_cx"], group_meta["disc_cy"]

    w_dens   = whole_image_density_features(vs_m, av_m, dc_m, valid_mask=valid)
    z_dens   = zone_density_features(vs_m, av_m, dc_m, zones, valid_mask=valid)
    w_skel   = whole_image_skeleton_features(vs_m, av_m, valid_mask=valid)
    w_seg    = whole_image_segment_features(vs_m, av_m, valid_mask=valid)
    w_cal    = whole_image_caliber_features(vs_m, av_m, valid_mask=valid, mm_per_px=mpp_v)
    knudtson = compute_crae_crve_avr(av_m, zones["zone_B"], mm_per_px=mpp_v)
    w_frac   = whole_image_fractal_features(vs_m, av_m, valid_mask=valid)
    w_sh_f   = sholl_features(vs_m, av_m, (fx, fy),   valid_mask=valid, center_name="fovea")
    w_sh_d   = sholl_features(vs_m, av_m, (dcx, dcy), valid_mask=valid, center_name="disc")
    w_color  = whole_image_color_features(rgb_a, vs_m, av_m, valid_mask=valid)
    w_tort   = whole_image_tortuosity_features(vs_m, av_m, valid_mask=valid)
    faz      = compute_faz(vs_m, (fx, fy), valid_mask=valid, mm_per_px=mpp_v, disc_diameter_px=dd_v)
    cross    = crossings_features(av_m, zones=zones, valid_mask=valid)
    w_bif    = whole_image_bifurcation_features(vs_m, av_m, valid_mask=valid)
    w_wv     = whole_image_width_variability(vs_m, av_m, valid_mask=valid)
    w_junc   = whole_image_junction_features(vs_m, av_m, valid_mask=valid)

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
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        v = float(o)
        return None if not np.isfinite(v) else v
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pd.Timestamp):
        return o.isoformat()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def save_patient_json(cfg: PipelineConfig, patient_id: str,
                      eye_meta: dict, feature_rows: List[dict]) -> Path:
    """Write `<features_root>/<patient>.json`."""
    pid = str(patient_id)
    payload = {
        "patient": pid,
        "config": {
            "mode": "aligned", "square_size": cfg.square_size,
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        },
        "meta_by_eye": eye_meta,
        "features": feature_rows,
    }
    out_path = cfg.features_root / f"{pid}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)
    return out_path


def load_patient_features(json_path: Path) -> pd.DataFrame:
    """Patient JSON → feature DataFrame (features list only)."""
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["features"])
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def merge_patient_jsons(features_root: Path) -> pd.DataFrame:
    """Concatenate every per-patient JSON into one DataFrame."""
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
# 7. Cleanup
# ═════════════════════════════════════════════════════════════════════

def cleanup_patient_masks(cfg: PipelineConfig, patient_id: str):
    """Delete preprocessed / inference / aligned trees for one patient (JSON stays)."""
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
    """Run the full pipeline for one patient. Returns the JSON path, or None on failure.

    Automatic skip / cache:
      - If `<features_root>/<pid>.json` exists → return immediately (use force=True to override).
      - If `<inference_root>/<pid>/inventory.csv` + every mask file exists
        → skip preprocess + inference; only re-run registration + feature extraction.
    """
    pid = str(patient_id)
    out_json = cfg.features_root / f"{pid}.json"

    if out_json.exists() and not force:
        if verbose:
            print(f"[{pid}] JSON already exists → skip ({out_json.name})")
        return out_json

    inv_cache = _inventory_cache_path(cfg, pid)

    if inv_cache.exists() and _all_masks_exist(cfg, pid, rows):
        inv = pd.read_csv(inv_cache, parse_dates=["date"])
        if verbose:
            print(f"[{pid}] reusing cached inventory + masks")
    else:
        ids = preprocess_patient(cfg, pid, rows)
        if len(ids) == 0:
            if verbose:
                print(f"[{pid}] no images")
            return None
        inf_dfs = infer_patient(cfg, models, pid, ids, device)
        inv = build_inventory(cfg, pid, rows, inf_dfs)
        if len(inv) == 0:
            if verbose:
                print(f"[{pid}] inventory empty")
            return None
        inv_cache.parent.mkdir(parents=True, exist_ok=True)
        inv.to_csv(inv_cache, index=False)

    eye_meta = {}
    feature_rows = []
    for eye in ["L", "R"]:
        visits = inv[inv["eye"] == eye].to_dict("records")
        if len(visits) < cfg.min_visits_per_eye:
            if verbose:
                print(f"[{pid}/{eye}] {len(visits)} visits (skip)")
            continue

        reg_records = register_group(cfg, eyeliner, device, visits, pid, eye)
        zones, valid, meta = build_zones_and_valid(cfg, pid, eye, reg_records, inv)
        if zones is None:
            continue
        eye_meta[eye] = meta

        for v in reg_records:
            feat = extract_features_for_visit(cfg, pid, eye, v, zones, valid, meta)
            if feat is not None:
                feature_rows.append(feat)

    if not feature_rows:
        if verbose:
            print(f"[{pid}] no features extracted")
        return None

    out_path = save_patient_json(cfg, pid, eye_meta, feature_rows)
    if verbose:
        print(f"[{pid}] saved {out_path.name} ({len(feature_rows)} visits, {len(eye_meta)} eyes)")

    if not cfg.keep_masks:
        cleanup_patient_masks(cfg, pid)

    return out_path
