"""Generate pipeline-step figures from the bundled sample dataset.

Writes PNGs into `docs/` at the repo root. Requires the sample pipeline to have
already been run (`notebooks/01-alignment.ipynb` or `scripts/run_pipeline.py`),
so that `samples/output/` is populated.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from features.zones import (
    make_etdrs_9subfields, make_disc_zones, mm_per_px_from_disc,
    DEFAULT_ETDRS_RADII_MM, DEFAULT_DISC_MARGIN_ZONES,
)
from features.skeleton import clean_and_skeletonize
from features.topology import sholl_curve


SAMPLE_ROOT = _PKG_ROOT / "samples" / "output"
DOCS        = _PKG_ROOT / "docs"
PATIENT     = "SAMPLE"
EYE         = "R"
SQUARE_SIZE = 512
FIG_H       = 3.6   # uniform panel height across all figures (inches)


def _resolve_rgb_dir() -> Path:
    """Script layout: <out>/preprocessed_512/<pid>/rgb. Notebook layout: <out>/preprocessed_512/rgb."""
    cand = SAMPLE_ROOT / f"preprocessed_{SQUARE_SIZE}" / PATIENT / "rgb"
    if cand.exists():
        return cand
    return SAMPLE_ROOT / f"preprocessed_{SQUARE_SIZE}" / "rgb"


def _resolve_inference_dir() -> Path:
    """Script layout: inference_512/<pid>. Notebook layout: vascx_output_512."""
    cand = SAMPLE_ROOT / f"inference_{SQUARE_SIZE}" / PATIENT
    if cand.exists():
        return cand
    return SAMPLE_ROOT / f"vascx_output_{SQUARE_SIZE}"


def _resolve_features_json() -> Path:
    for name in [f"features_{SQUARE_SIZE}_aligned", f"features_json_aligned_{SQUARE_SIZE}"]:
        p = SAMPLE_ROOT / name / f"{PATIENT}.json"
        if p.exists():
            return p
    raise FileNotFoundError("No features JSON found — run the pipeline first.")


def _load_payload():
    with open(_resolve_features_json(), encoding="utf-8") as f:
        return json.load(f)


def _av_rgb(av_mask):
    """AV mask (0 bg / 1 A / 2 V / 3 crossing) → coloured RGB."""
    rgb = np.zeros((*av_mask.shape, 3), dtype=np.uint8)
    rgb[av_mask == 1] = [230, 40, 40]
    rgb[av_mask == 2] = [40, 80, 230]
    rgb[av_mask == 3] = [200, 40, 200]
    return rgb


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — VascX masks for a single visit
# ─────────────────────────────────────────────────────────────────────

def fig01_vascx_masks(rgb_dir: Path, inf_dir: Path):
    vid = "pid_1"
    rgb = np.array(Image.open(rgb_dir / f"{vid}.png"))
    av  = np.array(Image.open(inf_dir / "av"      / f"{vid}.png"))
    ves = np.array(Image.open(inf_dir / "vessels" / f"{vid}.png"))
    dsc = np.array(Image.open(inf_dir / "discs"   / f"{vid}.png"))

    fig, axes = plt.subplots(1, 4, figsize=(FIG_H * 4, FIG_H))
    axes[0].imshow(rgb);              axes[0].set_title("RGB")
    axes[1].imshow(ves, cmap="gray"); axes[1].set_title("Vessels")
    axes[2].imshow(_av_rgb(av));      axes[2].set_title("AV")
    axes[3].imshow(dsc, cmap="gray"); axes[3].set_title("Disc")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "01_vascx_masks.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 2 — Laterality (fovea vs disc)
# ─────────────────────────────────────────────────────────────────────

def fig02_laterality(payload, rgb_dir: Path):
    meta = payload["meta_by_eye"][EYE]
    fx, fy = meta["fovea_x"], meta["fovea_y"]
    dcx, dcy = meta["disc_cx"], meta["disc_cy"]
    sep = dcx - fx
    eye_label = "L (OS)" if sep > 0 else "R (OD)"

    first_vid = payload["features"][0]["id"]
    rgb = np.array(Image.open(rgb_dir / f"{first_vid}.png"))

    fig, ax = plt.subplots(figsize=(FIG_H, FIG_H))
    ax.imshow(rgb)
    ax.plot([fx, dcx], [fy, dcy], color="yellow", linewidth=1.5, alpha=0.8)
    ax.scatter(fx, fy, s=120, marker="x", color="lime", linewidths=2.2, label="fovea")
    ax.scatter(dcx, dcy, s=120, marker="o", facecolors="none",
               edgecolors="cyan", linewidths=2.2, label="disc")
    ax.set_title(f"disc.x − fovea.x = {sep:+.0f}px → {eye_label}", fontsize=10)
    ax.axis("off")
    ax.legend(loc="lower right", facecolor="black", labelcolor="white",
              framealpha=0.6, fontsize=8)
    fig.tight_layout()
    fig.savefig(DOCS / "02_laterality.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 3 — Registration (before / after)
# ─────────────────────────────────────────────────────────────────────

def _overlay_rb(gray_fixed, gray_moving):
    """Red = fixed, Cyan = moving. Yellow where they overlap."""
    H, W = gray_fixed.shape
    out = np.zeros((H, W, 3), dtype=np.uint8)
    out[..., 0] = gray_fixed
    out[..., 1] = gray_moving
    out[..., 2] = gray_moving
    return out


def fig03_registration(payload, rgb_dir: Path, aligned_dir: Path):
    import cv2
    feats = payload["features"]
    fixed_vid  = feats[0]["id"]
    moving_vid = feats[-1]["id"]

    fixed_rgb  = np.array(Image.open(rgb_dir / f"{fixed_vid}.png"))
    moving_rgb = np.array(Image.open(rgb_dir / f"{moving_vid}.png"))

    fixed_gray  = cv2.cvtColor(fixed_rgb,  cv2.COLOR_RGB2GRAY)
    moving_gray = cv2.cvtColor(moving_rgb, cv2.COLOR_RGB2GRAY)

    # Aligned vessel masks give us a clean "after" comparison. For the "before"
    # we overlay the raw moving RGB on the fixed RGB (same canvas, no warp).
    fixed_vess = np.array(Image.open(aligned_dir / "vessels" / f"{fixed_vid}.png")) > 0
    moving_vess_aligned = np.array(Image.open(aligned_dir / "vessels" / f"{moving_vid}.png")) > 0

    before = _overlay_rb(fixed_gray, moving_gray)

    H, W = fixed_vess.shape
    after = np.zeros((H, W, 3), dtype=np.uint8)
    after[..., 0] = 220 * fixed_vess.astype(np.uint8)
    after[..., 1] = 220 * moving_vess_aligned.astype(np.uint8)

    dice = (2 * (fixed_vess & moving_vess_aligned).sum() /
            max(int(fixed_vess.sum()) + int(moving_vess_aligned.sum()), 1))

    fig, axes = plt.subplots(1, 2, figsize=(FIG_H * 2, FIG_H))
    axes[0].imshow(before)
    axes[0].set_title("Before")
    axes[1].imshow(after)
    axes[1].set_title(f"After  ·  Dice = {dice:.3f}")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "03_registration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 4 — Aligned vessel timeline
# ─────────────────────────────────────────────────────────────────────

def fig04_aligned_timeline(payload, aligned_dir: Path):
    feats = payload["features"]
    n = len(feats)
    fig, axes = plt.subplots(1, n, figsize=(FIG_H * n, FIG_H))
    if n == 1:
        axes = [axes]
    for i, (ax, v) in enumerate(zip(axes, feats)):
        av = np.array(Image.open(aligned_dir / "av" / f"{v['id']}.png"))
        ax.imshow(_av_rgb(av))
        tag = "fixed" if i == 0 else f"visit {i + 1}"
        ax.set_title(tag)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DOCS / "04_aligned_timeline.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 5 — ETDRS + Wong zones overlaid on the fixed frame
# ─────────────────────────────────────────────────────────────────────

ZONE_COLORS = {
    "C":  [255, 255, 100], "S1": [130, 200, 255], "N1": [180, 130, 255],
    "I1": [130, 255, 180], "T1": [255, 180, 130], "S2": [80,  130, 200],
    "N2": [130, 80,  200], "I2": [80,  200, 130], "T2": [200, 130, 80],
    "zone_B": [255, 100, 100],
    "zone_C": [180, 60,  60],
}


def fig05_zones(payload, rgb_dir: Path):
    meta = payload["meta_by_eye"][EYE]
    fx, fy   = meta["fovea_x"], meta["fovea_y"]
    dcx, dcy = meta["disc_cx"], meta["disc_cy"]
    dd       = meta["disc_diameter_px"]
    mpp      = meta["mm_per_px"]

    etdrs  = make_etdrs_9subfields(fx, fy, EYE, mpp, (SQUARE_SIZE, SQUARE_SIZE), DEFAULT_ETDRS_RADII_MM)
    disc_z = make_disc_zones(dcx, dcy, dd, (SQUARE_SIZE, SQUARE_SIZE), DEFAULT_DISC_MARGIN_ZONES)
    zones  = {**etdrs, **disc_z}

    fixed_vid = payload["features"][0]["id"]
    rgb = np.array(Image.open(rgb_dir / f"{fixed_vid}.png"))

    disp = rgb.copy().astype(np.float32)
    for name, mask in zones.items():
        c = np.array(ZONE_COLORS.get(name, [200, 200, 200]), dtype=np.float32)
        disp[mask] = disp[mask] * 0.6 + c * 0.4
    disp = disp.clip(0, 255).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(FIG_H, FIG_H))
    ax.imshow(disp)
    ax.scatter(fx, fy, s=100, marker="x", color="white", linewidths=1.8)
    ax.scatter(dcx, dcy, s=100, marker="o", facecolors="none",
               edgecolors="white", linewidths=1.8)
    ax.axis("off")

    handles = [Patch(color=np.array(c) / 255, label=n) for n, c in ZONE_COLORS.items()]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(DOCS / "05_zones.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 6 — A feature example (Sholl curve, fovea-centred)
# ─────────────────────────────────────────────────────────────────────

def fig06_features(payload, aligned_dir: Path):
    feats = payload["features"]
    meta = payload["meta_by_eye"][EYE]
    fx, fy = meta["fovea_x"], meta["fovea_y"]
    radii = list(range(10, 240, 5))

    fig, (ax_mask, ax_curve) = plt.subplots(1, 2, figsize=(FIG_H * 2, FIG_H))

    fixed_vid = feats[0]["id"]
    ves = np.array(Image.open(aligned_dir / "vessels" / f"{fixed_vid}.png")) > 0
    ax_mask.imshow(ves, cmap="gray")
    theta = np.linspace(0, 2 * np.pi, 200)
    for r in [40, 80, 120, 160, 200]:
        ax_mask.plot(fx + r * np.cos(theta), fy + r * np.sin(theta),
                     color="cyan", linewidth=0.9, alpha=0.7)
    ax_mask.scatter(fx, fy, s=100, marker="x", color="yellow", linewidths=1.8)
    ax_mask.set_title("Sholl rings (fovea-centred)")
    ax_mask.axis("off")

    for i, v in enumerate(feats):
        ves = np.array(Image.open(aligned_dir / "vessels" / f"{v['id']}.png"))
        skel = clean_and_skeletonize(ves > 0)
        curve = sholl_curve(skel, fx, fy, radii)
        ax_curve.plot(radii, curve, "-o", markersize=3, linewidth=1,
                      label=f"visit {i + 1}")
    ax_curve.set_xlabel("radius (px)")
    ax_curve.set_ylabel("# skeleton crossings")
    ax_curve.set_title("Sholl curve per visit")
    ax_curve.grid(alpha=0.3)
    ax_curve.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(DOCS / "06_features.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────

def main():
    DOCS.mkdir(parents=True, exist_ok=True)

    rgb_dir     = _resolve_rgb_dir()
    inf_dir     = _resolve_inference_dir()
    aligned_dir = SAMPLE_ROOT / f"aligned_{SQUARE_SIZE}" / PATIENT / EYE

    for p, name in [(rgb_dir, "RGB"), (inf_dir, "inference"), (aligned_dir, "aligned")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} dir not found: {p}\nRun the sample pipeline first.")

    payload = _load_payload()

    print(f"[figures] rgb_dir     = {rgb_dir}")
    print(f"[figures] inf_dir     = {inf_dir}")
    print(f"[figures] aligned_dir = {aligned_dir}")
    print(f"[figures] docs out    = {DOCS}")

    fig01_vascx_masks(rgb_dir, inf_dir);                     print("  wrote 01_vascx_masks.png")
    fig02_laterality(payload, rgb_dir);                      print("  wrote 02_laterality.png")
    fig03_registration(payload, rgb_dir, aligned_dir);       print("  wrote 03_registration.png")
    fig04_aligned_timeline(payload, aligned_dir);            print("  wrote 04_aligned_timeline.png")
    fig05_zones(payload, rgb_dir);                           print("  wrote 05_zones.png")
    fig06_features(payload, aligned_dir);                    print("  wrote 06_features.png")

    print("[done]")


if __name__ == "__main__":
    main()
