"""Per-patient aligned feature-extraction pipeline (CLI).

Reads a clinical CSV, then for each patient:
    1. Preprocess + VascX inference (7 models).
    2. Determine laterality + fovea/disc geometry + quality.
    3. EyeLiner registration (first visit = fixed, rest warped to it).
    4. Extract ~241 features per visit.
    5. Save patient JSON.
    6. If --drop-masks, delete per-patient preprocessed/inference/aligned masks.

Example (sample dataset bundled with the repo):

    python scripts/run_pipeline.py `
      --csv samples/sample_clinical.csv `
      --src-images samples/images `
      --output-dir samples/output `
      --keep-masks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from pipeline_core import (
    PipelineConfig, load_vascx_models, _setup_eyeliner,
    run_patient, merge_patient_jsons,
)


def parse_args():
    p = argparse.ArgumentParser(description="Per-patient aligned feature-extraction pipeline")
    p.add_argument("--csv", type=Path, required=True, help="clinical CSV path")
    p.add_argument("--src-images", type=Path, required=True, help="raw fundus images dir")
    p.add_argument("--output-dir", type=Path, required=True, help="output root")
    p.add_argument("--weights-dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "weights" / "vascx",
                   help="VascX weights dir")

    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")

    mask_grp = p.add_mutually_exclusive_group()
    mask_grp.add_argument("--keep-masks", dest="keep_masks", action="store_true",
                          help="keep preprocessed/inference/aligned masks after each patient (default: drop)")
    mask_grp.add_argument("--drop-masks", dest="keep_masks", action="store_false",
                          help="delete masks after the patient JSON is written (saves disk)")
    p.set_defaults(keep_masks=False)

    p.add_argument("--col-patient", default="ID")
    p.add_argument("--col-filename", default="Filename")
    p.add_argument("--col-date", default="LAB_DTM")
    p.add_argument("--col-shot", default="shot")

    p.add_argument("--patient-ids", nargs="+", default=None,
                   help="only process these patient IDs (space-separated); default: every patient in the CSV")
    p.add_argument("--top-n", type=int, default=None,
                   help="only the top-N patients by visit count (mutually exclusive with --patient-ids)")
    p.add_argument("--min-visits", type=int, default=2,
                   help="skip any eye with fewer than this many visits")

    p.add_argument("--force", action="store_true",
                   help="re-extract even if <features_root>/<pid>.json already exists")
    p.add_argument("--limit", type=int, default=None,
                   help="debug: process at most N patients")
    return p.parse_args()


def build_config(a) -> PipelineConfig:
    return PipelineConfig(
        csv_path=a.csv, src_images_dir=a.src_images, output_dir=a.output_dir,
        weights_dir=a.weights_dir, square_size=a.square_size, device=a.device,
        keep_masks=a.keep_masks,
        col_patient=a.col_patient, col_filename=a.col_filename,
        col_date=a.col_date, col_shot=a.col_shot,
        patient_ids=a.patient_ids, min_visits_per_eye=a.min_visits,
    )


def select_patients(cfg: PipelineConfig, df: pd.DataFrame, top_n=None) -> list:
    if cfg.patient_ids:
        return [str(p) for p in cfg.patient_ids]
    counts = df.groupby(cfg.col_patient).size().sort_values(ascending=False)
    if top_n is not None:
        counts = counts.head(top_n)
    return [str(p) for p in counts.index]


def main():
    args = parse_args()
    cfg = build_config(args)

    print(f"[config] square_size={cfg.square_size}  keep_masks={cfg.keep_masks}")
    print(f"[config] output_dir={cfg.output_dir}")
    print(f"[config] features_root={cfg.features_root}")

    df = pd.read_csv(cfg.csv_path, low_memory=False)
    df[cfg.col_date] = pd.to_datetime(df[cfg.col_date])
    print(f"[csv] rows={len(df)}, patients={df[cfg.col_patient].nunique()}")

    patient_ids = select_patients(cfg, df, top_n=args.top_n)
    if args.limit is not None:
        patient_ids = patient_ids[:args.limit]
    print(f"[patients] {len(patient_ids)} selected")

    print("[models] loading VascX ...")
    models, device = load_vascx_models(cfg)
    eyeliner = _setup_eyeliner(cfg, device)
    print(f"[models] device={device}")

    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()
    for i, pid in enumerate(patient_ids, 1):
        rows = df[df[cfg.col_patient].astype(str) == pid]
        if len(rows) == 0:
            print(f"[{i}/{len(patient_ids)}] {pid}  no CSV rows")
            n_skip += 1
            continue

        print(f"[{i}/{len(patient_ids)}] {pid}  visits={len(rows)}")
        try:
            r = run_patient(cfg, models, device, eyeliner, pid, rows,
                            verbose=True, force=args.force)
            if r is None:
                n_fail += 1
            else:
                n_done += 1
        except Exception as e:
            print(f"  {pid} failed: {e}")
            import traceback
            traceback.print_exc()
            n_fail += 1

    dt = time.time() - t0
    print(f"\n[done] {n_done} saved, {n_skip} skip, {n_fail} fail  ({dt/60:.1f} min)")

    merged = merge_patient_jsons(cfg.features_root)
    if len(merged):
        csv_path = cfg.features_root / "_merged_features_aligned.csv"
        merged.to_csv(csv_path, index=False)
        print(f"[merge] {csv_path}  rows={len(merged)}, cols={merged.shape[1]}")
    else:
        print("[merge] no patient JSON found")


if __name__ == "__main__":
    main()
