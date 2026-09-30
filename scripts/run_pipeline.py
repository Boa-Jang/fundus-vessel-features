"""Production feature-extraction pipeline (per-patient).

CSV 를 읽고 환자 단위로:
  1. 전처리 + VascX 추론 (7 model)
  2. 좌우안 판정 + fovea/disc geometry + quality
  3. Registration (aligned) or identity (naive)
  4. Feature 241 개 추출
  5. Patient JSON 저장
  6. --drop-masks 면 mask 삭제 (JSON 만 남김)

사용 예 (PowerShell) — 샘플 데이터로 검증:
  python scripts/run_pipeline.py `
    --csv samples/sample_clinical.csv `
    --src-images samples/images `
    --output-dir samples/output `
    --mode aligned --keep-masks
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Windows cp949 → utf-8 (ensures emoji/한글 print 안 깨짐)
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path: sys.path.insert(0, str(_HERE))
from pipeline_core import (
    PipelineConfig, load_vascx_models, _setup_eyeliner,
    run_patient, merge_patient_jsons,
)


def parse_args():
    p = argparse.ArgumentParser(description="Per-patient feature extraction pipeline")
    p.add_argument("--csv", type=Path, required=True, help="clinical CSV path")
    p.add_argument("--src-images", type=Path, required=True, help="raw fundus images dir")
    p.add_argument("--output-dir", type=Path, required=True, help="output root")
    p.add_argument("--weights-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "weights" / "vascx",
                    help="VascX weights dir")

    p.add_argument("--mode", choices=["aligned", "naive"], default="aligned")
    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")

    # mask disk 관리
    mask_grp = p.add_mutually_exclusive_group()
    mask_grp.add_argument("--keep-masks", dest="keep_masks", action="store_true",
                           help="preprocess + inference + aligned mask 모두 유지 (재분석 용이)")
    mask_grp.add_argument("--drop-masks", dest="keep_masks", action="store_false",
                           help="환자 처리 후 mask 삭제 (JSON 만 남김, 용량 절약)")
    p.set_defaults(keep_masks=False)

    # CSV 컬럼 이름 (필요 시 override)
    p.add_argument("--col-patient", default="ID")
    p.add_argument("--col-filename", default="Filename")
    p.add_argument("--col-date", default="LAB_DTM")
    p.add_argument("--col-shot", default="shot")

    # 환자 필터
    p.add_argument("--patient-ids", nargs="+", default=None,
                    help="특정 환자 ID 만 처리 (space-separated). 없으면 CSV 전체.")
    p.add_argument("--top-n", type=int, default=None,
                    help="visit 수 상위 N 명만 (patient-ids 와 배타적)")
    p.add_argument("--min-visits", type=int, default=2,
                    help="한 눈에 이 이하 visit 이면 그 눈 스킵")

    p.add_argument("--force", action="store_true",
                    help="이미 <features_root>/<pid>.json 있어도 재추출 (기본은 자동 스킵)")
    p.add_argument("--limit", type=int, default=None,
                    help="max 환자 수 (디버깅용)")
    return p.parse_args()


def build_config(a) -> PipelineConfig:
    return PipelineConfig(
        csv_path=a.csv, src_images_dir=a.src_images, output_dir=a.output_dir,
        weights_dir=a.weights_dir, square_size=a.square_size, device=a.device,
        mode=a.mode, keep_masks=a.keep_masks,
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

    print(f"[config] mode={cfg.mode}  square_size={cfg.square_size}  keep_masks={cfg.keep_masks}")
    print(f"[config] output_dir={cfg.output_dir}")
    print(f"[config] features_root={cfg.features_root}")

    # 1. CSV 로드
    df = pd.read_csv(cfg.csv_path, low_memory=False)
    df[cfg.col_date] = pd.to_datetime(df[cfg.col_date])
    print(f"[csv] rows={len(df)}, patients={df[cfg.col_patient].nunique()}")

    patient_ids = select_patients(cfg, df, top_n=args.top_n)
    if args.limit is not None:
        patient_ids = patient_ids[:args.limit]
    print(f"[patients] {len(patient_ids)} 선정")

    # 2. VascX + EyeLiner 준비
    print("[models] VascX 로딩...")
    models, device = load_vascx_models(cfg)
    eyeliner = _setup_eyeliner(cfg, device) if cfg.mode == "aligned" else None
    print(f"[models] device={device}, mode={cfg.mode}")

    # 3. 환자별 실행
    n_done, n_skip, n_fail = 0, 0, 0
    t0 = time.time()
    for i, pid in enumerate(patient_ids, 1):
        rows = df[df[cfg.col_patient].astype(str) == pid]
        if len(rows) == 0:
            print(f"[{i}/{len(patient_ids)}] {pid}  no CSV rows"); n_skip += 1; continue

        print(f"[{i}/{len(patient_ids)}] {pid}  visits={len(rows)}")
        try:
            r = run_patient(cfg, models, device, eyeliner, pid, rows,
                             verbose=True, force=args.force)
            if r is None: n_fail += 1
            else: n_done += 1
        except Exception as e:
            print(f"  ❌ {pid} 실패: {e}")
            import traceback; traceback.print_exc()
            n_fail += 1

    dt = time.time() - t0
    print(f"\n[done] {n_done} saved, {n_skip} skip, {n_fail} fail  ({dt/60:.1f} min)")

    # 4. Merge → 단일 CSV (편의)
    merged = merge_patient_jsons(cfg.features_root)
    if len(merged):
        csv_path = cfg.features_root / f"_merged_features_{cfg.mode}.csv"
        merged.to_csv(csv_path, index=False)
        print(f"[merge] {csv_path}  rows={len(merged)}, cols={merged.shape[1]}")
    else:
        print("[merge] no patient JSON 발견")


if __name__ == "__main__":
    main()
