"""Naive vs Aligned trend-revelation comparison — 샘플 환자 N 명.

목적:
  같은 환자 세트에 대해 naive + aligned 두 파이프라인 결과를 저장하고,
  feature 별 CV reduction + age-trend revelation 분석 → alignment 필요성 증명.

플로우:
  1. CSV 에서 visit 수 상위 N 명 (또는 min-visits 이상) 샘플링
  2. 각 환자 → naive JSON 저장, aligned JSON 저장 (같은 preprocess/inference 재사용)
  3. 두 CSV merge (naive_features.csv, aligned_features.csv)
  4. Trend analysis:
     - CV_naive vs CV_aligned per feature
     - time_r_naive vs time_r_aligned per feature × (patient, eye)
     - trend_verdict: revealed / both_trend / stationary / erased / mixed
  5. Summary CSV + 콘솔 리포트

사용 예:
  python scripts/run_compare.py `
    --csv path/to/longitudinal.csv --src-images path/to/raw_images `
    --output-dir path/to/compare_run `
    --n-patients 30 --min-total-visits 6
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path: sys.path.insert(0, str(_HERE))
from pipeline_core import (
    PipelineConfig, load_vascx_models, _setup_eyeliner,
    run_patient, merge_patient_jsons,
)


# ═════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="Naive vs Aligned trend comparison")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--src-images", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "weights" / "vascx")

    p.add_argument("--square-size", type=int, default=512)
    p.add_argument("--device", default="cuda:0")

    # 표본 선정
    p.add_argument("--n-patients", type=int, default=30)
    p.add_argument("--min-total-visits", type=int, default=6,
                    help="양안 합쳐 이 이상 visit 있는 환자만 후보")
    p.add_argument("--min-visits-per-eye", type=int, default=3,
                    help="한 눈에 이 이상 visit 있어야 그 눈 처리")
    p.add_argument("--seed", type=int, default=42)

    # 재실행 동작 (기본: 이미 JSON 있으면 스킵)
    p.add_argument("--force", action="store_true",
                    help="이미 JSON 존재해도 재추출 (기본은 자동 스킵)")
    p.add_argument("--keep-masks", action="store_true",
                    help="masks 유지 (기본은 환자 끝나면 삭제)")

    # 분석만 (추출 스킵)
    p.add_argument("--analyze-only", action="store_true",
                    help="추출 스킵, features_root 의 JSON 만 읽어 분석")

    p.add_argument("--col-patient", default="ID")
    p.add_argument("--col-filename", default="Filename")
    p.add_argument("--col-date", default="LAB_DTM")
    p.add_argument("--col-shot", default="shot")
    return p.parse_args()


# ═════════════════════════════════════════════════════════════════════
# 표본 선정
# ═════════════════════════════════════════════════════════════════════

def sample_patients(df: pd.DataFrame, args) -> list:
    """min-total-visits 이상 환자 중 상위 N + 랜덤 fallback."""
    counts = df.groupby(args.col_patient).size()
    eligible = counts[counts >= args.min_total_visits].index.tolist()
    print(f"[sample] eligible (>={args.min_total_visits} visits): {len(eligible)} / {counts.size}")

    if len(eligible) <= args.n_patients:
        return [str(p) for p in eligible]

    # 상위 visit 수 절반 + 나머지 랜덤 절반 (다양성)
    rng = np.random.default_rng(args.seed)
    top = counts.loc[eligible].sort_values(ascending=False).head(args.n_patients // 2).index.tolist()
    rest = [p for p in eligible if p not in top]
    sampled_rest = rng.choice(rest, size=args.n_patients - len(top), replace=False)
    return [str(p) for p in list(top) + list(sampled_rest)]


# ═════════════════════════════════════════════════════════════════════
# 분석
# ═════════════════════════════════════════════════════════════════════

META_COLS = {
    "id", "patient", "eye", "date", "shot",
    "disc_diameter_px", "mm_per_px",
    "q1", "q2", "q3", "x_fovea", "y_fovea", "x_disc", "y_disc",
}


def analyze(naive_csv: Path, aligned_csv: Path, out_dir: Path):
    """CV + trend revelation 분석."""
    if not naive_csv.exists() or not aligned_csv.exists():
        print(f"[analyze] CSV 없음: {naive_csv.exists()}, {aligned_csv.exists()}")
        return

    fn = pd.read_csv(naive_csv, parse_dates=["date"])
    fa = pd.read_csv(aligned_csv, parse_dates=["date"])
    print(f"[analyze] naive: {fn.shape}, aligned: {fa.shape}")

    feature_cols = [c for c in fn.columns
                     if c not in META_COLS and pd.api.types.is_numeric_dtype(fn[c])]
    common = [c for c in feature_cols if c in fa.columns]
    print(f"[analyze] {len(common)} numeric features 비교")

    # ── CV analysis (per patient × eye 평균) ──
    def cv_per_group(df, col):
        rows = []
        for (pid, eye), g in df.groupby(["patient", "eye"]):
            v = g[col].dropna()
            if len(v) < 3: continue
            m = v.mean()
            rows.append(100 * v.std() / m if m else np.nan)
        return np.nanmean(rows) if rows else np.nan

    cv_rows = []
    for c in common:
        cvn = cv_per_group(fn, c)
        cva = cv_per_group(fa, c)
        cv_rows.append({
            "feature": c,
            "cv_naive": cvn, "cv_aligned": cva,
            "reduction_%": (cvn - cva) / cvn * 100 if cvn and np.isfinite(cvn) else np.nan,
        })
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(out_dir / "cv_comparison.csv", index=False)
    print(f"[analyze] cv_comparison.csv saved  ({len(cv_df)} features)")

    # ── Trend revelation (per patient × eye time correlation) ──
    def time_r_per_group(df, col):
        rows = []
        for (pid, eye), g in df.groupby(["patient", "eye"]):
            g = g.sort_values("date").dropna(subset=[col, "date"])
            if len(g) < 5: continue
            t = (g["date"] - g["date"].min()).dt.days.values.astype(float)
            y = g[col].values.astype(float)
            if t.std() == 0 or y.std() == 0: continue
            rows.append(abs(float(np.corrcoef(t, y)[0, 1])))
        return np.nanmean(rows) if rows else np.nan

    tr_rows = []
    for c in common:
        trn = time_r_per_group(fn, c)
        tra = time_r_per_group(fa, c)
        tr_rows.append({
            "feature": c,
            "time_r_naive": trn, "time_r_aligned": tra,
            "time_r_gain": (tra - trn) if (np.isfinite(tra) and np.isfinite(trn)) else np.nan,
        })
    tr_df = pd.DataFrame(tr_rows)

    def classify(row):
        ra, rn = row["time_r_aligned"], row["time_r_naive"]
        if pd.isna(ra) or pd.isna(rn): return "unknown"
        if ra >= 0.5 and rn <  0.3: return "revealed"
        if ra >= 0.5 and rn >= 0.5: return "both_trend"
        if ra <  0.3 and rn <  0.3: return "stationary"
        if ra <  0.3 and rn >= 0.5: return "erased"
        return "mixed"
    tr_df["trend_verdict"] = tr_df.apply(classify, axis=1)
    tr_df.to_csv(out_dir / "trend_revelation.csv", index=False)

    # ── Combined summary ──
    summary = cv_df.merge(tr_df, on="feature")
    summary.to_csv(out_dir / "summary.csv", index=False)

    # ── 콘솔 리포트 ──
    print("\n" + "="*60)
    print("  CV reduction  (양수 = alignment 이 CV 낮춤)")
    print("="*60)
    print(f"  median reduction: {cv_df['reduction_%'].median():.1f}%")
    print(f"  # feat with reduction > 10%: "
          f"{(cv_df['reduction_%'] > 10).sum()} / {len(cv_df)}")
    print(f"  # feat with reduction < -10% (aligned 더 나쁨): "
          f"{(cv_df['reduction_%'] < -10).sum()} / {len(cv_df)}")

    print("\n" + "="*60)
    print("  Trend revelation")
    print("="*60)
    for v in ["revealed", "both_trend", "stationary", "erased", "mixed", "unknown"]:
        n = (tr_df["trend_verdict"] == v).sum()
        print(f"  {v:12s}: {n:>4} features")

    top_gain = tr_df.dropna(subset=["time_r_gain"]).sort_values("time_r_gain", ascending=False).head(15)
    print("\n  Top 15 time_r_gain (aligned trend > naive trend):")
    for _, r in top_gain.iterrows():
        print(f"    {r['feature']:50s}  gain={r['time_r_gain']:+.3f} "
              f"(naive={r['time_r_naive']:.2f} → aligned={r['time_r_aligned']:.2f})")

    print(f"\n[analyze] 저장:")
    for f in ["cv_comparison.csv", "trend_revelation.csv", "summary.csv"]:
        print(f"  - {out_dir / f}")


# ═════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    df[args.col_date] = pd.to_datetime(df[args.col_date])
    patient_ids = sample_patients(df, args)
    print(f"[patients] sampled {len(patient_ids)}: {patient_ids[:5]}...")

    naive_dir   = out_dir / f"features_{args.square_size}_naive"
    aligned_dir = out_dir / f"features_{args.square_size}_aligned"

    if not args.analyze_only:
        print("\n[models] VascX 로딩 (1회)...")
        cfg_shared = PipelineConfig(
            csv_path=args.csv, src_images_dir=args.src_images, output_dir=args.output_dir,
            weights_dir=args.weights_dir, square_size=args.square_size, device=args.device,
            mode="aligned", keep_masks=True,   # 첫 번째 (aligned) 는 keep 해서 naive 재사용
            col_patient=args.col_patient, col_filename=args.col_filename,
            col_date=args.col_date, col_shot=args.col_shot,
            min_visits_per_eye=args.min_visits_per_eye,
        )
        models, device = load_vascx_models(cfg_shared)
        eyeliner = _setup_eyeliner(cfg_shared, device)

        # 환자별 loop: aligned + naive 두 번 (preprocess/inference 는 공유됨)
        for i, pid in enumerate(patient_ids, 1):
            rows = df[df[args.col_patient].astype(str) == pid]
            if len(rows) == 0:
                print(f"[{i}/{len(patient_ids)}] {pid}  no CSV rows"); continue

            # --- 1) aligned mode (keep intermediate) ---
            cfg_a = PipelineConfig(
                csv_path=args.csv, src_images_dir=args.src_images, output_dir=args.output_dir,
                weights_dir=args.weights_dir, square_size=args.square_size, device=args.device,
                mode="aligned", keep_masks=True,
                col_patient=args.col_patient, col_filename=args.col_filename,
                col_date=args.col_date, col_shot=args.col_shot,
                min_visits_per_eye=args.min_visits_per_eye,
            )
            print(f"[{i}/{len(patient_ids)}] {pid}  visits={len(rows)}")
            try:
                run_patient(cfg_a, models, device, eyeliner, pid, rows,
                             verbose=True, force=args.force)
            except Exception as e:
                print(f"  ❌ {pid} aligned 실패: {e}")
                import traceback; traceback.print_exc()

            # --- 2) naive mode (기존 inference/preprocess 재사용) ---
            cfg_n = PipelineConfig(
                csv_path=args.csv, src_images_dir=args.src_images, output_dir=args.output_dir,
                weights_dir=args.weights_dir, square_size=args.square_size, device=args.device,
                mode="naive", keep_masks=args.keep_masks,
                col_patient=args.col_patient, col_filename=args.col_filename,
                col_date=args.col_date, col_shot=args.col_shot,
                min_visits_per_eye=args.min_visits_per_eye,
            )
            try:
                run_patient(cfg_n, models, device, None, pid, rows,
                             verbose=True, force=args.force)
            except Exception as e:
                print(f"  ❌ {pid} naive 실패: {e}")
                import traceback; traceback.print_exc()

            # 이 환자 처리 끝 → keep_masks=False 면 정리 (naive 가 이미 정리했을 수도)
            if not args.keep_masks:
                for base in [cfg_n.prep_root, cfg_n.inference_root, cfg_n.aligned_root]:
                    d = base / str(pid)
                    if d.exists(): shutil.rmtree(d, ignore_errors=True)

        # merge JSONs
        naive_merged   = merge_patient_jsons(cfg_n.features_root)
        aligned_merged = merge_patient_jsons(cfg_a.features_root)
        n_csv = cfg_n.features_root / f"_merged_features_naive.csv"
        a_csv = cfg_a.features_root / f"_merged_features_aligned.csv"
        if len(naive_merged):   naive_merged.to_csv(n_csv, index=False)
        if len(aligned_merged): aligned_merged.to_csv(a_csv, index=False)
        print(f"[merge] naive:   {n_csv}  rows={len(naive_merged)}")
        print(f"[merge] aligned: {a_csv}  rows={len(aligned_merged)}")

    # 분석
    n_csv = naive_dir   / f"_merged_features_naive.csv"
    a_csv = aligned_dir / f"_merged_features_aligned.csv"
    analyze(n_csv, a_csv, out_dir)


if __name__ == "__main__":
    main()
