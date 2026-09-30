# fundus-vessel-features-2

Longitudinal retinal vessel biomarker extraction with **image registration** for same-patient fundus timelines.

Given a patient with multiple fundus visits, this pipeline:

1. Runs **VascX** inference (vessels, artery/vein, disc, fovea, quality — 7 models)
2. Registers all visits to the first visit using **EyeLiner** (SuperPoint+LightGlue + affine)
3. Extracts **~241 biomarkers** per visit across 12 categories (density, skeleton, caliber, fractal, Sholl, color, tortuosity, FAZ, crossings, bifurcation, width-variability, junction exponent)
4. Saves per-patient JSON that merges to a flat CSV

Why alignment? Fovea/disc-based zones (ETDRS, Wong Zone B/C) and spatial features (Sholl, FAZ) require the same anatomical location across visits. Registration reduces measurement noise and enables age-trend detection in zone-specific features (paired Wilcoxon p < 10⁻³ on our 30-patient sample; see [notebooks/03-compare-run-analysis.ipynb](notebooks/03-compare-run-analysis.ipynb)).

---

## Repository layout

```
fundus-vessel-features-2/
├── features/                 # Feature extractors (one module per category)
│   ├── density.py            # 8 whole + 88 zone metrics
│   ├── skeleton.py           # branch/endpoint/segment + tortuosity + bifurcation
│   ├── caliber.py            # CRAE/CRVE/AVR + width variability + junction exponent
│   ├── topology.py           # box-counting fractal + Sholl (fovea/disc centered)
│   ├── color.py              # RGB stats + AV whiteness (sclerosis marker)
│   ├── faz.py                # foveal avascular zone (DR/DME indicator)
│   ├── crossings.py          # A-V crossing points (hypertension marker)
│   └── zones.py              # ETDRS 9-subfield + Wong disc zones
├── scripts/
│   ├── pipeline_core.py      # shared: VascX loader, alignment, extraction, JSON I/O
│   ├── run_pipeline.py       # production (per-patient loop, JSON per patient)
│   └── run_compare.py        # 30-patient naive vs aligned sample for trend proof
├── notebooks/
│   ├── 01-alignment.ipynb              # VascX inference + EyeLiner registration
│   ├── 02-features-align.ipynb         # aligned feature extraction (per-patient JSON)
│   ├── 02-features-naive.ipynb         # naive (no alignment) baseline
│   ├── 02-features-compare.ipynb       # CV + trend comparison
│   ├── 03-align-zones-timeline.ipynb   # visualize aligned vessels + ETDRS zones
│   └── 03-compare-run-analysis.ipynb   # Wilcoxon + category-based gain analysis
├── EyeLiner/                 # bundled EyeLiner + LightGlue (SPLG affine registration)
├── vascx_models/             # local VascX helpers
├── weights/vascx/            # (git-ignored) VascX model weights ~2.2 GB
└── samples/                  # 3 de-identified fundus images (one patient, 3 visits)
    ├── images/pid_1.png  pid_2.png  pid_3.png
    └── sample_clinical.csv
```

---

## Quick start — sample walkthrough

The `samples/` folder contains **one patient, 3 consecutive visits** (single eye, ~1.5 years apart) so you can verify the pipeline end-to-end before pointing it at real data.

```powershell
# 1. Install dependencies (see below)
# 2. Download VascX weights to weights/vascx/

# 3. Run the aligned pipeline on the sample
python scripts/run_pipeline.py `
  --csv samples/sample_clinical.csv `
  --src-images samples/images `
  --output-dir samples/output `
  --mode aligned `
  --keep-masks
```

**What happens per patient (SAMPLE in this case)**:

| Step | Action | Output |
|---|---|---|
| 1 | Preprocess 3 images to 512×512 RGB + CE | `samples/output/preprocessed_512/SAMPLE/{rgb,ce}/pid_{1,2,3}.png` |
| 2 | VascX 7 models inference | `samples/output/inference_512/SAMPLE/{vessels,av,discs}/*.png` + inventory.csv |
| 3 | Detect eye side (fovea vs disc x-position) + disc geometry | `inventory.csv` (fovea_x/y, disc_cx/cy, disc_diameter, quality) |
| 4 | EyeLiner: fix pid_1 as reference, warp pid_2 & pid_3 to its frame | `samples/output/aligned_512/SAMPLE/R/{vessels,av,discs}/*.png` (warped) |
| 5 | Build fixed ETDRS + Wong zones from pid_1's fovea/disc | in-memory |
| 6 | Extract 241 features per visit (using warped masks + fixed zones) | `samples/output/features_512_aligned/SAMPLE.json` |
| 7 | Merge all patient JSONs into flat CSV | `samples/output/features_512_aligned/_merged_features_aligned.csv` |

**Expected output** (this sample only has 1 patient / 1 eye / 3 visits):
```
[1/1] SAMPLE  visits=3
[SAMPLE] preprocessed 3 images
[SAMPLE] inferred 3 images
[SAMPLE/R] 3 visits registering...
[SAMPLE] saved SAMPLE.json (3 visits, 1 eye)
```

Then visualize the aligned vessels overlaid with ETDRS zones:
```
jupyter notebook notebooks/03-align-zones-timeline.ipynb
```
Set `SOURCE = "script"`, `COMPARE_RUN = "samples/output"`, `PATIENT_ID = "SAMPLE"`, `EYE = "R"`.

---

## Full pipeline (production)

Same command, pointed at a real longitudinal cohort CSV:

```powershell
python scripts/run_pipeline.py `
  --csv path/to/longitudinal.csv `
  --src-images path/to/raw_images `
  --output-dir path/to/output `
  --mode aligned
```

**Options**:

| Flag | Default | Description |
|---|---|---|
| `--mode {aligned,naive}` | `aligned` | naive skips EyeLiner (per-visit reference) |
| `--keep-masks / --drop-masks` | `--drop-masks` | drop deletes preprocessed/inference/aligned masks per patient after JSON saved (~50× disk saving) |
| `--top-n N` | — | only visit-count top N patients |
| `--patient-ids ID1 ID2 …` | — | specific patients only |
| `--force` | off | re-extract even if patient JSON exists |
| `--limit N` | — | debug: first N patients |

**Behavior**:
- **Auto-skip**: if `<output>/features_512_aligned/<pid>.json` already exists, that patient is skipped (safe to resume after crash).
- **Cache**: preprocess + inference results are kept per-patient during a run; if they exist for the patient, skipped even without a JSON.

---

## Trend-revelation study (naive vs aligned)

Prove alignment matters by extracting **both** naive and aligned features on N sample patients, then compare with paired stats:

```powershell
python scripts/run_compare.py `
  --csv path/to/longitudinal.csv `
  --src-images path/to/raw_images `
  --output-dir path/to/compare_run `
  --n-patients 30 --min-total-visits 6
```

Produces:
- `features_512_aligned/*.json`, `features_512_naive/*.json`
- `cv_comparison.csv` (per-feature CV_naive/CV_aligned/reduction%)
- `trend_revelation.csv` (per-feature time_r_naive/aligned + verdict)
- `summary.csv` (merged)

Then run [notebooks/03-compare-run-analysis.ipynb](notebooks/03-compare-run-analysis.ipynb) for:
- Paired Wilcoxon test (H1: aligned time_r > naive)
- Category-wise gain bar chart (sholl / faz-cross / zone / whole)
- Strict vs relaxed verdict distributions

**Our finding on 30 patients**: Wilcoxon p = 5×10⁻⁴. Whole-image aggregates (121 features) show no gain (rotation-invariant), but spatial features benefit: **Sholl 83% positive gain, FAZ/crossings 79%, zones 59%.**

---

## Installation

Python **3.11**. CUDA GPU strongly recommended (VascX + EyeLiner both use PyTorch).

```powershell
python -m venv .venv
.venv\Scripts\activate

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install rtnls_inference rtnls_fundusprep
pip install -r requirements.txt   # scipy, scikit-image, opencv, matplotlib, pandas, tqdm, monai
```

### VascX weights

Not in this repository (~2.2 GB). Download from [huggingface.co/Eyened/vascx](https://huggingface.co/Eyened/vascx) and place under `weights/vascx/`:

```
weights/vascx/
├── quality.pt
├── vessels_july24.pt
├── av_july24.pt
├── disc_july24.pt
├── fovea_july24.pt
├── discedge_july24.pt
└── odfd_march25.pt
```

### EyeLiner

Bundled locally under `EyeLiner/` (LightGlue + SuperPoint). No download required.

---

## Feature catalog (~241 columns)

| Category | Function | Count | Source |
|---|---|---|---|
| Whole density | `whole_image_density_features` | 8 | vessel/artery/vein/disc density + AV ratio |
| Zone density | `zone_density_features` | 88 | 11 zones (9 ETDRS + Wong B/C) × 8 metrics |
| Whole skeleton | `whole_image_skeleton_features` | 9 | branch/endpoint density per network |
| Whole segments | `whole_image_segment_features` | 15 | segment count + length stats |
| Whole caliber | `whole_image_caliber_features` | 18 | width mean/std/median × 3 network |
| Knudtson | `compute_crae_crve_avr` | 3 | CRAE, CRVE, AVR from Zone B |
| Fractal | `whole_image_fractal_features` | 3 | box-counting per network |
| Sholl | `sholl_features` × 2 centers | 24 | 2 centers × 3 network × 4 stats |
| Color | `whole_image_color_features` | 30 | RGB stats + AV whiteness (sclerosis marker) |
| Tortuosity | `whole_image_tortuosity_features` | 12 | arc/chord ratio stats |
| FAZ | `compute_faz` | 5 | foveal avascular zone area/diameter/circularity |
| Crossings | `crossings_features` | 9 | A-V crossings (whole + Zone B + Zone C) |
| Bifurcation | `whole_image_bifurcation_features` | 12 | angle stats + deviation from Murray 82.5° |
| Width variability | `whole_image_width_variability` | 6 | segment caliber CoV per network |
| Junction exponent | `whole_image_junction_features` | 9 | Murray junction exponent X (≈3 optimal) |

Zone definitions:
- **ETDRS 9** (fovea-centered): C (central 1 mm), inner ring S1/N1/I1/T1 (1–3 mm), outer ring S2/N2/I2/T2 (3–6 mm)
- **Wong disc zones** (Knudtson): Zone B = 0.5–1.0 DD from disc margin, Zone C = 1.0–2.0 DD

---

## Data schema

**Per-patient JSON** (`<pid>.json`):

```json
{
  "patient": "SAMPLE",
  "config": {"mode": "aligned", "square_size": 512, "extracted_at": "2026-10-01T..."},
  "meta_by_eye": {
    "R": {
      "fovea_x": 258.3, "fovea_y": 256.1,
      "disc_cx": 106.5, "disc_cy": 250.8,
      "disc_diameter_px": 67.7,
      "mm_per_px": 0.02657,
      "n_visits": 3
    }
  },
  "features": [
    {"id": "pid_1", "patient": "SAMPLE", "eye": "R", "date": "2020-01-15", "shot": 1,
     "disc_diameter_px": 66.4, "mm_per_px": 0.02710,
     "whole_vessel_density": 0.128, "whole_artery_density": 0.061, ...},
    {"id": "pid_2", ...},
    {"id": "pid_3", ...}
  ]
}
```

**Merged CSV**: one row per (patient, visit), columns = meta + all 241 features.

---

## Attribution

- **VascX**: Eyened Reading Center. Models under Eyened HuggingFace license. See [arxiv.org/abs/2409.16016](https://arxiv.org/abs/2409.16016).
- **EyeLiner**: SuperPoint + LightGlue for retinal alignment. Bundled locally under `EyeLiner/`.
- **rtnls_inference**, **rtnls_fundusprep**: Eyened preprocess/inference toolkits, installed via pip.

Sample images in `samples/images/` are de-identified from a longitudinal fundus cohort. All patient-identifying metadata (names, hospital IDs, real dates) has been removed; the patient ID is replaced with `SAMPLE` and dates are synthetic.
