# FundusAlign

Longitudinal retinal vessel biomarker extraction with per-patient image alignment.

Built on **[retinalysis-vascx](https://github.com/Eyened/retinalysis-vascx)** for segmentation and **[EyeLiner](https://github.com/QTIM-Lab/EyeLiner)** for pairwise registration.

Pipeline per patient:
1. VascX inference (vessels, A/V, disc, fovea, quality)
2. EyeLiner affine registration (first visit = fixed, rest warped to it)
3. Extract ~241 biomarkers per visit over fixed ETDRS + Wong disc zones
4. Save per-patient JSON (mergeable to CSV)

---

## Quick start — sample

```powershell
python scripts/run_pipeline.py `
  --csv samples/sample_clinical.csv `
  --src-images samples/images `
  --output-dir samples/output `
  --keep-masks
```

Produces `samples/output/features_512_aligned/SAMPLE.json` (one patient, 3 visits).

For an interactive walkthrough: run `notebooks/01-alignment.ipynb` then `notebooks/02-features-align.ipynb`.

---

## Full run

```powershell
python scripts/run_pipeline.py `
  --csv path/to/clinical.csv `
  --src-images path/to/raw_images `
  --output-dir path/to/output
```

CLI options: `--keep-masks` · `--top-n N` · `--patient-ids ...` · `--force` · `--limit N` · `--min-visits N`.
Auto-resume: an existing patient JSON is skipped; cached preprocess/inference is reused.

---

## Repository

```
features/      density, skeleton, caliber, topology, color, faz, crossings, zones
scripts/       pipeline_core.py, run_pipeline.py
notebooks/     01-alignment.ipynb, 02-features-align.ipynb
EyeLiner/      bundled EyeLiner + LightGlue (Apache 2.0 + BSD 3-Clause)
weights/vascx/ VascX weights — download from HuggingFace (git-ignored)
samples/       3 de-identified fundus images (one patient, 3 visits)
```

---

## Install

Python 3.11, CUDA GPU.

```powershell
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install rtnls_inference rtnls_fundusprep
pip install -r requirements.txt
```

VascX weights (~2.2 GB) — download from [huggingface.co/Eyened/vascx](https://huggingface.co/Eyened/vascx) → `weights/vascx/`.

---

## Attribution

- **VascX** (segmentation + biomarkers): Vargas-Quiros J.D. et al. *retinalysis-vascx: an explainable software toolbox for the extraction of retinal vascular biomarkers from color fundus images.* 2026. [arXiv:2602.08580](https://arxiv.org/abs/2602.08580). AGPL-3.0.
- **EyeLiner** (registration): QTIM-Lab. Apache 2.0.
- **LightGlue** (feature matching): ETH Zurich. BSD 3-Clause.

Sample images are de-identified from a longitudinal cohort. Patient ID → `SAMPLE`, dates synthetic.
