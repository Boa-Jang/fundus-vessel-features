# Fundus vessel features

Measure retinal vessel morphology from colour fundus photographs.
Segmentation with [VascX](https://github.com/Eyened), graph and geometry
measurement with `retinal_graph`. **1,098 columns per image**: 1,087 measured
features plus 11 carried over from VascX.

Whole-image graph metrics, disc-centred zones, anatomical sectors, vessel
calibre (CRAE / CRVE / AVR), and optic-disc vasculature - one flat table,
joined on `image_id`.

---

## Quick start

```bash
git clone <this-repo> && cd fundus-vessel-features
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# model weights are not in git -- see "Model weights" below
python check_env.py            # tells you exactly what is missing

python preprocess.py           # optional but recommended - see below
python extract.py --save-masks
python verify_sample.py
```

The repository ships **5 de-identified sample images** already wired up, so
a fresh clone runs end to end before you point it at your own data.

---

## Installation

Python **3.11**. A CUDA GPU is strongly recommended; CPU works but is slow.

`torch` is installed separately on purpose - the right CUDA build depends on
your driver and comes from a different package index. Change `cu128` to match
yours. Versions in `requirements.txt` are the set this was validated against,
not lower bounds.

### Model weights

**Weights are not in this repository.** VascX is about 2.2 GB, far past what
git should carry, so `vascx/weights/` is git-ignored.

All weights live under one folder so they can be distributed as a single
download. Unpack it at the repository root:

```
weights/
├── vascx/                      2.2 GB - required
│   ├── vessels_july24.pt
│   ├── av_july24.pt
│   ├── disc_july24.pt
│   ├── fovea_july24.pt
│   └── quality.pt
└── rag/                        333 MB - optional, retinal age only
    ├── rag_model.pth
    └── validation_correction.json
```

Point somewhere else with `FUNDUS_WEIGHTS=/path/to/weights` if you keep them
outside the repository.

**Download:** <https://drive.google.com/drive/folders/1ELyt70KiBsw8hwVNnwf7lHnD5HPjTgwq?usp=drive_link>

Unzip so that `weights/vascx/` and `weights/rag/` sit at the repository root,
then `python check_env.py` will confirm they are found.

The preprocessing code (`vascx/rtnls_fundusprep`) is vendored from
<https://github.com/Eyened/retinalysis-fundusprep>; the segmentation weights
come from the same group. Check their licence terms before redistributing.

`python check_env.py` verifies imports, CUDA, weights, worklist, images and
free disk space, and exits non-zero if the pipeline cannot run. **Run it
first** - it is much cheaper than discovering a missing weight file partway
through a long job.

```bash
python config.py      # prints every path it resolved
```

---

## Usage

### On the bundled samples

```bash
python extract.py --save-masks
python verify_sample.py
```

_No reference values are bundled in this build, so `verify_sample.py` runs
structural checks only._
`verify_sample.py` compares every numeric column against it, so passing means
your install reproduces our numbers - not merely that the script exited 0.
A percent or two of drift is normal (GPU model, cuDNN, driver); a column that
turns `NaN` or moves past the tolerance is not.

### Preprocessing (recommended)

```bash
python preprocess.py --images /data/raw --out /data/prepped --size 1024
FUNDUS_IMAGES=/data/prepped python extract.py --save-masks
```

Fundus photographs are not square and the retina rarely fills the frame.
Resizing straight to a square stretches the circle by however much the camera's
aspect ratio demands - so two eras of one cohort reach the model distorted by
different amounts. Measured on a single eye photographed either side of a
camera change, the optic disc came out **1.31x** different in a cache built that
way, and nothing downstream undoes it.

`preprocess.py` finds the fundus circle, pads the short axis with black so the
frame is square without stretching, centres the circle, and records an
estimated micron-per-pixel per image in `preprocess_manifest.csv`.

It is optional - the pipeline runs on raw images - but **required if you want
numbers comparable with the bundled reference values**, because both VascX and
the retinal-age model saw images prepared this way.

### On your own images

**Point it at a folder. That is all.**

```bash
FUNDUS_IMAGES=/data/fundus FUNDUS_OUT=/data/results python extract.py --save-masks
```

```powershell
$env:FUNDUS_IMAGES="D:\dataundus"; $env:FUNDUS_OUT="D:\data
esults"
python extract.py --save-masks
```

There is **no worklist to write**. If one does not exist, `extract.py` scans the
images directory and generates it, taking `image_id` from each filename:

```
no worklist found -- generated one for 1,284 images in /data/fundus
```

Accepted: `.jpg .jpeg .png .tif .tiff .bmp .webp`. Filename stems must be
unique, since they become the row keys.

Sub-directories are handled too: if the top level holds no images, it searches
below and builds ids from the relative path, so two files both called `01.jpg`
in different folders stay distinct.

<details>
<summary>Supplying your own worklist</summary>

Only needed if you already have measured laterality or fovea coordinates - they
take priority over what VascX detects. Write a CSV and point `FUNDUS_WORKLIST`
at it:

| column | required | meaning |
|---|---|---|
| `image_id` | yes | unique id; becomes the row key |
| `relpath` | yes | path relative to the images directory |
| `laterality` | no | `L` or `R` |
| `fovea_x`, `fovea_y` | no | fovea pixel coordinates |
| `w_img` | no | image width those coordinates refer to |

```bash
FUNDUS_IMAGES=/data/fundus FUNDUS_WORKLIST=/data/worklist.csv python extract.py
```

</details>

### Resuming

**Automatic.** Re-run the same command; it reads the finished chunks, reports
how many images are already done, and continues. There is no resume flag.

```bash
python extract.py --save-masks     # again, after an interruption
python extract.py --combine        # chunks -> one table, at the end
```

Useful flags: `--limit N` smoke test, `--jobs N` CPU workers, `--batch-size N`
GPU stage, `--shard i/n` split across machines, `--no-zones` for whole-image
features only (much faster, ~250 columns).

---

## Output

| path | contents |
|---|---|
| `output/chunks_graph/chunk_*.csv` | per-chunk feature tables, written as they finish |
| `output/graph_features.csv` | combined table, after `--combine` - **1,098 columns** |
| `output/masks/` | segmentation masks, with `--save-masks` |

Eleven of those columns come straight from VascX rather than from the vessel
graph, and are worth keeping even if you only want the morphometry:

| column | meaning |
|---|---|
| `vascx_laterality` | `L` / `R`, from fovea vs disc position |
| `vascx_fovea_x`, `vascx_fovea_y` | fovea location |
| `vascx_w_img`, `vascx_h_img` | image size those coordinates refer to |
| `vascx_disc_cx`, `vascx_disc_cy` | disc centroid |
| `vascx_sep_x_frac` | fovea-disc separation; small values mean laterality is doubtful |
| `vascx_quality_q1`, `q2`, `q3` | image quality scores - **filter on these before analysing** |

Skip the two extra GPU passes with `--no-vascx-extras`; you lose these columns
and, unless the worklist supplies orientation, every sector feature.

### Save the masks

Each image's four masks pack into one ~8.5 KB PNG. That is roughly 2.8 GB per
330k images against about 26 GPU-hours to regenerate them, and registration or
any repeat-measurement work needs the masks rather than the feature table.

> **Put them on a filesystem with small allocation units.** Masks are many tiny
> files, so the cluster size decides what they actually cost. On an **exFAT**
> volume with 1 MB clusters each 8.5 KB mask occupies a full megabyte - measured
> here, 80,765 masks holding **0.9 GB of data took 79 GB of disk**. NTFS
> defaults to 4 KB and is fine. Check with `fsutil fsinfo ntfsinfo G:` on
> Windows, and send the output elsewhere if needed:
>
> ```bash
> FUNDUS_OUT=/path/on/ntfs python extract.py --save-masks
> ```

---

## Retinal age (optional)

A separate model predicts apparent age and sex from the photograph. It is not
part of the vessel pipeline and needs its own weights at `weights/rag/`.

```bash
python extract_rag.py              # -> output/rag_predictions.csv
python extract_rag.py --merge      # also -> output/features_with_rag.csv
```

`--merge` joins the predictions onto `graph_features.csv` by `image_id` and
writes a **separate** file; `graph_features.csv` is left exactly as produced,
because that is what `verify_sample.py` checks.

| column | needs | notes |
|---|---|---|
| `predicted_age` | nothing | raw model output |
| `predicted_sex` | nothing | raw model output |
| `corrected_age` | `--ages` | bias correction fitted on one validation split |
| `rag_corrected` | `--ages` | `corrected_age` minus chronological age |

```bash
python extract_rag.py --ages /data/ages.csv --merge
```

`--ages` takes a CSV with `image_id` and an age column. Without it the two
corrected columns are simply absent - the script says so rather than inventing
them.

The model is described in
[GeroScience 10.1007/s11357-026-02538-8](https://link.springer.com/article/10.1007/s11357-026-02538-8).
**Cite it wherever `predicted_age` is reported.**

> **Check for a device step before reading any trend.** Predicted age is
> sensitive to the camera. In the cohort this model was built on, a camera
> change made predicted age jump several years overnight; read as a
> within-person trend that looked like accelerated ageing, and it was not.
> Before comparing retinal age across time, plot the cohort median against
> imaging date and look for a step.

---

## What a "vessel feature" actually is

Every number comes from the same short chain. `python make_figures.py`
regenerates all of these from a bundled sample, using the same code path the
extraction uses.

### 1. Segmentation - VascX turns the photograph into masks

![masks](docs/01_masks.png)

Vessels, artery/vein, and the optic disc. **Everything downstream is measured
off these**, so a segmentation error becomes a feature error.

Segmentation is **not** this repository's work. It is
[**VascX**](https://github.com/Eyened/rtnls_vascx_models) from the Eyened group
(Erasmus MC) - an ensemble of models trained for retinal vascular analysis on
colour fundus images. Five models ship with it and this pipeline runs all five:

| model | output | used for |
|---|---|---|
| `vessels` | binary vessel mask | every length, density and graph feature |
| `av` | artery / vein / uncertain | the `art_*` and `ven_*` families, AVR |
| `disc` | optic disc mask | the DD coordinate system, `disc_*` features |
| `fovea` | fovea location | which side is temporal |
| `quality` | image quality score | `vascx_quality_q1..q3` |

What this repository adds is everything after the masks: skeleton, graph,
zones, calibre and the feature table. The weights are run locally as
torchscript exports; preprocessing is vendored from
[`retinalysis-fundusprep`](https://github.com/Eyened/retinalysis-fundusprep).

> If you use this pipeline, **cite VascX as well** - see
> [License and third-party components](#license-and-third-party-components).

### 2. Skeleton - area becomes a one-pixel line

![skeleton](docs/02_skeleton.png)

A mask has area; length, branching and tortuosity need a line. Short spurs
thrown up by a ragged mask edge are pruned (red).

### 3. Graph - the line becomes nodes and edges

![graph](docs/03_graph.png)

Nodes are branch points and endpoints; edges are the vessel segments between
them. This is where `n_bifurcations`, `branch_order`, `tortuosity` and the rest
of the topology come from - **421 of the 1,087 columns need this graph**.

### 4. Calibre - thickness comes from the distance transform

![calibre](docs/04_calibre.png)

The graph has no thickness. For every pixel inside a vessel, the distance to
the nearest background pixel is its radius, so on the skeleton
`diameter = 2 x dt - 1`.

### 5. Zones - a disc-centred coordinate system

![zones](docs/05_zones.png)

Whole-frame counts are not comparable between images: they depend on how much
retina the photograph happened to contain. Measuring inside rings at a fixed
**disc-diameter** radius fixes that, because the optic disc is ~1.8 mm across
in nearly everyone. Each ring reports its `coverage` - the outermost ones run
off the edge of the picture, as the figure shows.

### 6. CRAE / CRVE / AVR - the clinical summary

![crae](docs/06_crae_crve.png)

Inside Zone B, the six widest arteries and six widest veins are combined by the
Knudtson formula into one arteriolar and one venular equivalent diameter.
`AVR = CRAE / CRVE` is their ratio, and being a ratio it is unaffected by
camera scale.

### 7. Sholl - density against distance

![sholl](docs/07_sholl.png)

Concentric circles, counting how many times each one crosses a vessel.

---

## What the columns are

`graph_features.csv` has **1,098 columns**:

| block | columns | from |
|---|---|---|
| measured features | **1,087** | `retinal_graph`, the product below |
| `vascx_*` | **11** | VascX quality, fovea, laterality - carried through, not measured |

With `extract_rag.py --merge` two more (`predicted_age`, `predicted_sex`) land
in `features_with_rag.csv`, for 1,100.

The 1,087 are not 1,087 independent measurements. The count is a product:

```
   3 vessel networks    x   spatial extents              x   measurements
   ves (all vessels)        whole frame                      143 distinct
   art (arteries)           zone_a  zone_b  zone_c
   ven (veins)              outer_c  periph_25_35  periph_35_45

                            + 4 anatomical sectors, inside zone_b and outer_c

   975 vessel columns + 60 optic disc + 8 CRAE/CRVE/AVR + 4 A/V crossing + meta
   = 1,087
```

The same quantity - vessel density, say - is computed once for the whole frame,
once per zone, and once per zone-sector cell. Treat the table as correlated,
not as 1,087 independent tests.

### Disc-centred zones

Zones are annuli at fixed **disc-diameter (DD)** radii from the disc centre.
The optic disc is about 1.8 mm across in nearly everyone, so 1.5 DD lands on
the same anatomy whatever the camera magnification. The disc margin sits at
0.5 DD, so subtract 0.5 to get the disc-margin convention used in the AutoMorph
paper.

| zone | from disc centre | notes |
|---|---|---|
| `zone_a` | 0.5-1.0 DD | proximal reference |
| `zone_b` | 1.0-1.5 DD | AutoMorph Zone B; where CRAE/CRVE are measured |
| `zone_c` | 1.0-2.5 DD | AutoMorph Zone C - **contains `zone_b`** |
| `outer_c` | 1.5-2.5 DD | the disjoint remainder of `zone_c` |
| `periph_25_35` | 2.5-3.5 DD | exploratory |
| `periph_35_45` | 3.5-4.5 DD | exploratory, often outside the field of view |

`zone_c` is kept nested so the numbers stay comparable with the literature.
**Calling `outer_c` "Zone C" is wrong.**

Every zone is reported with a `coverage` value: the fraction of the annulus
actually inside the photograph, measured against the analytic area of the full
ring. A half-visible ring is **not a scaled-down copy of the whole one** - it
is one particular sector of retina. Filter on coverage before comparing across
images.

### Pixels vs disc diameters

Lengths and calibres are reported both ways: `*_px` and `*_dd`.

**Prefer `_dd` whenever images may come from different cameras.** Pixel
measurements scale with ocular magnification and with the camera; dividing by
the disc diameter removes that. In this cohort a camera change moved raw pixel
calibre by 0.65 SD while the DD-normalised version moved 0.12 SD.

`AVR_Knudtson` is a ratio and therefore already scale-free - the most robust
single number in the table.

### Negative control

`disc_area_px` is useful as a control. Optic disc area has no reason to vary
with a vascular exposure, so **if it moves in your analysis, suspect
magnification rather than biology** - most likely refractive error, since
myopia both lengthens the eye and correlates with many exposures. It caught a
false finding during development that would otherwise have been reported.

---

## Repository layout

```
.
├── check_env.py            verify the machine before a long run
├── config.py               every path, resolved from the environment
├── extract.py              VascX segmentation + feature measurement  <- main entry
├── extract_rag.py          retinal age / sex (optional, separate model)
├── preprocess.py           square the fundus before extraction
├── fundusprep/             the cohort's preprocessing code
├── make_figures.py         regenerate docs/ from a sample
├── verify_sample.py        did the run reproduce the reference?
├── common.py               worklist loading, chunking, atomic writes
├── retinal_graph/          measurement: skeleton, graph, zones, calibre, disc
├── vascx/                  segmentation wrapper + vendored preprocessing
├── weights/                ALL model weights, NOT in git - one download
│   ├── vascx/              vessels, av, disc, fovea, quality  (2.2 GB)
│   └── rag/                retinal-age model                  (333 MB)
├── samples/                5 images + reference values (worklist is generated)
├── docs/                   the figures above
└── output/                 created by a run, not in git
```

## License and third-party components

This pipeline is a thin layer over other people's models. **Most of what makes
it work is not ours**, and the parts that are not carry their own terms.

| component | what it is | source | licence |
|---|---|---|---|
| **VascX models** | the five segmentation / quality models | [Eyened/rtnls_vascx_models](https://github.com/Eyened/rtnls_vascx_models), weights on [HuggingFace](https://huggingface.co/Eyened/vascx) | **none stated upstream** |
| `rtnls_fundusprep` | preprocessing, vendored into `vascx/rtnls_fundusprep/` | [Eyened/retinalysis-fundusprep](https://github.com/Eyened/retinalysis-fundusprep) | **AGPL-3.0** (`LICENSE` included) |
| retinal-age model | `rag/`, optional | [GeroScience 10.1007/s11357-026-02538-8](https://link.springer.com/article/10.1007/s11357-026-02538-8) - this repository's own work | see `LICENSE` |
| `retinal_graph/`, the scripts | skeleton, graph, zones, calibre - this repository | - | see `LICENSE` |

VascX is described in a preprint: <https://arxiv.org/abs/2602.08580>.
Note that upstream has moved to
[retinalysis-vascx](https://github.com/eyened/retinalysis-vascx) and says it
will stop supporting `rtnls_vascx_models`; the torchscript weights bundled here
came from the older release.


## Limitations

- **Sector features need orientation** and are `NaN` without it, by design.
- **Absolute pixel lengths are comparable only within one camera** and one
  preprocessing pipeline. Use `_dd` across devices.
- **No refraction or axial-length adjustment.** Ocular magnification moves
  every pixel measurement at once; `_dd` normalisation is the only defence
  here.
