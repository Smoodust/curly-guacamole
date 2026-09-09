# AIIJC segmentation experiments

Reusable code lives in `src/`, experiment settings in `configs/`, and results in
`runs/<run_name>/`. Use the `challenges` conda environment.

Start with the [experiment map](docs/experiment_map.md) for the current config-to-run
catalog, saved results, comparison groups, and differences between recipes and run
snapshots (local audit dated 2026-09-09).

## Training

Open `notebooks/baseline_pipeline.ipynb` with the `challenges` kernel. For a new
experiment, create a YAML inheriting from an existing config, set `run_name`, and
select the new YAML in the notebook. The same pipeline can run from a Python script:

```python
from src.config import load_experiment_config
from src.training.engine import run_experiment

run = run_experiment(load_experiment_config("configs/baseline.yaml"))
print(run.summary)
```

For example, `configs/my_experiment.yaml` can contain only the overrides:

```yaml
extends: baseline_mixed_original.yaml
paths:
  run_name: my_experiment
dataset:
  image_size: 672
train:
  batch_size: 2
  accum_steps: 8
```

`extends` names one parent YAML, relative to the file containing it (absolute
paths also work). Parents can inherit from other configs. Nested mappings merge
recursively; child values win, and lists replace the entire inherited list.
Omitted fields are inherited; an empty mapping does not clear inherited keys.
Cycles, missing parents, and invalid settings raise errors during loading.
Changing a parent affects its descendants on their next load. Run snapshots
contain the fully resolved settings, without `extends`, and remain self-contained.

Inspect `summary.json`, `metrics.csv`, and `notes.md` in the run directory.
`notebooks/16_forensic_maps_before_resize.ipynb` is a historical research archive;
its old API is not the supported training entry point.

## Mixed-frame experiment

Open `notebooks/baseline_mixed_original.ipynb` (restart the kernel after updating
the package). It uses `configs/baseline_mixed_original.yaml` and writes to
`runs/pvt_v2_b2_mixed_original`, keeping the completed baseline separate.
The architecture, fold, seed and training budget stay at 8 × 24,000 = 192,000
sample presentations.

- `full_frame_probability: 0.5`: first six epochs mix full frames and square crops.
- `foreground_crop_probability: 0.5`: half of crop attempts select a GT pixel
  and an 8-aligned window containing it; empty masks use ordinary random crops.
- `final_full_frame_epochs: 2`: last two epochs use full frames. Full-frame
  examples keep flips but do not rotate by 90 degrees.
- `full_frame: true` still overrides mixing and always selects a full frame.
- `eval.resolution: original`: restore sigmoid probabilities with bilinear
  interpolation before histogram threshold tuning against the original binary GT.
  Different mask sizes are retained in a list by `ValidationCollator`.
  Malformed GT with a different size from its image raises an error.

RNG streams advance on repeated samples and reset reproducibly per epoch and
worker; the epoch is shared with persistent workers. Sampling is seeded per
epoch as well. Reproducibility assumes the same worker count and batch setup;
the existing checkpoint format does not guarantee bit-identical training resume
for model-side randomness such as dropout.
`train/negative_fraction` records the observed fraction after cropping and resize,
which can differ from the sampler's configured negative fraction.

Legacy YAML files default to crop-only augmentation and resized validation.
The RNG fix applies to them too. Do not compare the completed baseline's resized
AIC directly with the new original-resolution AIC: re-evaluate its saved best
checkpoint on the same validation rows at original resolution first. Subsequent
fold checks should use a new run name and the same number of sample presentations.
Changing the new augmentation options or validation resolution when resuming an
existing run is rejected before its snapshot can be overwritten.

## EfficientViT experiment

`notebooks/efficientvit_b2_mixed_original.ipynb` uses
`configs/efficientvit_b2_mixed_original.yaml`. The existing `Segmenter` accepts
the MIT EfficientViT encoder through timm; no encoder adapter is required.
The backbone is `efficientvit_b2.r288_in1k` (ImageNet classification weights),
from [MIT EfficientViT](https://github.com/mit-han-lab/efficientvit), corresponding
to [the multi-scale linear attention paper](https://arxiv.org/abs/2205.14756).
This is the timm `efficientvit_b*` family, distinct from `efficientvit_m*`.

Its features have strides `[4, 8, 16, 32]` and channels `[48, 96, 192, 384]`.
The same DCT branch and gated residual fusion attach at strides 8/16/32;
decoder channels, auxiliary supervision and classification head are retained.
At **1024 × 1024**, the complete model has **16,310,395 parameters** and costs
**97.00815744 GFLOPS** per image (batch 1, eval, `FlopCounterMode`, confirmed by
a real CPU forward with random weights). The PVT baseline at 640 costs 97.1621.
The CPU DCT preprocessing, mask restoration and metric calculation are outside
this forward count. H100 latency and full-resolution training GPU memory are
not yet measured; the active GPU training run was not interrupted.

The experiment keeps the same fold, seed, augmentation schedule, original-size
validation and 192,000 presentations. Batch 2 with accumulation 8 preserves an
effective batch of 16; BatchNorm still observes the physical batch of 2.
Results go to `runs/efficientvit_b2_1024_mixed_original`.
First training use downloads the configured pretrained weights through timm;
the compatibility checks use `pretrained=False` and require no download.
If GPU memory is insufficient, use batch 1 / accumulation 16 in a copied config
with a new run name. Start this notebook after the current GPU experiment ends.

For reference, measured full-model alternatives with the same decoder:

| Encoder | Image size | GFLOPS |
| --- | ---: | ---: |
| EfficientViT-B1 | 1024 | 49.982 |
| EfficientViT-B2 | 896 | 74.272 |
| EfficientViT-B2 | 1024 | 97.008 |
| EfficientViT-B3 | 768 | 112.100 (over budget) |

## EfficientViT with aspect-preserving resize

`notebooks/efficientvit_b2_letterbox.ipynb` selects
`configs/efficientvit_b2_letterbox.yaml` and writes to
`runs/efficientvit_b2_1024_letterbox`. All training settings match the EfficientViT
mixed/original experiment; only `dataset.resize_mode: letterbox` changes geometry
and enables padding-aware loss, classification pooling and restoration.

The longer side is resized to 1024 (including upscaling smaller images), the
shorter side is rounded to the nearest pixel, and padding is added on the bottom
and right. For example, 640×480 becomes 1024×768 plus 256 bottom rows. The top-left
origin remains aligned with the DCT grid. Square random training crops remain
square; full-frame examples retain their aspect ratio.

DCT maps are extracted before resizing. Their block centers are resampled with
the same image scale, and partial boundary blocks are weighted by valid coverage.
RGB padding is reset to zero after normalization and photometric augmentation;
GT and forensic padding are zero. `valid_mask` excludes padding from BCE/Dice
(including auxiliary loss) and weights classification pooling at encoder resolution.
Convolutions, attention and normalization still operate on the full square.

Validation and submission share the same inverse: crop to `content_size`, resize
probabilities to the original dimensions, then binarize. Padding never enters the
metric area denominator, including the optional resized validation mode. The resize
mode is saved in the snapshot and automatically restored by submission inference.
Legacy configs/checkpoints default to `stretch`; changing geometry on resume is rejected.

The model still processes a dense 1024×1024 tensor, so padding does not save FLOPs.
The counted forward includes the masked classification path; preprocessing,
restoration, and validation histograms remain outside that count.

## Loss ablation

`notebooks/loss_ablation.ipynb` runs `configs/loss_l*.yaml` as one series and
tabulates the arms. Without Jupyter, the same series runs from the shell:

```bash
python -m src.training configs/loss_l*.yaml --data-path /workspace/data --csv runs/loss_ablation.csv
```

The objective is now selected from the config through a registry in
`src/losses` (`src/losses/README.md` documents the arms, the soft false-positive
surrogate and how to add another objective). Configs without a `loss:` section
keep the previous `BCE + Dice + 0.3 cls + aux` formula, verified against the old
implementation to the last digit, so completed runs stay comparable.

Two hypotheses are under test. First, a mask covering 0.5% of the frame produces
0.5% of the BCE terms and its gradient is lost in the background: arms L1
(per-image positive weighting), L2 (focal) and L3 (Tversky with `beta > alpha`)
each attack it differently. Second, the objective disagrees with the metric:
AIC scores Dice on positive frames only and charges a negative frame nothing
until its predicted mask reaches 1% of the frame, whereas Dice on a negative
frame is already near its maximum penalty for any blob at all. Arm L4 replaces
that term with a soft false-positive rate at the same 1% threshold; arm L5 also
replaces the sum with the metric's harmonic mean.

| arm | `loss.name` | change against the control |
| --- | --- | --- |
| L0 | `bce_dice` | control, formula unchanged |
| L1 | `balanced_bce_dice` | positive pixels reweighted per image, up to 20x |
| L2 | `focal_dice` | focal reweighting, `gamma=2`, term scale preserved |
| L3 | `focal_tversky` | Dice becomes Tversky (`alpha=0.3`, `beta=0.7`, `gamma=0.75`) |
| L4 | `aic_surrogate` | Dice over positives only; negatives get a soft FPR at 1% |
| L5 | `aic_harmonic` | one minus the batch AIC, from soft Dice and soft FPR |

Every arm shares the encoder, fold, seed, augmentation schedule and
original-resolution validation of `baseline_mixed_original`; only the `loss`
section, the run name and `epoch_size` differ. The budget is halved to
`epoch_size: 12000`, so **these scores are not comparable with the completed
full-budget runs** — L0 is the control for exactly that reason. Promote the
winning objective into a full-budget config under a new run name before reading
anything into the absolute number.

Arm L5 estimates both AIC components inside one micro-batch. At
`batch_size: 4` with a 25% negative fraction some micro-batches hold a single
negative frame, so its estimate is the noisiest of the six; a larger physical
batch would suit it better, but changing it here would confound the comparison.
The saved snapshot records `loss_name`, and resuming a run with a different
objective is rejected rather than silently continuing a different curve.

## Submission

From the project root in PowerShell:

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m src.inference runs/<run_name> submissions/<run_name>
```

This reads `ckpt/best.pt` (EMA weights when present), the saved configuration, and
the tuned thresholds in `summary.json`. It writes `submission.csv` and the PNG
masks under `predictions/`, retaining template paths and original image sizes.
For the competition archive, zip those two entries at the archive root.

Console output reports configuration/data/checkpoint loading, the selected device
and thresholds, prediction and PNG saving progress by batch, and the final output
directory. Progress is printed after the first batch, every 30 seconds, and after
the last batch, with elapsed time and an estimated remaining time once available.

Use `--data-path` to relocate the dataset, `--template-path` for an explicit
template, or `--device cpu` to override the saved device. To override the operating
point, pass all three flags: `--mask-threshold`, `--cls-threshold`, `--min-area`.
Histogram tuning reports the effective bin boundary used during validation.
Legacy validation scores use resized targets; final predictions are resized to
original dimensions before binarization. Set `eval.resolution: original` for the
same restoration and thresholding order during model selection.

## Experiment history

Open `notebooks/experiments.ipynb` to view the experiment tree and results table.
Edit the YAML header in `runs/<run_name>/notes.md` by hand:

```yaml
parent: pvt_v2_b2_mixed_original
baseline: null
change: "Replaced the decoder with SegFormer"
leaderboard_score: null
```

`parent` is the experiment this run was derived from, not an instruction to load
weights. A missing, empty or null `baseline` defaults to `parent`; set another
single run name to override it. References use run directory names. Root runs
leave `parent` empty. Existing baseline-only cards keep their meaning.
Replace `leaderboard_score: null` with a number from 0 to 1 after scoring.
No private score or submission metadata is required.

Cards are tracked by Git; training artifacts remain ignored. Existing cards are
never overwritten by training. New runs receive the expanded template automatically.
Local metrics still come from `summary.json`. The registry reports missing links,
cycles and malformed records without changing run files. Recreate the registry
or rerun the notebook after edits. Runs without cards remain visible.

```python
from src.training.experiments import ExperimentRegistry

registry = ExperimentRegistry("runs")
print(registry.tree())
print(registry.table())
print(registry.issues)
```

Compare validation scores only on matching validation rows and evaluation
protocols; in particular, resized and original-resolution AIC are not directly
comparable. The overview does not calculate automatic score differences.

## Regression checks

```powershell
& 'D:/Apps/anaconda3/envs/challenges/python.exe' -m pytest
```

The suite covers configuration, factories, training state, validation, dataset
modes and paths, DCT shapes, synchronized augmentation, model forward, and
template-aligned submission. Tests use synthetic inputs and mocked encoders, with
no training dataset or pretrained weight download required.
## Storage paths

Experiment YAML files only specify `paths.run_name`. Copy `.env.example` to
`.env` in the project root and set `AIIJC_DATA_PATH` and `AIIJC_RUNS_PATH` once
per machine. `.env` is ignored by Git; `.env.example` is the shared template.
The file is read each time an experiment config is loaded, including in notebooks.
Environment variables override `.env`; missing or empty values default to
the project's `data/` and `runs/` directories.
Relative storage paths resolve from the project root, regardless of the
working directory or YAML location.

For example, a server's `.env`:

```dotenv
AIIJC_DATA_PATH=/mnt/datasets/aiijc
AIIJC_RUNS_PATH=/mnt/experiments/aiijc
```

The same YAML and notebook work on both machines. Runtime snapshots retain
the resolved paths for reference; loading an old experiment YAML ignores its
stored `data_path` and `runs_path` in favor of the current machine settings.


## Positive-only Dice experiment and loss diagnostics

`configs/baseline_positive_dice_mixed_original.yaml` inherits the mixed/original
U-Net baseline and uses a new run name. Load it with `load_experiment_config`
and pass the result to `run_experiment` in the usual notebook pipeline.

```yaml
loss:
  dice_scope: positive
  dice_weight: 0.75
```

BCE still trains all masks. Dice averages only images with nonempty targets
inside the valid region, after augmentation/resize. A batch with no positives
contributes zero Dice and still receives BCE gradients. The same policy applies
to decoder and DCT auxiliary losses. Reductions use float32 under AMP.
The initial weight 0.75 roughly preserves the previous positive contribution
with 25% negatives; it is an ablation setting, not a measured optimum.

Without a `loss` section, configs retain `dice_scope: all`, `dice_weight: 1.0`.
The scalar `compute_loss` API retains that legacy policy. New training code uses
`SegmentationLoss`, which returns the objective and its components. Resuming a
run with changed Dice settings or auxiliary weights is rejected before its
snapshot is overwritten; use a new run name to change the objective.

Fractional validity weights on downsampled DCT/letterbox boundaries now enter
the Dice intersection once, rather than being squared. This corrects the old
boundary penalty even in all-image mode. Historical runs combining DCT aux and
letterbox should use a new run name after this correction. Binary validity masks
are unaffected by this correction.

New epoch fields in JSONL, CSV and TensorBoard:

- `train/loss`: total objective, averaged by batch image count.
- `train/loss_bce`, `train/loss_dice`, `train/loss_cls`: weighted main BCE,
  selected Dice and classification contributions (classification weight 0.3).
- `train/loss_aux_bce`, `train/loss_aux_dice`, `train/loss_dct_aux_bce`,
  `train/loss_dct_aux_dice`: weighted auxiliary contributions when enabled.
  These contributions plus the three main contributions sum to `train/loss`.
- `train/loss_dice_pos`, `train/loss_dice_neg`: unweighted main soft-Dice loss
  diagnostics, averaged by the actual number of images in each group over the
  epoch. Absent groups are omitted. Negative Dice is diagnostic only in the
  positive-only mode; it does not enter the objective.
- `val/loss_bce`, `val/loss_dice`, `val/loss_cls`, `val/loss_total`,
  `val/loss_dice_pos`, `val/loss_dice_neg`: corresponding EMA/eval diagnostics,
  with no auxiliary losses. `val/loss_main` is BCE plus weighted Dice.
- `val/aic_fixed`, `val/dice_fixed`, `val/fpr_fixed`: mask threshold 0.5,
  classifier threshold 0 and minimum area 0 (no classifier gating). Existing
  tuned metric fields and checkpoint selection remain based on tuned AIC.

Validation losses use resized targets on the network grid, excluding padding.
AIC uses `eval.resolution`, including original resolution when configured; the
loss and AIC therefore need not agree on tiny regions lost during resize.
Training contributions reflect the actual batch objective: the mean of batch
positive-only Dice values is not necessarily the global positive-image mean.
Use the separately counted `loss_dice_pos` diagnostic for the latter.
