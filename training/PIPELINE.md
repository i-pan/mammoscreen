# Corrected end to end training recipe

This is the active route for a new experiment. Historical notebooks/configs remain source evidence; do not combine their default split paths with this recipe. A new run can have different seeds, split membership and metrics. No full data preparation or training was executed. A bounded synthetic CPU lifecycle check is recorded in the [verification report](../docs/VERIFICATION.md).

## 1 Obtain inputs and install

Use the [prepared CBIS-DDSM dataset, version 1](https://www.kaggle.com/datasets/vaillant/cbis-ddsm/versions/1) and the [RSNA screening competition data](https://www.kaggle.com/competitions/rsna-breast-cancer-detection/data). Competition access may require your Kaggle account and acceptance of its terms. Arrange the prepared DDSM archive's inner `png/png/` directory as `/data/ddsm/png/` and its `cleaned_cbis_ddsm.csv` at `/data/ddsm/`.

From the repository root, in your chosen training environment with an appropriate PyTorch installation:

```bash
pip install -e 'training[processing,evaluation]'
```

This installs `mammoscreen-splits` and `mammoscreen-recipe`; equivalent source commands are shown below. CUDA hardware is required for the training recipe. JPEG2000 DICOM decoding needs Pillow built with OpenJPEG support (as in the tested wheels); if your build lacks it, install `pylibjpeg` and `pylibjpeg-openjpeg`. The active launcher adapts the preserved runner to save configuration and metrics locally in each fold's `logs/events.jsonl`; no tracking account is needed. `--project` is an optional local experiment label. Original public timm ImageNet initialization is used for DDSM pretraining. The published HF crop model can be used as a supplied preprocessing model, avoiding any circular dependency on retraining it.

## 2 Prepare DDSM and RSNA images

Use the [preparation guide](data_preparation/README.md) for the stages and the [corrected utilities](data_preparation/corrected/README.md) for mask geometry. Keep the archived files unchanged by making working copies when adapting their ROOT variable or notebook paths.

- RSNA: copy `data_preparation/rsna-breast-cancer-detection/convert_dicom_to_png.py`, set its ROOT to your RSNA download, and run it to create `train_png/`. Then copy/adapt `crop_images.py` and run it with the public crop model to create `train_cropped_png/`.
- DDSM: run the corrected mask command below, then use the main image/mask loop in `data_preparation/CBIS-DDSM/crop_images_and_masks.ipynb` with ROOT set to `/data/ddsm/`. It uses the same crop for each paired image/mask and creates `cropped_png/` and `cropped_masks/`. Then copy/adapt `CBIS-DDSM/create_kfold_cv.py` with that ROOT to create `/data/ddsm/train_kfold_cv.csv`.

```bash
python training/data_preparation/corrected/prepare.py ddsm-masks \
  --images /data/ddsm/png \
  --annotations /data/ddsm/cleaned_cbis_ddsm.csv \
  --output /data/ddsm/masks
```

The coordinate convention defaults to the historical notebook's yx mapping; confirm a few image/mask pairs before bulk use. Blank images, crop bounds and missing masks need inspection in adapted historical image-cropping loops. Original notebook rough edges remain clearly labeled; the two corrected mask/crop-label commands are separate implementations.

Training the crop model itself is optional for this screening recipe because the public crop model is an obtainable input. If you choose to retrain it, generate pixel coordinates with the corrected crop-label utility, merge patient/image identifiers and create grouped folds before invoking the historical crop config. That optional branch does not replace the screening split command below.

## 3 Generate valid patient splits

```bash
python training/data_preparation/corrected/splits.py \
  --annotations /data/rsna/train.csv \
  --output /data/rsna/new-splits --seed 88 --folds 9 --repeats 3
```

This reads the raw RSNA labels, encodes cancer/auxiliary targets and breast IDs, and writes:

- `train_kfold_v01.csv` with `holdout_test_v01.csv`;
- equivalent pairs for sets 02 and 03;
- `splits.json`, with filenames, hashes, seed and nominal proportions.

A locally owned RNG is never reset by the historical fold helper. Holdouts are stratified by patient-level cancer status; every patient's views stay together. Repeats may overlap, as independent holdouts normally do, but identical holdout memberships are rejected. Patient overlap between train/validation/holdout and validation folds lacking either cancer class are rejected. Sampling weights use training rows only. Changing CSV contents after generation invalidates their manifest hashes.

Nine validation folds target approximately 80% training / 10% validation / 10% holdout patients; use ten if you intend the historical generator's approximate 81%/9%/10% proportions. Rounding and images per patient affect realized fractions. These are valid new partitions, not recovered historical memberships. Too few patients in a cancer stratum produces a clear error. A new output directory is required; partial writes from filesystem failures are not automatically deleted or reused.

## 4 Bind the full recipe before launching jobs

```bash
python training/mammoscreen_recipe.py plan \
  --splits /data/rsna/new-splits/splits.json \
  --images /data/rsna/train_cropped_png \
  --ddsm-images /data/ddsm/cropped_png \
  --ddsm-masks /data/ddsm/cropped_masks \
  --ddsm-annotations /data/ddsm/train_kfold_cv.csv \
  --runs /runs/mammoscreen-new \
  --project mammoscreen-local \
  --output /runs/mammoscreen-recipe.json

python training/mammoscreen_recipe.py check --recipe /runs/mammoscreen-recipe.json
```

`plan` and `check` read CSVs/configuration, not images or checkpoint payloads, and do not launch training. The JSON is the authoritative routing record. All three models in set NN train exclusively from `train_kfold_vNN.csv`; evaluation reads `holdout_test_vNN.csv`. The full-dataset fivefold table and nonexistent historical CSV names cannot silently become screening inputs.

| Member | Screening base recipe | DDSM base recipe | Screening validation fold |
| --- | --- | --- | --- |
| 2048 × 1024 | `cfg_ddsm_pretrained_aux_losses` | `cfg_ddsm_seg_cls_v02` | 0 |
| 1920 × 1280 | `cfg_ddsm_pretrained_aux_losses_v02` | `cfg_ddsm_seg_cls_v03` | 1 |
| 1536 × 1536 | `cfg_ddsm_pretrained_aux_losses_v04` | `cfg_ddsm_seg_cls_v05` | 2 |

The active `cfg_recipe_ddsm`/`cfg_recipe_screen` wrappers clone these base values, replace data/output/tracking/initialization paths from the plan, enable screening EMA, and explicitly select reflection padding for the square member. They intentionally fail without the recipe environment supplied by the launcher. They do not mutate the archived config objects. Exact historical CLI overrides are not asserted.

## 5 Train the initializations, then screening members

These commands are the expensive steps and should be run only when you intend to train:

```bash
python training/mammoscreen_recipe.py run --recipe /runs/mammoscreen-recipe.json \
  --stage pretraining --devices 2

python training/mammoscreen_recipe.py run --recipe /runs/mammoscreen-recipe.json \
  --stage screening --set 1 --devices 2
```

The first command trains three DDSM initializations, each at fold 0. Their generated `last.ckpt` files feed the corresponding screening encoder. The launcher preserves those initialization paths instead of rewriting them to the screening fold. Set 1 trains three screening members; repeat the screening command with `--set 2` and `--set 3` for all three evaluation sets, or omit `--set` to train all nine members. `--member 2048`, `1920` or `1536` selects a single resolution when needed.

Output paths are generated, not copied from historical run IDs:

```text
/runs/mammoscreen-new/mammo.cfg_recipe_ddsm/ddsm-2048/fold0/checkpoints/last.ckpt
/runs/mammoscreen-new/mammo.cfg_recipe_screen/set01-2048/fold0/checkpoints/ema_weights.pt
/runs/mammoscreen-new/mammo.cfg_recipe_screen/set01-1920/fold1/checkpoints/ema_weights.pt
/runs/mammoscreen-new/mammo.cfg_recipe_screen/set01-1536/fold2/checkpoints/ema_weights.pt
```

The exact paths for all jobs are in the JSON. Existing selected run directories cause an error; no run is overwritten. All selected jobs' predictable path prerequisites are checked before any job starts. `--member`/`--set` let you select not-yet-started jobs after partial completion. A failed job's existing directory is not automatically resumed or deleted; inspect it and deliberately choose a fresh run plan/root or use the original runner's supported workflow. This launcher does not invent checkpoint-resume behavior.

## 6 Evaluate the matching fresh EMA ensemble

```bash
python training/mammoscreen_recipe.py evaluate \
  --recipe /runs/mammoscreen-recipe.json --set 1 \
  --device cuda:0 --output /runs/evaluation-set01
```

All three generated 11-output EMA checkpoints must exist. Evaluation uses matching image transforms, sigmoid cancer output 0, the mean probability across models, then the mean across images for each breast. It checks consistent breast targets and computes binary cancer AUC with tie handling. The output directory contains `image_predictions.csv` and `metrics.json`, with cohort counts, recipe identity and checkpoint SHA-256 values. This endpoint evaluates cancer discrimination; density-head metrics and HF export are outside this recipe.

Evaluate sets 2/3 with their own `--set` and fresh output directories. No historical checkpoint, original split CSV, or HF 11-to-5-head conversion is needed for this training-to-local-evaluation path. Original archived evaluation results remain unchanged as historical evidence.

## Verification limits

Synthetic checks cover local RNG state, distinct patient partitions, leak rejection, actual active config routing, generated checkpoint paths, launcher arguments, CSV identity and probability aggregation/AUC. Full dataset ETL, training, remote tracking and GPU evaluation were not run during implementation. Missing local datasets can be downloaded again; unavailable current training dependencies must be installed in the chosen environment before actual launches.

For runtime evidence, observed dependency versions, the opt-in tiny lifecycle check and remaining GPU/real-data limits, see [bounded pipeline verification](../docs/VERIFICATION.md).
