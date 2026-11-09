# Historical mammography data preparation

Recovered from [i-pan/datasets-src](https://github.com/i-pan/datasets-src/tree/74ad10027bed59759af119f8832f6fa6f45a2aec). All included source files match that Git revision before documented formatting/output removal. No scripts were executed during recovery. Data and CSV contents are not bundled. Original paths and some exploratory rough edges remain; review each script before running because scripts execute their dataset writes at top level.

## Corrected utilities

For new preparation work, [corrected/prepare.py](corrected/README.md) supplies tested crop-label and multi-lesion-mask commands. They retain original-image pixel coordinates and preserve disjoint lesion categories. They are new reference implementations, separate from the historical source below; they do not regenerate original split memberships or prove the original labels.

## RSNA screening workflow

In `rsna-breast-cancer-detection/`:

1. `convert_dicom_to_png.py` (also a notebook) reads `train_images`, applies VOI LUT, inverts MONOCHROME1, min/max converts to uint8, and writes `train_png/<patient>/<image>.png`.
2. `get_crop_coords.ipynb` generates foreground-component `xywh` labels, associates patient/image IDs, and writes `crop_coords_using_connected_components.csv` and `train_coords_kfold.csv`. The output names align with `mammo.cfg_crop`, but this scratchpad is **not a valid Run All label generator as preserved**: its loop stores pixel-valued boxes and later cells multiply them by image dimensions as if normalized. Resolve that inconsistent scaling before using the generated CSV. Its foreground heuristic also assumes usable nonblank images. A warning is included in the notebook.
3. `crop_images.py` applies the published `ianpan/mammo-crop` model and writes `train_cropped_png`. The historical script uses an unpinned HF model and remote code; it is preserved as evidence. The optional integrated pipeline elsewhere in this repository provides pinned model loading.
4. `create_kfold_cv_and_holdout_x3.py` derives filename/breast grouping, encodes density A/B/C/D as 0/1/2/3, handles missing auxiliary labels, computes sampling weights, and produces `holdout_test_v01.csv`–`v03.csv` plus `train_kfold_v01.csv`–`v03.csv`, with holdout selection at patient level and grouped inner/outer folds. This matches the historical evaluation notebook's holdout filenames. With one of ten outer folds used as validation on the remaining 90%, nominal patient fractions are approximately 81%/9%/10%, rather than exactly the model card's 80%/10%/10%; image fractions depend on studies per patient. The recovered fold helper resets NumPy's global seed, so later holdout draws can repeat rather than create independent sets when the helper consumes no additional randomness. Exact historical memberships cannot be recovered from this source alone: initial holdout sampling precedes the helper's seed reset and depends on RNG state.
5. `create_kfold_cv.py` is an alternative five-fold preparation that writes `train_kfold_cv.csv`; it is not the three-holdout generator. Config snapshots use more than one historical CSV naming convention. Do not silently substitute one split file for another.

`view_cropped.ipynb` is an inspection scratchpad. Notebook outputs and embedded attachments are removed, while code cells remain unchanged.

## CBIS-DDSM pretraining workflow

The prepared input data is available from [CBIS-DDSM PNG with Bounding Box ROIs](https://www.kaggle.com/datasets/vaillant/cbis-ddsm), [version 1](https://www.kaggle.com/datasets/vaillant/cbis-ddsm/versions/1). It provides `cleaned_cbis_ddsm.csv` and PNG mammograms, so users can start from this release rather than reconstructing the original DICOM conversion/annotation cleanup.

The verified CSV has 3,568 lesion rows referring to 3,032 unique image paths, with columns `filepath`, `patient_id`, `abnormality`, `pathology`, `subtlety`, `x1`, `y1`, `x2`, `y2`. The archive listing uses `png/png/<filename>.png`; arrange that inner image directory as `ROOT/png/` for these scripts, and place `cleaned_cbis_ddsm.csv` at ROOT. `datasets.json` records the version, schema, counts and annotation checksum. Only metadata and the CSV were inspected; the image payload/full image inventory was not verified.

Starting with those prepared inputs, in `CBIS-DDSM/`:

1. `generate_masks_from_bbox.ipynb` consumes existing `png/` images and `cleaned_cbis_ddsm.csv`, maps benign/malignant calcification/mass categories, draws labeled ellipse masks from bounding boxes, and writes `masks/`. Its helper also relabels all existing foreground on every lesion, so later lesions overwrite earlier categories on mixed-category multi-lesion images. Its final plotting cell refers to a commented-out `mask` variable. These limitations are annotated in the preserved notebook; it is not a validated Run All recipe.
2. `crop_images_and_masks.ipynb` applies the published crop model and the same box to image/mask pairs, writing `cropped_png/` and `cropped_masks/`. These are the paths expected by the DDSM training configs.
3. `create_kfold_cv.py` aggregates lesion labels per image and creates patient-grouped nested folds in `train_kfold_cv.csv`.
4. `generate_patches.ipynb` and `create_kfold_cv_patches.py` support historical patch-based alternatives; the published screening ensemble uses whole-image pretraining/fine-tuning recipes.

`eda.ipynb` inspects existing images/boxes. The original scripts that created these PNGs and the cleaned bounding-box CSV are not present in the recovered source. The linked prepared dataset supplies those prerequisites; this repository covers the downstream mask/crop/fold processing and does not claim to reconstruct the raw DICOM-to-prepared-release conversion.

## Using the source

Install the separate training framework (`pip install -e training`) for `skp.toolbox.functions.create_double_cv`. Additional preparation imports are covered by `pip install -e 'training[processing]'`; appropriate DICOM pixel decoders may also be needed. Adapt ROOT paths first. No original data conversion, splitting or model inference was launched during recovery. Source hashes and transformations are in `source-manifest.json`.

These scripts are Apache-2.0 licensed; see LICENSE. The training framework and configs are also Apache-2.0 in this repository; embedded third-party notices are retained separately.
