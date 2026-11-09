# Bounded pipeline verification

Reviewed 2026-11-09 against the active [training recipe](../training/PIPELINE.md).
The aim is a usable new training run from obtainable inputs, without a full
study rerun or a promise of identical weights and scores.

| Stage | Supplied/downloadable input | Producer and next consumer | Verification |
| --- | --- | --- | --- |
| Acquisition | RSNA competition data; prepared vaillant/cbis-ddsm v1; public ianpan/mammo-crop | Layout in PIPELINE.md; crop model supplies preprocessing | Public sources and layout documented; deleted local datasets are not required artifacts |
| RSNA ETL | DICOMs | Archived converter → train_png → crop_images.py → train_cropped_png | Generated 14-bit lossless JPEG2000 MONOCHROME1 DICOM decoded through Pillow and converted to exact independently expected 8-bit pixels; archived crop script exercised with preserved real crop weights |
| DDSM ETL | Prepared PNGs and lesion CSV | Corrected mask utility → archived paired crop loop → archived grouped-fold producer | 24 synthetic images with all four lesion categories; real crop weights; every image/mask pair aligned and retained foreground; generated ten-fold table has expected training columns |
| Screening labels/splits | Raw RSNA train.csv | Corrected splitter → hash-checked recipe | Determinism, distinct holdouts, patient isolation, auxiliary encoding and tamper rejection covered by tests |
| DDSM training | Cropped PNG/mask table | Actual cls_seg dataset/model/loss/task/optimizer/scheduler → last.ckpt | All three recipe members trained for two CPU steps at 64×64, with validation and checkpoint saving |
| Screening training | Matched split and generated DDSM checkpoint | Actual classification dataset/model/loss/task/optimizer/scheduler → EMA checkpoint | All three members exercised for three CPU steps at 64×64, including validation, local logging and EMA saving |
| Prediction/evaluation | Three freshly generated 11-output EMA files | Actual evaluator → image probabilities → breast means/AUC and metrics JSON | Real generated checkpoints strictly loaded and evaluated on synthetic holdout images; independent probability/AUC tests cover semantics |
| Distribution | Training wheel | Recipe/helpers/config wrappers/logger/archived code | Built wheel inspected and checked outside the checkout using the existing CPU dependency environment |

The bounded lifecycle disables pretrained downloads, uses one CPU, reduces all
spatial dimensions to 64×64, uses two-image batches and zero loader workers,
replaces augmentation with resize, and limits DDSM to two training batches, screening to three, and both stages to two
validation batches. Screening uses three steps because the EMA callback copies raw
weights on its first two updates; the check then verifies distinct EMA/raw weights. These are verification overrides, not the actual experiment
recipe. They exercise all six real model/task paths, not substitute networks.
They do not test ImageNet weight downloading, full-resolution throughput/memory,
CUDA precision, SyncBatchNorm or real multi-GPU execution.

A startup failure was found with current Lightning: its loggers package no longer
exports NeptuneLogger, which the archived runner imports. The active adapter now
supplies the local logger during that import and restores the original package
state immediately afterward. Regression coverage checks both older Lightning
with the export and newer Lightning without it. The archived source is unchanged.
No hosted tracking account is used. The retained Neptune SDK is used only by the
historical tasks' configuration-stringification helper.

## Repeatable checks

Run the ordinary tests from the repository root. In an environment with the
training dependencies installed, explicitly enable the bounded lifecycle check:

```bash
pytest -q
MAMMOSCREEN_RUN_LIFECYCLE=1 pytest -q tests/test_training_lifecycle.py
```

The lifecycle uses pytest's temporary directory, synthetic labels/pixels and random
model initialization. It creates no user dataset files and makes no pretrained
model downloads. The larger historical ETL checks above used temporary synthetic
files and the already preserved local crop model snapshot; they did not download
or process the full datasets.

## Observed environment and limits

Python 3.12.11; torch 2.15.0+cpu; torchvision 0.29.2+cpu; Lightning 2.6.6;
torchmetrics 1.9.0; timm 1.0.31; Albumentations 2.0.8; NumPy 2.5.4;
pandas 3.1.1; scikit-learn 1.9.1; scikit-image 0.26.0; MONAI 1.6.1;
torcheval 0.0.7; neptune 1.14.0.post2; Pillow 13.0.0; pydicom 3.0.2;
Transformers 4.57.6. This records the exercised versions, not a universal lockfile.

No complete real-data ETL, full training run or cohort-performance evaluation was
performed. Real-data visual inspection and GPU/distributed execution remain
unverified. A fresh operating-system environment was not rebuilt; installed-wheel
checks reused the available dependency environment. Independent read-only source
review and the bounded runtime checks establish the usable handoffs, rather than
certifying every hardware/environment combination or exact historical results.
