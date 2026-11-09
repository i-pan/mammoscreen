# Verification

The [active training recipe](../training/PIPELINE.md) is complete on inspection
and has passed bounded runtime checks. No full study rerun or exact historical
replication is claimed. Independent read-only reviews found no remaining concrete
blockers in the documented route.

[GitHub Actions](https://github.com/i-pan/mammoscreen/actions/runs/42644056447)
passed unit, lint and package-build checks plus the six-model bounded lifecycle
at code revision `34bb3bf`, including a fresh training-dependency installation on
clean Ubuntu runners. The default suite passed 32 tests; the opt-in lifecycle
passed separately.

## Training pipeline checks

| Stage | Route checked | Evidence |
| --- | --- | --- |
| Inputs | RSNA competition data, prepared vaillant/cbis-ddsm v1 and public ianpan/mammo-crop | Sources, access requirements and layouts documented in the active recipe |
| RSNA preparation | DICOM → PNG → cropped PNG | Synthetic 14-bit lossless JPEG2000 MONOCHROME1 conversion matched independently expected pixels; archived crop script ran with preserved real crop weights |
| DDSM preparation | Prepared PNG/lesion CSV → corrected masks → paired crops → grouped folds | 24 synthetic images covering all four lesion categories; real crop weights; aligned image/mask outputs with retained foreground; expected ten-fold training table |
| Screening splits | Raw annotations → patient partitions → hash-checked recipe | Determinism, distinct holdouts, patient isolation, auxiliary encoding, configuration routing and tamper rejection |
| DDSM training | Real datasets/models/losses/tasks/optimizers/schedulers → last.ckpt | All three members completed two CPU training batches, validation and checkpoint saving |
| Screening training | Matched split and generated DDSM weights → EMA checkpoint | All three members completed three CPU training batches and validation; encoder tensors matched the DDSM initialization and saved EMA weights differed from raw weights |
| Evaluation | Three generated 11-output EMA files → probabilities → breast means/AUC | Real checkpoints loaded strictly and evaluated on synthetic holdouts; independent aggregation/AUC expectations tested |
| Packaging | Training wheel → recipe, helpers, configs, logger and runner | Lifecycle repeated outside the checkout with the installed wheel and Hugging Face offline |

The lifecycle check uses random initialization, one CPU, 64×64 images,
two-image batches, zero loader workers and resize-only augmentation. Each stage
uses at most two validation batches. Screening runs three training batches because
the EMA callback copies raw weights on its first two updates. These are test
overrides; the actual experiment settings remain in the active recipe.

Additional tests check source-manifest hashes, Python syntax, internal and dynamic
imports, sanitized notebooks, recorded evaluation results, the recovered screening
loss against an independent numerical oracle, and missing-label gradients.
Preparation tests cover crop-coordinate units, foreground selection, mask class
mapping and overlaps, coordinate order, normalized paths and overwrite refusal.
Logger tests cover metric/config writes, rank suppression, DDP relaunch arguments
and compatibility with Lightning releases with or without the old Neptune export.
The active runner saves local JSONL logs; the retained Neptune SDK is used by the
historical tasks' configuration-stringification helper. Archived source remains
unchanged.

## Optional inference checks

The integration tests cover grouping, crop application/clipping, invalid inputs,
raster/DICOM conversion, multipage TIFF rejection, reflection padding, probability
aggregation and CLI output. Both pinned crop and screening models also ran offline
on a synthetic image through the built wheel's real CLI, producing finite cancer
probabilities and normalized density probabilities. Packaging checks exclude model
weights, private run records and recovery archives. Model-code origins and the
explicit reflection-padding patch are recorded in [model-code-provenance.json](model-code-provenance.json).

## Repeatable checks

From the repository root, with the appropriate test/training dependencies installed:

```bash
pytest -q
MAMMOSCREEN_RUN_LIFECYCLE=1 pytest -q tests/test_training_lifecycle.py
```

The lifecycle writes synthetic inputs and checkpoints to pytest's temporary
directory and downloads no pretrained models. CI runs it as a separate job.
The larger archived crop/fold checks used temporary synthetic files and the
already preserved local crop-model snapshot.

## Observed environment and limits

Reviewed on 2026-11-09 with Python 3.12.11; torch 2.15.0+cpu;
torchvision 0.29.2+cpu; Lightning 2.6.6; torchmetrics 1.9.0; timm 1.0.31;
Albumentations 2.0.8; NumPy 2.5.4; pandas 3.1.1; scikit-learn 1.9.1;
scikit-image 0.26.0; MONAI 1.6.1; torcheval 0.0.7; neptune 1.14.0.post2;
Pillow 13.0.0; pydicom 3.0.2; Transformers 4.57.6. These are observed versions,
not a universal lockfile. Local wheel checks reused this dependency environment;
the linked CI run separately exercised a clean dependency installation.

Full real-data ETL, visual inspection of real image/mask pairs, full-resolution
GPU training, mixed precision, SyncBatchNorm, actual multi-GPU execution, ImageNet
weight downloading and real-cohort performance remain unverified. Synthetic checks
do not establish clinical performance or calibration. Original split memberships,
exact historical effective overrides, final crop-run identity and tensor parity
with the HF screening export are not established; see [training provenance](TRAINING.md).
