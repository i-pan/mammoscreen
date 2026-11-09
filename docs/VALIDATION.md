# Validation

## Training companion

The expanded training subtree has five focused checks: source-manifest hashes and Python syntax, included internal imports and dynamic model/dataset/task references, recovered screening loss versus an independent numerical oracle (including masked auxiliary labels), consistency of recorded evaluation results, and provenance/notebook sanitization for the 13 data-preparation files. The actual recovered loss was exercised on synthetic logits with backward gradients. All 23 config modules were also imported successfully. The actual recovered screening model passed a reduced-size synthetic forward/loss/backward check and the crop model passed a reduced-size synthetic forward check on CPU, with pretrained downloads disabled. No optimizer step, training loop, GPU run, training dependency environment or original study evaluation was recreated.

## Optional inference integration

Validated on 2026-11-09 with Python 3.12.11, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, Transformers 4.57.6, timm 1.0.31, Albumentations 2.0.8 and pydicom 3.0.2.

- Twelve tests passed: breast/view grouping, exact crop application, clipping, invalid boxes and inputs, raster/DICOM conversion, multipage TIFF rejection, training-compatible reflection padding, actual model probability aggregation, and CLI serialization/options.
- Ruff passed for integration code. Model implementations are excluded from style checks. Three files retain upstream bytes; the screening implementation has a documented reflection-padding patch, with upstream and integrated hashes recorded separately.
- Wheel and source distribution built successfully. Inspection confirmed model weights, run pickles and raw recovery archives are excluded.
- Both pinned local models loaded with safetensors, with no dynamic remote-code loading or network required. A synthetic grayscale image passed through crop and screening on CPU. The crop was 508 × 193; output shapes were cancer (1,) and density (1, 4), with finite values and normalized density probabilities.
- The built wheel was imported in isolated Python mode and its real CLI path ran both preserved models offline on a synthetic image. JSON outputs were finite, with one cancer probability and four density probabilities summing to approximately one. CLI help and package entry point were also checked.

The real-weight check establishes that the integration executes. It does not evaluate clinical performance, validate calibration or reproduce the published test cohort. CUDA was unavailable on this host; GPU execution is unverified.

## Corrected preparation utilities

Six synthetic checks cover original-image pixel crop coordinates, largest-component selection, blank-image rejection, CSV outputs, disjoint/overlapping lesion labels, both CSV coordinate orders, normalized path aliases, path escape rejection and overwrite refusal. An independent reviewer identified a mask-path alias overwrite; the CLI regression failed before the fix and passed afterward. The final implementation groups normalized relative paths and creates mask files exclusively. No user dataset processing was run.

## Active split/training/evaluation recipe

The corrected splitter and active recipe are checked with synthetic patient tables: local RNG ownership and deterministic reruns, distinct holdouts, patient isolation, binary/auxiliary label encoding, train-only sampling weights, actual configuration routing, generated DDSM/EMA paths, launch arguments and config-wrapper imports, tampered CSV rejection and probability-mean/AUC semantics. A tiny parameter-only network fixture exercises the real evaluation adapter loading generated-format 11-head EMA checkpoint files and writing predictions/metrics. It does not constitute real-model inference or full training validation.

The historical data, config and notebook sources remain unchanged. The new recipe is a usable new-experiment route, not a claim to recreate historical split memberships or exact original weights. Independent source review found no remaining material handoff/leakage findings for the documented screening route.

The active runner uses a separate local Lightning logger; archived runner bytes remain unchanged. Two bounded tests exercise real Lightning 2.6.6 channel/metric writes, tensor serialization, nonzero-rank suppression and the training adapter's argument handling. DDP relaunch keeps the recipe module and `_train` arguments so each rank applies the logger adapter. No actual distributed process group or training loop was launched. The local suite now contains 30 passing tests.

## Bounded lifecycle follow-up

The [pipeline verification record](PIPELINE_VERIFICATION.md) adds actual six-model synthetic CPU train/validate/checkpoint/EMA/evaluation checks, compressed-DICOM conversion and real crop-model ETL checks. These checks found and corrected the current Lightning import incompatibility before the local logger adapter took over. Historical source hashes remain unchanged. Earlier statements that no training lifecycle was exercised describe the earlier review stages; this follow-up supersedes that limitation only for the bounded synthetic paths described.

The final default suite passes 32 tests with the opt-in lifecycle skipped. The explicit lifecycle passes separately, including DDSM encoder tensor transfer and distinct EMA/raw screening weights after averaging starts. A clean training-wheel installation outside the checkout repeats this lifecycle with Hugging Face offline, while reusing the observed CPU dependency environment. CI includes the bounded lifecycle as a separate job.
