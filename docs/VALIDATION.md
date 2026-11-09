# Validation

## Training companion

The expanded training subtree has four focused checks: source-manifest hashes and Python syntax, included internal imports and dynamic model/dataset/task references, recovered screening loss versus an independent numerical oracle (including masked auxiliary labels), and consistency of recorded evaluation results. The actual recovered loss was exercised on synthetic logits with backward gradients. All 23 config modules were also imported successfully. The actual recovered screening model passed a reduced-size synthetic forward/loss/backward check and the crop model passed a reduced-size synthetic forward check on CPU, with pretrained downloads disabled. No optimizer step, training loop, GPU run, training dependency environment or original study evaluation was recreated.

## Optional inference integration

Validated on 2026-11-09 with Python 3.12.11, PyTorch 2.15.0+cpu, torchvision 0.29.2+cpu, Transformers 4.57.6, timm 1.0.31, Albumentations 2.0.8 and pydicom 3.0.2.

- Twelve tests passed: breast/view grouping, exact crop application, clipping, invalid boxes and inputs, raster/DICOM conversion, multipage TIFF rejection, training-compatible reflection padding, actual model probability aggregation, and CLI serialization/options.
- Ruff passed for integration code. Model implementations are excluded from style checks. Three files retain upstream bytes; the screening implementation has a documented reflection-padding patch, with upstream and integrated hashes recorded separately.
- Wheel and source distribution built successfully. Inspection confirmed model weights, run pickles and raw recovery archives are excluded.
- Both pinned local models loaded with safetensors, with no dynamic remote-code loading or network required. A synthetic grayscale image passed through crop and screening on CPU. The crop was 508 × 193; output shapes were cancer (1,) and density (1, 4), with finite values and normalized density probabilities.
- The built wheel was imported in isolated Python mode and its real CLI path ran both preserved models offline on a synthetic image. JSON outputs were finite, with one cancer probability and four density probabilities summing to approximately one. CLI help and package entry point were also checked.

The real-weight check establishes that the integration executes. It does not evaluate clinical performance, validate calibration or reproduce the published test cohort. CUDA was unavailable on this host; GPU execution is unverified.
