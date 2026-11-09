# Mammography training guide

This subtree publishes the training implementation and evidence for the two Hugging Face models. It contains the original SKP runner and supporting modules, preserved mammography recipes, and evaluation notebooks. Reading the source and recorded results does not require recreating the original training environment.

## Stages

1. **Breast crop regression:** `skp/configs/mammo/cfg_crop.py` uses MobileNetV3-Small, GeM pooling, a sigmoid four-coordinate head, L1 loss and normalized `xywh` targets. The loader transforms boxes with the image and normalizes after resizing. The crop model card describes largest-component foreground boxes and a 90/10 split. The recovered config uses batch size 128, whereas the published card says 256; its exact final run is not established.
2. **CBIS-DDSM pretraining:** the `cfg_ddsm_seg_cls*` family trains a shared EfficientNetV2-S encoder with segmentation and classification heads. Targets cover benign/malignant calcifications and masses. This supplies the classification backbone initialization.
3. **RSNA fine-tuning:** `cfg_ddsm_pretrained_aux_losses`, `_v02` and `_v04` train the three screening resolutions. The output head has 11 logits and the six-task loss combines binary cancer/biopsy/invasive/difficult-case targets, three BIRADS categories and four density categories. Missing BIRADS/density labels use `-1` and are masked.
4. **EMA and evaluation:** `skp/callbacks/ema_callback.py` maintains EMA weights; `skp/train.py` saves them as `ema_weights.pt`. `notebooks/eval_model.ipynb` evaluates three ensembles, averages image predictions per breast and reports the recorded AUCs. `evaluation-evidence.json` retains the numerical results and associated run IDs.

Additional configs are historical alternatives, not additional members of the published ensemble. The 2400 × 960 `_v03` recipe is one such alternative.

## Code map

| Responsibility | Source |
| --- | --- |
| CLI overrides, model/task construction and training lifecycle | `skp/train.py` |
| Screening model and backbone initialization | `skp/models/classification/net2d.py` |
| Joint segmentation/classification pretraining | `skp/models/segmentation/seg_cls.py` |
| Screening images, splits and breast grouping | `skp/datasets/mammo/cls.py` |
| DDSM images, masks and classification targets | `skp/datasets/mammo/seg_cls.py` |
| Bounding-box images and targets | `skp/datasets/crop2d.py` |
| Screening multi-task loss | `skp/losses/custom.py` → `MammoCancerWithAuxLosses` |
| Breast-level AUC and pF1 | `skp/metrics/custom_cls.py` |
| EMA updates and validation weight swapping | `skp/callbacks/ema_callback.py` |
| Learning rate schedule | `skp/optim/linear_warmup_cosine_annealing.py` |

## Data contracts

The screening configs expect a CSV with `filename`, `fold`, `breast_id_index`, and targets `cancer`, `biopsy`, `invasive`, `difficult_negative_case`, `BIRADS`, `density`. The dataset trains where `fold != cfg.fold` and validates where `fold == cfg.fold`. With nested CV enabled it excludes the `outer` holdout fold and uses the corresponding `inner{outer}` assignment. The held-out evaluation CSVs are separate from this train/validation routing. Images are cropped grayscale PNGs; the rectangular recipes pad to their aspect ratio before resizing. The square recipe uses longest-side resize and padding.

DDSM configs expect grayscale image and mask paths plus benign/malignant mass/calcification labels; see each config's `inputs`, `masks`, `targets`, `data_dir`, and `seg_data_dir`. Crop training expects `filename`, `fold` and pixel-coordinate `x,y,w,h` labels. These source datasets and original split memberships are not distributed here.

## Environment and launch interface

The framework uses PyTorch Lightning, timm, Albumentations and Neptune. `pyproject.toml` records the recovered dependencies plus statically referenced optional packages and corrected package discovery. This is not an exact historical lockfile or a newly verified training environment. Keep it separate from other projects that install an unrelated `skp` package.

After installing PyTorch appropriate for your system, an environment can be prepared with:

```bash
cd training
pip install -e .
# Additional libraries used by the HF evaluation notebook:
pip install -e '.[evaluation]'
```

Before launching, adapt the configs' absolute data, annotation, pretrained-checkpoint and output paths, and change the Neptune project/tracking settings to your own. Configs retain the historical paths intentionally. Changing a path only through a CLI override will not recompute other values already constructed from it in Python; update the source config where necessary. Albumentations defaults have changed between versions, so inspect historical transforms when adapting the recipes, particularly the square model's reflection padding.

The historical entry point is run from this directory:

```bash
python skp/train.py mammo.cfg_crop --kfold 0
python skp/train.py mammo.cfg_ddsm_seg_cls_v02 --kfold 0
python skp/train.py mammo.cfg_ddsm_pretrained_aux_losses --kfold 0 --ema.on True
```

These illustrate the original launch interface; they are not launches performed or fully validated during recovery. GPU counts, precision, fold routing, EMA decay/switch settings and pretrained fold paths need to match the experiment being examined. EMA defaults to off, so the example explicitly enables it; the exact historical effective overrides have not all been recovered. Do not treat the example as the exact original command.

The three fine-tuning configs reference DDSM initialization variants `_v02`, `_v03` and `_v05` respectively. Retain `--keep_pretrained_weights_path` when deliberately reusing an initialization from another dataset/fold; otherwise the runner rewrites `/foldN/` to the selected training fold.

## Evidence and boundaries

The source comes from pinned SKP revision `81305a8aa81872791a01894ce6b1b6cb242b7157`, with local snapshot mammography configs. `source-manifest.json` records exact origins and hashes; it does not assert these are the exact historical dirty bytes. Shared modules contain generic code also used by other experiments, but unrelated project configs and datasets are omitted. `UPSTREAM_README.md` is retained as historical framework documentation; paths in this guide describe the included subset.

Notebook code is preserved with outputs cleared. The HF `eval_automodel.ipynb` scratchpad has unfinished copied tail cells and is labeled accordingly; use `notebooks/eval_model.ipynb` for the three-ensemble historical evaluation. See `notebooks/README.md` for each notebook's scope. Numerical result cells are separately recorded without patient tables or images. No training was rerun. Original EMA artifacts, data, split files, full effective environment and final HF conversion script are not bundled. See `../docs/TRAINING.md` for the model-to-run mapping and limits. The optional inference package lives separately at the repository root and does not replace this training implementation.
