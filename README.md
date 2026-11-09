# Mammoscreen training code

Training code, configurations, and evaluation notebooks behind [ianpan/mammoscreen](https://huggingface.co/ianpan/mammoscreen) and the associated [mammo-crop](https://huggingface.co/ianpan/mammo-crop) model. The trained models are hosted on Hugging Face; this repository documents how they were built and evaluated. Prepared CBIS-DDSM images and bounding-box annotations are available in [the associated Kaggle dataset](https://www.kaggle.com/datasets/vaillant/cbis-ddsm).

## Run a new experiment

Use the [corrected end-to-end recipe](training/PIPELINE.md) for downloadable inputs, preparation, patient-disjoint splits, DDSM initialization, all three screening models and evaluation of fresh EMA checkpoints. The historical defaults below are provenance; do not mix their split filenames into new runs.

## Model training

The screening ensemble uses three EfficientNetV2-S networks. Each backbone is first pretrained with joint classification and segmentation on CBIS-DDSM, then fine-tuned on cropped RSNA screening mammograms with cancer and auxiliary targets. Exponential moving average weights are used for evaluation. Networks operate at 2048 × 1024, 1920 × 1280, and 1536 × 1536; predictions are averaged across views and networks for each breast.

The crop model uses MobileNetV3-Small to regress an `xywh` bounding box and remove unnecessary background. Its recovered config, bounding-box dataset loader, L1 loss, and evaluation metrics are included alongside the screening code. The crop config is a provenance candidate, not a verified exact export recipe; its batch size differs from the HF card.

## Start here

| Content | Location |
| --- | --- |
| Training setup, stages, data contracts, and example commands | [Training guide](training/README.md) |
| DICOM conversion, crop/mask preparation and split generation | [Data preparation](training/data_preparation/README.md) |
| Active training/evaluation launcher | [training/mammoscreen_recipe.py](training/mammoscreen_recipe.py) |
| Preserved historical runner | [training/skp/train.py](training/skp/train.py) |
| Mammography recipes | [training/skp/configs/mammo/](training/skp/configs/mammo/) |
| Training models, datasets, losses, metrics, and EMA | [training/skp/](training/skp/) |
| Evaluation notebooks | [training/notebooks/](training/notebooks/) |
| Recorded evaluation results and run identities | [Evaluation evidence](training/evaluation-evidence.json) |
| Model-to-run mapping and source limitations | [Training provenance](docs/TRAINING.md) |
| Optional crop-plus-screen inference API and CLI | [Inference guide](docs/INFERENCE.md) |

## Published ensemble

| Resolution | Fine-tuning config | First evaluation set run |
| --- | --- | --- |
| 2048 × 1024 | `mammo.cfg_ddsm_pretrained_aux_losses` | `6a3b30f7` |
| 1920 × 1280 | `mammo.cfg_ddsm_pretrained_aux_losses_v02` | `ddfeb3f6` |
| 1536 × 1536 | `mammo.cfg_ddsm_pretrained_aux_losses_v04` | `b386692c` |

The preserved evaluation notebook records breast-level ensemble AUCs **0.9464, 0.9467, and 0.9422** across three evaluation sets, matching the HF model card. The card states the first set's weights were uploaded. This ties the recipes and recorded results to the published model; equality to the original training tensors has not been independently verified.

## Source and scope

The training subtree contains a mammography-focused selection of the original [SKP framework](https://github.com/i-pan/SKP/tree/81305a8aa81872791a01894ce6b1b6cb242b7157), plus preserved local mammography configs. Shared modules retain their original implementation, including some generic classes used by other projects. [Source hashes](training/source-manifest.json) distinguish upstream code from the local config snapshot. Notebook outputs are cleared; numerical evaluation evidence is preserved separately.

This is a training-code companion, not a claim of an exact historical environment or a newly rerun study. Original data, split CSVs, pretrained initialization files, private run records, and weights are not bundled. The active recipe binds your data/output paths and uses local logging; historical examples retain original paths and tracking settings. These gaps do not change the recorded training evidence. No training was launched during source recovery.

## Validation

Training source and import coverage are checked, and a bounded CPU lifecycle exercises all three real DDSM and screening models through validation, checkpoint/EMA saving and evaluation on synthetic data. Compressed-DICOM conversion and the paired crop/fold preparation handoffs were also exercised; see [bounded pipeline verification](docs/PIPELINE_VERIFICATION.md) for exact overrides and remaining limits. The optional inference integration has its own tests and a real-weight CPU smoke check; those checks do not certify training or clinical performance. See [validation](docs/VALIDATION.md) and [training review scope](docs/TRAINING_REVIEW.md).

## License

This repository, including Ian Pan-authored training code, data preparation, and the optional inference integration, is licensed under [Apache-2.0](LICENSE). Third-party implementations retain their required attribution and license notices in [training/THIRD_PARTY_NOTICES](training/THIRD_PARTY_NOTICES). Published model weights retain the licenses stated in their Hugging Face repositories.
