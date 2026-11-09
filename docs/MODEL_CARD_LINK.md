# Proposed mammoscreen model card addition

Add this section after the model introduction once the repository is published:

## Training code and evaluation

The [Mammoscreen training repository](https://github.com/i-pan/mammoscreen) contains the recovered training framework, CBIS-DDSM pretraining and RSNA fine-tuning configurations, model and dataset implementations, custom losses and metrics, EMA callbacks, and evaluation notebooks. It documents the recipes and recorded results associated with the published ensemble and includes the related mammo-crop training configuration. An optional integrated inference API is also included.

Historical data paths and tracking settings need adaptation; original data and split files are not bundled. The repository preserves training evidence rather than claiming an exact rerun of the original study.
