# Training provenance

The mammography recipes were recovered from i-pan/SKP and a preserved local source snapshot. The source revision used to recover the complete framework is `81305a8aa81872791a01894ce6b1b6cb242b7157` (historically MIT licensed). Ian Pan-authored code is distributed under Apache-2.0 in this repository; retained third-party notices are in `training/THIRD_PARTY_NOTICES`.

## Screening ensemble

The three configs below use EfficientNetV2-S and CBIS-DDSM pretrained initialization. Their 11-output training heads cover cancer, biopsy, invasive disease, difficult-negative cases, three BIRADS outputs and four density outputs. The HF model exposes only cancer and density (five outputs per network). A weight-row selection step is implied, but the original conversion script and tensor equivalence are unverified.

| Resolution | Config | First evaluation set run | Fold |
| --- | --- | --- | --- |
| 2048 × 1024 | `cfg_ddsm_pretrained_aux_losses.py` | `6a3b30f7` | 0 |
| 1920 × 1280 | `cfg_ddsm_pretrained_aux_losses_v02.py` | `ddfeb3f6` | 1 |
| 1536 × 1536 | `cfg_ddsm_pretrained_aux_losses_v04.py` | `b386692c` | 2 |

The upstream `notebooks/mammo/eval_model.ipynb` records three ensemble AUCs of 0.9464, 0.9467 and 0.9422, matching the HF card. The first result uses the three runs above; the card states the first split was uploaded. This is documentary identification, not verification that tensors match.

Static pickle-opcode inspection of the preserved `b386692c/config.pkl` records the square network's `PadIfNeeded` border mode as 4 (`cv2.BORDER_REFLECT_101`). No pickle code was executed. The integrated inference code selects this mode explicitly because Albumentations 2 changed its default to constant padding. The original HF source and its checksum remain recorded separately from this downstream patch.

The original nine evaluation EMA files are no longer available locally. Their config records survive in local preservation storage. Split CSVs and original dataset paths were not recovered. Private run pickles, machine-specific audit records, model weights and unrelated framework snapshots are excluded from Git.

## Cropping

`cfg_crop.py` records a four-coordinate crop regression model using MobileNetV3-Small, matching the published mammo-crop architecture. The crop model card describes 90/10 training and validation and a connected-component method to generate bounding-box labels. Exact crop checkpoint provenance has not been independently established.

## Recovery boundary

`training/skp/configs/mammo/` preserves historical recipes (two files have trailing whitespace normalized). The relevant SKP framework is included in `training/skp/`; importing or launching it requires the separate training dependencies and adaptation of the historical data/tracking paths. The training subtree has not been runtime-certified on a newly recreated historical environment. No training or new model evaluation was launched during recovery.
