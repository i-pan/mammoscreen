# Corrected preparation utilities

These are new reference utilities alongside the preserved historical notebooks. They fix the two documented implementation errors; they do not claim to reproduce the original training labels, regenerate the historical splits, or change the published models.

## Pixel crop labels

```bash
python training/data_preparation/corrected/prepare.py crop-labels \
  --images /path/to/train_png --output /path/to/crop_coords.csv
```

The utility ignores a five-pixel border, thresholds the interior at >10, and selects the largest 8-connected foreground component. It restores the border offset and writes `filename,x,y,w,h` in **original-image pixels** exactly once. The CSV is the pre-fold coordinate table: merge patient/image identifiers and create patient-grouped folds separately before using `cfg_crop`. The training loader handles coordinate normalization after image augmentation/resizing.

Blank/too-small images raise an error rather than fabricate a box. The command validates all crop rows before writing a new CSV and refuses to overwrite an existing file. It scans lowercase `.png` paths recursively and uses paths relative to the supplied image directory.

## DDSM ellipse masks

```bash
python training/data_preparation/corrected/prepare.py ddsm-masks \
  --images /path/to/png \
  --annotations /path/to/cleaned_cbis_ddsm.csv \
  --output /path/to/new_masks
```

This retains the historical notebook's column mapping by default: `left=y1`, `top=x1`, `width=y2-y1`, `height=x2-x1` (`--bbox-order yx`). Use `--bbox-order xy` only for an annotation source with conventional x=column/y=row boxes. The mapping is explicit and tested against synthetic geometry; it has not been visually revalidated on the full Kaggle image collection.

A fresh ellipse is drawn for each lesion. Its category is assigned only within that ellipse, so disjoint earlier lesions retain their labels. On overlapping pixels, later CSV rows win. Background is zero; the four categories are fixed:

| Value | Category |
| --- | --- |
| 1 | benign calcification |
| 2 | benign mass |
| 3 | malignant calcification |
| 4 | malignant mass |

The output contains lossless PNG masks preserving the annotation-relative image paths, plus `labels.json`. Paths such as `view.png` and `./view.png` are normalized into the same image group. Absolute paths, parent-directory components, and non-PNG filenames are rejected. Individual output files are created exclusively to prevent overwriting. Unknown categories, missing images, invalid boxes and paths outside the supplied image directory raise errors. Out-of-bounds boxes are rejected rather than silently clipped. The output directory must not exist. On a later failure, already-written masks remain in that new directory; inspect/remove that partial output or choose another fresh directory before retrying. Existing masks are never overwritten.

## Validation

Synthetic tests independently check original-image crop coordinates, largest-component selection, disjoint and overlapping mask categories, CSV coordinate ordering, command-line file outputs, and overwrite refusal. No original dataset preparation or model evaluation was run. The historical source and its manifests are unchanged. These new utilities are Apache-2.0 licensed under the parent directory's LICENSE.
