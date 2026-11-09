# Optional combined inference

From the repository root:

## Install

Requires Python 3.10 or newer. Install PyTorch and torchvision appropriate for your CPU/CUDA platform using the [PyTorch installation guide](https://pytorch.org/get-started/locally/), then install this project:

```bash
git clone https://github.com/i-pan/mammoscreen.git
cd mammoscreen
pip install -e '.[dicom]'
```

The optional DICOM extra uses pydicom. Compressed DICOM transfer syntaxes may require additional pixel decoders. Models download from Hugging Face on first use; their weights are not stored in Git.

## Python

```python
from mammoscreen import MammoscreenPipeline, load_image

pipeline = MammoscreenPipeline.from_pretrained(device="cpu")  # or cuda:0
result = pipeline({
    "cc": load_image("left_cc.dcm"),
    "mlo": load_image("left_mlo.dcm"),
})
print(result["cancer"])   # CPU tensor, shape (N,)
print(result["density"])  # CPU tensor, shape (N, 4)
```

Each mapping represents **one breast**, not a patient. Keep left and right separate. Supply `cc`, `mlo`, or both. For several breasts, pass a list of mappings. Direct NumPy inputs must be nonempty 2D uint8 grayscale arrays with pixels in [0, 255]. Single-view inference is supported by the published model but has lower reported performance. Density outputs follow the original model's four-class ordering; a clinically named class mapping has not been independently verified here.

The wrapper uses the published crop coordinates, clips them to image bounds, and rejects empty crops. DICOM loading follows the model card: VOI LUT, MONOCHROME1 inversion, and per-image conversion to uint8. Constant images produce zero pixels. Raster uint8 pixels are unchanged; higher-bit grayscale rasters are min/max converted. RGB and multiframe inputs are rejected.

The square network explicitly uses reflection padding, as recorded in its saved training configuration. This avoids the changed default padding between Albumentations 1.x and 2.x; the original HF source left that default implicit.

## Command line

```bash
mammoscreen --cc left_cc.png --mlo left_mlo.png --device cpu
```

The command prints JSON containing `cancer` and `density` arrays. Use a separate invocation for the other breast.

## Offline snapshots

```python
pipeline = MammoscreenPipeline.from_pretrained(
    crop_source="/path/to/mammo-crop-snapshot",
    screen_source="/path/to/mammoscreen-snapshot",
    local_files_only=True,
)
```

Both directories must contain `config.json` and `model.safetensors`. Native model implementations are included in this package, so `trust_remote_code=True` is unnecessary. Default HF loads are pinned:

| Model | Revision |
| --- | --- |
| mammo-crop | `bec2e302d47176d743a86d276d8402f876ad0f9f` |
| mammoscreen | `b619a889a8214f3b3f9e7427c2603608a44e6d0a` |

## Limits

The wrapper was checked on synthetic inputs; it has not undergone independent clinical validation. Published performance belongs to the original HF evaluation. CUDA execution is unverified.
