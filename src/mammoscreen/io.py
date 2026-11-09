"""Image loading without executing model repository code."""

from pathlib import Path

import cv2
import numpy as np


def _uint8(image: np.ndarray) -> np.ndarray:
    if image.ndim != 2 or 0 in image.shape:
        raise ValueError("Expected a nonempty, single-frame grayscale image")
    if image.dtype == np.uint8:
        return image
    image = image.astype(np.float64)
    if not np.isfinite(image).all():
        raise ValueError("Image contains nonfinite pixels")
    image -= image.min()
    maximum = image.max()
    if maximum == 0:
        return np.zeros(image.shape, dtype=np.uint8)
    return (image / maximum * 255).astype(np.uint8)


def load_image(path: str | Path) -> np.ndarray:
    """Load grayscale raster or DICOM; return uint8 pixels in [0, 255].

    DICOM follows the published model's VOI LUT, MONOCHROME1 inversion,
    and per-image min/max conversion. Constant DICOMs become zeros.
    """
    path = Path(path)
    if path.suffix.lower() in {".dcm", ".dicom"}:
        from pydicom import dcmread
        from pydicom.pixels import apply_voi_lut

        ds = dcmread(path)
        image = apply_voi_lut(ds.pixel_array, ds).astype(np.float64)
        if ds.PhotometricInterpretation == "MONOCHROME1":
            image = image.max() - image
        return _uint8(image)
    if cv2.imcount(str(path)) > 1:
        raise ValueError("Expected a single-frame raster image")
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return _uint8(image)
