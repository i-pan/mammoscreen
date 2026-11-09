"""Compressed-DICOM check of the archived converter on synthetic pixels."""

import io
from pathlib import Path

import cv2
import numpy as np
import pydicom
from PIL import Image
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import JPEG2000Lossless, generate_uid


def test_archived_converter_decodes_jpeg2000_and_inverts_monochrome1(tmp_path):
    root = tmp_path
    path = root / 'train_images/1/1.dcm'
    path.parent.mkdir(parents=True)
    original = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 3)
    stream = io.BytesIO()
    Image.fromarray(original).save(stream, format='JPEG2000', no_jp2=True, irreversible=False)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = JPEG2000Lossless
    meta.MediaStorageSOPClassUID = pydicom.uid.DigitalMammographyXRayImageStorageForPresentation
    meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b'\0' * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = original.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = 'MONOCHROME1'
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 14, 13, 0
    ds.PixelData = encapsulate([stream.getvalue()])
    ds['PixelData'].is_undefined_length = True
    ds.save_as(path, enforce_file_format=True)
    np.testing.assert_array_equal(pydicom.dcmread(path).pixel_array, original)
    source = (Path(__file__).resolve().parents[1] /
              'training/data_preparation/rsna-breast-cancer-detection/convert_dicom_to_png.py').read_text()
    source = source.replace('ROOT = "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/"', f'ROOT = {str(root)!r}')
    exec(compile(source, 'archived_convert_dicom_to_png.py', 'exec'), {})  # noqa: S102 - local archived source
    output = cv2.imread(str(root / 'train_png/1/1.png'), 0)
    expected = original.max() - original.astype('float32')
    expected = (expected / expected.max() * 255).astype('uint8')
    np.testing.assert_array_equal(output, expected)
    print('14-bit JPEG2000 MONOCHROME1 -> archived converter -> exact expected 8-bit PNG: passed')
