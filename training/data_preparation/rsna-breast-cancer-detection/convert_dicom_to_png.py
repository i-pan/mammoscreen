import cv2
import glob
import os
import pydicom

from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


ROOT = "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/"

images = glob.glob(os.path.join(ROOT, "train_images/*/*.dcm"))

save_dir = os.path.join(ROOT, "train_png")

for im in tqdm(images):
    dicom = pydicom.dcmread(im)
    arr = dicom.pixel_array
    arr = apply_voi_lut(arr, dicom)
    arr = arr.astype("float32")
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        arr = arr.max() - arr
    arr -= arr.min()
    arr /= arr.max()
    arr *= 255.0
    arr = arr.astype("uint8")
    tmp_save_dir = os.path.join(save_dir, os.path.dirname(im).split("/")[-1])
    os.makedirs(tmp_save_dir, exist_ok=True)
    save_path = os.path.join(tmp_save_dir, os.path.basename(im).replace(".dcm", ".png"))
    cv2.imwrite(save_path, arr)
