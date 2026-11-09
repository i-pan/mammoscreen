import cv2
import glob
import os
import torch

from transformers import AutoModel
from tqdm import tqdm


ROOT = "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_dir = os.path.join(ROOT, "train_png")
save_dir = os.path.join(ROOT, "train_cropped_png")

model = AutoModel.from_pretrained("ianpan/mammo-crop", trust_remote_code=True)
model = model.eval().to(device)

files = glob.glob(os.path.join(data_dir, "*/*.png"))
for f in tqdm(files):
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    img_shape = torch.tensor([img.shape[:2]]).to(device)
    x = model.preprocess(img)
    x = torch.from_numpy(x)
    x = x.expand(1, 1, -1, -1)
    x = x.to(device)
    with torch.inference_mode():
        y = model(x, img_shape)
    y = y[0].cpu().numpy()
    x, y, w, h = y
    cropped_img = img[y : y + h, x : x + w]
    pid = f.split("/")[-2]
    os.makedirs(os.path.join(save_dir, pid), exist_ok=True)
    save_path = os.path.join(save_dir, pid, os.path.basename(f))
    cv2.imwrite(save_path, cropped_img)
