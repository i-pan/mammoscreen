import glob
import os
import pandas as pd


ROOT = "/mnt/stor/datasets/CBIS-DDSM/"
df = pd.read_csv(os.path.join(ROOT, "train_kfold_cv.csv"))
patches = glob.glob(os.path.join(ROOT, "patches_1-8", "*.png"))
patches_df = pd.DataFrame({"filename": patches})
patches_df["filename"] = patches_df.filename.apply(lambda x: os.path.basename(x))
df["prefix"] = df.filepath.apply(lambda x: x.replace(".png", ""))
patches_df["prefix"] = patches_df.filename.apply(lambda x: "_".join(x.split("_")[:-1]))

df = df.merge(patches_df, on="prefix")
df.to_csv(os.path.join(ROOT, "train_kfold_cv_patches.csv"), index=False)
