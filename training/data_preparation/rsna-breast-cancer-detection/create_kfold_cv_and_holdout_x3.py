import numpy as np
import os
import pandas as pd

from skp.toolbox.functions import create_double_cv


ROOT = "/mnt/stor/datasets/kaggle/rsna-breast-cancer-detection/"
df = pd.read_csv(os.path.join(ROOT, "train.csv"))
df["filename"] = df.patient_id.astype(str) + "/" + df.image_id.astype(str) + ".png"
df["breast_id"] = df.patient_id.astype(str) + "_" + df.laterality.astype(str)
df["breast_id_index"] = pd.Categorical(df.breast_id).codes

df["BIRADS"] = df.BIRADS.fillna(-1)
density_dict = {k: v for v, k in enumerate(["A", "B", "C", "D"])}
df["density"] = df.density.map(density_dict)
df["density"] = df.density.fillna(-1)
df["difficult_negative_case"] = df.difficult_negative_case.astype(int)

# compare cancer % betwee nsites
print("site 1", df.loc[df.site_id == 1, "cancer"].mean())
print("site 2", df.loc[df.site_id == 2, "cancer"].mean())

# any invasive without cancer or biopsy?
print(len(df.loc[(df.invasive == 1) & (df.cancer == 0) & (df.biopsy == 0)]))

# any cancer without biopsy?
print(len(df.loc[(df.cancer == 1) & (df.biopsy == 0)]))

# fill BIRADS missing and biopsy = 1 with BIRADS 0
df.loc[(df.BIRADS == -1) & (df.biopsy == 1), "BIRADS"] = 0

# any biopsy not BIRADS 0?
print(len(df.loc[(df.biopsy == 1) & (df.BIRADS != 0)]))

# if cancer or invasive or biopsied, BIRADS must be 0
df.loc[df.biopsy == 1, "BIRADS"] = 0

sampling_wts = {each_cls: (df.cancer == each_cls).mean() ** -1 for each_cls in [0, 1]}
df["sampling_weight"] = df.cancer.map(sampling_wts)
df["sampling_weight"] = df["sampling_weight"] / df["sampling_weight"].min()
print(df.sampling_weight.value_counts())

pids = df.patient_id.unique().tolist()
num_holdout = int(0.1 * len(pids))  # 10%
for i in range(3):
    holdout_pids = np.random.choice(pids, size=num_holdout, replace=False)
    holdout_test = df.loc[df.patient_id.isin(holdout_pids)]
    holdout_test.to_csv(
        os.path.join(ROOT, f"holdout_test_v{i + 1:02d}.csv"), index=False
    )
    train_val = df.loc[~df.patient_id.isin(holdout_pids)]
    train_val = create_double_cv(train_val, "patient_id", 10, 10)
    train_val.to_csv(os.path.join(ROOT, f"train_kfold_v{i + 1:02d}.csv"), index=False)
