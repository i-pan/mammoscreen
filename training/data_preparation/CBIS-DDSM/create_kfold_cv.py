import os
import pandas as pd

from skp.toolbox.functions import create_double_cv


ROOT = "/mnt/stor/datasets/CBIS-DDSM/"
df = pd.read_csv(os.path.join(ROOT, "cleaned_cbis_ddsm.csv"))
df["label"] = (
    df.pathology.apply(lambda x: x.lower().split("_")[0])
    + "_"
    + df.abnormality.apply(lambda x: x.lower())
)
print(df.label.value_counts())

labels = df.label.unique()
for label in labels:
    df[label] = (df.label == label).astype("int")

df_list = []
for fp, fp_df in df.groupby("filepath"):
    tmp_dict = {
        "filepath": [fp], "patient_id": [fp_df.patient_id.iloc[0]]
    }
    for label in labels:
        tmp_dict[label] = [fp_df[label].max()]
    tmp_df = pd.DataFrame(tmp_dict)
    df_list.append(tmp_df)

df = pd.concat(df_list)
df = create_double_cv(df, "patient_id", 10, 10)
print(df)

df.to_csv(os.path.join(ROOT, "train_kfold_cv.csv"), index=False)
