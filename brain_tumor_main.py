import numpy as np
import pandas as pd
from preprocess import TumorImageProcessing

# Get data frame processed from extract_data.py
file_path = "C:\\Users\\deeno\\.cache\\processed_image_data.pkl.gz"
df = pd.read_pickle(file_path)

# The data frame has Train & Val data
df_train = df[df['set_type'] == 'Train']
df_val = df[df['set_type'] == 'Val']

# Original dataset had no test sample from training (80%, 10%, 10%)
df_test = df_train.sample(n=len(df_val), random_state=0)
index_to_drop = df_test.index
df_train = df_train.drop(index=df_test.index)

# Prepare input/output datasets
def prepare_dataset(dataframe, n_tumors_max):
    inputs, outputs = [], []
    for i in range(len(dataframe)):
        sample = dataframe.iloc[i]
        data_prep = TumorImageProcessing(sample['image'], sample['box_coor'], sample['class'], n_tumors_max)
        inputs.append(data_prep.resize_and_normalize_image())
        outputs.append(data_prep.create_output_array())
    return np.array(inputs), np.array(outputs)

max_tumor_count = df['box_coor'].apply(len).max()
X_train, y_train = prepare_dataset(df_train, max_tumor_count)
X_val, y_val = prepare_dataset(df_val, max_tumor_count)
X_test, y_test = prepare_dataset(df_test, max_tumor_count)