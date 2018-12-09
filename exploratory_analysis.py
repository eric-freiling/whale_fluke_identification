import numpy as np
import pandas as pd
from os.path import join, exists
from os import makedirs, listdir
import cv2


data_path = "../data/whale_fluke_data"
train_path = join(data_path, "train")
test_path = join(data_path, "test")
train_csv_path = join(data_path, "train.csv")

train_dir_names = listdir(train_path)
label_data = pd.read_csv(train_csv_path)
labels, counts = np.unique(label_data.Id, return_counts=True)

file_names = label_data.Image.values
file_ids = label_data.Id.values


def get_im_size_array():
    im_size_array = []
    for name in file_names:
        temp_im = cv2.imread(join(train_path, name))
        im_size_array.append(temp_im.shape)

    return im_size_array


# file_name = df.values[0, 0]
# file_id = df.values[0, 1]
# im = cv2.imread(join(train_path, file_name))

# loc = np.where(file_ids == file_id)[0]
# ims = []
# for idx in loc:
#    ims.append(cv2.imread(join(train_path, file_names[idx])))
