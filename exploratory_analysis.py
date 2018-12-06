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
