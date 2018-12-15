import numpy as np
from os import listdir, makedirs
from os.path import join, exists
import utils
import pandas as pd


def main(data_path):
    train_answers_path = join(data_path, "train.csv")
    train_data_dir = join(data_path, "transformed_train")

    df = pd.read_csv(train_answers_path)
    image_names = df.Image.values
    labels = df.Id.values
    loc = np.where(labels != "new_whale")[0]
    labels = labels[loc]
    image_names = image_names[loc]



if __name__ == "__main__":
    path = "../data/whale_fluke_data"
    main(path)
