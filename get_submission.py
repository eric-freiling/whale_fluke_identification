import numpy as np
from os import listdir, makedirs
from os.path import join, exists
import utils
import pandas as pd
import cv2
from parameters import *


def main(data_path):
    # Check if data is already transformed, if not transform it
    if bw_flag:
        test_data_dir = join(data_path, "transformed_test_" + str(input_shape[0])
                              + "_" + str(input_shape[1]) + "_" + "bw")
        train_data_dir = join(data_path, "transformed_train_" + str(input_shape[0])
                             + "_" + str(input_shape[1]) + "_" + "bw")
    else:
        test_data_dir = join(data_path, "transformed_test_" + str(input_shape[0])
                              + "_" + str(input_shape[1]) + "_" + "color")
        train_data_dir = join(data_path, "transformed_train_" + str(input_shape[0])
                             + "_" + str(input_shape[1]) + "_" + "color")
    if not exists(test_data_dir):
        makedirs(test_data_dir)
        print("Transforming Train Directory... \n")
        utils.transform_dir(join(data_path, "test"), test_data_dir, input_shape, bw_flag)
    train_answers_path = join(data_path, "train.csv")

    df = pd.read_csv(train_answers_path)
    image_names = df.Image.values
    labels = df.Id.values
    loc = np.where(labels != "new_whale")[0]
    labels = labels[loc]
    image_names = image_names[loc]

    model = utils.load_model_npy(
        model_path,
        input_shape,
        filter_sizes,
        conv_shapes,
        conv_acts,
        dense_shapes,
        dense_acts,
        learning_rate
    )
    test_dir = join(data_path, "test")
    test_file_names = listdir(test_dir)
    test_batch_size = 1000
    num_blocks = int(len(labels)/test_batch_size)
    remainder = len(labels) - num_blocks * test_batch_size
    left_batch = np.zeros([test_batch_size, *input_shape])
    right_batch = np.zeros([test_batch_size, *input_shape])
    answers = []
    test_counter = 0
    for test_file in test_file_names:
        if test_counter % 500 == 0:
            print("Test Image: {}/{}".format(test_counter, len(test_file_names)))
        test_counter += 1
        test_im = cv2.imread(join(test_data_dir, test_file))

        # For each test image test against
        counter = 0
        probs = np.zeros(len(labels))
        for block_num in range(num_blocks):
            for i in range(test_batch_size):
                left_batch[i, :, :, :] = test_im[:, :, [0]]
                if bw_flag:
                    right_batch[i, :, :, :] = cv2.imread(join(train_data_dir, image_names[counter]))[:, :, [0]]
                else:
                    right_batch[i, :, :, :] = cv2.imread(join(train_data_dir, image_names[counter]))[:, :, :]
                counter += 1
            y_hat = model.predict([left_batch, right_batch])
            probs[block_num * test_batch_size:(block_num + 1) * test_batch_size] = y_hat.flatten()
        for i in range(remainder):
            left_batch[i, :, :, :] = test_im[:, :, [0]]
            if bw_flag:
                right_batch[i, :, :, :] = cv2.imread(join(train_data_dir, image_names[counter]))[:, :, [0]]
            else:
                right_batch[i, :, :, :] = cv2.imread(join(train_data_dir, image_names[counter]))[:, :, :]
            counter += 1
            y_hat = model.predict([left_batch, right_batch])
            probs[num_blocks * test_batch_size::] = y_hat.flatten()
        answers.append(labels[np.argmax(probs)])

    if not exists("submissions"):
        makedirs("submissions")
    if bw_flag:
        submission_name = "submissions/" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_bw.csv"
    else:
        submission_name = "submissions/" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_color.csv"
    with open(submission_name, "w") as f:
        f.write("Image,Id\n")
        for i in range(len(labels)):
            f.write(image_names[i] + "," + answers[i])


if __name__ == "__main__":
    path = "../data/whale_fluke_data"
    main(path)
