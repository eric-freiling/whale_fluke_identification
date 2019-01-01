import numpy as np
from os import listdir, makedirs
from os.path import join, exists
import utils
import pandas as pd
import cv2
from parameters import *
from pathlib import Path
import time


def main():
    # Check if data is already transformed, if not transform it

    if bw_flag:
        test_name = "transformed_test_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_" + "bw"
        train_name = "transformed_train_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_" + "bw"
    else:
        test_name = "transformed_test_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_" + "color"
        train_name = "transformed_train_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_" + "color"

    test_data_dir = data_path / test_name
    train_data_dir = data_path / train_name

    if not exists(test_data_dir):
        makedirs(test_data_dir)
        print("Transforming Test Directory... \n")
        utils.transform_dir(test_path, test_data_dir, input_shape, bw_flag)

    train_answers_path = data_path / "train.csv"

    df = pd.read_csv(train_answers_path)
    image_names = df.Image.values
    labels = df.Id.values
    loc = np.where(labels != "new_whale")[0]
    labels = labels[loc]
    image_names = image_names[loc]

    test_file_names = listdir(test_path)

    num_blocks = int(len(labels) / batch_size)
    remainder = len(labels) - num_blocks * batch_size
    batch = np.zeros([batch_size, input_shape[0], input_shape[1], input_shape[2]])
    if not exists(data_path / "train_embeddings.npy"):
        print("Loading Embedded Model...\n")
        model = utils.load_embedding_net(
            model_path,
            input_shape,
            dense_shapes,
            dense_acts,
            learning_rate
        )
        print("Calculating Train Embeddings...\n")
        train_embeddings = np.zeros([len(image_names), 4096])
        for block_num in range(num_blocks):
            if block_num % 100 == 0:
                print("Processing Block {}/{}".format(block_num, num_blocks))
            start = block_num * batch_size
            stop = (block_num + 1) * batch_size
            batch_names = image_names[start:stop]
            counter = 0
            for im_name in batch_names:
                batch[counter, :, :, :] = cv2.imread(str(train_data_dir / im_name))
                counter += 1
            train_embeddings[start:stop, :] = model.predict(batch)
        if remainder > 0:
            batch_names = image_names[stop::]
            batch = np.zeros([remainder, input_shape[0], input_shape[1], input_shape[2]])
            counter = 0
            for im_name in batch_names:
                batch[counter, :, :, :] = cv2.imread(str(train_data_dir / im_name))
                counter += 1
            train_embeddings[stop::, :] = model.predict(batch)
        np.save(data_path / "train_embeddings", train_embeddings)
    else:
        train_embeddings = np.load(data_path / "train_embeddings.npy")

    if not exists(data_path / "test_embeddings.npy"):
        print("Calculating Test Embeddings...\n")
        test_embeddings = np.zeros([len(test_file_names), 4096])
        num_blocks = int(len(test_file_names) / batch_size)
        remainder = len(test_file_names) - num_blocks * batch_size
        batch = np.zeros([batch_size, input_shape[0], input_shape[1], input_shape[2]])
        for block_num in range(num_blocks):
            if block_num % 100 == 0:
                print("Processing Block {}/{}".format(block_num, num_blocks))
            start = block_num * batch_size
            stop = (block_num + 1) * batch_size
            batch_names = test_file_names[start:stop]
            counter = 0
            for im_name in batch_names:
                batch[counter, :, :, :] = cv2.imread(str(test_data_dir / im_name))
                counter += 1
            test_embeddings[start:stop, :] = model.predict(batch)
        if remainder > 0:
            batch_names = test_file_names[stop::]
            batch = np.zeros([remainder, input_shape[0], input_shape[1], input_shape[2]])
            counter = 0
            for im_name in batch_names:
                batch[counter, :, :, :] = cv2.imread(str(test_data_dir / im_name))
                counter += 1
            test_embeddings[stop::, :] = model.predict(batch)
        np.save(data_path / "test_embeddings", test_embeddings)
    else:
        test_embeddings = np.load(data_path / "test_embeddings.npy")
    num_train = train_embeddings.shape[0]
    num_test = len(test_file_names)
    # Apply top net
    if not exists(data_path / "siamese_name_matrix.npy"):
        name_matrix = []
        dist_matrix = np.zeros([num_test, 5])
        top_model = utils.load_top_net(
            model_path,
            dense_shapes,
            dense_acts
        )
        for i in range(num_test):
            if i % 100 == 0:
                print("Processing Test File {}/{}".format(i, num_test))
            left_batch = np.zeros([num_train, 4096])
            for j in range(num_train):
                left_batch[j, :] = test_embeddings[i, :]
            y_hat = top_model.predict([left_batch, train_embeddings])
            y = y_hat.flatten()
            sort_loc = np.argsort(y)
            counter = 0
            whale_names = []
            whale_set = set([])
            index = num_train - 1
            while counter < 5:
                whale_set.add(labels[sort_loc[index]])
                if len(whale_set) > counter:
                    whale_names.append(labels[sort_loc[index]])
                    dist_matrix[i, counter] = y[sort_loc[index]]
                    counter += 1
                index -= 1
            name_matrix.append(whale_names)

        name_matrix = np.array(name_matrix)
        np.save(data_path / "siamese_name_matrix", name_matrix)
        np.save(data_path / "siamese_dist_matrix", dist_matrix)
    else:
        name_matrix = np.load(data_path / "siamese_name_matrix.npy")
        dist_matrix = np.load(data_path / "siamese_dist_matrix.npy")

    # new_whale_thresh = 20
    # for i in range(len(test_file_names)):
    #     loc = np.where(dist_matrix[i, :] > new_whale_thresh)[0]
    #     if len(loc) > 0:
    #         names = name_matrix[i, :]
    #         temp = np.copy(names[loc[0]:-1])
    #         name_matrix[i, loc[0]] = "new_whale"
    #         if loc[0] < 4:
    #             name_matrix[i, loc[0] + 1::] = temp

    if not exists("submissions"):
        makedirs("submissions")
    if bw_flag:
        submission_name = "submissions/siamese_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_bw.csv"
    else:
        submission_name = "submissions/siamese_" + str(input_shape[0]) + "_" + str(input_shape[1]) + "_color.csv"
    with open(submission_name, "w") as f:
        f.write("Image,Id\n")
        for i in range(len(test_file_names)):
            line = test_file_names[i] + "," + " ".join(name_matrix[i,:].tolist()) + "\n"
            f.write(line)


def time_test():
    top_model = utils.load_top_net(
        model_path,
        dense_shapes,
        dense_acts
    )
    test_embeddings = np.load(data_path / "test_embeddings.npy")
    train_embeddings = np.load(data_path / "train_embeddings.npy")
    num_train = train_embeddings.shape[0]
    num_test = test_embeddings.shape[0]

    start = time.time()
    left_batch = np.zeros([num_train, 4096])
    for j in range(num_train):
        left_batch[j, :] = test_embeddings[0, :]
    y_hat = top_model.predict([left_batch, train_embeddings])
    total = time.time() - start
    total_time = (total * num_test)/ 60
    print(total_time)


if __name__ == "__main__":
    # time_test()
    main()
