from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)

import utils
import sys
import numpy as np
# from keras.models import load_model
import pandas as pd
import random
import cv2
import pickle

from os.path import join, exists
from os import listdir, makedirs, environ

import matplotlib.image as img
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from parameters import *

environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_batch_names(hist, counts, batch_size):
    half = batch_size // 2
    # we want 50% different and same pairs
    left = []
    right = []
    answers = []
    num_unique = len(counts)
    for i in range(half):
        # find two that are the same
        loc = np.where(counts > 1)[0]
        index = random.sample(set(loc), 1)[0]
        im_left, im_right = random.sample(set(hist[index]), 2)
        left.append(im_left)
        right.append(im_right)
        answers.append(1)

        # find two that are different
        loc_left, loc_right = np.random.choice(num_unique, 2, replace=False)
        im_left = random.sample(set(hist[loc_left]), 1)
        im_right = random.sample(set(hist[loc_right]), 1)
        left.append(im_left[0])
        right.append(im_right[0])
        answers.append(0)

    right = np.array(right)
    left = np.array(left)
    answers = np.array(answers)
    random_indices = np.random.permutation(np.arange(len(right)))

    return left[random_indices], right[random_indices], answers[random_indices]


def get_batch(folder, hist, counts, batch_size, input_shape, bw):
    left_names, right_names, answers = get_batch_names(hist, counts, batch_size)
    left_batch = np.zeros([batch_size, input_shape[0], input_shape[1], input_shape[2]])
    right_batch = np.zeros([batch_size, input_shape[0], input_shape[1], input_shape[2]])
    if bw:
        for i in range(batch_size):
            left_batch[i, :, :, :] = cv2.imread(join(folder, left_names[i]))[:, :, [0]]
            right_batch[i, :, :, :] = cv2.imread(join(folder, right_names[i]))[:, :, [0]]
    else:
        for i in range(batch_size):
            left_batch[i, :, :, :] = cv2.imread(join(folder, left_names[i]))[:, :, :]
            right_batch[i, :, :, :] = cv2.imread(join(folder, right_names[i]))[:, :, :]
    return left_batch, right_batch, answers


def main(data_path):

    print("Begin Siamese Training ... \n")

    # Create folder to save models
    if not exists(save_dir):
        makedirs(save_dir)

    if validation_flag:
        train_answers_path = join(data_path, "train_split.csv")
        val_answers_path = join(data_path, "val_split.csv")

        # Check if train-validation split has been made
        if not exists(train_answers_path):
            print("Creating Training-Validation Split... \n")

            with open(join(data_path, "train.csv")) as f:
                content = f.readlines()
            header = [content[0]]
            content = np.array(content[1::])
            num_files = len(content)
            random_indices = np.random.permutation(num_files)
            val_index = int(num_files * validation_percent)
            with open(val_answers_path, "w") as f:
                f.writelines(
                    np.concatenate([
                        header,
                        content[random_indices[0:val_index]]
                    ])
                )
            with open(train_answers_path, "w") as f:
                f.writelines(
                    np.concatenate([
                        header,
                        content[random_indices[val_index::]]
                    ])
                )
        val_hist, val_counts = utils.answers_to_hist(val_answers_path, return_counts=True)
    else:
        train_answers_path = join(data_path, "train.csv")

    train_hist, train_counts = utils.answers_to_hist(train_answers_path, return_counts=True)

    # Check if data is already transformed, if not transform it
    if bw_flag:
        train_data_dir = join(data_path, "transformed_train_" + str(input_shape[0])
                              + "_" + str(input_shape[1]) + "_" + "bw" )
    else:
        train_data_dir = join(data_path, "transformed_train_" + str(input_shape[0])
                              + "_" + str(input_shape[1]) + "_" + "color")
    if not exists(train_data_dir):
        makedirs(train_data_dir)
        print("Transforming Train Directory... \n")
        utils.transform_dir(join(data_path, "train"), train_data_dir, input_shape, bw_flag)

    # Check to see if model already exists
    # If exists load the model and pick up training where we left off
    # If doesn't exist, then create it
    if exists(model_path + ".npy"):
        print("Loading Saved Model...\n")
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
    else:
        print("Creating New Model...\n")
        model = utils.define_siamese_model(
            input_shape,
            filter_sizes,
            conv_shapes,
            conv_acts,
            dense_shapes,
            dense_acts,
            learning_rate
        )

    # Run Epochs
    if validation_flag:
        print("Starting to Train with Validation...\n")
    else:
        print("Starting to Train with no Validation...\n")
    for i in range(epochs):
        # Load training batch
        x_left_train, x_right_train, y_train = get_batch(
            train_data_dir,
            train_hist,
            train_counts,
            batch_size,
            input_shape,
            bw_flag
        )

        # Train on batch
        model.train_on_batch([x_left_train, x_right_train], y_train)

        # Print updates
        if i % print_iter == 0:
            print("Epoch {}/{}".format(i, epochs))
            score = model.evaluate([x_left_train, x_right_train], y_train, verbose=0)
            loss = round(score[0], 2)
            acc = round(score[1], 2)
            print('Training ---- loss: {} accuracy: {}'.format(loss, acc))

            if validation_flag:
                x_left_val, x_right_val, y_val = get_batch(
                    train_data_dir,
                    val_hist,
                    val_counts,
                    batch_size,
                    input_shape,
                    bw_flag
                )
                score = model.evaluate([x_left_val, x_right_val], y_val, verbose=0)
                loss = round(score[0], 2)
                acc = round(score[1], 2)
                print('Validation -- loss: {} accuracy: {}\n'.format(loss, acc))

        if i % save_iter == 0:
            # model.save(model_path) # Usual way to save model but throwing an error when reading for some reason
            utils.save_model_npy(model, model_path)

    # model.save(model_path)
    utils.save_model_npy(model, model_path)
    print('\nSaved trained model at %s ' % model_path)


if __name__ == "__main__":
    # Data path will need to be set according to your own folder structure
    path = "../data/whale_fluke_data"
    main(path)