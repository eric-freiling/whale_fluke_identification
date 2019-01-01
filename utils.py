from os.path import join, exists
from os import listdir, makedirs
import pandas as pd
import numpy as np
from random import *
import cv2
# from keras.models import model_from_json
from keras.optimizers import rmsprop
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, merge, Lambda

from keras.models import Model, Sequential
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def define_siamese_model(input_shape, filters, conv_shapes, conv_acts, dense_shapes, dense_acts, lr):
    # input shape (w, h, d)

    # Define Inputs for two images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()
    for i in range(len(filters)):
        if i == 0:
            convnet.add(Conv2D(filters[i], conv_shapes[i], activation=conv_acts[i], input_shape=input_shape))
            convnet.add(MaxPooling2D())
        elif i == len(filters) - 1:
            convnet.add(Conv2D(filters[i], conv_shapes[i], activation=conv_acts[i]))
            convnet.add(Flatten())
        else:
            convnet.add(Conv2D(filters[i], conv_shapes[i], activation=conv_acts[i]))
            convnet.add(MaxPooling2D())
    # Conv pipeline to use in each siamese 'leg'

    convnet.add(Dense(dense_shapes[0], activation=dense_acts[0]))

    # encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # merge two encoded inputs with the l1 distance between them
    # Getting the L1 Distance between the 2 encodings
    dist_func = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    distance_layer = dist_func([encoded_l, encoded_r])
    dense_input = distance_layer
    for i in range(1, len(dense_shapes)):
        layer = Dense(dense_shapes[i], activation=dense_acts[i])(dense_input)
        drop_out = Dropout(0.5)(layer)
        dense_input = drop_out

    prediction = Dense(1, activation='sigmoid')(dense_input)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    optimizer = Adam(lr, decay=2.5e-4)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return siamese_net


def define_siamese_vgg16(input_shape, dense_shapes, dense_acts, lr):
    # input shape (w, h, d)

    # Define Inputs for two images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    convnet = Sequential()
    convnet.add(VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)))
    convnet.add(Flatten())
    convnet.add(Dense(dense_shapes[0], activation=dense_acts[0]))

    # encode each of the two inputs into a vector with the convnet
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # merge two encoded inputs with the l1 distance between them
    # Getting the L1 Distance between the 2 encodings
    dist_func = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    distance_layer = dist_func([encoded_l, encoded_r])
    dense_input = distance_layer
    for i in range(1, len(dense_shapes)):
        layer = Dense(dense_shapes[i], activation=dense_acts[i])(dense_input)
        drop_out = Dropout(0.5)(layer)
        dense_input = drop_out

    prediction = Dense(1, activation='sigmoid')(dense_input)
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    optimizer = Adam(lr, decay=2.5e-4)
    siamese_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return siamese_net


def load_embedding_net(model_path, input_shape, dense_shapes, dense_acts, lr):
    # input shape (w, h, d)

    orig_weights = np.load(model_path + ".npy")
    input_layer = Input(input_shape)

    convnet = Sequential()
    convnet.add(VGG16(weights=None, include_top=False, input_shape=(224, 224, 3)))
    convnet.add(Flatten())
    convnet.add(Dense(dense_shapes[0], activation=dense_acts[0]))

    # encode each of the two inputs into a vector with the convnet
    embedded_layer = convnet(input_layer)

    embedded_net = Model(inputs=input_layer, outputs=embedded_layer)

    # optimizer = Adam(lr, decay=2.5e-4)
    # embedded_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    embedded_net.set_weights(orig_weights[0:len(embedded_net.weights)])

    return embedded_net


def load_top_net(model_path, dense_shapes, dense_acts):
    # input shape (w, h, d)

    orig_weights = np.load(model_path + ".npy")

    # encode each of the two inputs into a vector with the convnet
    encoded_l = Input(shape=(4096,))
    encoded_r = Input(shape=(4096,))

    # merge two encoded inputs with the l1 distance between them
    # Getting the L1 Distance between the 2 encodings
    dist_func = Lambda(lambda tensor: K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    distance_layer = dist_func([encoded_l, encoded_r])
    dense_input = distance_layer
    for i in range(1, len(dense_shapes)):
        layer = Dense(dense_shapes[i], activation=dense_acts[i])(dense_input)
        drop_out = Dropout(0.5)(layer)
        dense_input = drop_out

    prediction = Dense(1, activation='sigmoid')(dense_input)
    top_net = Model(inputs=[encoded_l, encoded_r], outputs=prediction)

    # optimizer = Adam(lr, decay=2.5e-4)
    # embedded_net.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=['accuracy'])
    top_net.set_weights(orig_weights[-4::])

    return top_net


def rgb2gray(im):
    gray = 0.2989 * im[:, :, 0] + 0.5870 * im[:, :, 1] + 0.1140 * im[:, :, 2]

    return gray


def transform_image(im, input_shape, bw_flag):
    # convert to gray scale
    if bw_flag:
        im = rgb2gray(im)
    resized_im = cv2.resize(im, (input_shape[1], input_shape[0]))

    return resized_im


def transform_dir(input_dir, output_dir, input_shape, bw_flag):
    if not exists(output_dir):
        makedirs(output_dir)
    files = listdir(input_dir)
    num_files = len(files)
    counter = 1
    for f in files:
        if counter % 1000 == 0:
            print("Transformed {}/{} images".format(counter, num_files))
        image = cv2.imread(join(input_dir, f))
        trans_image = transform_image(image, input_shape, bw_flag)
        cv2.imwrite(join(output_dir, f), trans_image)
        counter += 1


def save_model_npy(model, model_name):
    weights = model.get_weights()
    np.save(model_name, weights)


def load_model_npy(model_name, input_shape, filters, conv_shapes, conv_acts, dense_shapes, dense_acts, lr):
    model = define_siamese_model(
        input_shape,
        filters,
        conv_shapes,
        conv_acts,
        dense_shapes,
        dense_acts,
        lr
    )
    weights = np.load(model_name + ".npy")
    model.set_weights(weights)

    return model


def load_model_vgg16(model_name, input_shape, dense_shapes, dense_acts, lr):
    model = define_siamese_vgg16(
        input_shape,
        dense_shapes,
        dense_acts,
        lr
    )
    weights = np.load(model_name + ".npy")
    model.set_weights(weights)

    return model


def answers_to_hist(answers_path, return_counts=False, return_labels=False):
    df = pd.read_csv(answers_path)
    ids = df.Id.values
    im_names = df.Image.values

    unique_ids = np.unique(ids)
    hist = []
    hist_counts = []
    for u_id in unique_ids:
        loc = np.where(ids == u_id)[0]
        hist_counts.append(len(loc))
        hist.append(im_names[loc])

    if return_counts:
        return np.array(hist), np.array(hist_counts)
    elif return_labels:
        return np.array(hist), unique_ids
    else:
        return np.array(hist)
